# -*- coding: utf-8 -*-
"""
自主呼吸专用训练 —— 只用非机械通气（自主呼吸）窗口训练模型
============================================================
背景：主训练 train.py 用全部窗口（89% 机械通气），但机械通气下
      RR/EtCO₂ 由呼吸机设定，5 分类体系在自主呼吸语义上不成立。
      本项目部署场景（家用/健康监测 NDIR）是自主呼吸。

本脚本：
  1. 按 Primus/SET_RR_IPPV 判断通气模式（有设定值=机械通气，无=自主呼吸）
  2. 只保留自主呼吸时段内的 30s 窗口
  3. 用与 train.py 相同的流程训练 + 评估 + 导出 ONNX

注意：需联网下载 SET_RR_IPPV 轨道（未在缓存中），首次运行会下载。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from training.train_rf import (split_final_test, cross_validate,
                               train_final_model, evaluate_on_final_test,
                               evaluate_feature_importance)
import vitaldb

# ============================================================
# 配置
# ============================================================
SAMPLING_RATE = 100
WINDOW_SIZE = 3000         # 30 秒
WINDOW_STEP = 1500         # 半重叠
N_CASES = 150
FINAL_TEST_RATIO = 0.2
MODEL_NAME = 'rf'
OUTPUT_ONNX = 'etco2_model_sb.onnx'   # 自主呼吸版模型，不覆盖主模型

# 呼吸机设定频率轨道（判断通气模式）
VENT_SET_RR_TRACK = 'Primus/SET_RR_IPPV'


def is_spontaneous(co2, set_rr, start=0, sampling_rate=SAMPLING_RATE,
                   window_size=WINDOW_SIZE, sb_ratio=0.8):
    """判断一个 CO₂ 窗口是否落在"自主呼吸"时段。

    逻辑：窗口时间范围内，呼吸机设定 RR 的采样点中，
    无有效设定值（<=0 或 NaN）的比例 >= sb_ratio → 自主呼吸。

    参数:
        co2: 窗口 CO₂ 波形（mmHg）
        set_rr: 呼吸机设定 RR 轨道（整段，同时间轴，含 NaN）
        start: 窗口在整段中的绝对起始索引
    """
    end = start + len(co2)
    # 窗口在 set_rr 中对应的绝对段
    if start >= len(set_rr):
        return False
    seg_set_rr = set_rr[start:min(end, len(set_rr))]

    # 有 CO2 信号的区间内，判断设定 RR 是否有效
    has_co2 = ~np.isnan(co2)
    if not has_co2.any():
        return False

    # 对齐 co2 的有效部分
    seg_set_rr_valid = seg_set_rr[has_co2[:len(seg_set_rr)]]
    if len(seg_set_rr_valid) == 0:
        return False

    # 设定 RR 无效（NaN 或 <=0）= 呼吸机未强制送气 = 自主呼吸
    vent_active = ~np.isnan(seg_set_rr_valid) & (seg_set_rr_valid > 0)
    sb_fraction = 1.0 - vent_active.mean()
    return sb_fraction >= sb_ratio


def main():
    print('=' * 70)
    print('自主呼吸专用训练（非机械通气窗口）')
    print('=' * 70)

    print(f'\n[1/5] 查找合格病例（{N_CASES}）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    print(f'  找到 {len(case_ids)} 个')

    print('\n[2/5] 加载数据 + 筛自主呼吸窗口...')
    all_windows, all_cases = [], []
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid, interval=0.01)
            # 额外加载呼吸机设定 RR（判断通气模式）
            set_rr = vitaldb.load_case(caseid=cid,
                                       track_names=[VENT_SET_RR_TRACK],
                                       interval=0.01)
            set_rr = set_rr[:, 0] if set_rr.ndim > 1 else set_rr
        except Exception as e:
            print(f'  病例 {cid} 加载失败: {e}')
            continue

        # 手工切窗（与 make_windows_with_reference 一致，但要判断自主呼吸）
        from preprocessing.make_windows import make_windows_with_reference
        records = make_windows_with_reference(
            ref['co2'], ref['rr'], ref['etco2'],
            window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)

        sb_count = 0
        for rec in records:
            # 自主呼吸判断：窗口 CO₂ 对应时段内设定 RR 无值
            if is_spontaneous(rec['co2'], set_rr, start=rec['start']):
                all_windows.append(rec)
                all_cases.append(cid)
                sb_count += 1
        print(f'  病例 {cid}: {len(records)} 窗口, 自主呼吸 {sb_count}')

    print(f'\n  自主呼吸窗口总数: {len(all_windows)}（来自 {len(set(all_cases))} 病例）')
    if len(all_windows) < 200:
        print('  ⚠ 自主呼吸窗口过少，无法有效训练')
        return

    print('\n[3/5] 提取特征 + 生成标签...')
    X_list, y_list, kept_cases = [], [], []
    for i, rec in enumerate(all_windows):
        co2, _ = preprocess_window(rec['co2'], sampling_rate=SAMPLING_RATE)
        res = build_labels_for_window(None, rec['rr'], rec['etco2'], min_reference_points=2)
        if res['label'] is None:
            continue
        X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
        y_list.append(res['label'])
        kept_cases.append(all_cases[i])

    X = np.array(X_list)
    y = np.array(y_list)
    cases = np.array(kept_cases)
    class_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f'  有效样本: {len(y)}, 特征 {X.shape[1]}')
    print(f'  类别分布: {class_dist}')

    print('\n[4/5] 划分 + 训练 + 评估...')
    final_test_cases, cv_cases = split_final_test(cases, test_ratio=FINAL_TEST_RATIO)
    cv_mask = np.isin(cases, cv_cases)
    cv_results = cross_validate(X[cv_mask], y[cv_mask], cases[cv_mask], model_name=MODEL_NAME)
    if cv_results:
        print(f'  CV: Accuracy={np.mean([r["accuracy"] for r in cv_results]):.4f}, '
              f'Macro-F1={np.mean([r["macro_f1"] for r in cv_results]):.4f}')

    clf, _ = train_final_model(X, y, cases, final_test_cases, model_name=MODEL_NAME)
    final_eval = evaluate_on_final_test(clf, X, y, cases, final_test_cases,
                                        class_names=['正常', '急促', '缓慢', '屏息', '异常'])

    print('\n[5/5] 导出 ONNX...')
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx import convert_sklearn
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    with open(OUTPUT_ONNX, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f'  已保存 {OUTPUT_ONNX} ({os.path.getsize(OUTPUT_ONNX)/1024:.0f} KB, {X.shape[1]}维)')

    print('\n' + '=' * 70)
    print('自主呼吸训练完成')


if __name__ == '__main__':
    main()
