# -*- coding: utf-8 -*-
"""
90s 窗口内部一致性分析
=======================
问题：90s 窗口标签用中位数决定，但窗口内可能混入多种呼吸状态
      （如前半正常后半异常），导致"混合波形 → 单一标签"的模糊对应。

本脚本：
  1. 统计每个 90s 窗口内参考 RR / EtCO2 的变异程度
     （窗口内标准差、范围、状态变化次数）
  2. 评估"内部不一致"窗口占比
  3. 对比"一致" vs "不一致"窗口的预测正确率
     → 判断内部一致性是否影响模型
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from training.train_rf import split_final_test
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200


def window_internal_variability(rr_seg, etco2_seg):
    """计算窗口内参考值的内部变异指标。

    返回:
        rr_cv: RR 变异系数（std/mean，窗口内 RR 波动程度）
        rr_range: RR 最大值-最小值
        et_std: EtCO2 标准差
        n_state_changes: RR 状态变化次数（跨阈值 6/18）
        has_multiple_states: 是否出现多个 RR 状态（正常/急促/缓慢/屏息）
    """
    rr = rr_seg[~np.isnan(rr_seg) & (rr_seg > 0)]
    et = etco2_seg[~np.isnan(etco2_seg) & (etco2_seg > 0)]

    if len(rr) < 2:
        return None  # RR 参考点太少，无法评估

    rr_std = np.std(rr)
    rr_mean = np.mean(rr)
    rr_cv = rr_std / rr_mean if rr_mean > 0 else 0
    rr_range = np.ptp(rr)

    # RR 状态变化：正常(7-17)/急促(>18)/缓慢(<7)/屏息(<3)
    states = []
    for v in rr:
        if v < 3:
            states.append(3)
        elif v < 7:
            states.append(2)
        elif v > 18:
            states.append(1)
        else:
            states.append(0)
    n_changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
    n_unique_states = len(set(states))

    et_std = np.std(et) if len(et) > 1 else 0

    return {
        'rr_cv': rr_cv, 'rr_range': rr_range, 'et_std': et_std,
        'n_changes': n_changes, 'n_unique_states': n_unique_states,
    }


def main():
    print(f'90s窗口内部一致性分析（{N_CASES} 病例）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)

    # 收集 (标签, 内部变异指标, 是否预测正确)
    X_list, y_list, cases_list, var_list = [], [], [], []
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid)
            records = make_windows_with_reference(
                ref['co2'], ref['rr'], ref['etco2'],
                window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)
        except Exception:
            continue
        for r in records:
            res = build_labels_for_window(None, r['rr'], r['etco2'], min_reference_points=2)
            if res['label'] is None:
                continue
            v = window_internal_variability(r['rr'], r['etco2'])
            if v is None:
                continue  # 参考点太少
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(res['label'])
            cases_list.append(cid)
            var_list.append(v)

    X = np.array(X_list)
    y = np.array(y_list)
    g = np.array(cases_list)
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    # 内部一致性统计
    rr_cv = np.array([v['rr_cv'] for v in var_list])
    rr_range = np.array([v['rr_range'] for v in var_list])
    n_changes = np.array([v['n_changes'] for v in var_list])
    n_states = np.array([v['n_unique_states'] for v in var_list])

    print('\n' + '=' * 60)
    print('窗口内 RR 变异统计')
    print('=' * 60)
    print(f'  RR变异系数: 中位 {np.median(rr_cv):.3f}, 25分位 {np.percentile(rr_cv,25):.3f}, '
          f'75分位 {np.percentile(rr_cv,75):.3f}')
    print(f'  RR范围(bpm): 中位 {np.median(rr_range):.1f}, 90分位 {np.percentile(rr_range,90):.1f}')
    print(f'  状态变化次数: 0次 {int((n_changes==0).sum())} | 1次 {int((n_changes==1).sum())} '
          f'| 2次+ {int((n_changes>=2).sum())}')
    print(f'  窗口内不同RR状态数: 1种 {int((n_states==1).sum())} | 2种 {int((n_states==2).sum())} '
          f'| 3种+ {int((n_states>=3).sum())}')

    # 不一致窗口占比（变异系数高或状态变化多）
    inconsistent = (rr_cv > 0.15) | (n_changes >= 2)
    print(f'\n  内部不一致窗口(高变异或多状态): {int(inconsistent.sum())}/{len(y)} '
          f'({inconsistent.mean()*100:.1f}%)')

    # 对比一致/不一致窗口的标签可靠性
    # 用训练好的模型预测，看一致 vs 不一致窗口的准确率
    print('\n' + '=' * 60)
    print('模型预测：一致 vs 不一致窗口')
    print('=' * 60)
    final_test_cases, _ = split_final_test(g, test_ratio=0.2)
    train_mask = np.isin(g, final_test_cases, invert=True)
    test_mask = np.isin(g, final_test_cases)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                 class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
    clf.fit(X[train_mask], y[train_mask])
    yp = clf.predict(X[test_mask])

    for name, mask in [('一致窗口', ~inconsistent), ('不一致窗口', inconsistent)]:
        sub_test = test_mask & mask
        if sub_test.sum() == 0:
            print(f'  {name}: 无测试样本')
            continue
        acc = accuracy_score(y[sub_test], yp[sub_test])
        print(f'  {name}: n={int(sub_test.sum())}, 准确率 {acc:.3f}')

    print('\n分析完成')


if __name__ == '__main__':
    main()
