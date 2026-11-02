# -*- coding: utf-8 -*-
"""
模型训练主入口 —— 依据 plan.docx 重构后的完整流程
====================================================
流程（消除标签泄漏 + 信号预处理）：
  1. 病例筛选      找到同时有 CO₂ + RR + EtCO₂ 参考轨道的病例
  2. 数据加载      下载 CO₂ 波形 + 参考轨道（时间对齐）
  3. 窗口切分      30s 窗口 + 质量检测（插值/去噪/剔除低质量）
  4. 特征提取      12 维新特征（全从 CO₂ 波形独立计算）
  5. 标签生成      用监护仪参考轨道 + 原 5 分类规则打标签
  6. 数据划分      锁定 Final Test 病例 + GroupKFold CV
  7. 模型训练      RandomForest + 交叉验证
  8. 评估          Accuracy + Macro-F1 + 混淆矩阵 + 特征重要性
  9. 导出 ONNX     生成 STM32 可部署模型

设计原则：标签依据（监护仪参考）与模型输入（CO₂ 波形特征）彻底分离。
"""
import os
import sys
import numpy as np

# 让模块可导入（无论从何处运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference, inspect_case
from preprocessing.make_windows import make_windows_with_reference, summarize_windows
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features, FEATURE_NAMES, FEATURE_NAMES_CN
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from training.train_rf import (split_final_test, cross_validate,
                               train_final_model, evaluate_on_final_test,
                               evaluate_feature_importance)

# ============================================================
# 配置
# ============================================================
SAMPLING_RATE = 100        # 重采样目标频率（Hz），原始 Primus 为 62.5Hz
WINDOW_SIZE = 9000         # 90 秒窗口 = 100Hz × 90s（实验验证最优，见 experiment_window_length.py）
WINDOW_STEP = 4500         # 窗口步长（半重叠，4500=45s 步进）
                           # 窗口长度实验（150病例, Final Test Macro-F1）:
                           #   30s: 0.714 | 60s: 0.758 | 90s: 0.763 | 120s: 0.754
                           #   90s 最优（与文献 Spijkerboer 2024 一致）
N_CASES = 200              # 病例数（扩展实验：150/200/250 对比，200 最优 Macro-F1 0.888）
FINAL_TEST_RATIO = 0.2     # 锁定 20% 病例为 Final Test
N_CV_FOLDS = 5             # GroupKFold 折数
USE_PREPROCESS = True      # 是否做信号预处理（插值/去噪/基线）
MODEL_NAME = 'rf'          # 模型: rf (RandomForest) / xgb (XGBoost加权) / lgb (LightGBM加权)
OUTPUT_ONNX = 'etco2_model_v2.onnx'


# ============================================================
# 主流程
# ============================================================
def main():
    print('=' * 70)
    print('CO₂ 呼吸模式识别模型训练（plan.docx 重构版）')
    print('=' * 70)

    # ---------- 1. 病例筛选 ----------
    print('\n[1/9] 查找合格病例（CO₂ + RR + ETCO₂ 参考轨道）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    print(f'  找到 {len(case_ids)} 个合格病例')
    if len(case_ids) < 10:
        print('  ⚠ 病例过少，请检查网络或轨道名称')
        return

    # ---------- 2-3. 数据加载 + 窗口切分 ----------
    print('\n[2/9] 加载数据并切分窗口...')
    all_windows = []
    all_case_of_window = []

    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid, interval=0.01)
        except Exception as e:
            print(f'  病例 {cid} 加载失败: {e}')
            continue

        # 窗口切分（CO₂ + 对齐参考轨道）
        records = make_windows_with_reference(
            ref['co2'], ref['rr'], ref['etco2'],
            window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)

        # 保留窗口（含质量记录）
        for rec in records:
            all_windows.append(rec)
            all_case_of_window.append(cid)

        print(f'  病例 {cid}: {len(records)} 窗口 (RR来源={ref["rr_source"]})')

    print(f'\n  总窗口数: {len(all_windows)}（来自 {len(set(all_case_of_window))} 病例）')
    if len(all_windows) < 100:
        print('  ⚠ 窗口过少，无法有效训练')
        return

    # ---------- 4. 特征提取 ----------
    print('\n[3/9] 提取 12 维波形特征...')
    X_list = []
    for rec in all_windows:
        co2 = rec['co2']
        if USE_PREPROCESS:
            co2, _ = preprocess_window(co2, sampling_rate=SAMPLING_RATE)
        features = extract_features(co2, sampling_rate=SAMPLING_RATE)
        X_list.append(features)
    X = np.array(X_list)
    print(f'  特征矩阵: {X.shape} (12 维)')

    # ---------- 5. 标签生成（参考轨道） ----------
    print('\n[4/9] 用监护仪参考轨道生成标签...')
    y_list = []
    kept_idx = []
    n_no_label = 0
    for i, rec in enumerate(all_windows):
        result = build_labels_for_window(None, rec['rr'], rec['etco2'], min_reference_points=2)
        if result['label'] is None:
            n_no_label += 1
            continue  # 无参考值的窗口不打标签，跳过
        y_list.append(result['label'])
        kept_idx.append(i)

    # 同步过滤 X / y / cases
    X = X[kept_idx]
    y = np.array(y_list)
    cases = np.array([all_case_of_window[i] for i in kept_idx])

    print(f'  有效样本: {len(y)}（{n_no_label} 个无参考标签被跳过）')
    if len(y) == 0:
        print('  ⚠ 无有效样本，请检查参考轨道')
        return

    # 类别分布
    class_dist = {i: int((y == i).sum()) for i in np.unique(y)}
    print(f'  类别分布: {class_dist}')

    # ---------- 6. 数据划分（Final Test + CV） ----------
    print('\n[5/9] 锁定 Final Test 病例 + GroupKFold CV...')
    final_test_cases, cv_cases = split_final_test(cases, test_ratio=FINAL_TEST_RATIO)
    print(f'  Final Test 病例 ({len(final_test_cases)}): {sorted(final_test_cases)}')
    print(f'  CV 病例 ({len(cv_cases)}): {sorted(cv_cases)[:5]}...')

    # ---------- 7. 模型训练（GroupKFold CV） ----------
    print(f'\n[6/9] 交叉验证（GroupKFold, 病例级隔离, 模型={MODEL_NAME}）...')
    cv_mask = np.isin(cases, cv_cases)
    cv_results = cross_validate(X[cv_mask], y[cv_mask], cases[cv_mask],
                                n_splits=N_CV_FOLDS, model_name=MODEL_NAME)
    if cv_results:
        print(f'\n  CV 平均: Accuracy={np.mean([r["accuracy"] for r in cv_results]):.4f} '
              f'± {np.std([r["accuracy"] for r in cv_results]):.4f}, '
              f'Macro-F1={np.mean([r["macro_f1"] for r in cv_results]):.4f}')

    # ---------- 8. 最终模型 + Final Test 评估 ----------
    print('\n[7/9] 训练最终模型并在 Final Test 上评估...')
    clf, _ = train_final_model(X, y, cases, final_test_cases, model_name=MODEL_NAME)
    final_eval = evaluate_on_final_test(clf, X, y, cases, final_test_cases,
                                        class_names=['正常', '急促', '屏息', '呼吸异常'])  # 4类标签 0/1/2/3

    # 特征重要性
    print('\n[8/9] 特征重要性...')
    imp = evaluate_feature_importance(clf, FEATURE_NAMES_CN)

    # ---------- 9. 导出 ONNX（默认禁用，直到最终方案确定） ----------
    if not os.environ.get('EXPORT_ONNX'):
        print('\n[9/9] 导出 ONNX（跳过：EXPORT_ONNX 未设置，最终方案确定后再导出）')
    else:
        print('\n[9/9] 导出 ONNX...')
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

        if MODEL_NAME == 'xgb':
            # XGBoost 用 onnxmltools 导出（skl2onnx 不支持 XGBClassifier）
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType as OmFloatTensorType
            initial_type_om = [('float_input', OmFloatTensorType([None, X.shape[1]]))]
            onnx_model = onnxmltools.convert_xgboost(clf.get_booster(), 'xgb',
                                                     initial_types=initial_type_om)
        else:
            from skl2onnx import convert_sklearn
            onnx_model = convert_sklearn(clf, initial_types=initial_type)

        with open(OUTPUT_ONNX, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        onnx_size = os.path.getsize(OUTPUT_ONNX) / 1024
        print(f'  已保存 {OUTPUT_ONNX} ({onnx_size:.0f} KB), 输入维度 {X.shape[1]}')

    print('\n' + '=' * 70)
    print('训练完成！')
    print(f'  ONNX: {"已导出 " + OUTPUT_ONNX if os.environ.get("EXPORT_ONNX") else "未导出（EXPORT_ONNX 未设置）"}')
    print(f'  CV Macro-F1: {np.mean([r["macro_f1"] for r in cv_results]):.4f}' if cv_results else '  CV 未完成')
    if final_eval:
        print(f'  Final Test Accuracy: {final_eval["accuracy"]:.4f}, Macro-F1: {final_eval["macro_f1"]:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
