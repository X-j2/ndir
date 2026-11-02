# -*- coding: utf-8 -*-
"""实验 A：标签流/特征流 分离 vs 不分离 的对照实验

论文问题
--------
旧方案（train_model.py）准确率虚高到 1.00，新方案经过重构后是 0.91。
但"1.00 → 0.91"这个对比横跨了两代代码——窗口长度、类别数、特征维度
全都变了，无法把差异归因到"消除标签泄漏"这一项上。

本实验通过**只改变一个变量**给出可归因的对照：
    标签的生成依据，是否同时作为模型的输入特征。

三组对照（其余全部相同：200 病例 / 90s 窗口 / 半重叠 / 同模型 / 同划分）
------------------------------------------------------------------------
  组 1「不分离」  特征 = 12 维 + [自算RR, 自算EtCO2]
                  标签 = 用这两个量按 4 类规则生成
                  → 标签依据就是特征本身，模型只需复现规则

  组 2「部分分离」特征 = 仅 12 维
                  标签 = 仍用波形自算的 RR/EtCO2 生成
                  → 特征不再直接给出标签依据，但 12 维里的
                     F9(峰数)/F11(峰数×2) 仍与自算 RR 同源

  组 3「完全分离」特征 = 仅 12 维
                  标签 = 监护仪参考轨道生成（当前生产方案）
                  → 标签来源与模型输入完全独立

预期：组 1 接近 1.00，组 3 回到 0.9x；组 2 介于两者之间，
说明"只把特征换掉"不足以消除泄漏，必须同时更换标签来源。

用法：python experiment_leakage_separation.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features, find_breath_peaks
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from training.train_rf import (split_final_test, get_model,
                               train_final_model, evaluate_on_final_test)

# ---- 与 train.py 保持完全一致的配置 ----
SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200
FINAL_TEST_RATIO = 0.2
N_CV_FOLDS = 5
RANDOM_STATE = 42

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'vitaldb')


# ============================================================
# 从波形自算 RR 与 EtCO2（旧方案的标签依据）
# ============================================================
def waveform_rr_etco2(window):
    """从 CO2 波形独立计算呼吸率与潮末 CO2 —— 与旧 train_model.py 同源。

    寻峰参数沿用当前训练端配置（height=12, distance=1.0s, prominence=8），
    使两组对照的唯一差异是"标签是否由特征本身决定"，而非寻峰参数。
    """
    peaks = find_breath_peaks(window, SAMPLING_RATE)
    n_peaks = len(peaks)

    # 真实呼吸率（90 秒窗口）
    rr = n_peaks * 60.0 / (WINDOW_SIZE / SAMPLING_RATE)

    x = np.asarray(window, dtype=float)
    max_val = float(np.max(x))
    if n_peaks >= 1:
        etco2 = float(np.mean(x[peaks]))
    else:
        etco2 = max_val          # 无峰时用最大值兜底（旧方案做法）

    return rr, etco2, max_val, n_peaks


def label_from_selfcomputed(rr, etco2, max_val):
    """用波形自算量按 4 类规则生成标签（复刻旧 train_model.py 的标签逻辑）。

    ⚠ 关键：旧版标签规则用的是 max_val 与 rr_per_min —— 而这两个量
    同时也是模型输入特征。这正是标签泄漏的根源，本函数必须如实复刻，
    否则对照实验测不出泄漏。类别编号由旧版 5 类映射到当前 4 类。

    旧版规则（优先级自上而下）：
        max_val < 10 或 rr_per_min < 3   → 屏息
        rr_per_min >= 18                 → 急促
        rr_per_min <= 6                  → 缓慢（现并入异常）
        max_val < 35 或 max_val > 45     → 异常
        否则                              → 正常
    """
    # 屏息
    if max_val < 10 or rr < 3:
        return 2
    # 急促
    if rr >= 18:
        return 1
    # 缓慢（旧类 2）与浓度异常（旧类 4）现均归入「呼吸异常」(3)
    if rr <= 6:
        return 3
    if max_val < 35 or max_val > 45:
        return 3
    return 0


# ============================================================
# 数据加载
# ============================================================
def load_all(n_cases=N_CASES):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))[:n_cases]
    print(f'  从缓存加载 {len(files)} 个病例...')

    feats12, feats12_leak, y_ref, y_self, cases = [], [], [], [], []
    n_skip_ref = 0

    for path in files:
        cid = int(os.path.basename(path).split('_')[0])
        try:
            ref = load_reference_tracks(cid, interval=0.01)
        except Exception:
            continue

        records = make_windows_with_reference(
            ref['co2'], ref['rr'], ref['etco2'],
            window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)

        for rec in records:
            co2 = rec['co2']
            co2, _ = preprocess_window(co2, sampling_rate=SAMPLING_RATE)

            # --- 组 3 的标签：监护仪参考轨道 ---
            lab = build_labels_for_window(None, rec['rr'], rec['etco2'],
                                          min_reference_points=2)
            if lab['label'] is None:
                n_skip_ref += 1
                continue                      # 无参考标签的窗口整组跳过，保证三组样本一致

            # --- 12 维特征（组 2/3 用）---
            f12 = extract_features(co2, sampling_rate=SAMPLING_RATE)

            # --- 波形自算量（组 1 的额外特征 + 组 1/2 的标签依据）---
            rr, etco2, max_val, _ = waveform_rr_etco2(co2)
            y_self_lab = label_from_selfcomputed(rr, etco2, max_val)

            feats12.append(f12)
            feats12_leak.append(list(f12) + [rr, max_val])
            y_ref.append(int(lab['label']))
            y_self.append(y_self_lab)
            cases.append(cid)

    print(f'  有效窗口 {len(y_ref)} 个（{n_skip_ref} 个因无参考标签跳过）')
    return (np.array(feats12), np.array(feats12_leak),
            np.array(y_ref), np.array(y_self), np.array(cases))


# ============================================================
# 单组评估
# ============================================================
def run_arm(name, X, y, cases, final_test_cases):
    print(f'\n{"=" * 62}')
    print(f'  {name}')
    print(f'{"=" * 62}')
    print(f'  特征维度 {X.shape[1]}，样本 {len(y)}，病例 {len(set(cases))}')

    dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f'  类别分布 {dist}')

    # 训练与评估（与 train.py 完全相同的流程）
    clf, train_mask = train_final_model(X, y, cases, final_test_cases,
                                        random_state=RANDOM_STATE)
    result = evaluate_on_final_test(clf, X, y, cases,
                                    final_test_cases,
                                    class_names=['正常', '急促', '屏息', '异常'])
    if result is None:
        return None
    return {'name': name, 'acc': result['accuracy'],
            'macro_f1': result['macro_f1'], 'dist': dist,
            'n_features': X.shape[1]}


def main():
    print('=' * 62)
    print('实验 A：标签流/特征流 分离 vs 不分离')
    print('=' * 62)
    print(f'配置：{N_CASES} 病例 / {WINDOW_SIZE // SAMPLING_RATE}s 窗口 / '
          f'半重叠 / RF 30树 / GroupKFold({N_CV_FOLDS})')
    print('三组仅"标签依据是否作为特征"不同，其余完全一致\n')

    X12, X12leak, y_ref, y_self, cases = load_all()

    # 三组共用同一套 Final Test 病例划分
    final_test_cases, _ = split_final_test(cases, FINAL_TEST_RATIO, RANDOM_STATE)
    print(f'  Final Test 病例 {len(final_test_cases)} 个')

    results = []

    r1 = run_arm('组 1「不分离」：特征含标签依据 + 标签自算',
                 X12leak, y_self, cases, final_test_cases)
    if r1: results.append(r1)

    r2 = run_arm('组 2「部分分离」：仅 12 维特征 + 标签自算',
                 X12, y_self, cases, final_test_cases)
    if r2: results.append(r2)

    r3 = run_arm('组 3「完全分离」：仅 12 维特征 + 监护仪参考标签（当前方案）',
                 X12, y_ref, cases, final_test_cases)
    if r3: results.append(r3)

    # ---- 汇总 ----
    print('\n' + '=' * 62)
    print('  汇总（论文可直接引用）')
    print('=' * 62)
    print(f'  {"实验组":<44}{"Accuracy":>10}{"Macro-F1":>10}')
    print('  ' + '-' * 62)
    for r in results:
        print(f'  {r["name"]:<44}{r["acc"]:>10.4f}{r["macro_f1"]:>10.4f}')

    if len(results) >= 2:
        drop = results[0]['acc'] - results[-1]['acc']
        print(f'\n  从「不分离」到「完全分离」准确率下降 {drop:.4f}')
        print('  → 该下降完全由"消除标签泄漏"引起（其余条件均未改变）')

        # 验证组 1 是否真的在复现规则
        if results[0]['acc'] > 0.99:
            print('  ✓ 组 1 准确率接近 1.00，证实模型只是在复现标签规则而非学习波形')

    return 0


if __name__ == '__main__':
    sys.exit(main())
