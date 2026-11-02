# -*- coding: utf-8 -*-
"""
规则引擎 vs 随机森林 对比实验（含噪声鲁棒性）
================================================
核心问题：如果标签是阈值规则，那"从波形算 RR/EtCO₂ + 规则引擎"就够了，
          为什么还要随机森林？

本实验回答：
  1. 在干净波形下，规则引擎（波形自算RR/EtCO₂ → 阈值）与 RF 谁准？
  2. 在加噪波形下（真实设备噪声），谁更鲁棒？
  3. 结论：RF 的价值是否在"噪声鲁棒性"？

规则引擎设计（只用波形，与 RF 相同输入信息）：
  - 自算RR = 峰数 × 60 / 90s
  - 自算EtCO2 = 波形 95 百分位（呼气末浓度近似）
  - 套 4 类阈值规则（与标签规则一致）分类

噪声注入：对 CO₂ 波形加高斯噪声（不同 σ）模拟传感器噪声。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import find_breath_peaks, extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from training.train_rf import split_final_test
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200


def rule_engine_labels(co2, rr_ref=None):
    """规则引擎：从波形自算 RR/EtCO2，套 4 类阈值规则。

    co2: 预处理后的 CO₂ 波形（mmHg）
    rr_ref: 参考 RR（若提供则用参考值，模拟"完美估计"上界）
    """
    # 波形自算 RR（峰数 × 60/90s）
    peaks = find_breath_peaks(co2, sampling_rate=SAMPLING_RATE)
    wave_rr = len(peaks) * 60.0 / 90.0

    # 波形自算 EtCO2（呼气末近似 = 95 百分位）
    wave_et = np.percentile(co2, 95)

    # 套 4 类阈值规则（与标签规则一致）
    if wave_rr < 3:
        return 3  # 屏息
    if wave_rr >= 18:
        return 1  # 急促
    if wave_rr <= 6 or wave_et < 35 or wave_et > 45:
        return 4  # 异常
    return 0  # 正常


def add_noise(co2, sigma):
    """给波形加高斯噪声（sigma mmHg）。"""
    return co2 + np.random.normal(0, sigma, len(co2))


def build_dataset():
    """构建 (原始波形, y_true, groups)。"""
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    waves, y_list, cases_list = [], [], []
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
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue
            waves.append(co2)
            y_list.append(res['label'])
            cases_list.append(cid)
    return waves, np.array(y_list), np.array(cases_list)


def run_rule_engine(waves, y_true, groups, noise_sigmas=[0, 2, 5, 10]):
    """规则引擎：不同噪声下评估。"""
    final_test_cases, _ = split_final_test(groups, test_ratio=0.2)
    test_mask = np.isin(groups, final_test_cases)

    print('\n=== 规则引擎（波形自算RR/EtCO2 → 阈值规则） ===')
    for sigma in noise_sigmas:
        np.random.seed(0)
        preds = []
        for i, w in enumerate(waves):
            if not test_mask[i]:
                continue
            wn = add_noise(w, sigma) if sigma > 0 else w
            preds.append(rule_engine_labels(wn))
        yt = y_true[test_mask]
        yp = np.array(preds)
        acc = accuracy_score(yt, yp)
        macro = f1_score(yt, yp, average='macro', zero_division=0)
        print(f'  噪声σ={sigma:2d}: Accuracy={acc:.4f}, Macro-F1={macro:.4f}')


def run_rf(waves, y_true, groups, noise_sigmas=[0, 2, 5, 10]):
    """随机森林：不同噪声下评估（训练干净，测试加噪 = 部署场景）。"""
    final_test_cases, _ = split_final_test(groups, test_ratio=0.2)
    train_mask = np.isin(groups, final_test_cases, invert=True)
    test_mask = np.isin(groups, final_test_cases)

    # 训练特征（干净波形）
    X_tr = np.array([extract_features(w, sampling_rate=SAMPLING_RATE)
                     for w, m in zip(waves, train_mask) if m])
    y_tr = y_true[train_mask]
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                 class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
    clf.fit(X_tr, y_tr)
    yt = y_true[test_mask]

    print('\n=== 随机森林（干净训练，噪声测试） ===')
    for sigma in noise_sigmas:
        np.random.seed(0)
        X_te = []
        for i, w in enumerate(waves):
            if not test_mask[i]:
                continue
            wn = add_noise(w, sigma) if sigma > 0 else w
            X_te.append(extract_features(wn, sampling_rate=SAMPLING_RATE))
        yp = clf.predict(np.array(X_te))
        acc = accuracy_score(yt, yp)
        macro = f1_score(yt, yp, average='macro', zero_division=0)
        print(f'  噪声σ={sigma:2d}: Accuracy={acc:.4f}, Macro-F1={macro:.4f}')


def main():
    print(f'规则引擎 vs 随机森林（{N_CASES} 病例, 90s窗口, 4类）...')
    print('构建数据集...')
    waves, y, g = build_dataset()
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    print('\n' + '=' * 60)
    print('规则引擎')
    print('=' * 60)
    run_rule_engine(waves, y, g)

    print('\n' + '=' * 60)
    print('随机森林')
    print('=' * 60)
    run_rf(waves, y, g)

    print('\n' + '=' * 60)
    print('对比总结')
    print('=' * 60)
    print('关注：噪声σ增大时，谁的性能下降更快')


if __name__ == '__main__':
    main()
