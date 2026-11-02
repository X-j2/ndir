# -*- coding: utf-8 -*-
"""
优化实验 —— 对比不同标签策略/特征/模型参数
============================================
目的：找出比当前基线（8特征+监护仪标签+RF）更好的配置。

实验维度：
  A. 标签策略：只用参考值充足窗口 vs 放宽
  B. 特征工程：当前8特征 vs 增加峰数特征/周期变异系数等
  C. 模型：RF 不同 max_depth / 类别权重

一次性跑完，输出各配置的 CV 指标，选最优。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features, find_breath_peaks
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.signal import find_peaks

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
N_CASES = 30  # 用前30个病例做实验（快一些）

# ============================================================
# 1. 加载数据 + 基础特征（8维）和扩展特征
# ============================================================
def build_dataset(n_cases=30, extra_feature=False):
    X_list, y_list, cases_list = [], [], []
    case_ids = list(range(1, n_cases + 1))
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid)
            records = make_windows_with_reference(ref['co2'], ref['rr'], ref['etco2'])
        except Exception:
            continue
        for r in records:
            res = build_labels_for_window(None, r['rr'], r['etco2'], min_reference_points=2)
            if res['label'] is None:
                continue
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            f = extract_features(co2, sampling_rate=SAMPLING_RATE)
            if extra_feature:
                # 增加峰数特征（呼吸次数，不直接等于RR）
                peaks = find_breath_peaks(co2, SAMPLING_RATE)
                n_peaks = len(peaks)
                # 周期变异系数 = 峰间距std/mean
                f = np.append(f, [n_peaks])
            X_list.append(f)
            y_list.append(res['label'])
            cases_list.append(cid)
    return np.array(X_list), np.array(y_list), np.array(cases_list)


def run_cv(X, y, cases, max_depth=8, class_weight='balanced', n_est=100):
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for tr_idx, va_idx in gkf.split(X, y, cases):
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                     random_state=42, class_weight=class_weight)
        clf.fit(X[tr_idx], y[tr_idx])
        yp = clf.predict(X[va_idx])
        accs.append(np.mean(yp == y[va_idx]))
        f1s.append(f1_score(y[va_idx], yp, average='macro', zero_division=0))
    return np.mean(accs), np.mean(f1s)


# ============================================================
# 2. 实验对比
# ============================================================
print('加载数据集（30病例）...')
X8, y, cases = build_dataset(n_cases=N_CASES, extra_feature=False)
X9, _, _ = build_dataset(n_cases=N_CASES, extra_feature=True)
print(f'  8特征: {X8.shape}, 9特征: {X9.shape}')

print('\n=== 实验 1: 基线（8特征, max_depth=8, balanced）===')
acc, f1 = run_cv(X8, y, cases)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

print('\n=== 实验 2: 9特征（+峰数）===')
acc, f1 = run_cv(X9, y, cases)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

print('\n=== 实验 3: 8特征, 更浅树(max_depth=6)===')
acc, f1 = run_cv(X8, y, cases, max_depth=6)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

print('\n=== 实验 4: 8特征, 深树(max_depth=12)===')
acc, f1 = run_cv(X8, y, cases, max_depth=12)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

print('\n=== 实验 5: 9特征, 不加类别权重 ===')
acc, f1 = run_cv(X9, y, cases, class_weight=None)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

print('\n=== 实验 6: 9特征, 更多树(200)===')
acc, f1 = run_cv(X9, y, cases, n_est=200)
print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
