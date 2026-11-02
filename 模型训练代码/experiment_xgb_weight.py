# -*- coding: utf-8 -*-
"""
XGBoost 带类别权重实验
=======================
默认 XGBoost 的 Macro-F1 低于 RF（0.675 vs 0.697），
因为可能没正确处理类别不均衡。测试加权 XGBoost 能否超越。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from xgboost import XGBClassifier

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 100


def build_dataset():
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    X_list, y_list, cases_list = [], [], []
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
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(res['label'])
            cases_list.append(cid)
    return np.array(X_list), np.array(y_list), np.array(cases_list)


def run_cv(X, y, groups, weighted=False):
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    # 全局样本权重
    cnt = Counter(y)
    max_c = max(cnt.values())
    w_array = np.array([max_c / cnt[c] for c in y]) if weighted else None

    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = XGBClassifier(n_estimators=100, max_depth=8, random_state=42,
                            learning_rate=0.1, eval_metric='mlogloss')
        if weighted:
            clf.fit(X[train_idx], y[train_idx], sample_weight=w_array[train_idx])
        else:
            clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], yp))
        f1s.append(f1_score(y[val_idx], yp, average='macro', zero_division=0))
    return np.mean(accs), np.mean(f1s)


if __name__ == '__main__':
    print('构建数据集...')
    X, y, groups = build_dataset()
    cnt = Counter(y)
    print(f'样本: {len(y)}, 类别分布: {dict(cnt)}')

    print('\n=== XGBoost 默认 ===')
    acc, f1 = run_cv(X, y, groups, weighted=False)
    print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

    print('\n=== XGBoost 样本加权（少数类×max/cnt）===')
    acc, f1 = run_cv(X, y, groups, weighted=True)
    print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
