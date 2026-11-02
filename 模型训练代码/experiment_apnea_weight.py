# -*- coding: utf-8 -*-
"""
屏息加权实验 —— 提高屏息类权重，改善 precision 0.46
=====================================================
误报分析（experiment_apnea_analysis.py）：
  真屏息(357) 峰峰值 0.2/峰数 0，误判窗口(310) 峰峰值 23/峰数 2。
  模型过度依赖"峰数"判屏息，把低频异常也判成屏息。

本实验：
  1. 对比不同屏息类权重（class_weight 自定义: 屏息×N）
  2. 看屏息 precision/recall 能否提升，且不牺牲其他类
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 150

NAMES = {0: '正常', 1: '急促', 3: '屏息', 4: '呼吸异常'}


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
            if co2 is None:
                continue
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(res['label'])
            cases_list.append(cid)
    return np.array(X_list), np.array(y_list), np.array(cases_list)


def run_cv(X, y, groups, apnea_weight=1.0):
    """apnea_weight: 屏息类的额外权重。"""
    gkf = GroupKFold(n_splits=5)
    y_t, y_p = [], []
    # 自定义 class_weight：其他类 1.0，屏息 ×apnea_weight
    cnt = Counter(y)
    weights = {}
    for c in cnt:
        if c == 3:
            weights[c] = apnea_weight
        else:
            weights[c] = 1.0
    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, class_weight=weights)
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[val_idx])
        y_t.extend(y[val_idx].tolist())
        y_p.extend(yp.tolist())
    return np.array(y_t), np.array(y_p)


def main():
    print(f'屏息加权实验（{N_CASES} 病例, 90s窗口, 4类）...')
    X, y, g = build_dataset()
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    for w in [1.0, 2.0, 3.0, 5.0, 8.0]:
        print(f'\n=== 屏息权重 {w} ===')
        y_t, y_p = run_cv(X, y, g, apnea_weight=w)
        acc = accuracy_score(y_t, y_p)
        macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        # 屏息 P/R/F1
        ap_p = y_p[y_t == 3]
        ap_recall = (ap_p == 3).mean() if len(ap_p) else 0
        pred_ap = y_p[y_p == 3]
        ap_prec = (y_t[y_p == 3] == 3).mean() if len(pred_ap) else 0
        ap_f1 = 2 * ap_prec * ap_recall / (ap_prec + ap_recall) if ap_prec + ap_recall else 0
        print(f'  Accuracy={acc:.4f}, Macro-F1={macro:.4f}')
        print(f'  屏息: precision={ap_prec:.3f}, recall={ap_recall:.3f}, F1={ap_f1:.3f}')
        # 其他类 F1
        for c in [0, 1, 4]:
            cf1 = f1_score(y_t == c, y_p == c, zero_division=0)
            print(f'    {NAMES[c]} F1={cf1:.3f}')

    print('\n实验完成')


if __name__ == '__main__':
    main()
