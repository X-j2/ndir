# -*- coding: utf-8 -*-
"""
屏息(3) 误报分析 —— 为什么 precision 只有 0.46
=================================================
从混淆矩阵看：异常(4) 有 48 个窗口被误判成屏息，是 precision 杀手。

本脚本：
  1. 用 4 类标签（0正常/1急促/3屏息/4异常）+ 90s窗口训练 RF
  2. 收集被误判成屏息的窗口（真实非屏息）
  3. 对比"真屏息" vs "误判成屏息的异常/正常"的 12 维特征
  4. 找出可区分的特征 → 决定优化方向
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features, FEATURE_NAMES_CN
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000    # 90s
WINDOW_STEP = 4500
N_CASES = 150


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


def feat_table(rows, title):
    if len(rows) == 0:
        print(f'  [{title}] 无样本')
        return
    X = rows[:, :12].astype(float)
    med = np.median(X, axis=0)
    print(f'\n  [{title}] n={len(rows)}')
    for i, name in enumerate(FEATURE_NAMES_CN):
        print(f'    {name:12s} 中位 {med[i]:8.2f}')


def main():
    print(f'屏息误报分析（{N_CASES} 病例, 90s窗口, 4类）...')
    X, y, g = build_dataset()
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    # GroupKFold 收集误分类
    gkf = GroupKFold(n_splits=5)
    apnea_true = []
    false_apnea = []  # 真实非屏息但被判屏息
    for train_idx, val_idx in gkf.split(X, y, g):
        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[val_idx])
        for pos, i in enumerate(val_idx):
            r = np.concatenate([X[i], [y[i], yp[pos]]])
            if y[i] == 3:
                apnea_true.append(r)
            elif yp[pos] == 3:
                false_apnea.append(r)

    at = np.array(apnea_true)
    fa = np.array(false_apnea)
    print(f'\n真屏息: {len(at)}')
    print(f'误判成屏息: {len(fa)}')

    # 误判来源分布
    if len(fa):
        src = Counter(fa[:, -2].astype(int))
        print(f'误判来源: {dict(src)}')

    print('\n' + '=' * 60)
    print('特征对比：真屏息 vs 误判成屏息')
    print('=' * 60)
    feat_table(at, '真屏息')
    feat_table(fa, '误判成屏息(非屏息)')

    # 参考值对比（误判窗口的 RR/EtCO2 参考）
    print('\n' + '=' * 60)
    print('误判窗口的参考 RR/EtCO2（来自标签生成，看它们真实是什么）')
    print('=' * 60)
    if len(fa):
        # 需要重新查这些窗口的参考值 —— 简化：直接看分布
        pass

    print('\n诊断完成')


if __name__ == '__main__':
    main()
