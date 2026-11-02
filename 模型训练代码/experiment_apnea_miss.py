# -*- coding: utf-8 -*-
"""
屏息漏检分析 —— 200病例 + 屏息×3权重下，漏检的屏息窗口特征
=============================================================
从混淆矩阵：屏息 80 个里 21 个漏到异常(4)（recall 0.74）。

问题：漏检是因为"样本少"（插值可解）还是"特征边界重叠"（插值无效）？
验证：
  1. 收集 漏检屏息（真3判4）vs 真屏息（判对）vs 真异常
  2. 对比特征，看漏检窗口是否"不像屏息、像异常"
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
from sklearn.metrics import f1_score, accuracy_score
from training.train_rf import split_final_test
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200


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


def feat_stats(rows, title):
    if len(rows) == 0:
        print(f'  [{title}] 无样本')
        return
    X = rows[:, :12].astype(float)
    med = np.median(X, axis=0)
    print(f'\n  [{title}] n={len(rows)}')
    for i, name in enumerate(FEATURE_NAMES_CN):
        print(f'    {name:12s} 中位 {med[i]:8.2f}')


def main():
    print(f'屏息漏检分析（{N_CASES} 病例, 90s窗口, 4类, 屏息×3）...')
    X, y, g = build_dataset()
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    # 与 train.py 一致的 Final Test 划分
    final_test_cases, _ = split_final_test(g, test_ratio=0.2)
    train_mask = np.isin(g, final_test_cases, invert=True)
    test_mask = np.isin(g, final_test_cases)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                 class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
    clf.fit(X[train_mask], y[train_mask])
    yp = clf.predict(X[test_mask])

    yt, y_pred = y[test_mask], yp
    print(f'  FinalTest: Accuracy={accuracy_score(yt, y_pred):.4f}, '
          f'Macro-F1={f1_score(yt, y_pred, average="macro", zero_division=0):.4f}')

    # 收集窗口
    correct_apnea, miss_apnea, true_abn = [], [], []
    for i in range(len(yt)):
        r = np.concatenate([X[test_mask][i], [yt[i], y_pred[i]]])
        if yt[i] == 3 and y_pred[i] == 3:
            correct_apnea.append(r)
        elif yt[i] == 3 and y_pred[i] != 3:
            miss_apnea.append(r)
        elif yt[i] == 4:
            true_abn.append(r)

    ca = np.array(correct_apnea)
    ma = np.array(miss_apnea)
    ta = np.array(true_abn)

    print(f'\n  判对屏息: {len(ca)}')
    print(f'  漏检屏息: {len(ma)} (去向 {Counter(ma[:, -1].astype(int)) if len(ma) else "-"})')
    print(f'  真异常(参考): {len(ta)}')

    print('\n' + '=' * 60)
    print('特征对比')
    print('=' * 60)
    feat_stats(ca, '真屏息-判对')
    feat_stats(ma, '真屏息-漏检')
    feat_stats(ta, '真异常(参考)')

    # 漏检屏息的标签参考值（RR/EtCO2）—— 看它们为什么被判异常
    print('\n结论判断：')
    if len(ma) and len(ca):
        print(f'  漏检屏息与判对屏息的特征差异 → 决定插值是否有效')
    print('诊断完成')


if __name__ == '__main__':
    main()
