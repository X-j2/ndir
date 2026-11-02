# -*- coding: utf-8 -*-
"""
RF 瘦身实验 —— 减树数/深度对精度和 ONNX 大小的影响
=====================================================
背景：F407 1MB Flash 放不下 1.9MB ONNX。
RF 是树模型，大小来自节点数（不是权重），量化无效，需减树/减深。

本实验对比 (n_estimators, max_depth) 组合：
  - Final Test Macro-F1 / Accuracy
  - ONNX 文件大小（估算，能否塞进 1MB Flash）

注意：树模型 int8 量化无效，故只看节点数瘦身。
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

CONFIGS = [
    # (n_estimators, max_depth, 标签)
    (100, 8, '当前 (100树/depth8)'),
    (50, 8, '50树/depth8'),
    (30, 8, '30树/depth8'),
    (100, 6, '100树/depth6'),
    (50, 6, '50树/depth6'),
    (30, 6, '30树/depth6'),
    (50, 10, '50树/depth10'),
]


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


def main():
    print(f'RF 瘦身实验（{N_CASES} 病例, 90s窗口, 4类）...')
    X, y, g = build_dataset()
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')

    final_test_cases, _ = split_final_test(g, test_ratio=0.2)
    train_mask = np.isin(g, final_test_cases, invert=True)
    test_mask = np.isin(g, final_test_cases)

    print('\n' + '=' * 70)
    print(f'{"配置":<22} {"Acc":>6} {"Macro-F1":>9} {"ONNX估":>8}')
    print('=' * 70)

    results = []
    for n_est, depth, label in CONFIGS:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                     random_state=42, class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
        clf.fit(X[train_mask], y[train_mask])
        yp = clf.predict(X[test_mask])
        yt = y[test_mask]
        acc = accuracy_score(yt, yp)
        macro = f1_score(yt, yp, average='macro', zero_division=0)

        # ONNX 大小估算：用树节点数 × 每节点字节数（经验 ~24 字节/节点）
        n_nodes = sum(clf.estimators_[i].tree_.node_count for i in range(len(clf.estimators_)))
        onnx_est_kb = n_nodes * 24 / 1024
        print(f'{label:<22} {acc:.4f} {macro:.4f} {onnx_est_kb:7.0f}KB')
        results.append((label, acc, macro, onnx_est_kb))

    print('\n' + '=' * 70)
    print('对比总结')
    print('=' * 70)
    for label, acc, macro, kb in results:
        flag = '✅' if kb < 800 else '❌>1MB'
        print(f'{label:<22} Macro-F1={macro:.4f} ONNX≈{kb:.0f}KB {flag}')


if __name__ == '__main__':
    main()
