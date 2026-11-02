# -*- coding: utf-8 -*-
"""
窗口长度对比实验 —— 15s / 30s / 45s / 60s
=============================================
文献（Spijkerboer 2024）用 90s 窗口，指出更长窗口更易区分
伪影/真性呼吸暂停，周期特征更稳定。

本实验：
  用不同窗口长度（半重叠，步长=窗口/2），同一标签规则、同一 12 维特征，
  对比 GroupKFold CV 的 Macro-F1 和每类 P/R。

注意：特征提取中的"波形自算RR"(F11) 按 30s 窗口固定时长换算，
      长窗口需修正为按实际窗口长度。
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
from training.train_rf import split_final_test
from collections import Counter

SAMPLING_RATE = 100
N_CASES = 150
WINDOW_SECS = [90, 120]


def build_dataset(window_size, step):
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    X_list, y_list, cases_list = [], [], []
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid)
            records = make_windows_with_reference(
                ref['co2'], ref['rr'], ref['etco2'],
                window_size=window_size, step=step, sampling_rate=SAMPLING_RATE)
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


def run_cv(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    y_t, y_p = [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[val_idx])
        y_t.extend(y[val_idx].tolist())
        y_p.extend(yp.tolist())
    return np.array(y_t), np.array(y_p)


def run_final_test(X, y, groups, test_ratio=0.2):
    """锁定 Final Test 病例（模型完全未见），训练最终模型并评估。"""
    final_test_cases, cv_cases = split_final_test(groups, test_ratio=test_ratio)
    train_mask = np.isin(groups, final_test_cases, invert=True)
    test_mask = np.isin(groups, final_test_cases)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=42, class_weight='balanced')
    clf.fit(X[train_mask], y[train_mask])
    y_t, y_p = y[test_mask], clf.predict(X[test_mask])
    return y_t, y_p


NAMES = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}


def main():
    print(f'窗口长度对比实验（{N_CASES} 病例, 60/90s）...')
    results = []
    for sec in WINDOW_SECS:
        win = sec * SAMPLING_RATE
        step = win // 2  # 半重叠
        print(f'\n构建数据集: {sec}s 窗口 (step={step})')
        X, y, g = build_dataset(win, step)
        dist = {NAMES[int(k)]: int(v) for k, v in Counter(y).items()}
        print(f'  样本 {len(y)}, 分布 {dist}')
        y_t, y_p = run_cv(X, y, g)
        acc = accuracy_score(y_t, y_p)
        macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        per = {NAMES[int(k)]: round(f1_score(y_t == k, y_p == k, zero_division=0), 3)
               for k in sorted(set(y_t) | set(y_p))}
        print(f'  CV: Accuracy={acc:.4f}, Macro-F1={macro:.4f}')
        print(f'  CV 每类F1: {per}')

        # Final Test
        y_ft, y_fp = run_final_test(X, y, g)
        ft_acc = accuracy_score(y_ft, y_fp)
        ft_macro = f1_score(y_ft, y_fp, average='macro', zero_division=0)
        ft_per = {NAMES[int(k)]: round(f1_score(y_ft == k, y_fp == k, zero_division=0), 3)
                  for k in sorted(set(y_ft) | set(y_fp))}
        print(f'  FinalTest(20%锁定病例, {len(y_ft)}样本): '
              f'Accuracy={ft_acc:.4f}, Macro-F1={ft_macro:.4f}')
        print(f'  FinalTest 每类F1: {ft_per}')
        results.append((sec, len(y), acc, macro, per, dist, ft_macro, ft_per))

    print('\n' + '=' * 70)
    print('对比总结（按 Final Test Macro-F1 排序）')
    print('=' * 70)
    for sec, n, acc, macro, per, dist, ft_macro, ft_per in sorted(results, key=lambda x: -x[6]):
        print(f'\n{sec}s 窗口 ({n} 样本)')
        print(f'  CV: Accuracy={acc:.4f} | Macro-F1={macro:.4f} | 每类 {per}')
        print(f'  FT: Macro-F1={ft_macro:.4f} | 每类 {ft_per}')
        print(f'  分布: {dist}')


if __name__ == '__main__':
    main()
