# -*- coding: utf-8 -*-
"""
缓慢+异常合并实验 —— 5类 vs 4类（合并缓慢与异常）
=====================================================
背景：缓慢(2) precision 极低(0.17)，因标签把"低频+任意浓度"混在一起，
     与异常(4) 在低频+边界浓度区域严重重叠、几乎不可分。

合并思路：缓慢(2) + 异常(4) → 新"异常(4)"（非正常频率或浓度）
  0 正常 | 1 急促 | 3 屏息 | 4 呼吸异常(含低频/浓度异常)

对比（90s 窗口，150 病例，CV + Final Test）：
  5类（当前） vs 4类（合并）
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
WINDOW_SIZE = 9000    # 90s 窗口（当前基线）
WINDOW_STEP = 4500
N_CASES = 150


def build_dataset(merge_slow_abn=False):
    """构建数据集。merge_slow_abn=True 时缓慢(2)并入异常(4)。"""
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
            lb = res['label']
            if merge_slow_abn and lb == 2:
                lb = 4  # 缓慢并入异常
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(lb)
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


def run_final_test(X, y, groups):
    final_test_cases, _ = split_final_test(groups, test_ratio=0.2)
    train_mask = np.isin(groups, final_test_cases, invert=True)
    test_mask = np.isin(groups, final_test_cases)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=42, class_weight='balanced')
    clf.fit(X[train_mask], y[train_mask])
    return y[test_mask], clf.predict(X[test_mask])


NAMES5 = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}
NAMES4 = {0: '正常', 1: '急促', 3: '屏息', 4: '呼吸异常'}


def report(yt, yp, names, title):
    acc = accuracy_score(yt, yp)
    macro = f1_score(yt, yp, average='macro', zero_division=0)
    print(f'\n  {title}: Accuracy={acc:.4f}, Macro-F1={macro:.4f}')
    labels = sorted(set(yt) | set(yp))
    print(classification_report(yt, yp, labels=labels, zero_division=0,
                                target_names=[names.get(l, str(l)) for l in labels]))


def main():
    print(f'缓慢+异常合并实验（{N_CASES} 病例, 90s窗口）...')

    # 5类
    print('\n=== 5 类（当前） ===')
    X, y, g = build_dataset(merge_slow_abn=False)
    print(f'  样本 {len(y)}, 分布 {Counter(y)}')
    y_cv_t, y_cv_p = run_cv(X, y, g)
    report(y_cv_t, y_cv_p, NAMES5, 'CV')
    y_ft_t, y_ft_p = run_final_test(X, y, g)
    report(y_ft_t, y_ft_p, NAMES5, 'FinalTest')

    # 4类合并
    print('\n=== 4 类（合并缓慢+异常） ===')
    X2, y2, g2 = build_dataset(merge_slow_abn=True)
    print(f'  样本 {len(y2)}, 分布 {Counter(y2)}')
    y_cv_t2, y_cv_p2 = run_cv(X2, y2, g2)
    report(y_cv_t2, y_cv_p2, NAMES4, 'CV')
    y_ft_t2, y_ft_p2 = run_final_test(X2, y2, g2)
    report(y_ft_t2, y_ft_p2, NAMES4, 'FinalTest')

    print('\n' + '=' * 70)
    print('对比总结')
    print('=' * 70)
    print(f'5类 FT Macro-F1: {f1_score(y_ft_t, y_ft_p, average="macro", zero_division=0):.4f}')
    print(f'4类 FT Macro-F1: {f1_score(y_ft_t2, y_ft_p2, average="macro", zero_division=0):.4f}')


if __name__ == '__main__':
    main()
