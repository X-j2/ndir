# -*- coding: utf-8 -*-
"""
扩展缓存 + 训练对比 —— 150 vs 200 vs 250 病例
=================================================
步骤：
  1. 找到 250 个合格病例（VitalDB）
  2. 补齐缓存（新病例首次下载）
  3. 分别用 150/200/250 病例训练（4类合并、90s窗口、屏息×3权重）
  4. 对比 Final Test Macro-F1，观察数据量提升是否有效
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
import time

SAMPLING_RATE = 100
WINDOW_SIZE = 9000    # 90s
WINDOW_STEP = 4500
MAX_CASES = 250       # 最多 250 病例


def ensure_cache(case_ids, interval=0.01):
    """确保病例已缓存（缺失的下载）。返回需要新下载的数量。"""
    from labels.build_reference_labels import CACHE_DIR
    missing = []
    for cid in case_ids:
        p = os.path.join(CACHE_DIR, f'{cid}_i{interval}.npz')
        if not os.path.exists(p):
            missing.append(cid)
    print(f'  需新下载缓存: {len(missing)} 病例')
    if missing:
        t0 = time.time()
        for i, cid in enumerate(missing, 1):
            load_reference_tracks(cid, interval=interval, use_cache=True)
            if i % 20 == 0:
                print(f'    已下载 {i}/{len(missing)} ({time.time()-t0:.0f}s)')
        print(f'  新缓存完成: {len(missing)} 病例 ({time.time()-t0:.0f}s)')
    return len(missing)


def build_dataset(case_ids):
    """用指定病例构建数据集（4类标签、90s窗口）。"""
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


def train_eval(X, y, groups):
    """Final Test 评估（20%病例锁定，屏息×3权重）。"""
    final_test_cases, _ = split_final_test(groups, test_ratio=0.2)
    train_mask = np.isin(groups, final_test_cases, invert=True)
    test_mask = np.isin(groups, final_test_cases)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                 class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
    clf.fit(X[train_mask], y[train_mask])
    y_t, y_p = y[test_mask], clf.predict(X[test_mask])
    acc = accuracy_score(y_t, y_p)
    macro = f1_score(y_t, y_p, average='macro', zero_division=0)
    # 各类 F1
    per = {}
    for c in sorted(set(y_t) | set(y_p)):
        per[c] = round(f1_score(y_t == c, y_p == c, zero_division=0), 3)
    return acc, macro, per, len(y), len(set(test_mask.sum() * [0]) if False else groups)


def main():
    print(f'扩展缓存 + 训练对比（最多 {MAX_CASES} 病例）...')

    # 1. 找 250 个合格病例
    print('\n[1/4] 查找合格病例...')
    all_cases = find_cases_with_reference(n_cases=MAX_CASES)
    print(f'  找到 {len(all_cases)} 个')

    # 2. 补齐缓存
    print('\n[2/4] 补齐缓存（新病例下载）...')
    n_new = ensure_cache(all_cases)

    # 3. 分规模训练对比
    print('\n[3/4] 分规模训练对比...')
    results = []
    for n in [150, 200, 250]:
        if n > len(all_cases):
            print(f'  ⚠ 只有 {len(all_cases)} 病例，跳过 {n}')
            continue
        case_ids = all_cases[:n]
        print(f'\n  构建数据集: {n} 病例...')
        X, y, g = build_dataset(case_ids)
        dist = {int(k): int(v) for k, v in Counter(y).items()}
        print(f'    样本 {len(y)}, 分布 {dist}')
        acc, macro, per, n_ft, n_cases = train_eval(X, y, g)
        print(f'    FinalTest: Accuracy={acc:.4f}, Macro-F1={macro:.4f}, 各类F1={per}')
        results.append((n, len(y), acc, macro, per))

    # 4. 汇总
    print('\n[4/4] 对比总结')
    print('=' * 70)
    for n, nsamp, acc, macro, per in results:
        print(f'{n} 病例 ({nsamp} 样本): Accuracy={acc:.4f}, Macro-F1={macro:.4f}, 各类={per}')


if __name__ == '__main__':
    main()
