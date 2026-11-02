# -*- coding: utf-8 -*-
"""实验 B：病例级划分 vs 窗口级划分 的对照实验

论文问题
--------
同一病例的相邻窗口是 90 秒窗、45 秒步长切出来的——相邻窗口重叠 50%，
波形高度相似，标签也大概率相同。若按**窗口**随机划分，同一病例的
窗口会同时出现在训练集和测试集，模型在测试时"见过"几乎相同的波形，
指标被系统性高估。

本实验用**同一批数据、同一个模型、同一个随机种子**，
只改划分方式，量化这种高估的幅度。

两组对照
--------
  窗口级划分  StratifiedKFold / 分层随机划分
              → 同一病例的窗口可跨越训练/测试集（数据泄漏）

  病例级划分  GroupKFold / 病例锁定
              → 同一病例的全部窗口只在一侧（当前生产方案）

预期：窗口级显著高于病例级；差值即为"病例内相似性"带来的虚高。

用法：python experiment_split_strategy.py
"""
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200
RANDOM_STATE = 42
N_SPLITS = 5

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'vitaldb')


def build_model():
    """与部署模型完全一致的随机森林配置。"""
    return RandomForestClassifier(n_estimators=30, max_depth=8,
                                  random_state=RANDOM_STATE,
                                  class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})


def load_data(n_cases=N_CASES):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))[:n_cases]
    print(f'  从缓存加载 {len(files)} 个病例...')

    X, y, cases = [], [], []
    for path in files:
        cid = int(os.path.basename(path).split('_')[0])
        try:
            ref = load_reference_tracks(cid, interval=0.01)
        except Exception:
            continue

        records = make_windows_with_reference(
            ref['co2'], ref['rr'], ref['etco2'],
            window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)

        for rec in records:
            lab = build_labels_for_window(None, rec['rr'], rec['etco2'],
                                          min_reference_points=2)
            if lab['label'] is None:
                continue
            co2, _ = preprocess_window(rec['co2'], sampling_rate=SAMPLING_RATE)
            X.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y.append(int(lab['label']))
            cases.append(cid)

    X = np.array(X)
    y = np.array(y)
    cases = np.array(cases)
    print(f'  有效样本 {len(y)}，病例 {len(set(cases))} 个')
    return X, y, cases


# ============================================================
# 划分方式一：窗口级（有泄漏）
# ============================================================
def eval_window_level(X, y, cases, n_splits=N_SPLITS):
    """分层随机划分——同一病例的窗口可同时出现在训练集与测试集。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    leaked_counts = []

    for train_idx, test_idx in skf.split(X, y):
        clf = build_model()
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])

        accs.append(float(np.mean(pred == y[test_idx])))
        f1s.append(float(f1_score(y[test_idx], pred, average='macro', zero_division=0)))

        # 统计有多少测试窗口的"同病例兄弟窗口"出现在训练集里
        train_cases = set(cases[train_idx])
        leaked = sum(1 for c in cases[test_idx] if c in train_cases)
        leaked_counts.append(leaked / len(test_idx))

    return accs, f1s, float(np.mean(leaked_counts))


# ============================================================
# 划分方式二：病例级（无泄漏）
# ============================================================
def eval_case_level(X, y, cases, n_splits=N_SPLITS):
    """GroupKFold——同一病例的窗口全部落在同一侧。"""
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []

    for train_idx, test_idx in gkf.split(X, y, cases):
        clf = build_model()
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])

        accs.append(float(np.mean(pred == y[test_idx])))
        f1s.append(float(f1_score(y[test_idx], pred, average='macro', zero_division=0)))

    return accs, f1s, 0.0


# ============================================================
# 划分方式三：锁定 Final Test（当前生产方案）
# ============================================================
def eval_final_test(X, y, cases, test_ratio=0.2):
    unique_cases = np.unique(cases)
    test_cases, train_cases = train_test_split(
        unique_cases, test_size=1 - test_ratio, random_state=RANDOM_STATE)

    train_mask = np.isin(cases, train_cases)
    test_mask = np.isin(cases, test_cases)

    clf = build_model()
    clf.fit(X[train_mask], y[train_mask])
    pred = clf.predict(X[test_mask])

    acc = float(np.mean(pred == y[test_mask]))
    f1 = float(f1_score(y[test_mask], pred, average='macro', zero_division=0))
    return acc, f1, int(test_mask.sum()), len(test_cases)


def main():
    print('=' * 66)
    print('实验 B：病例级划分 vs 窗口级划分')
    print('=' * 66)
    print(f'配置：{N_CASES} 病例 / {WINDOW_SIZE // SAMPLING_RATE}s 窗口 / 步长 '
          f'{WINDOW_STEP // SAMPLING_RATE}s（半重叠）/ RF 30树')
    print('三组仅"划分方式"不同，数据与模型完全相同\n')

    X, y, cases = load_data()

    print('\n' + '-' * 66)
    print('  方式一：窗口级划分（StratifiedKFold，有泄漏）')
    print('-' * 66)
    acc_w, f1_w, leak_ratio = eval_window_level(X, y, cases)
    for i, (a, f) in enumerate(zip(acc_w, f1_w)):
        print(f'    Fold {i + 1}: acc={a:.4f}  macro-F1={f:.4f}')
    print(f'    均值: acc={np.mean(acc_w):.4f}  macro-F1={np.mean(f1_w):.4f}')
    print(f'    ⚠ {leak_ratio * 100:.1f}% 的测试窗口，其同病例窗口出现在训练集中')

    print('\n' + '-' * 66)
    print('  方式二：病例级划分（GroupKFold，无泄漏）')
    print('-' * 66)
    acc_g, f1_g, _ = eval_case_level(X, y, cases)
    for i, (a, f) in enumerate(zip(acc_g, f1_g)):
        print(f'    Fold {i + 1}: acc={a:.4f}  macro-F1={f:.4f}')
    print(f'    均值: acc={np.mean(acc_g):.4f}  macro-F1={np.mean(f1_g):.4f}')

    print('\n' + '-' * 66)
    print('  方式三：锁定 Final Test 病例（当前生产方案）')
    print('-' * 66)
    acc_f, f1_f, n_test, n_cases_test = eval_final_test(X, y, cases)
    print(f'    Final Test: {n_cases_test} 病例 / {n_test} 样本')
    print(f'    acc={acc_f:.4f}  macro-F1={f1_f:.4f}')

    # ---- 汇总 ----
    print('\n' + '=' * 66)
    print('  汇总（论文可直接引用）')
    print('=' * 66)
    print(f'  {"划分方式":<34}{"Accuracy":>10}{"Macro-F1":>10}')
    print('  ' + '-' * 62)
    print(f'  {"窗口级 StratifiedKFold（有泄漏）":<34}'
          f'{np.mean(acc_w):>10.4f}{np.mean(f1_w):>10.4f}')
    print(f'  {"病例级 GroupKFold（无泄漏）":<34}'
          f'{np.mean(acc_g):>10.4f}{np.mean(f1_g):>10.4f}')
    print(f'  {"病例级 Final Test（生产方案）":<34}'
          f'{acc_f:>10.4f}{f1_f:>10.4f}')

    gap = np.mean(acc_w) - np.mean(acc_g)
    print(f'\n  窗口级比病例级虚高 {gap:.4f}（{gap * 100:.2f} 个百分点）')
    print('  → 该差值即"同一病例相邻窗口高度相似"带来的乐观偏差')
    print('  → 论文中的泛化性能应引用病例级数值')

    return 0


if __name__ == '__main__':
    sys.exit(main())
