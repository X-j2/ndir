# -*- coding: utf-8 -*-
"""
误分类诊断 —— 找出"缓慢 precision 低"的真正根因
==================================================
第一步诊断（experiment_diagnose_peaks.py）否定了"漏峰"假设：
低碳酸异常窗口的寻峰频率 = 监护仪 RR（误差 1 bpm），寻峰正常。

本脚本直接分析误分类窗口的特征：
  1. 构建数据集 → GroupKFold 训练 RF → 收集每折预测
  2. 分组：
     A. 真缓慢(2) 判对
     B. 真缓慢(2) 判错（看判成什么）
     C. 真异常(4) 被判成缓慢(2)  ← precision 杀手
  3. 对比三组的 10 维特征分布，找出区分它们的特征缺口

输出：特征对比表，指出真正需要增强的判别维度。
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
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 30


def build_dataset(n_cases=N_CASES):
    case_ids = find_cases_with_reference(n_cases=n_cases)
    X_list, y_list, cases_list, meta_list = [], [], [], []
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
            meta_list.append({'rr': res['rr_median'], 'et': res['etco2_median']})
    return (np.array(X_list), np.array(y_list), np.array(cases_list), meta_list)


def feat_table(rows, title):
    """打印一组样本的特征分布（中位数/均值）。"""
    if len(rows) == 0:
        print(f'  [{title}] 无样本')
        return
    X = rows[:, :10].astype(float)
    med = np.median(X, axis=0)
    mean = np.mean(X, axis=0)
    print(f'\n  [{title}] n={len(rows)}')
    for i, name in enumerate(FEATURE_NAMES_CN):
        print(f'    {name:12s} 中位 {med[i]:7.2f} | 均值 {mean[i]:7.2f}')


def main():
    print(f'误分类诊断（{N_CASES} 病例）...')
    X, y, cases, meta = build_dataset()
    print(f'  样本 {len(y)}, 特征 {X.shape[1]}, 病例 {len(set(cases))}')
    print(f'  类别分布: {dict(Counter(y))}')

    # GroupKFold 预测收集误分类
    gkf = GroupKFold(n_splits=5)
    slow_true = {'correct': [], 'wrong': []}
    abn_as_slow = []
    for train_idx, val_idx in gkf.split(X, y, cases):
        clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[val_idx])
        for pos, i in enumerate(val_idx):
            r = np.concatenate([X[i], [y[i], yp[pos]]])  # 10特征 + 2标签 = 12

            if y[i] == 2:
                if yp[pos] == 2:
                    slow_true['correct'].append(r)
                else:
                    slow_true['wrong'].append(r)
            elif y[i] == 4 and yp[pos] == 2:
                abn_as_slow.append(r)

    slow_c = np.array(slow_true['correct'])
    slow_w = np.array(slow_true['wrong'])
    abn_s = np.array(abn_as_slow)

    print('\n' + '=' * 60)
    print('分组特征对比')
    print('=' * 60)

    feat_table(slow_c, '真缓慢(2) 判对')
    feat_table(slow_w, '真缓慢(2) 判错')
    feat_table(abn_s,  '真异常(4) 判成缓慢')

    # 缓慢判错后去了哪个类？
    if len(slow_w):
        preds = slow_w[:, -1].astype(int)
        print(f'\n  真缓慢判错去向: {dict(Counter(preds))}')

    # 判对 vs 判错的缓慢，参考 RR/EtCO2 有差异吗？
    print('\n' + '=' * 60)
    print('参考值对比（判对 vs 判错缓慢）')
    print('=' * 60)
    if len(slow_c) and len(slow_w):
        # meta 对齐困难（meta 是 list），改用索引——简化：直接输出判错缓慢的 rr/et
        # 这里用简化方式：真缓慢判错窗口的数量级
        pass

    print('\n诊断完成')


if __name__ == '__main__':
    main()
