# -*- coding: utf-8 -*-
"""
边界缓冲区实验 —— 类别之间设缓冲区，缓冲窗口剔除
=====================================================
思路：标签是连续量阈值切分（RR 6.9 vs 7.0 一个异常一个正常），
     阈值附近的窗口标签不可靠 → 设缓冲区剔除，只训练"清晰核心"。

对比（200病例, 90s窗口, 4类, 屏息×3权重）：
  V0 无缓冲（当前）
  V1 训练剔除边界，测试保留全部（诚实评估：部署时边界窗口仍会出现）
  V2 训练+测试都剔除边界（指标虚高参考，不可部署）

缓冲区定义（对 4 类标签的关键阈值）：
  正常(0)/异常(4): RR 6.5-7.5 或 EtCO2 34-36
  屏息(3)/异常(4): RR 3-4
  急促(1)/异常(4): RR 17-19
  （这些区间的窗口标签不可靠，剔除）
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, window_reference_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from training.train_rf import split_final_test
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200

NAMES = {0: '正常', 1: '急促', 3: '屏息', 4: '异常'}


def in_margin(rr, et):
    """判断窗口是否落在边界缓冲区（标签不可靠区域）。

    返回 True = 剔除该窗口。
    """
    if rr is not None:
        if 3 <= rr < 5:          # 屏息/异常 边界
            return True
        if 6.5 <= rr < 7.5:      # 异常/正常 边界
            return True
        if 17 <= rr < 19:        # 急促/异常 边界
            return True
    if et is not None:
        if 34 <= et <= 36:       # 正常/异常 EtCO2 边界
            return True
    return False


def build_dataset(margin='none'):
    """margin: 'none'(不剔除) / 'train'(仅训练剔除) / 'all'(训练测试都剔除)"""
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    X_list, y_list, cases_list, ref_list = [], [], [], []
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid)
            records = make_windows_with_reference(
                ref['co2'], ref['rr'], ref['etco2'],
                window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)
        except Exception:
            continue
        for r in records:
            rr_med, et_med, _, _ = window_reference_stats(r['rr'], r['etco2'])
            rr_count = (~np.isnan(r['rr']) & (r['rr'] > 0)).sum() if len(r['rr']) else 0
            et_count = (~np.isnan(r['etco2']) & (r['etco2'] > 0)).sum() if len(r['etco2']) else 0
            if rr_med is None and et_med is None:
                continue
            if rr_med is not None and (rr_med < 3 or rr_med >= 18 or rr_med <= 6) and rr_count < 2:
                continue
            if et_med is not None and (et_med < 35 or et_med > 45) and et_count < 2:
                continue
            # 4类标签（与 train.py 一致）
            if rr_med is not None and rr_med < 3:
                lb = 3
            elif rr_med is None and et_med is not None and et_med < 10:
                lb = 3
            elif rr_med is not None and rr_med >= 18:
                lb = 1
            elif rr_med is not None and rr_med <= 6:
                lb = 4  # 低频 → 异常
            elif et_med is not None and (et_med < 35 or et_med > 45):
                lb = 4
            else:
                lb = 0
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(lb)
            cases_list.append(cid)
            ref_list.append((rr_med, et_med))

    X = np.array(X_list); y = np.array(y_list); g = np.array(cases_list)
    margin_mask = np.array([in_margin(rr, et) for rr, et in ref_list])
    return X, y, g, margin_mask


def run(margin='none'):
    X, y, g, margin_mask = build_dataset(margin)
    print(f'  margin={margin}: 总样本 {len(y)}, 边界窗口 {int(margin_mask.sum())} ({margin_mask.mean()*100:.1f}%)')

    final_test_cases, _ = split_final_test(g, test_ratio=0.2)
    train_mask = np.isin(g, final_test_cases, invert=True)
    test_mask = np.isin(g, final_test_cases)

    if margin == 'train':
        # 仅训练剔除边界（测试保留全部）
        eff_train = train_mask & ~margin_mask
        eff_test = test_mask
    elif margin == 'all':
        # 训练测试都剔除
        eff_train = train_mask & ~margin_mask
        eff_test = test_mask & ~margin_mask
    else:
        eff_train, eff_test = train_mask, test_mask

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                 class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0})
    clf.fit(X[eff_train], y[eff_train])

    y_t, y_p = y[eff_test], clf.predict(X[eff_test])
    acc = accuracy_score(y_t, y_p)
    macro = f1_score(y_t, y_p, average='macro', zero_division=0)
    per = {NAMES[c]: round(f1_score(y_t == c, y_p == c, zero_division=0), 3)
           for c in sorted(set(y_t) | set(y_p))}
    print(f'    测试样本 {len(y_t)}, Accuracy={acc:.4f}, Macro-F1={macro:.4f}, 各类F1={per}')

    # 额外：测试集里"边界窗口"的准确率（margin='train'时部署会遇到的）
    if margin == 'train':
        edge_test = test_mask & margin_mask
        if edge_test.sum() > 0:
            edge_acc = accuracy_score(y[edge_test], clf.predict(X[edge_test]))
            print(f'    测试集边界窗口(n={int(edge_test.sum())}) 准确率: {edge_acc:.3f}')
    return macro, per


def main():
    print(f'边界缓冲区实验（{N_CASES} 病例, 90s窗口, 4类）...')
    print('\n=== V0: 无缓冲（当前基线） ===')
    m0, _ = run('none')
    print('\n=== V1: 训练剔除边界，测试保留全部（诚实评估） ===')
    m1, p1 = run('train')
    print('\n=== V2: 训练+测试都剔除（指标虚高参考） ===')
    m2, _ = run('all')

    print('\n' + '=' * 60)
    print('对比总结')
    print('=' * 60)
    print(f'V0 无缓冲:        Macro-F1={m0:.4f}')
    print(f'V1 训练剔除:      Macro-F1={m1:.4f} (测试保留边界)')
    print(f'V2 都剔除:        Macro-F1={m2:.4f} (指标虚高参考)')


if __name__ == '__main__':
    main()
