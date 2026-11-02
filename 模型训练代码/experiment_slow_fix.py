# -*- coding: utf-8 -*-
"""
缓慢类混叠修复实验 —— 只改缓慢定义，不动急促/屏息
=====================================================
当前问题：缓慢(2) = RR≤6，EtCO2 从 5-75 横跨全范围，
         混了 低碳酸/正常浓度/高碳酸 三种波形完全不同的状态。

本实验比较 3 种"缓慢"定义（其余类阈值不变）：
  V0 当前:  RR≤6 任意 EtCO2
  V1 限定:  RR≤6 且 35≤EtCO2≤45；低RR+浓度异常 → 异常(4)
  V2 文献:  RR<12 且 35≤EtCO2≤45；低RR+浓度异常 → 异常(4)（扩缓慢到文献范围）

对比：各类分布 + GroupKFold CV Macro-F1 + 每类 P/R
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 100


def label_v0(rr, et):
    """当前: 屏息<3|急促>=18|缓慢<=6(任意EtCO2)|异常EtCO2外|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr >= 18:
        return 1
    if rr is not None and rr <= 6:
        return 2
    if et is not None and (et < 35 or et > 45):
        return 4
    return 0


def label_v1(rr, et):
    """缓慢限定浓度: 屏息<3|急促>=18|缓慢<=6且EtCO2正常|异常EtCO2外|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr >= 18:
        return 1
    if rr is not None and rr <= 6:
        # 缓慢限定 EtCO2 正常范围
        if et is not None and 35 <= et <= 45:
            return 2
        else:
            return 4  # 低RR+浓度异常 → 异常
    if et is not None and (et < 35 or et > 45):
        return 4
    return 0


def label_v2(rr, et):
    """文献阈值+浓度限定: 屏息<3|急促>20|缓慢<12且EtCO2正常|异常EtCO2外|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr > 20:
        return 1
    if rr is not None and rr < 12:
        if et is not None and 35 <= et <= 45:
            return 2
        else:
            return 4
    if et is not None and (et < 35 or et > 45):
        return 4
    return 0


def build_dataset(label_fn):
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
            rr_med, et_med, _, _ = window_reference_stats(r['rr'], r['etco2'])
            rr_count = (~np.isnan(r['rr']) & (r['rr'] > 0)).sum() if len(r['rr']) else 0
            et_count = (~np.isnan(r['etco2']) & (r['etco2'] > 0)).sum() if len(r['etco2']) else 0
            if rr_med is None and et_med is None:
                continue
            if rr_med is not None and (rr_med < 3 or rr_med >= 18 or rr_med <= 6) and rr_count < 2:
                continue
            if et_med is not None and (et_med < 35 or et_med > 45) and et_count < 2:
                continue
            lb = label_fn(rr_med, et_med)
            if lb is None:
                continue
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


VARIANTS = [
    ('V0 当前(RR<=6任意EtCO2)', label_v0),
    ('V1 缓慢限定浓度(RR<=6且EtCO2正常)', label_v1),
    ('V2 文献阈值+限定(RR<12且EtCO2正常)', label_v2),
]
NAMES = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}


def main():
    print(f'缓慢混叠修复实验（{N_CASES} 病例）...')
    results = []
    for name, fn in VARIANTS:
        print(f'\n构建数据集: {name}')
        X, y, g = build_dataset(fn)
        dist = {NAMES[int(k)]: int(v) for k, v in Counter(y).items()}
        print(f'  样本 {len(y)}, 分布 {dist}')
        y_t, y_p = run_cv(X, y, g)
        acc = accuracy_score(y_t, y_p)
        macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        print(f'  Accuracy={acc:.4f}, Macro-F1={macro:.4f}')
        print(classification_report(y_t, y_p, zero_division=0,
                                    target_names=['正常', '急促', '缓慢', '屏息', '异常'],
                                    labels=sorted(set(y_t) | set(y_p))))
        results.append((name, acc, macro, dist))

    print('\n' + '=' * 70)
    print('对比总结（按 Macro-F1 排序）')
    print('=' * 70)
    for name, acc, macro, dist in sorted(results, key=lambda x: -x[2]):
        print(f'\n{name}')
        print(f'  Accuracy={acc:.4f} | Macro-F1={macro:.4f}')
        print(f'  分布: {dist}')


if __name__ == '__main__':
    main()
