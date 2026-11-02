# -*- coding: utf-8 -*-
"""
临床语义标签变体对比实验
=========================
目标：保留临床诊断术语（正常/急促/缓慢/屏息/异常），
但修复"缓慢"类的病理混叠（当前缓慢 EtCO2 从 5-75，混了3种状态）。

假设：缓慢类混叠是 precision 低的直接原因。
修复思路：让每个临床标签对应清晰的波形形态（浓度×频率）。

候选变体（都用文献阈值 正常12-20/急促>20/缓慢<12 为基准）：
  V1 当前:     屏息<3 | 急促>=18 | 缓慢<=6 | 异常EtCO2外 | 正常
  V2 文献+纯化: 屏息<3 | 急促>20 | 缓慢<12且EtCO2正常 | 异常EtCO2外 | 正常
       (缓慢限定 EtCO2 35-45，消除混叠)
  V3 五类去混: 屏息<3 | 急促>20 | 缓慢<12(任意EtCO2) | 异常EtCO2外 | 正常
       (文献阈值但缓慢不限定浓度，看混叠影响)
  V4 六类(备选): 屏息 | 急促 | 缓慢-低碳酸 | 缓慢-正常 | 异常 | 正常
       (缓慢按浓度拆分，最纯但样本更少)

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
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 100  # 用100病例加快实验（缓存已预热）


# ============================================================
# 标签变体
# ============================================================
def label_v1(rr, et):
    """当前: 屏息<3|急促>=18|缓慢<=6|异常EtCO2外|正常"""
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


def label_v2(rr, et):
    """文献+纯化: 屏息<3|急促>20|缓慢<12且EtCO2正常|异常EtCO2外|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr > 20:
        return 1
    if rr is not None and rr < 12:
        # 缓慢限定 EtCO2 正常范围（消除低碳酸/高碳酸混入）
        if et is not None and 35 <= et <= 45:
            return 2
        # 低RR+浓度异常 → 归异常(4)（呼吸频率低但浓度异常是不同状态）
        else:
            return 4
    if et is not None and (et < 35 or et > 45):
        return 4
    return 0


def label_v3(rr, et):
    """文献阈值: 屏息<3|急促>20|缓慢<12(任意EtCO2)|异常EtCO2外|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr > 20:
        return 1
    if rr is not None and rr < 12:
        return 2
    if et is not None and (et < 35 or et > 45):
        return 4
    return 0


def label_v4(rr, et):
    """六类: 屏息|急促|缓慢-低碳酸|缓慢-正常|异常|正常"""
    if rr is None and et is None:
        return None
    if rr is not None and rr < 3:
        return 3
    if rr is None and et is not None and et < 10:
        return 3
    if rr is not None and rr > 20:
        return 1
    if rr is not None and rr < 12:
        # 缓慢按浓度拆
        if et is not None and et < 35:
            return 5  # 缓慢-低碳酸
        elif et is not None and et > 45:
            return 4  # 异常(高碳酸)
        else:
            return 2  # 缓慢-正常
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
            # 简化：参考点数过滤（与训练一致）
            if rr_med is not None and (rr_med < 3 or rr_med > 20 or rr_med < 12) and rr_count < 2:
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
    ('V1 当前(7-17|>=18|<=6)', label_v1),
    ('V2 文献+缓慢纯化(12-20|>20|<12且EtCO2正常)', label_v2),
    ('V3 文献阈值(12-20|>20|<12任意)', label_v3),
    ('V4 六类(缓慢拆低碳酸/正常)', label_v4),
]
NAMES = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常', 5: '缓低碳酸'}


def main():
    print(f'临床语义标签变体对比（{N_CASES} 病例）...')
    results = []
    for name, fn in VARIANTS:
        print(f'\n构建数据集: {name}')
        X, y, g = build_dataset(fn)
        dist = {NAMES.get(int(k), str(k)): int(v) for k, v in Counter(y).items()}
        print(f'  样本 {len(y)}, 分布 {dist}')
        y_t, y_p = run_cv(X, y, g)
        acc = accuracy_score(y_t, y_p)
        macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        # 每类 F1
        labels = sorted(set(y_t) | set(y_p))
        per_class = {}
        for lb in labels:
            per_class[NAMES.get(int(lb), str(lb))] = round(
                f1_score(y_t == lb, y_p == lb, zero_division=0), 3)
        print(f'  Accuracy={acc:.4f}, Macro-F1={macro:.4f}')
        print(f'  每类F1: {per_class}')
        results.append((name, acc, macro, dist, per_class))

    print('\n' + '=' * 70)
    print('对比总结（按 Macro-F1 排序）')
    print('=' * 70)
    for name, acc, macro, dist, pc in sorted(results, key=lambda x: -x[2]):
        print(f'\n{name}')
        print(f'  Accuracy={acc:.4f} | Macro-F1={macro:.4f}')
        print(f'  分布: {dist}')
        print(f'  每类F1: {pc}')


if __name__ == '__main__':
    main()
