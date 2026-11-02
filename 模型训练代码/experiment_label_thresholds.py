# -*- coding: utf-8 -*-
"""
标签阈值对比实验 —— 文献标准 vs 现有规则
==========================================
对照医学文献阈值（见 README 查证记录）：
  现有规则: 正常 7-17 | 急促>=18 | 缓慢<=6 | 屏息<3 | 异常 EtCO2 35-45 外
  文献标准: 正常 12-20 | 急促>20 | 缓慢<12 | 屏息<3 | 异常 EtCO2 35-45 外

对比：
  1. 各阈值下五类样本分布（是否合理、少数类是否足量）
  2. 相同特征下 GroupKFold CV 的 Macro-F1 / 每类 P/R

目标：用数据决定采用哪个阈值基准（用户选择"先实验再定"）。
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
N_CASES = 150


# ============================================================
# 两种标签规则
# ============================================================
def label_old(rr, et):
    """现有规则：正常 7-17 | 急促>=18 | 缓慢<=6 | 屏息<3 | 异常 EtCO2 外"""
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


def label_lit(rr, et):
    """文献标准：正常 12-20 | 急促>20 | 缓慢<12 | 屏息<3 | 异常 EtCO2 外"""
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


def build_dataset(label_fn):
    """构建 (X, y, cases)。用给定的标签函数。"""
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
            # 参考点数过滤（与 build_labels_for_window 一致）
            rr_count = (~np.isnan(r['rr']) & (r['rr'] > 0)).sum() if len(r['rr']) else 0
            et_count = (~np.isnan(r['etco2']) & (r['etco2'] > 0)).sum() if len(r['etco2']) else 0
            if rr_med is None and et_med is None:
                continue
            need_rr = (rr_med is not None) and (rr_med < 3 or rr_med >= 18 or rr_med <= 6) if label_fn is label_old else \
                      (rr_med is not None) and (rr_med < 3 or rr_med > 20 or rr_med < 12)
            need_et = (et_med is not None) and (et_med < 10 or et_med < 35 or et_med > 45)
            if need_rr and rr_count < 2:
                continue
            if need_et and et_count < 2:
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
    """GroupKFold CV，返回每类 P/R/F1。"""
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


def report(name, X, y, groups):
    print(f'\n{"="*70}')
    print(f'【{name}】')
    print(f'{"="*70}')
    print(f'样本 {len(y)}, 特征 {X.shape[1]}, 病例 {len(set(groups))}')
    dist = {int(k): int(v) for k, v in Counter(y).items()}
    print(f'类别分布: {dist}')
    y_t, y_p = run_cv(X, y, groups)
    acc = accuracy_score(y_t, y_p)
    macro = f1_score(y_t, y_p, average='macro', zero_division=0)
    print(f'\nAccuracy={acc:.4f}, Macro-F1={macro:.4f}')
    print('分类报告:')
    print(classification_report(y_t, y_p, zero_division=0,
                                target_names=['正常','急促','缓慢','屏息','异常']))
    return {'name': name, 'dist': dist, 'acc': acc, 'macro': macro}


def main():
    print(f'标签阈值对比实验（{N_CASES} 病例）...')

    # 现有规则
    print('\n构建数据集（现有规则）...')
    X1, y1, g1 = build_dataset(label_old)
    r1 = report('现有规则 (正常7-17|急促>=18|缓慢<=6)', X1, y1, g1)

    # 文献标准
    print('\n构建数据集（文献标准）...')
    X2, y2, g2 = build_dataset(label_lit)
    r2 = report('文献标准 (正常12-20|急促>20|缓慢<12)', X2, y2, g2)

    print('\n' + '='*70)
    print('对比总结')
    print('='*70)
    for r in (r1, r2):
        print(f'{r["name"]}:')
        print(f'  分布 {r["dist"]}')
        print(f'  Accuracy={r["acc"]:.4f} | Macro-F1={r["macro"]:.4f}')


if __name__ == '__main__':
    main()
