# -*- coding: utf-8 -*-
"""
二阶段分类实验（借鉴 Spijkerboer 2024, J Clin Monit Comput）
=============================================================
文献核心（见 experiment 研究记录）：
  Stage1: 二分类"正常 vs 异常"，高 recall 抓所有潜在危险
  Stage2: 对异常子集细分（急促/缓慢/屏息/异常），优化 precision
  核心特征: "最长低于均值段"(抓屏息) + 一阶傅里叶系数(频率)

本实验：
  1. 新增 2 个特征: F13 最长低于均值段, F14 一阶傅里叶系数
  2. Stage1: RF 二分类 正常 vs 异常（报告 recall）
  3. Stage2: 异常子集 四分类（报告每类 P/R/F1）
  4. 与单阶段对比（同特征、同数据划分）
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features, find_breath_peaks
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 100


# ============================================================
# 新增特征（文献启发）
# ============================================================
def feat_longest_strike_below_mean(window):
    """F13: 最长连续低于均值段（采样点数）。文献核心特征（TSFresh）。
    呼吸停止(屏息) → CO₂ 长时间低于均值 → 该特征大。
    """
    x = np.asarray(window, dtype=float)
    mean = np.mean(x)
    below = x < mean
    # 找最长连续 True 段
    max_len = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return float(max_len)


def feat_fourier_freq(window):
    """F14: 一阶实傅里叶系数（实部）。文献核心特征。
    反映主要振荡频率，幅度无关，可用于区分高频/低频呼吸。
    """
    x = np.asarray(window, dtype=float) - np.mean(window)
    n = len(x)
    # 一阶傅里叶系数（频率 = 1/窗口周期）
    t = np.arange(n)
    # 找主导频率（0.05-1.2 Hz 内峰值）
    from scipy.fft import rfft, rfftfreq
    X = rfft(x * np.hanning(n))
    freqs = rfftfreq(n, d=1.0 / SAMPLING_RATE)
    mask = (freqs >= 0.05) & (freqs <= 1.2)
    if not mask.any():
        return 0.0
    idx = np.argmax(np.abs(X[mask]))
    return float(freqs[mask][idx])


def extract_features_14(window, sampling_rate=SAMPLING_RATE):
    """12 维 + F13 + F14 = 14 维。"""
    f12 = extract_features(window, sampling_rate)
    f13 = feat_longest_strike_below_mean(window)
    f14 = feat_fourier_freq(window)
    return np.concatenate([f12, [f13, f14]])


def build_dataset():
    """构建 14 维数据集 + 5 类标签。"""
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
            X_list.append(extract_features_14(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(res['label'])
            cases_list.append(cid)
    return np.array(X_list), np.array(y_list), np.array(cases_list)


def two_stage_cv(X, y, groups):
    """二阶段 GroupKFold CV。

    Stage1: 正常(0) vs 异常(1-4)
    Stage2: 异常子集的 4 分类（1/2/3/4）
    最终预测 = Stage2 结果（对 Stage1 判正常的用 Stage2 训练不覆盖的类？）
    这里采用"串联": Stage1 判正常→正常；判异常→Stage2 细分。
    """
    gkf = GroupKFold(n_splits=5)

    # Stage1 指标
    s1_recall_all = []
    # Stage2 指标
    s2_f1s = []
    # 整体串联指标（对 5 类）
    y_true_all, y_pred_all = [], []

    for train_idx, val_idx in gkf.split(X, y, groups):
        ytr, yva = y[train_idx], y[val_idx]

        # ---- Stage1: 正常 vs 异常 ----
        y1_tr = (ytr != 0).astype(int)   # 1=异常
        y1_va = (yva != 0).astype(int)
        clf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf1.fit(X[train_idx], y1_tr)
        p1 = clf1.predict(X[val_idx])
        # Stage1 recall: 异常被抓到多少
        mask_abn = y1_va == 1
        recall_abn = (p1[mask_abn] == 1).mean() if mask_abn.any() else 0
        s1_recall_all.append(recall_abn)

        # ---- Stage2: 异常子集 4 分类 ----
        abn_tr = train_idx[ytr != 0]
        if len(abn_tr) < 50:
            # 异常训练样本太少，跳过本折
            continue
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=8,
                                      random_state=42, class_weight='balanced')
        clf2.fit(X[abn_tr], ytr[ytr != 0])
        # Stage2 只预测 Stage1 判为异常且真实也是异常的
        abn_val = val_idx[(y1_va == 1)]
        if len(abn_val) == 0:
            continue
        p2 = clf2.predict(X[abn_val])
        f2 = f1_score(yva[y1_va == 1], p2, average='macro', zero_division=0)
        s2_f1s.append(f2)

        # ---- 串联整体预测 ----
        y_pred_fold = np.zeros(len(val_idx), dtype=int)
        # Stage1 判正常的 → 预测 0
        norm_idx = val_idx[p1 == 0]
        y_pred_fold[p1 == 0] = 0
        # Stage1 判异常 → Stage2 细分
        abn_pred_idx = val_idx[p1 == 1]
        if len(abn_pred_idx):
            p2_all = clf2.predict(X[abn_pred_idx])
            for pos, ai in enumerate(abn_pred_idx):
                y_pred_fold[p1 == 1][pos] = p2_all[pos]
        y_true_all.extend(yva.tolist())
        y_pred_all.extend(y_pred_fold.tolist())

    return y_true_all, y_pred_all, s1_recall_all, s2_f1s


def single_stage_cv(X, y, groups):
    """单阶段 5 分类 CV（基线对比）。"""
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


def main():
    print(f'二阶段分类实验（{N_CASES} 病例, 14 维特征）...')
    X, y, g = build_dataset()
    print(f'  样本 {len(y)}, 特征 {X.shape[1]}, 病例 {len(set(g))}')
    dist = {int(k): int(v) for k, v in Counter(y).items()}
    print(f'  类别分布: {dist}')

    # 单阶段基线
    print('\n=== 单阶段 5 分类（基线） ===')
    y_t, y_p = single_stage_cv(X, y, g)
    print(f'  Accuracy={accuracy_score(y_t, y_p):.4f}')
    print(f'  Macro-F1={f1_score(y_t, y_p, average="macro", zero_division=0):.4f}')
    print(classification_report(y_t, y_p, zero_division=0,
                                target_names=['正常','急促','缓慢','屏息','异常']))

    # 二阶段（分开评估，文献做法）
    print('\n=== 二阶段（分开评估，Spijkerboer 做法） ===')
    yt, yp, s1_rec, s2_f1 = two_stage_cv(X, y, g)
    yt, yp = np.array(yt), np.array(yp)
    print(f'  Stage1 异常 recall: {np.mean(s1_rec):.4f} ± {np.std(s1_rec):.4f}')
    print(f'  Stage2 异常细分 Macro-F1: {np.mean(s2_f1):.4f}')

    # Stage2 单独评估（只在真实异常子集上，不串联）
    gkf = GroupKFold(n_splits=5)
    abn_true, abn_pred = [], []
    for train_idx, val_idx in gkf.split(X, y, g):
        abn_tr = train_idx[y[train_idx] != 0]
        if len(abn_tr) < 50:
            continue
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=8,
                                      random_state=42, class_weight='balanced')
        clf2.fit(X[abn_tr], y[abn_tr])
        abn_va = val_idx[y[val_idx] != 0]
        if len(abn_va) == 0:
            continue
        abn_true.extend(y[abn_va].tolist())
        abn_pred.extend(clf2.predict(X[abn_va]).tolist())
    if len(abn_true):
        print('\n  Stage2 异常子集 4 分类评估：')
        labels = sorted(set(abn_true) | set(abn_pred))
        tnames = ['(1)急促','(2)缓慢','(3)屏息','(4)异常']
        print(classification_report(abn_true, abn_pred, labels=labels,
                                    zero_division=0, target_names=tnames))
        print(f'  Stage2 异常细分 Macro-F1: '
              f'{f1_score(abn_true, abn_pred, average="macro", zero_division=0):.4f}')


if __name__ == '__main__':
    main()
