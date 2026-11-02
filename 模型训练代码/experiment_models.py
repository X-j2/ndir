# -*- coding: utf-8 -*-
"""
模型对比实验 —— RF vs XGBoost vs LightGBM vs SVM
==================================================
在同一套数据（100病例、半重叠窗口、10特征、混合标签）上对比不同模型。

目标：找到比 RandomForest（当前 Macro-F1 0.71）更好的模型。
所有模型都需能导出 ONNX 部署到 STM32。
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 100


def build_dataset(n_cases=N_CASES):
    """构建数据集（与 train.py 一致的流程）。"""
    case_ids = find_cases_with_reference(n_cases=n_cases)
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
            X_list.append(extract_features(co2, sampling_rate=SAMPLING_RATE))
            y_list.append(res['label'])
            cases_list.append(cid)
    return np.array(X_list), np.array(y_list), np.array(cases_list)


def run_groupkfold(X, y, groups, model_fn, n_splits=5):
    """GroupKFold 交叉验证。model_fn: 返回 (model, needs_scaling)。"""
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model, needs_scaling = model_fn()
        if needs_scaling:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[train_idx])
            Xva = scaler.transform(X[val_idx])
        else:
            Xtr, Xva = X[train_idx], X[val_idx]
        model.fit(Xtr, y[train_idx])
        yp = model.predict(Xva)
        accs.append(accuracy_score(y[val_idx], yp))
        f1s.append(f1_score(y[val_idx], yp, average='macro', zero_division=0))
    return np.mean(accs), np.mean(f1s)


def main():
    print('=' * 60)
    print('模型对比实验（100病例, 10特征, 半重叠窗口）')
    print('=' * 60)

    print('\n构建数据集...')
    X, y, cases = build_dataset()
    print(f'  样本: {len(y)}, 特征: {X.shape[1]}, 病例: {len(set(cases))}')
    dist = {i: int((y == i).sum()) for i in np.unique(y)}
    print(f'  类别分布: {dist}')

    # RF 基线
    print('\n=== RandomForest (当前基线) ===')
    acc, f1 = run_groupkfold(X, y, cases, lambda: (
        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced'), False))
    print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')

    # XGBoost
    print('\n=== XGBoost ===')
    try:
        from xgboost import XGBClassifier
        acc, f1 = run_groupkfold(X, y, cases, lambda: (
            XGBClassifier(n_estimators=100, max_depth=8, random_state=42,
                          learning_rate=0.1, eval_metric='mlogloss'), False))
        print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
    except Exception as e:
        print(f'  ⚠ {e}')

    # LightGBM
    print('\n=== LightGBM ===')
    try:
        from lightgbm import LGBMClassifier
        acc, f1 = run_groupkfold(X, y, cases, lambda: (
            LGBMClassifier(n_estimators=100, max_depth=8, random_state=42,
                           learning_rate=0.1, class_weight='balanced', verbose=-1), False))
        print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
    except Exception as e:
        print(f'  ⚠ {e}')

    # SVM (RBF, 需标准化)
    print('\n=== SVM (RBF, 标准化) ===')
    try:
        acc, f1 = run_groupkfold(X, y, cases, lambda: (
            SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'), True))
        print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
    except Exception as e:
        print(f'  ⚠ {e}')

    # SVM (linear)
    print('\n=== SVM (linear, 标准化) ===')
    try:
        acc, f1 = run_groupkfold(X, y, cases, lambda: (
            SVC(kernel='linear', C=1.0, class_weight='balanced'), True))
        print(f'  Accuracy={acc:.4f}, Macro-F1={f1:.4f}')
    except Exception as e:
        print(f'  ⚠ {e}')

    print('\n' + '=' * 60)
    print('实验完成')


if __name__ == '__main__':
    main()
