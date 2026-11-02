# -*- coding: utf-8 -*-
"""
RF 训练模块 —— RandomForest + 病例级 GroupKFold 交叉验证
==========================================================
依据 plan.docx 第 7 节：
- 用 8 维新特征（波形独立计算）→ 预测参考标签
- 病例级 GroupKFold：同一病例窗口不跨集
- 锁定 Final Test 病例（完全未参与调参），剩余做 CV

训练与数据划分策略：
  1. 锁定 20% 病例为 Final Test（模型完全未见）
  2. 剩余 80% 病例做 GroupKFold CV（验证泛化稳定性）
  3. CV 后在全训练集（不含 Final Test）重新训练最终模型
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import json


def get_model(model_name='rf', random_state=42):
    """模型工厂：返回 (model, needs_scaling, fit_kwargs)。

    支持:
        rf        RandomForest（当前基线）
        xgb       XGBoost 加权（少数类样本权重）
        lgb       LightGBM 加权
    """
    if model_name == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=8, random_state=random_state,
                             learning_rate=0.1, eval_metric='mlogloss'), False, 'sample_weight'
    elif model_name == 'lgb':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, max_depth=8, random_state=random_state,
                              learning_rate=0.1, verbose=-1), False, 'sample_weight'
    else:
        # 自定义 class_weight：屏息(2)×3，其余×1（4类标签 0/1/2/3）
        # （balanced 在类别极不均衡时对屏息过度加权，precision 掉到 0.46；
        #   实验 experiment_apnea_weight.py 验证 屏息×3 最优）
        # n_estimators=30：STM32F407 部署瘦身（experiment_rf_size.py）
        #   30树 Macro-F1 0.8865（vs 100树0.888，差0.0015），ONNX 567KB（F407 1MB 安全）
        #   40/50树 ONNX 757/955KB 偏大；30树是精度/大小最优平衡
        return RandomForestClassifier(n_estimators=30, max_depth=8,
                                      random_state=random_state,
                                      class_weight={0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0}), False, None


def _sample_weights(y):
    """计算少数类样本权重（与类别数成反比）。"""
    cnt = Counter(y)
    max_c = max(cnt.values())
    return np.array([max_c / cnt[c] for c in y])


def split_final_test(case_ids, test_ratio=0.2, random_state=42):
    """锁定 Final Test 病例（与 CV 病例不重叠）。

    返回:
        final_test_cases: 最终测试病例（模型完全未见）
        cv_cases:         用于 CV 的病例
    """
    unique_cases = np.unique(case_ids)
    test_cases, cv_cases = train_test_split(
        unique_cases, test_size=1 - test_ratio, random_state=random_state)
    return test_cases, cv_cases


def stratified_group_split(X, y, groups, n_splits=5):
    """按病例 GroupKFold 划分，返回 (train_idx, test_idx) 迭代器。"""
    gkf = GroupKFold(n_splits=n_splits)
    return gkf.split(X, y, groups)


def cross_validate(X, y, groups, n_splits=5, random_state=42, model_name='rf'):
    """GroupKFold 交叉验证，返回每折指标。"""
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # 训练折内模型
        clf, needs_scaling, fit_kwarg = get_model(model_name, random_state)
        if fit_kwarg:
            clf.fit(X[train_idx], y[train_idx],
                    **{fit_kwarg: _sample_weights(y[train_idx])})
        else:
            clf.fit(X[train_idx], y[train_idx])

        y_pred = clf.predict(X[val_idx])
        y_true = y[val_idx]

        acc = float(np.mean(y_pred == y_true))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        fold_results.append({'fold': fold, 'accuracy': acc, 'macro_f1': macro_f1})
        print(f'  Fold {fold+1}: acc={acc:.4f}, macro-F1={macro_f1:.4f} '
              f'(训练{len(train_idx)}样本/验证{len(val_idx)}样本, {len(set(groups[train_idx]))}病例)')

    return fold_results


def train_final_model(X, y, groups, final_test_cases, random_state=42, model_name='rf'):
    """在排除 Final Test 后的全部训练数据上训练最终模型。"""
    train_mask = np.isin(groups, final_test_cases, invert=True)
    clf, needs_scaling, fit_kwarg = get_model(model_name, random_state)
    if fit_kwarg:
        clf.fit(X[train_mask], y[train_mask],
                **{fit_kwarg: _sample_weights(y[train_mask])})
    else:
        clf.fit(X[train_mask], y[train_mask])
    return clf, train_mask


def evaluate_on_final_test(clf, X, y, groups, final_test_cases, class_names=None):
    """在 Final Test 病例上评估最终模型。"""
    test_mask = np.isin(groups, final_test_cases)
    if test_mask.sum() == 0:
        print('  ⚠ Final Test 病例无样本')
        return None

    y_true = y[test_mask]
    y_pred = clf.predict(X[test_mask])

    if class_names is None:
        class_names = [str(i) for i in np.unique(np.concatenate([y_true, y_pred]))]

    acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    print(f'\n=== Final Test 评估（{test_mask.sum()} 样本, {len(final_test_cases)} 病例）===')
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro-F1: {macro_f1:.4f}')
    print('\n分类报告:')
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print('混淆矩阵:')
    print(cm)
    return {'accuracy': acc, 'macro_f1': macro_f1, 'y_true': y_true,
            'y_pred': y_pred, 'confusion': cm}


def evaluate_feature_importance(clf, feature_names):
    """输出特征重要性排序。"""
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    print('\n特征重要性排序:')
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f'F{i+1}'
        print(f'  {name:20s}  {imp[i]:.4f}')
    return imp
