# -*- coding: utf-8 -*-
"""
标签质量分析 —— 五类的 RR_median / EtCO2_median 分布统计
==========================================================
目的：判断标签本身是否有问题。

方法与 train.py 完全一致：
  - 150 病例，30s 半重叠窗口（步长 15s）
  - build_labels_for_window 生成标签（含 rr_median / etco2_median）
  - split_final_test 锁定 20% 病例（random_state=42）为 Final Test

对 Final Test 的五类分别统计 RR_median 和 EtCO2_median 的：
  范围(min-max)、均值、中位数、四分位数(Q1/Q3)、有效值数

重点：
  - 缓慢(2) 的 170 个真实样本：RR 应集中在 3-6，EtCO2 分布如何？
  - 缓慢 vs 异常 的参考空间是否真的可分（标签是否有病理混叠）
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from training.train_rf import split_final_test

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 150
FINAL_TEST_RATIO = 0.2


def build_dataset():
    """与 train.py 一致：返回窗口的 (label, rr_median, etco2_median, case)。"""
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    rows = []
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
            rows.append((res['label'], res['rr_median'], res['etco2_median'], cid))
    return rows


def stats(vals, title):
    """对数值数组输出分布统计。vals 是过滤掉 None 后的数组。"""
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        print(f'  {title}: 无有效值')
        return v
    q1, q3 = np.percentile(v, [25, 75])
    print(f'  {title}: n={len(v)} | 范围 {v.min():.1f}-{v.max():.1f} | '
          f'均值 {v.mean():.2f} | 中位 {np.median(v):.1f} | '
          f'Q1 {q1:.1f} | Q3 {q3:.1f}')
    return v


def main():
    print('构建数据集（150 病例，与训练一致）...')
    rows = build_dataset()
    labels = np.array([r[0] for r in rows])
    cases = np.array([r[3] for r in rows])
    print(f'  总样本: {len(rows)}')

    # 锁定 Final Test（与训练完全一致）
    final_test_cases, _ = split_final_test(cases, test_ratio=FINAL_TEST_RATIO, random_state=42)
    test_mask = np.isin(cases, final_test_cases)
    ft = [rows[i] for i in range(len(rows)) if test_mask[i]]
    ft_labels = np.array([r[0] for r in ft])
    print(f'  Final Test 样本: {len(ft)}（病例 {len(final_test_cases)}）')

    names = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}
    print('\n' + '=' * 72)
    print('Final Test 五类的 RR_median / EtCO2_median 分布')
    print('=' * 72)

    for lb in range(5):
        sub = [r for r in ft if r[0] == lb]
        if not sub:
            continue
        print(f'\n【{names[lb]}({lb})】 n={len(sub)}')
        rr = [r[1] for r in sub]
        et = [r[2] for r in sub]
        stats(rr, 'RR_median')
        stats(et, 'EtCO2_median')
        # 缓慢类额外输出：RR 和 EtCO2 的联合分布
        if lb == 2:
            print('\n  缓慢(2) 的 RR 分档：')
            rr_v = np.array([x for x in rr if x is not None], float)
            if len(rr_v):
                for lo, hi, name in [(3, 4, 'RR 3-4'), (4, 5, 'RR 4-5'), (5, 6, 'RR 5-6'), (6, 7, 'RR 6-7')]:
                    c = int(((rr_v >= lo) & (rr_v < hi)).sum())
                    print(f'    {name}: {c} 样本 ({c/len(rr_v)*100:.0f}%)')
                print(f'    RR 其他 (<3 或 >6): {int(((rr_v < 3) | (rr_v >= 6)).sum())} 样本')
            et_v = np.array([x for x in et if x is not None], float)
            if len(et_v):
                print('  缓慢(2) 的 EtCO2 分档：')
                for lo, hi, name in [(0, 10, 'EtCO2 <10'), (10, 20, '10-20'), (20, 30, '20-30'),
                                     (30, 35, '30-35'), (35, 45, '35-45'), (45, 100, '>45')]:
                    c = int(((et_v >= lo) & (et_v < hi)).sum())
                    print(f'    {name}: {c} 样本 ({c/len(et_v)*100:.0f}%)')

    # 缓慢 vs 异常 参考空间对比（关键：标签是否可分）
    print('\n' + '=' * 72)
    print('缓慢(2) vs 异常(4) 参考空间对比（判断标签可分性）')
    print('=' * 72)
    slow = [r for r in ft if r[0] == 2]
    abn = [r for r in ft if r[0] == 4]
    stats([r[1] for r in slow], '缓慢 RR_median')
    stats([r[2] for r in slow], '缓慢 EtCO2_median')
    stats([r[1] for r in abn], '异常 RR_median')
    stats([r[2] for r in abn], '异常 EtCO2_median')

    print('\n诊断完成')


if __name__ == '__main__':
    main()
