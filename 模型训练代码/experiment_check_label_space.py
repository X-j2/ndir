# -*- coding: utf-8 -*-
"""检查标签在 (RR, EtCO2) 参考空间上的划分——是否真有可分性。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window

N_CASES = 30
case_ids = find_cases_with_reference(n_cases=N_CASES)
rows = []  # (label, rr, et)
for cid in case_ids:
    try:
        ref = load_reference_tracks(cid)
        records = make_windows_with_reference(
            ref['co2'], ref['rr'], ref['etco2'],
            window_size=3000, step=1500, sampling_rate=100)
    except Exception:
        continue
    for r in records:
        res = build_labels_for_window(None, r['rr'], r['etco2'], min_reference_points=2)
        if res['label'] is None:
            continue
        rows.append((res['label'], res['rr_median'], res['etco2_median']))

names = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}
print('类别在 (RR, EtCO2) 参考空间的分布：')
for lb in range(5):
    sub = [r for r in rows if r[0] == lb]
    if not sub:
        continue
    rr = np.array([r[1] for r in sub if r[1] is not None], float)
    et = np.array([r[2] for r in sub if r[2] is not None], float)
    line = f'{names[lb]}: n={len(sub)}'
    if len(rr):
        line += f' | RR 中位 {np.median(rr):.1f} (范围 {rr.min():.0f}-{rr.max():.0f})'
    else:
        line += ' | RR 无'
    if len(et):
        line += f' | EtCO2 中位 {np.median(et):.1f} (范围 {et.min():.0f}-{et.max():.0f})'
    else:
        line += ' | EtCO2 无'
    print(line)
