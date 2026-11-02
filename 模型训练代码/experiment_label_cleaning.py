# -*- coding: utf-8 -*-
"""
标签清洗范围分析 —— 哪些类存在"标签与波形不符"的噪声
=====================================================
背景：漏检屏息(21)波形有2次呼吸但标屏息 → 标签噪声。
     但噪声不只屏息有，其他类也可能有。

本脚本：对每个类，检查"标签判类依据"与"波形实际"是否一致
  0 正常(0): 参考 RR 7-17 且 EtCO2 35-45 → 波形应规律有峰
  1 急促(1): RR>=18 → 波形应高频
  3 屏息(3): RR<3 → 波形应无峰/平坦
  4 异常(4): RR<=6 或 EtCO2<35/>45 → 波形低频或浓度异常

对每个窗口，用"波形自算RR"(F11) 与"参考RR"对比：
  - 自算RR 明显偏离参考RR（如参考屏息但波形自算RR>10）→ 标签噪声
输出：各类的"波形-标签不一致"占比，决定清洗范围。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import extract_features
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from collections import Counter

SAMPLING_RATE = 100
WINDOW_SIZE = 9000
WINDOW_STEP = 4500
N_CASES = 200

NAMES = {0: '正常', 1: '急促', 3: '屏息', 4: '异常'}


def wave_rr_bpm(window):
    """波形自算 RR = 峰数×60/90s（90s窗口）"""
    from features.extract_features import find_breath_peaks
    peaks = find_breath_peaks(window, sampling_rate=SAMPLING_RATE)
    return len(peaks) * 60.0 / 90.0  # 90s 窗口


def main():
    print(f'标签清洗范围分析（{N_CASES} 病例, 90s窗口）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)

    # 每类收集 (参考RR, 参考EtCO2, 波形自算RR)
    per_class = {c: [] for c in [0, 1, 3, 4]}
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
            if res['label'] is None or res['label'] not in per_class:
                continue
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue
            wrr = wave_rr_bpm(co2)
            per_class[res['label']].append((res['rr_median'], res['etco2_median'], wrr))

    print('\n' + '=' * 70)
    print('各类的 参考RR vs 波形自算RR 一致性')
    print('=' * 70)
    for c in [0, 1, 3, 4]:
        rows = per_class[c]
        if not rows:
            continue
        rr_ref = np.array([x[0] for x in rows if x[0] is not None], float)
        wrr = np.array([x[2] for x in rows], float)
        print(f'\n【{NAMES[c]}】 n={len(rows)}')
        if len(rr_ref):
            print(f'  参考RR: 中位 {np.median(rr_ref):.1f} | 波形自算RR: 中位 {np.median(wrr):.1f}')
        # 波形自算RR 异常偏高的占比（标签应为低频但波形高频 → 噪声）
        if c in [3, 4]:  # 屏息/异常应低频
            hi = (wrr >= 10).mean()
            print(f'  波形自算RR>=10 的占比: {hi*100:.1f}% (标签低频但波形高频 → 潜在噪声)')
        if c == 0:  # 正常应 7-20
            out = ((wrr < 7) | (wrr > 20)).mean()
            print(f'  波形自算RR 超出7-20 占比: {out*100:.1f}% (潜在噪声)')

    print('\n' + '=' * 70)
    print('结论判断')
    print('=' * 70)
    print('哪几类的"波形-标签不一致"占比高 → 需要清洗')


if __name__ == '__main__':
    main()
