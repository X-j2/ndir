# -*- coding: utf-8 -*-
"""
验证：新增"波形自算 RR"特征能否区分缓慢(2) vs 异常(4)
========================================================
现有 10 维特征里，F5 峰间距均值（秒）间接反映呼吸频率，但：
  - 低碳酸异常窗口峰幅小 → find_peaks 可能漏峰 → F5 偏大 → 误判为缓慢
  - F5 是"间隔秒数"，与标签用的"RR bpm"不是同一尺度，模型不易直接学

验证目标：
  1. 计算每个窗口的"波形自算 RR bpm"（峰值数 → 频率，不用参考轨道）
  2. 对比 真缓慢(2) vs 异常(4)判成缓慢 的自算 RR 分布
  3. 若分布可分（缓慢集中 3-6，异常集中 8-17）→ 新增频率特征有效

同时测试不同 height 阈值下的自算 RR 区分度，确定最佳寻峰参数。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import find_breath_peaks
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from scipy.signal import find_peaks

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 30


def wave_rr_bpm(window, height, sampling_rate=SAMPLING_RATE):
    """波形自算 RR = 有效峰数 × 2（30s 窗口 → 2 bpm/峰）。"""
    peaks, _ = find_peaks(window, height=height,
                          distance=int(0.8 * sampling_rate),
                          prominence=max(6.0, height * 0.5))
    return 2.0 * len(peaks)


def main():
    print(f'验证频率特征（{N_CASES} 病例）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    rows = []  # (label, ref_rr, ref_et, wrr15, wrr10, wrr8)
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
            rows.append((
                res['label'],
                res['rr_median'], res['etco2_median'],
                wave_rr_bpm(co2, 15), wave_rr_bpm(co2, 10), wave_rr_bpm(co2, 8),
            ))
    rows = np.array(rows, dtype=object)
    print(f'  有效窗口: {len(rows)}')

    labels = rows[:, 0].astype(int)
    ref_rr = rows[:, 1].astype(float)
    ref_et = rows[:, 2].astype(float)

    def summary(mask, title, use_et_thresh=None):
        sub = rows[mask]
        if use_et_thresh is not None:
            et = sub[:, 2].astype(float)
            if use_et_thresh[0] == 'lt':
                sub = sub[et < use_et_thresh[1]]
            else:
                sub = sub[et > use_et_thresh[1]]
        if len(sub) == 0:
            print(f'  [{title}] 无样本')
            return
        r = sub[:, 1].astype(float)
        w15 = sub[:, 3].astype(float)
        w10 = sub[:, 4].astype(float)
        w8 = sub[:, 5].astype(float)
        print(f'\n  [{title}] n={len(sub)}')
        print(f'    参考 RR : 中位 {np.median(r):5.1f} | 25-75分位 {np.percentile(r,25):.0f}-{np.percentile(r,75):.0f}')
        print(f'    自算h15 : 中位 {np.median(w15):5.1f} | 25-75分位 {np.percentile(w15,25):.0f}-{np.percentile(w15,75):.0f}')
        print(f'    自算h10 : 中位 {np.median(w10):5.1f} | 25-75分位 {np.percentile(w10,25):.0f}-{np.percentile(w10,75):.0f}')
        print(f'    自算h8  : 中位 {np.median(w8):5.1f} | 25-75分位 {np.percentile(w8,25):.0f}-{np.percentile(w8,75):.0f}')
        # 若用自算 RR≤6 判缓慢，precision/recall 如何（对缓慢类）
        if '缓慢' in title:
            for h, col in [('h15', 3), ('h10', 4), ('h8', 5)]:
                pred_slow = sub[:, col].astype(float) <= 6
                tp = pred_slow.sum()
                total_slow = len(sub)
                print(f'      [自算{h}≤6 判缓慢] recall={tp/total_slow:.2f}')

    names = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}
    print('\n' + '=' * 60)
    print('各类别自算 RR vs 参考 RR')
    print('=' * 60)
    for lb in range(5):
        summary(labels == lb, names[lb])

    # 重点：缓慢 vs 低碳酸异常（两个最接近的类）
    print('\n' + '=' * 60)
    print('缓慢(2) vs 低碳酸异常(4) 频率对比')
    print('=' * 60)
    summary(labels == 2, '缓慢(2) 全部')
    low_abn = (labels == 4) & (ref_et < 35)
    summary(low_abn, '异常(4) 低碳酸(Et<35)')

    # 若用自算RR≤6判缓慢，对"异常"的误报率
    print('\n  [用自算h15≤6 判缓慢] 对异常(4)的误报:')
    for name, mask in [('全部异常', labels == 4),
                       ('低碳酸', (labels == 4) & (ref_et < 35))]:
        sub = rows[mask]
        pred = sub[:, 3].astype(float) <= 6
        print(f'    {name}: {pred.sum()}/{len(sub)} 误判成缓慢 ({pred.mean()*100:.1f}%)')


if __name__ == '__main__':
    main()
