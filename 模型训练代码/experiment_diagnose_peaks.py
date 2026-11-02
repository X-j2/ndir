# -*- coding: utf-8 -*-
"""
寻峰漏峰诊断 —— 异常(4)类窗口的波形频率 vs 监护仪参考 RR
==========================================================
背景：缓慢(2) precision 只有 0.13，大量异常(4)窗口被误判为缓慢。
假设：固定寻峰阈值 height=15/prominence=8 对"低碳酸血症"波形（幅度小）
      漏掉大量真实呼吸峰 → 峰数变少、峰间距变大 → 特征与缓慢(2)混淆。

本脚本量化：
  1. 每个窗口：参考 RR（监护仪） vs 波形估算频率（不同 height 阈值）
  2. 按类别汇总漏峰程度（重点看异常类低碳酸/高碳酸细分）
  3. 评估自相关/FFT 周期估计（幅度不敏感）在低碳酸窗口的效果

输出：对比表，供决定"降阈值 or 加周期特征 or 两者都做"。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.build_cases import find_cases_with_reference
from preprocessing.make_windows import make_windows_with_reference
from preprocessing.signal_processing import preprocess_window
from features.extract_features import find_breath_peaks
from labels.build_reference_labels import load_reference_tracks, build_labels_for_window
from scipy.signal import find_peaks, welch

SAMPLING_RATE = 100
WINDOW_SIZE = 3000
WINDOW_STEP = 1500
N_CASES = 20


def peak_freq_bpm(window, height, sampling_rate=SAMPLING_RATE):
    """用给定 height 找峰，估算呼吸频率 (bpm) = 峰数/30s * 60。"""
    peaks, _ = find_peaks(window, height=height,
                          distance=int(0.8 * sampling_rate),
                          prominence=max(6.0, height * 0.5))
    n = len(peaks)
    return 2.0 * n  # 30s 窗口 → 2*bpm 每峰


def autocorr_period_bpm(window, sampling_rate=SAMPLING_RATE):
    """自相关主周期 → 呼吸频率 (bpm)。幅度不敏感。

    对去均值波形算自相关，在滞后 0.5~5s（12~120bpm 对应周期）
    范围内找第一个显著峰，对应周期即为呼吸周期。
    找不到 → 返回 None（可能屏息/噪声）。
    """
    x = window - np.mean(window)
    n = len(x)
    if np.std(x) < 1e-6:
        return None

    k_min = int(0.5 * sampling_rate)
    k_max = int(5.0 * sampling_rate)
    k_max = min(k_max, n // 2)

    # 用 FFT 加速自相关
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    S = np.abs(X) ** 2
    acorr = np.fft.irfft(S, n=nfft)[:k_max + 1]
    # 归一化
    if acorr[0] > 1e-9:
        acorr /= acorr[0]

    # 找 k_min..k_max 范围内第一个显著局部峰（前导上升沿）
    seg = acorr[k_min:k_max + 1]
    # 平滑
    seg = np.convolve(seg, np.ones(3) / 3, mode='same')
    if len(seg) < 3:
        return None

    peaks, props = find_peaks(seg, prominence=0.05)
    if len(peaks) == 0:
        return None

    first_k = k_min + peaks[0]
    period_s = first_k / sampling_rate
    if period_s < 0.4:
        return None
    return 60.0 / period_s


def fft_freq_bpm(window, sampling_rate=SAMPLING_RATE):
    """FFT 主频 → 呼吸频率 (bpm)。幅度不敏感（只看频率位置）。

    在 0.08~1.2 Hz（约 5~72 bpm）范围内找能量主峰。
    """
    x = window - np.mean(window)
    n = len(x)
    # 汉宁窗降低频谱泄漏
    win = np.hanning(n)
    xw = x * win
    f, Pxx = welch(xw, fs=sampling_rate, nperseg=min(n, 1024))
    mask = (f >= 0.08) & (f <= 1.2)
    if not mask.any():
        return None
    f_sel = f[mask]
    P_sel = Pxx[mask]
    # 找能量主峰（要求相对突出）
    i_main = np.argmax(P_sel)
    P_peak = P_sel[i_main]
    P_mean = np.mean(P_sel)
    if P_peak < 2.0 * P_mean:
        return None  # 无明显周期（屏息/噪声占主导）
    return f_sel[i_main] * 60.0


def main():
    print(f'诊断寻峰漏峰（{N_CASES} 病例）...')
    case_ids = find_cases_with_reference(n_cases=N_CASES)
    print(f'  合格病例: {len(case_ids)}')

    rows = []  # (label, etco2_median, ref_rr, f15, f10, f8, acorr, fft)
    for cid in case_ids:
        try:
            ref = load_reference_tracks(cid)
            records = make_windows_with_reference(
                ref['co2'], ref['rr'], ref['etco2'],
                window_size=WINDOW_SIZE, step=WINDOW_STEP, sampling_rate=SAMPLING_RATE)
        except Exception as e:
            print(f'  病例 {cid} 加载失败: {e}')
            continue

        for r in records:
            res = build_labels_for_window(None, r['rr'], r['etco2'], min_reference_points=2)
            if res['label'] is None:
                continue
            co2, _ = preprocess_window(r['co2'], sampling_rate=SAMPLING_RATE)
            if co2 is None:
                continue

            ref_rr = res['rr_median']
            et = res['etco2_median']
            f15 = peak_freq_bpm(co2, 15)
            f10 = peak_freq_bpm(co2, 10)
            f8 = peak_freq_bpm(co2, 8)
            ac = autocorr_period_bpm(co2)
            ff = fft_freq_bpm(co2)
            rows.append((res['label'], et, ref_rr, f15, f10, f8, ac, ff))

    print(f'  有效窗口: {len(rows)}')
    rows = np.array(rows, dtype=object)

    # ============================================================
    # 汇总：按类别（异常类再细分为低碳酸/高碳酸）
    # ============================================================
    names = {0: '正常', 1: '急促', 2: '缓慢', 3: '屏息', 4: '异常'}

    def summarize(rows_sub, title):
        if len(rows_sub) == 0:
            print(f'\n  [{title}] 无样本')
            return
        rr = rows_sub[:, 2].astype(float)
        f15 = rows_sub[:, 3].astype(float)
        f10 = rows_sub[:, 4].astype(float)
        f8 = rows_sub[:, 5].astype(float)
        ac = rows_sub[:, 6]
        ff = rows_sub[:, 7]

        # 过滤无效
        def med(arr):
            arr = arr[~np.isnan(arr)]
            return float(np.median(arr)) if len(arr) else float('nan')

        ac_valid = np.array([x for x in ac if x is not None], dtype=float)
        ff_valid = np.array([x for x in ff if x is not None], dtype=float)

        print(f'\n  [{title}] n={len(rows_sub)}')
        print(f'    参考 RR 中位    : {med(rr):6.1f} bpm')
        print(f'    波形频率 h=15   : {med(f15):6.1f} bpm   (漏峰差 {med(rr)-med(f15):+5.1f})')
        print(f'    波形频率 h=10   : {med(f10):6.1f} bpm   (漏峰差 {med(rr)-med(f10):+5.1f})')
        print(f'    波形频率 h=8    : {med(f8):6.1f} bpm   (漏峰差 {med(rr)-med(f8):+5.1f})')
        print(f'    自相关周期估计  : {med(ac_valid):6.1f} bpm (有效 {len(ac_valid)}/{len(rows_sub)})')
        print(f'    FFT 主频估计    : {med(ff_valid):6.1f} bpm (有效 {len(ff_valid)}/{len(rows_sub)})')
        # 用"频率误差"评估：|估计-参考|
        err15 = np.nanmedian(np.abs(f15 - rr))
        err_ac = np.nanmedian(np.abs(ac_valid - rr[np.array([x is not None for x in ac], dtype=bool)])) if len(ac_valid) else float('nan')
        err_ff = np.nanmedian(np.abs(ff_valid - rr[np.array([x is not None for x in ff], dtype=bool)])) if len(ff_valid) else float('nan')
        print(f'    频率误差 h=15   : {err15:5.1f} bpm | 自相关 {err_ac:5.1f} bpm | FFT {err_ff:5.1f} bpm')

    labels = rows[:, 0].astype(int)
    for lb in sorted(set(labels)):
        mask = labels == lb
        summarize(rows[mask], names[lb])

    # 异常类细分：低碳酸 (et<35) / 高碳酸 (et>45)
    mask_abn = labels == 4
    et = rows[:, 1].astype(float)
    if mask_abn.any():
        low = mask_abn & (et < 35)
        high = mask_abn & (et > 45)
        summarize(rows[low], '异常-低碳酸(Et<35)')
        summarize(rows[high], '异常-高碳酸(Et>45)')

    print('\n' + '=' * 60)
    print('诊断完成')


if __name__ == '__main__':
    main()
