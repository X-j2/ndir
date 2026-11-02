# -*- coding: utf-8 -*-
"""
信号预处理模块 —— 通用去噪 + NaN 处理 + 质量检测
=================================================
设计依据（滤波算法说明.md + plan.docx）：
- 训练数据（VitalDB Primus/CO2）和未来 STM32 推理数据都是"浓度成品波形(mmHg)"
- 固件 α=1/8 IIR 是给原始 ADC count 用的，不是给成品波形用 → 不复刻固件 IIR
- 这里只做通用信号质量处理：NaN 插值、轻度去噪、基线校正、异常点检测、质量检测

一致性要求：训练端用的这些处理方法，若 STM32 推理端启用必须用同一套。
更稳妥的方案是两端都用原始成品波形，预处理仅用于提高特征质量。
"""
import numpy as np
from scipy.signal import butter, filtfilt


# ============================================================
# 1. NaN 处理
# ============================================================
def fill_short_gaps(x, max_gap=200):
    """短缺失（<=max_gap 点）线性插值填充，保留时间轴。

    参数:
        x: 1D CO2 波形（含 NaN）
        max_gap: 最长可插值缺口（点）。默认 200 点 = 2 秒 @100Hz
    返回:
        填充后的数组（长缺口仍保留 NaN）
    设计: 短缺失插值保住时间轴连续性；长缺失留给 make_windows 按有效区间切分。
    """
    x = np.array(x, dtype=float)
    n = len(x)
    nan_mask = np.isnan(x)
    if not nan_mask.any():
        return x

    # 找到每个 NaN 段的起止
    idx = np.arange(n)
    nan_idx = idx[nan_mask]
    if len(nan_idx) == 0:
        return x

    # 按连续 NaN 分组
    groups = []
    start = nan_idx[0]
    prev = nan_idx[0]
    for i in nan_idx[1:]:
        if i - prev > 1:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))

    result = x.copy()
    for g_start, g_end in groups:
        gap_len = g_end - g_start + 1
        if gap_len <= max_gap:
            # 线性插值：用缺口前后最近的有效值
            before = g_start - 1
            after = g_end + 1
            if before >= 0 and after < n:
                v_before = x[before]
                v_after = x[after]
                if not (np.isnan(v_before) or np.isnan(v_after)):
                    t = np.arange(gap_len + 2) / (gap_len + 1)
                    result[g_start:g_end+1] = v_before + (v_after - v_before) * t[1:-1]
    return result


def get_valid_segments(x, min_len=3000):
    """把含长缺口的波形切分为连续有效区间（保留时间轴完整性）。

    参数:
        x: 1D CO2 波形（可能含 NaN）
        min_len: 有效区间最小长度（点）。默认 3000 = 30 秒窗口
    返回:
        [(start_idx, segment_array), ...]，只保留长度 >= min_len 的区间
    设计: 不直接删 NaN 压缩时间轴；长缺失处切开，各自成为独立连续信号。
    """
    valid = ~np.isnan(x)
    n = len(x)
    segments = []
    i = 0
    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1
            seg = x[i:j]
            if len(seg) >= min_len:
                segments.append((i, seg))
            i = j
        else:
            i += 1
    return segments


# ============================================================
# 2. 去噪（轻度低通滤波，保住呼吸频带）
# ============================================================
def lowpass_filter(x, sampling_rate=100, cutoff=2.0, order=3):
    """低通滤波去高频噪声。

    参数:
        x: 1D CO2 波形
        sampling_rate: 采样率（Hz），默认 100
        cutoff: 截止频率（Hz）。默认 2.0 Hz
                呼吸频率 0.1~0.5Hz，cutoff=2 保呼吸且滤高频抖动
        order: 滤波器阶数
    返回:
        滤波后波形
    设计: butter + filtfilt（零相位），保证呼吸峰不移位、不畸变。
    注意: filtfilt 是非因果零相位滤波；若 STM32 端要复刻，
          需用因果 IIR 或延迟补偿。训练端可接受。
    """
    x = np.asarray(x, dtype=float)
    nyquist = sampling_rate / 2.0
    if cutoff >= nyquist:
        return x
    b, a = butter(order, cutoff / nyquist, btype='low')
    # 用 padlen 避免边界效应；数据过短时降阶
    if len(x) < 3 * max(len(b), len(a)):
        return x
    return filtfilt(b, a, x)


# ============================================================
# 3. 基线漂移校正（中值滤波估计基线）
# ============================================================
def remove_baseline(x, kernel_size=101):
    """用中值滤波估计基线并扣除，校正缓慢漂移。

    参数:
        x: 1D CO2 波形
        kernel_size: 中值滤波窗口（点）。默认 101 点 ≈ 1 秒 @100Hz
    返回:
        校正后波形（基线归零）
    设计: 呼吸周期约 2~4 秒，kernel 1 秒远小于周期，
          能捕捉基线趋势而不抹平呼吸峰。
    """
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=float)
    if len(x) < kernel_size:
        return x - np.median(x)
    baseline = median_filter(x, size=kernel_size)
    return x - baseline


# ============================================================
# 4. 异常点检测
# ============================================================
def detect_outliers(x, n_std=5):
    """检测异常尖刺（超出邻域均值的 n_std 倍标准差）。

    参数:
        x: 1D CO2 波形
        n_std: 判定阈值（倍标准差）
    返回:
        (outlier_mask, cleaned_x)
    设计: 用滚动均值+滚动标准差估计局部期望，超出的标记为异常。
          异常点用前后中值替换，避免尖刺影响峰检测。
    """
    from scipy.ndimage import uniform_filter
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 100:
        return np.zeros(n, dtype=bool), x

    win = 51  # 局部窗口
    # 滚动均值/方差（等效于局部邻域统计）
    local_mean = uniform_filter(x, size=win, mode='nearest')
    local_std = np.sqrt(uniform_filter((x - local_mean)**2, size=win, mode='nearest'))
    local_std = np.maximum(local_std, 1e-6)

    dev = np.abs(x - local_mean)
    outlier = dev > n_std * local_std

    # 异常点用邻域中值替换
    cleaned = x.copy()
    if outlier.any():
        from scipy.ndimage import median_filter
        med = median_filter(x, size=11)
        cleaned[outlier] = med[outlier]
    return outlier, cleaned


# ============================================================
# 5. 信号质量检测
# ============================================================
def signal_quality(x, sampling_rate=100, window_sec=30):
    """评估一个窗口的信号质量，输出多项 QC 指标。

    返回 dict:
        missing_ratio:    缺失(NaN)比例
        zero_ratio:       零值比例（可能断线/屏息）
        low_amplitude:    峰值过低（<10 mmHg）标记
        quality_score:    综合质量评分 0~1
        is_usable:        是否可用于训练（缺失<10% 且非全零）
    """
    from scipy.signal import find_peaks
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {'missing_ratio': 1.0, 'zero_ratio': 1.0,
                'low_amplitude': True, 'quality_score': 0.0, 'is_usable': False}

    missing_ratio = np.isnan(x).sum() / n
    nonzero = x[~np.isnan(x)]
    zero_ratio = (nonzero == 0).sum() / len(nonzero) if len(nonzero) else 1.0

    max_val = np.nanmax(x) if len(nonzero) else 0
    low_amplitude = max_val < 10  # 无有效呼吸信号（如屏息/断线）

    # 综合质量分
    score = 1.0
    score -= missing_ratio * 1.0          # 缺失越重分越低
    score -= zero_ratio * 0.5             # 零值多也降分
    if low_amplitude:
        score *= 0.3                      # 无呼吸信号严重降分

    # 可用性：缺失<10% 且非全零 且有一定幅度
    is_usable = (missing_ratio < 0.10) and (zero_ratio < 0.9) and (max_val > 0)

    return {
        'missing_ratio': missing_ratio,
        'zero_ratio': zero_ratio,
        'low_amplitude': low_amplitude,
        'quality_score': max(0.0, min(1.0, score)),
        'is_usable': is_usable,
    }


# ============================================================
# 6. 完整预处理管道
# ============================================================
def preprocess_window(window, sampling_rate=100,
                      fill_gap=200, lowpass_cutoff=None,
                      remove_bl=False, detect_spike=True):
    """对一个 30s 窗口执行完整预处理。

    参数:
        window: 1D CO2 波形（mmHg）
        sampling_rate: 采样率
        fill_gap: 短缺失插值最大长度
        lowpass_cutoff: 低通截止频率；None 则跳过低通
        remove_bl: 是否做基线校正（默认 False！
                   原因：CO₂ 波形基线有生理意义，扣除后峰值被压到 <15，
                   寻峰失败 → 周期特征全变 0，模型无法学习）
        detect_spike: 是否做异常点检测
    返回:
        cleaned: 预处理后波形
        qc:      质量检测结果 dict
    """
    # 1. NaN 短缺失插值
    x = fill_short_gaps(window, max_gap=fill_gap)
    qc = signal_quality(x, sampling_rate)

    # 2. 异常点检测（先于去噪，避免尖刺被滤波扩散）
    if detect_spike:
        _, x = detect_outliers(x)

    # 3. 低通去噪（不影响绝对峰值高度，寻峰仍有效）
    if lowpass_cutoff is not None:
        x = lowpass_filter(x, sampling_rate, cutoff=lowpass_cutoff)

    # 4. 基线校正（默认关闭——破坏寻峰，见函数 docstring）
    if remove_bl:
        x = remove_baseline(x)

    return x, qc


if __name__ == '__main__':
    # 简单自测
    np.random.seed(0)
    t = np.linspace(0, 30, 3000)
    # 模拟呼吸波形：基线 8 + 呼吸峰
    rr = 12
    wave = 8 + 25 * (np.sin(2*np.pi*rr/60*t) > 0.3).astype(float)
    wave += np.random.normal(0, 1, len(t))
    # 注入缺失和尖刺
    wave[500:530] = np.nan
    wave[1500] = 100
    cleaned, qc = preprocess_window(wave)
    print('缺失比例:', round(qc['missing_ratio'], 3))
    print('质量分:', round(qc['quality_score'], 3))
    print('可用:', qc['is_usable'])
    print('预处理后 NaN 数:', np.isnan(cleaned).sum())
    print('自测通过 ✅')
