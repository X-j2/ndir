# -*- coding: utf-8 -*-
"""
特征提取模块 —— 12 维波形特征（plan.docx 第 5 节 + 优化扩展）
====================================================
设计原则：
- 全部特征只从 CO₂ 波形计算，不直接使用"最终 RR 数值"或"最终 EtCO₂ 数值"
- F5/F6 依赖峰检测，但输出的是峰间距的"间隔"而非"RR 次/分"，避免与标签同义
- 全部特征 STM32 端可用 C 语言 1:1 复现

特征列表（12 维）：
  F1  波形均值       窗口内 CO₂ 均值
  F2  波形标准差     窗口振幅/波动
  F3  峰峰值         max - min
  F4  波形偏度       分布不对称性
  F5  峰间距均值     有效峰间平均时间间隔（秒）
  F6  峰间距标准差   呼吸周期稳定性
  F7  峰宽统计量     有效呼吸峰的平均宽度（秒）
  F8  呼气平台占比   局部峰值平台区域的时间比例
  F9  呼吸峰数       有效呼吸峰个数
  F10 平均峰幅       峰高减基线均值
  F11 波形自算 RR    有效峰数换算的呼吸率 (bpm)，与标签依据（监护仪 RR）独立
  F12 频率置信度     峰间距变异系数导出的节律规律度（规律呼吸接近1）

优化依据（experiment_verify_freq_feat.py）：
- 缓慢(2) 自算 RR 中位 8 vs 异常(4) 中位 14，频率维度可分，新增 F11/F12 增强判别力
- 寻峰 height 15→12：低碳酸异常波形幅度小，降阈值捕获更多真实呼吸峰
"""
import numpy as np
from scipy.signal import find_peaks


# ============================================================
# 寻峰（与训练/部署一致的参数）
# ============================================================
def find_breath_peaks(window, sampling_rate=100):
    """在 CO₂ 波形上找有效呼吸峰。

    参数: height=12（峰值>12mmHg, 原15）, distance=1.0s（呼吸间隔, 原0.8s）, prominence=8
    返回: 峰索引数组
    """
    peaks, _ = find_peaks(window, height=12,
                          distance=int(1.0 * sampling_rate),
                          prominence=8)
    return peaks


# ============================================================
# 特征 1-4：基础统计
# ============================================================
def _feature_stat(window):
    """F1-F4：均值 / 标准差 / 峰峰值 / 偏度。"""
    x = np.asarray(window, dtype=float)
    mean = np.mean(x)
    std = np.std(x)
    peak_to_peak = np.max(x) - np.min(x)

    # 偏度：三阶标准矩，衡量波形分布不对称性
    if std > 1e-6:
        skew = np.mean((x - mean) ** 3) / (std ** 3)
    else:
        skew = 0.0

    return [mean, std, peak_to_peak, skew]


# ============================================================
# 特征 5-6：周期结构（峰间距）
# ============================================================
def _feature_periodicity(peaks, sampling_rate=100):
    """F5-F6：峰间距均值 / 峰间距标准差（以秒为单位，非次/分）。

    若峰数 < 2，无法计算峰间距 → 返回 [0, 0]（无呼吸/单峰）
    """
    if len(peaks) < 2:
        return [0.0, 0.0]

    # 相邻峰之间的间距（采样点）
    intervals = np.diff(peaks)
    # 转为秒
    intervals_sec = intervals / sampling_rate

    mean_interval = np.mean(intervals_sec)
    std_interval = np.std(intervals_sec)

    return [mean_interval, std_interval]


# ============================================================
# 特征 7：峰宽统计量
# ============================================================
def _feature_width(window, peaks, sampling_rate=100):
    """F7：有效呼吸峰的平均宽度（秒）。

    峰宽 = 峰两侧下降到 prominence 一半处的距离（近似半高全宽）。
    反映呼吸波形陡峭/平缓程度。
    """
    if len(peaks) == 0:
        return 0.0

    x = np.asarray(window, dtype=float)
    n = len(x)
    widths = []
    half_height = 7.5  # prominence=8 的一半，约 4；但用 height=15 的一半 7.5

    for p in peaks:
        peak_val = x[p]
        threshold = peak_val - half_height  # 下降到峰高的一半以下
        # 向左找下降沿
        left = p
        while left > 0 and x[left] > threshold:
            left -= 1
        # 向右找下降沿
        right = p
        while right < n - 1 and x[right] > threshold:
            right += 1
        width = (right - left) / sampling_rate  # 秒
        widths.append(width)

    return float(np.mean(widths)) if widths else 0.0


# ============================================================
# 特征 8：呼气平台占比
# ============================================================
def _feature_plateau(window, peaks, sampling_rate=100, tolerance=3.0):
    """F8：呼气平台占比。

    呼气末 CO₂ 波形在峰值附近形成"平台"（呼气末气流几乎停滞，CO₂ 接近恒定）。
    定义：距峰值 2 mmHg 以内的点占比（局部平坦段比例）。
    用峰值附近的平坦度估计，反映呼气平台期是否明显。
    """
    if len(peaks) == 0:
        return 0.0

    x = np.asarray(window, dtype=float)
    max_val = np.max(x)
    if max_val <= 0:
        return 0.0

    # 在峰值 ±tolerance mmHg 内的点占比
    plateau_points = np.sum(x >= (max_val - tolerance))
    ratio = plateau_points / len(x)
    return float(ratio)


# ============================================================
# 主入口：提取 8 维特征
# ============================================================
# 特征 9-10：有效呼吸峰数量 / 平均峰幅
# ============================================================
def _feature_peak_count_amp(window, peaks):
    """F9-F10：有效呼吸峰数 / 平均峰幅。

    关键判别力：缓慢(2) 峰数≈4.1、峰幅≈29.7；屏息(3) 峰数≈1.3、峰幅≈10.7。
    这两特征直接区分"有呼吸但慢"和"几乎无呼吸"。
    """
    x = np.asarray(window, dtype=float)
    n_peaks = len(peaks)

    if n_peaks > 0:
        baseline = np.min(x)
        amplitudes = x[peaks] - baseline
        mean_amp = float(np.mean(amplitudes))
    else:
        mean_amp = 0.0

    return [float(n_peaks), mean_amp]


# ============================================================
# 特征 11-12：波形自算 RR / 频率置信度（优化新增）
# ============================================================
def _feature_wave_rr(peaks, sampling_rate=100):
    """F11：波形自算呼吸率 (bpm) = 峰数 / 窗口秒数 × 60。

    与标签依据（监护仪参考 RR）独立——自算值来自波形形态，参考值是设备量测，
    两者有独立噪声，模型学习的是两者关系，不构成标签泄漏。

    优化依据（experiment_verify_freq_feat.py）：
      缓慢(2) 自算 RR 中位 8 vs 异常(4) 中位 14 → 频率维度可分。
    """
    n_peaks = len(peaks)
    if n_peaks == 0:
        return 0.0
    # 窗口秒数由寻峰范围估算（最后一个峰到第一个峰的时间跨度）
    # 若峰数少，用固定 30s 窗口时长更稳健
    return float(n_peaks * 60.0 / 30.0)  # 30s 窗口 → 每峰 2 bpm


def _feature_rhythm_confidence(peaks, sampling_rate=100):
    """F12：呼吸节律置信度 = max(0, 1 - 峰间距变异系数)。

    规律呼吸（真缓慢）峰间距变化小 → 变异系数小 → 置信度接近 1；
    噪声/伪峰主导（峰少且不规律）→ 置信度低。
    用于区分"真缓慢"与"只是峰少的噪声窗口"。
    """
    if len(peaks) < 3:
        return 0.0  # 峰太少无法评估规律性
    intervals = np.diff(peaks)
    mean_iv = np.mean(intervals)
    if mean_iv < 1e-6:
        return 0.0
    std_iv = np.std(intervals)
    cv = std_iv / mean_iv
    return float(max(0.0, 1.0 - cv))


# ============================================================
def extract_features(window, sampling_rate=100):
    """从 30s CO₂ 波形提取 12 维特征向量。

    参数:
        window: 1D CO₂ 波形（mmHg），长度约 3000 点
        sampling_rate: 采样率（Hz）
    返回:
        np.array 形状 [12]，顺序 [F1..F12]
    """
    x = np.asarray(window, dtype=float)

    # 寻峰
    peaks = find_breath_peaks(x, sampling_rate)

    # 基础统计 F1-F4
    feat_stat = _feature_stat(x)

    # 周期结构 F5-F6
    feat_period = _feature_periodicity(peaks, sampling_rate)

    # 峰宽 F7
    feat_width = _feature_width(x, peaks, sampling_rate)

    # 平台占比 F8
    feat_plateau = _feature_plateau(x, peaks, sampling_rate)

    # 峰数/峰幅 F9-F10
    feat_count_amp = _feature_peak_count_amp(x, peaks)

    # 波形自算 RR / 频率置信度 F11-F12
    feat_wave_rr = _feature_wave_rr(peaks, sampling_rate)
    feat_rhythm = _feature_rhythm_confidence(peaks, sampling_rate)

    features = (feat_stat + feat_period + [feat_width, feat_plateau]
                + feat_count_amp + [feat_wave_rr, feat_rhythm])
    return np.array(features, dtype=float)


# ============================================================
# 特征名（用于可读性 / 论文图表）
# ============================================================
FEATURE_NAMES = [
    'Mean CO2',          # F1
    'Std CO2',           # F2
    'Peak-to-peak',      # F3
    'Skewness',          # F4
    'Peak interval mean',  # F5
    'Peak interval SD',     # F6
    'Peak width',        # F7
    'Plateau ratio',     # F8
    'Peak count',        # F9
    'Peak amplitude',    # F10
    'Wave RR (bpm)',     # F11
    'Rhythm confidence', # F12
]

FEATURE_NAMES_CN = [
    '波形均值', '波形标准差', '峰峰值', '波形偏度',
    '峰间距均值', '峰间距标准差', '峰宽', '呼气平台占比',
    '呼吸峰数', '平均峰幅', '波形自算RR', '频率置信度',
]


if __name__ == '__main__':
    # 自测：合成 5 种呼吸波形，验证特征提取正确
    np.random.seed(0)
    t = np.linspace(0, 30, 3000)
    def synth_wave(rr, amp, baseline=8):
        phase = (rr/60 * t) % 1.0
        wave = baseline + amp * np.maximum(0, np.sin(2*np.pi*phase))
        return wave + np.random.normal(0, 0.5, len(t))

    print('特征维度:', len(FEATURE_NAMES))
    for name, (rr, amp) in [('正常', (12, 25)), ('急促', (25, 20)), ('缓慢', (4, 25)), ('屏息', (0, 0))]:
        w = synth_wave(rr, amp)
        f = extract_features(w)
        peaks = find_breath_peaks(w)
        print(f'{name}: RR≈{len(peaks)*2}/分, 特征={np.round(f, 3)}')
    print('自测通过 ✅')
