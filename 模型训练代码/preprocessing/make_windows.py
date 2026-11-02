# -*- coding: utf-8 -*-
"""
窗口切分模块 —— 构建 30s CO₂ 窗口 + 对齐参考轨道 + 质量记录
==============================================================
依据 plan.docx 第 4.2 节：
- 30s 非重叠窗口（后续可做 15/30/45/60s 敏感性实验）
- 窗口必须同时映射到两套时间轴：
    CO₂ 波形窗口 + 同一 [t0, t1] 范围内的 RR/EtCO₂ 参考数据
- 标签由窗口内参考值的中位数决定，不是简单取一个时间点
- 记录每个窗口的缺失率、质量、参考来源，供低质量窗口剔除
"""
import numpy as np
from .signal_processing import fill_short_gaps, get_valid_segments, signal_quality


def make_windows_from_clean(co2, window_size=3000, step=None, sampling_rate=100):
    """从连续有效 CO₂ 波形切 30s 窗口。

    参数:
        co2: 已去除 NaN 的连续波形（mmHg）
        window_size: 窗口长度（点），默认 3000
        step: 步长（点），默认 = window_size（非重叠）
    返回:
        窗口列表 [ (start_idx, window_array), ... ]
    注意: 修正边界——保证能切出最后一个完整窗口
    """
    if step is None:
        step = window_size
    n = len(co2)
    windows = []
    for start in range(0, n - window_size + 1, step):
        windows.append((start, co2[start:start + window_size]))
    return windows


def make_windows_with_reference(co2_raw, rr_raw, etco2_raw,
                                window_size=3000, step=None,
                                sampling_rate=100,
                                max_missing_ratio=0.10):
    """从带 NaN 的波形 + 参考轨道构建窗口，两套时间轴对齐。

    流程：
      1. 短缺失插值（fill_short_gaps），保留时间轴
      2. 按连续有效区间切分（长缺失处断开，各自独立）
      3. 每个有效区间内切 30s 窗口
      4. 每窗口取同时间轴的参考 RR/EtCO₂ 段
      5. 质量检查，剔除高缺失窗口

    返回:
        窗口记录列表，每项:
            {
                'start': 窗口起点（原波形索引）
                'co2':   30s CO₂ 波形（已插值，可能还有长缺口处被切走）
                'rr':    对齐的 RR 参考段（含 NaN）
                'etco2': 对齐的 EtCO₂ 参考段（含 NaN）
                'missing_ratio': 窗口内 CO₂ 缺失比例
                'qc':    质量检测结果
            }
    """
    co2 = np.asarray(co2_raw, dtype=float)
    rr = np.asarray(rr_raw, dtype=float)
    etco2 = np.asarray(etco2_raw, dtype=float)

    # 1. 短缺失插值
    co2_filled = fill_short_gaps(co2)

    # 2. 连续有效区间
    valid_segments = get_valid_segments(co2_filled, min_len=window_size)

    # 3-5. 切窗 + 对齐参考 + 质量检查
    records = []
    for seg_start, seg in valid_segments:
        for start, window in make_windows_from_clean(seg, window_size, step, sampling_rate):
            abs_start = seg_start + start

            # 对齐参考轨道（同时间轴切片）
            rr_seg = rr[abs_start:abs_start + window_size] if abs_start + window_size <= len(rr) else np.array([])
            et_seg = etco2[abs_start:abs_start + window_size] if abs_start + window_size <= len(etco2) else np.array([])

            # 质量检测（CO₂ 窗口本身）
            qc = signal_quality(window, sampling_rate, window_sec=window_size / sampling_rate)

            # 剔除缺失过高的窗口
            if qc['missing_ratio'] > max_missing_ratio:
                continue

            records.append({
                'start': abs_start,
                'co2': window,
                'rr': rr_seg,
                'etco2': et_seg,
                'missing_ratio': qc['missing_ratio'],
                'qc': qc,
            })

    return records


def summarize_windows(records):
    """统计窗口构建结果（论文/报告用）。"""
    n = len(records)
    if n == 0:
        return {'n_windows': 0, 'avg_missing': 0}
    missing = [r['missing_ratio'] for r in records]
    return {
        'n_windows': n,
        'avg_missing': float(np.mean(missing)),
        'max_missing': float(np.max(missing)),
    }


if __name__ == '__main__':
    # 自测
    np.random.seed(0)
    n = 100000  # 1000 秒波形
    t = np.arange(n) / 100
    wave = 8 + 25 * (np.sin(2*np.pi*0.2*t) > 0).astype(float)  # 0.2Hz 呼吸
    wave += np.random.normal(0, 1, n)
    # 注入缺失段
    wave[5000:5300] = np.nan  # 3 秒缺失
    wave[10000:20000] = np.nan  # 100 秒长缺失

    rr_ref = np.full(n, np.nan)
    et_ref = np.full(n, np.nan)
    rr_ref[::700] = 12  # 每 7 秒一个参考值
    et_ref[::700] = 40

    records = make_windows_with_reference(wave, rr_ref, et_ref)
    print(f'构建窗口数: {len(records)}')
    if records:
        print(f'平均缺失率: {np.mean([r["missing_ratio"] for r in records]):.3f}')
        print(f'首个窗口参考 RR: {records[0]["rr"][records[0]["rr"] > 0][:3]}')
    print('自测完成 ✅')
