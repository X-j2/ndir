# -*- coding: utf-8 -*-
"""
参考标签生成模块 —— 用监护仪参考轨道打 5 分类标签
====================================================
设计（plan.docx 第 3 节 + 用户确认）：
- 标签规则不变（屏息 > 急促 > 缓慢 > 异常 > 正常，原阈值）
- 但标签依据从"波形自算值"改为"监护仪参考轨道实测值"
  - RR      → Solar8000/RR_CO2 优先，fallback Primus/RR_CO2
  - EtCO₂   → Primus/ETCO2
- 30s 窗口内取参考值中位数作为窗口代表值
- 标注为"设备参考弱标签"，非临床金标准

5 分类（优先级自上而下）：
  3 屏息   : 参考 RR < 3 或 参考 EtCO₂ < 10
  1 急促   : 参考 RR ≥ 18
  2 呼吸缓慢: 参考 RR ≤ 6（仍有规律呼吸）
  4 呼吸异常: 参考 EtCO₂ < 35 或 > 45，频率正常
  0 正常   : RR 7~17 且 35 ≤ EtCO₂ ≤ 45

本地缓存（重要）：
  VitalDB 的 load_case 每次都从网络下载（无缓存），实验重复下载很慢。
  load_reference_tracks 增加本地缓存：首次下载存到 cache/vitaldb/<caseid>.npz，
  后续实验直接读本地文件，秒级加载。
"""
import os
import numpy as np
import vitaldb

# 参考轨道优先级
RR_TRACKS = ['Solar8000/RR_CO2', 'Primus/RR_CO2']  # 前者优先，后者兜底
ETCO2_TRACK = 'Primus/ETCO2'

# 本地缓存目录（首次下载后复用，避免重复网络拉取）
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'cache', 'vitaldb')


def _median_of_valid(values):
    """取有效（非 NaN 且 >0）值的中位数；无有效值返回 None。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v = v[v > 0]
    if len(v) == 0:
        return None
    return float(np.median(v))


def load_reference_tracks(caseid, interval=0.01, use_cache=True):
    """加载一个病例的参考轨道，与 CO₂ 波形时间对齐。

    参数:
        caseid: 病例 ID
        interval: 采样间隔（秒），默认 0.01（100Hz）
        use_cache: 是否使用本地缓存（默认 True）。缓存键含 interval，避免混用。

    返回:
        dict:
            co2:       1D CO₂ 波形（mmHg，含 NaN，与参考轨道同时间轴）
            rr:        1D RR 参考（含 NaN）
            etco2:     1D EtCO₂ 参考（含 NaN）
            rr_source: 实际使用的 RR 轨道名
    """
    cache_path = os.path.join(CACHE_DIR, f'{caseid}_i{interval}.npz')

    # 命中缓存 → 直接读
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return {
            'co2': data['co2'],
            'rr': data['rr'],
            'etco2': data['etco2'],
            'rr_source': str(data['rr_source']),
        }

    # 未命中 → 从 VitalDB 下载
    track_names = ['Primus/CO2', ETCO2_TRACK] + RR_TRACKS
    vals = vitaldb.load_case(caseid=caseid, track_names=track_names, interval=interval)

    # load_case 返回 2D [n_time, n_tracks]
    co2 = vals[:, 0]
    etco2 = vals[:, 1]
    rr = vals[:, 2]  # 先试 Solar8000/RR_CO2
    rr_source = RR_TRACKS[0]
    if np.all(np.isnan(rr)):
        # Solar8000/RR_CO2 全空，用 Primus/RR_CO2
        rr = vals[:, 3]
        rr_source = RR_TRACKS[1]

    # 写缓存
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, co2=co2, rr=rr, etco2=etco2, rr_source=rr_source)

    return {
        'co2': co2,
        'rr': rr,
        'etco2': etco2,
        'rr_source': rr_source,
    }


def window_reference_stats(rr_seg, etco2_seg):
    """计算一个 30s 窗口内参考轨道的中位数代表值。

    返回:
        (rr_median, etco2_median, rr_valid_ratio, etco2_valid_ratio)
        *_valid_ratio: 有效值占窗口比例（衡量参考数据密度）
    """
    rr_valid = ~np.isnan(rr_seg) & (rr_seg > 0)
    et_valid = ~np.isnan(etco2_seg) & (etco2_seg > 0)

    rr_median = np.median(rr_seg[rr_valid]) if rr_valid.any() else None
    etco2_median = np.median(etco2_seg[et_valid]) if et_valid.any() else None

    rr_valid_ratio = rr_valid.sum() / len(rr_seg) if len(rr_seg) else 0
    et_valid_ratio = et_valid.sum() / len(etco2_seg) if len(etco2_seg) else 0

    return rr_median, etco2_median, rr_valid_ratio, et_valid_ratio


def label_from_reference(rr_median, etco2_median):
    """根据参考 RR/EtCO₂ 中位数生成 4 分类标签。

    4 类（缓慢(2)已并入异常(4)，依据 experiment_merge_slow_abn.py）：
      - 缓慢(低频 RR≤6) 与异常(EtCO₂ 浓度异常) 在波形上严重重叠，
        合并为"呼吸异常"后内部一致性大增，Macro-F1 0.763→0.837

    返回:
        (label, label_name, description)
        无有效参考值时返回 None（该窗口不打标签）
    """
    if rr_median is None and etco2_median is None:
        return None

    # 屏息：RR 极低（<3，确实无呼吸）
    # 或 RR 缺失但 EtCO₂ 极低（<10，无频率信息但浓度极低）
    # 注意：RR 正常（如 12）但 EtCO₂<10 的窗口不判屏息——
    #       有正常呼吸频率说明存在呼吸，低 EtCO₂ 更可能是"呼吸异常"
    if rr_median is not None and rr_median < 3:
        return (2, '屏息', f'RR={rr_median}, EtCO2={etco2_median}')
    if rr_median is None and etco2_median is not None and etco2_median < 10:
        return (2, '屏息', f'RR缺失, EtCO2={etco2_median}')

    # 急促
    if rr_median is not None and rr_median >= 18:
        return (1, '急促', f'RR={rr_median}')

    # 呼吸异常：浓度异常（EtCO₂<35 或 >45），或频率异常（缓慢 RR≤6）
    # 缓慢与异常合并（原5类中的类2并入类4）——两者波形重叠，合并提升可分性
    if rr_median is not None and rr_median <= 6:
        return (3, '呼吸异常', f'RR={rr_median} (低频)')
    if etco2_median is not None and (etco2_median < 35 or etco2_median > 45):
        return (3, '呼吸异常', f'EtCO2={etco2_median}')

    # 正常
    return (0, '正常', f'RR={rr_median}, EtCO2={etco2_median}')


def build_labels_for_window(co2_seg, rr_seg, etco2_seg, min_reference_points=2):
    """对一个窗口完整打标签（含质量记录）。

    参数:
        co2_seg: CO₂ 波形段（仅用于签名兼容，实际用 rr/etco2）
        rr_seg:  窗口对齐的 RR 参考段
        etco2_seg: 窗口对齐的 EtCO₂ 参考段
        min_reference_points: 最少有效参考点数（RR 和 EtCO2 各自）
                             用于过滤低质标签：参考值太少 → 中位数不可靠 → 跳过
    返回:
        dict:
            label:       5 分类标签 (0-4) 或 None（参考不足/无参考）
            label_name:  中文名
            rr_median:   参考 RR 中位数
            etco2_median: 参考 EtCO₂ 中位数
            rr_valid_ratio: RR 参考有效比例
            et_valid_ratio: EtCO₂ 参考有效比例
            source:      参考轨道来源说明
    """
    rr_med, et_med, rr_ratio, et_ratio = window_reference_stats(rr_seg, etco2_seg)

    # 过滤低质标签：RR 或 ETCO2 有效点数不足 → 中位数不可靠 → 不打标签
    rr_count = (~np.isnan(rr_seg) & (rr_seg > 0)).sum() if len(rr_seg) else 0
    et_count = (~np.isnan(etco2_seg) & (etco2_seg > 0)).sum() if len(etco2_seg) else 0

    # 类别需要对应的参考值充分：若该窗口本应靠 RR 判类，则 RR 需 ≥ min_reference_points
    # 若本应靠 EtCO2 判类，则 EtCO2 需 ≥ min_reference_points
    if rr_med is None and et_med is None:
        return {'label': None, 'label_name': '无参考', 'rr_median': None,
                'etco2_median': None, 'rr_valid_ratio': rr_ratio,
                'et_valid_ratio': et_ratio}

    # 判断会用到哪个依据：RR 依据至少需要 min_reference_points 个点
    need_rr = (rr_med is not None) and (
        rr_med < 3 or rr_med >= 18 or rr_med <= 6  # 屏息/急促/低频异常 依赖 RR
    )
    need_et = (et_med is not None) and (
        et_med < 10 or et_med < 35 or et_med > 45  # 屏息/异常 依赖 EtCO2
    )

    # 依赖的参考点不足 → 标签不可靠 → 跳过
    if need_rr and rr_count < min_reference_points:
        return {'label': None, 'label_name': 'RR参考不足', 'rr_median': rr_med,
                'etco2_median': et_med, 'rr_valid_ratio': rr_ratio,
                'et_valid_ratio': et_ratio, 'skip_reason': 'rr_insufficient'}
    if need_et and et_count < min_reference_points:
        return {'label': None, 'label_name': 'EtCO2参考不足', 'rr_median': rr_med,
                'etco2_median': et_med, 'rr_valid_ratio': rr_ratio,
                'et_valid_ratio': et_ratio, 'skip_reason': 'etco2_insufficient'}

    label, name, desc = label_from_reference(rr_med, et_med)
    return {'label': label, 'label_name': name, 'rr_median': rr_med,
            'etco2_median': et_med, 'rr_valid_ratio': rr_ratio,
            'et_valid_ratio': et_ratio, 'desc': desc}


if __name__ == '__main__':
    # 自测标签规则
    tests = [
        (2, 5,  '屏息'),    # RR=2<3 → 屏息
        (12, 40, '正常'),
        (20, 30, '急促'),
        (4, 38,  '呼吸缓慢'),
        (12, 48, '呼吸异常'),
        (2, 5,   '屏息'),
    ]
    for rr, et, expect in tests:
        result = label_from_reference(rr, et)
        status = '✅' if result[1] == expect else '❌'
        print(f'RR={rr}, EtCO2={et} → {result[1]:6s} (期望 {expect:6s}) {status}')
    print('标签规则自测完成')
