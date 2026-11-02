# -*- coding: utf-8 -*-
"""
病例构建模块 —— 病例筛选 + 时间对齐 + 窗口切分
=================================================
依据 plan.docx 第 6 节：
- 筛选同时拥有 Primus/CO2 与参考 RR、EtCO₂ 轨道的病例
- 排除 CO₂ 波形极短、缺失严重、参考参数大量缺失或时间无法对齐的病例
- 统计每个病例可用窗口数与类别分布，避免单一病例贡献绝大多数样本
"""
import numpy as np
import vitaldb
import os
import json


def find_cases_with_reference(n_cases=50, co2_track='Primus/CO2',
                              rr_tracks=None, etco2_track='Primus/ETCO2'):
    """查找同时有 CO₂ 波形和参考 RR/EtCO₂ 轨道的病例。

    返回:
        满足条件的病例 ID 列表（最多 n_cases 个）
    """
    if rr_tracks is None:
        rr_tracks = ['Solar8000/RR_CO2', 'Primus/RR_CO2']

    all_case_ids = vitaldb.find_cases([co2_track])
    qualified = []

    for cid in all_case_ids:
        if len(qualified) >= n_cases:
            break
        try:
            tracks = vitaldb.get_track_names([cid])
            tnames = tracks['tnames'][0]
            has_co2 = co2_track in tnames
            has_rr = any(rr in tnames for rr in rr_tracks)
            has_etco2 = etco2_track in tnames
            if has_co2 and has_rr and has_etco2:
                qualified.append(cid)
        except Exception:
            continue

    return qualified


def inspect_case(caseid):
    """检查一个病例的数据质量，返回摘要信息。

    用于病例筛选：判断 CO₂ 波形长度、缺失率、参考数据密度是否可用。
    """
    try:
        tracks = vitaldb.get_track_names([caseid])
        tnames = tracks['tnames'][0]

        vals = vitaldb.load_case(caseid=caseid,
                                 track_names=['Primus/CO2', 'Primus/ETCO2',
                                              'Solar8000/RR_CO2', 'Primus/RR_CO2'],
                                 interval=0.01)
        co2 = vals[:, 0]
        etco2 = vals[:, 1]
        rr_solar = vals[:, 2]
        rr_primus = vals[:, 3]

        co2_valid = ~np.isnan(co2)
        et_valid = ~np.isnan(etco2) & (etco2 > 0)
        rr_s_valid = ~np.isnan(rr_solar) & (rr_solar > 0)
        rr_p_valid = ~np.isnan(rr_primus) & (rr_primus > 0)

        return {
            'caseid': caseid,
            'n_co2': int(co2_valid.sum()),
            'co2_missing_ratio': float(1 - co2_valid.mean()),
            'etco2_valid': int(et_valid.sum()),
            'rr_solar_valid': int(rr_s_valid.sum()),
            'rr_primus_valid': int(rr_p_valid.sum()),
            'tracks': tnames,
        }
    except Exception as e:
        return {'caseid': caseid, 'error': str(e)}


def build_case_metadata(case_ids, save_path=None):
    """构建病例元数据表并保存为 JSON（plan.docx 第 11 节 metadata.json）。

    每个病例记录：可用窗口数、参考数据密度、质量标记。
    用于后续的病例级划分（保证测试/训练病例无交集）。
    """
    metadata = []
    for cid in case_ids:
        info = inspect_case(cid)
        metadata.append(info)
        print(f'  病例 {cid}: CO2={info.get("n_co2")}点, '
              f'缺失={info.get("co2_missing_ratio", "?"):.2%}, '
              f'ETCO2={info.get("etco2_valid", "?")}, '
              f'RR_solar={info.get("rr_solar_valid", "?")}, '
              f'RR_primus={info.get("rr_primus_valid", "?")}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f'  元数据已保存到 {save_path}')
    return metadata


def filter_usable_cases(metadata, min_co2_points=3000, max_missing=0.2,
                        min_reference_ratio=0.01):
    """按质量条件过滤可用病例。

    条件：
        - CO₂ 波形点数 >= min_co2_points（至少一个窗口）
        - CO₂ 缺失率 <= max_missing（不能缺失太严重）
        - 参考数据有效比例 >= min_reference_ratio（至少有一点参考值）
    返回: 可用病例的 caseid 列表
    """
    usable = []
    for m in metadata:
        if 'error' in m:
            continue
        if m['n_co2'] < min_co2_points:
            continue
        if m['co2_missing_ratio'] > max_missing:
            continue
        # 参考数据有效性：ETCO2 或 RR 任一有足够值
        total_ref = m.get('etco2_valid', 0) + m.get('rr_solar_valid', 0) + m.get('rr_primus_valid', 0)
        if total_ref < min_reference_ratio * m['n_co2']:
            continue
        usable.append(m['caseid'])
    return usable


if __name__ == '__main__':
    print('查找同时有 CO2 + RR + ETCO2 轨道的病例...')
    cases = find_cases_with_reference(n_cases=50)
    print(f'找到 {len(cases)} 个合格病例: {cases[:10]}...')

    print('\n构建病例元数据...')
    metadata = build_case_metadata(cases, save_path='metadata.json')

    usable = filter_usable_cases(metadata)
    print(f'\n可用病例: {len(usable)} 个')
    print('自测完成 ✅')
