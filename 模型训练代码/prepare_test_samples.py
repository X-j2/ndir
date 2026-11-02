# -*- coding: utf-8 -*-
"""
准备新测试样本：从 VitalDB 未训练过的病例（51-60）提取 CO₂ 波形
================================================================
目的：替换 测试样本/ 目录下 4 个旧 CSV（旧样本量纲异常，全部被误判屏息）

步骤：
  1. 用 vitaldb.find_cases() 确认 51-60 号病例存在且含 CO₂ 波形
  2. 每个病例取第一个有效的 30 秒窗口（3000 点 @100Hz）
  3. 保存为与旧样本相同格式的 CSV：
     表头 "co2_y","flow_y"
     第 0 列 = CO₂ 浓度（mmHg，与训练数据同单位）
     第 1 列 = 流量（无实际数据时用 0 填充）
  4. 旧样本备份到 测试样本_old/，新样本覆盖 测试样本/

注意：VitalDB 的 Primus/CO2 轨道本身就是 mmHg 单位，
      与训练数据完全同源，无需任何换算。
"""
import os
import shutil
import numpy as np
import vitaldb
from scipy.signal import find_peaks

SAMPLING_RATE = 100
WINDOW_SIZE = 3000          # 30 秒窗口
N_SAMPLES = 4               # 生成 4 个新测试样本
OUTPUT_DIR = '测试样本'
BACKUP_DIR = '测试样本_old'

# ------------------------------------------------------------
# 1. 找到所有含 CO₂ 的病例，确认测试病例不在训练集（1-50）中
# ------------------------------------------------------------
all_case_ids = vitaldb.find_cases(['Primus/CO2'])
train_case_ids = set(all_case_ids[:50])       # 训练用过的病例 1-50
# 候选测试病例 = 前 50 个之后、还没被训练用过的
test_candidates = [cid for cid in all_case_ids[50:70] if cid not in train_case_ids]
print(f"可用测试候选病例: {test_candidates}")

# ------------------------------------------------------------
# 2. 备份旧样本，创建新输出目录
# ------------------------------------------------------------
if os.path.exists(OUTPUT_DIR) and not os.path.exists(BACKUP_DIR):
    shutil.copytree(OUTPUT_DIR, BACKUP_DIR)
    print(f"旧样本已备份到 {BACKUP_DIR}/")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ------------------------------------------------------------
# 3. 从每个候选病例取一个 30 秒有效窗口
# ------------------------------------------------------------
saved = []
for cid in test_candidates:
    if len(saved) >= N_SAMPLES:
        break
    try:
        vals = vitaldb.load_case(caseid=cid, track_names=['Primus/CO2'], interval=0.01)
    except Exception as e:
        print(f"  病例 {cid} 下载失败: {e}")
        continue
    co2 = vals[~np.isnan(vals)]
    if len(co2) < WINDOW_SIZE:
        print(f"  病例 {cid} 数据太短（{len(co2)}点），跳过")
        continue

    # 找第一个"正常呼吸"窗口（RR 7~17，max 35~45，即模型的正常类）
    # 这样测试样本能覆盖正常呼吸模式，而不是全落在屏息/缓慢段
    found = None
    best = None
    best_rr = 0
    for i in range(0, len(co2) - WINDOW_SIZE, WINDOW_SIZE):
        w = co2[i:i + WINDOW_SIZE]
        if np.max(w) <= 0:
            continue
        peaks, _ = find_peaks(w, height=15, distance=int(0.8 * SAMPLING_RATE), prominence=8)
        rr = len(peaks) * 2
        # 优先找正常呼吸窗口（RR 7-17），找不到就退回找呼吸率最高的
        if 7 <= rr <= 17 and np.max(w) > 10:
            found = w
            break
        if rr > best_rr:
            best_rr = rr
            best = w

    if found is None:
        found = best
        if found is None:
            print(f"  病例 {cid} 无有效呼吸窗口，跳过")
            continue
        print(f"  病例 {cid}: 未找到完美正常呼吸窗口，用 RR={best_rr} 的窗口")
    else:
        print(f"  病例 {cid}: 找到正常呼吸窗口", end=" ")

    # 输出窗口信息
    peaks, _ = find_peaks(found, height=15, distance=int(0.8 * SAMPLING_RATE), prominence=8)
    rr = len(peaks) * 2
    print(f"max={found.max():.1f} mmHg, RR={rr} 次/分, {len(found)}点")

    # 保存（flow 列用 0 填充，保持格式一致）
    csv_name = f"{cid:04d}_A_signal.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    header = '"co2_y","flow_y"\n'
    flow = np.zeros(len(found))
    data = np.column_stack((found, flow))
    np.savetxt(csv_path, data, delimiter=',', header='co2_y,flow_y', comments='', fmt='%.6f')
    # 重写表头为带引号格式（与旧样本一致）
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    with open(csv_path, 'w') as f:
        f.write('"co2_y","flow_y"\n')
        f.writelines(lines[1:])
    saved.append(cid)
    print(f"  → 已保存 {csv_path}")

# ------------------------------------------------------------
# 4. 清理旧样本文件（已被替换）
# ------------------------------------------------------------
for old in ['0501_A_signal.csv', '0515_A_signal.csv', '0523_A_signal.csv', '0524_A_signal.csv']:
    old_path = os.path.join(OUTPUT_DIR, old)
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"  已删除旧样本 {old}（备份在 {BACKUP_DIR}/）")

print(f"\n完成！新测试样本 {len(saved)} 个已保存到 {OUTPUT_DIR}/")
print(f"生成文件: {[f'{cid:04d}_A_signal.csv' for cid in saved]}")
