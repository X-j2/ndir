"""
验证脚本：对比 scipy.find_peaks 与手写状态机的呼吸率计算结果
证明两种寻峰方式对同一波形输出一致的呼吸率(RR)
"""
import numpy as np
from scipy.signal import find_peaks
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

SAMPLING_RATE = 100      # train_model.py 用 100Hz
SAMPLING_RATE_25 = 25    # 上位机用 25Hz

# ============================================================
# 手写状态机寻峰（直接复制 co2_monitor.py 的 extract_features 逻辑）
# ============================================================
def manual_state_machine(data, sampling_rate):
    """与 co2_monitor.py extract_features() 中寻峰逻辑完全一致"""
    n = len(data)
    if n == 0:
        return 0

    arr = np.array(data, dtype=np.float32)

    peak_count = 0
    last_peak_idx = -100
    min_distance = int(0.8 * sampling_rate)  # 0.8秒间隔
    state = 0
    current_max = arr[0]
    current_min = arr[0]

    for i in range(1, n):
        if state == 0:  # 上升段，追踪波峰
            if arr[i] > current_max:
                current_max = arr[i]
            if current_max - arr[i] >= 8.0:  # 下降超过8 → 确认波峰
                if current_max > 15.0 and (i - last_peak_idx) >= min_distance:
                    peak_count += 1
                    last_peak_idx = i
                state = 1
                current_min = arr[i]
        else:  # 下降段，追踪波谷
            if arr[i] < current_min:
                current_min = arr[i]
            if arr[i] - current_min >= 8.0 or arr[i] < 10.0:
                state = 0
                current_max = arr[i]

    # 换算为每分钟呼吸率（train_model: 30秒窗口×2, 上位机: 按实际秒数换算）
    duration_sec = n / sampling_rate
    rr = peak_count * (60.0 / duration_sec)
    return rr, peak_count


# ============================================================
# scipy 寻峰（复制 train_model.py 的参数）
# ============================================================
def scipy_peaks(data, sampling_rate):
    """与 train_model.py 中 find_peaks 参数完全一致"""
    prominence = 8.0
    height = 15.0
    distance = int(0.8 * sampling_rate)

    peaks, _ = find_peaks(data, height=height, distance=distance, prominence=prominence)
    peak_count = len(peaks)

    duration_sec = len(data) / sampling_rate
    rr = peak_count * (60.0 / duration_sec)
    return rr, peak_count


# ============================================================
# 生成仿真呼吸波形
# ============================================================
def generate_breath_wave(rr_target, amplitude=18, baseline=38, duration_sec=30,
                          sampling_rate=100, noise_std=0.8):
    """生成单次呼吸波形（四段式合成，与 co2_monitor.py 逻辑一致）"""
    n_points = duration_sec * sampling_rate
    t = np.linspace(0, duration_sec, n_points)
    breath_freq = rr_target / 60.0  # Hz
    phase = 0.0
    co2 = np.zeros(n_points)

    for i in range(n_points):
        phase += breath_freq / sampling_rate
        if phase >= 1.0:
            phase -= 1.0

        if phase < 0.55:
            frac = phase / 0.55
            co2[i] = baseline - amplitude * (1.0 - frac)
        elif phase < 0.85:
            frac = (phase - 0.55) / 0.30
            co2[i] = baseline + amplitude * frac
        else:
            frac = (phase - 0.85) / 0.15
            co2[i] = baseline + amplitude * (1.0 - frac * 0.3)

    co2 += np.random.normal(0, noise_std, n_points)
    return co2


# ============================================================
# 主测试
# ============================================================
print("=" * 65)
print("scipy.find_peaks vs 手写状态机 呼吸率一致性验证")
print("=" * 65)

# 测试用例：覆盖5种呼吸模式
test_cases = [
    {"name": "正常呼吸",  "rr": 14,  "amplitude": 18, "baseline": 38},
    {"name": "急促呼吸",  "rr": 28,  "amplitude": 10, "baseline": 33},
    {"name": "呼吸缓慢",  "rr": 5,   "amplitude": 22, "baseline": 42},
    {"name": "屏息",      "rr": 0.5, "amplitude": 2,  "baseline": 37},
    {"name": "呼吸异常",  "rr": 12,  "amplitude": 8,  "baseline": 28},
]

print(f"\n{'模式':<10} {'设定RR':<8} {'scipy RR':<10} {'手写 RR':<10} {'差异':<8} {'峰数(scipy/手写)':<18} {'判定'}")
print("-" * 65)

total_diff = 0
all_pass = True

for tc in test_cases:
    # 生成30秒波形
    wave = generate_breath_wave(
        tc["rr"], tc["amplitude"], tc["baseline"],
        duration_sec=30, sampling_rate=SAMPLING_RATE, noise_std=0.8
    )

    # 两种方式计算
    rr_sp, pk_sp = scipy_peaks(wave, SAMPLING_RATE)
    rr_man, pk_man = manual_state_machine(wave, SAMPLING_RATE)

    diff = abs(rr_sp - rr_man)
    total_diff += diff
    status = "✅" if diff < 0.5 or (diff / max(rr_sp, 0.1) < 0.05) else "⚠️"

    print(f"{tc['name']:<10} {tc['rr']:<8.0f} {rr_sp:<10.1f} {rr_man:<10.1f} {diff:<8.2f} "
          f"{pk_sp}/{pk_man:<15} {status}")

    if diff >= 0.5 and diff / max(rr_sp, 0.1) >= 0.05:
        all_pass = False

avg_diff = total_diff / len(test_cases)
print("-" * 65)
print(f"平均差异: {avg_diff:.2f} 次/分")

# ============================================================
# 补充测试：多轮随机波形
# ============================================================
print(f"\n{'='*65}")
print("多轮随机测试（每模式各10轮，随机抖动振幅和基线）")
print("=" * 65)
random.seed(42)
np.random.seed(42)

multi_results = []
for tc in test_cases:
    diffs = []
    for _ in range(10):
        amp = tc["amplitude"] + random.uniform(-3, 3)
        bl = tc["baseline"] + random.uniform(-3, 3)
        wave = generate_breath_wave(
            tc["rr"], amp, bl,
            duration_sec=30, sampling_rate=SAMPLING_RATE, noise_std=1.0
        )
        rr_sp, _ = scipy_peaks(wave, SAMPLING_RATE)
        rr_man, _ = manual_state_machine(wave, SAMPLING_RATE)
        diffs.append(abs(rr_sp - rr_man))
    avg_d = np.mean(diffs)
    multi_results.append((tc["name"], avg_d, max(diffs)))
    print(f"  {tc['name']:<10} 10轮 | 平均差异={avg_d:.3f} | 最大差异={max(diffs):.3f}")

overall_avg = np.mean([r[1] for r in multi_results])
overall_max = max([r[2] for r in multi_results])
print(f"\n全部50轮测试 | 总平均差异={overall_avg:.3f} 次/分 | 最大单次差异={overall_max:.3f} 次/分")

# ============================================================
# 结论
# ============================================================
print(f"\n{'='*65}")
print("结论:")
print("=" * 65)
if all_pass and overall_avg < 0.3:
    print("✅ scipy.find_peaks 与手写状态机的呼吸率计算结果高度一致。")
    print(f"   平均偏差仅 {overall_avg:.2f} 次/分，在实际临床呼吸监测中可忽略不计。")
    print("   验证了训练端→推理端的特征提取等效性。")
else:
    print("⚠️ 存在一定偏差，需进一步分析参数差异。")
