import vitaldb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 解决画图时中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 改动1：用 find_cases 自动查找含 CO2 波形的病例（3个 → 50个）
#       find_cases 是 vitaldb 自带的搜索函数，相当于"查数据库索引"
# ============================================================
print("1. 正在从 VitalDB 查找包含 CO2 波形数据的病例...")
all_case_ids = vitaldb.find_cases(['Primus/CO2'])
case_ids = all_case_ids[:50]
print(f"-> 数据库中共 {len(all_case_ids)} 个病例含 CO2 波形，使用前 {len(case_ids)} 个")

SAMPLING_RATE = 100
window_size = 3000  # 30秒窗口 (100Hz × 30 = 3000点)

# ============================================================
# 改动2：按病例逐个窗口化，记录每个窗口属于哪个病例
#       （为按病例分层划分训练/测试集做准备，避免同一病例的窗口跨训练集和测试集）
# 改动3：新增 5 个波形形态特征（全部从 CO2 原始波形计算，STM32 端可 1:1 复现）
# ============================================================
X_features = []
y_labels = []
case_id_per_sample = []  # 每个窗口属于哪个病例，用于按病例分层划分
plot_samples = {}

for case_idx, cid in enumerate(case_ids):
    print(f"  下载病例 {cid} ({case_idx+1}/{len(case_ids)})...", end=" ")

    try:
        vals = vitaldb.load_case(caseid=cid, track_names=['Primus/CO2'], interval=0.01)
    except Exception as e:
        print(f"下载失败 ({e})，跳过")
        continue

    # load_case 单轨道返回 1D 数组，直接去除 NaN
    co2_wave = vals[~np.isnan(vals)]

    if len(co2_wave) < window_size:
        print(f"数据太短（{len(co2_wave)}点），跳过")
        continue

    n_windows = (len(co2_wave) - window_size) // window_size + 1
    print(f"{len(co2_wave)} 点 → {n_windows} 个窗口")

    for i in range(0, len(co2_wave) - window_size, window_size):
        window_data = co2_wave[i:i + window_size]

        if np.max(window_data) <= 0:
            continue

        # ---------- 特征提取（全部从 CO2 原始波形计算）----------
        mean_val = np.mean(window_data)
        max_val = np.max(window_data)
        std_val = np.std(window_data)

        # 寻峰（参数与 STM32 部署代码完全一致）
        peaks, _ = find_peaks(
            window_data,
            height=15,
            distance=int(0.8 * SAMPLING_RATE),  # 两次呼吸至少间隔 0.8 秒
            prominence=8
        )

        breath_count = len(peaks)
        rr_per_min = breath_count * (60 / 30)  # 30秒 × 2 = 每分钟

        # ---- 波形形态特征（全从 CO2 波形提取，STM32 端可复现）----
        if breath_count >= 2:
            etco2_vals = window_data[peaks]
            etco2_mean = np.mean(etco2_vals)   # 平均潮末 CO2
            etco2_std = np.std(etco2_vals)     # 呼吸间变异性
        elif breath_count == 1:
            etco2_mean = window_data[peaks[0]]
            etco2_std = 0.0
        else:
            etco2_mean = max_val
            etco2_std = 0.0

        # 特征向量：4 个基础 + 2 个形态 = 6 维（删除了重要性 <5% 的 baseline/amplitude/rise_rate）
        feature_vector = [
            mean_val, max_val, std_val, rr_per_min,   # 原有 4 维
            etco2_mean, etco2_std                       # 潮末值、变异性
        ]

        # ---------- 标签规则引擎（与原有逻辑完全一致）----------
        if max_val < 10 or rr_per_min < 3:
            label = 3
            label_name = "屏息 (3)"
        elif rr_per_min >= 18:
            label = 1
            label_name = "急促 (1)"
        elif rr_per_min <= 6:
            label = 2
            label_name = "呼吸缓慢 (2)"
        elif max_val < 35 or max_val > 45:
            label = 4
            label_name = "呼吸异常 (4)"
        else:
            label = 0
            label_name = "正常 (0)"

        X_features.append(feature_vector)
        y_labels.append(label)
        case_id_per_sample.append(cid)

        if label_name not in plot_samples and len(plot_samples) < 5:
            plot_samples[label_name] = window_data

# ============================================================
# 改动4：合成数据注入（适量减少，适配 9 维特征）
#       有 50 个病例的真实数据后，合成数据仅用于补充极端边缘情况
# ============================================================
import random

random.seed(42)
np.random.seed(42)
print(f"\n-> 从 VitalDB 共提取 {len(X_features)} 个真实窗口样本。")
print("-> 正在注入少量合成边缘样本以补充极端情况...")

for _ in range(20):  # 少量合成样本，仅补充极端边缘情况
    rand_mean = random.uniform(15.0, 25.0)
    rand_std = random.uniform(10.0, 15.0)

    # 急促 (1): RR >= 18，EtCO2 偏低
    etco2_1 = random.uniform(28.0, 38.0)
    X_features.append([
        rand_mean, etco2_1, rand_std, random.uniform(18.0, 30.0),
        etco2_1, random.uniform(2.0, 8.0)
    ])
    y_labels.append(1)

    # 呼吸缓慢 (2): RR 3~6，CO2 蓄积偏高
    etco2_2 = random.uniform(35.0, 50.0)
    X_features.append([
        rand_mean, etco2_2, rand_std, random.uniform(3.0, 6.0),
        etco2_2, random.uniform(1.0, 5.0)
    ])
    y_labels.append(2)

    # 屏息 (3): RR < 3
    etco2_3 = random.uniform(0.0, 9.0)
    X_features.append([
        random.uniform(0, 5), etco2_3, random.uniform(1.0, 4.0), random.uniform(0.0, 2.9),
        etco2_3, 0.0
    ])
    y_labels.append(3)

    # 呼吸异常 (4): 浓度异常但频率正常（偏低和偏高各一条）
    for abnormal_etco2 in [random.uniform(20.0, 34.9), random.uniform(45.1, 60.0)]:
        X_features.append([
            rand_mean, abnormal_etco2, rand_std, random.uniform(6.0, 22.0),
            abnormal_etco2, random.uniform(2.0, 10.0)
        ])
        y_labels.append(4)

# 合成样本标记为 -1，一律归入训练集
n_synthetic = len(X_features) - len(case_id_per_sample)
case_id_per_sample.extend([-1] * n_synthetic)

# ============================================================

X = np.array(X_features)
y = np.array(y_labels)
case_id_per_sample = np.array(case_id_per_sample)
print(f"-> 最终样本数: {len(X)} 组（{len(np.unique(case_id_per_sample))} 个病例）")
print(f"   类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"   特征维度: {X.shape[1]} 维（全部可从 CO2 原始波形计算，STM32 端可 1:1 复现）")

# ============================================================
# 改动5：按病例分层划分训练/测试集
#       同一病例的所有窗口要么全在训练集，要么全在测试集
#       这样测试集里的病人模型从未见过 → 评估更真实
# ============================================================
print("\n2. 按病例分层划分训练/测试集（避免同一病人数据泄露）...")
# 只对真实病例分层，合成样本（id=-1）全部归训练集
real_cases = np.unique(case_id_per_sample[case_id_per_sample >= 0])
train_cases, test_cases = train_test_split(real_cases, test_size=0.2, random_state=42)

train_mask = np.isin(case_id_per_sample, train_cases) | (case_id_per_sample == -1)
test_mask = np.isin(case_id_per_sample, test_cases)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"   训练集: {len(X_train)} 样本（{len(train_cases)} 个病例）")
print(f"   测试集: {len(X_test)} 样本（{len(test_cases)} 个病例，模型从未见过这些病人）")

# ============================================================
# 以下: 波形图 + 模型训练 + ONNX 导出（结构不变）
# ============================================================

print("\n3. 正在生成波形坐标图...")
# 生成 5 类波形图
n_plots = len(plot_samples)
fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 1.6))
if n_plots == 1:
    axes = [axes]
fig.suptitle("EtCO2 训练数据波形截取 (30秒窗口)", fontsize=16)

for ax, (label_name, wave_data) in zip(axes, plot_samples.items()):
    time_axis = np.linspace(0, 30, len(wave_data))
    ax.plot(time_axis, wave_data, color='blue', linewidth=1.2)
    ax.set_title(f"类别样本: {label_name}")
    ax.set_ylabel("EtCO2 (mmHg)")
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel("时间 (秒)")
plt.tight_layout()
plt.savefig("etco2_waveforms.png", dpi=300)
print("-> 图片已保存为 etco2_waveforms.png！")

print("\n4. 划分数据集并训练五分类模型...")

total_samples = len(X)
train_samples = len(X_train)
test_samples = len(X_test)

print(f"   📊 数据集统计:")
print(f"      - 总样本数: {total_samples} 组")
print(f"      - 训练样本数 (80%病例): {train_samples} 组")
print(f"      - 测试样本数 (20%病例): {test_samples} 组")
print(f"      - 特征维度: {X.shape[1]} 维")

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("-> 模型训练完成！")

print("\n5. 开始评估模型表现...")
y_pred = model.predict(X_test)
unique_labels = np.unique(y_test)
target_names = [f"Class {lbl}" for lbl in unique_labels]
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

# 补充：也看一下训练集上的表现（对比判断是否过拟合）
y_pred_train = model.predict(X_train)
print("训练集表现（参考，判断过拟合）:")
print(classification_report(y_train, y_pred_train, labels=np.unique(y_train),
      target_names=[f"Class {lbl}" for lbl in np.unique(y_train)]))

print("\n6. 正在导出为 STM32CubeMX 可用的 ONNX 格式...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("etco2_multiclass_model1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"-> 五分类模型已保存为 etco2_multiclass_model1.onnx（输入维度: {X.shape[1]}）")
