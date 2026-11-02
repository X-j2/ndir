# Stage 2 U15/U30 模型、校准与评价规范 v1

> **注：本文件为 v1；已被 `stage2_model_eval_spec_v2.md` 取代。**
> v2 修正因果决策时间、低支撑忽略、无窗口病例真值、B0 同口径和缓存指纹。

> 版本：stage2_model_eval_spec_v1  
> 日期：2026-11-01  
> 依据：Stage 1 修订 3 报告与 `stage1_quality_rule_spec_v3`  
> 范围：200 例开发集；Final Test 继续封存；不改固件、不做气体标定  
> 结果：`report/第2阶段-实施报告.md`、`report/stage2/`

---

## 1. 目标与边界

本阶段训练两个**单尺度**模型：

- **U15**：15 s 窗口、7.5 s 步长；
- **U30**：30 s 窗口、15 s 步长。

不训练 U60/U90，不做多尺度融合，不实现 Stage 3 的事件引擎。Final Test 的
50 个病例只用于读取 case id 做互斥断言，**不加载波形、标签、事件或概率**。

训练、权重选择、校准器选择、事件阈值选择全部只使用开发集内部的病例级
划分。

---

## 2. 数据与标签

### 2.1 数据

- 冻结开发集：`data_splits/cases_200.json`（200 例）。
- 窗口由 `tools/stage0_common.py::iter_windows()` 生成，沿用 Stage 0 的
  窗口-事件重叠标签规则。
- 窗口标签：U0/U1/U2/U3/ambiguous/ignore/`ignore_low_support`/unlabeled/other。
- **RF 训练类别只有 U0/U1/U2**；U3 是规则观察态，不进入 RF，也不进入窗口级
  三分类指标。低支撑事件窗不当作正例或负例。

### 2.2 训练子集

RF 只在以下窗口上训练：

```
label ∈ {U0, U1, U2}
quality_state == VALID
```

原因：

- `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 是 Stage 1 质量规范的**直接 U2
  证据路径**，不依赖 z 峰特征；
- `FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / `GAP` / `SATURATED` /
  `POLARITY_INVALID` / `QUANTILE_ARTIFACT` / `NOISY` 进入 UNKNOWN，不训练；
- 避免把中性零向量（`A=0`）的平坦窗与 VALID 波形混在同一个 RF 分布中。

训练集规模（开发集）：

| 尺度 | 训练窗 | U0 | U1 | U2 |
|---|---:|---:|---:|---:|
| U15 | 272,154 | 240,987 | 30,465 | 702 |
| U30 | 135,567 | 120,159 | 15,173 | 235 |

### 2.3 推理/评价的混合策略

对窗口输出：

| 质量状态 | 窗口级输出 |
|---|---|
| `UNKNOWN_STATES` | `UNKNOWN`（弃权） |
| `FLAT_WITH_HISTORY` | U2（高置信直接证据） |
| `FLAT_EVER_HISTORY` | U2（低置信直接证据） |
| `VALID` | 校准后 RF 的 argmax |

`FLAT_EVER_HISTORY` 不单独触发正式 Stage 3 ACTIVE 报警；在本阶段的原始
事件代理中只进入 SUSPECT 流。

---

## 3. 模型与训练

每个尺度一个随机森林：

```python
RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    random_state=42,
    class_weight=<selected within outer training fold>,
    n_jobs=-1,
)
```

类别权重候选（只在训练折内选择）：

| 名称 | 权重 |
|---|---|
| `none` | 无 |
| `u2x3` | U0:1, U1:1, U2:3 |
| `u2x10` | U0:1, U1:1, U2:10 |
| `u2x30` | U0:1, U1:1, U2:30 |
| `balanced` | sklearn `balanced` |

选择指标：内层 OOF 的 Macro-F1，随后 U2 F1，随后 log loss。类别权重不使
用 Final Test，也不使用外层测试折。

---

## 4. 病例级嵌套划分

### 4.1 外层

- 200 个开发病例按可靠 U2 事件数分层（0 / 1 / 2 / ≥3 事件）；
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`；
- 每个病例的全部窗口只出现在一个外层测试折；
- 外层测试折完全不参与该折的权重、校准器或阈值选择。

### 4.2 内层

- 每个外层训练集再做 3 折病例级、事件分层划分；
- 内层训练折拟合 RF，内层验证折产生 OOF 原始概率；
- 类别权重、校准器和事件阈值只在内层 OOF 上选择；
- 外层模型只在内层选择完成后拟合一次。

### 4.3 最终部署配置

为保存一个可用的 U15/U30 模型，额外使用**全部开发集**做 5 折 OOF 配置选择：

- 对每个权重候选生成 OOF 原始概率；
- 选择校准器、最终权重和事件阈值；
- 最终 RF 在全部开发集训练窗上重训。

最终配置的选择只用于部署模型/`model_meta.json`；报告中的主要性能数字仍
来自外层嵌套 OOF，不使用全开发集重训模型的自评。

---

## 5. 概率校准

### 5.1 候选方法

1. `none`：原始 RF 概率；
2. `temperature`：单参数温度缩放；
   `softmax(log(p)/T)`；
3. `multinomial_logistic`：以类别 0 为参照的两组对数概率比作为输入，
   拟合多项 logistic 回归。

### 5.2 选择与拟合

- 在内层 OOF 上做 leave-one-inner-fold-out：校准器在其余内层折拟合，
  在留出内层折计算 log loss；取平均 log loss 最小的方法；
- 选定方法后，在内层 OOF 全部行上重新拟合，得到外层测试折使用的校准器；
- **校准器从不使用外层测试折的标签或预测**；
- 校准指标来自外层 OOF 的交叉拟合预测，而不是用同一批数据拟合后再自评。

### 5.3 指标

- 多类 log loss；
- 每个类别的 Brier、ECE（15 个等宽箱）、AUC（one-vs-rest）；
- top-label ECE；
- 可靠性曲线分箱表（保存在结果 JSON）。

ECE 不追求“看起来小”；U2 极稀疏时等宽箱会集中在低概率区，报告必须连同
prevalence、AUC 和事件级指标一起解读。

---

## 6. 事件级评价（单尺度原始告警代理）

本阶段**没有 Stage 3 的 ACTIVE 事件引擎**（无时间衰减证据、无双阈值迟滞、
无最短持续时间）。因此事件级评价使用明确标注的**单尺度原始告警代理**：

1. 对每个病例、每个尺度，生成窗口级阳性掩码：
   - `FLAT_WITH_HISTORY`：阳性；
   - `VALID` 且校准后的 `p(U2) >= threshold`：阳性；
   - `FLAT_EVER_HISTORY`：仅进入 SUSPECT，不进入原始 ACTIVE 代理；
   - 其他 UNKNOWN 状态：不产生告警。
2. 只使用参考轨道有效比例 ≥ 0.90 的窗口；
3. 将阳性窗口的 `[start, start+window]` 按起点排序，间隔 ≤10 s 合并为
   单尺度告警事件；
4. 真值只用**可靠 U2 事件**（≥2 条实际 EtCO₂<10 读数）；低支撑事件只计数，
   不进入 TP/FP/FN；
5. 使用冻结的 `event_matching_spec_v1`：
   - 原始 ACTIVE 代理：一对一、同病例同类、有效重叠 ≥1 s 且 ≥较短事件 5%；
   - SUSPECT：pre-onset lead 与 post-onset TTD 分开，不进入 ACTIVE TP/PPV/F1。

事件阈值在每个外层折内用内层 OOF 选择，目标为原始代理的 event F1，随后
recall、较少 FP。阈值候选为 0.05–0.95（步长 0.05）与 0.98。外层测试折
只应用已选阈值。

**报告措辞**：这些数字只能称为“单尺度原始告警代理”，不能称为 Stage 3
ACTIVE 事件引擎性能，也不能称为设备实时性能。

### 6.1 事件指标

- TP/FN/FP、event recall、event PPV、event F1；
- FA/h：FP / 参考可评估小时；同时报告 FP / 总小时；
- 告警事件数、告警持续时间占比；
- TTD（含负值早期预警，单独报告 early lead）、30/60/90 s 检出率；
- SUSPECT：pre-onset recall、post-onset recall、any recall、lead/TTD 中位数、
  unmatched suspect 数及每小时误报；
- 所有主要事件指标给 **case-cluster bootstrap 95% CI（1000 次）**。

---

## 7. 窗口级指标

在全部外层 OOF 窗口上，只统计真实标签 U0/U1/U2：

- Accuracy（全部/覆盖）、覆盖率；
- 三类 P/R/F1、Macro-F1；
- U2 二分类 P/R/F1（UNKNOWN 计为未检出，不计正例）；
- case-cluster bootstrap 95% CI。

窗口级准确率不能替代事件级指标；两者必须同时报告。

---

## 8. 结果文件

- `report/stage2/u15_results.json`
- `report/stage2/u30_results.json`
- `report/stage2/u15_run.log`
- `report/stage2/u30_run.log`
- `models/stage2/u15_rf.joblib`
- `models/stage2/u30_rf.joblib`
- `models/stage2/u15_meta.json`
- `models/stage2/u30_meta.json`
- `models/stage2/u15_calibrator.json`
- `models/stage2/u30_calibrator.json`

模型文件只用于开发/复现，不是已标定设备模型，也不能替代 Final Test 评价。

---

## 9. 已知限制

1. 训练标签来自监护仪参考轨道，U2 是“参考 EtCO₂ 低值/CO₂ 缺失样事件”，
   不是临床确诊屏息。
2. 只有 VALID 窗进入 RF；平坦窗走规则路径。模型不能学习平坦-历史模式。
3. 训练数据是术中机械通气，样机是自主呼吸；域偏移未消除。
4. 单尺度事件代理没有迟滞/持续时间，PPV 低和 FA/h 高是预期风险，不能当作
   Stage 3 最终性能。
5. U30 的 VALID U2 训练窗只有 235 个，校准与事件指标 CI 较宽。
6. Final Test 未评价；不能据本阶段数字推断 Final Test 性能。
7. 未上端侧、未做 25 Hz、未做气体标定。
