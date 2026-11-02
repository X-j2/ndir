# Stage 2 U15/U30 模型、校准与评价规范 v2

> 版本：stage2_model_eval_spec_v2  
> 日期：2026-11-01  
> 依据：Stage 1 rev3 质量规范、Stage 2 首轮外部审查意见  
> 范围：200 例开发集；Final Test 继续封存；不改固件、不做气体标定  
> 本规范取代 `stage2_model_eval_spec_v1.md`

---

## 0. 本轮修订（对应外部审查五项）

1. **因果决策时间**：窗口 `[start, end)` 的预测只在 `end` 可用；输入证据区间与
   告警输出区间分开，告警不得回填到 `start`；
2. **低支撑事件忽略**：低支撑区间不作正例/负例；完全落入其中的告警不计
   TP/FP/FN；跨越不可评分区间的告警裁剪/切分，不跨区间合并；
3. **无窗口病例真值保留**：无可用波形窗口的病例仍保留可靠/低支撑事件；
   全量可靠事件端到端召回计入 FN；
4. **B0/RF 同口径比较**：相同病例、窗口、真值类别、质量规则、UNKNOWN 计分；
   三分类只含真值 U0/U1/U2；
5. **缓存硬校验**：缓存命中后仍核对病例/划分/规范/实现指纹；不一致显式重建。

---

## 1. 目标与边界

训练两个单尺度模型：U15（15 s / 7.5 s）与 U30（30 s / 15 s）。

不做 U60/U90、多尺度证据累积、双阈值迟滞、Stage 3 ACTIVE 引擎、端侧部署或
气体标定。Final Test 只读取 `case_ids` 做互斥断言，不加载波形、标签、事件或
概率。

---

## 2. 数据与标签

### 2.1 训练子集

RF 只在以下窗口训练：

```
label ∈ {U0, U1, U2}
quality_state == VALID
```

- U3 不进入 RF；真值 U3 不进入三分类指标。
- `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 走规则直接 U2 路径；
- 其他 UNKNOWN 状态（`FLAT_NO_HISTORY`、`FLAT_HISTORY_UNKNOWN`、`GAP`、
  `SATURATED`、`POLARITY_INVALID`、`QUANTILE_ARTIFACT`、`NOISY`）弃权；
- 低支撑事件窗不可作正例或负例，不进入训练。

### 2.2 无窗口病例

`_build_case()` 必须先构建可靠/低支撑事件，再切窗口。即使 `rows` 为空：

- 病例仍保留 `reliable_u2_events` 和 `low_support_u2_events`；
- 事件分层使用完整事件表；
- 事件级召回的分母包含这些病例的可靠事件；
- 因为没有窗口/告警而无法检出的事件计 FN，不得从分母删除。

开发集冻结事件数必须逐病例核对为：

- 可靠 U2：**303 事件 / 143 病例**；
- 低支撑 U2：**60 事件 / 46 病例**。

### 2.3 缓存校验

缓存元数据必须包含：

- `cache_schema_version`；
- `window_sec`、`step_sec`；
- `dev_case_ids`（完整 200 例，与 `cases_200.json` 完全一致）；
- `dataset_fingerprint`；
- `fingerprint_components`（每个被校验文件/规范的 SHA256）。

指纹至少覆盖：

- `features/feature_spec_si_v2.json`、`features/si_features.py`、
  `features/quality_states.py`；
- `tools/stage0_common.py`、`preprocessing/make_windows.py`、
  `preprocessing/signal_processing.py`、`training/stage2_data.py`；
- `docs/specs/labels_spec_v1.md`、`docs/specs/event_spec_v1.md`、
  `docs/specs/quality_abstention_spec_v1.md`、
  `docs/specs/stage1_quality_rule_spec_v3.md`、本规范；
- `data_splits/cases_200.json`、`data_splits/final_test_seed42.json`、
  `data_splits/case_event_counts.json`、`data_splits/split_manifest.json`。

缓存命中时必须同时满足：

1. schema/window/step/版本匹配；
2. 指纹匹配；
3. 缓存 `dev_case_ids` 与当前冻结开发集完全一致；
4. 开发集与 Final Test 无交集；
5. 缓存内窗口 `case_id` 均属于开发集，且不包含 Final Test 病例。

任一条件不满足（包括缺少指纹的旧缓存）都必须打印明确原因并**重建**，不得
静默读取旧缓存。数据集指纹写入阶段结果 JSON 与模型 metadata。

---

## 3. 模型、划分与校准

模型、类别权重候选、外层 5 折/内层 3 折、校准候选与选择方式沿用 v1：

- RF 30 树/深 8；
- 外层病例级、事件分层 5 折；内层病例级 3 折；
- 权重/校准器/阈值只在内层 OOF 选择；
- 最终部署配置用全开发集 OOF 选择，只用于保存模型；
- 报告主指标来自外层嵌套 OOF。

类别权重、校准器、概率指标的定义与 v1 相同；唯一变化是事件阈值必须在
**因果决策时间修正后的事件代理**上选择。

---

## 4. 因果决策与告警输出

### 4.1 输入证据与输出时间分离

对每个窗口：

```
evidence_start = start
evidence_end   = end = start + window_sec
decision_time  = end
```

窗口 `[start, end)` 的预测在 `end` 才能产生。**禁止**把告警开始时间写成
`start`，也不得把证据区间和输出区间混为同一字段。

### 4.2 单尺度保持规则

对每个正决策（RF 阳性或 `FLAT_WITH_HISTORY`）：

```
hold_end = min(decision_time + step_sec, case_end)
raw_alarm = [decision_time, hold_end)
```

- 连续正决策（下一个决策时刻恰好等于当前 `hold_end`）可以合并为一个告警；
- 负决策、UNKNOWN 决策、缺失决策或数据中断处**不延伸**、**不跨区间合并**；
- 不使用“间隔 ≤10 s 合并”来跨越负决策/不可评分区间；
- 原始告警记录 `decision_time`，即使后续被低支撑/未知区间裁剪，该时间保留。

若 `hold_end <= decision_time`（病例已结束/无下一更新时段），不产生告警。

### 4.3 理论可用时间

本阶段所有 TTD、提前量、检出率都按 `decision_time` 计算，不含运行耗时；
只能称为“窗口结束时的理论可用时间”，不能称为实际设备延迟。

### 4.4 回归反例（必须通过）

```
窗口 [0, 30)，决策时刻 30；
真值事件起点 20；
首次正决策 30。
→ 告警从 30 开始；TTD = 30 - 20 = +10 s。
不得报告 TTD = -20 s 或“提前 20 s 预警”。
```

---

## 5. 低支撑与可评分时域

### 5.1 可评分区间

每病例在 1 s 网格上定义：

```
reference_valid = ffill(etco2, cap=10 s) 非 NaN
low_support     = 低支撑 U2 事件区间（全部，不判正/负）
scorable        = reference_valid AND NOT low_support
```

`unknown` 为 `reference_valid=False` 的区间。`scorable_intervals` 由连续
`scorable=True` 的 1 s 单元合并。

- 低支撑事件**不做可靠正例，也不做负例**；
- 参考未知区间不可评分；
- 可评分时长、被排除时长、低支撑排除时长、参考未知排除时长都要报告。

### 5.2 告警裁剪/切分

对每条原始告警 `[decision_time, hold_end)`：

1. 与每个 `scorable_interval` 求交；
2. 非空交集生成一个**可评分告警片段**，保留 `decision_time`；
3. 多个片段的并集不能跨不可评分区间合并；
4. 完全无可评分交集的告警不计 TP/FP/FN，计数为 `unscorable_alarm_episodes`；
5. 不能因为告警有一部分落在不可评分区间就丢弃整条告警。

**回归反例（必须通过）**：

```
病例只有 1 个低支撑事件；
告警完全位于该低支撑事件区间内；
→ FP = 0；该告警记为 unscorable，不参与 TP/FP/FN。
```

### 5.3 指标时域一致性

- **FA/h** 分子为可评分告警片段中未匹配的 FP 数，分母为
  `sum(scorable_sec)/3600`；
- **告警持续时间占比** 分子为可评分告警片段总时长，分母为同一
  `scorable_sec`；
- 另报总时长敏感性，但不得替代主口径；
- 报告被排除的小时数和原因；UNKNOWN 区间内的告警不参与主 FA/h。

---

## 6. 因果事件匹配与 TTD

事件匹配的**重叠判定**使用可评分告警片段；**序时判定**使用
`decision_time`：

- 活跃告警候选对：同病例、同类，可评分片段与真值事件有效重叠
  `>=1 s` 且 `>=5% * min(duration)`；
- 候选排序：`(overlap, intersection/union, -|decision_time - truth.onset|,
  truth.onset, decision_time, truth_id, alarm_id)`；
- 一对一贪心匹配；
- `TTD = decision_time - truth.onset`，负值单独报告为早期预警；
- 30/60/90 s 检出率使用该 TTD；
- SUSPECT 的 pre-onset lead 与 post-onset TTD 同样使用 `decision_time`；
  pre/post 分类以 `decision_time` 为准；
- 事件匹配不得使用窗口 `start` 回填告警起点。

**回归反例（必须通过）**：

```
真值 [20, 40)；窗口 [0,30) 的首次决策 = 30；
可评分告警片段从 30 开始。
→ active 匹配的 TTD = +10 s，early_lead = 0。
```

---

## 7. B0 与 RF 同口径比较

### 7.1 窗口级

- 同一批外层 OOF 窗口；
- 相同质量状态、相同 UNKNOWN 计分、相同真值类别 U0/U1/U2；
- B0 使用 Stage 1 `predict_rule` 的同一规则；
- B0 预测 U3 保留为一个非 U2 预测，不丢弃；
- 真值 U3 不进入三分类指标（RF 也不训练 U3），不能只把 B0 的 U3 当 FP；
- B0 的 Macro-F1/U2 P/R/F1由代码在同一 OOF 行上复算，不抄旧表。

### 7.2 事件级

若报告 RF 事件级提升，必须同时报告同口径 B0：

- B0 阳性决策 = B0 预测 U2 的窗口；
- B0 也使用相同的 `decision_time` 保持规则、可评分裁剪、低支撑排除、
  TTD/匹配和全量 303 事件分母；
- B0 没有概率阈值，直接以 U2 决策为阳性；
- 报告 B0/RF 的 TP/FN/FP、recall/PPV/F1、FA/h、TTD 和 SUSPECT。

---

## 8. 窗口级与事件级指标

窗口级仍报告 U0/U1/U2 的 P/R/F1、Macro-F1、覆盖率、U2 二分类指标及
case-cluster bootstrap 95% CI。

事件级主结果：

- 全量开发集 **303 个可靠 U2 事件**的端到端 recall（无窗口/不可评分导致的
  漏检计 FN）；
- TP/FN/FP、event recall/PPV/F1；
- 可评分 FA/h、告警持续时间占比、被排除时长；
- TTD（含负值早期预警）、30/60/90 s 检出率；
- SUSPECT pre/post 分开；
- case-cluster bootstrap 95% CI。

如另报“可评估事件子集”，必须给出：

- 排除规则；
- 病例数、事件数；
- 与全量结果的差异。

该子集不能替代全量 303 事件的主结果。

---

## 9. 结果与复现

- 结果：`report/stage2/u15_results.json`、`u30_results.json`；
- 模型：`models/stage2/u15_rf.joblib`、`u30_rf.joblib` 及 metadata；
- 数据集指纹写入 `results.json["dataset_fingerprint"]` 和
  `u{15,30}_meta.json["dataset_fingerprint"]`；
- 回归测试必须覆盖：
  1. `[0,30)` 真值 20 → TTD +10；
  2. 低支撑-only 告警 → FP=0、unscorable；
  3. 跨低支撑/未知告警 → 裁剪/切分且保留 decision_time；
  4. 无窗口病例事件保留、计入 FN；
  5. 缓存 case_ids/指纹失效 → 显式重建；
  6. B0/RF 同口径窗口指标；
  7. 303/60 事件逐病例核对。

---

## 10. 已知限制

- TTD 是窗口结束时的理论可用时间，不含推理/传输/渲染耗时；
- 单尺度原始代理没有 Stage 3 证据累积/迟滞，FA/h 仍可能偏高；
- 低支撑/参考未知的排除会降低事件召回，必须与全量结果同时解读；
- U2 弱标签、术中机械通气域偏移、未标定设备仍未解决；
- Final Test 未评价，不能据本阶段数字推断 Final Test 性能。
