# CF-REM 事件匹配规范 v1（Stage 0 冻结）

> 版本：event_matching_spec_v1（内容修订 rev3）  
> 日期：2026-10-29  
> 状态：Stage 0 冻结；Stage 3 事件引擎与 Stage 4 评价必须使用本规范  
> 依赖：`event_spec_v1.md`、`labels_spec_v1.md`

---

## 1. 目的

v1.0/v1.1 的 TTD、事件召回、FA/h、PPV/F1 使用了三套不相容的匹配规则，同一个告警可以同时是“召回”和“误报”。本规范定义**唯一、逐类、一对一**的真值-告警匹配，所有事件级指标都从同一匹配结果派生。

---

## 2. 输入

- **真值事件** `T`：来自 `event_spec_v1.md`，每项 `{case_id, class, onset, offset}`，半开区间 `[onset, offset)`。**主指标只用可靠 U2（≥2 条与事件网格关联的 EtCO2<10 原始读数运行段；来源可早于整数秒起点但在 cap 内）**；低支撑 U2（<2 读）是 `indeterminate`，不计 TP/FP/FN/PPV/F1/FA_h，单独报告。
- **告警事件** `A`：来自事件引擎，每项 `{case_id, class, onset, offset, stream}`：
  - `stream = SUSPECT`：窗口级早期嫌疑，不要求持续门槛；
  - `stream = ACTIVE`：正式确认事件，带迟滞/最短持续。
- 只对**同一病例、同一类别**匹配；不跨病例、不跨类别。

---

## 3. 候选对

### 3.1 ACTIVE（正式事件）

对每个真值 `t` 与 ACTIVE 告警 `a`（同类、同病例），必须满足**预冻结的有效重叠条件**：

```
overlap = max(0, min(t.offset, a.offset) - max(t.onset, a.onset))
min_dur = min(t.offset - t.onset, a.offset - a.onset)
candidate = (overlap >= 1.0s) and (overlap >= 0.05 * min_dur)
```

- `MIN_ACTIVE_OVERLAP_SEC = 1.0s`；
- `MIN_ACTIVE_OVERLAP_FRAC = 0.05`（较短事件时长的 5%）；
- 两个条件必须同时满足。仅重叠 0.1s 的“擦边”不是 TP。
- **无重叠的宽限时间不构成 ACTIVE 候选，也不可能是 TP。** 即使告警落在 `[t.onset-10s, t.offset+30s]` 内，只要没有有效重叠，它就是 FP（诊断上可标为 `early_alarm` 或 `late_alarm`，但不改变 TP/FP/FN）。
- 这两个阈值是 Stage 0 预冻结值，**不得**在 Final Test 上调。

### 3.2 SUSPECT（早期提示）

SUSPECT 由 `match_suspect` 单独处理，不进入 ACTIVE 匹配；分两条流：

1. **事件前预警（pre-onset）**
   ```
   same case/class
   0 < (t.onset - a.onset) <= pre_lead_allow   # 10s
   ```
   报告 `lead`；严格早于事件起点。

2. **事件开始后的早期检出（post-onset）**
   ```
   same case/class
   0 <= (a.onset - t.onset) <= post_onset_allow   # 30s
   a.onset <= t.offset
   ```
   报告**实际 TTD**（`a.onset - t.onset`），不是 lead。事件开始当秒的提示是 TTD=0，归入 post-onset，不算“事件前预警”。

- 同一真值优先取 pre-onset；没有 pre 才取最早的 post-onset。
- 一条 SUSPECT 最多匹配一个真值；重复/未匹配的 SUSPECT 单独报为 false early。
- SUSPECT 不参与 ACTIVE 的 TP/PPV/F1/FA/h。

### 3.3 宽限参数

- `pre_allow = 10s`、`post_allow = 30s` 只用于 **诊断/报告**（early_alarm、late_alarm），不用于把无重叠告警升级为 TP。
- 事件召回/PPV/F1/FA/h 的主指标不使用宽限时间。

---

## 4. 一对一匹配

### 4.1 评分

对每个候选对计算：

```
overlap = max(0, min(t.offset, a.offset) - max(t.onset, a.onset))
union   = (t.offset - t.onset) + (a.offset - a.onset) - overlap
iou     = overlap / union          # 交集/并集；不是交集/最大时长
onset_dist = abs(a.onset - t.onset)
```

排序键（降序优先）：

```
(overlap, iou, -onset_dist, t.onset, a.onset, t_id, a_id)
```

### 4.2 贪心匹配

1. 生成所有候选对并按上述键排序；
2. 依次取候选对；若 `t` 未被匹配且 `a` 未被匹配，则匹配；
3. 已匹配的真值/告警不再参与后续匹配；
4. 匹配结果 `M = {(t_i, a_j)}` 是**一对一**的。

### 4.3 TP/FP/FN

- `TP = |M|`；
- `FN = |T| - TP`；
- `FP = |A| - TP`（包括重复告警、类别错误、无真值覆盖的告警）。

**重复告警**：同一真值的第二个告警在第一次匹配后变成 FP；不得把多个告警都算 TP。

**类别错误**：类别不一致的告警不参与该类匹配，计为该告警类别的 FP；另报“类别混淆”计数，不改变 TP/FP/FN 定义。

**无重叠宽限告警**：落在 `[t.onset-10s, t.offset+30s]` 但无重叠的 ACTIVE 告警是 FP；只报为 early/late 诊断，不得计 TP。

---

## 5. 指标（全部从同一匹配派生）

### 5.1 事件召回 / PPV / F1

```
recall = TP / (TP + FN)
PPV    = TP / (TP + FP)
F1     = 2*TP / (2*TP + FP + FN)
```

### 5.2 TTD

对每个匹配对：

```
TTD = a.onset - t.onset
```

- `TTD >= 0`：检测延迟；报告 median、p90、30/60/90s 内检出率。
- `TTD < 0`：早期预警；单独报告 `early_lead = -TTD` 的 median/p90 和比例，**不得静默截断为 0**。
- `SUSPECT` 分两条流，不能混：
  - **pre-onset**：报告 `lead = t.onset - a.onset`（>0）；
  - **post-onset**：报告**实际 TTD = a.onset - t.onset**（≥0）；
  - 两条流的 median/召回分开报告；`SUSPECT` 不进入 ACTIVE 的 TTD。
- `ACTIVE` 的 TTD 用于正式检测延迟；与 `SUSPECT` 分开。

### 5.3 FA/h

```
FA/h = FP / evaluable_hours
```

- 主指标用 `evaluable_hours`（参考可用、非 UNKNOWN、非长缺口）。
- 同时报告 `FP / total_hours` 作为敏感性。
- UNKNOWN 区间内的告警不计入 FP；单独报告 `unknown_alarm_rate`。
- 无参考时段（unlabeled）不得默认成“无事件”；其告警也不计入主 FA/h，单独报告。

### 5.4 告警负担

- `alarms/hour = |A| / evaluable_hours`；
- `alarm_duration_fraction = sum(a.offset-a.onset) / evaluable_seconds`；
- `SUSPECT` 与 `ACTIVE` 分开报告。

### 5.5 置信区间

- 按 **case** 做 cluster bootstrap（1000 次）；事件数少时同时按 event 重采样；
- 报告两个口径，不能只报窗口级 bootstrap（重叠窗不独立）。

---

## 6. SUSPECT 与 ACTIVE 的分工

| 流 | 定义 | 进入 | 不进入 |
|---|---|---|---|
| SUSPECT | 窗口级早期嫌疑，无持续门槛 | **pre-onset lead**、**post-onset TTD**、早期检出率、早期误报率 | 事件 TP/PPV/F1、主 FA/h |
| ACTIVE | 事件引擎确认，带双阈值迟滞/最短持续 | 事件 TP/FP/FN、召回、PPV、F1、主 FA/h | — |

**“15s 出疑似”只承诺 SUSPECT；不能把它写成 ACTIVE 事件确认。**  
v1.1 的 U15 每 5s 更新 + `L>2` 持续 10s，最早 ACTIVE 可能约 25s；Stage 3 必须把 SUSPECT 和 ACTIVE 分开报告。

---

## 7. 工作例子

**例 1：ACTIVE 重叠 → TP**

真值：`[100, 110)`；ACTIVE：`[95, 105)`。

- overlap = 5；union = 10+10-5 = 15；IoU = 1/3；
- 候选（overlap>0）→ TP；
- TTD = 95 - 100 = **-5s**；early lead=5s，单独报告。

**例 2：ACTIVE 无重叠但落在 pre_allow → FP，不是 TP**

真值：`[100, 110)`；ACTIVE：`[95, 96)`。

- overlap = 0；即使 `95 ∈ [90, 140]`，也不构成 ACTIVE 候选；
- FP；诊断上记为 `early_alarm`；
- 若是 SUSPECT，则进入早期提示评估，lead=5s，不计 ACTIVE TP。

**例 3：ACTIVE 无重叠但落在 post_allow → FP，不是 TP**

真值：`[100, 110)`；ACTIVE：`[112, 113)`。

- overlap = 0；即使 `112 ∈ [90, 140]`，也不构成 ACTIVE 候选；
- FP；诊断上记为 `late_alarm`。

**例 3b：极小重叠 → FP**

真值：`[100, 110)`；ACTIVE：`[100, 100.1)`。

- overlap = 0.1s；`overlap < 1.0s` 且 `< 0.05 * min_dur`；
- 不构成有效 ACTIVE 候选 → FP；不得因 0.1s 擦边算 TP。

**例 4：ACTIVE 在事件内开始并延伸出去 → TP**

真值：`[100, 110)`；ACTIVE：`[105, 115)`。

- overlap = 5；union = 15；IoU = 1/3；
- TP；TTD = 5s。

**例 5：重复告警 → 1 TP + 1 FP**

真值：`[100, 110)`；ACTIVE `[105, 106)` 与 `[107, 108)`：

- 两个都与真值重叠；排序后 `[105,106)`（overlap=1）先匹配 → TP；
- `[107,108)` 未匹配 → FP；不得把两个都算 TP。

**例 6：SUSPECT 分 pre-onset / post-onset 评估**

真值：`[100, 110)`；SUSPECT `[95, 96)`：

- `0 < 100-95 = 5 <= 10` → pre-onset 预警；lead=5s；
- 不计 ACTIVE TP/PPV/F1/FA_h。

SUSPECT `[100, 101)`：

- lead=0，不是严格 pre-onset；按 post-onset，TTD=0。

SUSPECT `[105, 106)`：

- post-onset，TTD=5s；不能写成 lead=5s。

SUSPECT `[112, 113)`：

- 晚于 offset=110，不是事件开始后的早期检出；不匹配。

---

## 8. 监测小时定义

- `total_hours`：病例有任意数据的时长；
- `evaluable_hours`：参考轨道可用、非 UNKNOWN、非长缺口、非饱和的时长；
- 主 FA/h 用 `evaluable_hours`；
- 报告覆盖率：`evaluable_hours / total_hours`；
- 报告每类事件的 `events_per_evaluable_hour`，避免只报绝对数。

---

## 9. 必须报告的诊断量

- 真值事件数、病例数、每类事件率；
- 匹配对数、未匹配真值数、未匹配告警数；
- 重复告警数、类别错误数、UNKNOWN 区间告警数；
- SUSPECT/ACTIVE 分开：ACTIVE TTD；SUSPECT pre-onset lead 与 post-onset 实际 TTD；各自召回；
- 低支撑 U2 事件数（indeterminate，不进主分母）；
- case/event cluster bootstrap 95% CI；
- 每折/测试集的事件数；若低于预注册最小值，降低结论强度。

---

## 10. 验收测试

1. 真值 `[100,110)`、ACTIVE `[95,96)` → FP（early_alarm），不是 TP。
2. 真值 `[100,110)`、ACTIVE `[100,100.1)` → 0.1s 重叠，不满足有效重叠 → FP，不是 TP。
3. 真值 `[100,110)`、ACTIVE `[100,101)` → 1.0s 重叠，满足有效重叠 → TP，TTD=0。
4. 同一真值两个重叠告警 `[105,106)`、`[107,108)` → 1 TP + 1 FP。
5. 真值 `[100,110)`、ACTIVE `[112,113)` → FP（late_alarm），不是 TP。
6. 真值 `[100,110)`、ACTIVE `[105,115)` → TP，TTD=5，IoU=1/3。
7. 真值 `[100,110)`、ACTIVE `[95,105)` → TP，TTD=-5，IoU=1/3。
8. SUSPECT `[95,96)` → pre-onset lead=5s，不计 ACTIVE TP。
9. SUSPECT `[100,101)` → post-onset，TTD=0；不能写成 lead。
10. SUSPECT `[105,106)` → post-onset，TTD=5s。
11. SUSPECT `[112,113)` → 晚于 offset，不匹配。
12. 类别错误 → 该类 FP，不匹配另一类真值。
13. 低支撑 U2 事件 → 不进主 TP/FN；单独报告 indeterminate。
14. UNKNOWN 区间告警 → 不计 FP，单独报告。
15. case/event bootstrap 的 CI 可复现（固定 seed）。
16. 同一输入重复运行，匹配结果完全一致。
17. 事件匹配不使用 Final Test 调参；Final Test 只评估一次。

---

## 11. 已知限制

- `pre_allow/post_allow`、`MIN_ACTIVE_OVERLAP_SEC=1.0s`、`MIN_ACTIVE_OVERLAP_FRAC=0.05` 是工程参数，不是临床金标准；必须在开发集预注册，不能按测试结果调。
- 低支撑 U2 事件（<2 条与事件网格关联的实际读数）不进入主 TP/FN；它们可能是真事件也可能是 ffill 伪迹，当前数据无法区分，只能单独报告。
- 事件起止精度约 1–2s；TTD 的秒级差异不宜过度解读。
- 机械通气数据中，EtCO₂ 低值事件可能不是自主呼吸暂停；匹配只表示“与参考轨道事件一致”，不表示临床确诊。
- 事件数少时，PPV/F1/FA/h 的 CI 会很宽；必须报告实际事件数，不能只报点估计。
