# Stage 3 事件引擎、相对模式与隔离回放规范 v1

> **注：本文件为 v1；已被 `stage3_event_engine_spec_v2.md` 取代。**
> v2 修正严格连续计数、缺失决策中断、转换事件快照、回放病例集校验和
> 累计信号时间调度。以 v2、`report/第3阶段-实施报告-修订1.md` 和当前
> SSOT 为准。

> 版本：stage3_event_engine_spec_v1  
> 日期：2026-11-01  
> 依据：Stage 2 rev1 报告、`stage2_model_eval_spec_v2`、`stage1_quality_rule_spec_v3`  
> 范围：U15/U30 单尺度事件引擎；上位机相对模式；隔离回放；不改固件、不做气体标定  
> Final Test：继续封存；不加载波形/标签/事件/概率  
> 多尺度融合、U60/U90：留到 Stage 4

---

## 1. 事件语义与边界

正式名称：

> **呼气 CO₂ 低值/消失样事件**  
> reference low-CO₂ / CO₂-disappearance-like event

没有独立气流/呼吸努力/人工标注时，不得写成“临床确诊屏息/呼吸暂停”。

本阶段只运行 **U15 与 U30 两个独立单尺度引擎**。两个引擎可以同时显示，
但**不做证据累积/融合**；任一引擎的 ACTIVE 只代表该尺度自己的告警。

---

## 2. 时间语义

对每个输入窗口：

```
evidence_start = start
evidence_end   = end
decision_time  = end
```

- 证据只能在 `decision_time` 使用；
- **estimated onset**：本 episode 第一次出现 SUSPECT/ACTIVE 证据的决策时间；
- **first prompt time**：SUSPECT 首次输出的决策时间；
- **first ACTIVE time**：ACTIVE 首次输出的决策时间；
- **actual detection delay**：真值事件起点到首次实际输出（SUSPECT 用
  first prompt，ACTIVE 用 first ACTIVE）的时间；
- 所有时间不含推理/传输/渲染耗时，只能称“窗口结束时的理论可用时间”。

估计起点不能替代首次输出时间；不得把 estimate onset 回填成告警起点。

---

## 3. 每个决策的输入

每个尺度在每个 `decision_time` 获得：

- `quality_state`（Stage 1 v3）；
- `p_u2`（仅 `VALID`：校准后的 U2 概率）；
- `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 状态。

证据分类：

| 类别 | 条件 | SUSPECT | ACTIVE |
|---|---|---|---|
| `H` | `FLAT_WITH_HISTORY` | 是 | **是** |
| `L` | `FLAT_EVER_HISTORY` | 是 | **否** |
| `V` | `VALID` 且 `p_u2 >= suspect_p` | 是 | 看 active 阈值 |
| `A` | `VALID` 且 `p_u2 >= active_enter_p` | 是 | **是** |
| `U` | 任何 UNKNOWN 状态 | 无 | 无 |

`FLAT_EVER_HISTORY` 是低置信提示，**永远不能单独触发 ACTIVE**。

---

## 4. 状态机

状态：`WAITING`、`UNKNOWN`、`NORMAL`、`SUSPECT`、`ACTIVE`。

### 4.1 参数

每个尺度独立选择：

```
suspect_p              SUSPECT 概率阈值
active_enter_p         ACTIVE 进入阈值
active_exit_p          ACTIVE 退出阈值（< active_enter_p）
suspect_on_count       SUSPECT 进入所需连续证据决策数
suspect_off_count      SUSPECT 退出所需连续非证据决策数
active_on_count        ACTIVE 进入所需连续 active 证据决策数
active_off_count       ACTIVE 退出所需连续 active-exit 决策数
```

边界约束：`0 < active_exit_p < active_enter_p`、`suspect_p <= active_enter_p`、
`suspect_on_count <= active_on_count`。

### 4.2 进入 SUSPECT

连续 `suspect_on_count` 个 SUSPECT 证据（H/L/V）→ 输出 SUSPECT。

- `first_prompt_time = decision_time`（本次满足门槛的决策）；
- `estimated_onset = 本 episode 第一个 SUSPECT/ACTIVE 证据的 decision_time`；
- SUSPECT 事件从 `first_prompt_time` 开始，不早于首次输出。

### 4.3 进入 ACTIVE

连续 `active_on_count` 个 ACTIVE 证据（H/A，不含 L）→ 输出 ACTIVE。

- `first_active_time = decision_time`；
- 若尚未输出 SUSPECT，则同一决策先产生 SUSPECT；
- ACTIVE 事件从 `first_active_time` 开始。

### 4.4 退出 ACTIVE / SUSPECT

- ACTIVE：连续 `active_off_count` 个 **active-exit** 决策
  （`VALID` 且 `p_u2 < active_exit_p`）→ ACTIVE 结束于该 decision_time；
  `L` 不是 active-exit（保持/中性），`H` 是 active 证据；
- SUSPECT：连续 `suspect_off_count` 个无 SUSPECT 证据的决策 →
  SUSPECT 结束于该 decision_time；
- ACTIVE 结束时若 SUSPECT 仍成立，UI 回到 SUSPECT；
- ACTIVE 结束不自动结束 SUSPECT；SUSPECT 结束才结束本 episode。

### 4.5 UNKNOWN 规则

UNKNOWN **不当作正常，也不当作负证据**：

1. 在 UNKNOWN 的 decision_time，立即结束当前 ACTIVE/SUSPECT 输出
   （输出区间截至该 decision_time，不延伸到未知区间）；
2. 清空 SUSPECT/ACTIVE 连续计数、episode 记忆和低置信标记；
3. 状态置 `UNKNOWN`；不产生新告警；
4. 数据恢复后必须从 `NORMAL` 重新累计，**不得跨 UNKNOWN 延续证据**；
5. 会话/病例切换或数据中断超过一个决策步长时，同样按上述规则断开。

### 4.6 结束

`finish(case_end)` 在最后一个决策后的保持时段结束未闭合输出；不得把
未观测未来当作正常。

---

## 5. 输出对象

事件引擎输出两类区间，便于冻结匹配：

```
SUSPECT interval: [first_prompt_time, suspect_end)
ACTIVE  interval: [first_active_time, active_end)
```

episode 字段：

```
episode_id
scale                 U15/U30
estimated_onset
first_prompt_time
first_active_time
suspect_start / suspect_end
active_start / active_end
end_reason            NORMAL/UNKNOWN/FINISH
low_confidence        episode 是否只由 L 触发
```

同一 episode 可包含多个 ACTIVE 区间（ACTIVE 退出后 SUSPECT 继续，再次 ACTIVE）；
SUSPECT 区间连续覆盖整个 episode。低支撑/参考未知的评分裁剪沿用 Stage 2 v2。

ACTIVE 事件必须来自 H/A；任何仅由 L 触发的 episode 不得产生 ACTIVE。

---

## 6. 参数选择与验证

### 6.1 划分

- 只用 200 例开发集病例级外层 5 折；
- 每折内层 3 折用于模型权重/校准器/状态机参数选择；
- 外层测试折不参与模型训练、校准器拟合或状态机调参；
- Final Test 不参与任何选择或评价；
- **不得用在全部 200 例上训练的部署候选模型在验证病例上报告泛化性能**。

### 6.2 候选参数

每尺度使用预注册小网格；具体候选写入结果 JSON。选择目标：

1. ACTIVE event F1；
2. ACTIVE recall；
3. 较低可评分 FA/h；
4. 参数优先级（确定性 tie-break）。

在内层 OOF 上选择，外层只应用；最终主机演示参数在开发集 OOF 上选择并
单独标注“部署候选参数”，不用于报告泛化性能。

### 6.3 指标

与 Stage 2 使用相同病例、303 可靠事件、可评分区间和统计分母：

- ACTIVE/SUSPECT TP/FN/FP、recall/PPV/F1；
- 可评分 FA/h、告警持续占比；
- estimated onset、first prompt、first ACTIVE 分开报告；
- 实际检测延迟 = first prompt/ACTIVE - truth onset；
- UNKNOWN 决策比例、UNKNOWN 终止事件数；
- case-cluster bootstrap 95% CI；
- 同口径 Stage 2 原始代理对照。

若状态机没有改善，如实报告；不得用部署候选自评替代嵌套验证。

---

## 7. 上位机相对模式

### 7.1 默认模式

- 启动默认 `relative`；
- 未标定时不显示绝对 EtCO₂ 数值、mmHg 或标定浓度；
- CO₂ 波形只显示“相对 CO₂”形态；隐藏绝对刻度和单位；
- RR 可显示，但不与 EtCO₂ 绝对阈值组合报警；
- C 模式实时监控保持禁用；不能靠手动登记标定记录解锁。

### 7.2 告警来源

旧固件帧中的四分类只作为**旧分类显示/诊断参考**，不得进入任何用户告警链：

- 不写旧 `alarm_records`；
- 不调用旧 `_trigger_alert`、托盘通知、蜂鸣或状态栏告警；
- 不因旧类别触发趋势预警；
- 新告警只来自 Stage 3 事件引擎。

新事件写独立 `event_records`（或内存会话），字段至少包括：
`session_id, mode, scale, state, estimated_onset, first_prompt_time,
first_active_time, end_time, source="stage3_event_engine"`。
旧 `alarm_records` 保留兼容，但相对模式不写。

### 7.3 会话清空

以下操作必须清空窗口、质量历史、事件状态/计数/episode、旧 AI 缓存和告警限频：

- 模式切换；
- 串口断开/重连；
- 实时会话开始/结束；
- 病例切换；
- 回放跳转/重新开始。

不得把上一会话的 evidence/事件带入新会话。

---

## 8. 隔离回放工作台

### 8.1 数据

- 只使用开发集或合成数据；
- 不加载 Final Test 波形/标签/事件；
- 回放包标记 `source="dev"` 或 `source="synthetic"`；
- 多段拼接时标记 `stitched=true`，UI 明确显示“拼接”。

### 8.2 隔离

回放使用独立内存会话和独立 runtime：

- 不写生产数据库；
- 不触发生产通知/托盘/蜂鸣；
- 不控制真实设备/串口；
- 不改变实时会话的窗口、历史或事件状态。

### 8.3 时间与操作

- 以 `signal_time` 而不是墙钟驱动算法；
- 播放速度只改变每墙钟秒消费的信号样本数，不改变窗口/步长/状态机参数；
- 暂停保留当前位置；继续从当前位置；
- 重新开始/跳转必须重建 runtime，并从起点按因果顺序重放到目标时刻；
- 不得保留未来证据；
- 同一输入、配置、播放速度下，按 signal_time 产生一致状态/事件。

回放输出显示 ACTIVE/SUSPECT/UNKNOWN 和 Replay/拼接标识，但不进入生产告警。

---

## 9. 验收测试

1. UNKNOWN 不触发新告警，且结束当前输出/reset 计数；
2. `FLAT_EVER_HISTORY` 只能产生 SUSPECT，不能单独 ACTIVE；
3. 相对模式注入旧固件类别 0/1/2/3 均不触发旧告警链/DB 告警；
4. 回放前后生产 DB 行数、告警/通知不变；
5. 暂停、变速、重启、跳转、模式切换、断开重连、病例切换符合清空规则；
6. 时间戳、持续门槛、迟滞进入/退出、UNKNOWN 恢复有边界测试；
7. 现有上位机测试和训练仓库回归测试通过。

---

## 10. 已知限制

- 单尺度、无多尺度融合/U60/U90；
- UNKNOWN 后必须重新累计，可能降低连续性召回；
- 事件引擎参数仅在开发集选择，Final Test 未评价；
- 相对模式不提供绝对浓度；
- 协议无帧序号/时间戳，不能声称能检测所有真实丢帧；
- 回放是算法工作台，不是设备实时性能证明。
