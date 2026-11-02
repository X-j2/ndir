# Stage 3 事件引擎、相对模式与隔离回放规范 v2

> 版本：stage3_event_engine_spec_v2  
> 日期：2026-11-01  
> 依据：Stage 2 rev1 报告、`stage2_model_eval_spec_v2`、`stage1_quality_rule_spec_v3`  
> 取代：`stage3_event_engine_spec_v1`  
> 范围：U15/U30 单尺度事件引擎；上位机相对模式；隔离回放；不改固件、不做气体标定  
> Final Test：继续封存；不加载波形/标签/事件/概率  
> 多尺度融合、U60/U90：留到 Stage 4

---

## 0. v2 修订摘要

v1 的外部审查发现五类实现缺口，v2 明确并收紧如下：

1. **严格连续计数**：ACTIVE 进入/退出、SUSPECT 进入/退出的“连续”计数，
   遇到任何不属于该计数的证据都必须归零；阈值中间区间（neutral）也打断
   连续进入和连续退出。
2. **缺失决策即中断**：两个观测决策的间隔超过一个 `step_sec` 时，必须在
   预期更新期限 `last_decision_time + step_sec` 结束旧输出、清空全部连续
   证据，再处理迟到的当前决策。离线评价与上位机溢出/中断采用同一规则。
3. **转换携带事件快照**：每条状态转换必须携带 `transition_seq`、
   `episode_id` 和转换发生后的完整 episode 快照；上位机按快照落库，不再
   在处理后的当前状态里回查已经清空的 episode。
4. **回放病例集校验**：不仅检查 `source` 文本，还必须校验开发集白名单、
   Final Test 黑名单、顶层 `case_ids`、每个分段的 `case_id` 及分段区间。
5. **累计信号时间调度**：回放推进按累计小数信号样本调度，禁止逐 tick
   取整；U15/U30 决策时间按精确 `window_sec + k * step_sec` 调度，禁止用
   固定 188 点步长造成 7.52 秒漂移。

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

### 2.1 决策节拍

每个尺度的 `step_sec` 是预期决策间隔：

- U15：`window_sec=15.0`，`step_sec=7.5`；
- U30：`window_sec=30.0`，`step_sec=15.0`。

离线评价使用窗口 `start` 加 `window_sec` 得到决策时间。上位机使用精确
`decision_time_k = window_sec + k * step_sec`，不得使用取整后的样本间隔
（例如 188/25 = 7.52 秒）作为状态机 `step_sec`。

---

## 3. 每个决策的输入

每个尺度在每个 `decision_time` 获得：

- `quality_state`（Stage 1 v3）；
- `p_u2`（仅 `VALID`：校准后的 U2 概率）；
- `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 状态。

证据分类：

| 类别 | 条件 | SUSPECT 证据 | ACTIVE 证据 | ACTIVE-EXIT |
|---|---|---|---|---|
| `H` | `FLAT_WITH_HISTORY` | 是 | **是** | 否 |
| `L` | `FLAT_EVER_HISTORY` | 是 | **否** | 否 |
| `V` | `VALID` 且 `suspect_p <= p_u2 < active_enter_p` | 是 | 否 | 视 `p_u2 < active_exit_p` |
| `A` | `VALID` 且 `p_u2 >= active_enter_p` | 是 | **是** | 否 |
| `N` | 其他 `VALID`（`p_u2 < suspect_p`） | 否 | 否 | 视 `p_u2 < active_exit_p` |
| `U` | 任何 UNKNOWN 状态 | 无 | 无 | 无 |

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
suspect_on_count       SUSPECT 进入所需连续 SUSPECT 证据决策数
suspect_off_count      SUSPECT 退出所需连续无 SUSPECT 证据决策数
active_on_count        ACTIVE 进入所需连续 ACTIVE 证据决策数
active_off_count       ACTIVE 退出所需连续 active-exit 决策数
```

边界约束：`0 < active_exit_p < active_enter_p`、`suspect_p <= active_enter_p`、
`suspect_on_count <= active_on_count`。

### 4.2 严格连续计数的定义

“连续 N 次”必须按类别逐次判定，不允许跳过中间证据：

- `active_run`：当前决策是 ACTIVE 证据（H/A）时加 1，**否则归零**；
- `active_clear`：当前决策是 active-exit（`VALID` 且
  `p_u2 < active_exit_p`）时加 1，**否则归零**；
- `suspect_run`：当前决策是 SUSPECT 证据（H/L/V/A）时加 1，否则归零；
- `suspect_clear`：当前决策不是 SUSPECT 证据时加 1，否则归零。

特别地，阈值中间区间（`active_exit_p <= p_u2 < active_enter_p`）和
`FLAT_EVER_HISTORY` 对 ACTIVE 是 neutral：

- neutral **不能**累计进 `active_run`；
- neutral **不能**累计进 `active_clear`；
- neutral 必须同时把 `active_run` 和 `active_clear` 归零。

示例：`active_on_count=2` 时，`0.9 → 0.2（neutral）→ 0.9` 不得进入
ACTIVE；必须连续两个 ACTIVE 证据。退出同理：`exit → neutral → exit`
不得累计两次 active-exit。

### 4.3 进入 SUSPECT

连续 `suspect_on_count` 个 SUSPECT 证据（H/L/V/A）→ 输出 SUSPECT。

- `first_prompt_time = decision_time`（本次满足门槛的决策）；
- `estimated_onset = 本 episode 第一个 SUSPECT/ACTIVE 证据的 decision_time`；
- SUSPECT 事件从 `first_prompt_time` 开始，不早于首次输出。

### 4.4 进入 ACTIVE

连续 `active_on_count` 个 ACTIVE 证据（H/A，不含 L）→ 输出 ACTIVE。

- `first_active_time = decision_time`；
- 若尚未输出 SUSPECT，则同一决策先产生 SUSPECT；
- ACTIVE 事件从 `first_active_time` 开始。

### 4.5 退出 ACTIVE / SUSPECT

- ACTIVE：连续 `active_off_count` 个 **active-exit** 决策
  （`VALID` 且 `p_u2 < active_exit_p`）→ ACTIVE 结束于该 decision_time；
- SUSPECT：连续 `suspect_off_count` 个无 SUSPECT 证据的决策 →
  SUSPECT 结束于该 decision_time；
- ACTIVE 结束时若 SUSPECT 仍成立，UI 回到 SUSPECT；
- ACTIVE 结束不自动结束 SUSPECT；SUSPECT 结束才结束本 episode。

### 4.6 UNKNOWN 规则

UNKNOWN **不当作正常，也不当作负证据**：

1. 在 UNKNOWN 的 decision_time，立即结束当前 ACTIVE/SUSPECT 输出
   （输出区间截至该 decision_time，不延伸到未知区间）；
2. 清空 SUSPECT/ACTIVE 连续计数、episode 记忆和低置信标记；
3. 状态置 `UNKNOWN`；不产生新告警；
4. 数据恢复后必须从 `NORMAL` 重新累计，**不得跨 UNKNOWN 延续证据**；
5. 会话/病例切换或数据中断超过一个决策步长时，同样按上述规则断开。

### 4.7 缺失决策与数据中断

引擎记录最近一次已处理决策时间 `last_decision_time`。处理新决策
`decision_time` 前先检查间隔：

```
if decision_time - last_decision_time > step_sec + tol:
    deadline = last_decision_time + step_sec
    按 UNKNOWN 规则在 deadline 结束 ACTIVE/SUSPECT
    清空全部连续计数/episode 记忆
    state = UNKNOWN
    记录一次 gap_interruption
    然后才处理当前迟到决策（从 NORMAL 重新累计）
```

- `tol` 仅用于浮点误差，取 `max(1e-6, step_sec * 1e-6)`；
- 间隔恰好等于 `step_sec` 是正常决策，不触发中断；
- 中断本身不是一次观测决策，不计入 `total_decisions`；
- 当前迟到决策是新观测，正常计入 `total_decisions`，但不得继承中断前计数；
- 输出区间截止到 `deadline`，不是迟到决策时间；
- 离线评价、上位机采样队列溢出/数据中断使用同一 `interrupt()` 语义；
- 上位机不得用“直接 reset”静默丢弃已打开的 ACTIVE/SUSPECT；必须先按
  预期更新期限结束输出并落库，再清空会话。

### 4.8 结束

`finish(case_end)` 在最后一个决策后的保持时段结束未闭合输出；不得把
未观测未来当作正常。

---

## 5. 输出对象与转换快照

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
active_intervals      每段 {start, end}
end_time
end_reason            NORMAL/UNKNOWN/FINISH
low_confidence        episode 是否只由 L 触发
```

同一 episode 可包含多个 ACTIVE 区间（ACTIVE 退出后 SUSPECT 继续，再次 ACTIVE）；
SUSPECT 区间连续覆盖整个 episode。低支撑/参考未知的评分裁剪沿用 Stage 2 v2。

ACTIVE 事件必须来自 H/A；任何仅由 L 触发的 episode 不得产生 ACTIVE。

### 5.1 每条转换的字段

每次状态转换必须产生一条记录：

```
transition_seq         引擎内单调递增序号
kind                   SUSPECT_START/ACTIVE_START/ACTIVE_END/SUSPECT_END/UNKNOWN
time                   转换发生的 decision_time 或中断 deadline
scale                  U15/U30
state                  转换后的状态
episode_id             episode 标识；无 episode 时为 None
episode                转换发生后的完整 episode 深拷贝快照
reason                 NORMAL/UNKNOWN/FINISH/MISSING_DECISION/DATA_INTERRUPTION
```

快照必须在本次转换的状态变更（如 `suspect_end`、active interval `end`、
`end_time`、`end_reason`）完成之后采集。

### 5.2 上位机落库

- 上位机不得在处理转换后回查 `current_event()` 来补字段；
- 必须直接使用 `transition.episode` 快照；
- `event_records` 至少保存：
  `episode_id, transition_seq, session_id, mode, scale, state, label,
  estimated_onset, first_prompt_time, first_active_time, end_time, source`；
- 旧库通过 `ALTER TABLE` 增量补列，不重建、不丢历史；
- SUSPECT_START/ACTIVE_START 的 `end_time=None`；
- ACTIVE_END 的 `end_time` 取该次 ACTIVE 区间的结束时间；
- SUSPECT_END 的 `end_time` 取 episode 的 `end_time`。

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
- **缺失决策中断数 `gap_interruptions`、显式中断数 `interruptions`**；
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

新事件写独立 `event_records`（或内存会话），字段见 5.2。旧
`alarm_records` 保留兼容，但相对模式不写。

### 7.3 会话清空

以下操作必须清空窗口、质量历史、事件状态/计数/episode、旧 AI 缓存和告警限频：

- 模式切换；
- 串口断开/重连；
- 实时会话开始/结束；
- 病例切换；
- 回放跳转/重新开始。

清空前若存在打开的 ACTIVE/SUSPECT，必须先调用与 4.7 相同的中断语义，
在预期更新期限结束输出并按快照落库，再清空。不得把上一会话的
evidence/事件带入新会话。

---

## 8. 隔离回放工作台

### 8.1 数据与病例集校验

- 只使用开发集或合成数据；
- 不加载 Final Test 波形/标签/事件；
- 回放包标记 `source="dev"` 或 `source="synthetic"`；
- 多段拼接时标记 `stitched=true`，UI 明确显示“拼接”。

加载时必须执行以下校验：

1. 读取训练仓库 `data_splits/cases_200.json` 开发集白名单与
   `data_splits/final_test_seed42.json` Final Test 黑名单；两者必须互斥，
   否则拒绝加载；
2. `source="dev"`：
   - 顶层 `case_ids` 非空；
   - 顶层每个 case id 都在开发集白名单；
   - `segments` 非空，每个 segment 必须有 `case_id`，且属于顶层
     `case_ids` 和开发集白名单；
   - 每个 segment 的 `start_s < end_s`，`start_s >= 0`；
3. `source="synthetic"`：不得声明任何真实病例号（顶层或分段），避免用
   synthetic 标签绕过开发集/Final Test 校验；
4. 任何来源的顶层 case id、分段 case id、`truth_events.case_id` 命中
   Final Test 黑名单，立即拒绝；
5. 文件名命中 Final Test 文件同样拒绝（保留纵深防御）。

### 8.2 隔离

回放使用独立内存会话和独立 runtime：

- 不写生产数据库；
- 不触发生产通知/托盘/蜂鸣；
- 不控制真实设备/串口；
- 不改变实时会话的窗口、历史或事件状态。

### 8.3 时间与操作

- 以 `signal_time` 而不是墙钟驱动算法；
- 回放会话维护累计小数信号样本 `signal_accum`：
  - 每个 tick：`signal_accum += wall_dt * speed * sample_rate_hz`；
  - 消费目标：`target = floor(signal_accum)`，不得对每个 tick 的增量单独
    取整后再相加；
  - 暂停不推进 `signal_accum`；继续从当前位置继续累计；
  - 跳转/重新开始重建 runtime，并把 `signal_accum` 设为跳转后的整数样本；
- 播放速度只改变每墙钟秒消费的信号样本数，不改变窗口/步长/状态机参数；
- 同一输入、配置、播放速度下，按 signal_time 产生一致状态/事件；
- 回放输出显示 ACTIVE/SUSPECT/UNKNOWN 和 Replay/拼接标识，但不进入生产告警。

### 8.4 U15/U30 运行时调度

- `decision_time_k = window_sec + k * step_sec`，精确浮点表达，不用累加
  取整后的 `step_samples`；
- 窗口样本长度 `round(window_sec * 25)`；
- 窗口结束样本 `floor(decision_time_k * 25 + eps)`，保证不使用未来样本；
- U15 的 7.5 秒步长在 25 Hz 下对应交替 188/187 个样本；状态机
  `step_sec` 必须保持 7.5，窗口开始时间用 `decision_time_k - window_sec`；
- 任何实现不得把 188 点（7.52 秒）当作状态机步长。

---

## 9. 验收测试

1. UNKNOWN 不触发新告警，且结束当前输出/reset 计数；
2. `FLAT_EVER_HISTORY` 只能产生 SUSPECT，不能单独 ACTIVE；
3. **ACTIVE 连续进入**：`active_on_count=2`，`0.9 → 0.2 → 0.9` 不进入；
4. **ACTIVE 连续退出**：`active_off_count=2`，`exit → neutral → exit` 不退出；
5. **缺失决策中断**：15 秒一次决策，下一次 150 秒才到，旧输出在 30 秒结束，
   新决策不继承计数；
6. **转换快照**：SUSPECT/ACTIVE 开始与结束转换均携带 `episode_id` 和完整
   快照；结束快照含正确 `end_time`；
7. 相对模式注入旧固件类别 0/1/2/3 均不触发旧告警链/DB 告警；
8. 上位机按快照落库：15 秒首次 SUSPECT/ACTIVE、22.5 秒结束，落库起点仍为
   15 秒，首次提示/ACTIVE 时间不为 None；
9. 回放拒绝 `source="dev"` 但 case ids 含 Final Test 的包；拒绝分段 case id
   不在开发集白名单的包；拒绝 synthetic 包声明真实病例号；
10. 回放累计信号时间：1 秒 1× 与 0.5 秒 2× 消费相同样本；暂停/继续/跳转
    与真实 runtime 的 states/events 一致；
11. U15 决策时间间隔为 7.5 秒，不是 7.52 秒；
12. 回放前后生产 DB 行数、告警/通知不变；
13. 现有上位机测试和训练仓库回归测试通过。

---

## 10. 已知限制

- 单尺度、无多尺度融合/U60/U90；
- UNKNOWN/缺失决策后必须重新累计，可能降低连续性召回；
- 事件引擎参数仅在开发集选择，Final Test 未评价；
- 相对模式不提供绝对浓度；
- 协议无帧序号/时间戳，不能声称能检测所有真实丢帧；
- 回放是算法工作台，不是设备实时性能证明。
