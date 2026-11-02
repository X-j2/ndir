# CF-REM 标签规范 v1（Stage 0 冻结）

> 版本：labels_spec_v1（内容修订 rev3）  
> 日期：2026-10-29  
> 状态：Stage 0 冻结；后续修改必须升版本并重算全部统计  
> 依赖：`event_spec_v1.md`、`data_splits/*.json`  
> 实现：`tools/stage0_common.py::classify_window`  
> 复算：`python tools/recompute_stage0_stats.py`

---

## 1. 目的与边界

本规范定义 **U 模式**（免标定输入）的训练/评估标签。C 模式（旧四类、已标定研究数据）不在本规范内。

**主标签的正式名称：**

> **U2：参考 EtCO₂ 低值/呼气 CO₂ 缺失样事件**  
> 英文：reference low-EtCO₂ / CO₂-disappearance-like event

**禁止**在没有独立呼吸努力、气流或人工标注证据时，把 U2 写成“临床确诊屏息/呼吸暂停”。论文、UI、专利中“屏息”只能作为候选解释，并必须同时写出参考轨道与阈值。

---

## 2. 标签来源与输入分离

- **标签来源**：VitalDB 监护仪参考轨道 `etco2`（训练时可用）。
- **模型输入**：只使用未标定 CO₂ 波形计算的 SI 特征；`etco2`/`rr` 不得作为模型输入。
- **为什么不是标签泄漏**：标签与输入来源不同。用 EtCO₂ 生成弱标签，但推理时不提供 EtCO₂，不构成“用标签生成特征”。是否能从波形预测该标签，是 Stage 1 的实验问题。
- **Fallback**：若开发集上 SI 特征不能预测 EtCO₂<10 弱标签，论文 pivot 到“节律事件的可观测性”，不要硬做 U90 屏息主模型。

---

## 3. 事件时间线（前置依赖）

U2 标签依赖 `event_spec_v1.md` 的 U2 事件时间线：

```
轨道: etco2
1s 网格；<=0 视为缺失
前向填充上限: 10s；超过记 unknown
阈值: etco2 < 10
合并: 间隔 <=10s，且间隔内没有 unknown
最短持续: 10s
顺序: 先合并，后丢弃短段
```

U2 事件时间线是**每病例**独立构建的；窗口标签只使用该病例的事件时间线，不跨病例。**unknown 不允许被合并跨越**；详细规则见 `event_spec_v1.md`。

---

## 4. 窗口标签判定顺序（唯一顺序）

对每个窗口（任意尺度），计算其与 U2 事件时间线的重叠比例：

```
ov = overlap_seconds(window, U2_events) / window_seconds
```

按以下顺序判定，先命中先返回：

| 优先级 | 条件 | 标签 | 是否进 RF |
|---:|---|---|---|
| 1 | 可靠事件重叠 `ov_rel >= 0.50` | **U2** | 是 |
| 2 | 低支撑事件重叠 `ov_low >= 0.50` | **ignore_low_support** | 否 |
| 3 | 可靠事件边界 `0.10 <= ov_rel < 0.50` | **ignore** | 否 |
| 4 | 低支撑事件边界 `0.10 <= ov_low < 0.50` | **ignore_low_support** | 否 |
| 5 | RR 中位数缺失，或 RR 有效点 < 2 | **unlabeled** | 否 |
| 6 | `RR < 3` | **ambiguous** | 否 |
| 7 | `RR >= 18` | **U1 急促** | 是 |
| 8 | `7 <= RR < 18` | **U0 节律正常** | 是 |
| 9 | `3 <= RR < 7` | **U3 慢呼吸/观察态** | 否（规则观察态） |
| 10 | 其他（RR 非有限/越界） | **other** | 否 |

**可靠 vs 低支撑**：可靠事件 = **与事件 1s 网格实际关联**的 `EtCO2<10` 原始读数运行段 ≥2；低支撑 = <2（本次为单读）。关联规则见 `event_spec_v1.md` §2.4：某条原始读数只要是事件内至少一个低值网格点的 ffill 来源，就计入；它可以位于事件整数秒起点之前（仍在 cap 内）。低支撑窗只进 `ignore_low_support`，既不能当可靠 U2，也不能改作负例。

**关键决定（相对 v1.1 的变化）：**

- **删除**附录 A.1 中额外的 `etco2_med < 10` 门槛。U2 只由“事件重叠 ≥50%”决定；否则同一窗口在正文与附录会得到不同标签（Stage 0 审查 P0-2）。
- **删除** v1.1 中“3≤RR≤6 / 7≤RR≤17”的整数写法，改为半开区间，避免连续 RR 在 6–7、17–18 之间出现空档。
- **U3 的“RR 变异性高”暂不实现**：当前没有冻结的变异性定义；U3 只按 `3<=RR<7` 进入规则观察态。若后续要加，必须单独定义 `rr_cv` 阈值并在开发集验证，不能悄悄混入。
- **ambiguous** 只表示“RR<3 但无 EtCO₂ 事件重叠”，是标签噪声层，不训练、不评估主指标。
- **unlabeled** 包括 RR 缺失/参考点不足；不能默认成 U0。
- **other** 目前为 0；保留给越界/非有限参考值。

---

## 5. 参考值有效性

- RR 中位数：窗口内 `rr > 0` 且非 NaN 的有效点中位数；有效点 < 2 → `unlabeled`。
- EtCO₂ 中位数：窗口内 `etco2 > 0` 且非 NaN 的有效点中位数；仅用于诊断/审计，不参与 U2 窗标签（U2 由事件重叠决定）。
- 事件时间线的前向填充上限为 10s；超过上限的区间记 unknown，不用于事件构造。
- 病例边界：窗口起点/终点按 100Hz 样本换算到 1s 网格；事件时间线不跨病例。

---

## 6. 标签语义与 UI 措辞

| 标签 | 正式措辞 | 禁止措辞 |
|---|---|---|
| U0 | 节律正常，未评估气体交换 | 健康、正常气体交换、正常 CO₂ |
| U1 | 参考 RR 急促 | 临床确诊呼吸急促 |
| U2 | 参考 EtCO₂ 低值/呼气 CO₂ 缺失样事件 | 临床确诊屏息/呼吸暂停 |
| U3 | 慢呼吸/观察态（规则） | 正常、异常（未定义） |
| ambiguous | 参考 RR<3 但无 EtCO₂ 事件重叠 | 屏息 |
| ignore | 可靠事件边界窗口 | 正常/异常 |
| ignore_low_support | 低支撑事件（≤1 条实际读数）窗 | 可靠屏息/正常/异常 |
| unlabeled | 参考不足/不可判断 | 正常 |

**U0 的重要限制**：90s 中位数口径下有约 27.8k 个 U0 窗同时是旧类 3（EtCO₂ 在 35–45 外）。U0 只能表示“节律正常”，不能表示“气体交换正常”。

---

## 7. 数据划分与冻结

- 病例总表：`data_splits/cases_250.json`（250 例，按数值 case id 排序）。
- 开发集：`data_splits/cases_200.json`（200 例）。
- 锁定 Final Test：`data_splits/final_test_seed42.json`（50 例；事件分层配额 + 最小事件修复）。
- 划分只使用参考轨道事件数，不使用任何模型结果。
- **Final Test 不参与训练、阈值选择、特征选择、校准、融合调参。**
- Stage 0 rev3 的 Final Test 事件数：**U2 可靠 73 事件/35 例；U2 低支撑 15/13；RR<3 18/15**。v1.1 的 70/37 与 27/20 未能复现，不得当作已复现结果引用；rev2 的 58/32 也不再使用。
- 全量 90s U2 窗：386 窗/99 例；15s/30s/60s U2 窗：3,139/1,507/649。低支撑窗另计 `ignore_low_support`（90s：37 窗）。

---

## 8. 可执行定义（伪代码）

```python
ov_rel = overlap_fraction(window, reliable_u2_events)
ov_low = overlap_fraction(window, low_support_u2_events)
if ov_rel >= 0.50:
    return "U2"
if ov_low >= 0.50:
    return "ignore_low_support"
if ov_rel >= 0.10:
    return "ignore"
if ov_low >= 0.10:
    return "ignore_low_support"
if rr_median is None or rr_count < 2:
    return "unlabeled"
if rr_median < 3:
    return "ambiguous"
if rr_median >= 18:
    return "U1"
if rr_median >= 7:
    return "U0"
if rr_median >= 3:
    return "U3"
return "other"
```

实现：`tools/stage0_common.py::classify_window`。

---

## 9. 验收测试

1. 可靠事件覆盖 60% + RR=12 → U2（不是 U0）。
2. 低支撑事件覆盖 60% + RR=12 → ignore_low_support，不是 U2，也不是 U0。
3. 可靠事件覆盖 30% + RR=12 → ignore。
4. 低支撑事件覆盖 30% + RR=12 → ignore_low_support。
5. 事件覆盖 0% + RR=12 → U0。
6. 事件覆盖 0% + RR=2 → ambiguous。
7. 事件覆盖 0% + RR 缺失 → unlabeled。
8. 事件覆盖 0% + RR=4 → U3。
9. 事件覆盖 0% + RR=20 → U1。
10. 删除 `etco2_med` 门槛后，同一窗口在正文/附录得到同一标签。
11. 同一病例的事件时间线不跨病例泄漏。
12. Final Test 的 case id 不出现在开发集；Final Test 只用可靠 U2 事件做分层。
13. 0.5 s 单读（来源在事件整数秒起点前）→ 事件 `[1,11)`、关联 1 读、`ignore_low_support`，不能算 U2。
14. 0.5 s + 10.5 s 两读 → 事件 `[1,21)`、关联 2 读、U2。
15. 不传事件表的 `iter_windows()` 必须拆分低支撑，不能把单读事件当 U2。

---

## 10. 已知限制

- VitalDB 是术中机械通气数据，参考 RR/EtCO₂ 由呼吸机/监护仪设定；`EtCO₂<10` 可能来自呼吸暂停、漏气、脱管、采样管故障或低幅信号。没有独立气流/呼吸努力标签，U2 只能叫“参考 EtCO₂ 低值/CO₂ 缺失样事件”。
- 术前/术中参考轨道密度低（RR 约每 2s 一个值，EtCO₂ 约 8–14% 样本）；事件时间线对前向填充上限敏感，必须报告 5/10/20s 敏感性。
- 旧类 2 的 169/132（生产过滤）或 209/149（未过滤）是 **90s 窗段**，不是事件；不得与 U2 可靠 376/178、低支撑 75/59、RR<3 98/72 混用。
