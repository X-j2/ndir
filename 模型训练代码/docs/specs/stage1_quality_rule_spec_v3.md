# Stage 1 质量状态与 B0 规则基线规范 v3

> 日期：2026-11-01  
> 状态：Stage 1 rev3；只使用 200 例开发集；Final Test 保持封存  
> 相对 v2：不再把 `q90-q10=0` 直接等同于完全平坦；区分真正常数窗与分位幅度退化但仍含脉冲/变化的窗，后者新增 `QUANTILE_ARTIFACT` 判定。  
> 特征：`features/feature_spec_si_v2.json`（15 维向量未变）  
> 实现：`features/si_features.py`、`features/quality_states.py`、`features/rule_baseline.py`

---

## 1. 输入与边界

- 输入只有未标定 CO₂ 波形（100Hz）；`rr`/`etco2` 只用于冻结标签与评估，不进入特征。
- 本阶段不部署 25Hz、不改固件、不做气体标定。
- 特征与质量判定对 `x -> a*x+b (a>0)` 不变；负增益不静默翻转。
- 评估只用 `data_splits/cases_200.json`；脚本在加载时断言与 `final_test_seed42.json` 无交集，不读取 Final Test 波形。

## 2. 15 维 SI 特征（不变）

唯一规范是 `features/feature_spec_si_v2.json`，实现为 `extract_si_features()`：

1. `B=q10(x)`、`A=q90(x)-q10(x)`、`z=(x-B)/A`；
2. 15 维：`rel_mean, rel_std, rel_range, skewness, peak_interval_mean,
   peak_interval_std, peak_width_mean, plateau_ratio, peak_count,
   mean_prominence, peak_rate, rhythm_confidence, band_ratio, acf_peak,
   flat_run_frac`；`skewness` 是标准三阶矩
   `mean((z-mean(z))^3)/std(z)^3`；
3. 寻峰与 band/ACF 参数保持 v2 不变；
4. `A=0` 或非有限时，15 维向量仍返回有限中性值，`flat_run_frac=1`。

### 2.1 新增诊断：分位幅度退化

`A=q90-q10` 为 0 有两种完全不同的来源：

| 诊断 | 条件 | 语义 |
|---|---|---|
| `constant_flat=True` | `A_invalid` 且有限样本数 >0 且 `finite_span = max-min = 0` | 真正常数窗；在可信历史下保留 U2 候选路径 |
| `quantile_artifact=True` | `A_invalid` 且 `finite_span > 0` | 分位幅度退化但仍含稀疏脉冲/阶跃/其他变化；属伪迹，不得走平坦候选 |

`finite_span` 在补短缺口后的有限样本上计算。该判据使用精确 0 比较，不引入绝对幅度门限，因此在正仿射变换下保持。典型反例：15 秒内 >80% 样本为常数基线、仅含 3 个 0.2 秒脉冲时，`q10=q90=基线`、`A=0`，但 `finite_span>0`，应判 `QUANTILE_ARTIFACT`，不是完全平坦。

## 3. 质量状态（因果）

状态顺序：

```
GAP（失联/长缺口） -> SATURATED -> POLARITY_INVALID
-> QUANTILE_ARTIFACT -> NOISY（非零幅度） -> FLAT_* / VALID
```

| 状态 | 条件 | 用途 |
|---|---|---|
| `GAP` | NaN 比例 >10% 或最长 NaN >2s | UNKNOWN，不报警 |
| `SATURATED` | 给定 rail 时，贴轨比例 >5% 或连续贴轨 >2s | UNKNOWN |
| `POLARITY_INVALID` | 原始波形中位数为负且负峰数多于正峰数（负增益）；大正偏置下的负增益无法从幅度辨认，需已知增益符号 | UNKNOWN，不翻转 |
| `QUANTILE_ARTIFACT` | `A_invalid` 且 `finite_span>0`（稀疏脉冲/阶跃/其他变化） | **UNKNOWN**；不得成为 U2 候选 |
| `FLAT_WITH_HISTORY` | 真正常数/平坦/低幅，且近 300s 有已验证呼吸 | U2 候选（高置信） |
| `FLAT_EVER_HISTORY` | 近 300s 无，但更早已有已验证呼吸 | U2 候选（低置信，不单独报警） |
| `FLAT_NO_HISTORY` | 真正常数/平坦，且过去从未有已验证呼吸 | **UNKNOWN** |
| `FLAT_HISTORY_UNKNOWN` | 历史不足 30s | **UNKNOWN** |
| `NOISY` | 非 A=0 且 `noise_ratio_hf > 0.25`（开发集暂定）；低幅非零噪声也走此分支 | UNKNOWN |
| `VALID` | 非平坦、无上述问题 | 进入规则 |

**平坦/低幅判定（无绝对幅度门限）**：

```
flat = A_invalid(A=0/non-finite)
       OR flat_run_sec >= 10
       OR (history.A_ref is finite and A < 0.05 * history.A_ref)
```

其中 `A_invalid` **只有**在 `constant_flat=True` 或真实低幅/平台条件成立时才可进入
`FLAT_*`；`quantile_artifact=True` 已在平坦分支之前被判为 UNKNOWN。
不使用 `A<1 mmHg` 或 `A<1e-6` 作为分类门限。

**因果历史（Stage 1，质量筛选 + SI 寻峰）**：

- 15s bin、15s 步长；只使用 `bin.end_s <= window.start_s` 的已结束 bin；
- 每个 bin 用 `extract_si_features()` 的 SI 自适应寻峰；
- 只有同时满足以下条件的 bin 才是可信呼吸历史：
  1. 非 GAP（NaN 比例 ≤10%、最长缺口 ≤2s）；
  2. 非 SATURATED（若提供 rail）；
  3. 非 POLARITY_INVALID；
  4. 非 QUANTILE_ARTIFACT；
  5. `noise_ratio_hf <= 0.25`；
  6. SI `peak_count >= 2`。
- **GAP/饱和/极性错误/分位退化伪迹/NOISY 的 bin 不能贡献 `has_breathing` 或 `A_ref`**；
- **`A_ref` 只取“质量合格且 SI 峰≥2”的 bin 的 A 最大值**（尺度相对，不设绝对下限）；
- 未来 bin 永不参与；`HistoryIndex` 用二分查找实现同一语义。
- 开发数据没有设备 rail，SATURATED 只能由合成削顶测试验证。

## 4. B0 规则

```
if state in UNKNOWN_STATES:                 UNKNOWN
elif state in {FLAT_WITH_HISTORY,
               FLAT_EVER_HISTORY}:          U2
elif state == VALID:
    peak_rate >= 18:                        U1
    7 <= peak_rate < 18:                    U0
    3 <= peak_rate < 7:                     U3
    else:                                   UNKNOWN
else:                                       UNKNOWN
```

`UNKNOWN_STATES` 现在包含 `QUANTILE_ARTIFACT`。`FLAT_EVER_HISTORY` 的预测置信度
标为 `low`，报告必须与高置信分开解读。

## 5. 评估与结果

```bash
python tools/run_stage1_rule_baseline.py --window-sec 15 --step-sec 7.5
python tools/run_stage1_rule_baseline.py --window-sec 30 --step-sec 15
```

输出：

- `report/stage1/rule_baseline_15s.json/.md`
- `report/stage1/rule_baseline_30s.json/.md`

主指标为四类 P/R/F1、Macro-F1、U2 二分类 P/R/F1、覆盖率、质量状态分布与
混淆矩阵。`UNKNOWN` 不计为任何类别；U2 recall 把 UNKNOWN 计为漏检。

### 5.1 开发集结果（rev3，分位退化伪迹单独判 UNKNOWN）

| 尺度 | 有标签窗 | 覆盖 | Acc 全部/覆盖 | Macro-F1 | U2 P/R/F1 |
|---|---:|---:|---:|---:|---:|
| 15s/7.5s | 277,692 | 99.7% | 0.9015 / 0.9041 | 0.5045 | 0.448 / 0.608 / 0.516 |
| 30s/15s | 138,989 | 99.7% | 0.9377 / 0.9402 | 0.5751 | 0.326 / 0.649 / 0.434 |

分位退化伪迹与真正常数窗：

| 尺度 | 伪迹总数（按真值） | U2 伪迹 | U2 真正常数 | U2 A=0 总数 |
|---|---:|---:|---:|---:|
| 15s/7.5s | 288（U0 53 / U1 20 / U2 158 / U3 57）| 158 | 1,143 | 1,301 |
| 30s/15s | 201（U0 36 / U1 6 / U2 116 / U3 43）| 116 | 416 | 532 |

15s U2 的 1,301 个 A=0 窗中，预测 U2 1,050、UNKNOWN 251；
30s 的 532 个 A=0 窗中，预测 U2 383、UNKNOWN 149。
真正常数窗仍按可信历史进入候选；分位退化伪迹一律 UNKNOWN，不再被误当作
“完全平坦”的 U2 候选。

### 5.2 与 rev2 的差异

rev2 将全部 `A=0` 窗视为同一平坦路径，导致含稀疏脉冲的分位退化窗在有历史时
被预测为 U2。rev3 将其改判 UNKNOWN：

- 15s U2 recall 0.667 → 0.608，U2 F1 0.541 → 0.516；
- 30s U2 recall 0.742 → 0.649，U2 F1 0.471 → 0.434；
- 覆盖率 99.8%/99.9% → 99.7%/99.7%。

这是“宁可弃权、不把伪迹当呼吸消失”的保守代价，不应表述为模型性能提升。
真正常数窗的候选路径仍保留：有可信历史 → U2，无历史 → UNKNOWN。
