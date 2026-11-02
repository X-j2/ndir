> **注：本文件为 Stage 1 规范 v1；已被 `docs/specs/stage1_quality_rule_spec_v2.md` 取代。**

# Stage 1 质量状态与 B0 规则基线规范 v1

> 日期：2026-10-31  
> 状态：Stage 1 实现，只使用 200 例开发集；Final Test 保持封存  
> 特征：`features/feature_spec_si_v1.json`  
> 实现：`features/si_features.py`、`features/quality_states.py`、`features/rule_baseline.py`

---

## 1. 输入与边界

- 输入只有未标定 CO₂ 波形（100Hz）；`rr`/`etco2` 只用于冻结标签与评估，不进入特征。
- 本阶段不部署 25Hz、不改固件、不做气体标定。
- 特征对 `x -> a*x+b (a>0)` 不变；负增益不静默翻转。
- 评估只用 `data_splits/cases_200.json`；脚本在加载时断言与 `final_test_seed42.json` 无交集，不读取 Final Test 波形。

## 2. 15 维 SI 特征

唯一规范是 `features/feature_spec_si_v1.json`，实现为 `extract_si_features()`：

1. `B=q10(x)`、`A=q90(x)-q10(x)`、`z=(x-B)/A`；
2. 15 维：`rel_mean, rel_std, rel_range, skewness, peak_interval_mean,
   peak_interval_std, peak_width_mean, plateau_ratio, peak_count,
   mean_prominence, peak_rate, rhythm_confidence, band_ratio, acf_peak,
   flat_run_frac`；`skewness` 是三阶标准矩
   `mean((z-mean(z))^3)/std(z)^3`，不是未标准化的 `mean(z^3)`；
3. 寻峰：5 点平滑；第一遍 `height_z=0.20, prominence_z=0.12,
   distance=1s, width=0.25–5s`；峰数<2 时仅在
   `acf_peak>0.30` 且 `noise_ratio<0.25` 时启用第二遍
   (`0.10/0.06`)；
4. `A=0` 或非有限：15 维向量仍返回有限值，z 类特征为 0，
   `flat_run_frac=1`；是否作为候选由质量层决定，特征层不丢弃它。

## 3. 质量状态（因果）

状态顺序：

```
GAP（失联/长缺口） -> SATURATED -> POLARITY_INVALID
-> FLAT_* -> NOISY / VALID
```

| 状态 | 条件 | 用途 |
|---|---|---|
| `GAP` | NaN 比例 >10% 或最长 NaN >2s | UNKNOWN，不报警 |
| `SATURATED` | 给定 rail 时，贴轨比例 >5% 或连续贴轨 >2s | UNKNOWN |
| `POLARITY_INVALID` | 原始波形中位数为负且负峰数多于正峰数（负增益）；大正偏置下的负增益无法从幅度辨认，需已知增益符号 | UNKNOWN，不翻转 |
| `FLAT_WITH_HISTORY` | 平坦/低幅，且近 300s 有已验证呼吸 | U2 候选（高置信） |
| `FLAT_EVER_HISTORY` | 近 300s 无，但更早已有已验证呼吸 | U2 候选（低置信，不单独报警） |
| `FLAT_NO_HISTORY` | 平坦，且过去从未有已验证呼吸 | **UNKNOWN** |
| `FLAT_HISTORY_UNKNOWN` | 历史不足 30s | **UNKNOWN** |
| `NOISY` | 非平坦且 `noise_ratio_hf > 0.25`（开发集暂定） | UNKNOWN |
| `VALID` | 非平坦、无上述问题 | 进入规则 |

**平坦/低幅判定（无绝对幅度门限）**：

```
flat = A_invalid(A=0/non-finite)
       OR flat_run_sec >= 10
       OR (history.A_ref is finite and A < 0.05 * history.A_ref)
```

`A=0` 本身不是弃权理由。平坦窗若近期有可信呼吸历史，必须成为 U2 候选；
若没有可信历史，则进入 UNKNOWN，不能标成 U0 或丢弃。不使用 `A<1 mmHg` 或
`A<1e-6` 作为分类门限，因此状态在未知正增益下保持不变。

**因果历史（Stage 1，质量筛选 + SI 寻峰）**：

- 15s bin、15s 步长；只使用 `bin.end_s <= window.start_s` 的已结束 bin；
- 每个 bin 用 `extract_si_features()` 的 SI 自适应寻峰；
- 只有同时满足以下条件的 bin 才是可信呼吸历史：
  1. 非 GAP（NaN 比例 ≤10%、最长缺口 ≤2s）；
  2. 非 SATURATED（若提供 rail）；
  3. 非 POLARITY_INVALID；
  4. `noise_ratio_hf <= 0.25`；
  5. SI `peak_count >= 2`。
- **NOISY/GAP/饱和/极性错误的 bin 不能贡献 `has_breathing` 或 `A_ref`**；
  `A_ref` 只取合格 bin 的 A 最大值（尺度相对，不设绝对下限）；
- 未来 bin 永不参与。`HistoryIndex` 用二分查找实现同一语义。
- 开发数据没有设备 rail，SATURATED 只能由合成削顶测试验证，不能在 VitalDB
  上报告设备饱和率。

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

U3 只作规则观察态，不进入 RF；`peak_rate<3` 且非平坦为 UNKNOWN，不硬映射。
`FLAT_EVER_HISTORY` 的预测置信度标为 `low`，报告必须与高置信分开解读。

## 5. 评估与结果

脚本：

```bash
python tools/run_stage1_rule_baseline.py --window-sec 15 --step-sec 7.5
python tools/run_stage1_rule_baseline.py --window-sec 30 --step-sec 15
```

输出：

- `report/stage1/rule_baseline_15s.json/.md`
- `report/stage1/rule_baseline_30s.json/.md`

主指标为四类 P/R/F1、Macro-F1、U2 二分类 P/R/F1、覆盖率、质量状态分布与
混淆矩阵。`UNKNOWN` 不计为任何类别；U2 recall 把 UNKNOWN 计为漏检。
本阶段不训练 RF、不校准、不融合、不接触 Final Test。

### 5.1 开发集结果（rev1，质量筛选 SI 历史）

| 尺度 | 有标签窗 | 覆盖 | Acc 全部/覆盖 | Macro-F1 | U2 P/R/F1 |
|---|---:|---:|---:|---:|---:|
| 15s/7.5s | 277,692 | 99.8% | 0.9021 / 0.9038 | 0.5100 | 0.454 / 0.670 / 0.541 |
| 30s/15s | 138,989 | 99.9% | 0.9385 / 0.9398 | 0.5845 | 0.345 / 0.743 / 0.471 |

15s U2 中 A=0/non-finite 1,301 窗：预测 U2 1,195、UNKNOWN 106；
30s 为 532 窗：预测 U2 493、UNKNOWN 39。低幅判断只用相对历史幅度，
不使用 `A<1` 或 `A<1e-6` 的分类门限。
