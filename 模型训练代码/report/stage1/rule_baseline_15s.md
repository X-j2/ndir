# Stage 1 B0 规则基线（15s）

- 规则版本：`b0_rule_si_v1`；SI 规范：`si_feature_spec_v2`；质量规范：`stage1_quality_rule_spec_v3`
- 窗口：15s / 步长 7.5s；开发集 200 例（Final Test 未加载）
- 有标签窗：277692；覆盖：276899 (99.7%)；UNKNOWN：793
- Accuracy（全部/覆盖）：0.9015 / 0.9041；Macro-F1：0.5045

## 1. 四类结果

| 类别 | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| U0 | 241984 | 0.916 | 0.985 | 0.949 |
| U1 | 30784 | 0.830 | 0.338 | 0.480 |
| U2 | 2469 | 0.448 | 0.608 | 0.516 |
| U3 | 2455 | 0.131 | 0.051 | 0.073 |

## 2. U2 二分类（UNKNOWN 不当作正例）

- U2 support：2469；precision 0.448，recall 0.608，F1 0.516
- U2 预测分布：{'U0': 391, 'U1': 148, 'U2': 1501, 'U3': 82, 'UNKNOWN': 347}
- U2 质量状态：{'VALID': 702, 'FLAT_WITH_HISTORY': 900, 'FLAT_EVER_HISTORY': 601, 'FLAT_NO_HISTORY': 88, 'FLAT_HISTORY_UNKNOWN': 20, 'GAP': 0, 'SATURATED': 0, 'POLARITY_INVALID': 0, 'QUANTILE_ARTIFACT': 158, 'NOISY': 0}
- U2 中 A=0/non-finite（尺度无关 invalid）：1301；其中真正常数窗 1143、分位退化伪迹窗 158；预测 U2 1050，UNKNOWN 251
- `A=0` 只表示 q90-q10=0，不等于常数窗。有限样本完全相等才是真正常数窗并保留候选路径；仍含稀疏脉冲/变化的窗为 `QUANTILE_ARTIFACT`/UNKNOWN。
- 低幅判断按 `A < 0.05 * 因果历史 A_ref`，不再使用绝对 A 门限；U2 的 `FLAT_*` 状态分布见上表。

## 3. 分位退化伪迹与常数窗

- 全部有标签窗中，分位退化伪迹：288；按真值 {'U0': 53, 'U3': 57, 'U2': 158, 'U1': 20}
- 全部有标签窗中，真正常数窗：1740；按真值 {'U2': 1143, 'U3': 303, 'U0': 225, 'U1': 69}
- 分位退化伪迹先于噪声/平坦判定，进入 UNKNOWN，不能成为 U2 候选。

## 4. 平坦候选与 UNKNOWN

- 平坦候选窗：3352；按真值 {'U2': 1501, 'U0': 939, 'U1': 296, 'U3': 616}
- 平坦候选中预测 U2：3352；UNKNOWN：0
- `FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / GAP / 饱和 / 极性错误 / 分位退化伪迹 / 噪声均进入 UNKNOWN，不被当作 U0。

## 5. 质量状态分布

| 真值 | VALID | FLAT_WITH_HISTORY | FLAT_EVER_HISTORY | FLAT_NO_HISTORY | FLAT_HISTORY_UNKNOWN | GAP | SATURATED | POLARITY_INVALID | QUANTILE_ARTIFACT | NOISY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 | 240987 | 939 | 0 | 4 | 1 | 0 | 0 | 0 | 53 | 0 |
| U1 | 30465 | 296 | 0 | 2 | 0 | 0 | 0 | 0 | 20 | 1 |
| U2 | 702 | 900 | 601 | 88 | 20 | 0 | 0 | 0 | 158 | 0 |
| U3 | 1758 | 616 | 0 | 20 | 3 | 0 | 0 | 0 | 57 | 1 |

## 6. 混淆矩阵

| 真值 \ 预测 | U0 | U1 | U2 | U3 | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| U0 | 238333 | 1801 | 939 | 683 | 228 |
| U1 | 19984 | 10395 | 296 | 59 | 50 |
| U2 | 391 | 148 | 1501 | 82 | 347 |
| U3 | 1360 | 187 | 616 | 124 | 168 |

特征样本：每 50 个有标签窗取 1 个，共 5554 个；分位数见 JSON `feature_summary`。
