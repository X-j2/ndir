# Stage 1 B0 规则基线（30s）

- 规则版本：`b0_rule_si_v1`；SI 规范：`si_feature_spec_v2`；质量规范：`stage1_quality_rule_spec_v3`
- 窗口：30s / 步长 15.0s；开发集 200 例（Final Test 未加载）
- 有标签窗：138989；覆盖：138608 (99.7%)；UNKNOWN：381
- Accuracy（全部/覆盖）：0.9377 / 0.9402；Macro-F1：0.5751

## 1. 四类结果

| 类别 | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| U0 | 121063 | 0.963 | 0.977 | 0.970 |
| U1 | 15423 | 0.869 | 0.728 | 0.792 |
| U2 | 1186 | 0.326 | 0.649 | 0.434 |
| U3 | 1317 | 0.191 | 0.072 | 0.105 |

## 2. U2 二分类（UNKNOWN 不当作正例）

- U2 support：1186；precision 0.326，recall 0.649，F1 0.434
- U2 预测分布：{'U0': 104, 'U1': 69, 'U2': 770, 'U3': 28, 'UNKNOWN': 215}
- U2 质量状态：{'VALID': 235, 'FLAT_WITH_HISTORY': 493, 'FLAT_EVER_HISTORY': 277, 'FLAT_NO_HISTORY': 54, 'FLAT_HISTORY_UNKNOWN': 11, 'GAP': 0, 'SATURATED': 0, 'POLARITY_INVALID': 0, 'QUANTILE_ARTIFACT': 116, 'NOISY': 0}
- U2 中 A=0/non-finite（尺度无关 invalid）：532；其中真正常数窗 416、分位退化伪迹窗 116；预测 U2 383，UNKNOWN 149
- `A=0` 只表示 q90-q10=0，不等于常数窗。有限样本完全相等才是真正常数窗并保留候选路径；仍含稀疏脉冲/变化的窗为 `QUANTILE_ARTIFACT`/UNKNOWN。
- 低幅判断按 `A < 0.05 * 因果历史 A_ref`，不再使用绝对 A 门限；U2 的 `FLAT_*` 状态分布见上表。

## 3. 分位退化伪迹与常数窗

- 全部有标签窗中，分位退化伪迹：201；按真值 {'U3': 43, 'U2': 116, 'U0': 36, 'U1': 6}
- 全部有标签窗中，真正常数窗：609；按真值 {'U2': 416, 'U3': 127, 'U0': 52, 'U1': 14}
- 分位退化伪迹先于噪声/平坦判定，进入 UNKNOWN，不能成为 U2 候选。

## 4. 平坦候选与 UNKNOWN

- 平坦候选窗：2362；按真值 {'U2': 770, 'U0': 861, 'U3': 487, 'U1': 244}
- 平坦候选中预测 U2：2362；UNKNOWN：0
- `FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / GAP / 饱和 / 极性错误 / 分位退化伪迹 / 噪声均进入 UNKNOWN，不被当作 U0。

## 5. 质量状态分布

| 真值 | VALID | FLAT_WITH_HISTORY | FLAT_EVER_HISTORY | FLAT_NO_HISTORY | FLAT_HISTORY_UNKNOWN | GAP | SATURATED | POLARITY_INVALID | QUANTILE_ARTIFACT | NOISY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 | 120159 | 859 | 2 | 7 | 0 | 0 | 0 | 0 | 36 | 0 |
| U1 | 15173 | 244 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| U2 | 235 | 493 | 277 | 54 | 11 | 0 | 0 | 0 | 116 | 0 |
| U3 | 764 | 485 | 2 | 19 | 4 | 0 | 0 | 0 | 43 | 0 |

## 6. 混淆矩阵

| 真值 \ 预测 | U0 | U1 | U2 | U3 | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| U0 | 118228 | 1544 | 861 | 350 | 80 |
| U1 | 3914 | 11232 | 244 | 24 | 9 |
| U2 | 104 | 69 | 770 | 28 | 215 |
| U3 | 571 | 87 | 487 | 95 | 77 |

特征样本：每 50 个有标签窗取 1 个，共 2780 个；分位数见 JSON `feature_summary`。
