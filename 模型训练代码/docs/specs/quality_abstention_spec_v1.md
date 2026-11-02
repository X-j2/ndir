# CF-REM 质量与弃权规范 v1（Stage 0 冻结）

> 版本：quality_abstention_spec_v1（内容修订 rev3）  
> 日期：2026-10-29  
> 状态：Stage 0 冻结；Stage 1 实现并只在开发集上调阈值  
> 依赖：`labels_spec_v1.md`、`event_spec_v1.md`  
> 证据：`report/stage0/quality_analysis.md`、`report/stage0/quality_analysis.json`

---

## 1. 要解决的问题

v1.1 把“平坦波形”同时当作屏息证据和质量门禁：

- 平坦 15s 窗 `A=q90-q10=0` → 无法计算 `z` → 被 `A<1e-6` 判 invalid；
- 有微弱噪声的平坦窗 `A>1e-6` → 分位数归一化把噪声放大成有起伏的 `z` → `flat_run_sec` 未必成立；
- 结果：最典型的无呼吸片段可能被全部弃权，U2 的 3,244 个 15s 正窗不能等同于可检测正窗。

本规范把**平坦候选**从 **断线/饱和/极性问题/噪声** 中分离出来，规定：

1. 先做“信号是否可用”的物理判别；
2. 平坦但**过去**（近 300s 或更早的已结束 bin）有已验证呼吸 → `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY`，进入 U2 证据，不弃权；
3. 过去从未有呼吸/历史不足/长缺口/饱和/极性错误 → `FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / UNKNOWN，不报警；
4. 不得用一个 `A<1e-6` 门把两者混在一起；
5. 不得使用未来数据（完整病例的 future breathing）来决定当前窗状态。

---

## 2. Stage 0 证据（rev3，250 例，90s/45s 与 15s/7.5s，因果历史）

| 子集 | n | A<1e-6 | A<1 | flat 候选 | 近 300s 有呼吸 | 仅更早有呼吸 | 从未有呼吸 | 历史不足 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U2 90s（可靠） | 386 | 110 (28.5%) | 273 (70.7%) | 373 (96.6%) | 120 | 160 | 85 | 8 |
| U2 15s（可靠） | 3,139 | 1,673 (53.3%) | 2,281 (72.7%) | 2,364 (75.3%) | 564 | 1,252 | 518 | 30 |
| 非 U2 90s 抽样 | 2,000 | 104 (5.2%) | 134 (6.7%) | 242 (12.1%) | 125 | 27 | 78 | 12 |

**判读：**

- 28.5% 的 U2 90s 窗和 53.3% 的 U2 15s 窗会被 v1.1 的 `A<1e-6` 直接判 invalid；这是最典型的无呼吸片段，不能弃权。
- 70.7%/72.7% 的 U2 窗 `A<1 mmHg`；非 U2 抽样只有 6.7%。A 有很强的分离度，但阈值必须用相对/上下文方式，不能写死设备单位。
- **只分析可靠 U2 事件窗（≥2 条关联读数）**；低支撑 U2 事件（<2 读）映射到 `ignore_low_support`，不进入这里的 U2 质量统计。rev3 的支撑关联修正改变了可靠/低支撑集合，故本节数字与 rev2 不同。
- **历史必须是因果的**：`FLAT_WITH_HISTORY` 只看最近 300s 的已结束 bin；`FLAT_EVER_HISTORY` 只看更早的已结束 bin；未来 bin 永远不参与。不得再用“完整病例有呼吸”这种未来信息。
- U2 90s 的 373 个平坦窗中，120 个有近 300s 呼吸、160 个仅更早有呼吸、85 个从未有呼吸、8 个历史不足；U2 15s 的 2,364 个平坦窗中分别是 564/1,252/518/30。**从未有呼吸**的窗进入 UNKNOWN，不能靠未来数据翻案。
- **`MAD(diff(x))` 在平台/量化段上会退化为 0**：U2 90s 的 `zero_diff_frac` 均值 97.7%，U2 15s 96.5%，非 U2 84.3%。因此 v1.1 的 `noise_ratio` 会在大多数窗口为 0，SQI 会把平台误判为“很干净”。Stage 1 必须改用平台鲁棒的噪声估计（建议 2Hz 残差 `noise_hf` + 绝对噪声下限），不能沿用 `MAD(diff)` 阈值。

---

## 3. 状态定义

| 状态 | 含义 | 是否出特征/报警 | UI |
|---|---|---|---|
| `VALID` | 信号可用，非平坦 | 是 | 正常显示 |
| `FLAT_WITH_HISTORY` | 平坦/低幅，且最近 300s 有已验证呼吸 | 是（直接给 U2 证据，不计算 z 峰特征） | “疑似 CO₂ 低值/缺失样事件” |
| `FLAT_EVER_HISTORY` | 平坦，最近 300s 无呼吸，但更早的过去有已验证呼吸 | 是（长屏息/低置信候选；不能单独报警） | “长时 CO₂ 低值/缺失样事件（低置信）” |
| `FLAT_NO_HISTORY` | 平坦，且当前记录过去从未有已验证呼吸 | 否 | “无有效呼吸信号/无法判断” |
| `FLAT_HISTORY_UNKNOWN` | 历史不足 30s | 否 | “历史不足/无法判断” |
| `GAP` | 长缺口/NaN 过多 | 否 | “信号中断/数据不足” |
| `SATURATED` | 上/下限饱和或长时间贴轨 | 否 | “传感器饱和” |
| `POLARITY_INVALID` | 主峰方向为负/极性可疑 | 否 | “信号极性异常” |
| `NOISY` | 高频噪声过高，峰/周期不可信 | 否 | “信号噪声过大” |

`FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 不是 `UNKNOWN`。`FLAT_NO_HISTORY` / `FLAT_HISTORY_UNKNOWN` / `GAP` / `SATURATED` / `POLARITY_INVALID` / `NOISY` 进入 UNKNOWN，不报警。

---

## 4. 处理顺序（必须按顺序）

### 4.1 缺失/缺口

```
nan_frac = 窗口内 NaN 比例
max_nan_run_sec = 最长连续 NaN / 100Hz
if max_nan_run_sec > 2.0 or nan_frac > 0.10:
    state = GAP
```

- 短缺口 ≤2s 用 `fill_short_gaps(max_gap=200)` 插值；
- 长缺口不插值，窗口直接 `GAP`；
- 事件时间线/监测小时的 unknown 区间与此一致。

### 4.2 饱和/贴轨

- 设备需提供 `rail_low`/`rail_high`（ADC/ppm 上下限；当前固件 ppm 裁剪到 [0, 50000]）。
- 若窗口内 `fraction(x == rail_low or x == rail_high) > 0.05` 或连续贴轨 >2s → `SATURATED`。
- VitalDB 参考波形没有设备 ADC rail；Stage 0 的 `max_value_frac>0.8 且 max>=40` 只作代理，不能直接移植到设备。
- 饱和窗口不参与 U2 证据；不能把“饱和”当“平坦屏息”。

### 4.3 极性

- 期望主峰向上（呼气 CO₂ 高值）。
- 若 `z` 的主峰方向为负（如 `mean_prominence<0` 或负峰数 > 正峰数），或已知增益符号为负 → `POLARITY_INVALID`。
- **不静默翻转**；不把负增益数据送进模型。
- Stage 1 需在开发集上用合成负增益测试；Final Test 不用于调阈值。

### 4.4 幅度/平坦候选

```
A = q90(x) - q10(x)
A_ref = 最近 5 分钟已结束 VALID 窗口的 A 中位数（不足则用当前窗 A）
```

- VitalDB 暂定 `A_floor = 1.0 mmHg`；设备输入单位任意，必须用相对量：
  - `A < 0.05 * A_ref` 或 `A < 3 * noise_floor` → 低幅；
  - `A_ref` 只取**当前窗之前**的 VALID 窗；不得用未来窗。
- **因果历史**（只使用当前窗和已结束的过去）：
  - 15s bin、15s 步长；每个 bin 计算 `A` 与峰数；
  - 只有 `bin.end_s <= window.start_s` 的 bin 才是历史；
  - `recent`：`bin.start_s >= window.start_s - 300s`；
  - `ever`：任意已结束的历史 bin；
  - 历史总时长 <30s → 历史不足；
  - **未来 bin 永远不参与**；不得再用完整病例的 `case_has_breathing`。
- 若 `A < A_floor` 或 `flat_run_sec >= 10`：
  - 历史不足 → **`FLAT_HISTORY_UNKNOWN`**；
  - 近 300s 有已验证呼吸 → **`FLAT_WITH_HISTORY`**（高置信候选）；
  - 近 300s 无、但更早的过去有 → **`FLAT_EVER_HISTORY`**（长屏息/低置信候选，不能单独报警）；
  - 从未有已验证呼吸 → **`FLAT_NO_HISTORY`**（UNKNOWN，不报警）。
- `flat_run_sec`：
  - 若 `A >= A_floor`：在 `z=(x-q10)/A` 上计算 `z < 0.15` 的最长连续段；
  - 若 `A < A_floor`：`flat_run_sec = window_sec`（窗口本身已平坦），但状态由上面的因果历史决定；
  - 不用 v1.0 的 `flatness<0.25` 方向错误的写法。

### 4.5 噪声

- **不要**只用 `noise_ratio = 1.4826*MAD(diff(x))/sqrt(2)/A`；Stage 0 证明它在 84–98% 的窗口上退化为 0。
- 建议 Stage 1 使用：
  - `noise_hf = 1.4826 * MAD(x - lowpass_2Hz(x))`（绝对噪声，mmHg/设备单位）；
  - `noise_ratio_hf = noise_hf / max(A, eps)`；
  - `noise_floor` = 设备静态噪声（未标定也可从静息/无呼吸段估计）。
- 若 `noise_hf` 超过绝对上限，或 `noise_ratio_hf` 超过开发集冻结阈值 → `NOISY`。
- 平坦窗口的 `noise_hf` 接近 0 是“干净平坦”，不是“高质量正常”；SQI 不能只看噪声。

### 4.6 SQI（建议公式，Stage 1 调权重）

```
sqi = clip(1
           - w_gap   * min(1, nan_frac / 0.10)
           - w_sat   * min(1, rail_frac / 0.05)
           - w_noise * min(1, noise_hf / noise_limit)
           - w_polar * polarity_invalid
           , 0, 1)
```

- `SQI < 0.35` → 该尺度权重 0；全部尺度 <0.35 → `UNKNOWN`，不报警；
- `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` 不因 `SQI` 低而自动弃权；它们的证据来自“平坦+已验证历史”，不是峰/周期特征；
- 任何阈值必须在开发集上冻结，Final Test 只评估。

---

## 5. 与 v1.1 的差异

| v1.1 | v1 Stage 0 修订 | 原因 |
|---|---|---|
| `A<1e-6` → invalid/abstain | `A<A_floor` + 因果历史 → `FLAT_WITH_HISTORY` / `FLAT_EVER_HISTORY` / `FLAT_NO_HISTORY` | 28.5%/53.3% 的 U2 窗会被误弃权 |
| 先算 `z`，再判 `flat_run` | 先判幅度/缺口/饱和/极性，平坦候选不依赖 `z` | `A=0` 时无法算 `z` |
| `noise_ratio = MAD(diff)/A` | 增加 `noise_hf`（2Hz 残差）+ 绝对噪声下限 | 平台/量化使 `MAD(diff)=0` |
| 全低 SQI → UNKNOWN | 平坦+已验证历史 → 候选；从未有呼吸 → NO_HISTORY/UNKNOWN | 区分“屏息”和“断线/无信号” |
| 使用完整病例 `case_has_breathing` | 只用当前窗 + 已结束历史 bin；未来 bin 不参与 | 可部署、无未来泄漏 |
| 负增益未定义 | `POLARITY_INVALID` 弃权，不静默翻转 | 正仿射不变性只对 a>0 成立 |

---

## 6. 输出与 UI

- `VALID`：正常显示相对波形；U 模式结果可用。
- `FLAT_WITH_HISTORY`：显示“疑似 CO₂ 低值/缺失样事件”；不显示“临床屏息”；不显示 EtCO₂ 数值（未标定）。
- `FLAT_EVER_HISTORY`：显示“长时 CO₂ 低值/缺失样事件（低置信）”；不得单独触发正式报警，需更高尺度/其他证据确认。
- `FLAT_NO_HISTORY`/`FLAT_HISTORY_UNKNOWN`/`GAP`/`SATURATED`/`POLARITY_INVALID`/`NOISY`：显示“无法判断：<原因>”；不报警；不得默认正常。
- 相对模式：EtCO₂ 卡片显示“—/未标定”；不显示 mmHg。
- 任何 UNKNOWN 都不驱动用户告警；旧固件四分类结果也不得驱动 U 模式告警（Stage 3）。

---

## 7. 验收测试（Stage 1 实现）

1. 常数 `x=0`、过去 300s 有已验证呼吸 → `FLAT_WITH_HISTORY`，不是 `GAP`/`UNKNOWN`。
2. 常数 `x=0`、过去 300s 无、更早有呼吸 → `FLAT_EVER_HISTORY`。
3. 常数 `x=0`、过去从未有呼吸 → `FLAT_NO_HISTORY`（UNKNOWN，不报警）。
4. 常数 `x=0`、历史不足 30s → `FLAT_HISTORY_UNKNOWN`。
5. **未来呼吸不改变状态**：同一平坦窗，未来 60s 有正常呼吸，但过去无呼吸 → 仍是 `FLAT_NO_HISTORY`，不能变成 `FLAT_WITH_HISTORY`。
6. 注入 5s NaN → `GAP`。
7. 注入上限饱和 5% → `SATURATED`。
8. 负增益 `x → -x`（峰向下） → `POLARITY_INVALID`。
9. 正常 capnogram → `VALID`。
10. 平台正常波形（`MAD(diff)=0` 但 `noise_hf` 低） → `VALID`，不是 `NOISY`。
11. 平坦+近期呼吸的 U2 窗 → 不因 `A<1e-6` 弃权。
12. 全部尺度 SQI<0.35 → `UNKNOWN`，0 报警。
13. 相对模式 UI 不显示 EtCO₂ 数值/mmHg。
14. 阈值只在开发集调；Final Test 不参与。
15. 复算质量分析：可靠 U2 90s 的 `A<1e-6`≈28.5%、U2 15s≈53.3%；因果历史拆分见 `report/stage0/quality_analysis.md`（rev3）。
16. 低支撑 U2 窗（`ignore_low_support`）不进入 U2 质量统计，也不能当负例。

---

## 8. 已知限制

- `A_floor`、`noise_limit`、`SQI` 权重是暂定值；Stage 1 必须在开发集上冻结，不能按 Final Test 调。
- VitalDB 参考波形不是设备 ADC，饱和/贴轨代理不能直接当设备饱和；设备侧需要 rail 定义。
- 当前用 300s 近史 + 全部已结束历史的 15s bin；`FLAT_EVER_HISTORY` 是低置信长屏息候选，不能单独报警。最近史窗口、bin 长度、`A_floor` 都要在开发集冻结，Final Test 不参与。
- 因果历史只能区分“过去有/无已验证呼吸”，不能区分“长屏息”与“传感器拔管后长时间平坦”；后者需要额外的基线连续性/断线判别，Stage 1 未解决。
- 平坦候选仍可能是漏气、脱管、采样管故障；论文只能写“参考 EtCO₂ 低值/缺失样事件”，不能写临床屏息。
