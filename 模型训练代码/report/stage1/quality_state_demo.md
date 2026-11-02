# Stage 1 质量状态合成演示

仅用合成波形；不是设备实测。

| case | state | UNKNOWN | rule | A | finite_span | constant_flat | quantile_artifact | flat_run_sec | peak_rate |
|---|---|---|---|---:|---:|---|---|---:|---:|
| normal + recent history | VALID | False | U0 | 35.000 | 35.000 | False | False | 2.39 | 16.0 |
| flat A=0 + recent history | FLAT_WITH_HISTORY | False | U2 | 0.000 | 0.000 | True | False | 15.00 | 0.0 |
| flat A=0 + no history | FLAT_NO_HISTORY | True | UNKNOWN | 0.000 | 0.000 | True | False | 15.00 | 0.0 |
| flat A=0 + insufficient history | FLAT_HISTORY_UNKNOWN | True | UNKNOWN | 0.000 | 0.000 | True | False | 15.00 | 0.0 |
| 5s missing gap | GAP | True | UNKNOWN | 35.000 | 35.000 | False | False | 6.39 | 12.0 |
| clipped at rail | SATURATED | True | UNKNOWN | 35.000 | 35.000 | False | False | 2.39 | 16.0 |
| inverted trace | POLARITY_INVALID | True | UNKNOWN | 35.000 | 35.000 | False | False | 1.61 | 12.0 |
| high-frequency noise | NOISY | True | UNKNOWN | 65.777 | 73.042 | False | False | 0.04 | 16.0 |
| low-amplitude noise | NOISY | True | UNKNOWN | 0.702 | 1.522 | False | False | 0.06 | 12.0 |
| sparse pulses + recent history | QUANTILE_ARTIFACT | True | UNKNOWN | 0.000 | 40.000 | False | True | 15.00 | 0.0 |
| sparse pulses + no history | QUANTILE_ARTIFACT | True | UNKNOWN | 0.000 | 40.000 | False | True | 15.00 | 0.0 |
