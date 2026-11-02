# Stage 0 上位机测试基线

- 日期：2026-10-29
- 仓库：`e:/大创/任务书/上位机代码`
- 分支：`main`；HEAD：`1dab603382e8ed7a3f3197990ba554a2b26893be`
- 环境：Python 3.12.6，numpy 2.3.1，pandas 2.2.3，PySide6 6.10.2，onnxruntime 1.26.0，scikit-learn 1.7.2
- 运行设置：`QT_QPA_PLATFORM=offscreen`、`PYTHONIOENCODING=utf-8`
- 完整输出：`report/stage0/host_tests.log`

## 结果

| 测试文件 | 测试数 | 退出码 | 耗时 | 结果 |
|---|---:|---:|---:|---|
| test_business.py | 10 | 0 | 0s | 通过 |
| test_db_log.py | 22 | 0 | 0s | 通过 |
| test_e2e.py | 9 | 0 | 7s | 通过 |
| test_report.py | 11 | 0 | 0s | 通过 |
| test_serial.py | 48 | 0 | 2s | 通过 |
| test_serial_connect.py | 5 | 0 | 3s | 通过（打印 COM 端口错误属预期） |
| test_sim_ai.py | 10 | 0 | 3s | 通过 |
| **合计** | **115** | **0** | **约 15s** | **全部通过** |

## 备注

- `test_serial_connect.py` 会尝试打开不存在的 `COM_NOPE_XYZ`，打印“设备未找到/打开失败”，但测试本身通过；这是预期行为。
- `test_sim_ai.py` 有 `ResourceWarning: unclosed file`，不影响测试结果；Stage 3 重构时顺手修。
- 测试会写 `co2_monitor.db` 与 `logs/co2_monitor_2026-10-29.log`；二者不在 git 跟踪范围（或为忽略文件）。测试前后 `git status --short` 的已跟踪文件集合未变，未产生新的 tracked 修改。
- 测试运行结束时打印“每日摘要…共 12 条记录”“共 20 条记录”等，是测试数据，不是真实病人数据。

## 与 v1.1 的差异

v1.1 写“115 项静态 `def test_` 数，本次未运行”；Stage 0 实际运行了 7 个文件，115 项全部通过。后续如新增测试，必须更新本表和 SSOT，不能只写静态函数数。

## rev1 重跑（2026-10-29）

- 日志：`report/stage0/host_tests_rev1.log`
- 结果：7 个文件、**115 项全部通过**，退出码均为 0；无失败项。
- 测试未改上位机代码；只重跑回归。DB/log 副作用与首次相同，未产生新的 tracked 修改。

## rev2 重跑（2026-10-29）

- 日志：`report/stage0/host_tests_rev2.log`
- 结果：7 个文件、**115 项全部通过**，退出码均为 0；无失败项。
- 只改训练端 Stage 0 规则/统计/测试，未改上位机代码。

## rev3 重跑（2026-10-29）

- 日志：`report/stage0/host_tests_rev3.log`
- 结果：7 个文件、**115 项全部通过**，退出码均为 0；无失败项。
- 只改训练端事件支撑关联/统计/测试，未改上位机代码。
