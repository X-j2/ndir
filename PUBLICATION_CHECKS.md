# 发布核验记录

2026-11-02，独立克隆整理副本，未修改原开发目录业务代码。

| 检查 | 结果 |
|---|---|
| Python文件语法解析 | 通过 |
| 所选源码及模型与源文件SHA256比较 | 一致；Markdown副本仅调整链接 |
| Markdown本地文件链接 | 未发现失效目标；未发布材料已改为文字说明 |
| 常见GitHub Token、AWS Key、私钥模式扫描 | 未发现匹配；此项为模式检查，并非无限范围安全保证 |
| 排除清单 | 未包含DOCX/PDF/PPTX/ZIP/EXE/数据库/CSV/视频、个人config.json；未包含论文、专利或固件目录 |
| 无原始缓存运行149项训练测试 | 148项通过，1项因缺少病例缓存报错 |
| 排除唯一原始缓存依赖项 | 148项通过，0失败 |
| 内存临时指向原工作区已有缓存 | 149项通过，0失败；没有修改代码、复制或上传原始缓存 |

完整测试入口（在“模型训练代码”目录）：

```powershell
python -m unittest discover -s tests -q
```

缺数据项是`test_stage2_data.TestStage2Data.test_no_window_cases_keep_truth_events`，会读取病例7与106。其余148项的本次运行是在内存中展开unittest套件，准确排除此测试ID，保留其余测试，没有改写测试源文件。

本次未做GUI、串口实物、打包EXE、固件编译、新实验训练或Final Test模型评价。测试仍出现原有`preprocessing/signal_processing.py:171`的sqrt运行警告，未影响上述通过结果；未在发布时修复该代码。

原始阶段报告可能记载其他测试数量，代表对应阶段的历史验证；本表只记录这次发布整理副本实际核验。
