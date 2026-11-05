# 独立复现验证记录

日期：2026-11-04。此记录描述实际执行的测试，不是仅检查文件是否存在。

## 测试方式

1. 使用Python 3.10.20创建新的venv，按requirements.txt从pip安装依赖，不继承原数值环境的site-packages。
2. 将整个项目复制到另一个带中文与空格的路径；清空PYTHONPATH，禁用用户site-packages，从不同工作目录启动。
3. 执行quick，重新构造线路、32个理想query、局域pulse及R5/R6两个代表性整机case。
4. 在空运行目录执行full，实际重新计算全部32个整机相干case、局域/步长/能级/规模/阶段检查、理想原生矩阵，以及25张图。
5. 与包内参考的原始矩阵和指标比较；核对项目模块均从迁移后的包内加载。

本机为Windows x64；不是对其他物理电脑、Linux或macOS的实测。包中没有Windows数值扩展二进制，其他电脑由pip安装适配其系统的依赖。

## 已通过的结果

| 检查 | 实际结果 |
| --- | --- |
| 全部相干case | 32/32通过 |
| 与参考case关键指标最大差 | 0.0 |
| 与参考Choi矩阵最大元素差 | 0.0 |
| 理想完整query最大元素误差 | 8.482556806932493e−15 |
| local.npz所有数值数组最大差 | 0.0 |
| tracking.npz所有数值数组最大差 | 0.0 |
| pulse_convergence/scaling/resources | 解析后JSON一致 |
| 新生成分析图 | 25张PNG及25张矢量PDF |
| 新生成节点线路 | R5/R6 × down/up四份SVG及完整JSON |
| 新生成完整整网调度 | 1/2/3层 × R5/R6共六份 |
| 实际有效算子 | 108个，NPZ内容与调度SHA256逐一匹配 |
| full耗时 | 本机约168.4秒，其他电脑可不同 |
| 报告工程准备 | report --prepare-only通过 |

新生成的完整线路已另存于包根目录[../circuits/](../circuits/)。原32个参考case没有被覆盖。

## 可选重新校准的验证范围

局域pulse模拟后端实际执行了一个1000 shots的准备/读出任务，在两个不同输出根目录下使用相同相对种子，得到一致计数；未导入pyQCat。完整波形重新优化及全部shot-based校准流程没有在本次打包过程中重跑，复现冻结基准也不需要这一步。

## 机器可读证据

- [SELF_TEST.json](SELF_TEST.json)：整体结果。
- [all_32_case_comparisons.json](all_32_case_comparisons.json)：每个case的比较。
- [fresh_checks.json](fresh_checks.json)：新跑的理想、正性、层调度与数值检查。
- [full_result.json](full_result.json)、[quick_result.json](quick_result.json)：运行结果与模块来源。
- [runtime_input_audit.json](runtime_input_audit.json)：本包数值文件访问记录。
- [clean_environment_run_manifest.json](clean_environment_run_manifest.json)：测试环境及输入指纹。
- [calibration_backend_smoke.json](calibration_backend_smoke.json)：局域shot模拟后端的单任务检查。
