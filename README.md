# QRAM/Grover 独立复现包

这是2026-11-04整理的独立数值项目。**复制整个文件夹，或解压整个ZIP，即可在新路径中安装环境并复现线路与相干模拟，无需原电脑工作区、实验平台或pyQCat。**

先读[AI_SIMULATION_HANDOFF.md](AI_SIMULATION_HANDOFF.md)：目标、全部R5/R6上下行门序列、17角色query/Grover步骤、相位约定、矩阵/数组格式、已有结果与模型边界都在其中。

实际重跑证据见[独立复现验证记录](validation/PORTABILITY_TEST.md)。本包已在新venv与迁移目录中重算全部32个case，并保留[新生成的线路](circuits/README.md)。

## 1. 环境安装

推荐Python **3.10**。锁定依赖在[requirements.txt](requirements.txt)。计算、画线路和分析不要求GPU，不要求TeX，也不调用量子硬件。

[requirements-lock.txt](requirements-lock.txt)另外记录干净环境实际安装的全部直接/间接依赖；若需要最严格地匹配本次测试环境，可将安装命令的requirements.txt换为requirements-lock.txt。

Windows PowerShell / PyCharm Terminal，在本文件夹中执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py verify
```

也可在已经安装好依赖的Python环境中直接使用`python reproduce.py ...`。PyCharm的解释器选择新建的`.venv/Scripts/python.exe`，脚本为`reproduce.py`，参数填`quick --run-name quick-v1`等，工作目录为此文件夹。

若 PyCharm 自动创建了 `venv`，可直接使用 `./venv/Scripts/python.exe`。Git 仓库排除本地解释器目录和 `runs/`，克隆后按上面的步骤创建环境即可。

Linux/macOS安装同一Python版本和依赖后，使用`.venv/bin/python`替代上述解释器路径。本包路径由脚本位置解析，不含运行时固定盘符；已验证的操作系统与数值结果见`validation/`，跨平台BLAS数值比较使用1e-6容差。

## 2. 推荐运行顺序

```powershell
python reproduce.py verify
python reproduce.py circuits --run-name circuits-v1
python reproduce.py ideal --run-name ideal-v1
python reproduce.py quick --run-name quick-v1
python reproduce.py full --run-name full-v1
```

| 命令 | 执行内容 |
| --- | --- |
| verify | 核对随包文件SHA256与依赖版本，不重算传播 |
| circuits | 实际重建4份down/up门序列与SVG、1/2/3层六份完整调度、所用有效算子NPZ、线路及波形PNG/PDF |
| ideal | 逐原生门检查R5/R6各16数据库完整query与四目标理想Grover，并核验局部入口 |
| pulse | 重算局域16态、down/up/实际往返、控制扫描、配对消融及四/五能级步长检查 |
| case | 实际传播一个完整两层query和Grover，默认bond128，与随包冻结结果比较 |
| quick | circuits＋全部32个理想query＋局域pulse＋R5/R6各一个M=0010完整相干case |
| full | 从空运行目录重算32个相干case、局域/收敛/阶段/规模检查、理想原生检查和25张图，并与冻结基线比较 |
| plots | 从完整保存数据重新绘制25张分析图，默认只读取随包参考数据 |
| report | 准备或编译可编辑中文TeX报告，需要另装XeLaTeX或Tectonic |
| calibrate | 可选：从参考器件开始重新优化原生波形及运行shot-based局域校准流程，输出与冻结基准分离 |

quick包含真实传播，耗时大于文件校验。full包含三层张量网络和步长检查，适合在有足够内存、时间的电脑上执行。用户可以先用quick确认环境和关键指标，再进行full。

## 3. 独立任务命令

```powershell
python reproduce.py case --scheme r5 --memory 0010 --run-name r5-case-v1
python reproduce.py case --scheme r6 --memory 0010 --run-name r6-case-v1
python reproduce.py pulse --run-name local-pulse-v1
python reproduce.py plots --run-name figures-v1
python reproduce.py plots --data-run full-v1 --run-name figures-full-v1
```

`--memory`顺序为M00,M01,M10,M11，`0010`标记地址10。case默认bond128；改bond不再宣称与128基线逐数相等。

重新标定可选，不是复现冻结基准的必要步骤：

```powershell
python reproduce.py calibrate --run-name new-native-v1 --calibrate-only
python reproduce.py calibrate --run-name new-calibration-v1 --shots 1000
```

第一条只做原生波形优化；第二条在本地pulse模拟器中继续局域readout/paired/Ramsey/truth等shot-based流程。它通过模拟后端替换实验执行函数，不需要实验平台。新校准结果不会自动替换基准门库。

## 4. 结果位置与续跑

```text
runs/<run-name>/
  run_manifest.json             # 源码、冻结参数、依赖版本
  feasibility/cases/            # 新算case指标
  feasibility/arrays/           # 新算Choi/K0/态NPZ
  feasibility/figures/          # full/plots生成的分析图
  circuits/nodes/               # 四方向SVG和完整门/帧JSON
  circuits/schedule_*.json       # 1/2/3层完整原生调度
  circuits/operators.npz         # 按SHA256寻址的实际有效算子
  circuits/figures/              # 新生成线路/拓扑/波形图
  *_comparison.json             # 新结果与参考的比较
  runtime_input_audit.json       # 本包内数值文件访问记录
```

随包`Experiment/DistributedGroverNoDecoherence20261102/feasibility/`是参考快照；统一入口不覆盖它。full第一次不会把参考32个case复制成新结果；它会真正重算。中断后用同名、同源码、同冻结参数运行可续算已完成的新case。

源码、冻结参数、NumPy或SciPy版本改变时，同名运行会拒绝混用。请建立新run-name。跨操作系统续跑建议也用新run-name。

若有意开发修改源码，分发校验会先报告文件变化。请在单独副本中明确运行`python bundle_integrity.py --refresh`接受自己的修改，再用新run-name。此操作保留原始来源指纹与参考数据；物理改动导致与旧基线不同，数值比较仍会明确报错，不能将新参数结果混入本报告。

## 5. 包内内容与模型边界

```text
reproduce.py / requirements.txt / AI_SIMULATION_HANDOFF.md
Experiment/
  DistributedGroverNoDecoherence20261102/  # 当前数值脚本、全部参考NPZ/JSON、报告
  DistributedGroverSimulation20261102/     # circuit、model、MPS和共享辅助函数
  RouterPulseCalibration20261002/          # pulse积分、参考器件、冻结门库与可选校准
  LabRouterR5R6/router_r5_r6/              # R5/R6协议、分析、局域模拟复用接口
  LabPartialCSWAPVZ4/partial_cswap_vz4/    # 01酉分解、两跃迁帧与层调度
source_documents/                         # 原始四页规划PDF
provenance/                               # 原来源指纹与便携性改动说明
validation/                               # 本包验证记录（分发版）
```

部分原实验接口源文件为协议指纹完整性保留，但统一数值入口不导入硬件执行器。历史含噪主程序不作为支持入口；Noise默认值有噪声，当前统一入口始终使用ZERO。

冻结JSON中仍可能存在原电脑的`readout_file`等历史元数据。当前传播只读取本包中的paired、angles和原生参数，不访问这些历史地址。数值输入审计会拒绝从本包以外读取device/native/direction参数。

局域router使用4能级pulse导出过程；完整算法已覆盖，但跨片、部分接口和存储门仍是理想01过程。存储保持是经典基态模型结构保证；这里不是17体共享参数的连续全波形硬件仿真。保真度约99%也不是对真实芯片的保证。

## 6. 报告

直接阅读随包的[中文PDF](Experiment/DistributedGroverNoDecoherence20261102/results/线路可行性_中文LaTeX报告.pdf)即可，不必安装编译器。

```powershell
python reproduce.py report --run-name report-v1 --prepare-only
python reproduce.py report --run-name report-v2
python reproduce.py report --run-name report-v3 --compiler "你的tectonic或xelatex完整路径"
```

report重编译的是冻结参考版，输出新的独立TeX工程；不将新结果悄悄混入旧正文。中文字体采用TeX常见Fandol，不依赖Windows专有字体；finalize_pdf补齐缺失的Unicode映射以便复制中文，保留原内容流。TeX编译器是可选外部工具，不捆绑某个系统的可执行文件。

## 7. 版本一致性

所有传播内核、pulse模型、线路协议和冻结数值参数均保留原内容。为跨路径重新标定，局域shot模拟的种子键改用相对目录；它只影响重新采样的新校准，不改变随包冻结参数和当前整网基线。

`bundle_manifest.json`记录分发文件完整性，`provenance/`说明来源。请通过统一入口运行，避免直接执行内部历史脚本的原默认输出路径。
