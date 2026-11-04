# 面向另一位AI的交接：完整QRAM/Grover相干仿真、具体线路与数据接口

便携复现包版本：2026-11-04。数值结果主要生成于2026-11-02；本次整理没有重新优化门，也没有新增真机实验。

## 0. 先读结论与范围

本项目已经在模拟中走完“两层QRAM、四地址经典数据库、一次完整query、一轮Grover”的全部算法步骤。原生理想矩阵证明、局域四能级波形传播、整树模块过程传播是三种不同层次的证据。

当前应采用的结论是：**在规定入口、冻结局域门库及声明的理想接口假设下，R5/R6可以构成完整的相干QRAM查询，并支持固定线路的四目标单轮Grover。**

算法流程包含初态给定、地址保存、数据编码/下行、存储访问、数据上行、地址恢复、解码、diffusion和最终概率计算。初态是直接初始化，读出是直接取概率；这两端没有完整的实际准备/测量脉冲模拟。

物理模型范围：

| 已纳入 | 当前模型的处理 |
| --- | --- |
| 局域单门/CZ波形 | 四个Duffing振子，每位4能级，保留高能级相位和相干泄露 |
| R5/R6 down、up和实际往返 | 使用实际模板与分别冻结的方向参数，未将实际down矩阵直接取dagger代替up |
| 完整QRAM/Grover | 组合局域脉冲导出过程，传播整个查询与扩散 |
| QPU、模块外搬运、跨片和存储访问门 | 理想01过程延拓，并计入相应门数、层数与参考时长 |
| 四个存储位 | 处于经典基态，以条件变量精确收缩；没有连续求解其物理脉冲 |
| T1/T2、随机门错误、准备翻转、读出混淆 | 当前主结果全部关闭 |
| shots与实验QST重建 | 当前主结果没有抽样；密度矩阵由传播直接计算 |

因此，“全流程已经模拟”不等于“一套共享参数的17体连续全pulse Hamiltonian已经求解”。特别是跨片误差、存储访问失真和真实读出性能，还不是这里百分数的验证对象。

## 1. 工作区、版本和优先阅读入口

本复现包根目录为`reproduce.py`所在目录，可以放在任意盘符和路径。

本文所在研究目录，以下记为`STUDY`：

```text
Experiment/DistributedGroverNoDecoherence20261102
```

优先阅读：

1. 本文：统一术语、线路、模型与数据语义。
2. [CIRCUIT_DETAILS.md](Experiment/DistributedGroverNoDecoherence20261102/CIRCUIT_DETAILS.md)：更长的线路讲解及四方向逐门相位表。
3. [feasibility_data.py](Experiment/DistributedGroverNoDecoherence20261102/feasibility_data.py)：当前无附加噪声主传播入口和`NativeCircuit`。
4. [节点协议 protocol.py](Experiment/LabRouterR5R6/router_r5_r6/protocol.py)：R5/R6、CNOT、方向反转、V展开调用。
5. [全局 circuit.py](Experiment/DistributedGroverSimulation20261102/circuit.py)与[过程库 model.py](Experiment/DistributedGroverSimulation20261102/model.py)：地址协议、网关、模块映射、理想接口延拓。
6. [feasibility_checks.py](Experiment/DistributedGroverNoDecoherence20261102/feasibility_checks.py)：理想原生矩阵、完整query、数值与层调度检查。
7. [冻结参数与数据指纹](Experiment/DistributedGroverNoDecoherence20261102/feasibility/pulse_sources.json)、[整机结果](Experiment/DistributedGroverNoDecoherence20261102/feasibility/SUMMARY.json)、[资源](Experiment/DistributedGroverNoDecoherence20261102/feasibility/resources.json)。

当前中文TeX正文位于[latex_report/main.tex](Experiment/DistributedGroverNoDecoherence20261102/latex_report/main.tex)，具体段落在`latex_report/sections/`，成稿为[中文LaTeX报告](Experiment/DistributedGroverNoDecoherence20261102/results/线路可行性_中文LaTeX报告.pdf)。它是当前人工维护的叙述版本。

已有独立MD正文：[results/REPORT.md](Experiment/DistributedGroverNoDecoherence20261102/results/REPORT.md)。它由旧的报告生成器输出，与TeX正文的组织方式不同；TeX修改不会自动同步回这份MD。

### 1.1 当前入口与历史代码的区别

- 当前相干结果一律从`STUDY/feasibility/`读取。
- `STUDY/feasibility_data.py`明确传入`ZERO`，关闭附加噪声。
- 兄弟目录`DistributedGroverSimulation20261102/runner.py`含旧的随机轨迹、读出与耗散入口。当前仅复用其中若干辅助函数，例如输入bank、目标态、数组打包；不要因此运行其旧worker来复现当前结果。
- `model.Noise()`的默认值包含非零错误和耗散。要复现当前结果，应使用`feasibility_data.ZERO`，不能只实例化默认`Noise()`。
- 当前整机线路用`NativeCircuit`，它把所有外部H展开为两个物理脉冲；不要直接用旧`Circuit`的H占位去复现当前资源数。
- 原工作区的旧研究入口不随本包提供；旧版含噪结果不随本包提供。旧版本约70%的搜索结果不属于本轮相干口径。

## 2. 要实现的数学任务

地址寄存器为`A0,A1`，数据位为`D`。数据库按地址次序写成

$$m=(m_{00},m_{01},m_{10},m_{11}).$$

临时树寄存器记为`R_aux`，它不是局域节点右输出R。目标查询是

$$
Q_m|a\rangle_A|d\rangle_D|0\rangle_{R_{aux}}|m\rangle_M
=|a\rangle_A|d\oplus m_a\rangle_D|0\rangle_{R_{aux}}|m\rangle_M.
$$

要求包括：地址相对相位正确、数据异或读取正确、临时树恢复、经典存储保持。只检查计算基人口不足以验收该任务。

对数据负号态，同一查询满足

$$Q_m|a,-\rangle=(-1)^{m_a}|a,-\rangle,$$

成为地址上的phase oracle。四个one-hot数据库与目标对应：

| 存储字M00 M01 M10 M11 | 目标地址 |
| --- | --- |
| 1000 | 00 |
| 0100 | 01 |
| 0010 | 10 |
| 0001 | 11 |

**四个任务只能改变存储初始化。** 查询体、路由校准、QPU制备和扩散都应共用；目标地址不作为参数进入主体门选择。

## 3. 位序与索引：必须始终一致

### 3.1 单节点

线序固定为`C,I,L,R`，代码索引`0,1,2,3`。控制0走左，控制1走右。

```text
down input:  0000, 0100, 1000, 1100
down output: 0000, 0010, 1000, 1001
```

四维合法入口的任意复数叠加都应满足该映射。up从上面的输出像返回入口。节点并没有被要求在任意四位输入上实现通用完整CSWAP。

### 3.2 两层整网

| 角色 | 具体名称 | 数量 |
| --- | --- | --- |
| 外部QPU | A0、A1、D | 3 |
| 树控制 | C0、C1、C2 | 3 |
| 树数据端口 | T0、T1、T2、L00、L01、L10、L11 | 7 |
| 显式存储的物理角色 | M00、M01、M10、M11 | 4 |

合计17角色。当前量子传播显式包含前13个四能级站点；4个M为基态条件变量。数学参考系统不计为硬件角色。

```text
quantum order:
A0,A1,D,C0,C1,C2,T0,T1,T2,L00,L01,L10,L11

external output order: A0,A1,D
memory string order:   M00,M01,M10,M11
```

节点映射：

| 节点 | C | I | L | R |
| --- | --- | --- | --- | --- |
| root | C0 | T0 | T1 | T2 |
| left | C1 | T1 | L00 | L01 |
| right | C2 | T2 | L10 | L11 |

T1/T2在这里是端口名，与寿命时间无关。根输出与子节点输入共用物理位，不能将同一个位同时赋予两个参考角色的不同工作频率。

数值矩阵采用行=输出、列=输入。AD的01基序是`000,001,010,011,100,101,110,111`；四能级全空间的索引需用`np.ravel_multi_index(word,(4,)*n)`转换，不能把前`2**n`个索引直接当作计算子空间。

## 4. 原生门、相位与分层约定

所有门列表和箭头按**时间顺序从左向右**。矩阵乘积则右侧先作用。

目标跃迁内的约定是

$$R_{j,j+1}(\theta,\phi)=
\begin{pmatrix}
\cos(\theta/2)&-i e^{-i\phi}\sin(\theta/2)\\
-i e^{i\phi}\sin(\theta/2)&\cos(\theta/2)
\end{pmatrix}.$$

| 协议名 | 含义 |
| --- | --- |
| 01_half | R01(π/2,φ) |
| 01_pi | R01(π,φ)，原生完整X01 |
| 12_pi | R12(π,φ)，原生完整X12 |
| cz | 校准CZ及其原生帧补偿 |
| frame / frame_01 / frame_12 | 软件参考帧更新 |

`+Y/2`是01半门φ=+π/2，`−Y/2`是φ=−π/2。π脉冲有`−i`相位；有旁路能级时不能将其随意视为公共全局相位。

基础宏均已展开到允许门集：

```text
CNOT(a→b): X01/2_b(−π/2) → CZ(a,b) → X01/2_b(+π/2)

T(a→b):    CNOT(a→b) → CNOT(b→a)
           2 CZ + 4 half pulses; requires b initially |0>

S(a,b):    CNOT(a→b) → CNOT(b→a) → CNOT(a→b)
           3 CZ + 6 half pulses; full SWAP only on 01⊗01

H(q):      X01/2_q(+π/2) → X01_q(0)
           equals −iH on 01; 2 pulses, not a free primitive
```

异位单门允许并行；不相交CZ允许并行；物理单门与CZ不能同层。采用[physical_layers / layerize](Experiment/LabPartialCSWAPVZ4/partial_cswap_vz4/timing_plan.py)维持依赖、同线占用及barrier边界。

### 4.1 两跃迁参考帧

每条线缓存`(f01,f12)`：01轴用`φ+f01`，12轴用`φ+f12`。

- `frame(φ)`：两个缓存同时加φ。
- `frame_01(φ)`：仅f01加φ。
- `frame_12(φ)`：仅f12加φ。
- 每个单门按原生补偿二元组更新缓存；每个CZ的逐线标量补偿同时加到该线两条跃迁缓存。

扣除已知空闲参考旋转后，逻辑帧定义为

$$F(f01,f12)=\operatorname{diag}(1,e^{-if01},e^{-i(f01+f12)},e^{-i(f01+2f12)}).$$

普通frame在012内因此为`diag(1,e^−iφ,e^−2iφ)`。这些更新改变驱动参考，不是额外物理Z脉冲。

`Library.compile`会将局部时钟、驱动缓存与已知空闲旋转变换到有效过程算子。其导出的`name='frame'`算子可能是一个已编译对角矩阵，不能再当成原始同名软件帧重复上传。实际拼接控制波形需要一致的全局参考；不能硬抄每块从零开始计算的驱动轴，再将帧重置。

## 5. R5/R6的具体上下行线路

### 5.1 R5 down

定义编码

```text
E = X01(I) → X12(I) → CNOT(C→I) → X12(I) → X01(I)
```

下行：

```text
E
→ V_down(I)
→ T(I→L)
→ V_down†(L)
→ X12(I)
→ T(I→R)
→ output frames on C,L,R
```

共5 CZ、19物理单门、18层、713.025368 ns。

理想高能级CZ旁路取恒等、V=I时，CI经过E变成`+00,−01,+10,−12`。最终四个合法输出的相位是`1,−1,1,i`。理想端点帧约为`(0,−π,+π/2)`；实际冻结角度见第7节，不能与理想值混用。

### 5.2 R5 up

```text
input frames on C,L,R
→ T(R→I)
→ X12(I)
→ V_up†(L)
→ T(L→I)
→ V_up(I)
→ reverse_same_cz(E)
```

资源与down相同。V_up由上行参数文件给出，不能默认等于V_down或其dagger。

### 5.3 R6 down

```text
CNOT(I→L)
→ X01(I) → X12(I)
→ CNOT(C→I)
→ X12(I) → X01(I)
→ CNOT(L→I)
→ X12(I)
→ CNOT(I→L)
→ CNOT(I→R)
→ CNOT(R→I)
→ output frames on C,L,R
```

共6 CZ、17物理单门、19层、769.396679 ns，没有paired V。

### 5.4 R6 up

```text
input frames on C,L,R
→ CNOT(R→I) → CNOT(I→R) → CNOT(I→L)
→ X12(I) → CNOT(L→I)
→ X01(I) → X12(I) → CNOT(C→I)
→ X12(I) → X01(I) → CNOT(I→L)
```

资源与R6 down相同。

### 5.5 reverse_same_cz的准确语义

反向遍历原门列表：01半门相位加π并折回主值；显式软件帧取负；01/12 π脉冲与CZ继续复用同一物理脉冲。

**这不是完整qutrit/ququart传播子的数学dagger。** 高能级CZ未必自逆，π脉冲在目标块与旁路间有相位区别。上行独立标定，并以实际down输出接实际up验收；不允许数值强行取U_down†来展示虚假的完美恢复。

完整名义门、显式帧、逐门缓冲轴见[机器可读四方向门表](Experiment/DistributedGroverNoDecoherence20261102/feasibility/native_operation_details.json)以及[CIRCUIT_DETAILS.md第12节](Experiment/DistributedGroverNoDecoherence20261102/CIRCUIT_DETAILS.md)。波形相位应以协议和JSON全精度为准。

## 6. R5的2态问题和V修正：为何只在受限入口成立

01空间CZ校准固定的是`χ11−χ10−χ01+χ00≈π`，不同时固定`χ21−χ20`。当含2态的线控制CNOT时，目标看到近似块

$$B=R_y(\pi/2)\operatorname{diag}(e^{i\chi_{20}},e^{i\chi_{21}})R_y(-\pi/2).$$

它可能把目标0旋转成`a|0>+b|1>`。这是目标人口变化，末端对角虚拟Z无法任意消掉b。真实脉冲还保留同激发扇区的振幅交换，不是仅用两个相位描述。

令左搬运`K=T(I→L)`。近似封闭、泄露小且L初始0时：

$$K|\psi,0\rangle\simeq|0,\psi\rangle,\qquad K|2,0\rangle\simeq|2\rangle M|0\rangle.$$

选V的第一列匹配`M|0>`归一化方向，可得到

```text
normal: |ψ,0> → V(I) → |Vψ,0> → K → |0,Vψ> → V†(L) → |0,ψ>
bypass: |2,0> → V(I) → |2,0>  → K → |2>M|0> → V†(L) → e^{iγ}|2,0>
```

V修目标旋转，端点帧处理剩余分支相位γ；漏出该块的振幅不被V恢复。这个配对需要L入口0、正常搬运接近理想、辅助块近似封闭，不能推广为任意未知输入的完整CSWAP修正。

V目标作用为`V⊕1`（012内允许公共全局相位）；实际由两个半门与两跃迁帧展开。若

$$V=e^{i\alpha}R_z(\beta)R_y(\theta)R_z(\gamma),$$

定义`Z̃(δ)=frame_01(−δ)→frame_12(δ/2)`，时间顺序为

```text
Z̃(γ) → Z̃(π/2) → X01/2(+π/2)
→ Z̃(θ) → X01/2(−π/2)
→ Z̃(−π/2) → Z̃(β) → frame_12(α)
```

末尾frame_12(α)保存01块与2态的相对相位，不可省略。V†由其矩阵重新分解。

## 7. 冻结参数及局域pulse模型

本轮复用[RouterPulseCalibration20261002](Experiment/RouterPulseCalibration20261002)已经保存的合成参考校准，没有重新对每个整树节点跑一遍真实器件校准。

| 角色 | 参考标签 | f01/GHz | 非谐性/GHz |
| --- | --- | --- | --- |
| C | q111 | 4.600 | −0.220 |
| I | q121 | 5.000 | −0.220 |
| L | q130 | 5.500 | −0.220 |
| R | q112 | 5.650 | −0.235 |

每位4能级；主积分步长0.1 ns。01门32 ns，12门40 ns。单门的幅度、失谐、DRAG和补偿位于[native_parameters.json](Experiment/RouterPulseCalibration20261002/results/native/native_parameters.json)。这些幅度是模拟器单位，不能直接解释为AWG电压。

Hamiltonian在公共旋转系中包含Duffing能级、交换耦合和微波驱动：

$$\frac H\hbar=2\pi\sum_q[(f_q-f_{rot})n_q+\frac{\alpha_q}2n_q(n_q-1)]
+2\pi\sum_{ab}g_{ab}(t)(a_a^\dagger a_b+a_a a_b^\dagger)
+\pi\sum_q[\widetilde\Omega_q(t)a_q^\dagger+\widetilde\Omega_q^*(t)a_q].$$

单门包络为`s(t)=[1−cos(2πt/T)]/2`，复DRAG包络为`A[s−iη s_dot/α]`。CZ用起落半余弦权重w同时调频和开选中边耦合，其余边保持0：

$$f_q(t)=f_q^0+w(t)(f_q^{CZ}-f_q^0),\qquad
g_{ab}(t)=\rho_{ab}\sqrt{f_a(t)f_b(t)}w(t)p.$$

三边ρ=0.0031，p=1。

校准失谐δ按每个脉冲的局部调制处理并重置；名义f01/f12载波参考才跟随绝对时钟。重建传播时，不能额外把δ乘整条线路的起始时间，否则会人为改变后续轴相位。

| CZ边 | 调谐线 | 目标频率/GHz | T/ns | 坡时间/ns |
| --- | --- | --- | --- | --- |
| C–I | I | 4.818937693 | 30.641834 | 5.725930 |
| I–L | L | 5.244278591 | 56.371311 | 11.894511 |
| I–R | R | 5.259764008 | 64.820457 | 16.500000 |

方向参数文件位于

```text
Experiment/RouterPulseCalibration20261002/results/lab_equivalent_runs/
  {r5,r6}/{down,up}/pulse-L4-shots1000/parameters.json
```

其中`pulse-L4-shots1000`是既有校准目录名，不表示本轮整机传播每点抽取1000个shots。

| 方案/方向 | φC | φL | φR | 放置 |
| --- | --- | --- | --- | --- |
| R5 down | 2.1368149331 | −0.1345071692 | 2.3481146096 | 输出 |
| R5 up | 2.1591588279 | −0.1532027867 | 2.3348394093 | 输入 |
| R6 down | 2.1390297917 | −0.1334499509 | 2.9569575892 | 输出 |
| R6 up | 2.1142720857 | −0.1438178304 | 3.0067455533 | 输入 |

R5的V矩阵以`[real,imag]`复数对编码；读法是`array[...,0]+1j*array[...,1]`。完整精度保存在原文件和`native_operation_details.json`。V_down与V_up近似为

$$V_{down}=\begin{pmatrix}
0.923048&-0.001145-0.384684i\\
0.274545-0.269459i&0.648521+0.656839i
\end{pmatrix},\quad
V_{up}=\begin{pmatrix}
0.922191&0.273149-0.273776i\\
0.000251-0.386734i&0.652412+0.651763i
\end{pmatrix}.$$

矩阵显示值已四舍五入，不用于重新传播。原up参数元数据声明其相干校准针对匹配down准备的往返；独立up结果是当前研究额外计算的检验。

## 8. 完整QRAM的宏观时间线路

假定QPU内部有A0–D、A1–D、A0–A1边，唯一跨片边为D–T0，叶端各有Lj–Mj访问边。

定义网关

```text
G(Aj) = S(Aj,D) → S(D,T0) → S(Aj,D)
```

对任意01输入，标签`(a,d,t)→(d,a,t)→(d,t,a)→(t,d,a)`，所以远程交换Aj和T0、恢复D。每个G含9个CZ，其中3个跨片。

### 8.1 一次query的全部19步

入口临时树全0，M为给定经典存储字；`||`表示并行。

| 步 | 时间顺序中的操作 | 作用 |
| --- | --- | --- |
| 1 | G(A0) | 高地址位到T0，D恢复 |
| 2 | S(T0,C0) | 保存高位，T0回0 |
| 3 | G(A1) | 低地址位到T0，D恢复 |
| 4 | root down | 低位送T1或T2 |
| 5 | S(T1,C1) &#124;&#124; S(T2,C2) | 保存低位，数据端口清空 |
| 6 | H(D)，按Y/2→X展开 | 数据编码 |
| 7 | S(D,T0) | 数据跨片入树 |
| 8 | root down | 数据分到子树 |
| 9 | left down &#124;&#124; right down | 到被选叶端 |
| 10 | 四个CZ(Lj,Mj)并行 | 固定存储访问 |
| 11 | left up &#124;&#124; right up | 返回子输入 |
| 12 | root up | 返回T0 |
| 13 | S(D,T0) | 数据跨片返回 |
| 14 | S(T1,C1) &#124;&#124; S(T2,C2) | 取出低位，子控制回0 |
| 15 | root up | 低位回T0 |
| 16 | G(A1) | 恢复A1 |
| 17 | S(T0,C0) | 取出高位，C0回0 |
| 18 | G(A0) | 恢复A0，临时树回0 |
| 19 | H(D)，按Y/2→X展开 | 数据解码 |

代码还包含`prepare`阶段32 ns空等待，计入总时长/层数，但没有实际准备门。

router调用总数为8：地址低位装载/卸载调用root两次，数据经三个节点down/up共六次；总计down四次、up四次。

### 8.2 固定存储访问的数学解释

将地址保存和数据下行记为理想W。数据H编码后的计算基标签为x，存储层有`n_Lj=x δ(j,a)`。四CZ共同只产生`(−1)^(x*m_a)`。返回后两侧H把Z读取变成X读取：

$$Q_m=H_DW^\dagger\left[\prod_j CZ(L_j,M_j)\right]WH_D.$$

该式是理想功能证明。实际表中入口H在地址加载后；理想地址加载恢复D，故可交换。物理传播严格按表，上行未被数学W†替换。

M=0010时，按A0A1D基序有

$$Q_{0010}=\operatorname{diag}(I_2,I_2,X,I_2),$$

只交换100与101。其他数据库同理由Q_m定义给出完整8×8目标矩阵。

### 8.3 完整整网原生层表

- [R5：283层逐层表](Experiment/DistributedGroverNoDecoherence20261102/feasibility/native_layers_r5.md)、[对应JSON](Experiment/DistributedGroverNoDecoherence20261102/feasibility/schedule_2_r5.json)。
- [R6：289层逐层表](Experiment/DistributedGroverNoDecoherence20261102/feasibility/native_layers_r6.md)、[对应JSON](Experiment/DistributedGroverNoDecoherence20261102/feasibility/schedule_2_r6.json)。

这些层表包含query与一次diffusion。层索引从0起；JSON内`operator_sha256`是有效算子的指纹，不能由指纹反推出矩阵；重建算子需调用原模型库。

## 9. 完整Grover线路

直接初始化外部入口`|+>_A0 |+>_A1 |−>_D`，运行上面的完整query，再按时间顺序执行：

```text
H(A0) || H(A1)       # 每个H展开为Y01/2→X01，共两个单门层
→ X01(A0) || X01(A1)
→ CZ(A0,A1)
→ X01(A0) || X01(A1)
→ H(A0) || H(A1)     # 再展开
→ 直接计算地址Z人口
```

扩散在01空间与`2|++><++|−I`只差公共负号。四个one-hot数据库每个都使用相同query和diffusion，理想一轮成功率为1。

每个整网case都保存query与diffusion后的快照，但“单目标搜索成功率”只对恰有一个1的数据库定义。其他数据库也能计算固定diffusion后的态，不应随便取argmax数据库当成四选一目标。

真实入口准备与读出开销未进入当前资源和误差。模拟没有在实验端做破坏性测量后再继续扩散；中间快照来自同一次相干传播。

## 10. 已完成的模拟与检查

| 层次 | 已做工作 | 结果或边界 |
| --- | --- | --- |
| 理想协议 | 128组(a,d,m)，256个H编码分支，2048次合法入口核对 | 最终辅助零、存储不变，误差约2.22e−16 |
| 理想原生整网 | R5/R6 × 16数据库，完整8×8复数query | 最大元素误差8.482556806932493e−15；四目标理想Grover为1 |
| 局域冻结pulse | down、独立合法入口up、实际down→up；16个CI输入 | 保留真实高能级相位与振幅 |
| 连续控制 | I=1，C角度0到π共101点 | 同一冻结波形的末态四线人口 |
| 2态机制 | 0000入口原生层边界P2、R5配对V消融 | 临时2态为设计；固定其他参数的消融不是全局无V不可能性证明 |
| 整网相干 | R5/R6 × 全部16数据库，共32 case | query的完整Choi/K0、11种入口样本、Grover后快照 |
| 算法快照 | 目标10的oracle前后、diffusion后地址态 | 符号标记到目标人口的过程 |
| 辅助恢复 | 每个case的真实辅助0概率；代表性case的11阶段人口 | 没有采用二值读出标签后选择 |
| 截断/步长 | 局域4能级dt=.1、4能级dt=.05、5能级dt=.05 | 冻结参数，不重新优化 |
| 张量收敛 | 主bond128，对M=0010两方案另跑bond64 | Fe差约2.18e−6与3.79e−5 |
| 有限扩展 | 1/2/3层资源；R5全1/单目标的相位query趋势 | 三层只覆盖两种数据库模式，未证明全部输入 |

整网不是将单节点平均保真度连乘；真实传播保留振幅、相位、辅助纠缠和泄露。

## 11. 关键数值与资源：用于版本核对

### 11.1 局域

| 方案/过程 | 16态平均原始重叠 | 合法四线过程Fe |
| --- | --- | --- |
| R5 down | 99.624410% | 99.569921% |
| R5 up | 99.574168% | 99.559563% |
| R5实际往返 | 99.322553% | 99.310178% |
| R6 down | 99.606820% | 99.551637% |
| R6 up | 99.568204% | 99.555984% |
| R6实际往返 | 99.444068% | 99.417102% |

down的16态指标比较CLR，up/往返比较CI；合法过程Fe使用完整四线目标像。二者定义不同。

R5固定其他参数移除V后，CI=11目标四线人口为84.844483%；保留配对为99.734035%。无配对合法Fe约95.737463%。

### 11.2 整网

| 指标 | R5 | R6 |
| --- | --- | --- |
| 16数据库平均原始query Fe | 0.9798401334131317 | 0.9830798423866871 |
| 四目标平均原始Grover成功率 | 0.9890066951971707 | 0.9872370775199624 |
| M=0101、Bell地址的原始GHZ态重叠 | 0.9847267698621853 | 0.9863443122452584 |
| 一轮Grover CZ总数 | 105 | 113 |
| 其中跨片CZ | 18 | 18 |
| 物理单门脉冲 | 288 | 272 |
| CZ层深 | 86 | 92 |
| 全调度层数，含入口等待 | 283 | 289 |
| query时长/ns | 11184.152208837388 | 11522.380071997126 |
| 含扩散总时长/ns | 11426.152208837386 | 11764.380071997126 |

总CZ可核对为`8×(5或6)+65`。额外65来自树控制SWAP 18、跨片SWAP 18、QPU中转SWAP 24、四存储CZ 4、diffusion 1。跨片已包含在总数，不再相加。

query本身为104/112 CZ；105/113含一次扩散。只有n=2线路额外加diffusion，n=1/3的资源/时间比较需统一口径。

## 12. 保存的数据文件与数组语义

主目录`STUDY/feasibility/`：

| 文件 | 内容 |
| --- | --- |
| manifest.json | 当前传播内核指纹及scope |
| pulse_sources.json | 器件、冻结原生参数、四方向参数、pulse模型的指纹 |
| cases/r5_0010.json等 | 单case的标签、指标、bond与数值诊断 |
| arrays/r5_0010.npz等 | 单case矩阵及状态；32组case |
| SUMMARY.json | 32个case的指标汇总 |
| metrics.json | 绘图脚本整理的平均值及局域/算法结果 |
| checks.json | 理想原生解、理想完整query、合法性、正性与收敛检查 |
| ideal_validation.json | 稀疏理想协议入口/恢复检查 |
| local.json、local.npz | 局域16态、K、连续扫描、层跟踪与配对消融 |
| tracking.npz | 代表性整网case的11个阶段末四能级人口 |
| pulse_convergence.json | 固定脉冲的能级与积分步长检查 |
| scaling.json | R5在n=1/3的两个数据库模式，n=2从case读取 |
| resources.json | n=1/2/3、R5/R6的完整资源 |
| schedule_n_scheme.json | 完整有效过程调度，矩阵以指纹表示 |
| figures/、figure_catalog.json | 25张PNG及矢量PDF与图注 |

### 12.1 arrays/中单个两层case

query输入样本顺序固定：

```text
0:00, 1:01, 2:10, 3:11, 4:0+, 5:1+, 6:+0, 7:+1, 8:++, 9:Bell, 10:phase
```

前9个为地址标签、D=0；Bell是地址`(|00>+|11>)/√2`且D=0；phase是均匀地址且D=−。不要将Bell当作单独四位字符串解析。

| key | shape | 语义 |
| --- | --- | --- |
| query_block | (11,2,8,8) | A0A1D的01计算块，未按块迹归一 |
| query_address | (11,2,4,4) | 先迹掉D全部四能级，再截取地址01块 |
| query_adiag | (11,2,16) | 两地址四能级全空间的对角人口 |
| query_norm | (11,2) | 限制AD计算空间之前的该分支总范数 |
| choi | (64,64) | 原始query的AD输出01 Choi块 |
| choi_aux0 | (64,64) | 真实辅助0分支的同一Choi块 |
| K0 | (8,8) | 真实辅助0且AD在01的含相位Kraus块 |
| ideal_U | (8,8) | 当前数据库的目标异或query |
| grover_block | (1,2,8,8) | phase输入经历diffusion后的AD块 |
| grover_address | (1,2,4,4) | Grover后的地址01块 |
| grover_adiag | (1,2,16) | Grover后的地址四能级全对角 |
| grover_norm | (1,2) | 两分支总范数 |

第二轴的两个元素仅为：

```text
0 = 原始：迹掉所有临时树末态
1 = 真实R_aux全0：只保留全部10个临时树位为|0>的分支
```

当前没有旧版“二值读出标签选择”第三轴。条件AD密度块应取

```python
rho_raw  = arrays['query_block'][k,0] / arrays['query_norm'][k,0]
rho_aux0 = arrays['query_block'][k,1] / arrays['query_norm'][k,1]
p_aux0   = arrays['query_norm'][k,1] / arrays['query_norm'][k,0]
```

若AD本身有泄露，截取块的迹可以小于1。不能改除以`trace(block)`，否则会额外后选择AD无泄露，改变指标；也不能将原始block除以接受率来伪造条件态。

### 12.2 Choi、K0与指标定义

归一化参考态为`|Φ8>=Σ_x|x,x>/√8`，目标向量采用

```python
v_goal = ideal_U.T.reshape(-1) / np.sqrt(8)
```

即向量化顺序是`[reference input, output]`，不要改成`U.reshape(-1)`而不同时更改数据约定。

$$F_e=v^\dagger Jv,\quad F_e^{(0)}=v^\dagger J_0v,\quad
P_0=\operatorname{Tr}J_0,\quad S=\operatorname{Tr}J.$$

条件Choi重叠是`Fe0/P0`，不是原始确定性门Fe。`K0`未逐列归一；完整通道还有其他辅助末态Kraus块。

误差分支分解为

$$1-F_e=(1-S)+[S-P_0-(F_e-F_e^{(0)})]+[P_0-F_e^{(0)}].$$

三项分别是AD计算空间外、辅助非零分支内不匹配、辅助零分支内不匹配。它不是“X12/片内CZ/跨片门/耗散”等物理来源预算，不能据此归因某个具体门。

特定输入的原始成功、条件成功、接受率、联合正确比例也必须分开。Choi平均接受率和Grover特定输入接受率不相同；平均后的条件值与接受率不能直接相乘代替平均联合量。

### 12.3 local.npz与tracking.npz

局域16初态顺序是

```text
00,01,0+,0-,10,11,1+,1-,+0,+1,++,+-,-0,-1,-+,--
```

- `{scheme}_down_rho`：(16,8,8)，CLR约化态的01块；`down_K`：(4,4)，完整四线合法映射。
- `{scheme}_{up,roundtrip}_rho`：(16,4,4)，CI约化态的01块；对应K为(4,4)。
- `{scheme}_sweep`：(101,16)，每行按C/I/L/R各四能级人口展开；`theta`：(101,)。
- `r5_trace`：(18,17)，`r6_trace`：(19,17)，首列时间ns，后16列为各线能级人口。
- `r5_without_pair_K`：(4,4)，固定其他参数的无配对消融。

局域约化块不自动等于“四线都无泄露条件态”。例如只截取CLR的01块，没有额外证明被迹掉的I无泄露。

tracking的`roles`为13个量子站点，`{scheme}_population`：(11,13,4)，`{scheme}_times`：(11,)。阶段为

```text
0 prepare
1 address_load
2 data_encode
3 data_load
4 data_down
5 memory_access
6 data_up
7 data_return
8 address_unload
9 data_decode
10 diffusion
```

这是阶段末人口，不是每个波形积分点的连续峰值。

## 13. 本独立包的运行命令

本包已收齐前文五个目录的必要源码、冻结参数、32个case的NPZ/JSON以及图和报告。运行统一从复现包根目录开始；无需原工作区或pyQCat。推荐Python 3.10，环境安装与完整选项见[README](README.md)。

```powershell
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py circuits --run-name circuits-v1
python reproduce.py ideal --run-name ideal-v1
python reproduce.py quick --run-name quick-v1
python reproduce.py full --run-name full-v1
```

`verify`核对包文件与冻结指纹；`circuits`重建四方向门序列、SVG和1–3层全网调度与实际算子；`ideal`检查全部32个理想query；`quick`实际重算局域脉冲及R5/R6代表性整网case；`full`从空运行目录重算完整32个case、局域与规模检查、理想验证和25张图。

新结果写入`runs/<run-name>/`，随包的历史结果保留为只读参考用途。默认第一次full不会复用历史32个case，续跑同名同配置任务会复用已经完成的新case。改变源码或参数时运行指纹检查拒绝混用旧任务。

可独立指定case：

```powershell
python reproduce.py case --scheme r5 --memory 0010 --run-name case-r5-v1
python reproduce.py case --scheme r6 --memory 0010 --run-name case-r6-v1
python reproduce.py plots --run-name figures-v1
```

默认基准使用本包冻结门库。`calibrate`是可选的重新标定演示，运行本地pulse模拟而非硬件；输出单独写入runs，并不自动替换基准门库。新校准与当前冻结校准不应混为一组数据。

下面仍可直接读取随包的参考数组：

```python
from pathlib import Path
import json
import numpy as np
base = Path('Experiment/DistributedGroverNoDecoherence20261102/feasibility')
meta = json.loads((base/'cases/r5_0010.json').read_text(encoding='utf-8'))
with np.load(base/'arrays/r5_0010.npz') as a:
    rho_A = a['query_address'][10,0] / a['query_norm'][10,0]
    rho_A0 = a['query_address'][10,1] / a['query_norm'][10,1]
    p0 = a['query_norm'][10,1] / a['query_norm'][10,0]
    raw = a['grover_address'][0,0,2,2].real / a['grover_norm'][0,0]
    cond = a['grover_address'][0,1,2,2].real / a['grover_norm'][0,1]
print(meta['process']['Fe'], p0, raw, cond)
```

报告源码也已包含。`python reproduce.py report --run-name report-v1`准备独立TeX工程并调用已安装的XeLaTeX或Tectonic；计算和画线路不要求安装TeX。源码无需Windows专有字体，输出PDF还会核对中文可复制性。

## 14. 原始论文规划的覆盖度与后续工作

原始规划文件为`source_documents/存算分离的分布式Grover算法演示规划.pdf`。用户后来明确将当前版本收敛为“线路可行性”，关闭未经依据的随机门错误和读出模型。

| 项目 | 当前状态 |
| --- | --- |
| 完整算法线路、16数据库、四目标单轮Grover | 已覆盖 |
| R5/R6局域pulse、V修正、相位与泄露、实际往返 | 已覆盖参考节点 |
| 全部节点共享频率的一套17位连续pulse模型 | 未完成 |
| 真实跨片、QPU与存储访问pulse过程 | 当前采用理想01过程；误差尚未独立评估 |
| 独立存储保持验证 | 当前按模型结构保持；没有含访问失真/串扰的动态验证与参考组 |
| 局域双叠加代表性密度矩阵、query按1/2/4地址分组 | 尚需把已有数据整理成图 |
| query/GHZ原始与辅助0条件态的并排对照 | 已保存相关条件数据，当前正文未完整展示 |
| 条件Grover完整4×4矩阵、25/50/100%基线 | 可从现有数据补画；已有条件指标表 |
| 按X12/片内CZ/跨片门/耗散/读出划分的物理源预算 | 当前相干分支预算不能替代它 |
| 跨片门退极化率扫描、真实统计误差与实验QST | 本版范围外，尚未完成 |
| 1–3层趋势 | 已有两个数据库模式；不是三层全输入或多轮验证 |

未来若补图，优先复用已有数组，不给确定性传播虚构shots误差条。条件选择始终使用明确定义的真实辅助0事件；实验二值标签选择还需单独读出模型。

## 15. 接手AI的工作约束与建议顺序

1. 先读本文与源码，核对scope、字序、门相位和矩阵行列约定，再解释结果。
2. 以`feasibility/`为当前数据源，阅读case标签与指纹，不按文件名猜当前噪声模型。
3. 从保存矩阵重算一个已知指标，核对第11节结果，确认数组与条件归一化理解正确。
4. 需要补图时，先读`feasibility_plots.py`及原始数组；改图和改物理模型分别记录版本。
5. 新门优化、接口pulse或噪声研究建立新目录/版本，保留当前基线；不要覆盖旧校准与case。
6. 变更线路、并行、单门合并、帧位置或方向后，重验合法入口、完整query复数矩阵及实际往返，不只看基态真值表。
7. 不直接将R5/R6用于接收端未知的任意四位输入，不把末端Z当作人口/泄露修复，也不将up当作已实现的数学dagger。
8. 保留原始、条件、接受率和联合量；不能靠按列归一、投影每个门回01或选择容易输入来提高报告数值。
9. 报告叙述标注“理想原生”“局域pulse”“整树模块过程”各自证据；当前高保真数字不升级为真实17位硬件保证。

本独立包已经包含交接需要的源码、冻结JSON与全部32个NPZ。交接时传递整个文件夹或最终ZIP；参考TeX工程包仅用于文档，实际数值入口统一为根目录的reproduce.py。

继续分析可先复用现有相干数组补图；新增接口pulse、误差模型或线路优化应另建版本，以保留当前可比较基线。
