---
title: "MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation"
method_name: "MuseVLA"
authors: [Xingyuming Liu, Ruichun Ma, Heyu Guo, Qixiu Li, Qingwen Yang, Lin Luo, Shiqi Jiang, Chenren Xu, Jiaolong Yang, Baining Guo]
year: 2026
venue: arXiv
tags: [vla, multimodal-sensing, dexterous-manipulation, adaptive-sensing, sensor-fusion, diffusion-policy, grounded-representation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17598
created: 2026-06-17
---

# 论文笔记：MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, Microsoft Research Asia |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[π0.md\|π₀]], [[π0.5.md\|π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17598) |

---

## 一句话总结

> MuseVLA 将热成像、声学、mmWave 雷达等多模态传感器作为"按需工具"，通过自适应传感器选择与统一的 Grounded Sensor Image 表征，实现灵巧手操作任务的多模态感知，平均成功率达 80.6%。

---

## 核心贡献

1. **自适应传感器选择机制**: [[VLA（Vision-Language-Action Model）|VLA]] 模型生成传感器 token 以动态选择当前任务所需传感模态，避免冗余传感器引入噪声并降低推理显存至 6.61GB。
2. **Grounded Sensor Image 统一表征**: 将异构传感器数据（热图、声学热图、雷达点云）通过语义分割蒙版叠加到 RGB 图像的任务相关区域，形成可被标准视觉编码器直接处理的统一输入。
3. **多传感器数据合成流水线**: 通过向现有 RGB 数据集注入物理属性关键词并叠加彩色编码蒙版，生成 9.6K 合成多传感器演示片段，有效缓解真实多传感器数据稀缺问题。

---

## 问题背景

### 要解决的问题

现实世界中的灵巧机器人操作任务（如按温度分拣饮品、根据声音找目标、透过遮挡识别隐藏物体）超出了 RGB 相机的感知能力，需要热感、声学、毫米波雷达等补充传感器。

### 现有方法的局限

- 现有 [[VLA（Vision-Language-Action Model）|VLA]] 模型（如 [[π₀.md|π₀]]、[[π0.5.md|π₀.₅]]）仅依赖 RGB 输入，无法处理需要超视觉感知的操作任务。
- 直接拼接原始传感器数据（Raw Sensor）引入大量噪声，且需要模态特定编码器，扩展新传感器时需要完整重训练。
- 同时处理所有传感器会增加不必要的计算开销（推理显存达 13.23GB）。

### 本文的动机

将传感器视为"按需工具"（on-demand tools）而非固定输入，仅在任务需要时激活对应传感器；同时通过将传感器数据"接地"到 RGB 图像的空间特定区域，形成统一的视觉表征，使现有视觉编码器无需修改即可处理多模态感知信息。

---

## 方法详解

### 模型架构

MuseVLA 采用 **条件生成** 架构，基于 [[VLA（Vision-Language-Action Model）|VLA]] 框架扩展多传感器感知能力：

- **输入**: 语言指令 $l$ + RGB 观测 $o_t$（基础输入）+ 按需选择的传感器数据 $s_{i,t}$
- **VLM Backbone**: [[Vision-Language Model|PaliGemma-2]]（SigLIP 视觉编码器 + Gemma-2 语言模型）
- **核心模块 1**: 传感器选择头，生成 [[传感器 Token|Sensor Token]] $l_s$ 和目标描述 $l_d$
- **核心模块 2**: [[SAM-3]] 语义分割 + [[Homography|单应矩阵]] 对齐，构建 Grounded Sensor Image $m_{i,t}$
- **动作头**: [[扩散Transformer|扩散 Transformer（DiT）]] 处理增强后的多模态输入，预测操作动作序列
- **预训练来源**: 基于 VITRA 权重初始化

整体推理流程：

1. VLM Backbone 接收 RGB 图像 + 指令，输出传感器 token $l_s$ 和目标描述 $l_d$
2. 调用对应传感器采集 $s_{i,t}$
3. SAM3 以 $l_d$ 为提示生成二值蒙版 $M$
4. 将传感器热图 $s_{i,t}$ 叠加至蒙版区域，形成 Grounded Sensor Image $m_{i,t}$
5. DiT 动作专家以 $(l, o_t, l_s, l_d, m_{i,t})$ 为条件生成动作序列 $A$

### 核心模块

#### 模块 1: 传感器 Token 生成（Adaptive Sensor Selection）

**设计动机**: 不同任务需要不同传感模态，利用 [[Vision-Language Model|VLM]] 的语言理解能力动态选择最合适的传感器，避免固定多传感器输入引入不相关噪声。

**具体实现**:
- 在 VLM 词汇表中新增四个可学习 token：`<None>`、`<Thermal>`、`<Acoustic>`、`<mmWave>`
- VLM 同时生成传感器 token $l_s$ 和目标对象的文本描述 $l_d$（如 "the hot drink"）
- 训练中通过 [[交叉熵|交叉熵损失]] 监督传感器 token 分类和目标描述生成

#### 模块 2: Grounded Sensor Image 构建

**设计动机**: 将各异构传感器的空间测量信息"接地"到 RGB 图像的任务相关区域，形成标准视觉编码器可直接消费的统一输入，无需引入模态特定编码器。

**具体实现**:
- 使用 [[SAM-3]] 以目标描述 $l_d$ 为提示，对 RGB 图像执行语义分割，得到二值蒙版 $M$
- 通过 [[Homography|单应矩阵]] $H_i$ 将传感器坐标系对齐到 RGB 图像坐标系（一次性离线标定）
- 在蒙版区域叠加传感器热图，非蒙版区域保留原始 RGB，形成 Grounded Sensor Image $m$

#### 模块 3: 数据合成流水线

**设计动机**: 真实多传感器机器人数据极度稀缺，利用现有大规模 RGB 演示数据集合成带物理属性标注的多传感器数据。

**具体实现**:
1. 从现有 RGB 数据集（MolmoAct、AgiBotWorld-Alpha、VITRA）随机采样演示片段
2. 随机选择传感模态及物理属性关键词（如 "hot"/"cold"），注入任务指令
3. 使用 [[Vision-Language Model|VLM]] 生成目标对象描述
4. 在分割出的目标区域叠加彩色编码蒙版（模拟传感器热图）
5. 最终生成 9.6K 合成演示片段（1.05M 帧，覆盖 1000+ 对象）

---

## 关键公式

### 公式 1: [[VLA（Vision-Language-Action Model）|任务定义]]

$$
\pi: (l,\, o_t,\, s_{1,t},\, \dots,\, s_{N,t}) \rightarrow A
$$

**含义**: 传统多传感器 VLA 策略将语言指令 $l$、RGB 观测 $o_t$、所有 $N$ 个传感器的观测 $s_{i,t}$ 作为固定输入，输出动作序列 $A$。

**符号说明**:
- $l$: 自然语言任务指令
- $o_t$: 时刻 $t$ 的 RGB 图像观测
- $s_{i,t}$: 时刻 $t$ 第 $i$ 个传感器（热像仪/麦克风/雷达）的观测
- $A$: 输出动作序列

### 公式 2: [[传感器 Token|自适应传感器解耦架构]]

$$
\begin{aligned}
\pi&: (l,\, o_t) \rightarrow (l_s,\, l_d) \\
G&: (o_t,\, s_{i,t},\, l_d) \rightarrow m_{i,t} \\
\pi&: (l,\, o_t,\, l_s,\, l_d,\, m_{i,t}) \rightarrow A
\end{aligned}
$$

**含义**: MuseVLA 将原始多传感器输入解耦为三步：首先 VLM 选择传感器并生成目标描述，然后构建 Grounded Sensor Image，最后以增强输入生成动作序列。

**符号说明**:
- $l_s$: 传感器选择 token（如 `<Thermal>`）
- $l_d$: 目标对象文本描述（如 "the hot drink"）
- $G$: Grounded Sensor Image 构建函数
- $m_{i,t}$: 时刻 $t$ 的 Grounded Sensor Image

### 公式 3: [[SAM-3|Grounded Sensor Image 构建]]

$$
\begin{aligned}
M &= f_{\text{seg}}(o_{\text{RGB}},\, l_d) \\
m &= M \odot s + (1 - M) \odot o_{\text{RGB}}
\end{aligned}
$$

**含义**: 以目标描述为提示对 RGB 图像进行语义分割得到二值蒙版 $M$，再将传感器热图 $s$ 按蒙版加权叠加到 RGB 图像，形成 Grounded Sensor Image $m$。

**符号说明**:
- $f_{\text{seg}}$: SAM3 语义分割函数
- $M$: 二值分割蒙版（蒙版内为 1，蒙版外为 0）
- $s$: 对齐到 RGB 坐标系后的传感器热图
- $\odot$: 逐元素乘法（Hadamard 积）

### 公式 4: [[Vision-Language Model|VLM 训练损失]]

$$
\mathcal{L}_{\text{VLM}} = \mathcal{L}_{\text{sensor}} + \mathcal{L}_{\text{target}}
$$

**含义**: VLM 的训练损失由两部分组成：传感器 token 分类的[[交叉熵]]损失与目标描述文本生成的[[交叉熵]]损失之和。

**符号说明**:
- $\mathcal{L}_{\text{sensor}}$: 传感器 token 预测的交叉熵损失
- $\mathcal{L}_{\text{target}}$: 目标对象描述的文本生成交叉熵损失

### 公式 5: [[扩散Transformer|联合 VLM-VLA 训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{VLM}} + \lambda \,\mathbb{E}_{\tau,\varepsilon}\bigl[\|\varepsilon - \varepsilon_\theta(a^\tau,\, \tau,\, c)\|_2^2\bigr]
$$

**含义**: 端到端联合训练目标将 VLM 损失与扩散动作专家的 MSE 去噪损失结合，$\lambda$ 控制两者权重平衡。

**符号说明**:
- $\lambda$: 损失平衡系数（实验中设为 $10^{-2}$）
- $\tau$: 扩散时间步（均匀采样自 $[0,1]$）
- $\varepsilon$: 随机噪声
- $\varepsilon_\theta$: 去噪网络预测的噪声
- $a^\tau$: 时间步 $\tau$ 的加噪动作序列
- $c$: 条件上下文（语言、RGB、传感器图像等）

### 公式 6: [[mmWave 雷达|数字波束成形（mmWave 热图生成）]]

$$
\begin{aligned}
P(\theta, \phi) &= 20 \log_{10} \left\| \sum_k a_k e^{j\varphi_k} e^{-j\Delta_k(\theta,\phi)} \right\| \\
\Delta_k(\theta,\phi) &= \frac{2\pi}{\lambda}\bigl(d_k^x \cos\phi \sin\theta + d_k^y \sin\phi\bigr)
\end{aligned}
$$

**含义**: 对毫米波雷达天线阵列的接收信号进行数字波束成形，在方位角 $\theta$、俯仰角 $\phi$ 方向上合成空间功率谱 $P(\theta,\phi)$，生成二维热图。

**符号说明**:
- $P(\theta, \phi)$: 方向 $(\theta, \phi)$ 上的功率（dB）
- $a_k, \varphi_k$: 第 $k$ 根天线的幅度和相位
- $\Delta_k(\theta,\phi)$: 第 $k$ 根天线相对于参考点的相位差
- $\lambda$: 雷达信号波长
- $d_k^x, d_k^y$: 第 $k$ 根天线在 $x, y$ 方向上的位置坐标

### 公式 7: [[Homography|单应矩阵传感器对齐]]

$$
(u,\, v,\, 1)^T \sim H_i\, (u_i,\, v_i,\, 1)^T
$$

**含义**: 利用单应矩阵 $H_i$ 将第 $i$ 个传感器坐标 $(u_i, v_i)$ 映射到 RGB 图像坐标 $(u, v)$，实现传感器数据与 RGB 的二维空间对齐。只需一次离线标定。

**符号说明**:
- $H_i$: 第 $i$ 个传感器到 RGB 相机的 $3 \times 3$ 单应矩阵
- $(u_i, v_i)$: 传感器坐标系中的像素坐标
- $(u, v)$: RGB 图像坐标系中的像素坐标

### 公式 8: [[Homography|传感器热图投影]]

$$
M_i(u_i,\, v_i) = M\!\left(\pi\!\left(H_i [u_i,\, v_i,\, 1]^T\right)\right)
$$

**含义**: 将 RGB 坐标系中的分割蒙版 $M$ 逆向映射到传感器坐标系，确定传感器热图中哪些区域对应目标对象，从而只提取相关区域的传感器信息。

**符号说明**:
- $M_i$: 传感器坐标系中的蒙版
- $\pi(\cdot)$: 齐次坐标到像素坐标的投影
- $M(\cdot)$: 查询 RGB 坐标系蒙版值的函数

---

## 关键图表

### Figure 1: Adaptive Multisensory Robotic Manipulation / 自适应多传感器操作概览

![Figure 1](https://arxiv.org/html/2606.17598v1/x1.png)

**说明**: 展示 MuseVLA 的目标场景。系统针对 RGB 之外需要多模态感知的操作任务，自适应选择合适的传感器（热成像、声学、mmWave），并生成目标描述构建 Grounded Sensor Image 来引导操作。

### Figure 2: MuseVLA 模型总览

![Figure 2](https://arxiv.org/html/2606.17598v1/x2.png)

**说明**: MuseVLA 完整架构。给定 RGB 图像和任务指令，VLM Backbone 生成传感器 token 和目标描述；选定的传感器被激活，构建 Grounded Sensor Image，附加到输入中用于[[扩散Transformer|扩散 Transformer]] 动作生成。整个系统在真实世界数据和合成多传感器数据上端到端联合训练。

### Figure 3: Grounded Sensor Image 处理流程

![Figure 3](https://arxiv.org/html/2606.17598v1/x3.png)

**说明**: 以目标描述为提示通过 [[SAM-3]] 进行语义分割，将传感器热图叠加到分割蒙版对应的 RGB 区域，形成视觉编码器可直接处理的统一表征。

### Figure 4: 数据合成流水线

![Figure 4](https://arxiv.org/html/2606.17598v1/x4.png)

**说明**: 通过向现有 RGB 数据集任务指令中注入物理属性关键词（如"hot"、"cold"），并在分割目标对象上叠加彩色编码蒙版，合成多传感器训练数据，生成 9.6K 演示片段。

### Figure 5: 评估硬件设置

![Figure 5a - Robot Setup](https://arxiv.org/html/2606.17598v1/x5.png)
![Figure 5b - Task Objects](https://arxiv.org/html/2606.17598v1/x6.png)

**说明**: 实验平台：桌面机械臂 + 12 自由度灵巧手 + 多传感器模块（RGB-D、双热成像相机、mmWave 雷达、麦克风阵列），以及涵盖三种传感模态的多类任务对象。

### Figure 6: 未见任务评估执行轨迹

![Figure 6](https://arxiv.org/html/2606.17598v1/x7.png)
![Figure 6](https://arxiv.org/html/2606.17598v1/x8.png)
![Figure 6](https://arxiv.org/html/2606.17598v1/x9.png)

**说明**: 在零样本未见任务（Radar→RGB、Radar→Thermal 多阶段任务）上的执行轨迹可视化，展示 MuseVLA 跨传感器链式推理能力。

### Figure 7: 训练任务评估执行轨迹

![Figure 7](https://arxiv.org/html/2606.17598v1/x10.png)
![Figure 7](https://arxiv.org/html/2606.17598v1/x11.png)
![Figure 7](https://arxiv.org/html/2606.17598v1/x12.png)
![Figure 7](https://arxiv.org/html/2606.17598v1/x13.png)
![Figure 7](https://arxiv.org/html/2606.17598v1/x14.png)

**说明**: 热成像引导拾放、声学目标定位、mmWave 雷达辅助隐藏物体检索三类训练任务的完整执行轨迹，MuseVLA 平均成功率 76.4%。

### Figure 8: 多阶段任务评估执行轨迹

![Figure 8](https://arxiv.org/html/2606.17598v1/x15.png)
![Figure 8](https://arxiv.org/html/2606.17598v1/x16.png)
![Figure 8](https://arxiv.org/html/2606.17598v1/x17.png)
![Figure 8](https://arxiv.org/html/2606.17598v1/x18.png)

**说明**: 展示 MuseVLA 处理异构多阶段任务（先 mmWave 检测再 RGB 开盖；先 mmWave 检测再 Thermal 选择）的执行轨迹，验证跨模态链式感知能力。

### Figure 9: 零样本未见任务评估执行轨迹

![Figure 9](https://arxiv.org/html/2606.17598v1/x19.png)
![Figure 9](https://arxiv.org/html/2606.17598v1/x20.png)
![Figure 9](https://arxiv.org/html/2606.17598v1/x21.png)
![Figure 9](https://arxiv.org/html/2606.17598v1/x22.png)
![Figure 9](https://arxiv.org/html/2606.17598v1/x23.png)
![Figure 9](https://arxiv.org/html/2606.17598v1/x24.png)

**说明**: 零样本未见任务（使用训练时未见过的新对象或新传感器组合）的执行轨迹，平均成功率 66.7%，验证模型的泛化能力。

### Table 1: 主实验——任务成功率对比

| Method | Thermal | Acoustic | mmWave | Average | Sensing Score | Manipulation Score | Task Score | p-value |
|--------|---------|----------|--------|---------|---------------|--------------------|-----------|---------|
| π₀-RGB | 33.3% | 25.0% | 4.17% | 20.8% | 48.6% | 43.1% | 0.458 | <0.001 |
| π₀.₅-RGB | 16.7% | 33.3% | 8.33% | 19.4% | 41.7% | 33.3% | 0.375 | <0.001 |
| π₀-Raw | 16.7% | 41.7% | 25.0% | 27.8% | 86.1% | 27.8% | 0.569 | <0.001 |
| π₀.₅-Raw | 16.7% | 33.3% | 20.8% | 23.6% | 83.3% | 29.2% | 0.563 | <0.001 |
| MuseVLA-RGB | 12.5% | 33.3% | 20.8% | 22.2% | 41.7% | 43.1% | 0.424 | <0.001 |
| MuseVLA-Raw | 41.7% | 25.0% | 33.3% | 33.3% | 91.7% | 33.3% | 0.625 | <0.001 |
| MuseVLA-RawAdapt | 70.8% | 41.7% | 66.7% | 59.7% | 93.1% | 59.7% | 0.764 | <0.05 |
| **MuseVLA (Ours)** | **83.3%** | **58.3%** | **87.5%** | **76.4%** | **95.8%** | **77.8%** | **0.868** | — |

**说明**: MuseVLA 相比 RGB-only 基线（π₀-RGB）提升 58%，相比 Raw Sensor 基线（π₀-Raw）提升 47%。统计显著性通过 Fisher 精确检验验证。

### Table 2: 自适应传感器选择精度消融

| Method | 训练任务-传感器 | 训练任务-目标描述 | 未见任务-传感器 | 未见任务-目标描述 |
|--------|----------------|------------------|----------------|------------------|
| PaliGemma-2（基础）| 0% | 13.0% | 0% | 9.5% |
| MuseVLA w/o pretrain | 100% | 100% | 85% | 40.5% |
| **MuseVLA（预训练）** | **100%** | **93.5%** | **100%** | **82.0%** |

**说明**: 合成数据预训练显著提升未见任务的传感器选择（85%→100%）和目标描述生成（40.5%→82.0%）准确率，验证数据合成流水线的有效性。

### Table 3: 预训练对合成数据的影响——见过与未见任务

| Method | 见过-Thermal | 见过-Acoustic | 见过-mmWave | 见过平均 | 未见-Thermal | 未见-Acoustic | 未见-mmWave | 未见平均 | p-value |
|--------|------------|--------------|------------|---------|-------------|--------------|------------|---------|---------|
| MuseVLA-Raw | 41.7% | 25.0% | 33.3% | 33.3% | 31.3% | 25.0% | 18.8% | 25.0% | <0.001 |
| MuseVLA w/o pretrain | 83.3% | 58.3% | 87.5% | 76.4% | 25.0% | 31.3% | 25.0% | 27.1% | <0.001 |
| **MuseVLA（预训练）** | **87.5%** | **70.8%** | **83.3%** | **80.6%** | **75.0%** | **56.3%** | **68.8%** | **66.7%** | — |

**说明**: 不使用合成预训练数据时，未见任务性能仅 27.1%（与 Raw 基线相当）；加入合成预训练后大幅提升至 66.7%，说明合成数据是泛化能力的关键。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实多传感器演示 | 720 条 | 覆盖 10 种任务变体、7 种对象、3 种传感模态 | 微调训练 |
| 合成多传感器数据 | 9.6K 条 / 1.05M 帧 | 从 MolmoAct、AgiBotWorld-Alpha、VITRA 合成 | 预训练 |

### 实现细节

- **Backbone**: VITRA 初始化（PaliGemma-2 + Gemma-2 + SigLIP + [[扩散Transformer|DiT]] 动作专家）
- **分割模型**: [[SAM-3]]（Segment Anything with Concepts）
- **优化器**: AdamW，学习率 $1 \times 10^{-5}$
- **Batch Size**: 512
- **训练步数**: 20K 步（约 20 小时）
- **损失权重**: $\lambda = 10^{-2}$
- **硬件**: 64 × A100（40GB）GPU
- **传感器模块**: RGB-D 相机、双热成像相机、mmWave 雷达、麦克风阵列（刚性联结）
- **遥操作设备**: MANUS5 灵巧控制手套

### 可视化结果

MuseVLA 在三类训练任务（热成像拾放、声学定位、mmWave 隐藏物体检索）上均表现出稳定的感知与操作能力，在多阶段链式任务（Radar→RGB：66.7%，Radar→Thermal：75.0%）上也展示出跨传感器组合推理能力。

---

## 批判性思考

### 优点

1. **传感器解耦设计优雅**: 将传感器视为工具而非固定输入，通过传感器 token 实现按需激活，显存从 13.23GB 降至 6.61GB，扩展新传感器时无需重训练整个模型。
2. **Grounded Sensor Image 统一表征巧妙**: 将异构传感器数据统一为 RGB 兼容的视觉表征，使现有视觉编码器无需修改即可处理多模态感知，工程复杂度低。
3. **数据合成流水线有效**: 9.6K 合成数据使未见任务成功率从 27.1% 提升至 66.7%，验证了低成本扩展多传感器数据的可行性。

### 局限性

1. **真实数据规模较小**: 仅 720 条真实演示，任务种类和对象多样性有限，泛化能力依赖合成数据质量。
2. **依赖分割模型效果**: 整个流水线高度依赖 SAM3 的分割精度，当目标对象在 RGB 图像中难以分割时（如透明物体、遮挡场景）性能可能退化。
3. **标定成本**: 虽然单应矩阵标定是一次性的，但更换传感器位置或添加新传感器时需要重新标定。
4. **传感器种类受限**: 目前仅验证了热成像、声学、mmWave 三类传感器，触觉、嗅觉等其他模态的整合未被探索。

### 潜在改进方向

1. 扩展到更多传感器类型（触觉、力矩、嗅觉）并验证扩展性。
2. 探索端到端学习单应矩阵，减少人工标定依赖。
3. 提高真实数据规模并引入更多任务多样性。
4. 研究当 SAM3 分割失败时的鲁棒性策略（如置信度感知的 fallback 机制）。

### 可复现性评估

- [ ] 代码开源（未提及）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（backbone、优化器、学习率、硬件均有报告）
- [ ] 数据集可获取（真实演示数据未开放，合成流水线有描述）

---

## 关联笔记

### 基于

- [[π₀.md|π₀]]: VLA 基础 backbone，MuseVLA 对比基线之一
- [[π0.5.md|π₀.₅]]: 另一 VLA 基础 backbone，MuseVLA 对比基线之一
- [[SAM-3]]: 目标对象语义分割，Grounded Sensor Image 的关键组件
- [[扩散Transformer|扩散 Transformer（DiT）]]: 动作专家，负责生成操作动作序列

### 对比

- [[π₀.md|π₀-RGB]] / [[π0.5.md|π₀.₅-RGB]]: RGB-only VLA 基线，说明 RGB 单独无法处理超视觉感知任务
- [[π₀.md|π₀-Raw]] / [[π0.5.md|π₀.₅-Raw]]: 原始传感器拼接基线，说明未接地的传感器数据噪声大

### 方法相关

- [[VLA（Vision-Language-Action Model）|VLA]]: 核心框架
- [[Vision-Language Model|VLM]]: 用于传感器选择和目标描述生成
- [[Homography|单应矩阵]]: 传感器坐标对齐
- [[传感器 Token|Sensor Token]]: 自适应传感器选择的关键设计

### 硬件/数据相关

- [[mmWave 雷达]]: 毫米波雷达传感器，用于隐藏物体检测
- [[热成像相机]]: 热感传感器，用于温度区分任务
- [[麦克风阵列]]: 声学传感器，用于声源定位任务

---

## 速查卡片

> [!summary] MuseVLA
> - **核心**: 将多模态传感器作为"按需工具"，通过 Sensor Token 自适应选择传感器 + Grounded Sensor Image 统一表征
> - **方法**: VLM 生成传感器 token 和目标描述 → SAM3 分割 → 热图叠加 RGB → DiT 生成动作
> - **结果**: 训练任务 80.6% / 未见任务 66.7% 平均成功率；相比 RGB-only 基线提升 58%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-17*
