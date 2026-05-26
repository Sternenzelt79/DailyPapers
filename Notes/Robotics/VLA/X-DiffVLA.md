---
title: "X-DiffVLA: X-Embodied Diffusion Action Heads for Vision-Language-Action Models"
method_name: "X-DiffVLA"
authors: [Boyu Li, Chaoyi Xu, Haoqi Yuan, Xinrun Xu, Börje F. Karlsson, Dongbin Zhao, Haoran Li, Zongqing Lu]
year: 2026
venue: arXiv
tags: [cross-embodied, vla, diffusion-policy, dexterous-manipulation, robot-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.25044
created: 2026-05-26
---

# 论文笔记：X-DiffVLA: X-Embodied Diffusion Action Heads for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | SKL-MAIS, Institute of Automation, Chinese Academy of Sciences; UCAS; Beijing Academy of AI; BeingBeyond; Peking University |
| 日期 | May 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[pi0|π0]], [[Pi05|π0.5]], [[X-VLA]], [[UniVLA]], [[XR-1]], [[GR00T-N1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.25044) |

---

## 一句话总结

> X-DiffVLA 通过扩散动作头结合体现强制（EBF）和形态树扩散（MPTD），实现跨异构机器人末端执行器的统一策略学习与知识迁移。

---

## 核心贡献

1. **统一跨体现扩散动作头**: 基于 [[Being-H0]] 2B 骨干，建立涵盖机械臂底座、并联夹爪和灵巧手的统一动作空间，低自由度系统以零填充对齐
2. **体现强制（EBF）**: 将形态先验注入去噪过程，通过全局结构感知噪声初始化（体现特定噪声幅度 $\sigma_e$）和局部功能感知空间金字塔噪声（$\boldsymbol{\Sigma}_{e,f}$）实现跨体现判别
3. **形态树扩散（MPTD）**: 将 [[MCTD|Monte Carlo Tree Diffusion]] 扩展到异构末端执行器，通过元动作节点和值函数引导识别跨体现行为相关性，促进知识迁移

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 在跨异构机器人体现（heterogeneous embodiments）上的泛化能力严重受限：每当机器人形态变化时，都需要从头微调新的动作头，无法利用不同体现之间共享的任务知识。

### 现有方法的局限

- **体现专用微调**: 如 [[pi0|π0]]、[[GR00T-N1]] 等需要针对每种末端执行器独立微调，导致孤立的任务-末端执行器绑定
- **分布干扰**: 联合训练并联夹爪和灵巧手数据时，因动作维度和结构差异导致次优性能
- **跨体现知识浪费**: 不同体现之间存在根本相似性（如抓取的共性），但现有方法无法利用

### 本文的动机

不同机器人末端执行器在执行相同任务时共享底层行为规律（如"抓取"动作的空间轨迹）。通过扩散模型的多峰分布建模能力，可以同时捕捉体现特定的判别特征和跨体现的行为相关性，从而在单一模型中实现异构体现的统一控制。

---

## 方法详解

### 模型架构

X-DiffVLA 采用 **扩散动作头 + 预训练 VLM** 架构：

- **输入**: 视觉观测 $o_t$ + 语言指令 $l$ + 本体感知信息 $p_t$
- **Backbone**: [[Being-H0]] 2B（预训练 VLM），输出条件 token $n_t$
- **核心模块**: [[EBF|体现强制（EBF）]] 用于形态先验注入 + [[MCTD|MPTD]] 用于跨体现知识迁移
- **输出**: [[Action Chunking|动作块]] $\tau_t = [a_t, a_{t+1}, \ldots, a_{t+H}]$，维度统一为最大体现的自由度数
- **精度**: BF16 后训练

**统一动作空间**: 建立涵盖机器人底座（base）、夹爪（gripper）和灵巧手（dexterous hand）的最大维度标准化表示。低自由度系统（如并联夹爪）在末尾维度用**零填充**对齐到统一空间。

### 核心模块

#### 模块1: 体现强制 Embodied Forcing (EBF)

**设计动机**: 利用 [[CFG|Classifier-Free Guidance]] 机制，将形态学先验嵌入 [[Diffusion Model|扩散模型]] 的去噪过程，使模型在多体现联合训练时保持对各体现的判别能力。

**具体实现**——两层噪声初始化策略：

**（1）全局结构感知噪声初始化**:

为每种体现 $e$ 分配特定的噪声幅度 $\sigma_e$，反映不同机器人形态的整体运动特性。在前向扩散中引入体现特定缩放：

$$
\tau_t^k = \sqrt{1-\beta_k}\tau_t^{k-1} + \sqrt{\beta_k} \cdot \sigma_e \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

**（2）局部功能感知空间金字塔噪声**:

根据各关节组件距末端执行器的**空间接近程度**分配差异化噪声幅度，利用对角协方差矩阵 $\boldsymbol{\Sigma}_{e,f}$，按 $\delta$ 系数衰减：

$$
\tau_t^k = \sqrt{1-\beta_k}\tau_t^{k-1} + \sqrt{\beta_k} \cdot \boldsymbol{\Sigma}_{e,f} \epsilon
$$

$$
\boldsymbol{\Sigma}_{e,f} = \text{diag}(\sigma_e,\ \delta\sigma_e,\ \delta^2\sigma_e,\ \ldots)
$$

其中窗口大小 $W_n = 4$、$\delta = 0.5$ 时性能最优。

---

#### 模块2: 形态树扩散 Morphological Tree Diffusion (MPTD)

**设计动机**: 扩展 [[MCTD|MCTS]] 范式到跨体现去噪，通过树结构搜索识别不同末端执行器之间的行为相关性，引导扩散去噪向高价值动作轨迹方向进行。

**具体实现**——四阶段树搜索过程：

**（1）选择（Selection）**: 从形态树的白噪声出发，扩散头可以自去噪，或以外部体现的动作序列作为条件信号

**（2）扩展（Expansion）**: 从选定节点生成候选动作段，以 EBF 噪声初始化为条件进行采样

**（3）模拟（Simulation）**: 对扩展路径执行快速跳跃去噪（jump denoising），评估节点价值

**（4）反向传播（Backpropagation）**: 将模拟步骤获得的价值分数传播回树层次结构

**节点价值函数**: 衡量生成动作 $\mathbf{x}$ 与参考元动作 $\mathbf{y}$ 的距离，支持同维度和异维度体现的比较：

$$
v = \begin{cases}
-\|\mathbf{x}-\mathbf{y}\|^2 + M_0, & \text{if } \text{dim}(\mathbf{x})=\text{dim}(\mathbf{y}) \\
-\|\mathbf{x}_{1:d_{\min}}-\mathbf{y}_{1:d_{\min}}\|^2 + M_1, & \text{if } \text{dim}(\mathbf{x})\neq\text{dim}(\mathbf{y})
\end{cases}
$$

其中 $M_1 > M_0$，以激励异质体现之间的知识迁移，$d_{\min}$ 为两体现的最小功能维度。

---

#### 模块3: 体现软提示 Soft Prompts for Embodied Feature

**设计动机**: 识别扩散模型对潜在提示空间特别敏感，利用可学习的体现特定参数来吸收跨体现异质性，进一步增强体现判别。

**具体实现**: 学习参数集合 $P^E = \{p^i\}_{i=1}^e$，每种体现对应独立的软提示向量，在推理时与 VLM 输出的条件 token $n_t$ 拼接，为扩散头提供体现标识信息。

---

## 关键公式

### 公式1: [[Diffusion Model|前向扩散过程]]

$$
q(\mathbf{x}^k|\mathbf{x}^{k-1}) = \mathcal{N}(\mathbf{x}^k;\ \sqrt{1-\beta_k}\mathbf{x}^{k-1},\ \beta_k\mathbf{I})
$$

**含义**: 前向扩散过程，逐步向数据添加高斯噪声

**符号说明**:
- $\mathbf{x}^k$: 第 $k$ 步扩散后的数据
- $\beta_k$: 第 $k$ 步噪声调度系数
- $k$: 扩散时间步（从 0 到 $K$）

---

### 公式2: [[Diffusion Model|反向去噪过程]]

$$
p_\theta(\mathbf{x}^{k-1}|\mathbf{x}^k) = \mathcal{N}(\mathbf{x}^{k-1};\ \boldsymbol{\mu}(\mathbf{x}^k, k),\ \gamma_k\mathbf{I})
$$

**含义**: 参数化反向去噪过程，从噪声恢复数据分布

**符号说明**:
- $\boldsymbol{\mu}(\mathbf{x}^k, k)$: 网络预测的均值
- $\gamma_k$: 反向过程方差
- $\theta$: 网络参数

---

### 公式3: [[Action Chunking|动作块]]定义

$$
\tau_t = [a_t,\ a_{t+1},\ \ldots,\ a_{t+H}]
$$

**含义**: 时间步 $t$ 处预测的 $H$ 步动作序列块

**符号说明**:
- $a_t$: 时间步 $t$ 的动作
- $H$: 预测步长（horizon）

---

### 公式4: [[EBF|体现强制]] — 全局结构感知噪声

$$
\tau_t^k = \sqrt{1-\beta_k}\tau_t^{k-1} + \sqrt{\beta_k} \cdot \sigma_e \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

**含义**: 在标准扩散噪声中引入体现特定幅度 $\sigma_e$，使不同体现的噪声分布产生差异化

**符号说明**:
- $\sigma_e$: 体现 $e$ 对应的噪声幅度系数
- $\epsilon$: 标准高斯噪声
- $\tau_t^k$: 第 $k$ 扩散步的动作块

---

### 公式5: [[EBF|体现强制]] — 局部功能感知空间金字塔噪声

$$
\tau_t^k = \sqrt{1-\beta_k}\tau_t^{k-1} + \sqrt{\beta_k} \cdot \boldsymbol{\Sigma}_{e,f} \epsilon
$$

$$
\boldsymbol{\Sigma}_{e,f} = \text{diag}(\sigma_e,\ \delta\sigma_e,\ \delta^2\sigma_e,\ \ldots)
$$

**含义**: 对关节维度施加基于空间位置的差异化噪声，末端执行器附近关节获得更大噪声幅度

**符号说明**:
- $\boldsymbol{\Sigma}_{e,f}$: 体现 $e$ 的功能感知对角协方差矩阵
- $\delta$: 噪声衰减系数（最优值 0.5）
- $f$: 功能组（按距末端执行器的接近程度划分，窗口大小 $W_n=4$）

---

### 公式6: [[MCTD|形态树扩散]] — 节点价值函数

$$
v = \begin{cases}
-\|\mathbf{x}-\mathbf{y}\|^2 + M_0, & \text{if } \text{dim}(\mathbf{x})=\text{dim}(\mathbf{y}) \\
-\|\mathbf{x}_{1:d_{\min}}-\mathbf{y}_{1:d_{\min}}\|^2 + M_1, & \text{if } \text{dim}(\mathbf{x})\neq\text{dim}(\mathbf{y})
\end{cases}
$$

**含义**: 衡量生成动作与参考元动作的相似度，值越高表示动作越合理；异维度时取共享最小维度比较以实现跨体现知识迁移

**符号说明**:
- $\mathbf{x}$: 生成的候选动作序列
- $\mathbf{y}$: 参考元动作（meta-action node）
- $M_0, M_1$: 基础价值（$M_1 > M_0$，激励异质体现迁移）
- $d_{\min}$: 两体现的最小功能维度

---

### 公式7: [[Diffusion Policy|扩散头]]训练目标

$$
\hat{\tau}_t = x_{0,\theta}(\tau_t^k,\ p_t,\ n_t,\ k,\ e,\ f)
$$

$$
\mathcal{L} = \text{MSE}(x_{0,\theta}(\tau_t^k,\ p_t,\ n_t,\ k,\ e,\ f),\ \tau_t)
$$

**含义**: 扩散头以噪声动作块、本体感知、VLM token、扩散步、体现标识、功能标识为条件，预测干净动作块，用 MSE 监督

**符号说明**:
- $x_{0,\theta}$: 参数化扩散头
- $p_t$: 本体感知信息（proprioception）
- $n_t$: VLM 输出的条件 token
- $k$: 扩散步编号
- $e$: 体现标识（embodiment ID）
- $f$: 功能组标识

---

## 关键图表

### Figure 1: 动机对比图

![Figure 1](https://arxiv.org/html/2605.25044/x1.png)

**说明**: X-DiffVLA 的核心动机对比。左侧展示体现专用后训练方式：VLA 模型针对不同末端执行器（夹爪、灵巧手）分别进行孤立的 [[迁移学习|domain-specific fine-tuning]]，导致策略无法互用。右侧展示 X-DiffVLA：通过 [[跨体现学习]] 架构，利用多样化数据实现从专用专家到统一通用机器人控制器的跨越。

---

### Figure 2: X-DiffVLA 架构总览

![Figure 2](https://arxiv.org/html/2605.25044/x2.png)

**说明**: X-DiffVLA 完整架构。框架利用统一动作空间和 [[Diffusion Policy|扩散动作头]] 对多体现的多峰动作分布建模。关键组件：（1）[[EBF|体现强制（EBF）]] — 将形态先验集成到去噪过程，增强体现特定判别；（2）[[MCTD|形态树扩散（MPTD）]] — 捕捉不同末端执行器间的行为相关性，促进跨体现知识迁移。

---

### Figure 3: T-SNE 特征可视化

![Figure 3](https://arxiv.org/html/2605.25044/x3.png)

**说明**: Isaac Gym 中三种体现（Panda、Inspire、Shadow）末端执行器最终关节位置的 [[T-SNE]] 可视化（50 次试验）。X-DiffVLA 的特征聚类相比基线更加清晰分离，验证了 EBF 在跨体现判别上的有效性。

---

### Figure 4: 真实机器人实验可视化

![Figure 4](https://arxiv.org/html/2605.25044/x4.png)

**说明**: 真实世界实验结果。Inspire 灵巧手成功执行从并联夹爪数据中学习的策略（黄色标注），展示 X-DiffVLA 通过跨体现数据集实现知识迁移的能力——无需灵巧手专用演示数据即可完成抓取任务。

---

### Table 1: RoboCasa 环境主要对比结果

| 方法 | Panda | Robotiq-85 | Inspire | 平均 | 移动失败 | 交互失败 |
|------|-------|-----------|---------|------|---------|---------|
| UniVLA | 39.5% | 37.0% | 35.1% | 37.2% | 889 | 241 |
| XR-1 | 45.2% | 43.8% | 34.9% | 41.3% | 863 | 193 |
| X-VLA | 50.1% | 46.9% | 45.8% | 47.6% | 816 | 127 |
| GR00T-N1 | 39.6% | 40.4% | 38.5% | 39.5% | 854 | 235 |
| π₀+FAST | 44.3% | 45.8% | 39.2% | 43.1% | 820 | 204 |
| π₀ | 45.1% | 45.2% | 37.8% | 42.7% | 834 | 197 |
| π₀.₅ | 52.3% | 51.1% | 44.2% | 49.2% | 784 | 130 |
| **X-DiffVLA** | **67.2%** | **68.0%** | **58.3%** | **64.5%** | **550** | **89** |

**说明**: X-DiffVLA 在所有三种末端执行器上均大幅领先，平均成功率 64.5%，较最强基线 π₀.₅ 提升 **15.3%**。同时显著减少移动失败和交互失败次数。

---

### Table 2: Isaac Gym 跨体现抓取结果

| 方法 | Panda | Inspire | Shadow | 平均 |
|------|-------|---------|--------|------|
| X-VLA | 56.5% | 57.0% | 58.0% | 57.2% |
| π₀ | 55.5% | 52.5% | 54.0% | 54.0% |
| π₀.₅ | 59.5% | 58.5% | 57.5% | 58.5% |
| **X-DiffVLA** | **73.5%** | **69.5%** | **70.0%** | **71.0%** |

**说明**: Isaac Gym 中 10 类物体抓取任务，三种灵巧手均大幅提升，平均成功率 71.0%，较最强基线提升 **12.5%**。

---

### Table 3: 真实机器人实验

| 方法 | Panda 夹爪 | Inspire 灵巧手 | 平均 |
|------|-----------|--------------|------|
| X-VLA | 51.0% | 47.0% | 49.0% |
| π₀ | 55.0% | 47.0% | 51.0% |
| π₀.₅ | 59.0% | 55.0% | 57.0% |
| **X-DiffVLA** | **67.0%** | **60.0%** | **63.5%** |

**说明**: FR3 机械臂真实场景验证（5 类物体 × 20 次试验），X-DiffVLA 实现 63.5% 平均成功率，验证夹爪到灵巧手的知识迁移有效性。

---

### Table 4: 消融研究 (RoboCasa)

| 配置 | Panda | Robotiq-85 | Inspire | 平均 |
|------|-------|-----------|---------|------|
| **X-DiffVLA (完整)** | **67.2%** | **68.0%** | **58.3%** | **64.5%** |
| w/o SP（无软提示） | 66.1% | 65.4% | 55.4% | 62.3% |
| w/o MPTD（无形态树扩散） | 56.9% | 58.3% | 47.1% | 54.1% |
| w/o EBF（无体现强制） | 46.7% | 48.3% | 35.5% | 43.5% |
| w/o ALL（纯扩散头基线） | 41.2% | 39.4% | 41.8% | 40.8% |

**关键发现**: EBF 贡献最大（去除后下降 21.0%），MPTD 次之（去除后下降 10.4%），二者叠加效果显著超过各自单独效果，软提示提供少量额外增益（+2.2%）。

---

### Table 5: EBF 在扩散头 vs. 流匹配头上的效果

| 方法 | Panda | Robotiq-85 | Inspire | 平均 |
|------|-------|-----------|---------|------|
| 流匹配头（FH） | 43.3% | 43.1% | 38.1% | 41.5% |
| FH + EBF | 42.5% | 44.7% | 40.6% | 42.6% |
| 扩散头（DH） | 41.2% | 39.4% | 41.8% | 40.8% |
| **DH + EBF** | **56.9%** | **58.3%** | **47.1%** | **54.1%** |

**关键发现**: EBF 对扩散头增益显著（+13.3%），对流匹配头效果微弱（+1.1%），说明 EBF 的噪声初始化机制与扩散模型的随机去噪特性高度契合。

---

### Table 6: 超参数消融

| $\delta$ 值 | 平均成功率 | 最优标记 |
|------------|----------|---------|
| 0.9 | 55.7% | |
| **0.5** | **64.5%** | ✓ |
| 0.1 | 37.5% | |

| $W_n$ 值 | 平均成功率 | 最优标记 |
|---------|----------|---------|
| 9 | 43.8% | |
| **4** | **64.5%** | ✓ |
| 3 | 43.0% | |

**说明**: $\delta=0.5$（中等衰减）和 $W_n=4$（4 维窗口分组）为最优超参数配置。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboCasa | 30 任务/体现 × 50 轨迹 | 3 种末端执行器（Panda、Robotiq-85、Inspire） | 后训练 + 测试（20 次/任务） |
| Isaac Gym | 10 类物体 × 10 轨迹/体现 | 3 种灵巧手（Panda、Inspire、Shadow） | 仿真测试 |
| 真实场景 | 5 类物体 × 10 轨迹/体现 | FR3 机械臂 + 并联夹爪 / Inspire 灵巧手 | 真实验证（20 次/任务） |

### 实现细节

- **Backbone**: [[Being-H0]] 2B（预训练 VLM）
- **精度**: BF16 后训练
- **最优超参数**: $\delta=0.5$，$W_n=4$
- **数据收集工具**: GELLO、Manus 数据手套
- **动作空间**: 统一最大维度，低 DoF 系统零填充对齐

### 可视化结果

T-SNE 分析（Figure 3）显示三种体现在 Isaac Gym 中的末端执行器关节位置聚类清晰分离，验证 EBF 成功学到了体现特异性表征。真实机器人实验（Figure 4）中，Inspire 灵巧手能够执行从并联夹爪数据中迁移的抓取策略。

---

## 批判性思考

### 优点
1. **清晰的模块化设计**: EBF 和 MPTD 分别解决"体现判别"和"知识迁移"两个独立问题，消融实验验证各模块贡献
2. **跨硬件泛化**: 首次在仿真和真实环境中同时验证了夹爪→灵巧手的知识迁移，具有实用意义
3. **扩散模型契合性**: EBF 利用扩散模型随机性的内在优势，在流匹配头上几乎无效，说明方法设计有针对性

### 局限性
1. **Backbone 依赖性强**: 基于 Being-H0 2B，未在其他 VLM 骨干（如 OpenVLA、π0）上验证迁移性
2. **数据规模偏小**: 每种任务仅 50 条后训练轨迹，训练数据规模在工业级应用中偏少
3. **超参数敏感**: $\delta$ 和 $W_n$ 对性能影响极大（从 37.5% 到 64.5%），但缺乏自适应超参数机制

### 潜在改进方向
1. 研究 EBF 中 $\sigma_e$ 的自适应学习，而非人工设定不同体现的噪声幅度
2. 将 MPTD 扩展到在线学习场景，利用实时执行反馈更新树结构
3. 探索零样本新体现适应能力（zero-shot embodiment adaptation）

### 可复现性评估
- [ ] 代码开源（论文中未提及）
- [ ] 预训练模型（未提供）
- [ ] 训练细节完整（部分缺失，如 GPU 型号）
- [x] 数据集可获取（RoboCasa、Isaac Gym 均为公开仿真平台）

---

## 关联笔记

### 基于
- [[Being-H0]]: 使用其 2B 参数 VLM 作为骨干
- [[Diffusion Policy]]: 扩散动作头的基础框架
- [[MCTD]]: MPTD 的前身方法

### 对比
- [[Pi05|π0.5]]: 最强基线，RoboCasa 提升 15.3%
- [[X-VLA]]: 跨体现软提示方法，被 MPTD 超越
- [[UniVLA]]: 潜在动作表示方法
- [[GR00T-N1]]: NVIDIA 通用机器人 VLA

### 方法相关
- [[EBF|Embodied Forcing]]: 本文提出的核心组件之一
- [[MCTD|MPTD]]: 本文提出的形态树扩散
- [[Action Chunking]]: 输出动作序列块的范式
- [[CFG|分类器无关引导]]: EBF 的理论基础

### 硬件/数据相关
- [[RoboCasa]]: 主要仿真基准
- [[Isaac Gym]]: 灵巧手仿真环境
- [[Inspire Hand]]: 灵巧手硬件平台

---

## 速查卡片

> [!summary] X-DiffVLA
> - **核心**: 跨体现扩散 VLA，统一并联夹爪和灵巧手的策略学习
> - **方法**: EBF（体现特定噪声初始化）+ MPTD（形态树扩散跨体现迁移）
> - **结果**: RoboCasa +15.3%，Isaac Gym +12.5%，真实机器人 63.5%
> - **代码**: 未开源

---

*笔记创建时间: 2026-05-26*
