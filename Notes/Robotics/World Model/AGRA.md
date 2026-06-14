---
title: "Making Foresight Actionable: Repurposing Representation Alignment in World Action Models"
method_name: "AGRA"
authors: [Lu Qiu, Yizhuo Li, Yi Chen, Yuying Ge, Yixiao Ge, Xihui Liu]
year: 2026
venue: arXiv
tags: [world-action-model, representation-alignment, robot-manipulation, video-diffusion, action-grounding]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.12217v1
created: 2026-06-11
---

# 论文笔记：Making Foresight Actionable — AGRA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong, XPENG Robotics |
| 日期 | June 2026 |
| 项目主页 | [https://xpeng-robotics.github.io/agra](https://xpeng-robotics.github.io/agra) |
| 对比基线 | [[World-Action Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12217) |

---

## 一句话总结

> AGRA 通过将视频扩散中间特征与冻结 DINOv2 编码器的空间语义表示对齐，解决了世界动作模型中"视觉预测准确但动作提取失败"的动作基础差距问题。

---

## 核心贡献

1. **诊断 Action-Grounding Gap**: 通过注意力分析和因果干预，发现 [[World-Action Model|WAM]] 动作解码头无法聚焦于任务相关交互区域，且对背景扰动敏感
2. **提出 AGRA 目标函数**: 用 [[REPA|表示对齐]] 将视频扩散中间特征与 [[DINOv2]] 空间语义特征对齐，作为世界-动作接口的正则化器
3. **真实机器人验证**: 在 IRON-R01-1.11 人形机器人上实现 80% 成功率（对比基线 34%），OOD 场景提升 27-32%

---

## 问题背景

### 要解决的问题

[[World-Action Model|世界动作模型（WAM）]] 先通过视频扩散预测未来场景，再从中提取动作控制信号。然而实验发现：**视觉预测准确并不保证动作提取准确**——即使模型能生成合理的未来帧，动作解码器也可能输出错误的控制信号。

### 现有方法的局限

- [[Video Diffusion Model|视频扩散模型]] 的隐藏状态针对像素重建优化，特征与低层外观表示深度耦合
- 动作解码器的 [[Cross-Attention]] 注意力图显示它经常忽略任务关键的手-物交互区域
- 对背景扰动（swap/shuffle 无关区域）敏感，鲁棒性差

### 本文的动机

[[DINOv2]] 等基础视觉编码器的特征具有空间相干的语义组织——它能够区分任务相关区域和背景。将视频扩散的中间特征对齐到这种语义特征空间，可以在不替换视觉预测机制的情况下，使动作解码器的输入更具"可操作性"。

---

## 方法详解

### 模型架构

AGRA 基于 **双 DiT（Dual Diffusion Transformer）** 架构构建：

- **输入**: 语言指令 $c$ + 参考观测 $o_0$ + 机器人状态 $s_0$
- **Video DiT Backbone**: [[Cosmos]] Predict-2.5-2B（28 层），通过 [[Flow Matching]] 预测未来视频帧
- **核心对齐模块**: AGRA 从 Video DiT 第 $l_k$ 层提取中间隐藏态，经 [[Cross-Attention]] 投影为语义特征，与 [[DINOv2]] 目标对齐
- **Action DiT**: 8 层 [[Diffusion Transformer]]（500M 参数），通过多层 [[Cross-Attention]] 桥接世界模型特征以解码动作
- **输出**: [[Action Chunking|动作块]] $\hat{a}_{1:K} \in \mathbb{R}^{K \times d_a}$

### 核心模块

#### 模块 1: Action-Grounding Gap 诊断

**设计动机**: 在提出解决方案前，用两种方式定量证明问题存在。

**注意力分析**:
- 对动作 DiT 的 [[Cross-Attention]] 图进行可视化，计算注意力落在手-物交互掩码内的比例（Attention in Mask Ratio）和注意力中心到交互区域的距离（Centroid Error）
- 基线 WAM 注意力分散且中心偏移，AGRA 显著集中于交互区域

**因果干预**:
- 对参考帧的不同区域（任务相关区 vs 背景）施加三种扰动：mean/zero/shuffle/swap
- 用 Matched Sensitivity Ratio 量化模型对"正确区域"扰动的响应程度
- AGRA 的 Mean/Zero 干预比率更高，说明它更敏感于任务相关区域

#### 模块 2: AGRA 对齐目标函数

**设计动机**: 利用 [[REPA]] 技术——原本用于改善生成质量——重新定义为 **动作基础接口正则化器**。

**具体实现**:

1. **目标特征构建**: 对参考帧 $\bar{o}_t$ 用冻结 [[DINOv2]] 编码器提取特征，并插值到视频潜变量分辨率：

$$
\mathbf{Y}_t = g_\psi(\bar{o}_t) \in \mathbb{R}^{H_d \times W_d \times d_d}
$$

$$
\tilde{Y}_t = \text{Interp}(\mathbf{Y}_t; H_v, W_v) \in \mathbb{R}^{H_v \times W_v \times d_d}
$$

2. **Video DiT 中间层特征投影**:

$$
\mathbf{H}^{vid}_{l_k^{agra}} = f^{vid}_{\theta, l_k^{agra}}(\mathbf{z}^{\tau_v}_{1:T_v}, o_0, c, \tau_v)
$$

$$
\mathbf{Z}_k = P_k(\mathbf{H}^{vid}_{l_k^{agra}}) \in \mathbb{R}^{T_v \times H_v \times W_v \times d_d}
$$

3. **余弦相似度损失**（AGRA Loss）:

$$
\mathcal{L}_{AGRA} = -\frac{1}{KT_vH_vW_v} \sum_{k=1}^{K}\sum_{t=1}^{T_v}\sum_{u=1}^{H_v}\sum_{v=1}^{W_v} \cos(\mathbf{Z}_{k,t,u,v}, \tilde{Y}_{t,u,v})
$$

其中 $\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2 + \epsilon}$

#### 模块 3: 多层桥接（Multi-Layer Bridge）

**设计动机**: 单层对齐效果有限，均匀选取 Video DiT 多层特征传给 Action DiT。

**层选择策略**: 在 Video DiT（共 $M$ 层）中均匀选取 $N$ 个对齐层：

$$
l_j = \left\lfloor \frac{j(M-1)}{N-1} \right\rceil, \quad j = 0, \ldots, N-1
$$

- 实验表明最优位置在约 1/3 网络深度（第 8 层，共 28 层）
- 过深的层会破坏几何和动态细节；语义对齐应分配在浅层

#### 模块 4: 训练解耦策略（Separate Forward Pass）

**设计动机**: 避免训练/推理不一致——训练时动作解码器无需等待完整视频扩散结果。

Action DiT 训练采用固定高噪声级别 $\tau_a = 1$，与推理时一致：

$$
a^{\tau_a}_{1:K} = (1-\tau_a)a_{1:K} + \tau_a \epsilon_a, \quad \epsilon_a \sim \mathcal{N}(0, I)
$$

---

## 关键公式

### 公式 1: [[Flow Matching|视频流匹配预测目标]]

$$
\mathbf{z}^{t_v}_{1:T_v} = (1-\tau_v)\mathbf{z}_{1:T_v} + \tau_v \epsilon_v, \quad \epsilon_v \sim \mathcal{N}(0, \mathbf{I})
$$

$$
\mathbf{v}^{vid}_\theta = f^{vid}_\theta(\mathbf{z}^{\tau_v}_{1:T_v}, o_0, c, \tau_v)
$$

**含义**: 在噪声级别 $\tau_v$ 下对视频潜变量加噪，训练模型预测从噪声到数据的流方向

**符号说明**:
- $\mathbf{z}_{1:T_v}$: 干净视频潜变量序列
- $\tau_v \sim \mathcal{U}(0,1)$: 随机噪声级别
- $\epsilon_v$: 高斯噪声
- $\mathbf{v}^{vid}_\theta$: 预测的流方向

### 公式 2: [[Flow Matching|视频预测损失]]

$$
\mathcal{L}_{vid} = \mathbb{E}_{\mathbf{z}, \epsilon_v, \tau_v} \left[ \left\| f^{vid}_\theta(\mathbf{z}^{\tau_v}_{1:T_v}, o_0, c, \tau_v) - (\epsilon_v - \mathbf{z}_{1:T_v}) \right\|^2_2 \right]
$$

**含义**: Flow Matching 损失，最小化预测流与真实流之间的 L2 距离

### 公式 3: [[Cross-Attention|动作解码器跨注意力]]

$$
\mathcal{G}_j = \text{Proj}_j(\mathbf{H}^{vid}_{l_j}) \in \mathbb{R}^{N_v \times d_{act}}
$$

$$
\mathbf{X}^{act}_j = \text{CrossAttn}_j(Q=\mathbf{X}^{act}_{j-1}, K=\mathcal{G}_j, V=\mathcal{G}_j)
$$

**含义**: Action DiT 的第 $j$ 层通过跨注意力从视频特征中提取与动作相关的信息

**符号说明**:
- $\mathcal{G}_j$: 投影后的世界模型特征
- $N_v$: 视频特征 token 数
- $d_{act}$: Action DiT 的特征维度

### 公式 4: [[DINOv2]] 目标特征提取与插值

$$
\mathbf{Y}_t = g_\psi(\bar{o}_t) \in \mathbb{R}^{H_d \times W_d \times d_d}
$$

$$
\tilde{Y}_t = \text{Interp}(\mathbf{Y}_t; H_v, W_v) \in \mathbb{R}^{H_v \times W_v \times d_d}
$$

**含义**: 用冻结 DINOv2 编码器提取参考帧的空间语义特征，并插值到视频分辨率

### 公式 5: [[AGRA|AGRA 对齐损失]]

$$
\mathcal{L}_{AGRA} = -\frac{1}{KT_vH_vW_v} \sum_{k=1}^{K}\sum_{t=1}^{T_v}\sum_{u=1}^{H_v}\sum_{v=1}^{W_v} \cos(\mathbf{Z}_{k,t,u,v}, \tilde{Y}_{t,u,v})
$$

$$
\cos(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2 + \epsilon}
$$

**含义**: 最小化投影后视频扩散特征与 DINOv2 目标特征之间的负余弦相似度，实现空间语义对齐

**符号说明**:
- $K$: 对齐层数量
- $T_v, H_v, W_v$: 视频时间、高度、宽度维度
- $\mathbf{Z}_{k,t,u,v}$: 第 $k$ 层投影后的视频特征
- $\tilde{Y}_{t,u,v}$: 插值后的 DINOv2 目标特征

### 公式 6: [[WAM|完整训练目标]]

$$
\mathcal{L}_{WAM} = \mathcal{L}_{vid} + \lambda_{act}\mathcal{L}_{act}
$$

$$
\mathcal{L} = \mathcal{L}_{WAM} + \lambda_{agra}\mathcal{L}_{AGRA}
$$

**含义**: 总损失 = 视频预测损失 + 动作预测损失 + AGRA 对齐正则项

**符号说明**:
- $\lambda_{act}$: 动作损失权重
- $\lambda_{agra}$: AGRA 损失权重

### 公式 7: [[Flow Matching|动作预测损失]]

$$
v^{act}_\phi = f^{act}_\phi(a^{\tau_a}_{1:K}, s_0, \tau_a, \mathcal{G})
$$

$$
\mathcal{L}_{act} = \mathbb{E}_{a, \epsilon_a, \tau_a} \left[ \left\| f^{act}_\phi(a^{\tau_a}_{1:K}, s_0, \tau_a, \mathcal{G}) - (\epsilon_a - a_{1:K}) \right\|^2_2 \right]
$$

**含义**: Action DiT 的流匹配损失，在固定 $\tau_a = 1$ 下训练

---

## 关键图表

### Figure 1: 问题动机与方法概览

![Figure 1](https://arxiv.org/html/2606.12217v1/x1.png)

**说明**: 展示基线 WAM 的注意力漂移问题与 AGRA 改进后的任务相关区域聚焦对比。基线 WAM 动作解码头注意力分散到背景，AGRA 使注意力集中于手-物交互区域。

### Figure 2: Action-Grounding Gap 诊断

![Figure 2](https://arxiv.org/html/2606.12217v1/x2.png)

**说明**: 通过 [[Cross-Attention]] 图可视化和因果干预热图，定量展示基线 WAM 的注意力失焦问题。左侧注意力图显示基线模型关注背景，右侧因果干预分析揭示模型对背景扰动同样敏感。

### Figure 3: DINOv2 vs. Cosmos 特征的 PCA 对比

![Figure 3](https://arxiv.org/html/2606.12217v1/x3.png)

**说明**: PCA 可视化对比 [[DINOv2]] 的空间相干语义组织与 [[Cosmos-Predict2.5|Cosmos]] 的外观纠缠特征。DINOv2 特征能清晰分离任务相关区域，而 Cosmos 中间特征与低层像素外观高度耦合，动机了 AGRA 对齐方向。

### Figure 4: 真实世界评估结果总览

![Figure 4](https://arxiv.org/html/2606.12217v1/x4.png)

**说明**: 在 IRON-R01-1.11 人形机器人上的 ID 和 OOD 场景综合评估结果，包含注意力分析定量指标。AGRA 在 ID 场景达到 80% 成功率，各 OOD 类别提升 27-32%。

### Figure 5: AGRA 注意力集中效果定性分析

![Figure 5](https://arxiv.org/html/2606.12217v1/x5.png)

**说明**: 定性展示 AGRA 如何引导动作解码头注意力聚焦到任务关键的手-物交互区域，对比基线模型的注意力漂移。

### Figure 6: 真实机器人执行案例

![Figure 6](https://arxiv.org/html/2606.12217v1/x6.png)

**说明**: 展示 AGRA 在真实操作任务（抓取稳定性、物体 Affordance 理解）上的改进案例，包括 Pick-and-Place 和 Open-Steamer-Transfer-Bun 等复杂操作任务。

### Figure 7: 基线 WAM 与 AGRA 架构对比图

![Figure 7](https://arxiv.org/html/2606.12217v1/x7.png)

**说明**: 双 DiT 架构示意图。左侧为基线 WAM，右侧为加入 AGRA 的版本。Video DiT 的中间层隐藏态经投影层与 [[DINOv2]] 特征对齐，对齐后的特征通过 [[Cross-Attention]] 桥传递给 Action DiT。

### Figure 8: RoboCasa GR1 Benchmark 评估

![Figure 8](https://arxiv.org/html/2606.12217v1/x8.png)

**说明**: RoboCasa GR1 模拟环境的基准评估，包含代表性策略执行轨迹示例。AGRA 整体达到 66.4%，比 GR00T 高出 18.8%。

### Figure 9: RoboCasa-GR1 完整数据集评估结果

![Figure 9](https://arxiv.org/html/2606.12217v1/x9.png)

**说明**: 使用完整 24,000 条轨迹训练时，跨 24 个桌面操作任务的评估结果。包含与 UWM、Diffusion Policy、FLARE、DiT4DiT、LDA-1B、GR00T-N1.6 的对比。

### Figure 10: 层选择对下游操作性能的影响

![Figure 10](https://arxiv.org/html/2606.12217v1/x10.png)

**说明**: 分析不同 [[Cosmos-Predict2.5|Cosmos]] 层（1-28 层）对下游操作任务性能的影响。实验表明约 1/3 深度（第 8 层）效果最佳；AGRA-DinoL8 优于 AGRA-DinoL15；AGRA-SiglipL8 表明 DINOv2 优于 SigLIP 作为对齐目标；多层对齐（AGRA-DinoL4/8/12）优于单层配置。

### Figure 11: EgoDex 跨具身迁移实验

![Figure 11](https://arxiv.org/html/2606.12217v1/x11.png)

**说明**: 添加人类第一视角数据集 EgoDex 进行跨具身训练时，基线 WAM 几乎无提升，而 AGRA 显著受益（尤其 OOD 场景）。说明 AGRA 的对齐特征具有跨具身迁移能力，能有效利用人类演示数据。

### Table 1: Matched Sensitivity Ratio（因果干预分析）

| 干预类型 | WAM 基线 | AGRA |
|---------|---------|------|
| Mean | 8.41 | **10.31** |
| Zero | 1.95 | **2.82** |
| Shuffle | 0.99 | 1.01 |
| Swap w/ Comp | 8.36 | **10.19** |

**说明**: 高 Matched Sensitivity Ratio 意味着模型对任务相关区域扰动更敏感。AGRA 在 Mean/Zero/Swap 三种干预下比率更高，说明动作解码器更关注任务相关区域；Shuffle 接近 1 表明两者对无关区域扰动都不太敏感。

### Table 2: RoboCasa GR1 消融实验（Few-Shot, 100 轨迹/任务）

| 变体 | ID | Unseen App. | Unseen Obj. Types | Unseen Comb. |
|-----|-----|------------|-------------------|--------------|
| Freeze backbone | 52.89 | 49.33 | 43.87 | 51.42 |
| WAM 基线 | 58.41 | 53.77 | 43.18 | 59.57 |
| AGRA-SharedDenoisingPass | 58.24 | 57.77 | 41.87 | 58.57 |
| AGRA-VideoPass | 60.41 | 55.33 | 44.56 | 60.14 |
| **AGRA-ActionCondPass（完整）** | **61.75** | **56.55** | **45.31** | **66.28** |

**关键发现**: 
- Action-Conditioned Pass（动作条件前向传播）最优，比基线 WAM 在 OOD（Unseen Comb.）上提升 6.71 个百分点
- Shared Denoising Pass 在 ID 场景甚至略低于基线（训练/推理不一致问题）
- 完整 AGRA 在数据稀缺（few-shot）下 OOD 提升最显著

---

## 实验

### 数据集与平台

| 平台/数据集 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| IRON-R01-1.11 | 真实人形机器人 | Pick-and-Place + 复杂操作任务 | 真实世界验证 |
| RoboCasa GR1 Full | 24,000 条轨迹（1000/任务） | 24 任务，29 DoF | 模拟主实验 |
| RoboCasa GR1 Few-Shot | 2,400 条轨迹（100/任务） | 数据稀缺场景 | 消融实验 |

**任务分类**:
- 18 个 Pick & Place 重排任务
- 6 个关节交互任务（开/关门等）
- OOD 测试：Unseen Appearance / Unseen Object Types / Unseen Combinations

### 实现细节

- **Video Backbone**: [[Cosmos-Predict2.5|Cosmos]] Predict-2.5-2B（28 层 [[Diffusion Transformer]]）
- **Action Head**: 8 层 [[Diffusion Transformer]]（500M 参数）
- **输入分辨率**: 视频 192×336，DINOv2 输入 448×448
- **去噪步数**: 视频 1 步（$\tau_v = 1$），动作 4 步
- **训练目标**: [[Flow Matching]] 用于视频和动作分支

### 主要实验结果

**真实世界（IRON-R01-1.11）**:
- ID 成功率: AGRA **80%** vs. WAM 基线 **34%**（提升 2.35×）
- OOD-Semantic: +27%
- OOD-Instance Level: +32%
- OOD-Attribute: +32%

**模拟（RoboCasa GR1 Full）**:
- AGRA: **66.4%**
- GR00T: 47.6%（差距 +18.8%）

---

## 批判性思考

### 优点

1. **问题诊断严谨**: 用注意力分析 + 因果干预双重手段定量证明 action-grounding gap 存在，而非单纯凭直觉提出方法
2. **优雅地复用现有技术**: AGRA 将 [[REPA]] 对齐技术"重新定向"为接口正则化器，不改变视觉预测机制，对原 WAM 改动最小
3. **真实机器人验证**: 不止做仿真实验，在人形机器人上展示了显著提升（34% → 80%）

### 局限性

1. **DINOv2 依赖**: 对齐目标依赖 [[DINOv2]] 的质量；若任务涉及 DINOv2 不擅长的场景（如极端遮挡），效果未知
2. **计算开销**: 额外的 AGRA 前向传播（Video DiT + 多个投影层）增加训练成本，论文未定量报告开销
3. **超参数敏感性**: 最优对齐层（第 8 层）和层数 $K$ 的选择依赖消融实验，泛化到其他架构（非 Cosmos）的鲁棒性待验证

### 潜在改进方向

1. **动态层选择**: 根据任务复杂度自适应选择对齐层，而非固定选第 8 层
2. **多编码器对齐**: 除 DINOv2 外，引入 [[Affordance]] 专用编码器提供更细粒度的交互区域先验
3. **推广到语言-动作模型**: 探索在 VLA 架构（而非专用 WAM）中引入类似接口正则化

### 可复现性评估

- [ ] 代码开源（项目主页已列出，但截至撰写时未确认代码发布）
- [ ] 预训练模型（未提供）
- [x] 训练细节基本完整（网络规模、输入分辨率、去噪步数均已说明）
- [x] 数据集可获取（RoboCasa 公开可用）

---

## 关联笔记

### 基于

- [[World-Action Model]]: AGRA 的基础框架，先预测未来视觉场景再提取动作
- [[REPA]]: 原始表示对齐技术（用于改善生成质量），AGRA 将其重新定向为接口正则化器
- [[Cosmos-Predict2.5|Cosmos]]: 作为 Video DiT 的视频扩散 Backbone

### 对比

- [[Fast-WAM]]: 同为 WAM 改进工作，专注推理效率
- [[HiMem-WAM]]: WAM 系列，关注记忆增强

### 方法相关

- [[DINOv2]]: 提供空间语义对齐目标的冻结基础视觉编码器
- [[Flow Matching]]: 视频和动作分支的流匹配训练框架
- [[Diffusion Transformer]]: Video DiT 和 Action DiT 的核心架构
- [[Cross-Attention]]: Action DiT 从世界模型特征提取信息的机制
- [[Action Chunking]]: 动作输出格式，预测 $K$ 步动作序列

### 硬件/数据相关

- [[RoboCasa]]: 模拟基准数据集，24 个桌面操作任务

---

## 速查卡片

> [!summary] Making Foresight Actionable: AGRA
> - **核心**: 视觉预测准确 ≠ 动作提取准确；WAM 存在 action-grounding gap
> - **方法**: AGRA 损失将 Video DiT 中间层特征与冻结 DINOv2 语义特征对齐
> - **结果**: 真实机器人 80% vs. 34%（基线），OOD 提升 27-32%
> - **代码**: [xpeng-robotics.github.io/agra](https://xpeng-robotics.github.io/agra)

---

*笔记创建时间: 2026-06-11*
