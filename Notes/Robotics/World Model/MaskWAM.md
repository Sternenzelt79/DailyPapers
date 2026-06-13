---
title: "MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models"
method_name: "MaskWAM"
authors: [Hanyang Yu, Haitao Lin, Jingbo Zhang, Wenyao Zhang, Chenghao Gu, Heng Li, Ping Tan]
year: 2026
venue: arXiv
tags: [world-action-model, mask-prediction, visual-prompting, robotic-manipulation, flow-matching]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.13515
created: 2026-06-12
---

# 论文笔记：MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The Hong Kong University of Science and Technology, Tencent Robotics X, Tsinghua University |
| 日期 | June 2026 |
| 项目主页 | [MaskWAM Project Page](https://hanyangyu1021.github.io/maskwam.github.io/) |
| 对比基线 | [[FastWAM]], [[π0]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13515) |

---

## 一句话总结

> MaskWAM 将 Mask 作为输入提示（视觉锚点）与预测目标（语义监督）统一到 [[World Action Model]] 中，同时解决空间定位弱和语言歧义两大瓶颈，在仿真与真机任务中均达到 SOTA。

---

## 核心贡献

1. **统一 Mask 输入与预测**: 将未来帧 Mask 预测与首帧 Mask 提示统一在单一 [[Mixture of Transformers]] 架构中，Mask 既作为视觉提示消除语言歧义，又作为预测监督信号提供对象中心语义约束
2. **双重收益设计**: [[Future Mask Prediction|未来 Mask 预测]]提供对象中心语义监督（抑制背景视觉噪声），首帧 Mask 提示提供空间锚点（精确指定目标对象）
3. **SOTA 实验性能**: 在 LIBERO（98.4%）、RoboTwin 2.0（92.2%）和真机任务上均超越现有最优方法，语言歧义场景下尤为显著

---

## 问题背景

### 要解决的问题

[[World Action Model]]（WAM）通过联合预测未来视频帧和动作轨迹进行机器人控制，但现有方法存在两大空间瓶颈：
1. **弱空间定位**: 纯 RGB 预测缺乏对任务相关区域的语义监督，容易被背景噪声干扰
2. **语言歧义**: 文本指令（如"pick up the cup"）在杂乱场景中无法精确区分相似对象（如多个不同颜色的杯子）

### 现有方法的局限

- [[VLA|视觉-语言-动作模型]] 直接端到端学习，语言指令无法提供精确空间指向
- 基于坐标文本提示（如 "object at (x, y)"）的方法：稀疏坐标与空间语义理解差距大，仅能提供粗糙定位（消融实验中仅 18.2% 成功率 vs MaskWAM 的 84.9%）
- 已有 WAM（如 [[FastWAM]]、[[Motus]]）仅预测 RGB 帧，缺乏对象中心的结构监督

### 本文的动机

通过将 [[Instance Segmentation Mask|实例分割 Mask]] 作为模型输入与输出的双重角色：
- 输入侧：首帧目标 Mask 提供密集空间-语义先验，精确告知"操作哪个对象"
- 输出侧：强制预测未来帧 Mask 迫使模型关注任务相关区域，获得对象中心语义监督

---

## 方法详解

### 模型架构

MaskWAM 采用 **[[Mixture of Transformers]]（MoT）** 统一架构：
- **输入**: 语言指令 $\ell$ + 初始观测帧 $I_0$ + 首帧目标 Mask $M_0$（可选）+ 机器人状态 $s_0$
- **Backbone**: 冻结 [[Video VAE]] 编码器（处理 RGB 与 Mask 的联合潜空间）
- **核心模块**: [[Diffusion Transformer|DiT]] 联合去噪 RGB 潜变量、Mask 潜变量和动作
- **输出**: 未来 RGB 帧 $I_{1:T}$ + 未来 Mask $M_{1:T}$ + [[Action Chunking|动作块]] $a_{1:K}$
- **训练框架**: [[Flow Matching]] 解耦噪声调度

### 核心模块

#### 模块1: 统一 RGB 与 Mask 编码

**设计动机**: 利用 [[Video VAE]] 共享潜空间实现稳定的通道级融合，避免分辨率不匹配与表示差异

**具体实现**:
- Mask 渲染为 RGB 兼容的三通道图像（复制到三通道），与 RGB 帧一同送入冻结的 [[Video VAE]] 编码器
- RGB 潜变量 $z_v$ 与 Mask 潜变量 $z_m$ 在通道维度拼接：$z = [z_v; z_m]$
- [[Diffusion Transformer|DiT]] Patch Embedding 从 $C$ 通道扩展为 $2C$ 通道（Mask 对应通道初始化为零，保留预训练权重）

**为何不用简单下采样？** 直接下采样（最近邻/双线性）缺乏语义先验，VAE 编码统一利用预训练视觉特征，实验证明 VAE 编码显著优于两种下采样备选

#### 模块2: 首帧 Mask 作为策略条件（视觉提示）

**设计动机**: 提供比语言指令更精确的对象空间定位，消除语言歧义场景中的不确定性

**具体实现**:
- 首帧目标 Mask $M_0 \in \{0,1\}^{H \times W}$ 由 [[SAM-3]] 分割模型获取（真机部署时需用户一次性点击提示，无需实时跟踪）
- 训练时以 $p=0.5$ 的概率随机 **Mask Dropout**，使模型同时兼容"有 Mask 提示"（语言歧义场景）和"无 Mask 提示"（语言清晰场景）两种模式
- $M_0$ 经过相同 VAE 编码后，与初始帧 $I_0$ 的潜变量拼接输入条件流

**为何不用 Zero-initialized Gating（类 ControlNet）？** 可学习门控 $g$ 初始化为零时，模型易学到忽略 Mask 信号；直接监督强制模型利用 Mask 特征，消融实验验证此设计必要性

#### 模块3: 联合 Flow Matching 目标

**设计动机**: 通过解耦但同步的噪声调度实现 RGB、Mask、动作的统一去噪，支持推理时的部分去噪加速策略

**具体实现**:
- RGB 与 Mask 共享同一噪声调度 $\tau_v$（保持几何一致性）
- 动作使用独立噪声调度 $\tau_a$
- 推理时仅对 RGB-Mask 流执行**单步部分去噪**（partial denoising），避免完整视频生成的计算开销
- 结合 KV-Cache 高效生成动作序列

---

## 关键公式

### 公式1: [[联合生成目标|联合概率分布]]

$$
p_\theta(a_{1:K}, I_{1:T}, M_{1:T} \mid I_0, M_0, s_0, \ell)
$$

**含义**: MaskWAM 的核心建模目标——在初始帧、首帧 Mask、机器人状态和语言指令条件下，联合生成未来 RGB 帧、未来 Mask 和动作序列

**符号说明**:
- $a_{1:K}$: 长度为 $K$ 的动作块（[[Action Chunking]]）
- $I_{1:T}$: 未来 $T$ 帧 RGB 图像
- $M_{1:T}$: 未来 $T$ 帧对应的目标对象 Mask
- $I_0, M_0$: 初始 RGB 帧与首帧目标 Mask（$M_0$ 为可选条件）
- $s_0$: 初始机器人状态
- $\ell$: 语言指令
- $\theta$: 模型参数

### 公式2: [[通道拼接|RGB-Mask 潜空间拼接]]

$$
z = [z_v;\, z_m] \in \mathbb{R}^{2C \times L \times H' \times W'}
$$

**含义**: 将 RGB 潜变量与 Mask 潜变量在通道维度拼接，形成统一的双通道潜表示供 DiT 处理

**符号说明**:
- $z_v \in \mathbb{R}^{C \times L \times H' \times W'}$: RGB 潜变量（由冻结 Video VAE 编码）
- $z_m \in \mathbb{R}^{C \times L \times H' \times W'}$: Mask 潜变量（Mask 渲染为三通道后同一 VAE 编码）
- $C$: 单模态潜变量通道数
- $L$: 时间帧数，$H', W'$: 空间下采样后的高宽

### 公式3: [[Flow Matching|联合 Flow Matching 训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{act}}
$$

**含义**: 三路解耦但联合优化的流匹配损失，RGB 预测、Mask 预测和动作预测各有独立损失，共同训练同一 [[Mixture of Transformers]] 骨干

**符号说明**:
- $\mathcal{L}_{\text{video}}$: RGB 帧流匹配重建损失（噪声调度 $\tau_v$）
- $\mathcal{L}_{\text{mask}}$: Mask 帧流匹配重建损失（与 $\mathcal{L}_{\text{video}}$ 共享 $\tau_v$，保证 RGB-Mask 时序对齐）
- $\mathcal{L}_{\text{act}}$: 动作流匹配损失（独立噪声调度 $\tau_a$，解耦动作与视频生成）

### 公式4: [[Zero-initialized Gating|零初始化门控融合]]（备选设计，已弃用）

$$
e_{\text{fused}} = e_{\text{rgb}} + g \cdot e_{\text{mask}}
$$

**含义**: ControlNet 风格的门控融合，可学习标量 $g$ 初始化为零；实验证明此设计导致模型忽略 Mask 信号，最终被弃用，改为直接监督

**符号说明**:
- $e_{\text{rgb}}$: RGB token 嵌入
- $e_{\text{mask}}$: Mask token 嵌入（独立编码器输出）
- $g$: 可学习门控标量，初始化为 $0$

---

## 关键图表

### Figure 1: 方法概览（Teaser）

![[MaskWAM_fig1_teaser.png]]

**说明**: MaskWAM 整体概览。左侧展示无 Mask 提示的语言清晰场景（标准 WAM），右侧展示有首帧 Mask 提示的语言歧义场景。预测未来 Mask 作为对象中心语义监督，首帧 Mask 提示消除语言歧义，两者通过统一 [[Mixture of Transformers]] 架构实现端到端学习。

### Figure 2: 模型架构

![[MaskWAM_fig2_architecture.png]]

**说明**: 训练阶段（左）与推理阶段（右）的完整架构。训练时，带噪 RGB 和 Mask 潜变量经通道拼接后送入 [[Diffusion Transformer|DiT]] 联合去噪；推理时采用单步部分去噪策略获取视觉上下文，结合 KV-Cache 高效预测动作块。

### Figure 3: 真机任务设置

![Figure 3](https://arxiv.org/html/2606.13515/2606.13515v1/x3.png)

**说明**: 8 个真机任务。Tasks 1-4 为语言清晰场景（标准操作任务），Tasks 5-8 为语言歧义场景（多个相似对象，需要精确空间指定）。

### Figure 4: 语言歧义泛化性能

![[MaskWAM_fig4_bar_chart.png]]

**说明**: 在四种泛化设置下（In-Domain、Distractors、Novel Instances、Lighting）对比各方法的成功率。MaskWAM 在所有设置下均大幅领先，尤其在 Novel Instances 设置下（74.6% vs π₀-mask 的 44.6%），体现了 Mask 提示与预测联合训练带来的强泛化能力。

### Figure 5: 真机实验平台

![Figure 5](https://arxiv.org/html/2606.13515/2606.13515v1/x6.png)

**说明**: 双臂 Xtrainer 机器人平台。搭载 RealSense D455（头顶 Eye-on-Base）和 RealSense D405（手腕 Eye-on-Hand），提供第三视角与手部近景双路视觉反馈。

### Figure 6: Mask 标注流水线

![Figure 6](https://arxiv.org/html/2606.13515/2606.13515v1/x7.png)

**说明**: 自动化 Mask 标注流程。语言清晰场景：Qwen3-VL 识别目标对象 → SAM-3 自动跟踪生成 Mask；语言歧义场景：人工在第一帧点击目标对象 → SAM-3 跟踪。91% 的 episode 无需人工修正，约每 50 个 episode 需 3 分钟人工审核。

### Figure 7: 不完美 Mask 提示示例

![Figure 7](https://arxiv.org/html/2606.13515/2606.13515v1/x8.png)

**说明**: 可视化各类噪声 Mask：腐蚀、膨胀、平移偏移、区域 Dropout。MaskWAM 对轻度噪声（<25 像素平移、<20% 区域 Dropout）具有良好鲁棒性；严重腐蚀（丢失目标区域中心）导致性能下降。

### Figure 8: 对噪声首帧 Mask 的鲁棒性

![Figure 8](https://arxiv.org/html/2606.13515/2606.13515v1/x9.png)

**说明**: 定量鲁棒性分析曲线。横轴为扰动强度，纵轴为任务成功率。Mask 主要提供身份识别与空间锚定功能，对边界精度要求不高，轻度形变不影响性能。

### Figure 9: 域内与域外真机演示

![Figure 9](https://arxiv.org/html/2606.13515/2606.13515v1/x10.png)

**说明**: 补充展示 MaskWAM 在域内（训练见过的对象）和域外（新实例、新背景、光照变化）场景下的操作演示序列。

### Figure 10: 真实数据预测可视化

![[MaskWAM_fig5_visualization.png]]

**说明**: 可视化 MaskWAM 预测的 RGB 帧、Mask 帧和注意力图。模型在真实数据上准确预测目标对象的未来 Mask，注意力集中在任务相关区域。

### Figure 11: RoboTwin 预测可视化

![Figure 11](https://arxiv.org/html/2606.13515/2606.13515v1/x12.png)

**说明**: RoboTwin 2.0 仿真环境中的 RGB 帧、Mask 和注意力图可视化，展示模型在仿真环境下的视觉预测质量。

### Figure 12: 文本提示 vs 视觉提示的注意力对比

![Figure 12](https://arxiv.org/html/2606.13515/2606.13515v1/x13.png)

**说明**: 对比文本提示 WAM 与视觉 Mask 提示 WAM（MaskWAM）的注意力图。Mask 提示模型注意力精确聚焦于目标对象，文本提示模型注意力分散，难以区分场景中的相似对象。

---

## 实验

### 数据集与 Benchmark

| 数据集/Benchmark | 规模/设置 | 特点 | 用途 |
|---|---|---|---|
| [[LIBERO]] | 4 子任务（Spatial/Object/Goal/Long） | 仿真桌面操作，25 任务 × 500 demo | 主要仿真评测 |
| [[RoboTwin 2.0]] | 6 任务（Hammer/Bell/Card/Burger/Stand/Shoe） | 仿真双臂操作 | 仿真 SOTA 对比 |
| 真机语言清晰任务 | 4 任务（Tasks 1-4） | 双臂 Xtrainer 真实操作 | 真机评测 |
| 真机语言歧义任务 | 4 任务（Tasks 5-8） × 4 泛化设置 | 多相似对象场景，需 Mask 提示 | 核心创新评测 |

### 实现细节

- **Backbone**: 冻结 [[Video VAE]]（与 [[FastWAM]] 共享预训练权重）+ [[Diffusion Transformer|DiT]] 联合去噪
- **分割模型**: [[SAM-3]]（用于首帧 Mask 提示获取，真机部署时一次性点击）
- **Mask Dropout 概率**: $p = 0.5$（训练时随机丢弃首帧 Mask）
- **推理策略**: 单步部分去噪（仅对 RGB-Mask 流）+ KV-Cache 加速动作预测
- **硬件**: 双臂 Xtrainer 机器人，RealSense D455（Eye-on-Base）+ RealSense D405（Eye-on-Hand）

### Table 1: LIBERO Benchmark 结果

| Method | Type | Spatial | Object | Goal | Long | **Avg** |
|--------|------|---------|--------|------|------|---------|
| WorldVLA | VLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| GR00T-N1 | VLA | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀ | VLA | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | VLA | 98.6 | 98.2 | 98.0 | 92.4 | 96.8 |
| Motus | WAM | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| FastWAM | WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Ours (RGB-only) | WAM | 96.8 | 99.6 | 97.0 | 95.8 | 97.3 |
| Ours (Mask-only) | WAM | 97.2 | 99.8 | 97.4 | 96.0 | 97.6 |
| **MaskWAM (Ours)** | WAM | **98.8** | **100.0** | **98.2** | **96.4** | **98.4** |

**关键发现**: MaskWAM 以 98.4% 的平均成功率超越所有 VLA 和 WAM 基线。Spatial 子任务提升最显著（+2.0% over RGB-only，+0.6% over FastWAM），印证 Mask 监督对空间理解的关键作用。

### Table 2: RoboTwin 2.0 Benchmark 结果

| Method | Hammer | Bell | Card | Burger | Stand | Shoe | **Average** |
|--------|--------|------|------|--------|-------|------|-------------|
| π₀ | 68 | 72 | 81 | 79 | 63 | 74 | 72.8 |
| FastWAM | 83 | 87 | 92 | 94 | 80 | 90 | 87.7 |
| Ours (RGB-only) | 82 | 87 | 91 | 93 | 79 | 92 | 87.3 |
| Ours (Mask-only) | 85 | 90 | 93 | 93 | 81 | 91 | 88.8 |
| **MaskWAM (Ours)** | **88** | **93** | **95** | **97** | **85** | **95** | **92.2** |

**关键发现**: 相比 π₀ 提升 +19.4%，相比 FastWAM 提升 +4.5%。Mask-only（88.8%）> RGB-only（87.3%）> FastWAM（87.7%），说明 Mask 预测提供了比纯 RGB 预测更强的语义监督。

### Table 3: 真机语言清晰任务结果

| Method | Type | Task 1 | Task 2 | Task 3 | Task 4 | **Avg** |
|--------|------|--------|--------|--------|--------|---------|
| π₀ | VLA | 57 | 54 | 54 | 58 | 55.8 |
| π₀.₅ | VLA | 83 | 55 | 74 | 77 | 72.3 |
| FastWAM | WAM | 88 | 76 | 77 | 75 | 79.0 |
| Ours (RGB-only) | WAM | 86 | 77 | 76 | 78 | 79.3 |
| **MaskWAM (Ours)** | WAM | **91** | **82** | **81** | **83** | **84.3** |

**关键发现**: 在不需要 Mask 提示（无语言歧义）的任务上，MaskWAM 仍比 FastWAM 高 +5.3%，说明 **Mask 预测本身**（非提示）也对基础操作能力有实质性提升。

### Table 4: 消融——Mask 条件与预测的必要性

| 配置 | Future Mask Pred | Mask Prompt | Coord Prompt | **Avg 成功率** |
|------|:---:|:---:|:---:|:---:|
| Ours-no-pred（仅 Mask 提示，无预测）| – | ✓ | – | 21.6% |
| Ours-coord（仅坐标提示）| ✓ | – | ✓ | 18.2% |
| **MaskWAM（完整）** | ✓ | ✓ | – | **84.9%** |

**关键发现**: 仅有 Mask 提示但无未来 Mask 预测监督时，性能急剧下降至 21.6%——说明**未来 Mask 预测是视觉提示有效 grounding 的必要条件**。坐标文本提示效果更差（18.2%），印证密集空间语义先验优于稀疏坐标描述。

### Table 5: 语言歧义泛化实验（完整）

| 泛化设置 | Method | Type | Task 5 | Task 6 | Task 7 | Task 8 | **Average** |
|---|---|---|---|---|---|---|---|
| **In Domain** | π₀-mask | VLA | 96.7 | 93.3 | 38.3 | 23.3 | 62.9 |
| | π₀-coord | VLA | 53.3 | 88.3 | 20.0 | 5.0 | 41.7 |
| | FastWAM-coord | WAM | 35.0 | 55.0 | 13.3 | 1.7 | 26.3 |
| | **MaskWAM** | WAM | **100.0** | **100.0** | **88.3** | **83.3** | **92.9** |
| **Distractors** | π₀-mask | VLA | 86.7 | 75.0 | 31.7 | 18.3 | 52.9 |
| | π₀-coord | VLA | 46.7 | 61.7 | 11.7 | 0.0 | 30.0 |
| | FastWAM-coord | WAM | 15.0 | 23.3 | 5.0 | 0.0 | 10.8 |
| | **MaskWAM** | WAM | **100.0** | **98.3** | **85.0** | **78.3** | **90.4** |
| **Novel Instances** | π₀-mask | VLA | 71.7 | 65.0 | 26.7 | 15.0 | 44.6 |
| | π₀-coord | VLA | 41.7 | 51.7 | 8.3 | 0.0 | 25.4 |
| | FastWAM-coord | WAM | 18.3 | 28.3 | 6.7 | 0.0 | 13.3 |
| | **MaskWAM** | WAM | **86.7** | **76.7** | **70.0** | **65.0** | **74.6** |
| **Lighting** | π₀-mask | VLA | 75.0 | 68.3 | 28.3 | 13.3 | 46.3 |
| | π₀-coord | VLA | 48.3 | 66.7 | 15.0 | 3.3 | 33.3 |
| | FastWAM-coord | WAM | 31.7 | 36.7 | 8.3 | 0.0 | 19.2 |
| | **MaskWAM** | WAM | **93.3** | **91.7** | **73.3** | **68.3** | **81.7** |

**关键发现**: MaskWAM 在所有泛化设置下均大幅领先。最重要的是 Novel Instances 设置（74.6% vs π₀-mask 44.6%），说明统一 Mask 预测使模型学到了对象中心的可泛化表示，而非过拟合训练对象的外观。

---

## 批判性思考

### 优点
1. **设计直觉清晰**: Mask 的双重角色（输入提示 + 预测监督）各自独立有效，消融实验区分干净，说服力强
2. **无需实时分割**: 推理时 SAM-3 仅处理首帧，不增加实时推理成本，工程实用性高
3. **语言-视觉协同**: Mask Dropout 训练策略优雅地统一了两类应用场景（语言清晰 vs 歧义），无需分别训练两个模型

### 局限性
1. **标注依赖**: 训练需要 RGB-Mask-Action 三元组数据，虽然自动化程度高（91% 无需修正），但相比纯 RGB-Action 数据，收集成本仍有所增加
2. **SAM-3 部署依赖**: 真机部署依赖 SAM-3 分割能力，在极度杂乱或遮挡严重的场景中，SAM-3 定位失败会传导到策略失效
3. **大规模预训练未探索**: 论文指出 RGB-Mask-Action 联合大规模预训练是未来工作，当前仅在任务特定数据上微调，预训练数据规模未充分利用

### 潜在改进方向
1. 探索无需人工点击的全自动首帧 Mask 获取（用语言指令驱动 SAM-3 零样本定位）
2. 构建大规模 RGB-Mask-Action 预训练数据集，充分发挥联合预测的监督优势
3. 扩展到多对象协同操作场景（当前 Mask 仅标注单一目标对象）

### 可复现性评估
- [ ] 代码开源（项目主页存在，代码待确认）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（详细描述了 Mask Dropout、噪声调度等关键超参）
- [x] 数据集可获取（LIBERO、RoboTwin 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[FastWAM]]: 直接基线，共享 Video VAE 预训练权重，MaskWAM 在此基础上加入 Mask 流
- [[π0]]: 核心对比 VLA 基线，真机任务提升 +28.5%
- [[Flow Matching]]: 核心训练框架，解耦 RGB、Mask、动作三路噪声调度

### 对比
- [[π0.5]]: LIBERO 上被超越（96.8% → 98.4%），代表最强 VLA 基线
- [[Motus]]: WAM 类方法，LIBERO 上成绩接近（97.7% vs 98.4%）

### 方法相关
- [[World Action Model]]: 本文所属研究范畴
- [[Video VAE]]: 用于 RGB 与 Mask 的统一潜空间编码
- [[Diffusion Transformer]]: 核心去噪骨干架构
- [[Mixture of Transformers]]: 统一处理多模态（RGB/Mask/动作）的架构选择
- [[Flow Matching]]: 训练目标框架
- [[Action Chunking]]: 动作输出格式
- [[SAM-3]]: 首帧 Mask 获取工具

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评测 benchmark
- [[RoboTwin]]: 仿真双臂操作 benchmark
- [[Xtrainer]]: 真机实验平台（双臂机器人）

---

## 速查卡片

> [!summary] MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models
> - **核心**: 将 Mask 统一为 WAM 的输入提示（消歧义）与预测目标（语义监督），一石二鸟解决空间定位与语言歧义两大瓶颈
> - **方法**: 冻结 Video VAE 通道拼接 RGB+Mask 潜变量 → Mixture of Transformers 联合 Flow Matching 去噪；Mask Dropout 统一两类场景
> - **结果**: LIBERO 98.4%、RoboTwin 92.2%、真机语言歧义场景 84.9%，全面 SOTA
> - **代码**: [项目主页](https://hanyangyu1021.github.io/maskwam.github.io/)

---

*笔记创建时间: 2026-06-12*
