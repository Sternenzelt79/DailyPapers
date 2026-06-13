---
title: "Cosmos 3: Omnimodal World Models for Physical AI"
method_name: "Cosmos3"
authors: [NVIDIA Team]
year: 2026
venue: arXiv
tags: [world-model, omnimodal, mixture-of-transformers, physical-ai, video-generation, robot-policy, action-generation, diffusion-transformer]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/abs/2606.02800
created: 2026-06-04
---

# 论文笔记：Cosmos 3: Omnimodal World Models for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA (291 位作者) |
| 日期 | June 2026 |
| 项目主页 | [research.nvidia.com/labs/cosmos-lab/cosmos3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) |
| 对比基线 | [[扩散世界模型]]、[[视频基础模型]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02800) / [Code](https://github.com/nvidia/cosmos) / [HuggingFace](https://huggingface.co/collections/nvidia/cosmos3) |

---

## 一句话总结

> Cosmos 3 是一个统一的 omnimodal 世界模型，在单一 [[Mixture-of-Transformers]] 架构中同时处理语言、图像、视频、音频和动作，将 VLM、视频生成器、世界模拟器和动作模型整合为一个框架。

---

## 核心贡献

1. **统一 Omnimodal 架构**: 提出 [[Two-Tower MoT]] 架构，将 [[自回归变换器|自回归 Transformer]]（AR Tower）和 [[Diffusion Transformer]]（DM Tower）融合在同一前向传播中，支持语言/图像/视频/音频/动作的理解与生成
2. **灵活 Physical AI 接口**: 单一模型覆盖 VLM 推理、文本转图像/视频、音视频同步生成、机器人 [[前向动力学模型|前向动力学]]、[[逆向动力学模型|逆向动力学]]、直接策略生成等 7 种任务配置
3. **开源生态与数据集**: 发布模型权重、代码、训练配方及 6 个合成数据集，在 RoboArena、PAI-Bench、Physics-IQ 等多个 Physical AI 基准上达到开源 SOTA

---

## 问题背景

### 要解决的问题
Physical AI 系统（机器人、自动驾驶车辆）需要能够**同时理解和生成**多种模态（语言、图像、视频、音频、动作）的模型。现有方案将 VLM、视频生成器、世界仿真器、动作模型分开训练，造成模态信息孤岛和系统复杂度高。

### 现有方法的局限
- VLM 只能理解不能生成视觉内容
- 视频生成模型缺乏物理推理能力
- 机器人策略模型不具备世界模拟能力
- 多个独立模型维护成本高，无法共享跨模态知识

### 本文的动机
将 [[自回归模型|自回归]] 和 [[扩散模型]] 的优势整合在一个 [[Mixture-of-Transformers]] 架构中：AR 子系列处理离散 token（文本/推理），DM 子系列处理连续 token（图像/视频/音频/动作），二者通过 joint attention 相互交互，一个模型完成所有 Physical AI 任务。

---

## 方法详解

### 模型架构

**Cosmos 3** 采用 **[[Two-Tower MoT|Two-Tower Mixture-of-Transformers]]** 架构：

- **输入**: 语言指令 + 图像/视频帧 + 音频 + 动作序列（可任意组合）
- **AR Tower（Reasoner）**: [[自回归变换器]] (因果自注意力)，处理离散 token，负责理解、规划、推理
- **DM Tower（Generator）**: [[Diffusion Transformer]] (全注意力)，处理连续 token，负责生成图像/视频/音频/动作
- **Shared Backbone**: AR 和 DM token 通过 **joint attention** 共享信息，使推理能力直接条件化生成
- **输出**: 文本（next-token 自回归解码）或图像/视频/音频/动作（迭代去噪）
- **总参数**: Nano 16B（8B + 8B） / Super 64B（32B + 32B）

### 核心模块

#### 模块1: Omnimodal Tokenizer

**设计动机**: 将五种模态统一编码到共享表示空间，同时保留模态特有的结构信息。

**具体实现**:
- **视觉理解**: [[视觉变换器|ViT]] 将图像/视频帧编码为离散 token，加入 AR 子序列
- **视觉/音频生成**: [[变分自编码器|VAE]]（图像用 2D VAE，视频用 [[Causal 3D VAE]]）将图像/视频/音频编码为连续 latent，加入 DM 子序列
- **动作**: 使用 domain-aware 向量表示各机器人平台的动作（9D~57D），加入 DM 子序列
- 全部投影至 shared representation space 后送入统一 Transformer

**Causal 3D VAE**: 对视频做时序压缩，理解帧间像素关系，在 latent 空间中保持运动语义，尤其针对快速运动目标做了优化。

#### 模块2: Two-Tower MoT 前向计算

**设计动机**: AR token 需要因果注意力确保自回归属性，DM token 需要全注意力确保生成质量，两种注意力模式无法直接共用同一层参数。

**具体实现**:
- Token 序列被分为两个子序列：
  - **AR 子序列**: 离散文本 token + ViT 视觉 token（用于 Reasoner 模式）
  - **DM 子序列**: 连续 VAE/动作 token（用于 Generator 模式）
- AR 和 DM token 使用 **独立的参数集**（MoT 中的"不同专家"），在 Transformer 层内分别计算
- **Joint Attention**: AR 和 DM 子序列相互可见，推理理解直接影响生成过程
- 位置编码采用 [[多维旋转位置编码|3D mRoPE]]，统一编码空间-时间结构

#### 模块3: 推理与生成的双模式切换

**Reasoner 模式**（仅 AR 塔激活）:
- 语言和视觉理解 token 通过因果自注意力处理
- Next-token 预测，用于感知、规划、世界推理任务
- 单独可运行，不需激活生成塔

**Generator 模式**（AR + DM 两塔同时激活）:
- 噪声图像/视频/音频/动作 token 通过全注意力去噪
- 以 AR 塔的推理结果为条件，生成多模态输出
- 生成任务必须同时激活两塔

#### 模块4: 支持的动作体（Action Embodiments）

Cosmos 3 支持 9 种机器人平台的动作表示，动作维度从 9D 到 57D：

| 平台 | 动作维度 | 用途 |
|------|---------|------|
| General Camera Motion | 9D | 相机控制 |
| Autonomous Vehicles | 9D | 自动驾驶 |
| Egocentric Motion | 57D | 第一人称运动 |
| Franka Panda | 10D / 20D | 单臂/双臂操作 |
| Agibot | 29D | 人形机器人 |
| UR Robots | 10D | 通用机械臂 |
| Google Robot | 10D | Google 操作平台 |
| WidowX 250 | 10D | 小型操作臂 |
| UMI | 9D | 通用操作接口 |

---

## 关键公式

### 公式1: [[Two-Tower MoT|MoT Token 序列划分]]

$$
\mathbf{X} = [\underbrace{x_1^{AR}, x_2^{AR}, \ldots, x_m^{AR}}_{\text{AR 子序列（离散）}}, \underbrace{x_1^{DM}, x_2^{DM}, \ldots, x_n^{DM}}_{\text{DM 子序列（连续）}}]
$$

**含义**: 输入 token 序列 $\mathbf{X}$ 被分为 AR 子序列（文本+ViT视觉 token）和 DM 子序列（VAE连续 token），二者使用独立参数集但通过 joint attention 交互。

**符号说明**:
- $x_i^{AR}$: 第 $i$ 个自回归离散 token（文本或 ViT 编码的视觉 token）
- $x_j^{DM}$: 第 $j$ 个扩散连续 token（VAE 编码的图像/视频/音频/动作）
- $m$: AR 子序列长度
- $n$: DM 子序列长度

### 公式2: [[Diffusion Transformer|扩散去噪目标]]

$$
\mathcal{L}_{DM} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{AR}) - \boldsymbol{\epsilon} \right\|^2 \right]
$$

**含义**: DM Tower 的训练目标是在给定 AR Tower 推理上下文 $\mathbf{c}_{AR}$ 的条件下，预测加入到 $\mathbf{x}_t$ 中的噪声 $\boldsymbol{\epsilon}$，从而学习物理感知的生成能力。

**符号说明**:
- $\mathbf{x}_0$: 原始连续 token（图像/视频/音频/动作）
- $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$: $t$ 时刻的加噪版本
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 高斯噪声
- $\boldsymbol{\epsilon}_\theta$: DM Tower 的去噪网络（参数为 $\theta$）
- $\mathbf{c}_{AR}$: AR Tower 输出的推理上下文条件

### 公式3: [[自回归模型|AR 自回归生成目标]]

$$
\mathcal{L}_{AR} = -\sum_{i=1}^{m} \log p_\theta(x_i^{AR} \mid x_1^{AR}, \ldots, x_{i-1}^{AR}, \mathbf{x}^{DM})
$$

**含义**: AR Tower 使用标准的因果语言模型损失，在给定所有连续 DM token 和之前所有离散 token 的条件下预测下一个离散 token，实现文本和推理输出的自回归生成。

**符号说明**:
- $p_\theta$: AR Tower 的 next-token 概率分布
- $\mathbf{x}^{DM}$: DM 子序列（通过 joint attention 可见）
- $m$: AR 序列总长度

### 公式4: [[多维旋转位置编码|3D mRoPE 位置编码]]

$$
\text{RoPE}_{3D}(\mathbf{q}, \mathbf{k}, p_t, p_h, p_w) = \text{RoPE}_t(p_t) \oplus \text{RoPE}_h(p_h) \oplus \text{RoPE}_w(p_w)
$$

**含义**: 对视频 token 分别在时间维度 $p_t$、高度维度 $p_h$、宽度维度 $p_w$ 施加旋转位置编码，并拼接，从而在统一 latent 空间中编码 3D 时空结构，使模型能够理解不同分辨率和帧率的视频。

**符号说明**:
- $\mathbf{q}, \mathbf{k}$: Query 和 Key 向量
- $p_t, p_h, p_w$: 时间、高度、宽度维度的位置索引
- $\oplus$: 向量拼接操作

---

## 关键图表

### Figure 1: 系统概览 — Omnimodal 输入输出配置

![Cosmos 3 Architecture Overview](https://cdn-uploads.huggingface.co/production/uploads/6799309995f2227228bc38f3/IBreD1akJz8T47xKva3_B.png)

**说明**: Cosmos 3 的整体架构。左侧展示所有支持的输入模态（文本、图像、视频、音频、动作），经过 [[Two-Tower MoT]] 的 AR Tower（Reasoner）和 DM Tower（Generator）处理后，右侧输出对应模态的理解或生成结果。

### Figure 2: 机器人操作 Demo — Pick & Place

![Cosmos 3 Robotics Pick and Place](https://cdn-uploads.huggingface.co/production/uploads/6799309995f2227228bc38f3/Wcx8G3cD6e3tib-Zj4Ryw.gif)

**说明**: Cosmos 3 作为机器人策略模型，从语言指令和图像输入直接生成 Franka Panda 机械臂的关节角度动作序列，完成拾取放置任务。展示了[[前向动力学模型]]能力。

### Figure 3: 自动驾驶 Demo — 视频生成

![Cosmos 3 Autonomous Driving](https://cdn-uploads.huggingface.co/production/uploads/6799309995f2227228bc38f3/FP6Mh3FqTyThuZ_RN4mt4.gif)

**说明**: 以自动驾驶场景视频为条件，Cosmos 3 生成时序连贯的预测视频，展示[[前向动力学模型]]在自动驾驶场景中的世界模拟能力。

### Figure 4: 智能仓库 Demo — Image-to-Video

![Cosmos 3 Warehouse Operations](https://cdn-uploads.huggingface.co/production/uploads/6799309995f2227228bc38f3/WgIJWLbDhavkxYV8mBYjh.gif)

**说明**: 以仓库环境图片为输入，生成包含工人操作的动态视频，展示了 Image-to-Video 生成能力在工业 Physical AI 场景中的应用。

### Figure 5: 自动驾驶 Chain-of-Thought 推理

![Cosmos 3 Autonomous Driving Chain-of-Thought](https://cdn-uploads.huggingface.co/production/uploads/6799309995f2227228bc38f3/g06bFosM9Br-cwgfmt7cl.png)

**说明**: Cosmos 3 的 Reasoner 模式进行交通场景推理，先逐步分析场景（CoT），再输出驾驶决策，展示了 AR Tower 的长链推理能力（256K token 上下文）。

### Table 1: 模型变体规格

| 变体 | 参数量 | AR Tower | DM Tower | 部署目标 |
|------|-------|----------|----------|---------|
| **Cosmos 3 Nano** | 16B | 8B | 8B | 工作站（RTX PRO 6000） |
| **Cosmos 3 Super** | 64B | 32B | 32B | 数据中心（Hopper/Blackwell） |
| **Cosmos 3 Edge** | 4B | 2B | 2B | 边缘设备（计划中） |

### Table 2: 支持的任务输入/输出配置

| 输入 | 输出 | 任务类型 |
|------|------|---------|
| 文本、图像、视频 | 视频 | 视频生成模型 |
| 文本、视频 | 文本 | 视觉语言模型（VLM） |
| 动作 + 图像 + 文本 | 视频 | 前向动力学模型 |
| 文本 + 视频 | 动作 | 逆向动力学模型 |
| 图像 + 文本 | 视频 + 动作 | 策略模型 |
| 文本 + 图像 | 图像 | 文本转图像 |
| 文本 + 视频 | 视频 + 音频 | 音视频同步生成 |

### Table 3: 训练数据规模

| 模态 | 理解样本 | 生成样本 |
|------|---------|---------|
| 文本 | 22M | — |
| 图像 | 19M | 767M |
| 视频 | 1M | 348M |
| 音频 | — | 139M |
| 动作 | — | 8M |
| **总计** | **~42M** | **~1.26B** |

**说明**: 训练数据总量约 20 万亿 token，涵盖约 10 亿图片、4 亿真实与合成视频，以及环境音频、文本和机器人动作数据。

### Table 4: 综合 Benchmark 对比（含具体分数）

| 模型 | 推理:General | 推理:Robotics | 推理:Smart Infra. | 推理:Driving | T2I | T2V | I2V | Audio | FD:Robot | Policy:Robot |
|------|------------|-------------|-----------------|------------|-----|-----|-----|-------|---------|-------------|
| **Cosmos3-Super** | **73.7** | **57.8** | **62.6** | **79.3** | 91.26* | **80.0** | **82.8** | 7.31 | **26.0*** | — |
| **Cosmos3-Nano** | 69.6 | 55.1 | **61.0** | **76.0** | 84.22 | 79.4 | 82.7 | **7.34** | 25.5* | **39.7*** |
| Gemini 3.1 Pro† | 77.5 | **58.2** | 58.6 | 47.2 | — | — | — | — | — | — |
| Qwen3-VL-32B | 72.8 | 52.6 | 56.1 | 40.7 | — | — | — | — | — | — |
| Qwen3-VL-8B | 68.9 | 48.5 | 52.7 | 46.4 | — | — | — | — | — | — |
| Gemma-4-31B | 69.8 | 51.0 | 51.3 | 36.6 | — | — | — | — | — | — |
| Gemma-4-E4B | 53.1 | 39.3 | 29.4 | 26.0 | — | — | — | — | — | — |
| Gemini 3 Pro Image† | — | — | — | — | **90.85** | — | — | — | — | — |
| Qwen-Image-2512 | — | — | — | — | 84.25 | — | — | — | — | — |
| Veo-3.1† | — | — | — | — | — | 79.1 | 82.6 | **7.45** | — | — |
| Wan2.2-A14B | — | — | — | — | — | 78.0 | 81.3 | — | — | — |
| Ctrl-World | — | — | — | — | — | — | — | — | 23.0 | — |
| π₀.₅ | — | — | — | — | — | — | — | — | — | 28.1 |

**说明**: † 表示闭源模型；* 表示开源 SOTA；T2I = text-to-image；T2V = text-to-video；I2V = image-to-video；FD:Robot = forward dynamics（机器人）；Policy:Robot = 机器人策略。Cosmos3-Super 在 Driving 推理（79.3）、T2V、I2V 上领先所有开源模型；Cosmos3-Nano 在 Policy:Robot（39.7）大幅超越 π₀.₅（28.1）。

### Table 5: 基准测试排名汇总

| 基准 | 类别 | Cosmos 3 排名 |
|------|------|-------------|
| Artificial Analysis | 文本转图像/图像转视频 | **#1 开源** |
| RoboArena | 机器人策略 | **#1 开源** |
| PAI-Bench | Physical AI 综合 | **#1 开源** |
| Physics-IQ | 物理推理 | **#1 开源** |
| R-Bench | 机器人感知 | **#1 开源** |
| RoboLab | 机器人操作 | **#1 开源** |
| VANTAGE-Bench | 智能基础设施场景理解 | **#1 开源（各级别）** |
| TAR (AI City Challenge 2026 Track 3) | 交通异常推理 | **榜首** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OpenImage | 1.2M | 真实世界图像 | 图像理解训练 |
| Coyo700M | 100M | 大规模图文对 | 视觉-语言对齐 |
| YouTube Video | 340M | 真实视频 | 视频生成训练 |
| UMI | 4.5M | 机器人操作轨迹 | 动作生成训练 |
| Embodied-Robot-Scenes (合成) | — | Isaac Sim 仿真 | 机器人场景 SDG |
| Physical-Interaction-Scenes (合成) | — | 物理仿真数据 | 物理推理 |
| Spatial-Reasoning (合成) | — | 空间推理数据 | 具身空间推理 |
| Digital-Human-Scenes (合成) | — | 人体运动数据 | 人形机器人 |
| Autonomous-Driving-Scenarios (合成) | — | 驾驶仿真 | 自动驾驶 |
| Warehouse-Operations-Scenes (合成) | — | 仓库环境 | 工业场景 |

### 实现细节

- **精度**: BF16（官方支持），支持 FP8 / NVFP4（4-bit 量化，推理速度最高 2x）
- **视频输入**: 5 帧（最大），256p-720p，带/不带音频
- **音频**: 立体声，48kHz，最长 0.5 秒
- **动作帧数**: 16-400 帧（输入/输出）
- **视频输出**: 默认 189 帧
- **文本 context**: 4096 tokens（生成），256K tokens（推理模式）
- **推理引擎**: vLLM（推理任务）、vLLM-Omni（多模态生成）、Hugging Face Diffusers
- **训练配方**: 发布 SFT 后训练配方，支持自定义视觉生成微调和动作后训练

### 后训练策略

**预训练阶段**:
- 数据混合：约 2200 万样本，覆盖 OCR、grounding、QA、推理、字幕和指令跟随
- 强调强图文对齐、阅读能力、空间 grounding，保留轻量视频组件为后续 SFT 准备

**监督微调（SFT）**:
- AI-judge 过滤：阈值 5 时保留 46% 高置信度样本
- 增强通用空间和时间理解能力
- 特定任务配方：视觉生成后训练（自定义视频数据集）、动作后训练（机器人 Physical AI）

### 优化技术

- **[[高效视频采样|Efficient Video Sampling (EVS)]]**: 减少视频理解任务中的 token 数量
- **NVFP4 量化**: 4-bit 浮点数推理，最高 2x 加速
- **vLLM 推理引擎**: 优化批处理和 KV 缓存

---

## 批判性思考

### 优点
1. **统一框架减少工程复杂度**: 单模型替代 VLM + 视频生成 + 世界仿真 + 策略模型四个独立系统，降低 Physical AI 开发门槛
2. **推理指导生成**: AR Tower 的物理理解直接通过 joint attention 条件化 DM Tower 的生成，使生成内容符合物理规律，而非仅仅基于统计相关性
3. **开源生态完整**: 模型权重、训练代码、6 个合成数据集同步发布，可复现性强
4. **多体动作支持**: 内置 9 种机器人平台的动作接口，降低机器人应用迁移成本

### 局限性
1. **生成任务必须激活双塔**: 每次视频/图像/动作生成都需要同时运行 AR 和 DM Tower，相比专用扩散模型计算开销更大，实时推理在复杂场景下仍有挑战
2. **灵巧操作能力尚早期**: 论文明确指出复杂灵巧操作（dexterous manipulation）能力处于早期阶段，主要价值在于合成数据生成而非直接部署
3. **音视频同步不完美**: 在复杂动态场景中音频-视频同步存在缺陷
4. **长序列动作漂移**: 在扩展动作序列中存在 action-state drift 问题

### 潜在改进方向
1. 专门的 DM Tower 蒸馏版，用于计算受限的部署环境
2. 针对灵巧操作的专用后训练数据和策略
3. 引入在线强化学习改善长序列任务的累积误差

### 可复现性评估
- [x] 代码开源（[GitHub](https://github.com/nvidia/cosmos)）
- [x] 预训练模型（[HuggingFace](https://huggingface.co/collections/nvidia/cosmos3)）
- [x] 训练细节完整（发布 SFT 后训练配方）
- [x] 合成数据集可获取（6 个 SDG 数据集）
- [x] 许可证: OpenMDW-1.1（Linux Foundation）

---

## 关联笔记

### 基于
- [[扩散世界模型]]: Cosmos 3 的 DM Tower 沿用扩散世界模型的核心思路
- [[视频基础模型]]: 在大规模视频数据上预训练的基础
- [[自回归扩散模型]]: AR + 扩散的混合范式来源

### 对比
- [[CoME]]: 同类条件世界模型，但专注于操作场景，不支持 omnimodal
- [[WAV]]: World-Action-Video 模型，与 Cosmos 3 的 policy 模式类似但架构不同
- [[扩散策略]]: 专用机器人策略，Cosmos 3 将其能力集成为一个子任务

### 方法相关
- [[Two-Tower MoT]]: 核心架构创新
- [[Diffusion Transformer]]: DM Tower 基础
- [[Causal 3D VAE]]: 视频/音频编码关键组件
- [[多维旋转位置编码]]: 统一时空位置编码
- [[前向动力学模型]]: Cosmos 3 支持的核心 Physical AI 任务
- [[逆向动力学模型]]: 从视频恢复动作的能力

### 硬件/数据相关
- [[Isaac Sim]]: 合成数据生成仿真平台
- [[Franka Panda]]: 主要支持的机器人平台之一

---

## 速查卡片

> [!summary] Cosmos 3: Omnimodal World Models for Physical AI
> - **核心**: 统一 [[Two-Tower MoT]] 架构，AR 推理 + DM 生成，单模型覆盖所有 Physical AI 任务
> - **方法**: [[自回归变换器]] + [[Diffusion Transformer]] 通过 joint attention 交互，[[Causal 3D VAE]] 统一多模态 tokenization
> - **结果**: 在 RoboArena、PAI-Bench、Physics-IQ、Artificial Analysis 等 8 个基准达到开源 SOTA
> - **代码**: [github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)

---

*笔记创建时间: 2026-06-04*
