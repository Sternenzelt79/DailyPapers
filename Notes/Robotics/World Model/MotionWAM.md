---
title: "MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
method_name: "MotionWAM"
authors: [Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang]
year: 2026
venue: arXiv
tags: [world-action-model, humanoid, loco-manipulation, flow-matching, egocentric-vision]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.09215
created: 2026-06-09
---

# 论文笔记：MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mondo Robotics; HKUST (GZ); HKUST |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[GR00T N1.5\|GR00T-N1.7]], [[Pi05\|π₀.₅]], ACT, Diffusion Policy |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09215) |

---

## 一句话总结

> MotionWAM 将视频扩散世界模型的中间去噪特征作为条件，驱动人形机器人实时全身运动操作，以 76.1% 成功率超越最强 VLA 基线 43.9%。

---

## 核心贡献

1. **实时 WAM 策略**: 首个封闭环、端到端驱动全身人形机器人[[Loco-Manipulation|全身运动操作]]的[[WAM|World Action Model]]，推理频率 4.9 Hz，比 Cosmos Policy 快 7 倍。
2. **统一运动潜变量**: 用统一的[[Flow Matching|流匹配]]运动潜变量 $m_t$ 替代上下身分离控制器，天然支持任务驱动的足部交互（踢球等）。
3. **三阶段训练框架**: 以自我中心视频预训练 → 跨体型动作后训练 → 全身微调的渐进学习策略，将视频世界模型迁移到目标体型。

---

## 问题背景

### 要解决的问题

现有机器人操作策略通常将上身操作和下身运动解耦处理，使用独立的控制器，导致无法完成需要全身协调的复杂任务（如踢球取物、蹲下装卸等）。

### 现有方法的局限

- [[VLA|VLA 方法]]（如 [[GR00T N1.5|GR00T-N1.7]]、[[Pi05|π₀.₅]]）直接预测动作，缺乏对物理场景动态的感知。
- 以往的 [[WAM|World Action Model]] 方法（如 Cosmos Policy）推理速度太慢（0.7 Hz），无法在实机上实时控制。
- 层次化策略的上下身分离控制无法支持任务驱动的足部参与（足部只维持平衡，不主动完成任务）。

### 本文的动机

视频扩散模型在去噪过程中学到了丰富的场景未来动态信息，其中间去噪特征（intermediate denoising features）可作为隐式世界模型条件。关键洞察：**只运行一次 Video DiT 的前向传播即可获得编码未来场景走向的隐状态**，而无需真正生成视频帧，从而将计算成本控制在实时范围内。

---

## 方法详解

### 模型架构

MotionWAM 采用 **Dual-DiT（双扩散 Transformer）** 架构：

- **输入**: 自我中心 RGB 图像帧 $o_t$（Intel RealSense D435i）+ 语言指令 $l$ + 本体状态 $p_t$ + 体型标签 $e$
- **Video DiT**: [[Cosmos-Predict2.5|Cosmos-Predict2.5-2B]] 将视频帧压缩为 [[VAE]] 潜变量，在固定流时间步 $\tau_f \approx 1$ 时输出隐状态 $h_t^{\tau_f}$
- **Motion DiT**: DiT-B（隐藏维度 2560），以 $h_t^{\tau_f}$ 为条件，通过[[Flow Matching|流匹配]]预测运动潜变量 $m_t$
- **输出**: 统一运动潜变量 $m_t = (m_t^{\text{cont}}, k_t)$，其中 $m_t^{\text{cont}}$ 驱动末端执行器，$k_t$ 为 [[SONIC]] 全身控制器的离散运动令牌索引
- **总参数**: Video DiT ~2B（冻结 VAE/文本编码器）+ Motion DiT ~100M

### 核心模块

#### 模块 1: 中间去噪特征提取（Intermediate Denoising Features）

**设计动机**: 利用视频扩散模型在去噪中途（$\tau_f \approx 1$，接近纯噪声端）的隐状态，该状态已编码了场景将要发生的变化，但不需要完整去噪。

**具体实现**:
- 将当前帧编码为 [[VAE]] 潜变量 $z_t^0$
- 向潜变量注入固定量噪声：$z_t^{\tau_f} = \tau_f \cdot \epsilon_v + (1-\tau_f) \cdot z_t^0$，其中 $\epsilon_v \sim \mathcal{N}(0, I)$
- Video DiT 做一次前向传播，提取所有层的隐状态序列 $h_t^{\tau_f}$
- 此操作只做**一次前向传播**，不做去噪迭代，因此速度极快

#### 模块 2: 统一运动潜变量（Unified Motion Latent）

**设计动机**: 替代上下身分离的层次化控制，使策略能够学习全身协调的统一运动表示。

**具体实现**:
- 连续部分 $m_t^{\text{cont}} \in \mathbb{R}^{14}$：双臂末端执行器的 6D 位姿 + 夹爪开度
- 离散部分 $k_t \in \mathbb{Z}$：[[SONIC]] 编解码器的 VQ 令牌索引，编码躯干高度、蹲伏幅度、腰部旋转、足部接触意图
- Motion DiT 同时预测 $m_t^{\text{cont}}$ 和 $\tilde{k}_t$（连续松弛值），推理时用最近邻取整恢复离散索引 $\hat{k}_t$
- [[SONIC]] 全身控制器将 $(\hat{k}_t, m_t^{\text{cont}})$ 解码为 29 自由度关节角度，以 50 Hz 执行

#### 模块 3: 三阶段训练（Three-Stage Training）

**Stage 1 — 自我中心视频预训练**:
- 数据：~2136 小时自我中心视频（人类 30% + G1 类人形 50% + 其他机器人 20%）
- 仅训练 Video DiT，目标：$\mathcal{L}_{\text{video}}$
- 将世界模型的视觉先验从第三视角迁移到第一人称

**Stage 2 — 跨体型动作后训练**:
- 挂载 Motion DiT，联合训练
- 在 Unitree G1 异构数据集上训练（GR00T、RoboCOIN、UnifoLM-WBT 等）
- 目标：$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$

**Stage 3 — 全身微调**:
- 每任务 200 条遥操作演示（PICO VR 头显 + XRoboToolkit 遥操作框架）
- 端到端训练全网络
- 目标：$\mathcal{L}_{\text{Stage3}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$

---

## 关键公式

### 公式 1: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{\text{video}} = \mathbb{E} \left[ \left\| v_\theta^{\text{video}}\!\left(z_{t+1}^{\tau_v},\, \tau_v \mid z_t^0, l\right) - \left(\epsilon_v - z_{t+1}^0\right) \right\|_2^2 \right]
$$

**含义**: 训练 Video DiT 预测从当前帧潜变量 $z_t^0$ 到下一帧的视频流速度场。

**符号说明**:
- $v_\theta^{\text{video}}$: Video DiT 的速度预测网络，参数为 $\theta$
- $z_{t+1}^{\tau_v}$: 下一帧潜变量在流时间步 $\tau_v$ 处的加噪版本
- $\tau_v \sim \mathcal{U}(0, 1)$: 视频流时间步，均匀采样
- $z_{t+1}^0$: 下一帧的干净潜变量
- $\epsilon_v \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $l$: 语言指令条件

### 公式 2: [[Flow Matching|运动流匹配损失]]

$$
\mathcal{L}_{\text{motion}} = \mathbb{E} \left[ \left\| v_\phi^{\text{motion}}\!\left(m_t^{\tau_a},\, \tau_a \mid h_t^{\tau_f}, p_t, e\right) - \left(\epsilon_m - m_t^0\right) \right\|_2^2 \right]
$$

**含义**: 训练 Motion DiT 以 Video DiT 隐状态为条件，预测运动潜变量的流速度场。

**符号说明**:
- $v_\phi^{\text{motion}}$: Motion DiT 的速度预测网络，参数为 $\phi$
- $m_t^{\tau_a}$: 运动潜变量在流时间步 $\tau_a$ 处的加噪版本
- $\tau_a \sim \mathcal{U}(0, 1)$: 运动流时间步，均匀采样
- $h_t^{\tau_f}$: Video DiT 在固定时间步 $\tau_f$ 处输出的隐状态
- $p_t$: 机器人本体状态（关节角度、速度等）
- $e$: 体型嵌入标签（embodiment tag）
- $m_t^0$: 干净的目标运动潜变量
- $\epsilon_m \sim \mathcal{N}(0, I)$: 标准高斯噪声

### 公式 3: 联合训练损失（Stage 2 & 3）

$$
\mathcal{L}_{\text{Stage2/3}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}
$$

**含义**: 联合优化视频预测和运动预测，使视频世界模型与运动策略协同更新。

### 公式 4: 视频前向加噪（用于提取条件特征）

$$
z_t^{\tau_f} = \tau_f \cdot \epsilon_v + (1 - \tau_f) \cdot z_t^0, \quad \tau_f \approx 1
$$

**含义**: 在推理时，将当前帧潜变量 $z_t^0$ 加噪到时间步 $\tau_f \approx 1$（接近纯噪声），使 Video DiT 的隐状态编码丰富的场景动态信息而无需完整去噪迭代。

**符号说明**:
- $z_t^{\tau_f}$: 加噪后的潜变量，用于 Video DiT 一次前向传播
- $\tau_f \approx 1$: 固定的大流时间步，使隐状态处于"将发生什么"的感知最强区域

### 公式 5: 离散令牌恢复（推理阶段）

$$
\hat{k}_t = \operatorname*{arg\,min}_{k \in \mathcal{K}} \| \tilde{k}_t - k \|
$$

**含义**: 将 Motion DiT 预测的连续松弛值 $\tilde{k}_t$ 通过最近邻取整恢复为 [[SONIC]] 编码本中的合法离散索引。

**符号说明**:
- $\tilde{k}_t$: Motion DiT 输出的连续标量（对应 SONIC 码本索引的松弛）
- $\hat{k}_t$: 取整后的离散索引
- $\mathcal{K}$: SONIC VQ 码本中所有合法索引集合

---

## 关键图表

### Figure 1: MotionWAM 系统概览

![Figure 1](https://arxiv.org/html/2606.09215v1/x2.png)

**说明**: MotionWAM 在 Unitree G1 上的实物演示轨迹，展示腰部控制、高度调节、蹲伏运动、身体-手部协调以及任务驱动的足部交互（如踢球）。体现了统一全身运动策略的能力，这是分层上下身控制无法实现的。

### Figure 2: 系统架构与三阶段训练

![Figure 2](https://arxiv.org/html/2606.09215v1/x3.png)

**说明**: Dual-DiT 架构及三阶段训练流程。**Stage 1**：Video DiT 单独在自我中心视频上预训练；**Stage 2**：挂载 Motion DiT，在异构 G1 数据集上联合训练；**Stage 3**：在遥操作全身演示上端到端微调。Motion DiT 从 Video DiT 隐状态中提取未来场景动态信息以预测运动潜变量。

### Figure 3: 九任务真实场景套件

![Figure 3](https://arxiv.org/html/2606.09215v1/x4.png)

**说明**: 在 Unitree G1 上设计的九个全身运动操作任务，每个任务都要求超越纯平衡维持的主动腿部与躯干参与：PnP Bottle、Kick Soccer、Retrieve Item、Load Cart、Toss Garbage、Lift Basket、Stock Shelves、Wipe Board、Do Laundry。

### Figure 4: 与 SOTA VLA 的任务成功率对比

![Figure 4](https://arxiv.org/html/2606.09215v1/x5.png)

**说明**: MotionWAM 在九任务上的逐任务成功率（20 次试验），对比 GR00T-N1.7、π₀.₅、Qwen3DiT、Diffusion Policy、ACT。MotionWAM 在需要全身协调的任务上领先最大（Kick Soccer +40%、Wipe Board +45%、Do Laundry +30%）。

### Figure 5: 典型失败案例

![Figure 5](https://arxiv.org/html/2606.09215v1/x6.png)

**说明**: 九个任务上 MotionWAM 的代表性失败模式。大多数失败源于**视觉基底丢失**：被操作物体离开自我中心视野，或头部摄像头视角偏离训练分布。

### Figure 6: 推理演示轨迹

![Figure 6](https://arxiv.org/html/2606.09215v1/x7.png)

**说明**: MotionWAM 在九个真实任务上的代表性推理演示，展示从自我中心视角驱动全身协调动作的完整轨迹。

### Table 1: 与 SOTA VLA 的整体成功率对比

| 方法 | 平均成功率 |
|------|-----------|
| **MotionWAM（本文）** | **76.1%** |
| GR00T-N1.7 | 43.9% |
| π₀.₅ | ~20% |
| Qwen3DiT | ~0%（运动任务失败）|
| Diffusion Policy | 大多数任务失败 |
| ACT | 低性能 |

**表格说明**: MotionWAM 超越最强 VLA 基线 GR00T-N1.7 超过 30 个百分点，整体成功率领先显著。Qwen3DiT 在需要运动的任务上完全失败。

### Table 2: 消融实验（平均成功率）

| 配置 | 平均成功率 |
|------|-----------|
| w/o Stage 1（无自我中心视频预训练） | 59.0% |
| w/o Stage 2（无跨体型动作后训练） | 42.0% |
| **Full MotionWAM（完整三阶段）** | **70.0%** |

**关键发现**: Stage 2 跨体型后训练贡献最大（去除后下降 28 个百分点），Stage 1 自我中心预训练次之（去除后下降 11 个百分点）。两个阶段缺一不可。

### Table 3: 推理频率对比

| 方法 | 推理频率 |
|------|---------|
| Qwen3DiT | 9.0 Hz |
| GR00T-N1.7 | 6.5 Hz |
| **MotionWAM** | **4.9 Hz** |
| Cosmos Policy | 0.7 Hz |

**关键发现**: MotionWAM 比 Cosmos Policy 快 7 倍（4.9 Hz vs 0.7 Hz），实现了实时全身控制；同时仅比 GR00T-N1.7 慢 1.6 Hz，但任务成功率大幅领先。

### Table 4: 训练配置（Stage 1–3）

| 超参数 | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| GPU 数 | 128 | 32 | 8 |
| 训练步数 | 100k | 50k | 15k |
| 优化器 | AdamW | AdamW | AdamW |
| 目标函数 | $\mathcal{L}_{\text{video}}$ | $\mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$ | $\mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$ |

### Table 5: Stage 1 数据混合配比

| 数据来源 | 比例 | 具体数据集 |
|---------|------|-----------|
| 人类自我中心 | 30% | EgoDex |
| G1 类人形 | 50% | GR00T、RoboCOIN、Humanoid-Everyday、UnifoLM-WBT、PSI |
| 其他机器人 | 20% | R1_Lite、AIDA-L |
| **总计** | ~2136 小时 | — |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自我中心视频混合（Stage 1） | ~2136 小时 | 人类+G1+其他机器人 | Stage 1 预训练 |
| Unitree G1 异构数据 | — | 多来源 G1 演示 | Stage 2 后训练 |
| 遥操作全身演示（Stage 3） | 每任务 200 条 | PICO VR + XRoboToolkit | Stage 3 微调 |

### 实现细节

- **Video DiT**: Cosmos-Predict2.5-2B（冻结 VAE 和文本编码器，训练 Transformer 主体）
- **Motion DiT**: DiT-B，隐藏维度 2560，动作/状态维度 64
- **优化器**: AdamW，[[Flow Matching|流匹配]]目标
- **Stage 1**: 128 GPU，100k 步
- **Stage 2**: 32 GPU，50k 步
- **Stage 3**: 8 GPU，15k 步
- **硬件（推理）**: NVIDIA RTX 4090 工作站（策略服务器）+ RTX A100 部署
- **机器人**: [[Unitree G1]] + 双 [[ALOHA]] ALOHA2 夹爪
- **相机**: Intel RealSense D435i（头部安装，自我中心视角）
- **[[SONIC]] 控制频率**: 50 Hz（底层关节控制）

### 可视化结果

MotionWAM 能够完成需要全身协调的复杂任务：从蹲下捡地面物品（Retrieve Item），到用脚踢球（Kick Soccer），再到需要大范围身体移动的叠衣服（Do Laundry）。在这些任务上，GR00T-N1.7 因为腿部策略固定于平衡控制而无法执行。

---

## 批判性思考

### 优点

1. **实时性突破**: 通过"一次前向传播提取 Video DiT 隐状态"的设计，在保留世界模型物理先验的同时将推理速度提升至 4.9 Hz，是 WAM 领域的重要工程突破。
2. **全身统一表示**: 统一运动潜变量设计优雅，离散 + 连续的混合表示既保留了[[SONIC]]全身控制器的可靠性，又允许策略学习连续的末端执行器动作。
3. **扎实的实验**: 9 任务×20 次试验的真实机器人评估，消融实验覆盖三个训练阶段，结果说服力强。

### 局限性

1. **单体型验证**: Stage 3 微调仅在 Unitree G1 上验证，跨体型泛化性未知。
2. **视野依赖**: 被操作物体离开自我中心视野后性能急剧下降，缺乏记忆机制。
3. **新颖物体泛化**: 没有受控的新颖物体泛化实验，泛化边界不清晰。
4. **新奇性有限**: "使用 Video DiT 中间特征"的思路与 Flash-WAM 等近期工作有重叠，novelty 主要体现在全身统一表示和实时性工程实现上。

### 潜在改进方向

1. 增加外部摄像头或历史帧记忆，缓解视野丢失问题。
2. 探索在更多人形平台（H1、Atlas 等）上的跨体型迁移。
3. 研究是否可以完全端到端训练 SONIC 令牌，而非依赖预训练 VQ 码本。

### 可复现性评估

- [ ] 代码开源（截至论文发布时未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、数据配比均在附录中详细说明）
- [ ] 数据集可获取（Stage 1 数据部分公开，Stage 3 演示数据未公开）

---

## 关联笔记

### 基于

- [[WAM|World Action Model (WAM)]]: 核心范式，MotionWAM 是其在全身人形控制的延伸
- [[Cosmos-Predict2.5|Cosmos-Predict2.5-2B]]: Video DiT 基础模型
- [[SONIC]]: 全身运动控制器，提供离散运动令牌框架

### 对比

- [[GR00T N1.5|GR00T-N1.7]]: 最强 VLA 基线，MotionWAM 超越其 32 个百分点
- [[Pi05|π₀.₅]]: 另一强基线，成功率约 20%
- [[Flash-WAM]]: 同期 WAM 实时化工作，思路相近

### 方法相关

- [[Flow Matching|流匹配（Flow Matching）]]: 视频和运动预测的核心训练目标
- [[Loco-Manipulation|Loco-Manipulation]]: 全身运动操作任务定义
- [[VAE|变分自编码器（VAE）]]: 视频帧的潜变量编码器
- [[Action Chunking|Action Chunking]]: 运动潜变量可视为延伸的动作表示

### 硬件/数据相关

- [[Unitree G1]]: 实验平台，29 自由度人形机器人
- [[ALOHA|ALOHA2 夹爪]]: 双臂末端执行器

---

## 速查卡片

> [!summary] MotionWAM
> - **核心**: 将 Video DiT 中间去噪特征作为物理先验，驱动统一全身运动潜变量，实现实时人形 Loco-Manipulation
> - **方法**: Dual-DiT（Video DiT + Motion DiT）+ 三阶段训练 + SONIC 全身控制器
> - **结果**: 76.1% 成功率（9 任务），比 GR00T-N1.7（43.9%）高 32 个百分点；4.9 Hz 实时推理，比 Cosmos Policy 快 7 倍
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-06-09*
