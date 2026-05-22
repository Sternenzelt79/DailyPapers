---
title: "stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation"
method_name: "stable-worldmodel"
authors: [Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, reproducibility, benchmarking, planning, embodied-ai, evaluation, data-infrastructure]
zotero_collection: 3-Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2605.21800
created: 2026-05-22
---

# 论文笔记：stable-worldmodel — A Platform for Reproducible World Modeling Research

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Brown University · Meta FAIR · Mila |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[LeWM]] · [[DINO-WM]] · [[PLDM]] · [[TD-MPC2]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.21800) / [HTML](https://arxiv.org/html/2605.21800) |

---

## 一句话总结

> 一个统一的开源平台（swm），通过标准化数据层、规划控制接口和多环境评估套件，解决[[World Model|世界模型]]研究中的代码碎片化、数据瓶颈和缺乏 OOD 鲁棒性基准三大难题。

---

## 核心贡献

1. **高性能数据层**：基于 [[Lance]] 列式存储，本地吞吐量达 4,815 样本/秒（vs HDF5 的 1,416 样本/秒），支持 MP4、HDF5、[[LeRobot]] 格式转换及 S3 云端流式读取。
2. **标准化控制层**：统一 MPCPolicy 接口，内置 7 种规划求解器（CEM、MPPI、GRASP 等），任意[[World Model|世界模型]]均可无缝接入。
3. **多维度评估套件**：跨 5 大环境族，支持视觉、几何、物理三类[[因素变异|Factors of Variation (FoV)]]干预，系统评估 OOD 泛化能力。

---

## 问题背景

### 要解决的问题

[[World Model|世界模型]]研究面临三大基础设施瓶颈：

1. **代码库不稳定**：关键算法（如 [[Cross-Entropy Method|CEM]]）在 TDMPC、PLDM、DINO-WM、LeWM、V-JEPA2 等至少 5 篇论文中被独立重新实现，实现质量参差不齐，难以公平对比。
2. **视频数据加载缓慢**：传统逐帧存储带来高 I/O 开销和存储冗余；压缩视频虽节省空间却严重降低随机访问性能。
3. **缺乏鲁棒性基准**：现有基准主要评估训练分布内性能，无法判断模型是否学到了可复用的动力学规律，还是仅记住了数据偏差。

### 现有方法的局限

- [[DINO-WM]]、[[PLDM]]、[[LeWM]]、[[TD-MPC2]] 等各自有独立代码库，评估协议不统一
- 数据管道各异，复现时需大量工程投入
- 几乎没有系统性的[[分布外泛化|Out-of-Distribution (OOD)]]评估

### 本文的动机

通过构建最小侵入性的统一平台，让研究者在保留自己训练框架的同时，享受标准化的数据收集、评估和控制基础设施，从而降低研究门槛、加速可靠进展。

---

## 方法详解

### 平台架构

**stable-worldmodel (swm)** 采用三层极简抽象设计：

- **数据层（Data Layer）**：[[Lance]] 格式存储，统一数据收集 → 训练 → 评估管道
- **控制层（Control Layer）**：[[Model Predictive Control|MPC]] Policy 统一接口，解耦世界模型与规划求解器
- **评估层（Evaluation Layer）**：跨环境族、可控变异因素的鲁棒性评估套件

![[StableWorldModel_fig1_overview.jpeg]]

**说明**：数据从 World 环境收集后存入 Lance 格式，用于训练世界模型基线；训练好的模型与 Solver 组合成 MPCPolicy，再回到 World 中执行规划与评估。

### 核心模块

#### 模块1：World（统一环境包装器）

**设计动机**：提供单一入口点支持数据收集、策略执行和评估，同时支持[[因素变异|可控干预]]。

**具体实现**：
- 与 [[Gymnasium]] 兼容的统一接口
- 支持向量化执行与渲染
- 内置三类干预控制器：
  - **视觉变异**：颜色、光照、纹理、遮挡
  - **物理变异**：质量、摩擦、重力、动力学参数
  - **几何变异**：形状、大小、位置
- 支持"边界级别 FoV"，用于 Atari 等无法访问模拟器内部的环境

#### 模块2：Policy（策略接口）

**设计动机**：将策略类型与底层模型解耦，支持灵活组合。

**具体实现**：三类策略统一接口：
- **随机策略**：用于数据收集基线
- **专家策略**（如 [[Soft Actor-Critic|SAC]]）：用于收集高质量演示数据
- **MPCPolicy**：将任意[[World Model|世界模型]]与规划 Solver 包装为统一 MPC 控制器，每步编码当前观测后委托 Solver 求解

#### 模块3：Solver（规划求解器）

**设计动机**：标准化规划算法实现，消除各论文间的实现差异。

**具体实现**：Solver 仅需世界模型提供 `get_cost` 方法，内置 7 种求解器：

| 类别 | 算法 |
|------|------|
| 采样方法 | Predictive Sampling、[[Cross-Entropy Method\|CEM]]、iCEM、[[MPPI]] |
| 梯度方法 | Gradient Descent、Projected Gradient Descent、[[GRASP]] |

#### 模块4：高性能数据层

**设计动机**：解决传统存储在随机访问速度与存储效率之间的两难困境。

**具体实现**：
- 采用 [[Lance]] 列式格式（零拷贝操作、原生版本控制、云对象存储无缝流式传输）
- 提供 MP4、[[HDF5]]、[[LeRobot]] 格式一键转换工具
- 支持本地存储和 S3 远程流式读取

---

## 关键公式

### 公式1：[[Optimal Control|有限时域最优控制问题]]

$$
\pi^* = \arg\min_{\pi} \sum_t c(s_t, a_t)
$$

$$
\text{s.t.} \quad s_{t+1} = P(s_t, a_t), \quad a_t = \pi(s_t)
$$

**含义**：世界模型规划的核心目标——在预测器 $P$ 约束下，找到最小化累积代价的策略 $\pi^*$。

**符号说明**：
- $\pi^*$：最优策略
- $c(s_t, a_t)$：时间步 $t$ 的阶段代价（任务相关）
- $s_t$：时间步 $t$ 的状态（可以是潜空间中的表示）
- $a_t$：时间步 $t$ 的动作
- $P$：[[World Model|世界模型]]预测器，近似真实环境动力学

### 公式2：[[Encoder-Predictor|编码器-预测器框架]]

$$
z_t = E(o_t), \quad z_{t+1} = P(z_t, a_t)
$$

**含义**：编码器 $E$ 将原始观测映射到紧凑潜表示，预测器 $P$ 在潜空间中学习动力学模型。

**符号说明**：
- $o_t \in \mathbb{R}^n$：原始观测（图像像素）
- $z_t \in \mathbb{R}^D$：潜在状态表示
- $E: \mathbb{R}^n \to \mathbb{R}^D$：编码器网络
- $P: \mathbb{R}^D \times \mathbb{R}^A \to \mathbb{R}^D$：动力学预测器

---

## 关键图表

### Figure 1：平台整体流程

![[StableWorldModel_fig1_overview.jpeg]]

**说明**：stable-worldmodel 的端到端工作流。数据从 World 环境高效收集并存入 Lance 格式，训练多种世界模型基线，再通过统一 Solver 接口进行规划控制。

### Figure 2：支持的环境族与视觉变异

![[StableWorldModel_fig2_classic.png]]
![[StableWorldModel_fig2_mujoco.png]]
![[StableWorldModel_fig2_arcade.png]]
![[StableWorldModel_fig2_robotics.png]]
![[StableWorldModel_fig2_openworld.png]]
![[StableWorldModel_fig2_classic_perturbed.png]]
![[StableWorldModel_fig2_mujoco_perturbed.png]]
![[StableWorldModel_fig2_arcade_perturbed.png]]
![[StableWorldModel_fig2_robotics_perturbed.png]]
![[StableWorldModel_fig2_openworld_perturbed.png]]

**说明**：上行为各环境族的默认（未扰动）渲染；下行为所有视觉因素变异（智能体、物体、场景、几何、光照）联合扰动后的结果。物理参数变化（质量、密度、重力、摩擦）未在此可视化。

### Figure 3：数据格式性能对比

> Figure 3 为矢量图（matplotlib 绘制），未嵌入位图。详见论文 Figure 3：对比 Lance、HDF5、MP4 在本地和 S3 的吞吐量及磁盘占用。

**说明**：基于 Push-T 数据集的存储格式性能比较。（左）本地与 S3 远程流式读取的数据加载吞吐量对比；（右）磁盘占用量对比。Lance 在本地吞吐量上比 HDF5 高出约 3.4×。

### Figure 4：预测误差与规划成功率的相关性

> Figure 4 为矢量图（matplotlib 绘制），未嵌入位图。详见论文 Figure 4：LeWM 在 Push-T 上跨 4 级分布偏移的 MSE 分布直方图。

**说明**：使用 [[LeWM]] 在 Push-T 上，成功（蓝色）和失败（红色）规划轨迹的预测 MSE 分布，跨四个递增的[[分布外泛化|分布偏移]]等级。揭示了**预测误差悖论**：预测误差低不代表规划成功，在 OOD 设置下二者相关性极差。

### Figure 5：视觉干扰物与因素变异的鲁棒性分析

> Figure 5 为矢量图（matplotlib 绘制），未嵌入位图。详见论文 Figure 5：干扰物数量 vs 成功率 + FoV 条件下各方法对比折线图。

**说明**：（左）视觉干扰物数量对规划成功率的影响——模型对少量干扰物有一定容忍度，但超过阈值后呈二次衰减；（右）不同[[因素变异|FoV]]类别下各方法的规划成功率，揭示颜色和形状变化是最主要的性能杀手。

### Table 1：分布内性能对比（成功率 %）

| Method | Push-T | OGBench-Cube |
|--------|--------|--------------|
| TD-MPC2 | 12 | 4 |
| GCBC | 75 | **86** |
| PLDM | 78 | 62 |
| DINO-WM | 92 | **86** |
| **LeWM** | **94** | 72 |

**说明**：[[LeWM]] 在 Push-T 上以 94% 成功率领先，但在 OGBench-Cube 上不如 DINO-WM 和 GCBC，显示不同方法的优势场景不同。[[TD-MPC2]] 在两项任务上表现均明显落后，可能与其无解码器设计不适合目标条件规划有关。

### Table 2(a)：Push-T 上因素变异下的规划成功率（%）

| 变异类型 | 作用实体 | LeWM | PLDM | DINO-WM |
|---------|---------|------|------|---------|
| None（基线） | — | 50.8 | 50.8 | 20.0 |
| Color | Agent | 12.0 | 8.0 | 18.0 |
| Color | Block | 22.0 | 18.0 | 18.0 |
| Color | Canvas | 6.0 | 6.0 | 10.0 |
| Size | Agent | 22.0 | 18.0 | 4.0 |
| Size | Block | 20.0 | 18.0 | 16.0 |
| Shape | Agent | 26.0 | **52.0** | 18.0 |
| Shape | Block | 12.0 | 14.0 | 8.0 |

**关键发现**：即便是轻微的视觉变异（如颜色变化），所有方法成功率均大幅下降（Canvas 颜色变化后下降超 85%）。PLDM 在 Shape-Agent 变异下意外优于 LeWM，说明不同归纳偏置对不同扰动类型的鲁棒性差异显著。

---

## 实验

### 环境与数据集

| 环境族 | 代表任务 | 特点 |
|--------|---------|------|
| Classic Control | CartPole、Push-T | 低维 2D 控制，快速验证 |
| MuJoCo | DMControl Suite | 连续控制，物理仿真 |
| Atari (ALE) | 多款游戏 | 高维离散动作 |
| Manipulation | OGBench-Cube | 目标条件机器人操作 |
| Open-World | Craftax | 部分可观测，长时序规划 |

### 实现基线

| 类别 | 方法 | 特点 |
|------|------|------|
| 目标条件 RL | GCBC | 行为克隆，无需世界模型 |
| 目标条件 RL | GCIQL / GCIVL | 隐式 Q-learning 变体 |
| 潜在世界模型 | [[DINO-WM]] | 冻结 DINOv2 编码器 + ViT 预测器 |
| 潜在世界模型 | [[PLDM]] | JEPA 风格，多正则项稳定训练 |
| 潜在世界模型 | [[LeWM]] | 极简 JEPA，端到端从像素训练 |
| 无解码器 WM | [[TD-MPC2]] | 离散奖励/价值预测引导 |

### 关键发现

1. **预测误差悖论**：轨迹级预测 MSE 与规划成功率在 OOD 设置下相关性极低，说明传统的预测精度评估指标不足以衡量世界模型的规划能力。
2. **视觉扰动的脆弱性**：即便轻微的视觉变化，大多数模型规划成功率急剧下降，证实当前世界模型更多依赖训练分布中的可利用相关性，而非可复用的环境动力学。
3. **干扰物的二次衰减**：视觉干扰物数量少时模型有一定容忍性，但超过阈值后成功率呈二次衰减，揭示注意力竞争机制的脆弱性。
4. **数据吞吐提升约 3.4×**：Lance 格式相比 HDF5 本地吞吐量提升显著（4,815 vs 1,416 样本/秒），缓解了大规模世界模型训练的数据瓶颈。

---

## 批判性思考

### 优点

1. **工程价值高**：切中世界模型研究的实际痛点，数据层和控制层的标准化有望大幅降低社区复现成本。
2. **OOD 评估设计用心**：三类可控变异因素（视觉/几何/物理）组合设计细致，是目前世界模型领域最系统的鲁棒性评估框架之一。
3. **来自同一作者团队的可信度**：论文作者与 [[LeWM]]、[[PLDM]] 重合度高，对各基线实现质量有保障。

### 局限性

1. **评估环境仍以仿真为主**：缺乏真实机器人评估，sim-to-real 迁移能力未验证。
2. **扰动强度设定主观**：FoV 的"轻微扰动"定义依赖超参数设定，不同强度下结论可能不同。
3. **缺乏生成式世界模型基线**：Dreamer 系列等重要的基于重建的世界模型未纳入对比。
4. **OGBench 结果偏低**：多数方法在 OGBench-Cube 上成功率仅个位数到 86%，说明平台在难任务上的鲁棒性评估能力有待进一步验证。

### 潜在改进方向

1. 扩展到真实机器人数据（支持异步实时交互）
2. 引入在线训练机制，支持持续学习范式
3. 纳入 Dreamer 等生成式世界模型基线

### 可复现性评估

- [x] 代码开源（stable-worldmodel 平台本身即为开源工具）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（各基线实现细节在论文中有描述）
- [x] 数据集可获取（支持标准公开数据集格式转换）

---

## 关联笔记

### 基于

- [[Lance]]：高性能列式存储格式，平台数据层核心
- [[JEPA]]：联合嵌入预测架构，多个基线的训练范式基础

### 对比基线

- [[LeWM]]：同一作者团队的极简 JEPA 世界模型
- [[DINO-WM]]：基于 DINOv2 冻结编码器的世界模型
- [[PLDM]]：多正则项 JEPA 世界模型
- [[TD-MPC2]]：无解码器的时序差分 MPC 世界模型

### 方法相关

- [[Cross-Entropy Method]]：平台内置的采样规划核心算法
- [[Model Predictive Control]]：swm 控制层的核心框架
- [[MPPI]]：基于信息论的采样规划方法
- [[World Model]]：本文评估的核心对象

### 评估相关

- [[Push-T]]：主要评估任务，来自仿真操作环境
- [[OGBench]]：目标条件离线 RL 评估套件
- [[MuJoCo]]：连续控制物理仿真引擎

---

## 速查卡片

> [!summary] stable-worldmodel (swm)
> - **核心**：解决世界模型研究的代码碎片化、数据瓶颈和 OOD 评估缺失三大问题的开源平台
> - **方法**：Lance 高性能数据层 + 统一 MPCPolicy 接口 + 多环境可控变异评估套件
> - **关键发现**：预测误差与规划成功率在 OOD 下相关性极低；视觉轻微扰动即导致大多数模型大幅退化
> - **代码**：[stable-worldmodel on arXiv](https://arxiv.org/abs/2605.21800)

---

*笔记创建时间: 2026-05-22*
