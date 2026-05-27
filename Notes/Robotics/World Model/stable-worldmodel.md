---
title: "stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation"
method_name: "stable-worldmodel"
authors: [Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, reproducibility, benchmark, planning, robotics, offline-rl]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.21800
created: 2026-05-22
---

# 论文笔记：stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila, NYU, ServiceNow |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[DINO-WM]], [[PLDM]], [[LeWM]], [[TD-MPC2]], [[GCBC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.21800) |

---

## 一句话总结

> stable-worldmodel (swm) 是一个统一的世界模型研究开源平台，通过高性能数据层、标准化规划接口和系统性分布偏移 benchmark，揭示当前世界模型在零样本泛化上的脆弱性。

---

## 核心贡献

1. **高性能数据层**: 基于 [[Lance]] 列式存储格式，本地吞吐量达 4,815 样本/秒（约是 HDF5 的 3.4 倍），原生支持 MP4、HDF5、LeRobot 数据集的一键转换
2. **统一规划接口**: 实现 CEM、[[MPPI]]、Predictive Sampling（采样类）以及 Gradient Descent、Projected Gradient Descent、[[GRASP]]（梯度类）六种规划求解器，通过统一 API 与任意世界模型对接
3. **系统性泛化 Benchmark**: 构建涵盖视觉、几何、物理变异因子（FoV）的评测套件，证明当前世界模型在轻微分布偏移下规划成功率急剧下降

---

## 问题背景

### 要解决的问题

[[世界模型|World Model]] 研究存在三大瓶颈：
1. **代码碎片化**：各论文使用独立一次性代码库，实验无法复现
2. **数据加载低效**：视频训练数据的 I/O 吞吐量成为瓶颈
3. **缺乏标准化泛化 Benchmark**：不同工作使用不兼容的评测协议，无法横向比较

### 现有方法的局限

- 现有 [[世界模型]] 实现分散，无统一接口
- 评测仅在 in-distribution 数据上进行，无法反映真实泛化能力
- 数据格式各异（HDF5/MP4/LeRobot），互操作性差

### 本文的动机

通过标准化基础设施降低研究门槛，同时通过系统性分布偏移测试暴露当前世界模型的真实局限，为可信赖的世界模型研究提供基础。

---

## 方法详解

### 系统架构

stable-worldmodel (swm) 采用三层最小化抽象架构：

- **World 层**: 统一环境包装器，支持 Gymnasium 兼容仿真器，提供向量化执行和可控变异因子
- **Policy 层**: 简单观测→动作映射接口，包含用于世界模型+规划求解器组合的 `MPCPolicy` 封装
- **Solver 层**: 自包含规划算法实现，具有完整的单元测试

整体流程：
- **数据采集**: 从 World 环境中高效采集轨迹数据
- **模型训练**: 使用提供的 [[潜在世界模型|Latent World Model]] baseline 训练
- **规划控制**: 训练好的世界模型配合 Solver 进行 [[模型预测控制|MPC]] 规划

### 核心模块

#### 模块1：数据层（Lance Backend）

**设计动机**: 利用 [[Lance]] 列式 ML 优化格式解决视频训练的 I/O 瓶颈

**具体实现**:
- Lance 格式本地吞吐：**4,815 样本/秒**，支持零开销远程 S3 流式传输（3,184 样本/秒）
- 对比 HDF5 本地（1,416 样本/秒）和视频本地（1,331 样本/秒），提升约 3.4 倍
- 一键转换工具：`MP4 → Lance`、`HDF5 → Lance`、`LeRobot → Lance`

#### 模块2：规划求解器（Planning Solvers）

**设计动机**: 统一规划接口使世界模型可即插即用

**采样类方法**:
- [[CEM|Cross-Entropy Method (CEM)]]: 通过精英样本迭代更新动作分布
- [[MPPI|Model Predictive Path Integral]]: 基于重要性采样的轨迹优化
- Predictive Sampling: 前向预测采样

**梯度类方法**:
- Gradient Descent / Projected Gradient Descent: 直接对代价函数梯度优化
- [[GRASP]]: 梯度辅助采样规划

#### 模块3：变异因子（Factors of Variation, FoV）

**设计动机**: 提供系统性零样本泛化评测，暴露模型真实鲁棒性

**三类可控扰动**:
- **视觉因子**: 颜色、光照、纹理、遮挡
- **物理因子**: 质量、摩擦系数、重力
- **几何因子**: 物体形状和尺寸

---

## 关键公式

### 公式1: [[模型预测控制|规划目标函数]]

$$
\pi^* = \arg\min_{\pi} \sum_t c(\mathbf{s}_t, \mathbf{a}_t) \quad \text{s.t.} \quad \mathbf{s}_{t+1} = \mathcal{P}(\mathbf{s}_t, \mathbf{a}_t), \quad \mathbf{a}_t = \pi(\mathbf{s}_t)
$$

**含义**: 通过世界模型 $\mathcal{P}$ 预测未来状态序列，寻找最小化累积代价的最优策略 $\pi^*$

**符号说明**:
- $\pi^*$: 最优规划策略
- $c(\mathbf{s}_t, \mathbf{a}_t)$: 时刻 $t$ 的阶段代价函数
- $\mathcal{P}: \mathbb{R}^D \times \mathbb{R}^A \to \mathbb{R}^D$: 世界模型预测器，从当前潜在状态+动作预测下一状态
- $\mathbf{s}_t \in \mathbb{R}^D$: 潜在状态表示
- $\mathbf{a}_t \in \mathbb{R}^A$: 动作向量

### 公式2: [[编码器|编码器映射]]

$$
\mathcal{E}: \mathbb{R}^n \to \mathbb{R}^D
$$

**含义**: 将高维观测空间 $\mathbb{R}^n$ 映射到低维潜在状态空间 $\mathbb{R}^D$

**符号说明**:
- $n$: 原始观测维度（如像素数）
- $D$: 潜在空间维度

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2605.21800v1/x1.png)

**说明**: stable-worldmodel 整体工作流。数据从 World 环境中高效采集，用于训练世界模型基线，训练好的模型配合规划求解器用于控制任务。

### Figure 2: 支持的环境族

**2a — Classic Control（经典控制）**

![Classic Control](https://arxiv.org/html/2605.21800v1/figures/classic.png)

**2b — MuJoCo 连续控制**

![MuJoCo](https://arxiv.org/html/2605.21800v1/figures/dmc.png)

**2c — Arcade 游戏（Atari）**

![Arcade Games](https://arxiv.org/html/2605.21800v1/figures/ale.png)

**2d — 机器人操作（Push-T, OGBench）**

![Robotics](https://arxiv.org/html/2605.21800v1/figures/robotics.png)

**2e — 开放世界游戏（Craftax）**

![Craftax](https://arxiv.org/html/2605.21800v1/figures/craftax.png)

**说明**: 顶部一行为默认渲染，底部一行为视觉变异因子联合扰动后的渲染效果，覆盖经典控制、MuJoCo、Arcade、机器人操作、开放世界五个环境族。

**视觉变异对比（底部一行）**:

![Classic FoV](https://arxiv.org/html/2605.21800v1/figures/wrapper_fig2_classic.png) ![MuJoCo FoV](https://arxiv.org/html/2605.21800v1/figures/wrapper_fig2_mujoco.png) ![Arcade FoV](https://arxiv.org/html/2605.21800v1/figures/wrapper_fig2_arcade.png) ![Robotics FoV](https://arxiv.org/html/2605.21800v1/figures/wrapper_fig2_robotics.png) ![OpenWorld FoV](https://arxiv.org/html/2605.21800v1/figures/wrapper_fig2_openworld.png)

### Figure 3: 数据格式性能对比

![Figure 3](https://arxiv.org/html/2605.21800v1/x2.png)

**说明**: Push-T 数据集下不同存储格式的对比。左图为有/无缓存条件下本地和 S3 远程存储的吞吐量，右图为磁盘占用。Lance 在各场景下均达到最高吞吐。

### Table: 数据格式吞吐量对比（样本/秒）

| 格式 | 无缓存 | 有缓存 |
|------|--------|--------|
| HDF5 (本地) | 1,416 | 1,474 |
| HDF5 (S3) | 9 | 757 |
| **Lance (本地)** | **4,815** | **4,431** |
| **Lance (S3)** | **3,184** | **3,253** |
| Video (本地) | 1,331 | 1,348 |

**关键发现**: Lance 本地吞吐约为 HDF5 的 3.4 倍；Lance S3 流式传输吞吐（3,184）显著优于 HDF5 S3 缓存后（757），证明 Lance 对远程存储的原生支持优势。

### Figure 4: 预测 MSE 与规划成功率的关系

![Figure 4](https://arxiv.org/html/2605.21800v1/x3.png)

**说明**: Push-T 上使用 [[LeWM]] 在四个递增分布偏移级别下，规划成功轨迹（蓝色）和失败轨迹（红色）的轨迹级预测 MSE 分布对比。关键发现：预测误差与规划成功率相关性差，说明是分布外输入（而非原始误差大小）驱动了失败。

### Figure 5: Push-T 鲁棒性分析

![Figure 5](https://arxiv.org/html/2605.21800v1/x4.png)

**说明**: (a) 随视觉干扰物增加的成功率演化曲线；(b) 颜色/尺寸/形状等变异因子施加到不同实体时的规划成功率。成功率呈二次衰减模式——模型能容忍极少量干扰，但在轻微视觉变化下即迅速退化（从约 50% 降至 20% 以下）。

### Figure 6（附录）: 边界级 FoV 视觉包装器

![Figure 6](https://arxiv.org/html/2605.21800v1/x5.png)

**说明**: 附录 C.2 中展示边界级别的视觉变异因子实现细节，通过 Visual Wrappers 实现可控边界扰动。

### Figure 7（附录）

![Figure 7](https://arxiv.org/html/2605.21800v1/x6.png)

**说明**: 附录中的补充分析图。

### Table 1: In-Distribution 基线对比（成功率 %）

| 方法 | Push-T | OGBench-Cube |
|------|--------|--------------|
| TD-MPC2 | 12 | 4 |
| GCBC | 75 | 84 |
| [[LeWM]] | **94** | 72 |
| [[PLDM]] | 78 | 62 |
| [[DINO-WM]] | 92 | **86** |

**关键发现**: TD-MPC2 在离线设置下成功率极低（12%/4%），因为离线训练数据不包含其所需的探索性动作分布，导致动作空间分布外生成。LeWM 在 Push-T 上最优，DINO-WM 在 OGBench-Cube 上最优。

### Table 2: 分布偏移下的规划成功率（Push-T，%）

| 变异因子 | 实体 | LeWM | PLDM | DINO-WM |
|----------|------|------|------|---------|
| 无（基线） | — | 50.8 | 50.8 | 20.0 |
| 颜色 | Agent | 12.0 | 8.0 | 18.0 |
| 颜色 | Block | 22.0 | 18.0 | 18.0 |
| 颜色 | Canvas | 6.0 | 6.0 | 10.0 |
| 尺寸 | Agent | 22.0 | 18.0 | 4.0 |
| 尺寸 | Block | 20.0 | 18.0 | 16.0 |
| 形状 | Agent | 26.0 | 52.0 | 18.0 |
| 形状 | Block | 12.0 | 14.0 | 8.0 |

**关键发现**: 即使只改变画布颜色（Canvas Color），所有模型成功率骤降至 6-10%（基线 20-50.8%）。PLDM 在 Shape/Agent 变异下保持相对较好（52%），但整体趋势显示当前世界模型对视觉/几何扰动高度敏感。

---

## 实验

### 环境支持

| 环境族 | 代表任务 | 特点 |
|--------|----------|------|
| 经典控制 | CartPole | 低维状态，确定性 |
| MuJoCo | DMControl Suite | 连续控制，高维状态 |
| Arcade | Atari | 离散动作，像素输入 |
| 操作 | Push-T, OGBench | 机器人操作，部分可观 |
| 开放世界 | Craftax | 部分可观，长时序 |

### 实现细节

- **规划求解器**: CEM、MPPI、Predictive Sampling（采样类）+ Gradient Descent、PGD、GRASP（梯度类）
- **数据存储**: Lance 格式（默认），兼容 HDF5、MP4、LeRobot
- **环境接口**: Gymnasium 兼容，支持向量化执行
- **开源协议**: Creative Commons Attribution 4.0

### 核心发现

1. **预测误差 ≠ 规划性能**: 即使模型预测 MSE 相近，规划成功率差异可能很大；失败主要由分布外输入驱动
2. **视觉扰动呈二次衰减**: 成功率随干扰物数量的增加呈二次下降，而非线性
3. **TD-MPC2 离线失效**: 在离线设置下动作漂移严重，在线训练 DMControl 任务表现与 SAC 可比

---

## 批判性思考

### 优点

1. **工程实用性强**: Lance 数据层和统一规划 API 真实解决了研究痛点，平台化思路有助于领域标准化
2. **诊断价值高**: 预测 MSE 与规划成功率相关性差这一发现，对未来世界模型设计有深刻启发
3. **评测体系完整**: FoV 三类扰动（视觉/几何/物理）覆盖面广，可拓展到 sim-to-real 研究

### 局限性

1. **论文贡献以工程为主**: 缺乏新的世界模型算法，核心学术贡献是平台和发现，而非方法创新
2. **评测环境偏向 2D 操作**: Push-T 和 OGBench 相对简单，对复杂 3D 机器人任务的泛化性待验证
3. **FoV 仅测试零样本**: 未探索 few-shot 适应或 domain randomization 训练策略

### 潜在改进方向

1. 引入 domain randomization 训练支持，测试 FoV 是否能提升训练鲁棒性
2. 扩展至真实机器人平台，验证 sim-to-real 转移
3. 基于预测 MSE 与规划成功率解耦的发现，设计新的世界模型训练目标

### 可复现性评估

- [x] 代码开源（平台即产品）
- [ ] 预训练模型（未明确提供）
- [x] 训练细节完整（提供完整实现）
- [x] 数据集可获取（支持标准数据集格式转换）

---

## 关联笔记

### 基于

- [[PLDM]]: 潜在预测世界模型 baseline
- [[LeWM]]: LeRobot 世界模型 baseline
- [[DINO-WM]]: 基于 DINO 特征的世界模型 baseline
- [[Lance]]: 核心数据存储格式

### 对比

- [[TD-MPC2]]: 在线 RL 世界模型，离线设置下失效
- [[GCBC]]: Goal-Conditioned 行为克隆 baseline
- [[DINO-WM]]: 在 OGBench 上取得最佳表现

### 方法相关

- [[模型预测控制|Model Predictive Control (MPC)]]: 核心规划框架
- [[MPPI]]: 采样类规划求解器
- [[CEM]]: Cross-Entropy Method 规划求解器
- [[潜在世界模型|Latent World Model]]: 核心建模范式
- [[世界模型]]: 研究领域

### 数据/环境相关

- [[MuJoCo]]: 连续控制仿真器
- [[OGBench]]: 机器人操作评测数据集
- [[Push-T]]: 主要评测操作任务

---

## 速查卡片

> [!summary] stable-worldmodel (swm)
> - **核心**: 统一世界模型研究平台，解决可复现性、数据效率、标准评测三大痛点
> - **方法**: Lance 数据层 + 六种规划求解器 + FoV 泛化 Benchmark
> - **结果**: 揭示轻微分布偏移（改变颜色/形状）导致规划成功率骤降，预测 MSE 与规划性能相关性差
> - **代码**: [arXiv](https://arxiv.org/abs/2605.21800)

---

*笔记创建时间: 2026-05-22*
