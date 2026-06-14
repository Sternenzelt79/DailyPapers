---
title: "WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation"
method_name: "WEAVER"
authors: [Arnav Kumar Jain, Yilin Wu, Jesse Farebrother, Gokul Swamy, Andrea Bajcsy]
year: 2026
venue: arXiv
tags: [world-model, robotic-manipulation, flow-matching, policy-improvement, test-time-planning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.13672
created: 2026-06-12
---

# 论文笔记：WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila – Québec AI Institute / Université de Montréal / Carnegie Mellon University / McGill University |
| 日期 | June 2026 |
| 项目主页 | [arnavkj1995.github.io/WEAVER](https://arnavkj1995.github.io/WEAVER/) |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13672) / [Code](https://github.com/arnavkj1995/WEAVER) / [HuggingFace](https://huggingface.co/arnavkj1995/WEAVER) |

---

## 一句话总结

> WEAVER 是一个多视角潜空间世界模型，通过流匹配损失和 KV 缓存加速推理，同时满足高保真度、长程一致性和高效性，在机器人操作中实现策略评估（ρ=0.870）、策略提升（+38%）和测试时规划（+14%）。

---

## 核心贡献

1. **三位一体的世界模型设计**: 同时满足 Fidelity（保真度）、Consistency（长程一致性）和 Efficiency（效率）三个核心需求，现有方法均无法同时达到。
2. **多视角潜空间预测架构**: 结合外部摄像头与腕部摄像头的多视角预测，配合 Memory+History 时间结构，显著提升对接触丰富任务的感知能力。
3. **三大下游应用**: 统一框架支持策略评估（ρ=0.870 与真实成功率相关）、合成数据生成用于策略微调（仅 1000 段合成数据匹敌真实数据）、以及测试时最优动作规划（Best-of-N，5-10× 加速对比前作）。

---

## 问题背景

### 要解决的问题

世界模型在机器人学中有三大潜力应用：**策略评估**（无需真实交互估计策略质量）、**策略改进**（生成合成数据微调策略）和**测试时规划**（在线选择最优动作）。然而，目前没有任何机器人世界模型能同时满足三个核心要求：保真度、长程一致性和效率。

### 现有方法的局限

- **[[视频生成模型]]**（如 Ctrl-World）：生成质量高但推理极慢（>14s/chunk），无法用于实时规划。
- **[[JEPA]] 类模型**：潜空间不可解码，无法用于视觉运动策略的评估。
- **[[DreamerV4]]**：从头训练编码器，分布外（OOD）鲁棒性差。
- **专用操作模型**（Ctrl-World）：速度瓶颈使测试时规划不可行。

### 本文的动机

WEAVER 融合三类设计决策：
1. 视频生成领域的 [[Diffusion Forcing]] 和 [[Flow Matching]]
2. 潜空间世界模型的奖励头设计
3. 多视角架构

并结合 KV 缓存、渐进式噪声调度和 [[Rectified Flow Distillation]] 实现高效推理。

---

## 方法详解

### 模型架构

WEAVER 采用**潜空间预测**架构，核心组件包括：

- **输入**: 多视角观测 $o_t$（外部+腕部摄像头）+ 语言指令 $\ell$ + 动作块 $\mathbf{a}_t$
- **编码器**: 预训练 [[VAE]] 编码器 $\mathcal{E}_\psi$（基于 Stable Diffusion 3），将观测映射到潜空间
- **Memory 结构**: 稀疏长程记忆帧（步长 k=5）
- **History 结构**: 最近 2 帧的稠密历史
- **核心模块**: 32 层 [[2D Transformer]] 潜动力学模型 $f_\phi$
- **奖励头**: [[AdaPool]] 聚合后的 MSE 奖励预测
- **Critic 网络**: 基于 λ-return 的价值估计
- **解码器**: 预训练 [[VAE]] 解码器 $\mathcal{D}_\eta$
- **总参数**: 928M

```
外部摄像头 ┐
腕部摄像头 ├─→ [VAE 编码器] → 潜向量 z_t
本体感觉   ┘
                    ↓
         Memory: z_{t-2k}, z_{t-k} (稀疏)
         History: z_{t-m}, ..., z_t (稠密)
                    ↓
         [32层 2D Transformer f_φ] + SPRINT Token Dropping
                    ↓
         预测潜向量 ẑ_{t+1:t+h+1}
                    ↓
         [奖励头 R] → 预测奖励 r̂_t
         [Critic V] → 价值估计
         [VAE 解码器] → 重建观测 ô_t
```

### 核心模块

#### 模块1: 多视角预测（Multi-View Prediction）

**设计动机**: 操作任务中，腕部摄像头捕获手-物接触细节，外部摄像头提供全局场景，仅用单视角无法完整建模动力学。

**具体实现**:
- 同时预测外部摄像头和腕部摄像头的未来潜表示
- 显式预测本体感觉状态（关节角度、夹爪宽度）
- 多视角特征拼接后送入动力学 [[Transformer]]

#### 模块2: Memory-History 时序结构

**设计动机**: 利用 [[长程依赖]] 理解物体持久性，同时用近期帧感知动作后果。

**具体实现**:

$$
\mathbf{z}^{\text{mem}}_t := (\ldots, z_{t-2k}, z_{t-k}), \quad k = 5
$$

$$
\mathbf{z}^{\text{hist}}_t := (z_{t-m}, \ldots, z_t), \quad m = 2
$$

- 稀疏记忆覆盖长程历史，密集历史捕捉近期动作效果
- 两者拼接后作为 Transformer 的上下文输入

#### 模块3: Diffusion Forcing 训练

**设计动机**: 独立采样各时间步的噪声级别，强制模型在任意噪声混合条件下保持一致，提升长程生成的连贯性。

**具体实现**:
- 每个未来时间步 $t+1, \ldots, t+h$ 独立采样噪声时刻 $\tau \sim \mathcal{U}(0,1)$
- 模型学习在部分去噪的未来预测下保持内部一致性

#### 模块4: SPRINT Token Dropping

**设计动机**: 激进地丢弃冗余 patch token，在不明显损失质量的前提下大幅加速推理。

**具体实现**:
- 在 [[2D Transformer]] 的中间层选择性丢弃低显著性 token
- 结合 KV 缓存使推理速度提升 3× 以上

#### 模块5: 奖励模型与 Critic 网络

**设计动机**: 将语言-条件奖励信号直接从潜空间预测，无需解码到像素级别，结合自举 λ-return 支持长程价值估计。

**具体实现**:
- 奖励头：AdaPool 池化潜向量 + MLP，输出每步奖励
- Critic：预测超出想象域的价值估计

---

## 关键公式

### 公式1: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}^{\text{WM}}(\phi) = \mathbb{E}_{x^0_t, x^1_t, \tau}\left[\left\| (x^1_t - x^0_t) - f_\phi(\mathbf{z}^{\text{mem}}_t, \mathbf{z}^{\text{hist}}_t, \mathbf{a}_t, x^\tau_t) \right\|^2_2\right]
$$

**含义**: 流匹配损失，要求模型预测从噪声 $x^0$ 到真实潜向量 $x^1$ 的速度场。

**符号说明**:
- $x^1_t := z_{t+1:t+h+1}$: 真实未来潜向量（ground-truth）
- $x^0_t \sim \mathcal{N}(0, I)$: 高斯噪声起点
- $x^\tau_t := \tau x^1_t + (1-\tau)x^0_t$: $\tau$ 时刻的插值点
- $\tau \sim \mathcal{U}(0, 1)$: 均匀采样的时间步

### 公式2: [[Rectified Flow|潜向量预测（推理过程）]]

$$
\hat{\mathbf{z}}_t \sim f_\phi(\cdot \mid \mathbf{z}^{\text{mem}}_t, \mathbf{z}^{\text{hist}}_t, \mathbf{a}_t)
$$

**含义**: 给定记忆、历史和动作块，通过流匹配采样预测未来 $h$ 步的潜向量序列。

**符号说明**:
- $\mathbf{a}_t$: $h$ 步动作块
- $h$: 预测地平线（imagination horizon）

### 公式3: [[Reward Model|奖励模型损失]]

$$
\mathcal{L}^{\text{reward}} = \left\| R(z_t, \ell) - r_t \right\|^2_2
$$

**含义**: MSE 损失，训练奖励头从潜向量和语言指令预测每步奖励信号。

**符号说明**:
- $R(\cdot)$: 奖励头网络
- $z_t$: 当前潜向量
- $\ell$: 语言指令
- $r_t$: Robometer 提供的真实奖励标注

### 公式4: [[TD Lambda|自举 λ-Return]]

$$
\mathbf{v}^\lambda_t = R(z_t, \ell) + \gamma\left[(1-\lambda)V(z_{t+1}, \ell) + \lambda \mathbf{v}^\lambda_{t+1}\right]
$$

**含义**: 混合蒙特卡洛和自举估计的 λ-return，用于训练 Critic 网络估计长程价值。

**符号说明**:
- $\gamma = 0.995$: 折扣因子
- $\lambda = 0.95$: TD-λ 混合系数
- $V(\cdot)$: Critic 网络（价值函数）

### 公式5: [[Critic Network|Critic 损失]]

$$
\mathcal{L}^{\text{critic}}(V) = \left\| V(z_t, \ell) - \mathbf{v}^\lambda_t \right\|^2_2
$$

**含义**: MSE 损失，使 Critic 网络的预测逼近自举的 λ-return 目标。

### 公式6: [[Monte Carlo Advantage|蒙特卡洛优势估计]]

$$
\hat{A}^b_t = \sum_{\ell=1}^{H} \gamma^{\ell-1} R(\hat{z}^b_{t+\ell}, \ell) + \gamma^H V(\hat{z}^b_{t+H}, \ell) - V(z_t, \ell)
$$

**含义**: 在 $B$ 条并行想象轨迹中，计算每条轨迹相对于当前状态价值估计的优势，用于筛选最优动作。

**符号说明**:
- $b \in \{1, \ldots, B\}$: 第 $b$ 条想象轨迹
- $H = Kh$: 总想象地平线（$K$ 个动作块，每块长度 $h$）
- $\hat{z}^b_{t+\ell}$: 第 $b$ 条轨迹第 $\ell$ 步的预测潜向量

---

## 关键图表

### Figure 1: WEAVER 三大应用概览

![Figure 1](https://arxiv.org/html/2606.13672v1/x1.png)

**说明**: WEAVER 同时满足三个核心需求（保真度、长程一致性、效率），并支持三种下游应用：策略评估（无真实交互估计成功率）、策略提升（生成合成轨迹微调策略）、测试时规划（Best-of-N 动作选择）。

### Figure 2: WEAVER 架构

![Figure 2](https://arxiv.org/html/2606.13672v1/x2.png)

**说明**: 左：多视角世界模型，输入 Memory+History 潜向量和动作块，输出预测未来潜向量序列；中：潜空间验证器，通过奖励头和 Critic 头评估轨迹价值；右：通过预训练 [[VAE]] 解码器重建的视觉预测结果。

### Figure 3: 长程 FID 对比

![Figure 3](https://arxiv.org/html/2606.13672v1/x3.png)

**说明**: 横轴为预测地平线步数（最长 150 步），纵轴为 FID。WEAVER（NFE=16）在整个地平线上的 FID 均低于 Ctrl-World（NFE=50），验证了 WEAVER 的长程一致性优势。

### Figure 4: 奖励预测与优势过滤

![Figure 4](https://arxiv.org/html/2606.13672v1/x4.png)

**说明**: 左：WEAVER 预测奖励与 Robometer 真实奖励随轨迹步骤的对比，曲线高度吻合；右：通过优势过滤区分不同动作样本（高优势蓝色 vs 低优势橙色），验证奖励模型的判别能力。

### Figure 5: FVD vs 推理时间 Pareto 曲线

![Figure 5](https://arxiv.org/html/2606.13672v1/x5.png)

**说明**: 在 DROID 和 OOD 任务数据集上，WEAVER 的 FVD 与推理时间的权衡曲线全面 Pareto 主导 Ctrl-World。同等 FVD 质量下，WEAVER 速度快 3-16×；同等推理预算下，WEAVER FVD 更低。

### Figure 6: 策略评估相关性

![Figure 6](https://arxiv.org/html/2606.13672v1/x6.png)

**说明**: 左：PnP Towel 和 Pour Beans 任务的定性预测对比（预测帧 vs 真实帧）；右：WEAVER-FT 预测成功率与真实成功率的散点相关图，Pearson ρ=0.870，显著优于 Ctrl-World 和 Pretrained WM 基线。

### Figure 7: 策略改进结果

![Figure 7](https://arxiv.org/html/2606.13672v1/x7.png)

**说明**: 左：五类任务上使用不同微调数据策略的成功率柱状图（Base Policy / FT w/ Real / FT w/ Synthetic / FT w/ Mixed）；右：Pour Beans 任务合成数据规模扩展实验，随合成数据量从 1000 增至 5000 段，成功率持续提升并超越纯真实数据基线。

### Figure 8: 策略改进前后对比

![Figure 8](https://arxiv.org/html/2606.13672v1/x8.png)

**说明**: 微调前后实际机器人轨迹的对比展示，证明通过 WEAVER 生成的合成数据微调后，机器人行为质量显著提升。

### Figure 9: 测试时规划结果

![Figure 9](https://arxiv.org/html/2606.13672v1/x9.png)

**说明**: 五个操作任务（Stack Bowls、PnP Bag、PnP Marker、PnP Towel、Pour Beans）上，测试时 Best-of-N（B=4）规划相比 Base Policy 的成功率提升，平均提升 14%，弱基线上最高达 20%。

### Figure 10: 硬件平台与任务配置

![Figure 10](https://arxiv.org/html/2606.13672v1/x11.png)

**说明**: Franka Emika Panda 机器人实验平台，配备 2 个外部 Zed 2i 摄像头和 1 个腕部 Zed Mini 摄像头，共 5 个操作任务的实物配置展示。

---

### Table 1: 世界模型评估（FID / FVD / 推理时间）

| 数据集 | 方法 | NFE | 外部 FID ↓ | 外部 FVD ↓ | 腕部 FID ↓ | 腕部 FVD ↓ | 时间 (s) ↓ |
|--------|------|-----|-----------|-----------|-----------|-----------|-----------|
| DROID | Ctrl-World | 16 | 26.09 | 78.73 | 33.83 | 195.37 | 14.65 |
| DROID | Ctrl-World | 50 | 22.44 | 55.05 | 25.32 | 91.77 | 42.33 |
| DROID | **WEAVER** | 16 | **10.20** | **27.83** | **21.50** | **90.72** | **4.78** |
| DROID | **WEAVER** | 50 | **9.51** | **26.54** | **16.75** | **66.89** | **14.25** |
| OOD Task | Ctrl-World | 16 | 36.16 | 139.54 | 38.76 | 277.13 | 14.65 |
| OOD Task | Ctrl-World | 50 | 31.44 | 91.48 | 33.47 | 145.86 | 42.33 |
| OOD Task | **WEAVER** | 16 | **23.95** | **88.27** | **30.77** | **184.62** | **4.78** |
| OOD Task | **WEAVER** | 50 | **23.48** | **87.03** | **27.37** | **145.04** | **14.25** |

**关键发现**: WEAVER 在等同甚至更少推理步数下全面优于 Ctrl-World，OOD 场景下同样保持优势，最高可实现 16× 推理加速。

### Table 2: 策略评估相关性

| 方法 | Pearson ρ ↑ | MMRV ↓ |
|------|------------|--------|
| Pretrained WM | 低（低估策略性能）| 高 |
| Ctrl-World-FT | 中 | 中 |
| **WEAVER-FT** | **0.870** | **最低** |

**关键发现**: 微调后的 WEAVER 与真实世界成功率的相关性达 ρ=0.870，显著优于所有基线。

### Table 3: 策略改进（平均成功率）

| 微调策略 | 平均成功率 | 相对 Base 提升 |
|----------|-----------|---------------|
| Base Policy (π₀.₅) | 基线 | — |
| FT w/ Real Data（1000段） | 基线 + X% | +X% |
| FT w/ Synthetic Data（1000段） | ≈ Real Data | ≈ Real（差 4%） |
| **FT w/ Mixed Data（2000段）** | **最高** | **+38%** |

**关键发现**: 合成数据质量接近真实数据（仅差 4%），混合数据训练超越单纯真实数据微调 11%，整体成功率提升 38%。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | 大规模 | 多场景桌面操作 | 预训练 |
| $\mathcal{D}^{\text{FT}}_{\text{real}}$ | 每任务 50 条轨迹 | 目标任务真实数据 | 微调 |
| $\mathcal{D}^{\text{val}}_{\text{real}}$ | 每任务 20 条轨迹 | 真实验证数据 | 评估 |
| OOD Task Data | 100 条轨迹 | 分布外操作任务 | OOD 评估 |
| [yilin-wu/droid_ood_data](https://huggingface.co/datasets/yilin-wu/droid_ood_data) | — | HuggingFace 开源 | OOD 测试 |

### 任务列表

| 任务 | 难度特点 |
|------|---------|
| Stack Bowls | 精确放置 |
| PnP Bag | 软体抓取 |
| PnP Marker | 小物体操作 |
| PnP Towel | 可形变物体 |
| Pour Beans | 液态/颗粒物倾倒 |

基座策略 [[pi0.5]] 在每个任务成功率 ≥ 20%。

### 实现细节

- **Backbone**: [[Stable Diffusion 3]] VAE 编码器（$\mathcal{E}_\psi$）作为感知基础
- **动力学模型**: 32 层 [[2D Transformer]]，隐藏维度 1536，16 头，共 928M 参数
- **Transformer 组件**: [[RMSNorm]]、[[RoPE]]（位置编码）、[[QKNorm]]、[[SwiGLU]] 激活
- **优化器**: AdamW，预训练 LR 1e-4，微调 LR 2e-5
- **Batch Size**: 预训练 32
- **预训练**: 1M steps，4×H100 GPU，约 10 天
- **微调**: 16k steps，4×H100，约 6 小时
- **动作表示**: 关节位置增量（Joint Position Deltas）
- **控制频率**: 5Hz（原始数据下采样 3×）
- **图像分辨率**: 190×320
- **基础策略**: [[pi0.5]]（基于 DROID 数据集训练的 [[VLA]]）
- **机器人**: Franka Emika Panda，8-DOF（7 关节 + 夹爪）
- **推理加速**: KV 缓存 + 余弦噪声调度 + [[Rectified Flow Distillation]]

### 可视化结果

定性对比（Figure 6 左）显示 WEAVER 在 PnP Towel 和 Pour Beans 任务上生成的预测帧与真实帧在纹理、物体形状和运动轨迹上高度吻合，尤其是腕部摄像头视角的手-物接触区域预测较为精确。

---

## 批判性思考

### 优点

1. **设计决策系统化**: 明确以三个 desiderata 为出发点，每个架构选择均有对应消融验证，科学严谨。
2. **实用性强**: 三种下游应用全部在真实机器人平台验证，非仿真结果，说服力高。
3. **效率突破**: 相比 Ctrl-World 在等质量下快 16×，使测试时规划从理论上可行变为实际可用。
4. **数据效率**: 仅用 1000 段合成轨迹即可匹敌 1000 段真实数据，大幅降低真实数据采集成本。

### 局限性

1. **部分可观测性**: 仅使用视觉观测，缺乏触觉传感；可形变物体（如 PnP Towel）仍面临感知盲区。
2. **规划地平线受限**: 当前推理延迟限制了测试时规划仅能覆盖单个动作块（h=12 步），无法进行多步预见性规划。
3. **奖励标注噪声**: 依赖 Robometer 提供奖励监督，该系统本身存在噪声，可能影响奖励模型精度。
4. **可形变物体物理性**: 对可变形物体（布料、液体）缺乏物理先验，预测一致性较硬物体弱。

### 潜在改进方向

1. 引入触觉传感器数据作为额外模态，缓解接触区域的部分可观测问题。
2. 结合物理引擎先验（如 FEM 形变模型）改善可形变物体预测。
3. 延长想象地平线以支持多步测试时树搜索规划。

### 可复现性评估

- [x] 代码开源（GitHub: arnavkj1995/WEAVER）
- [x] 预训练模型（HuggingFace: arnavkj1995/WEAVER）
- [x] 训练细节完整（附录提供超参数）
- [x] 数据集可获取（DROID 公开，OOD 数据集已上传 HuggingFace）

---

## 关联笔记

### 基于

- [[pi0.5]]: 基础策略（VLA），WEAVER 在其上进行策略改进和规划
- [[Flow Matching]]: WEAVER 的核心训练目标
- [[Diffusion Forcing]]: 独立噪声采样策略，提升长程一致性
- [[Stable Diffusion 3]]: 提供预训练 VAE 编码器/解码器

### 对比

- [[Ctrl-World]]: 主要对比基线，WEAVER 在保真度和效率上全面超越
- [[DreamerV4]]: 从零训练编码器的 WM 代表，OOD 鲁棒性弱

### 方法相关

- [[Flow Matching]]: 核心训练方法
- [[Rectified Flow Distillation]]: 推理加速关键技术
- [[Monte Carlo Advantage]]: 策略改进的优势估计
- [[TD Lambda]]: Critic 网络训练的 λ-return
- [[KV Cache]]: Transformer 推理加速

### 硬件/数据相关

- [[DROID]]: 预训练数据集
- [[Franka Emika Panda]]: 实验机器人平台

---

## 速查卡片

> [!summary] WEAVER: World Estimation Across Views for Embodied Reasoning
> - **核心**: 同时满足保真度、长程一致性、效率的机器人操作世界模型
> - **方法**: 多视角潜空间预测 + Flow Matching + Diffusion Forcing + KV 缓存加速
> - **结果**: 策略评估 ρ=0.870，策略改进 +38%，测试时规划 +14%，速度快 Ctrl-World 最高 16×
> - **代码**: [github.com/arnavkj1995/WEAVER](https://github.com/arnavkj1995/WEAVER)

---

*笔记创建时间: 2026-06-12*
