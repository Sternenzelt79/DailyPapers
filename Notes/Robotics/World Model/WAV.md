---
title: "World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry"
method_name: "WAV"
authors: [Yuejiang Liu, Fan Feng, Lingjing Kong, Weifeng Lu, Jinzhou Tang, Kun Zhang, Kevin Murphy, Chelsea Finn, Yilun Du]
year: 2026
venue: arXiv
tags: [world-model, inverse-dynamics, active-learning, sample-efficiency, out-of-distribution]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2604.01985
created: 2026-06-01
---

# 论文笔记：World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多机构合作 |
| 日期 | April 2026 |
| 项目主页 | [world-action-verifier.github.io](https://world-action-verifier.github.io) |
| 对比基线 | [[Dreamer-v3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.01985) / [Code-MiniGrid](https://github.com/world-action-verifier/wav_minigrid) / [Code-Robot](https://github.com/world-action-verifier/wav_robot) |

---

## 一句话总结

> WAV 通过将世界模型预测验证分解为"状态可信性"与"动作可达性"两个互补因素，利用正逆动力学不对称性实现自我改进，在九个基准任务上获得 2× 样本效率提升和 22%+ 的策略性能提升。

---

## 核心贡献

1. **前向-逆向不对称性框架**: 提出利用[[逆向动力学模型|逆向动力学]]验证世界模型预测误差比直接用前向模型更容易且更准确的理论和实践框架
2. **两因素分解验证**: 将验证问题分解为"状态可信性（state plausibility）"和"动作可达性（action reachability）"两个互补因素，分别用视频数据训练的子目标生成器和稀疏逆模型处理
3. **有理论支撑的目标导向探索**: 给出 Proposition 3.1 精确刻画三个决定逆向验证优越性的因素（维度、随机性、样本量），并将其用于自适应探索策略

---

## 问题背景

### 要解决的问题

[[世界模型]] 在训练时依赖有标签的动作轨迹，但在分布外（OOD）场景预测质量下降。关键挑战在于：如何在**没有真实后继状态**的情况下，让世界模型识别并修正自己的预测错误？

### 现有方法的局限

- **基于不确定性的探索**：只能估计模型置信度，无法判断预测是否错误
- **基于学习进度的探索**：难以区分哪些区域的误差来自模型自身缺陷 vs 环境随机性
- **前向预测验证**：在高维状态空间中，前向误差估计受环境噪声和维度诅咒双重影响

### 本文的动机

逆向动力学（给定前后状态推断动作）具有三大天然优势：
1. 动作相关特征维度 $d_z \ll$ 状态维度 $d_s$（维度优势）
2. 环境噪声影响前向预测但不影响逆向推断（随机性优势）
3. 动作标签数据稀缺时，逆向模型样本效率更高（数据优势）

---

## 方法详解

### 模型架构

WAV 采用 **半监督验证 + 目标导向探索** 架构：

- **输入**: 当前状态 $s^t$，动作 $a^t$，预测后继状态 $\hat{s}^{t+1}$
- **数据源**:
  - 动作标签数据 $\mathcal{D}_{act} = \{(s^t, a^t, s^{t+1})\}$（机器人交互）
  - 无动作视频数据 $\mathcal{D}_{vid} = \{(s^t, s^{t+1}, \ldots)\}$（视频语料库）
- **核心模块**: [[子目标生成器]] $g_\phi$ + [[稀疏逆动力学模型]] $h_\psi$
- **输出**: 验证分数 $\hat{\varepsilon}(s^t, a^t, \hat{s}^{t+1})$（估计预测误差）
- **下游应用**: 引导[[主动学习|目标导向探索]]，收集高信息量轨迹

### 核心模块

#### 模块1: 半监督验证器构建

**设计动机**: 构建无需观测真实后继状态的验证函数，利用[[贝叶斯分解]]将联合分布解耦

**具体实现**:
- 验证目标：$\hat{\varepsilon}(s^t, a^t, \hat{s}^{t+1})$ 满足[[排序保持性质]]，即正确排序预测质量
- 将验证分解为两个独立因素：
  - **状态可信性**：$\hat{s}^{t+1}$ 是否是 $s^t$ 的可能后继状态（由 $g_\phi$ 评估）
  - **动作可达性**：从 $s^t$ 通过某个动作能否到达 $\hat{s}^{t+1}$（由 $h_\psi$ 评估）

#### 模块2: 子目标生成器 $g_\phi$（状态可信性）

**设计动机**: 利用大量无动作标签视频数据训练[[状态转移先验]]

**具体实现**:
- 从视频数据 $\mathcal{D}_{vid}$ 训练：学习 $p(s^{t+1} | s^t)$ 的分布
- 采样 $K$ 个候选子目标：$\{\tilde{s}_k^{t+1}\}_{k=1}^K \sim g_\phi(\cdot | s^t)$
- 提供多样化的可行后继状态分布，无需动作标签即可训练

#### 模块3: 稀疏逆动力学模型 $h_\psi$（动作可达性）

**设计动机**: 通过[[稀疏性约束]]强制模型只关注动作相关特征，降低问题维度

**具体实现**:
- 引入可学习的稀疏掩码 $M \in [0,1]^{d_s}$
- 逆向推断：$\hat{a}^t = h_\psi(M \odot s^t, M \odot s^{t+1})$
- $M$ 自动筛选出维度 $d_z \ll d_s$ 的动作相关子空间
- 骨干网络使用 [[CLAM]]（基于潜在动作空间的稀疏性）

#### 模块4: 目标导向探索循环

**设计动机**: 通过前向-逆向循环一致性检测世界模型的预测错误区域

**具体实现**: 循环流程如下：

$$
s^t \xrightarrow{g_\phi} \tilde{s}_{1:K}^{t+1} \xrightarrow{h_\psi} \hat{a}_{1:K}^t \xrightarrow{f_\theta} \hat{s}_{1:K}^{t+1} \xrightarrow{\ell} \hat{\varepsilon}_{1:K} \xrightarrow{\max} a^\star
$$

选择使前向预测与子目标分歧最大的动作执行，从高误差区域收集新数据。

---

## 关键公式

### 公式1: [[预测误差|真实预测误差]]

$$
\varepsilon(s^t, a^t; s^{t+1}) := \ell(f_\theta(s^t, a^t), s^{t+1})
$$

**含义**: 世界模型 $f_\theta$ 的预测 $f_\theta(s^t, a^t)$ 与真实后继状态 $s^{t+1}$ 之间的损失

**符号说明**:
- $f_\theta$: 前向世界模型，参数为 $\theta$
- $\ell(\cdot, \cdot)$: 损失函数（如 MSE）
- $s^t, a^t$: 当前状态与动作
- $s^{t+1}$: 真实后继状态（验证时不可观测）

### 公式2: [[排序保持性质|验证器排序保持条件]]

$$
\varepsilon(s^t, a_i^t; s_i^{t+1}) < \varepsilon(s^t, a_j^t; s_j^{t+1}) \implies \hat{\varepsilon}(s^t, a_i^t, \hat{s}_i^{t+1}) < \hat{\varepsilon}(s^t, a_j^t, \hat{s}_j^{t+1})
$$

**含义**: 无需真实后继状态的估计验证器 $\hat{\varepsilon}$ 应保持对预测质量的正确排序

**符号说明**:
- $\hat{\varepsilon}$: 估计的验证分数，无需观测真实 $s^{t+1}$
- 下标 $i, j$: 两条不同轨迹

### 公式3: [[贝叶斯分解|状态转移贝叶斯分解]]

$$
p(s^{t+1} | s^t, a^t) = \frac{p(a^t | s^t, s^{t+1}) \cdot p(s^{t+1} | s^t)}{p(a^t | s^t)} \propto p(s^{t+1} | s^t) \cdot p(a^t | s^t, s^{t+1})
$$

**含义**: 将动作条件下的状态转移分解为"状态先验"与"动作可能性"乘积，使两个因素可以分别建模

**符号说明**:
- $p(s^{t+1}|s^t)$: 状态转移先验，由子目标生成器 $g_\phi$ 建模
- $p(a^t|s^t, s^{t+1})$: 动作可能性，由逆动力学模型 $h_\psi$ 建模
- $p(a^t|s^t)$: 归一化常数（边际动作概率）

### 公式4: [[子目标采样]]

$$
\{\tilde{s}_k^{t+1}\}_{k=1}^K \sim g_\phi(\cdot | s^t)
$$

**含义**: 从视频预训练的状态先验中采样 $K$ 个多样化的候选后继状态

**符号说明**:
- $g_\phi$: 子目标生成器（由视频数据 $\mathcal{D}_{vid}$ 训练）
- $K$: 候选数量（超参数）
- $\tilde{s}_k^{t+1}$: 第 $k$ 个候选子目标

### 公式5: [[稀疏逆动力学模型|稀疏逆向动力学]]

$$
\hat{a}^t = h_\psi(M \odot s^t, M \odot s^{t+1})
$$

**含义**: 通过可学习稀疏掩码 $M$ 提取动作相关特征子空间，再用逆模型推断动作

**符号说明**:
- $h_\psi$: 逆动力学模型，参数为 $\psi$
- $M \in [0,1]^{d_s}$: 可学习稀疏掩码（element-wise）
- $\odot$: Hadamard 积（逐元素乘法）
- $d_z = \|M\|_0$: 有效动作相关特征维度，$d_z \ll d_s$

### 公式6: [[线性高斯动力学|理论分析假设]]

$$
s' = As + Ba + \xi, \quad \xi \sim \mathcal{N}(0, \sigma_s^2 I_{d_s})
$$

$$
a = h(z, z') + \eta, \quad h(z, z') := H[z; z'], \quad \eta \sim \mathcal{N}(0, \sigma_a^2 I_{d_a})
$$

**含义**: 线性高斯设定下，状态转移受环境噪声 $\sigma_s$ 影响，而动作从低维特征 $z$ 可恢复，只受动作恢复歧义 $\sigma_a$ 影响

**符号说明**:
- $A, B$: 状态和动作转移矩阵
- $\sigma_s$: 环境噪声标准差（影响前向预测）
- $z = Cs$: 动作相关低维特征，$d_z \ll d_s$
- $\sigma_a$: 动作恢复歧义标准差（影响逆向模型）
- $H$: 逆向动力学线性映射

### 公式7: [[前向逆向误差比|Proposition 3.1 误差边界]]

$$
\frac{\mathbb{E}[\mathcal{E}_F]}{\mathbb{E}[\mathcal{E}_I]} \geq \underbrace{\frac{d_s + d_a}{2d_z} \cdot \frac{d_s}{d_a}}_{\text{维度因素}} \cdot \underbrace{\left(\frac{\sigma_s}{\lambda \sigma_a}\right)^2}_{\text{随机性因素}} \cdot \underbrace{\frac{n - 2d_z - 1}{n - (d_s + d_a) - 1}}_{\text{样本量因素}}
$$

**含义**: 在线性高斯设定下（OLS 训练，$n > d_s + d_a + 1$），前向模型误差对逆向误差之比由三个可解释因素的乘积下界确定

**符号说明**:
- $\mathcal{E}_F = \frac{1}{d_s}\mathbb{E}[\|f_\theta(s,a) - f(s,a)\|_2^2]$: 前向模型归一化误差
- $\mathcal{E}_I = \frac{1}{d_s}\mathbb{E}[\|f(s, h_\psi(z,z')) - f(s, h(z,z'))\|_2^2]$: 逆向验证归一化误差
- $d_s, d_a, d_z$: 状态、动作、动作相关特征维度
- $\lambda$: 动作到状态空间的放大因子
- $n$: 动作标签样本数

---

## 关键图表

### Figure 1: WAV 系统概览

![Figure 1 - WAV Overview](https://arxiv.org/html/2604.01985v2/x1.png)

**说明**: WAV 整体流程。前向世界模型 $f_\theta$ 的预测由子目标生成器 $g_\phi$（状态可信性）和稀疏逆模型 $h_\psi$（动作可达性）联合验证，形成目标导向探索循环。

### Figure 2: 两因素分解示意

![Figure 2 - State Plausibility vs Action Reachability](https://arxiv.org/html/2604.01985v2/x2.png)

**说明**: 展示[[状态可信性]]与[[动作可达性]]的分解。左侧：视频数据训练的子目标生成器评估状态转移是否物理可行；右侧：稀疏逆模型评估预测后继状态是否可通过某个动作到达。

### Figure 3: MiniGrid 鲁棒性验证（三个理论因素）

![Figure 3a - Sample Efficiency](https://arxiv.org/html/2604.01985v2/x3.png)

![Figure 3b - State Complexity](https://arxiv.org/html/2604.01985v2/x4.png)

![Figure 3c - Stochasticity](https://arxiv.org/html/2604.01985v2/x5.png)

![Figure 3d - Combined](https://arxiv.org/html/2604.01985v2/x6.png)

**说明**: 分别验证 Proposition 3.1 的三个理论因素：
- **样本量因素**：数据预算 $\{400, 800, 1200, 1600, 2000\}$ 下，稀疏逆模型（WAV）持续优于前向模型
- **维度因素**：对象数量从 6 增加到 14 时，WAV 优势随状态复杂度上升而扩大
- **随机性因素**：噪声地板数量 0→4 增加时，WAV 鲁棒性显著优于基线

### Figure 4: MiniGrid 世界模型学习效果

![Figure 4a](https://arxiv.org/html/2604.01985v2/x7.png)

![Figure 4b](https://arxiv.org/html/2604.01985v2/x8.png)

![Figure 4c](https://arxiv.org/html/2604.01985v2/x9.png)

![Figure 4d](https://arxiv.org/html/2604.01985v2/x10.png)

![Figure 4e](https://arxiv.org/html/2604.01985v2/x11.png)

**说明**: 在 200 初始转移 + 3 轮 × 100 转移的主动学习设置下，WAV 达到接近 Oracle（使用真实预测误差）的性能，并在 action following score 上表现最佳，说明学到了更准确的动作依赖动力学。

### Figure 5: 机器人操作世界模型学习

![Figure 5 - RoboMimic ManiSkill World Model Learning](https://arxiv.org/html/2604.01985v2/x12.png)

**说明**: 在 RoboMimic（Lift, Can, Square）和 ManiSkill（PullCube, PokeCube, LiftPeg）六个任务上，以 32 帧预测 MSE 为指标，在不同 warm-up 数据规模 $\{200, 400, 600, 800, 1000\}$ 下评估。WAV 在所有任务和数据规模上均优于基线，尤其在低数据量场景优势明显。

### Figure 6: OOD 适应性实验

![Figure 6 - OOD Adaptation](https://arxiv.org/html/2604.01985v2/x13.png)

**说明**: 在 RoboMimic Can 的两种 OOD 变体上：
- **视觉偏移**：背景颜色和机器人颜色变化
- **对象/交互偏移**：多对象场景 + 混合优化质量的示范数据

WAV 使用 200 目标域轨迹适应后，在策略下游性能上超越基线 22%+。

---

## Algorithm 1: WAV 引导探索

```
# 输入：世界模型 f_θ，逆模型 h_ψ，子目标生成器 g_φ
# 当前状态 s，数据集 D，候选数 K

for each exploration iteration:
    s_g = g_φ.sample(s, K)         # 采样 K 个候选子目标
    a = h_ψ.inverse(s, s_g)        # 逆向推断候选动作
    s_p = f_θ.predict(s, a)        # 前向滚动预测后继状态
    scores = dist(s_g, s_p)        # 计算子目标与预测的分歧
    idx = argmax(scores)            # 选最大分歧的候选动作
    s_n = env.step(a[idx])          # 在环境中执行
    D.append((s, a[idx], s_n))      # 收集新转移
    f_θ.update(D), h_ψ.update(D)   # 更新前向和逆向模型
```

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MiniGrid (自定义) | 50k 序列，3 个任务 | 离散格子世界，可控维度/噪声 | 理论验证 |
| RoboMimic | {200-1000} 轨迹 warm-up | 专家示范，6-DOF 操作 | 机器人 WM 学习 |
| ManiSkill | {200-1000} 轨迹 warm-up | 多样操作任务 | 机器人 WM 学习 |
| RoboMimic OOD | 200 目标域轨迹 | 视觉偏移 / 对象偏移变体 | OOD 适应 |

### 实现细节

- **世界模型**: [[Dreamer-v3]]（潜在循环状态空间模型 RSSM）
- **稀疏逆模型骨干**: [[CLAM]]（基于潜在动作空间稀疏性）
- **MiniGrid 种子集**: 200 条带标签序列 + 20k 无标签候选池
- **机器人 Warm-up**: {200, 400, 600, 800, 1000} 轨迹（约 200 样本/轨迹）
- **探索轮数**: 2-3 轮，每轮收集 100-200 个新转移
- **评估指标**: 32 帧预测 MSE（3 个随机种子均值）
- **两阶段数据整理**:
  1. Warm-up：专家示范 + 最优策略的在线轨迹
  2. 探索：不完美策略检查点的轨迹

### 基线方法

| 基线 | 描述 |
|------|------|
| Random | 均匀随机采样（下界） |
| Uncertainty | 基于预测不确定性的探索 |
| Progress | 基于学习进度的探索 |
| Vanilla IDM | 无稀疏掩码的标准逆动力学模型 |
| Oracle | 使用真实预测损失（上界） |

### 主要结果

- **2× 样本效率提升**：在 MiniGrid 和机器人任务上的探索效率
- **22%+ 策略性能提升**：OOD 对象/交互场景下的下游策略
- **稀疏 vs 稠密 IDM**：稀疏掩码版本在所有条件下均优于普通 IDM，验证维度因素
- **OOD 鲁棒性**：视觉偏移和对象偏移下均优于 Uncertainty 和 Progress 基线

---

## 批判性思考

### 优点

1. **理论严谨**: Proposition 3.1 给出精确的误差比下界，三个因素可量化且实验分别验证
2. **数据效率**: 充分利用无标签视频数据训练子目标生成器，缓解动作标签数据稀缺问题
3. **实用性强**: 框架不依赖特定世界模型架构，可与 Dreamer-v3 等现有模型结合

### 局限性

1. **视频数据依赖**: 子目标生成器需要大量高质量无动作视频语料库，在某些领域（如水下机器人）难以获取
2. **逆向歧义问题**: 多个动作可能导致相同状态转移（逆向动力学本质上是欠定问题），稀疏掩码只能部分缓解
3. **计算开销**: 每步需要采样 $K$ 个候选子目标并分别做前向滚动，推断时间随 $K$ 线性增长

### 潜在改进方向

1. 将子目标生成器替换为扩散模型，提高子目标多样性和质量
2. 将探索策略与强化学习策略联合优化，而非分开训练
3. 在 3D 视觉表示空间中应用，扩展到更复杂的真实机器人场景

### 可复现性评估

- [x] 代码开源（MiniGrid + Robot 两个独立仓库）
- [ ] 预训练模型（项目主页有视频演示但未明确提供模型权重）
- [x] 训练细节完整（Appendix E 含超参数表）
- [x] 数据集可获取（RoboMimic、ManiSkill 均公开）

---

## 关联笔记

### 基于

- [[Dreamer-v3]]: 世界模型骨干（潜在 RSSM）
- [[CLAM]]: 稀疏逆动力学模型的骨干网络
- [[主动学习]]: 目标导向探索的理论基础

### 对比

- [[FeedbackWM]]: 同类世界模型自改进工作，WAV 专注前向-逆向不对称性
- [[StableWorldModel]]: 稳定世界模型训练，WAV 关注分布外适应

### 方法相关

- [[逆向动力学模型]]: WAV 核心验证机制
- [[稀疏性约束]]: 特征掩码 $M$ 的实现基础
- [[贝叶斯分解]]: 两因素分解的理论依据
- [[世界模型]]: 本文改进的核心对象

### 硬件/数据相关

- [[RoboMimic]]: 操作任务评估数据集
- [[ManiSkill]]: 多任务机器人仿真基准
- [[MiniGrid]]: 离散格子环境，用于可控实验验证

---

## 速查卡片

> [!summary] World Action Verifier (WAV)
> - **核心**: 前向-逆向不对称性 → 逆向动力学验证比前向预测更稳定
> - **方法**: 子目标生成器（状态可信性）+ 稀疏逆模型（动作可达性）→ 循环一致性验证
> - **结果**: 2× 样本效率，22%+ OOD 策略提升，9 个基准任务
> - **代码**: [MiniGrid](https://github.com/world-action-verifier/wav_minigrid) / [Robot](https://github.com/world-action-verifier/wav_robot)

---

*笔记创建时间: 2026-06-01*
