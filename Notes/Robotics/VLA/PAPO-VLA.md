---
title: "PAPO-VLA: Planning-Aware Policy Optimization for Vision-Language-Action Models"
method_name: "PAPO-VLA"
authors: [Peizheng Guo, Jingyao Wang, Changwen Zheng, Wenwen Qiang]
year: 2026
venue: arXiv
tags: [vla, policy-optimization, causal-reasoning, reinforcement-learning, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.19580
created: 2026-05-20
---

# 论文笔记：PAPO-VLA: Planning-Aware Policy Optimization for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Institute of Software, Chinese Academy of Sciences; University of Chinese Academy of Sciences |
| 日期 | May 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[OpenVLA]]、[[TGRPO]]、[[Nora]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19580) / Code: 暂无 |

---

## 一句话总结

> PAPO-VLA 将 [[Vision-Language-Action Model|VLA]] 策略分解为规划器与执行器双重角色，通过识别关键规划动作并利用因果充分性与必要性衡量其重要度，强化 [[GRPO]] 优化对关键时刻的权重分配，从而大幅提升机器人操作任务的策略可靠性。

---

## 核心贡献

1. **规划器/执行器双重角色分析**: 重新审视 VLA 执行过程，区分改变执行方向的"规划动作"（如从接近切换到抓取）与实现平滑连接的"执行动作"，指出规划动作的可靠性是任务成功的关键瓶颈。
2. **规划动作识别与因果重要度建模**: 结合动作变化幅度（action variation）和轨迹结果门控（outcome-aware gate）识别规划动作，再通过[[因果充分性]]和[[因果必要性]]的调和平均计算其对任务成功的因果重要度。
3. **规划感知策略优化（PAPO）**: 将因果重要度融入 [[GRPO]] 的优势估计，使优化过程自动向关键规划时刻集中梯度，在 LIBERO 和 RoboTwin2.0 多个 benchmark 上取得显著提升。

---

## 问题背景

### 要解决的问题

[[Vision-Language-Action Model|VLA]] 模型在机器人操作任务中的**策略可靠性**问题：任务通过闭环交互完成，每个动作影响后续执行，因此不仅需要局部动作合理，还要维持整体执行走向任务成功。

### 现有方法的局限

- [[OpenVLA]]、[[RIPT-VLA]]、[[TGRPO]] 等方法在 [[GRPO]] 中对轨迹内所有时间步采用**均匀的优化权重**，未区分对任务成功至关重要的关键动作。
- 现有方法忽视了 VLA 策略的**双重角色**：既是规划器（做出关键决策），也是执行器（产生连续动作）。

### 本文的动机

关键洞察：操作轨迹中存在少数**规划动作**（planning actions），它们标志着执行方向的改变（如从接近转换为抓取）；即使执行动作平滑，不稳定的规划动作（如不稳定的抓取）也会导致后续执行失效。因此应该识别这些关键动作并在优化时给予更大权重。

---

## 方法详解

### 模型架构

PAPO-VLA 建立在 [[OpenVLA-OFT]] 基础骨干上，核心在于为 [[GRPO]] 训练引入规划感知的优势估计：

- **输入**: 语言指令 $l$ + 观测窗口 $o_{t-H+1:t}$
- **Backbone**: OpenVLA-OFT（[[Vision-Language-Action Model]] 骨干）
- **核心模块**: [[因果充分性]] + [[因果必要性]] 联合建模规划动作重要度
- **输出**: 动作 $a_t$（连续控制信号）
- **优化框架**: 规划感知 [[GRPO]]（PAPO）

整体流程：轨迹 rollout → 轨迹级奖励评估 → 规划动作识别 → 因果重要度计算 → 融入 GRPO 优势估计 → 策略更新。

### 核心模块

#### 模块1: 规划动作识别（Planning Action Identification）

**设计动机**: 利用[[动作变化幅度]]和[[轨迹结果]]共同识别改变执行方向的关键时刻。

**具体实现**:

1. 计算动作变化幅度（action variation magnitude）：

$$
u_t = \begin{cases} \frac{1}{d}\|a_0\|_1, & t=0 \\ \frac{1}{d}\|a_t - a_{t-1}\|_1, & t>0 \end{cases}
$$

其中 $d$ 为动作维度，$\|\cdot\|_1$ 为 L1 范数。

2. 归一化变化量：$\tilde{u}_t = \frac{u_t}{\frac{1}{T}\sum_{j=0}^{T-1}u_j}$

3. 结果感知门控（outcome-aware gate），将轨迹奖励归一化为 $[0,1]$ 权重：

$$
g(\tau) = \frac{r(\tau) - r_{min}}{r_{max} - r_{min}} \in [0,1]
$$

4. 规划动作得分：$s_t = \tilde{u}_t \cdot g(\tau)$

5. 取得分前 $k$ 的时间步作为规划动作集合：$\mathcal{K}_\tau = \{t \mid m_t^{plan} = 1\}$

**关键设计**: 门控 $g(\tau)$ 确保只有成功轨迹中的高变化动作被标记为规划动作，避免失败轨迹中的无效动作干扰识别。

#### 模块2: 因果充分性重要度（Causal Sufficiency Importance）

**设计动机**: 衡量保留动作 $a_t$ 是否能支撑后续执行成功（即 $a_t$ 对成功是否"足够"）。

$$
C_t^{suff} = m_t^{plan} \cdot \left[\mathbb{E}_{\tau \sim \pi(\cdot|\mathcal{A}_{\leq t})} r(\tau) - \mathbb{E}_{\tau \sim \pi(\cdot|\bar{\mathcal{A}}_{\leq t})} r(\tau)\right]_+
$$

其中 $[\cdot]_+ = \max(\cdot, 0)$，$\bar{\mathcal{A}}_{\leq t}$ 表示对 $t$ 时刻及之前序列进行扰动后的替代序列。

#### 模块3: 因果必要性重要度（Causal Necessity Importance）

**设计动机**: 衡量在原序列成功的条件下，扰动 $a_t$ 是否会导致任务失败（即 $a_t$ 对成功是否"必要"）。

$$
C_t^{nec} = m_t^{plan} \cdot \left[\mathbb{E}_{\tau \sim \pi(\cdot|\mathcal{A})} r(\tau) - \mathbb{E}_{\tau \sim \pi(\cdot|\bar{\mathcal{A}}_t)} r(\tau)\right]_+
$$

其中 $\bar{\mathcal{A}}_t$ 表示仅对 $t$ 时刻动作进行扰动的替代序列。

#### 模块4: 整体规划动作重要度与 PAPO 优化

综合充分性与必要性，取调和平均确保高重要度需要两者同时高：

$$
C_t^{plan} = \frac{2C_t^{suff}C_t^{nec}}{C_t^{suff} + C_t^{nec}}
$$

---

## 关键公式

### 公式1: [[Vision-Language-Action Model|VLA 策略]]

$$
a_t \sim \pi_\theta(a_t \mid o_{t-H+1:t}, l)
$$

**含义**: VLA 策略基于过去 H 步观测窗口和语言指令生成当前动作。

**符号说明**:
- $o_{t-H+1:t}$: 最近 H 步观测序列
- $l$: 语言指令
- $\theta$: 策略参数
- $H$: 历史窗口长度

### 公式2: [[GRPO]] 目标函数

$$
J_{GRPO}(\theta) = \mathbb{E}_{x \sim P,\, \tau^i \sim \pi_{\theta_{old}}} \left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=0}^{T_i-1}\left(\min\!\left(\rho_{i,t}(\theta)A_i,\; \mathrm{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)A_i\right) - \beta D_{KL}(\pi_\theta\|\pi_{ref})\right)\right]
$$

**含义**: GRPO 对 G 条轨迹中所有时间步均匀加权，通过裁剪比率防止策略更新过大，同时 KL 正则化约束偏离参考策略。

**符号说明**:
- $\rho_{i,t}(\theta) = \frac{\pi_\theta(a_t^i | o_{t-H+1:t}^i, l)}{\pi_{\theta_{old}}(a_t^i | o_{t-H+1:t}^i, l)}$: 重要性比率
- $A_i = \frac{r^i - \mathrm{mean}(r^1,\ldots,r^G)}{\mathrm{std}(r^1,\ldots,r^G)}$: 相对优势估计
- $G$: 每次采样的轨迹数
- $\epsilon$: 裁剪阈值
- $\beta$: KL 正则化系数

### 公式3: [[因果重要度|概率因果重要度]] 定义

$$
C(a_t) = P_{suff}\, P(E(\bar{\mathcal{A}}_t, X) = 0,\, \bar{\mathcal{A}}_t, X) + P_{nec}\, P(E(\mathcal{A}, X) = 1,\, \mathcal{A}, X)
$$

**含义**: 结合因果充分性概率 $P_{suff}$ 和因果必要性概率 $P_{nec}$，联合衡量动作 $a_t$ 对任务成功的因果贡献。

**符号说明**:
- $P_{suff} = P(E_{do(a_t)}(\bar{\mathcal{A}}_t, X) = 1 \mid E(\bar{\mathcal{A}}_t, X) = 0, \bar{\mathcal{A}}_t, X)$: 因果充分性（在无该动作会失败的条件下，执行该动作能成功的概率）
- $P_{nec} = P(E_{do(\bar{a}_t)}(\mathcal{A}, X) = 0 \mid E(\mathcal{A}, X) = 1, \mathcal{A}, X)$: 因果必要性（在原序列成功的条件下，替换该动作会失败的概率）
- $E(\cdot)$: 任务执行结果函数（0 失败，1 成功）

### 公式4: 动作变化幅度

$$
u_t = \begin{cases} \frac{1}{d}\|a_0\|_1, & t=0 \\ \frac{1}{d}\|a_t - a_{t-1}\|_1, & t>0 \end{cases}
$$

**含义**: 通过相邻动作的 L1 距离衡量每个时间步的动作变化幅度，初始时刻取动作绝对值。

**符号说明**:
- $d$: 动作维度
- $\|\cdot\|_1$: L1 范数

### 公式5: 结果感知门控

$$
g(\tau) = \frac{r(\tau) - r_{min}}{r_{max} - r_{min}} \in [0,1]
$$

**含义**: 将轨迹奖励归一化为 $[0,1]$ 门控权重，确保只有高质量轨迹中的动作变化才被标记为规划动作。

**符号说明**:
- $r(\tau)$: 轨迹 $\tau$ 的奖励
- $r_{min}, r_{max}$: 批次内最小/最大奖励

### 公式6: 整体规划动作重要度（调和平均）

$$
C_t^{plan} = \frac{2C_t^{suff}C_t^{nec}}{C_t^{suff} + C_t^{nec}}
$$

**含义**: 充分性和必要性的调和平均，确保高重要度需要两者同时高，任一为零则整体重要度为零。

### 公式7: [[PAPO]] 规划感知优势估计

$$
\widehat{A}_{i,t} = A_{i,t} + \eta \cdot C_{i,t}^{plan}
$$

**含义**: 在原始 GRPO 优势基础上叠加规划动作重要度，使关键规划时刻获得更强的梯度信号。

**符号说明**:
- $A_{i,t}$: 原始 GRPO 相对优势
- $\eta$: 规划重要度权重系数（最优值 0.15）
- $C_{i,t}^{plan}$: 规划动作因果重要度

### 公式8: [[PAPO]] 完整优化目标

$$
J_{PAPO}(\theta) = \mathbb{E}_{x \sim P,\, \tau^i \sim \pi_{\theta_{old}}} \left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=0}^{T_i-1}\left(\min\!\left(\rho_{i,t}(\theta)\widehat{A}_{i,t},\; \mathrm{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\widehat{A}_{i,t}\right) - \beta D_{KL}(\pi_\theta\|\pi_{ref})\right)\right]
$$

**含义**: 将规划感知优势 $\widehat{A}_{i,t}$ 替换原始 GRPO 中的均匀优势 $A_i$，其余结构完全一致，使策略优化更关注关键规划时刻。

---

## 关键图表

### Figure 1: Planner 与 Executor 示意图

![Figure 1](https://arxiv.org/html/2605.19580v1/x1.png)

**说明**: 展示 VLA 策略的双重角色。(a) **Planner** 由规划动作组成，这些动作与任务成功更紧密相关（如从接近阶段切换到抓取阶段的关键帧）；(b) **Executor** 通过连续运动连接并实现这些规划动作。图中高亮了轨迹中的规划动作时刻，对比了规划动作稳定 vs 不稳定时的执行差异。

### Figure 2: PAPO-VLA 方法总览

![Figure 2](https://arxiv.org/html/2605.19580v1/x2.png)

**说明**: (a) **方法概览**：轨迹 rollout 首先被轨迹级奖励评估，规划动作被选出并通过[[因果充分性]]和[[因果必要性]]分析获得重要度，融入 [[GRPO]] 优势估计以引导优化；(b) **直觉示意**：操作轨迹中不同动作如何获得不同重要度权重，从而优化 [[Vision-Language-Action Model|VLA]] 策略。[[PAPO]] 的核心创新在于将均匀的时间步权重替换为因果感知的差异化权重。

### Figure 3: 与 OpenVLA 对比可视化

![Figure 3](https://arxiv.org/html/2605.19580v1/x3.png)

**说明**: 两个代表性任务的轨迹对比：(1) "将两个摩卡壶放到炉子上"；(2) "将黄色/白色马克杯放入微波炉并关门"。PAPO-VLA 产生更完整的轨迹，在多个阶段维持任务进展；而 [[OpenVLA]] 在规划动作处出现失败（如不稳定抓取导致后续执行失效）。

### Table 1: LIBERO Benchmark 结果

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| PackNet | 0.63 | 0.60 | 0.75 | 0.25 | 0.56 |
| MTL | 0.83 | 0.54 | 0.80 | 0.48 | 0.66 |
| Octo | 0.78 | 0.85 | 0.84 | 0.51 | 0.75 |
| [[OpenVLA]] | 0.85 | 0.88 | 0.79 | 0.53 | 0.76 |
| TGRPO | 0.90 | 0.92 | 0.81 | 0.59 | 0.81 |
| Nora | 0.92 | 0.95 | 0.89 | 0.74 | 0.87 |
| **PAPO-VLA (Ours)** | **0.93** | **0.98** | **0.98** | **0.94** | **0.96** |

**关键发现**: PAPO-VLA 在所有四个子集均达到最优，Long horizon 子集从 Nora 的 0.74 提升到 0.94（+27%），平均成功率 0.96 远超第二名 Nora 的 0.87。Long 子集提升最显著，说明关键规划动作在长时域任务中尤为重要。

### Table 2: RoboTwin2.0 Benchmark 结果（按时域长度分组）

**短时域（100–130 步）:**

| 方法 | Lift Pot | Beat Hammer | Pick Bottles | Place Phone | Avg. |
|------|----------|-------------|--------------|-------------|------|
| π₀ | 51.0 | 59.0 | 50.0 | 22.0 | 45.5 |
| RDT | 45.0 | 22.0 | 18.0 | 13.0 | 24.5 |
| [[OpenVLA-OFT]] | 10.1 | 28.1 | 29.7 | 17.1 | 21.3 |
| **PAPO-VLA** | **62.7** | **78.5** | **60.2** | **33.1** | **58.6** |

**中时域（150–230 步）:**

| 方法 | Move Can | Place A2B | Empty Cup | Handover Mic | Avg. |
|------|----------|-----------|-----------|--------------|------|
| π₀ | 41.0 | 38.0 | 84.0 | 96.0 | 64.8 |
| RDT | 33.0 | 21.0 | 42.0 | 95.0 | 47.8 |
| [[OpenVLA-OFT]] | 28.1 | 37.5 | 77.3 | 45.3 | 47.1 |
| **PAPO-VLA** | **59.3** | **41.9** | **90.1** | **86.7** | **69.5** |

**长/超长时域（280–650 步）:**

| 方法 | Handover Block | Stack Bowls | Blocks Rank | Put Bottles | Avg. |
|------|----------------|-------------|-------------|-------------|------|
| π₀ | 39.0 | 53.0 | 45.0 | 54.0 | 47.8 |
| RDT | 26.0 | 42.0 | 17.0 | 26.0 | 27.8 |
| [[OpenVLA-OFT]] | 33.1 | 40.6 | 70.2 | 42.2 | 46.5 |
| **PAPO-VLA** | **50.4** | **69.9** | **77.2** | **57.8** | **63.8** |

**关键发现**: 在所有三个时域尺度上 PAPO-VLA 均优于 π₀、RDT 和 [[OpenVLA-OFT]]；中时域 Handover Mic 一个任务 π₀ 反超（96.0 vs 86.7），可能与任务特性有关；但整体平均仍领先。

### Table 3: 消融实验（LIBERO）

| 配置 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| w/o Suff. & Nec.（纯 GRPO 基线） | 0.85 | 0.88 | 0.79 | 0.53 | 0.76 |
| w/o Suff.（仅 Nec.） | 0.89 | 0.90 | 0.87 | 0.80 | 0.87 |
| w/o Nec.（仅 Suff.） | 0.88 | 0.92 | 0.89 | 0.85 | 0.89 |
| **Full PAPO-VLA** | **0.93** | **0.98** | **0.98** | **0.94** | **0.96** |

**关键发现**: 充分性和必要性缺少任一都会显著降低性能，两者缺失时退化为 [[OpenVLA]] 基线（0.76）；充分性贡献略大于必要性（0.89 vs 0.87），但完整模型的调和平均综合效果明显优于任一单独使用（+7% 以上）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套各 10 个任务 | Franka Panda 仿真，每任务 50 训练集 | 训练+测试 |
| [[RoboTwin 2.0]] | 50 个任务（选 12） | 域随机化，3 类时域长度（100–650 步） | 测试 |

### 实现细节

- **Backbone**: OpenVLA-OFT（预训练 [[Vision-Language-Action Model]]）
- **LIBERO 设置**: 每任务 50 训练集，50 测试 rollout
- **RoboTwin2.0 设置**: 每任务 100 测试场景，12 任务按时域分 3 组
- **超参数**: $\eta = 0.15$（敏感性测试范围 [0, 0.3]）
- **硬件**: H100 GPU 集群

### 可视化结果

Figure 3 直观对比了 PAPO-VLA 和 [[OpenVLA]] 在两个代表性任务上的执行轨迹：PAPO-VLA 在多阶段操作（接近→抓取→放置）中保持稳定的规划动作，而 OpenVLA 在关键抓取时刻出现失稳导致后续阶段无法执行。

---

## 批判性思考

### 优点

1. **理论严谨**: 用概率因果框架（充分性+必要性）系统量化动作重要度，比直觉启发式更有说服力。
2. **即插即用**: PAPO 只修改 GRPO 的优势估计，对训练框架侵入性极低，易于集成到现有 VLA 训练流程。
3. **长时域收益显著**: Long horizon 任务的大幅提升（+0.20 以上）验证了关键规划动作对复杂任务的核心作用。

### 局限性

1. **估计近似性**: 因果充分性/必要性需要通过扰动轨迹估计条件期望，在实践中可能引入近似误差，文中未详述扰动方式的鲁棒性。
2. **超参数敏感性**: $\eta$ 和 top-k 规划动作比例需要调参，不同任务的最优值可能不同，泛化性存疑。
3. **仅限仿真评估**: 所有实验均在仿真环境中进行，真实机器人的验证缺失，sim-to-real 迁移效果未知。
4. **计算开销**: 需要额外的轨迹扰动估计（充分性和必要性），相比标准 GRPO 训练开销更大，文中未报告具体计算成本。

### 潜在改进方向

1. **自适应 k 选择**: 用学习的方式自动确定规划动作的比例，而非固定 top-k。
2. **真实机器人验证**: 在物理机器人上验证 sim-to-real 迁移效果。
3. **与扩散策略结合**: 探索在 [[Diffusion Policy]] 类 VLA 中引入类似规划感知机制。

### 可复现性评估

- [ ] 代码开源（暂未提供）
- [ ] 预训练模型（暂未提供）
- [x] 训练细节完整（超参数、benchmark 设置均有说明）
- [x] 数据集可获取（LIBERO 和 RoboTwin2.0 均公开）

---

## 关联笔记

### 基于
- [[OpenVLA-OFT]]: 使用其作为骨干模型
- [[GRPO]]: 基础策略优化算法，PAPO 在其之上扩展
- [[Vision-Language-Action Model]]: 整体技术框架

### 对比
- [[OpenVLA]]: 无规划感知的 GRPO 基线（=消融 w/o Suff. & Nec.）
- [[TGRPO]]: 另一个基于 GRPO 的 VLA 优化方法，LIBERO 上 Avg. 0.81 vs 0.96
- [[RIPT-VLA]]: 类似的 VLA 强化学习优化工作

### 方法相关
- [[GRPO]]: 核心优化框架
- [[因果充分性]]: 衡量动作对成功的充分性因果贡献
- [[因果必要性]]: 衡量动作对成功的必要性因果贡献
- [[Action Chunking]]: VLA 动作序列生成机制

### 硬件/数据相关
- [[LIBERO]]: 主要评估 benchmark
- [[RoboTwin 2.0]]: 域随机化评估 benchmark

---

## 速查卡片

> [!summary] PAPO-VLA: Planning-Aware Policy Optimization for VLA
> - **核心**: 识别规划动作 + 因果重要度 → 规划感知 GRPO 优化
> - **方法**: 动作变化幅度 × 结果门控 → 规划动作识别；因果充分性 + 必要性调和平均 → 重要度；叠加到 GRPO 优势估计
> - **结果**: LIBERO Avg. 0.96（+0.09 vs Nora），Long 子集 0.94（+0.20）；RoboTwin2.0 全面领先 π₀、RDT、OpenVLA-OFT
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-05-20*
