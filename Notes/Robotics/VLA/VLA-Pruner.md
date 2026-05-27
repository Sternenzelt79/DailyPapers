---
title: "Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference"
method_name: "VLA-Pruner"
authors: [Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao]
year: 2025
venue: arXiv
tags: [vla, token-pruning, efficient-inference, visual-tokens, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2511.16449
created: 2026-05-27
---

# 论文笔记：Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MINT Lab, SJTU |
| 日期 | November 2025 |
| 项目主页 | — |
| 对比基线 | [[FastV]], [[SparseVLM]], [[DivPrune]], [[VLA-Cache]] |
| 链接 | [arXiv](https://arxiv.org/abs/2511.16449) / [Code](https://github.com/MINT-SJTU/VLA-Pruner) |

---

## 一句话总结

> VLA-Pruner 提出语义-动作双重重要性估计 + Combine-then-Filter 策略，解决 VLM Token 剪枝方法直接用于 VLA 时的"语义-动作鸿沟"问题，实现最高 1.99× 推理加速。

---

## 核心贡献

1. **语义-动作鸿沟分析**: 揭示 [[VLA|VLA 模型]] 推理中 Prefill 阶段（语义理解）与 Decode 阶段（动作执行）所关注的视觉 Token 存在约 50% 的不重叠，直接套用 VLM Token 剪枝方法会导致严重性能下降。
2. **时序感知双层重要性估计**: 利用[[指数移动平均|EMA]] 对历史动作注意力进行时序平滑，在 Prefill 阶段无需当前动作注意力即可估计动作相关 Token 的重要性。
3. **Combine-then-Filter 策略**: 先取语义候选集与动作候选集的并集，再通过 [[最大最小多样性问题|Max-Min Diversity Problem]] 去冗余，实现紧凑且任务相关的 Token 保留。

---

## 问题背景

### 要解决的问题

[[VLA|VLA 模型]]在机器人操控部署时需要对连续视觉流进行实时处理，计算开销巨大。现有 [[视觉 Token 剪枝|Visual Token Pruning]] 方法借自 VLM 加速，但直接应用于 VLA 会导致严重的操控性能退化。

### 现有方法的局限

- **[[FastV]]**: 基于 Prefill 阶段的注意力分数剪枝，仅关注语义显著性。
- **[[SparseVLM]]**: 使用 Text-to-Vision 注意力，同样以语义为中心。
- **[[DivPrune]]**: 多样性驱动的选择，缺乏动作相关性建模。
- **[[VLA-Cache]]**: 利用时序连续性做特征缓存，效率不足（比 VLA-Pruner 慢 16%、多 20% FLOPs）。
- 这些方法均忽视了 VLA 推理中"动作解码对视觉局部精确信息的强依赖"。

### 本文的动机

VLA 推理包含两个具有不同视觉需求的阶段：
- **Prefill 阶段**：关注语义，注意力广而分散。
- **Action Decode 阶段**：关注操控精度，注意力局部集中。

两阶段 Top-k Token 重叠率仅约 50%（极端情况低于 30%），因此保留语义显著 Token 不足以支撑动作决策。同时，相邻时刻动作注意力高度一致（时序连续性），可用于跨时步估计动作相关 Token。

---

## 方法详解

### 模型架构

VLA-Pruner 是**训练无关（Training-Free）的即插即用模块**，可接入任意含有动作-视觉[[交叉注意力]]或[[自注意力机制]]的 VLA 架构：

- **输入**: 视觉 Token 序列 $\mathbf{E}_v \in \mathbb{R}^{M \times d}$（$M$ 为 Token 数，$d$ 为特征维度）
- **Backbone**: OpenVLA（[[自回归策略|Autoregressive]]）、OpenVLA-OFT（Action-Chunk）、π0（[[扩散策略|Diffusion-Based]]）
- **核心模块**: 语义-动作重要性估计 + [[Combine-then-Filter]] 策略
- **输出**: 剪枝后的视觉 Token 子集 $\tilde{\mathbf{E}}_v$，$|\tilde{\mathbf{E}}_v| = \tilde{M}$
- **剪枝位置**: Transformer 第 3 层之后

### 核心模块

#### 模块1: 语义-动作重要性估计（Semantic-Action Importance Estimation）

**设计动机**: 同时捕获语义显著性和动作相关性，弥合两阶段视觉需求差异。

**Prefill 注意力（语义）**:

[[多头自注意力]] 计算视觉语言 Prefill 阶段的注意力矩阵，再对所有 Token 维度取均值得到每个视觉 Token 的语义重要性分数。

**动作解码注意力（动作相关）**:

从[[自注意力机制|Action-to-Vision Cross-Attention]] 中提取动作 Token 对视觉 Token 的注意力均值，反映每个视觉 Token 对动作决策的贡献。

**时序平滑（解决当前时步动作注意力不可用问题）**:

利用[[指数移动平均|EMA]] 对历史动作注意力进行有限窗口加权平均，以历史估计代替当前动作注意力。

#### 模块2: Combine-then-Filter 策略

**设计动机**: 避免分数融合引入超参敏感性，同时消除候选集中的冗余 Token。

**三步流程**:

1. **语义-动作候选集构建**: 分别按语义分数和动作分数取 Top-$\tilde{M}$ Token，得到 $\mathcal{C}_{vl}$ 和 $\mathcal{C}_{act}$。
2. **相关性最大化合并**: 取并集 $\mathcal{C}_{dual} = \mathcal{C}_{vl} \cup \mathcal{C}_{act}$，确保两阶段关键 Token 均被保留。
3. **多样性过滤去冗余**: 在 $\mathcal{C}_{dual}$ 上求解[[最大最小多样性问题|Max-Min Diversity Problem (MMDP)]]，贪心选出 $\tilde{M}$ 个互相距离最大的 Token。

---

## 关键公式

### 公式1: [[多头自注意力|Prefill 阶段注意力矩阵]]

$$
\mathcal{A}_{vl} = \text{Softmax}\!\left(\frac{\mathbf{Q}_{vl} \mathbf{K}_{vl}^\top}{\sqrt{d_k}}\right)
$$

**含义**: 视觉-语言 Prefill 阶段计算的注意力矩阵，反映所有 Token 对视觉 Token 的关注程度。

**符号说明**:
- $\mathbf{Q}_{vl}, \mathbf{K}_{vl}$: Prefill 阶段的 Query 和 Key 矩阵
- $d_k$: Key 的维度（缩放因子）

### 公式2: [[视觉 Token 剪枝|Prefill 语义重要性分数]]

$$
\mathcal{S}_{vl}[m] = \frac{1}{M+N} \sum_{i=1}^{M+N} \mathcal{A}_{vl}[i, m], \quad m = 1, \ldots, M
$$

**含义**: 对所有 Token（视觉 $M$ + 语言 $N$）维度取均值，衡量第 $m$ 个视觉 Token 的语义显著性。

**符号说明**:
- $M$: 视觉 Token 数量
- $N$: 语言 Token 数量
- $\mathcal{A}_{vl}[i,m]$: 第 $i$ 个 Token 对第 $m$ 个视觉 Token 的注意力权重

### 公式3: [[交叉注意力|Action Decode 阶段注意力矩阵]]

$$
\mathcal{A}_{act} = \text{Softmax}\!\left(\frac{\mathbf{Q}_{act} \mathbf{K}_{vl}^\top}{\sqrt{d_k}}\right)
$$

**含义**: 动作解码阶段的 Query 对视觉 Token 的 Key 做注意力，反映动作决策对视觉 Token 的依赖。

**符号说明**:
- $\mathbf{Q}_{act}$: 动作解码阶段的 Query 矩阵
- $\tilde{N}$: 动作 Token 数量

### 公式4: [[视觉 Token 剪枝|动作相关视觉重要性分数]]

$$
\mathcal{S}_{act}[m] = \frac{1}{\tilde{N}} \sum_{i=1}^{\tilde{N}} \mathcal{A}_{act}[i, m], \quad m = 1, \ldots, M
$$

**含义**: 对所有动作 Token 维度取均值，衡量第 $m$ 个视觉 Token 对动作执行的贡献。

**符号说明**:
- $\tilde{N}$: 动作 Token 数量

### 公式5: [[视觉 Token 剪枝|Token 剪枝优化目标]]

$$
\min_f \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}}) \quad \text{s.t.} \quad |f(\mathbf{E}_v)| = \tilde{M}
$$

**含义**: 在保留 $\tilde{M}$ 个视觉 Token 的约束下，最小化原始模型分布与剪枝后模型分布的差异。

**符号说明**:
- $f$: Token 选择函数
- $\mathcal{P}, \tilde{\mathcal{P}}$: 原始和剪枝后的模型分布
- $\tilde{M}$: 目标 Token 保留数量

### 公式6: [[指数移动平均|时序平滑（EMA）]]

$$
\hat{\mathcal{S}}_{act}^t = \frac{\sum_{i=1}^{w} \gamma^i \mathcal{S}_{act}^{t-i}}{\sum_{i=1}^{w} \gamma^i}
$$

**含义**: 用历史 $w$ 步的动作注意力分数做指数加权平均，估计当前时步的动作相关 Token 重要性（因当前动作注意力在 Prefill 时不可用）。

**符号说明**:
- $w$: 历史窗口大小（实验设为 3）
- $\gamma$: 衰减率（实验设为 0.8），$\gamma \in [0,1]$
- $\mathcal{S}_{act}^{t-i}$: $t-i$ 时刻的动作重要性分数

### 公式7: [[最大最小多样性问题|MMDP 多样性过滤]]

$$
\tilde{\mathcal{C}} = \arg\max_{\mathcal{C} \subset \mathcal{C}_{dual},\, |\mathcal{C}| = \tilde{M}} \min_{\substack{i,j \in \mathcal{C} \\ i \neq j}} d(v_i, v_j)
$$

**含义**: 在双候选集 $\mathcal{C}_{dual}$ 中选出 $\tilde{M}$ 个 Token，使任意两个 Token 特征间的余弦距离最小值最大化，从而去除冗余。

**符号说明**:
- $\mathcal{C}_{dual} = \mathcal{C}_{vl} \cup \mathcal{C}_{act}$: 语义和动作候选集的并集
- $d(v_i, v_j)$: 余弦距离（见公式8）

### 公式8: [[余弦相似度|余弦距离]]

$$
d(v_i, v_j) = 1 - \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}
$$

**含义**: 衡量两个 Token 特征向量的多样性，距离越大表示越不相似，用于 MMDP 的贪心求解。

**符号说明**:
- $v_i, v_j$: 视觉 Token 的特征向量

---

## 关键图表

### Figure 1: 不同剪枝方法性能对比

![[VLA-Pruner_fig1.png]]

**说明**: 在不同剪枝比例下，[[FastV]]、[[SparseVLM]]、[[DivPrune]] 等 VLM 剪枝方法直接用于 VLA 推理时性能严重下降，而 VLA-Pruner 在高剪枝率下仍保持接近原始性能。

### Figure 2: VLA 推理的差异化注意力模式

![[VLA-Pruner_fig2.png]]

![[VLA-Pruner_fig3.png]]

![[VLA-Pruner_fig4.png]]

![[VLA-Pruner_fig5.png]]

**说明**: (a-b) Prefill 与 Action Decode 阶段 Top-k Token 重叠率约 50%，单次推演可低至 30% 以下，揭示两阶段视觉 Token 重要性不匹配问题。(c) Prefill 注意力广泛分布（语义覆盖），(d) Action Decode 注意力局部集中（操控精度导向）。

### Figure 3: VLA-Pruner 整体流程

![[VLA-Pruner_fig6.png]]

**说明**: 以预算 $k=3$ 为例，VLA-Pruner 分别选取语义显著和动作显著的候选 Token，再通过 MMDP 去冗余，最终保留紧凑且任务相关的 Token 集合。

### Figure 4: LIBERO 基准测试结果（OpenVLA）

![[VLA-Pruner_fig7.png]]

![[VLA-Pruner_fig8.png]]

![[VLA-Pruner_fig9.png]]

![[VLA-Pruner_fig10.png]]

**说明**: 在 LIBERO 四个子任务（Spatial/Object/Goal/Long）的不同剪枝比例下，VLA-Pruner 始终优于所有基线，87.5% 剪枝率下仍保持 88.9% 相对精度（FastV 仅 63%）。

### Figure 5: 真实机器人实验（xArm6）

![[VLA-Pruner_fig11.png]]

**说明**: 在真实 6-DoF xArm6 机械臂上，75% 剪枝率下 VLA-Pruner 在四个操控任务中相对精度最高，验证了实际部署价值。

### Figure 6: VLA-Pruner 消融与分析

![[VLA-Pruner_fig12.png]]

![[VLA-Pruner_fig13.png]]

![[VLA-Pruner_fig14.png]]

**说明**: 消融实验验证 Combine-then-Filter 策略优于 Prefill-only、Action-only 和 Score-fusion；敏感性分析表明窗口大小 $w$ 的选择具有鲁棒性；可视化展示 VLA-Pruner 同时保留了语义广覆盖和局部精确信息。

### Table 1: LIBERO 基准测试定量结果（OpenVLA，不同剪枝率）

| 保留率 | 方法 | Spatial | Object | Goal | Long | 相对精度 | FLOPs(T) | 延迟(ms) |
|--------|------|---------|--------|------|------|----------|----------|---------|
| 50% | Vanilla | 87.6 | 84.6 | 78.6 | 52.2 | 100.0% | 1.906 | 236.41 |
| 50% | FastV | 86.2 | 81.6 | 77.2 | 50.6 | 97.43% | 1.136 | 172.32 |
| 50% | SparseVLM | 85.6 | 80.5 | 75.0 | 48.6 | 95.44% | 1.155 | 175.77 |
| 50% | DivPrune | 82.6 | 78.8 | 71.8 | 47.6 | 92.64% | 1.105 | 173.88 |
| 50% | VLA-Cache | 87.1 | 80.7 | 78.6 | 51.8 | 98.48% | 1.384 | 192.20 |
| 50% | **VLA-Pruner** | **88.2** | **85.8** | **79.4** | **56.4** | **102.45%** | **1.139** | **178.38** |
| 25% | Vanilla | 95.8 | 98.7 | 96.3 | 90.7 | 100.0% | 3.903 | 135.78 |
| 25% | FastV | 81.6 | 69.6 | 71.6 | 43.8 | 87.62% | 0.757 | 141.25 |
| 25% | SparseVLM | 83.4 | 72.8 | 67.6 | 45.6 | 88.67% | 0.772 | 144.12 |
| 25% | DivPrune | 77.4 | 61.3 | 65.4 | 41.4 | 80.96% | 0.743 | 142.12 |
| 25% | VLA-Cache | 78.1 | 73.2 | 70.2 | 45.5 | 88.08% | 0.961 | 164.17 |
| 25% | **VLA-Pruner** | **85.4** | **82.5** | **78.4** | **51.8** | **98.48%** | **0.758** | **144.81** |
| 12.5% | Vanilla | 62.0 | 58.5 | 55.8 | 18.8 | 63.08% | 0.568 | 125.76 |
| 12.5% | FastV | 62.0 | 58.5 | 55.8 | 18.8 | 63.08% | 0.568 | 125.76 |
| 12.5% | SparseVLM | 65.8 | 55.3 | 55.4 | 19.0 | 63.20% | 0.581 | 128.77 |
| 12.5% | DivPrune | 55.4 | 54.8 | 51.4 | 17.6 | 58.06% | 0.569 | 125.36 |
| 12.5% | VLA-Cache | 52.5 | 50.1 | 52.0 | 15.1 | 54.79% | 0.710 | 145.93 |
| 12.5% | **VLA-Pruner** | **80.2** | **78.4** | **69.0** | **42.8** | **88.90%** | **0.571** | **129.01** |

**关键发现**: 在极高剪枝率（87.5% Token 被剪除）下，VLA-Pruner 相对精度（88.9%）远超 FastV（63%）和 VLA-Cache（54.8%），且 FLOPs 与 FastV 相当。

### Table 2: SIMPLER 跨环境评估（75% 剪枝率）

| 方法 | Move Near | Pick Coke Can | Open/Close Drawer | Overall | 相对精度 |
|------|-----------|---------------|-------------------|---------|----------|
| OpenVLA (Vanilla) | 54.0% | 52.8% | 49.5% | 52.1% | 100% |
| FastV | 38.7% | 41.9% | 33.7% | 38.1% | 73.1% |
| VLA-Cache | 41.2% | 40.6% | 38.8% | 40.2% | 77.2% |
| **VLA-Pruner** | **52.4%** | **50.1%** | **48.8%** | **50.4%** | **96.8%** |

**关键发现**: SIMPLER 跨环境泛化测试中，VLA-Pruner 以 96.8% 相对精度大幅领先 FastV（73.1%）和 VLA-Cache（77.2%）。

---

## 实验

### 数据集 / 评测环境

| 环境 | 规模 | 特点 | 用途 |
|------|------|------|------|
| LIBERO | 4 子任务 × 10 任务 × 500 episodes | 模拟操控基准，含 Spatial/Object/Goal/Long | 主要评测 |
| SIMPLER | Visual matching + Variant aggregation | 跨环境泛化，涵盖 Move Near / Pick / Drawer | 泛化评测 |
| Real Robot (xArm6) | 4 操控任务 × 100 trials | 真实 6-DoF 机械臂 | 实际部署验证 |

### 实现细节

- **适用架构**: [[自回归策略|OpenVLA]]（自回归）、OpenVLA-OFT（Action-Chunk）、[[扩散策略|π0]]（扩散式）
- **剪枝层**: Transformer 第 3 层
- **Prefill 注意力**: 最后一层 Prefill 注意力
- **动作注意力**: 后半层动作-视觉注意力均值
- **EMA 参数**: 窗口 $w=3$，衰减率 $\gamma=0.8$
- **Warm-up**: 前 $w$ 步用于建立注意力历史
- **训练**: 无需训练，即插即用

### 可视化结果

消融实验表明，Combine-then-Filter 策略始终优于三种退化变体（Prefill-only、Action-only、Score-fusion）。敏感性分析显示窗口大小 $w=3$ 在合理范围内具有鲁棒性。

---

## 批判性思考

### 优点

1. **问题洞察深刻**: 首次系统量化 VLA 推理两阶段注意力不匹配（~50% 重叠），为后续工作提供理论基础。
2. **无需训练**: 即插即用，适配多种 VLA 架构（自回归、Action-Chunk、扩散式），工程实用性强。
3. **性能提升显著**: 高剪枝率下大幅超越所有基线，同时在 FLOPs 和延迟上与 FastV 相当。

### 局限性

1. **依赖历史注意力**: EMA 方法需要 Warm-up 阶段，在首帧或短序列场景中效果可能受限。
2. **跨注意力依赖**: 需要模型提供动作-视觉注意力矩阵，对仅有隐式动作表示的 VLA 架构适用性待验证。
3. **剪枝层固定**: 固定在第 3 层剪枝，不同深度模型可能需要重新调参。

### 潜在改进方向

1. **自适应剪枝率**: 根据任务复杂度动态调整 $\tilde{M}$，而非全局固定。
2. **无 Warm-up 初始化**: 研究首帧的动作注意力替代估计（如语言-视觉注意力代理）。

### 可复现性评估

- [x] 代码开源（https://github.com/MINT-SJTU/VLA-Pruner）
- [ ] 预训练模型（依赖 OpenVLA / π0 基础权重）
- [x] 训练细节完整（超参数明确）
- [x] 数据集可获取（LIBERO / SIMPLER 均公开）

---

## 关联笔记

### 基于

- [[VLA]]: VLA-Pruner 加速的目标架构
- [[FastV]]: 启发语义注意力重要性估计的先驱方法
- [[指数移动平均]]: 时序平滑的核心技术

### 对比

- [[FastV]]: 仅用语义注意力剪枝，高剪枝率下 VLA 性能崩溃
- [[SparseVLM]]: Text-to-Vision 注意力剪枝，忽视动作相关性
- [[DivPrune]]: 多样性选择，缺乏两阶段重要性联合建模
- [[VLA-Cache]]: 时序缓存加速，效率低于 VLA-Pruner

### 方法相关

- [[视觉 Token 剪枝]]: 核心问题域
- [[最大最小多样性问题]]: 去冗余优化方法
- [[指数移动平均]]: 时序注意力平滑
- [[交叉注意力]]: 动作-视觉注意力来源

### 硬件/数据相关

- [[LIBERO]]: 主要评测基准
- [[SIMPLER]]: 跨环境泛化评测

---

## 速查卡片

> [!summary] VLA-Pruner (arXiv 2511.16449)
> - **核心**: 揭示 VLA 推理语义-动作注意力鸿沟，双重重要性估计 + Combine-then-Filter 策略
> - **方法**: EMA 时序平滑动作注意力 + 候选集合并 + MMDP 去冗余，训练无关
> - **结果**: LIBERO 87.5% 剪枝率下 88.9% 相对精度（FastV 仅 63%），最高 1.99× 加速
> - **代码**: [MINT-SJTU/VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner)

---

*笔记创建时间: 2026-05-27*
