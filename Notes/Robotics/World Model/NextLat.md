---
title: "Next-Latent Prediction Transformers Learn Compact World Models"
method_name: "NextLat"
authors: [Jayden Teoh, Manan Tomar, Kwangjun Ahn, Edward S. Hu, Tim Pearce, Pratyusha Sharma, Akshay Krishnamurthy, Riashat Islam, Alex Lamb, John Langford]
year: 2025
venue: arXiv
tags: [world-model, latent-prediction, transformer, speculative-decoding, belief-state]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2511.05963v4
created: 2026-06-18
---

# 论文笔记：Next-Latent Prediction Transformers Learn Compact World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft Research |
| 日期 | November 2025 (v4: June 2026) |
| 项目主页 | — |
| 对比基线 | [[GPT]]、[[多头预测 (MTP)|MTP]]、[[JTP]]、[[BST]] |
| 链接 | [arXiv](https://arxiv.org/abs/2511.05963) / [Code](https://github.com/JaydenTeoh/NextLat) |

---

## 一句话总结

> NextLat 通过在[[潜在空间]]中进行自监督下一状态预测，使 [[Transformer]] 的隐状态可证明收敛到[[信念状态]]，从而在不增加推理开销的前提下实现更紧凑的世界模型和最高 3.3× 的推理加速。

---

## 核心贡献

1. **下一潜态预测（Next-Latent Prediction）**: 引入辅助目标，在训练时监督 Transformer 的中间隐状态预测未来隐状态，无需改变推理架构。
2. **理论保证（Belief State 收敛）**: 证明当同时满足下一 token 一致性和转移一致性时，学到的隐状态可证明收敛到[[信念状态]]（Belief State），即预测未来所必要的历史充分统计量。
3. **可变长度自投机解码（Variable-Length Self-Speculative Decoding）**: 利用学到的潜态动力学进行超出训练步长 $d$ 的自回归草稿生成，实现最高 3.32× 的推理加速。

---

## 问题背景

### 要解决的问题

标准[[自回归 Transformer]]（GPT 风格）在下一 token 预测中，中间隐状态 $h_t$ 不受显式约束——模型可以采用任意"捷径"完成预测而不学习到可泛化的世界状态表示，导致在推理、规划、世界建模等任务上的泛化能力有限。

### 现有方法的局限

- **GPT**：隐状态无约束，倾向于局部 n-gram 记忆，缺乏前瞻性表示。
- **[[多头预测 (MTP)|MTP]]（Multi-Token Prediction）**：在 token 空间同时预测多个未来 token，可提供额外梯度信号，但仍在 token 空间监督，无法约束中间隐状态。
- **JTP（Joint Token Prediction）**：联合预测，计算量随步长 $d$ 增长，且不保证 belief state 性质。
- **BST（Belief State Transformer）**：通过前向-后向计算显式学习信念状态，理论完备，但计算复杂度为 $O(T^2)$，参数量翻倍（2.57B vs 1.32B），训练速度仅为 GPT 的 29%。

### 本文的动机

若能在隐状态层面施加"转移一致性"约束，则无需 $O(T^2)$ 代价即可获得等价的 belief state 保证。同时，学到的潜态动力学网络可在推理时充当草稿生成器，自然支持[[投机解码]]。

---

## 方法详解

### 模型架构

NextLat 在标准 [[Transformer]] 基础上增加一个轻量级**潜态动力学网络** $p_\psi$（MLP 实现），在训练时预测未来隐状态：

- **输入**: token 序列 $X_{1:T}$
- **Backbone**: 标准因果 [[Transformer]]（GPT 架构），生成隐状态序列 $h_1, \ldots, h_T$
- **核心模块**: 潜态动力学网络 $p_\psi$（[[MLP]]），接受 $(h_t, X_{t+1})$ 预测 $h_{t+1}$
- **输出**: 与标准 GPT 相同的下一 token 分布（推理时不启用 $p_\psi$）
- **总参数（d=1）**: 1.40B（vs GPT 1.32B，仅增加 6%）

### 核心模块

#### 模块1: 潜态动力学网络（Latent Dynamics Network）

**设计动机**: 通过[[转移一致性]]约束，使隐状态可递归地预测未来状态，实现[[信念状态]]收敛。

**具体实现**:
- 使用轻量 [[MLP]] 实现 $p_\psi: (h_t, X_{t+1}) \mapsto \hat{h}_{t+1}$
- 多步递归展开：$\hat{h}_{t+i} = p_\psi(\hat{h}_{t+i-1}, X_{t+i})$，覆盖 $d$ 步前瞻
- 对目标隐状态使用[[停止梯度（Stop-Gradient）]]，防止[[表示坍塌]]

#### 模块2: KL 辅助损失头（KL Token-Prediction Head）

**设计动机**: 确保预测的潜态 $\hat{h}_{t+i}$ 不仅几何上接近真实 $h_{t+i}$，还能输出等价的 token 概率分布。

**具体实现**:
- 使用冻结输出头 $p_\theta^{sg}$ 分别计算真实状态和预测状态的 token 分布
- 用[[KL 散度]]约束两者之间的分布差异（[[知识蒸馏]]风格）
- 联合与 $L_{\text{next-h}}$ 相加构成完整 NextLat 目标

#### 模块3: 可变长度自投机解码（Variable-Length Self-Speculative Decoding）

**设计动机**: 利用学到的 $p_\psi$ 在推理时递归生成草稿 token，然后用主模型并行验证，实现无损加速。

**具体实现**:
- 从当前隐状态 $h_t$ 出发，用 $p_\psi$ 递归展开 $k$ 步生成草稿序列
- 主模型并行验证草稿，拒绝不匹配的 token（标准[[投机解码]]协议）
- 由于 $p_\psi$ 学到真正的信念状态转移，草稿接受率极高（d=2 时平均接受 4.86 个 token/步）
- 支持超出训练步长 $d$ 的外推（训练 d=2，推理可用到 d=10+）

---

## 关键公式

### 公式1: [[下一 Token 一致性]]

$$
p_\theta(X_{t+1} \mid h_t) = P(X_{t+1} \mid X_{1:t})
$$

**含义**: 模型输出头满足因果语言模型的标准条件分布。

**符号说明**:
- $p_\theta$: 参数化 Transformer 模型
- $h_t$: 第 $t$ 步隐状态
- $X_{1:t}$: 历史 token 序列

---

### 公式2: [[转移一致性]]

$$
p_\psi(h_{t+1} \mid h_t, X_{t+1}) = P(h_{t+1} \mid X_{1:t+1})
$$

**含义**: 潜态动力学网络满足 Markov 转移一致性——给定当前隐状态和下一 token，即可精确预测下一隐状态。

**符号说明**:
- $p_\psi$: 潜态动力学网络（MLP）
- $h_{t+1}$: 第 $t+1$ 步真实隐状态

---

### 公式3: [[下一隐状态损失]] (Next-Hidden State Loss)

$$
\mathcal{L}_{\text{next-h}}(\theta, \psi; d) = \mathbb{E}_t \left[ \frac{1}{d} \sum_{i=1}^{d} \text{SmoothL1Loss}\bigl(\text{sg}[h_{t+i}],\; \hat{h}_{t+i}\bigr) \right]
$$

**含义**: 在 $d$ 步前瞻范围内，用 [[Huber 损失|SmoothL1Loss]] 回归预测的隐状态与真实隐状态之间的差距；对目标施加[[停止梯度（Stop-Gradient）|stop-gradient]]防止坍塌。

**符号说明**:
- $d$: 前瞻步数（训练超参数）
- $\hat{h}_{t+i}$: 动力学网络递归预测的第 $t+i$ 步隐状态
- $\text{sg}[\cdot]$: stop-gradient 操作（目标不回传梯度）

---

### 公式4: [[KL 辅助损失]] (KL Token-Prediction Loss)

$$
\mathcal{L}_{\text{KL}}(\theta, \psi; d) = \mathbb{E}_t \left[ \frac{1}{d} \sum_{i=1}^{d} D_{\text{KL}}\!\Bigl(p_\theta^{sg}(\cdot \mid \text{sg}[h_{t+i}]) \;\|\; p_\theta^{sg}(\cdot \mid \hat{h}_{t+i})\Bigr) \right]
$$

**含义**: 确保预测隐状态在 token 分布层面与真实隐状态等价，类似[[知识蒸馏]]中的软标签对齐；输出头参数固定（stop-gradient），只优化 $\psi$。

**符号说明**:
- $p_\theta^{sg}$: 冻结输出头（不反传梯度）
- $D_{\text{KL}}$: [[KL 散度]]

---

### 公式5: [[NextLat 总目标]] (Complete NextLat Objective)

$$
\mathcal{L}_{\text{NextLat}}(\theta, \psi;\; d, \lambda_{\text{next-h}}, \lambda_{\text{KL}}) = \mathcal{L}_{\text{next-token}}(\theta) + \lambda_{\text{next-h}} \mathcal{L}_{\text{next-h}}(\theta, \psi; d) + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}(\theta, \psi; d)
$$

**含义**: 三项损失之和：标准下一 token 预测损失 + 隐状态回归损失 + KL 分布对齐损失；$\lambda$ 为权重超参数。

**符号说明**:
- $\mathcal{L}_{\text{next-token}}$: 标准语言模型交叉熵损失
- $\lambda_{\text{next-h}}, \lambda_{\text{KL}}$: 两个辅助损失的权重系数

---

## 关键图表

### Figure 1: 预测机制对比

![Figure 1 - Model Comparison](https://arxiv.org/html/2511.05963v4/images/model_comparison_2.png)

**说明**: 对比 GPT、MTP、JTP、NextLat 四种方法的预测机制。其他方法仅在 token 空间监督中间表示（隐式），而 NextLat 显式训练模型在潜在空间预测下一隐状态 $h_{t+1}$，从而约束了[[信念状态]]的形成。

---

### Figure 2: 可变长度自投机解码

![Figure 2 - Variable-Length Speculative Decoding](https://arxiv.org/html/2511.05963v4/images/flexible_spec_decoding.png)

**说明**: MTP 使用固定草稿长度（例如 d=2），每步消耗固定数量 token；而 NextLat 的潜态动力学网络可以递归展开任意步，无需固定草稿长度，从而减少草稿-验证轮次，进一步加速推理。

---

### Figure 3: 曼哈顿出租车路线重建地图（世界建模能力对比）

**Figure 3a — GPT**:

![Figure 3a - GPT map](https://arxiv.org/html/2511.05963v4/images/manhattan/close_gpt_random_walks_seq6400_deg8_dist0.5_randerr-1.png)

**Figure 3b — MTP**:

![Figure 3b - MTP map](https://arxiv.org/html/2511.05963v4/images/manhattan/close_mtp_random_walks_seq6400_deg8_dist0.5_randerr-1.png)

**Figure 3c — JTP**:

![Figure 3c - JTP map](https://arxiv.org/html/2511.05963v4/images/manhattan/close_jtp_random_walks_seq6400_deg8_dist0.5_randerr-1.png)

**Figure 3d — NextLat**:

![Figure 3d - NextLat map](https://arxiv.org/html/2511.05963v4/images/manhattan/nhs_manhattan.png)

**说明**: 通过从模型生成的轨迹重建曼哈顿街道地图，评估世界建模质量。NextLat 重建的地图最清晰，幻觉最少；GPT 产生大量不一致的道路走向；JTP 表现最差（Sequence Compression 仅 0.32）。

---

### Figure 5: Countdown 方程有效性

![Figure 5 - Countdown Validity](https://arxiv.org/html/2511.05963v4/x1.png)

**说明**: 在 Countdown 推理任务上，所有模型（d=1）的方程有效率（LHS = RHS）。NextLat 有效方程比例显著高于 GPT、MTP、BST，表明其前瞻性表示减少了"后悔性妥协"错误（即生成过程中的次优中间步骤）。

---

### Figure 6: Path-Star 图任务准确率

![Figure 6 - Path-Star Accuracy](https://arxiv.org/html/2511.05963v4/x2.png)

**说明**: 在不同规模的 Path-Star 图（$G_{2,10}$、$G_{5,5}$、$G_{7,7}$）上的路径规划准确率。NextLat 在所有图规模上均保持接近 100%；BST 在 $G_{7,7}$ 上失败；MTP/JTP 随图规模增大显著退化。NextLat 避免了 token 空间方法的"Clever Hans"捷径学习问题。

---

### Figure 7: Path-Star 图示例

![Figure 7 - Path-Star Illustration](https://arxiv.org/html/2511.05963v4/x3.png)

**说明**: $G_{5,5}$ Path-Star 图的结构示意。任务要求模型从星型图的一条臂（星臂）出发，找到通往中心再延伸至目标臂末端的正确路径，考察长程规划能力。

---

### Figure 8: TinyStories 线性探针结果

![Figure 8 - TinyStories Linear Probe](https://arxiv.org/html/2511.05963v4/images/tinystories_results_2.png)

**说明**: 在冻结的隐状态上训练线性探针，预测未来不同偏移位置的 token。与 GPT 相比，NextLat 的隐状态对未来 token 有更强的预测力（更低的交叉熵差值），证明其学到了更多的前瞻信息。

---

### Figure 9: FineWeb-Edu 投机解码加速结果

**Figure 9a — 各模型推理加速比**:

![Figure 9a - Speedup](https://arxiv.org/html/2511.05963v4/images/fineweb/fineweb_spec_graph.png)

**Figure 9b — 累积接受率**:

![Figure 9b - Cumulative Acceptance](https://arxiv.org/html/2511.05963v4/images/fineweb/fineweb_cumulative_acceptance.png)

**说明**: (a) NextLat 在 FineWeb-Edu 验证集上的推理加速比显著超越 JTP 和 MTP；d=2 时最高达 3.32×。(b) 累积接受率（全部草稿 token 均被接受的概率）随草稿长度下降，但 NextLat 虽仅用 d=2 训练，其曲线在 d=10 时仍保持有效，展示了超出训练步长的外推能力。

---

### Figure 10: A₅ 词问题长度泛化

![Figure 10 - A5 Length Generalization](https://arxiv.org/html/2511.05963v4/images/a5_length_gen.png)

**说明**: 在 NC¹-complete 的 A₅ 词问题上测试不同序列长度下的准确率泛化。使用 NextLat 协同训练的 RNN 在 12-token 训练下泛化到 36-token 序列达 >95% 准确率；而浅层 Transformer（2 层）在相同条件下无法泛化，即使直接在 36-token 上训练也无法解决该任务。

---

### Table 1: 曼哈顿世界建模指标

| Metric | GPT | MTP | JTP | NextLat | True |
|--------|-----|-----|-----|---------|------|
| Next-Token Test (↑) | 100% | 100% | 100% | 100% | 100% |
| Valid Trajectories (↑) | 97.0% | 98.1% | 97.1% | **98.7%** | 100% |
| Sequence Compression (↑) | 0.65 | 0.64 | 0.32 | **0.71** | 1.00 |
| Effective Latent Rank (↓) | 160.1 | 57.7 | 215.8 | **52.7** | — |
| Detour Robustness (↑) | 85.0% | 95.0% | 87.0% | **95.0%** | 100% |

**说明**: NextLat 在所有世界建模指标上均达到最优或并列最优。有效潜态秩（Effective Latent Rank）最低（52.7），表明其学到了最紧凑的表示；序列压缩率最高（0.71），说明其生成的轨迹最接近真实地图的熵率。

---

### Table 2: 语言建模评估（1.3B 参数，100B token FineWeb-Edu）

| Model | FW-Edu ppl ↓ | Wiki ppl ↓ | LAMB ppl ↓ | LAMB acc ↑ | PIQA | HellaS | Wino | ARC-e | ARC-c | SIQA | SciQ | Avg ↑ |
|-------|-------------|-----------|-----------|-----------|------|--------|------|-------|-------|------|------|------|
| GPT | 10.52 | 17.93 | 20.26 | 42.07% | 73.45% | 58.79% | 60.46% | 68.18% | 39.16% | 42.32% | 86.10% | 58.82% |
| JTP (d=1) | 11.08 | 19.28 | 21.88 | 41.35% | 74.92% | 57.43% | 58.64% | 68.73% | 39.25% | 42.99% | 87.30% | 58.83% |
| JTP (d=2) | 11.18 | 19.60 | 22.11 | 41.37% | 73.34% | 56.84% | 59.98% | 68.86% | 38.57% | 43.35% | 86.70% | 58.63% |
| MTP (d=1) | 10.90 | 18.82 | 20.23 | 41.26% | 74.32% | 58.05% | 60.54% | 68.52% | 38.91% | 42.84% | 85.40% | 58.76% |
| MTP (d=2) | 11.00 | 18.61 | 18.34 | 43.43% | 72.80% | 57.92% | 59.35% | 68.35% | 39.08% | 41.97% | 86.60% | 58.69% |
| NextLat (d=1) | 10.83 | 18.39 | 19.77 | 41.08% | 73.07% | 58.35% | 59.27% | 69.65% | 39.68% | 43.24% | 86.00% | 58.79% |
| **NextLat (d=2)** | 10.88 | 18.44 | **17.83** | **43.86%** | 73.61% | 57.79% | 59.20% | 69.74% | **40.10%** | 41.91% | **87.50%** | **59.21%** |

**说明**: NextLat (d=2) 在 LAMBADA 困惑度（17.83）、LAMBADA 准确率（43.86%）和平均准确率（59.21%）上均达到最优，展现出更强的长程依赖建模能力。

---

### Table 3: 各域投机解码加速比（d=1 和 d=2）

| Model | Wikipedia | Books | Code | Math |
|-------|-----------|-------|------|------|
| JTP (d=1) | 1.46× / 0.96 tkns | 1.47× / 0.97 tkns | 1.47× / 0.98 tkns | 1.46× / 0.97 tkns |
| JTP (d=2) | 1.88× / 1.84 tkns | 1.90× / 1.89 tkns | 1.88× / 1.85 tkns | 1.89× / 1.86 tkns |
| MTP (d=1) | 1.38× / 0.91 tkns | 1.39× / 0.95 tkns | 1.40× / 0.97 tkns | 1.39× / 0.95 tkns |
| MTP (d=2) | 1.68× / 1.72 tkns | 1.72× / 1.83 tkns | 1.75× / 1.91 tkns | 1.72× / 1.84 tkns |
| **NextLat (d=1)** | **2.68× / 3.52 tkns** | **2.72× / 3.64 tkns** | 2.29× / 2.66 tkns | 2.30× / 2.72 tkns |
| **NextLat (d=2)** | 3.21× / 4.59 tkns | **3.32× / 4.86 tkns** | 2.38× / 2.83 tkns | 2.87× / 3.94 tkns |

**说明**: NextLat 的投机解码加速效果远超 JTP 和 MTP——即使仅使用 d=1，NextLat 在 Wikipedia 上即达到 2.68×，而其他方法 d=1 仅约 1.4×。Books d=2 时每步平均接受 4.86 个草稿 token，超过训练步长 $d=2$，体现了超出训练范围的外推能力。

---

### Table 4: 计算资源对比

| 指标 | GPT | BST | MTP (d=1/2/8) | JTP (d=1/2/8) | NextLat (d=1/2/8) |
|------|-----|-----|---------------|---------------|-------------------|
| 训练参数量 | 1.32B | 2.57B | 1.37B/1.42B/1.72B | 1.34B | 1.40B |
| 推理参数量 | 1.32B | 1.32B/2.57B | 1.32B | 1.34B | **1.32B** |
| 训练速度 (steps/sec) | 3.09 | 0.89 | 2.80/2.58/1.70 | 3.15/2.92/2.02 | **3.09**/2.79/1.73 |
| 梯度信号量 | $O(T)$ | $O(T^2)$ | $O(Td)$ | $O(Td)$ | $O(Td)$ |

**说明**: NextLat (d=1) 的训练速度与 GPT 完全持平（均为 3.09 steps/sec），但获得了接近 BST 的理论保证，且推理时无额外参数开销。BST 虽然信念状态最完备，但训练成本为 GPT 的 3.5×，参数量翻倍。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Manhattan Taxi Rides | — | 真实出租车轨迹，用于世界建模评估 | 世界建模测试 |
| Countdown | — | 数字推理任务（24 点风格） | 推理/规划 |
| Path-Star Graphs | $G_{k,n}$ 系列 | 图导航，考察长程规划 | 规划测试 |
| TinyStories | — | 简单英文故事，小规模 LM | 隐状态线性探针 |
| FineWeb-Edu | 100B tokens | 教育内容网络文本 | 大规模语言建模 |
| Wikipedia, Books, Code, Math | — | 多域文本 | 投机解码评估 |
| A₅ Word Problem | 12→36 token | NC¹-complete 状态追踪 | 长度泛化测试 |

### 实现细节

- **Backbone**: GPT 风格 [[Transformer]]（1.32B 参数）
- **动力学网络**: 轻量 [[MLP]]，无需复杂架构
- **优化器**: AdamW（细节见附录）
- **训练数据**: FineWeb-Edu 100B tokens（大规模实验）
- **前瞻步长 $d$**: 主要实验使用 d=1 和 d=2
- **损失权重**: $\lambda_{\text{next-h}}$ 和 $\lambda_{\text{KL}}$ 通过小规模消融确定
- **硬件**: 未明确列出

### 可视化结果

- 曼哈顿地图重建展示 NextLat 的道路一致性最高，GPT 产生大量幻觉街道
- Path-Star 图上 NextLat 接近 100% 正确率，其他方法在大图上显著退化
- TinyStories 线性探针证明 NextLat 隐状态编码了更多的未来 token 信息

---

## 批判性思考

### 优点

1. **理论完备**：Theorem 3.2 给出了可证明的信念状态收敛性，是少数在方法设计上有理论支撑的工作之一。
2. **计算高效**：d=1 时与标准 GPT 训练速度完全相同，参数增加仅 6%，推理零开销。
3. **多任务泛化**：在世界建模、推理、规划、语言建模四个不同域均展现提升，说明学到的表示确实更基础性。
4. **自投机解码的独特优势**：通过潜态递归实现超出训练步长的外推，是其他方法（如 MTP 固定步长）所不具备的。

### 局限性

1. **超参选择依赖小规模消融**：stop-gradient、KL 辅助损失等设计选择未在大规模上充分验证，可能不是最优配置。
2. **多步监督必要性存疑**：作者承认在大规模下 $d>1$ 的必要性尚不明确。
3. **动力学网络架构过简**：仅使用 MLP，更表达力的潜态动力学（如 Transformer 块）可能进一步提升。
4. **投机解码使用固定草稿长度**：未探索自适应草稿长度（可能进一步提升加速比）。
5. **A₅ 实验中 Transformer 本身失败**：NextLat 借助 RNN 解决了 NC¹ 任务，但实际上是 RNN 的胜利，非 Transformer 本身能力的突破。

### 潜在改进方向

1. 探索更表达力的潜态动力学网络（Transformer 块代替 MLP）
2. 自适应草稿长度的投机解码策略
3. 在更大规模（>7B 参数）上验证多步监督的必要性
4. 分析学到的信念状态的结构（哪类信息被压缩进了隐状态）

### 可复现性评估

- [x] 代码开源（https://github.com/JaydenTeoh/NextLat）
- [ ] 预训练模型（未提供检查点）
- [x] 训练细节完整（附录含伪代码）
- [x] 数据集可获取（FineWeb-Edu 公开）

---

## 关联笔记

### 基于

- [[自回归 Transformer]]: 标准 GPT 架构作为 Backbone
- [[信念状态]]: 核心理论目标，证明隐状态收敛于此
- [[潜在世界模型]]: 本文方法在潜在空间建模的直接动机

### 对比

- [[多头预测 (MTP)|MTP（Multi-Token Prediction）]]: token 空间多步预测，无潜态约束，投机解码加速有限
- [[BST（Belief State Transformer）]]: 理论等价但计算 $O(T^2)$，参数翻倍
- [[JTP]]: 联合 token 预测，无信念状态保证

### 方法相关

- [[投机解码]]: NextLat 推理加速的核心机制
- [[停止梯度（Stop-Gradient）]]: 防止潜态坍塌的关键技巧
- [[知识蒸馏]]: KL 辅助损失的设计思路
- [[KL 散度]]: KL 损失的度量核心
- [[Huber 损失]]: 隐状态回归使用 SmoothL1Loss

### 方法在 World Model 语境下

- [[潜在世界模型]]: NextLat 本质上是一种紧凑潜态动力学学习方法

---

## 速查卡片

> [!summary] Next-Latent Prediction Transformers (NextLat)
> - **核心**: 训练时在潜在空间预测下一隐状态，可证明收敛到信念状态
> - **方法**: 潜态动力学 MLP + SmoothL1 隐状态回归 + KL 辅助损失
> - **结果**: 世界建模/推理/规划全面提升，投机解码最高 3.32× 加速
> - **代码**: https://github.com/JaydenTeoh/NextLat

---

*笔记创建时间: 2026-06-18*
