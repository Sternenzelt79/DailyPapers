---
title: "HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation"
method_name: "HiMem-WAM"
authors: [Xiaoquan Sun, Ruijian Zhang, Chen Cao, Yihan Sun, Jiahui Chen, Zetian Xu, Bo Chen, Haijier Chen, Zhen Yang, Jiarun Zhu, Yijun Hong, JingZhe Xu, Jingrui Pang, Mingqi Yuan, Jiayu Chen]
year: 2026
venue: arXiv
tags: [world-action-model, hierarchical-policy, memory-gated, robotic-manipulation, latent-action, long-horizon]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.10363
created: 2026-06-10
---

# 论文笔记：HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未披露 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[Fast-WAM]], [[WorldVLA]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.10363) |

---

## 一句话总结

> 通过分层潜动作（低级运动 + 高级技能）与边界触发的记忆门控，HiMem-WAM 让机器人无需测试时光流即可高效完成长时域操作任务。

---

## 核心贡献

1. **分层潜动作表示**: 将运动分解为低级运动原语（[[Latent Action|低级潜动作]]）与高级可复用技能（[[Skill Latent|技能潜变量]]），实现短时域执行与长时域规划的桥梁
2. **边界感知记忆门**: 在预测的技能转换边界处写入紧凑任务状态，避免了密集历史聚合，实现因果推理，无需测试时视频生成或光流估计
3. **三阶段训练流水线**: Stage I 学习低级潜动作分词器；Stage II 发现技能潜变量并预训练规划器/执行器；Stage III 激活记忆并端到端微调

---

## 问题背景

### 要解决的问题

机器人长时域操作任务需要跟踪任务进度并记忆历史关键状态。现有 [[World Action Model|世界动作模型]] 在处理需要记忆的多步操作序列时性能显著下降。

### 现有方法的局限

- 现有 WAM 方法（如 [[Fast-WAM]]）缺乏显式的历史状态记忆机制，在需要记忆的长时域任务（RMBench M(n) 类型）上表现较差
- 部分方法需要测试时进行光流估计或未来帧视频生成，增加了推理复杂度和延迟
- 基于固定时间间隔的密集历史聚合缺乏对任务结构的对齐，导致冗余信息干扰

### 本文的动机

将 [[Temporal Abstraction|时间抽象]] 与事件驱动的记忆机制结合：通过学习到的技能边界来触发记忆写入，既保证了因果推理，又避免了无关历史信息的干扰。

---

## 方法详解

### 模型架构

**HiMem-WAM** 采用**分层世界动作模型**架构：

- **输入**: 语言指令 $\ell$ + 多视角 RGB 观测 $o_t$ + 本体感知 $p_t$ + 外部记忆 $M_t$
- **Backbone**: [[Qwen3-VL]] 作为高级规划器
- **核心模块**: [[Hierarchical Latent Action|分层潜动作]] 框架 + [[Memory-Gated Module|记忆门控模块]]
- **输出**: 动作块 $a_{t:t+K-1}$
- **三阶段**: Stage I（低级潜动作学习）→ Stage II（技能发现与预训练）→ Stage III（记忆激活与微调）

架构由三个组件组成：

- **规划器（Planner）**: 基于 Qwen3-VL，预测当前技能潜变量 $\hat{z}^h_t$ 与边界得分 $\hat{b}_t$
- **执行器（Executor）**: 将技能潜变量展开为低级动作块序列 $\hat{Z}^l_{t:t+K-1}$
- **解码器（Decoder）**: 将潜动作解码为可执行控制信号 $a_{t:t+K-1}$

### 核心模块

#### 模块1：低级潜动作分词器

**设计动机**: 利用 [[Optical Flow|光流]] 作为运动的无监督监督信号，学习紧凑的运动表示

**具体实现**:
- 使用 [[Variational Autoencoder|变分编码器]] $q_\phi(z^l_t | c_t)$ 将上下文（观测、本体感知、语言、光流）编码为连续潜变量
- 通过 [[Optical Flow Reconstruction|光流重建]] 提供自监督信号，辅以动作对齐和 [[KL Divergence|KL 散度]] 正则
- 训练时依赖光流和动作标注，推理时不需要

#### 模块2：高级技能潜变量发现

**设计动机**: 利用 [[Dynamic Chunking|动态分块]] 将低级潜动作序列组织为语义上连贯的可变长度技能片段

**具体实现**:
- 通过归一化余弦距离计算相邻低级潜变量的**异质性得分**，检测技能转换边界 $b^{(s)}_i$
- [[Attention-Based Pooling|注意力池化]] 将可变长度片段聚合为单一技能潜变量
- 多阶段层级化（$H$ 层），逐层将低级 token 压缩为高级 skill token
- 使用一致性约束（$\mathcal{L}_{cons}$）鼓励片段内部表示的连贯性

#### 模块3：边界感知记忆门控

**设计动机**: 在预测的技能边界处写入紧凑任务状态，避免密集历史聚合，实现稀疏可解释的记忆更新

**具体实现**:
- **读门**：[[Attention|注意力]] 检索外部记忆 $M_t$，标量门 $\alpha^r_t$ 控制记忆注入量
- **写门**：标量门 $\alpha^w_t$ 根据预测边界 $\hat{b}_t$ 决定是否更新记忆
- 外部记忆 $M_t$ 存储技能级 token，仅在边界处追加新 token
- **教师强制预热（Teacher-Forced Warmup）**：Stage III 训练初期用发现的真实边界 $\bar{b}_t$ 驱动记忆写入，稳定训练

---

## 关键公式

### 公式1：[[Hierarchical Policy|分层策略分解]]

$$
\pi_\theta(a_{t:t+K-1} | o_t, p_t, \ell, M_t) = \int p_\theta(a_{t:t+K-1} | Z^l_{t:t+K-1}, o_t, p_t)
\cdot p_\theta(Z^l_{t:t+K-1} | z^h_t, o_t, p_t, M_t)
\cdot p_\theta(z^h_t | o_t, p_t, \ell, M_t)\, dZ^l\, dz^h
$$

**含义**: 将动作生成分解为技能选择 → 运动展开 → 动作解码三个条件独立步骤，允许分层监督

**符号说明**:
- $a_{t:t+K-1}$: 动作块，从时刻 $t$ 到 $t+K-1$ 的动作序列
- $Z^l_{t:t+K-1}$: 低级潜动作序列
- $z^h_t$: 当前时刻的高级技能潜变量
- $M_t$: 外部记忆，存储历史任务状态
- $\ell$: 语言指令
- $p_t$: 本体感知状态

### 公式2：[[Variational Tokenizer|变分潜动作分词器]]

$$
q_\phi(z^l_t | c_t) = \mathcal{N}(\mu_t,\, \text{diag}(\sigma^2_t)), \quad z^l_t = \mu_t + \sigma_t \odot \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 将上下文信息 $c_t$（观测、本体感知、语言、光流）编码为高斯分布，采样得到低级潜动作

**符号说明**:
- $c_t$: 上下文输入（观测 + 本体感知 + 语言 + 光流）
- $\mu_t, \sigma^2_t$: 编码器输出的均值和方差
- $\odot$: 逐元素乘法
- $\varepsilon$: 重参数化噪声

### 公式3：[[Tokenizer Training Loss|低级分词器训练损失]]

$$
\mathcal{L}_l = \|\hat{\Phi}_t - \Phi_t\|_1 + \lambda_a \mathbf{1}^{\text{act}}_t \|\hat{a}_t - a_t\|^2_2 + \beta D_{\text{KL}}(q_\phi(z^l_t | c_t) \| \mathcal{N}(0, I))
$$

**含义**: 联合优化光流重建精度、动作对齐（有标注时）和潜空间正则化

**符号说明**:
- $\hat{\Phi}_t$: 重建光流，$\Phi_t$: 真实光流
- $\mathbf{1}^{\text{act}}_t$: 动作标注可用指示符
- $\hat{a}_t$: 预测动作，$a_t$: 真实动作
- $\lambda_a, \beta$: 平衡各项损失的权重系数

### 公式4：[[Hierarchical Chunking|层级动态分块]]

$$
Z^{(s+1)} = \text{Chunk}_s(E_s(Z^{(s)});\, b^{(s)}), \quad Z^h = Z^{(H)}
$$

**含义**: 在 $H$ 个层级中，依据边界 $b^{(s)}$ 逐层将低级潜变量序列压缩为高级技能潜变量

**符号说明**:
- $Z^{(s)}$: 第 $s$ 层的潜变量序列
- $E_s$: 第 $s$ 层的编码器
- $b^{(s)}$: 第 $s$ 层的边界指示向量
- $H$: 总层级数

### 公式5：[[Boundary Scoring|边界得分计算]]

$$
r^{(s)}_i = \begin{cases} 1 & i = 1 \\ \frac{1}{2}\left(1 - (\hat{q}^{(s)}_{i-1})^\top \hat{k}^{(s)}_i\right) & i > 1 \end{cases}, \quad
b^{(s)}_i = \begin{cases} 1 & i = 1 \\ \mathbf{1}[r^{(s)}_i \geq \delta_s] & i > 1 \end{cases}
$$

**含义**: 通过相邻 token 的归一化余弦距离衡量相似度，超过阈值 $\delta_s$ 则标记为技能转换边界

**符号说明**:
- $r^{(s)}_i$: 第 $i$ 个位置的异质性得分
- $\hat{q}^{(s)}_{i-1}, \hat{k}^{(s)}_i$: 相邻位置的归一化查询/键向量
- $\delta_s$: 第 $s$ 层的边界检测阈值

### 公式6：[[Attention-Based Pooling|注意力片段池化]]

$$
\alpha^{(s)}_{j,i} = \frac{\exp\left((w_s)^\top h^{(s)}_i\right)}{\sum_{k \in \mathcal{I}_j} \exp\left((w_s)^\top h^{(s)}_k\right)}, \quad
z^{(s+1)}_j = \sum_{i \in \mathcal{I}_j} \alpha^{(s)}_{j,i} h^{(s)}_i
$$

**含义**: 用可学习权重向量 $w_s$ 对每个片段内的 token 进行加权池化，生成下一层技能 token

**符号说明**:
- $\mathcal{I}_j$: 第 $j$ 个技能片段内的时间步集合
- $h^{(s)}_i$: 第 $s$ 层第 $i$ 个 token 的隐状态
- $w_s$: 可学习的池化权重向量

### 公式7：[[Memory Read Gate|记忆读门]]

$$
c^m_t = \text{Attn}(W_q x_t,\, W_k M_t,\, W_v M_t), \quad
x̃_t = x_t + \alpha^r_t W_m c^m_t, \quad
\alpha^r_t = \sigma(G_r(x_t, c^m_t))
$$

**含义**: 注意力检索外部记忆，标量门 $\alpha^r_t$ 控制记忆上下文注入量，自适应调整历史信息的影响

**符号说明**:
- $c^m_t$: 从外部记忆检索的上下文向量
- $\alpha^r_t \in [0,1]$: 可学习读门标量
- $W_q, W_k, W_v, W_m$: 可学习权重矩阵
- $G_r$: 读门评分网络

### 公式8：[[Memory Write Gate|记忆写门与更新]]

$$
\alpha^w_t = \sigma(G_w(\tilde{x}_t,\, \hat{z}^h_t,\, \hat{b}_t)), \quad
M_{t+1} = \begin{cases} U_\psi(M_t, \gamma_t) & \text{if } \alpha^w_t > \eta \\ M_t & \text{otherwise} \end{cases}
$$

**含义**: 写门基于增强后的特征、预测技能潜变量和边界得分决定是否将当前状态写入记忆，实现事件驱动的稀疏更新

**符号说明**:
- $\alpha^w_t \in [0,1]$: 可学习写门标量
- $\eta$: 写入阈值
- $U_\psi$: 记忆更新函数
- $\gamma_t$: 当前写入的记忆 token

### 公式9：[[Gate Training Loss|门控训练损失]]

$$
\mathcal{L}_{\text{gate}} = \text{BCE}(\alpha^w_t, \bar{b}_t) + \lambda_r \|\alpha^r_t\|_1 + \lambda_w \|\alpha^w_t\|_1
$$

**含义**: 以边界标签监督写门，同时用 L1 稀疏正则鼓励读/写门仅在必要时激活

**符号说明**:
- $\bar{b}_t$: 真实边界标签（由 Stage II 发现的边界映射回时间步得到）
- $\lambda_r, \lambda_w$: 稀疏正则权重

### 公式10：[[Stage II Pretraining Loss|Stage II 潜动作预训练损失]]

$$
\mathcal{L}_{\text{latent}} = \lambda_h \text{MSE}(\hat{z}^h_t, \bar{z}^h_t) + \lambda_l \text{MSE}(\hat{Z}^l_{t:t+K-1}, Z^l_{t:t+K-1}) + \lambda_b \text{BCE}(\hat{b}_t, \bar{b}_t)
$$

**含义**: 联合监督技能预测（高级）、潜动作重建（低级）和边界检测，不使用记忆

**符号说明**:
- $\bar{z}^h_t$: Stage I 发现的目标技能潜变量
- $\lambda_h, \lambda_l, \lambda_b$: 各项权重

### 公式11：[[Stage III Fine-tuning Loss|Stage III 端到端微调损失]]

$$
\mathcal{L}_{\text{ft}} = \mathcal{L}_{\text{act}} + \alpha_h \mathcal{L}_{\text{plan}} + \alpha_l \mathcal{L}_{\text{exec}} + \alpha_b \mathcal{L}_{\text{bd}} + \alpha_m \mathcal{L}_{\text{gate}}
$$

**含义**: 端到端整合动作预测、规划、执行、边界检测和记忆门控的联合损失

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 动作预测损失
- $\mathcal{L}_{\text{plan}}, \mathcal{L}_{\text{exec}}$: 规划器/执行器辅助损失
- $\alpha_h, \alpha_l, \alpha_b, \alpha_m$: 各项权重系数

### 公式12：[[Skill Discovery Loss|技能发现损失]]

$$
\mathcal{L}_{\text{skill}} = \mathcal{L}_{\text{next}} + \lambda_m \mathcal{L}_{\text{motion}} + \lambda_r \mathcal{L}_{\text{ratio}} + \lambda_c \mathcal{L}_{\text{cons}}
$$

**含义**: 联合优化下一潜变量预测、运动保真度、边界比例控制和片段内一致性

**符号说明**:
- $\mathcal{L}_{\text{next}}$: 下一潜动作预测损失
- $\mathcal{L}_{\text{motion}}$: 运动信息保留损失
- $\mathcal{L}_{\text{ratio}}$: 控制边界频率的比例损失
- $\mathcal{L}_{\text{cons}}$: 片段内一致性正则

### 公式13：[[Teacher-Forced Memory Warmup|教师强制记忆预热]]

$$
M^{\text{teach}}_{t+1} = \begin{cases} U_\psi(M^{\text{teach}}_t, \gamma^{\text{teach}}_t) & \text{if } \bar{b}_t = 1 \\ M^{\text{teach}}_t & \text{otherwise} \end{cases}, \quad
\gamma^{\text{teach}}_t = \Gamma_\psi(x_t, \bar{z}^h_t, \text{Pool}(Z^l_{t:t+K-1}))
$$

**含义**: Stage III 训练初期使用发现的真实边界驱动记忆写入，稳定记忆模块的初始训练

**符号说明**:
- $\bar{b}_t$: 真实边界标签
- $\Gamma_\psi$: 记忆 token 生成网络

---

## 关键图表

### Figure 1: 系统总览

![Figure 1 - HiMem-WAM Overview](https://arxiv.org/html/2606.10363v1/pictures/overview.png)

**说明**: HiMem-WAM 的三阶段训练框架总览。Stage I 从演示中提取低级动作 token 和高级技能潜变量；Stage II 从视频和语言输入学习预测潜动作；Stage III 引入门控记忆模块实现历史感知的动作预测。底部展示真实世界与仿真评估结果。

### Figure 2: 从 WAM 到 HiMem-WAM

![Figure 2 - WAM vs HiMem-WAM](https://arxiv.org/html/2606.10363v1/pictures/wamvsours.png)

**说明**: 对比 [[WAM]] 与 HiMem-WAM 的架构差异。HiMem-WAM 通过引入记忆专家模块，使动作预测同时条件化于当前观测和任务历史，克服了原始 WAM 在长时域任务上的局限。

### Figure 3: 真实世界评估结果

![Figure 3 - Real-world Evaluation](https://arxiv.org/html/2606.10363v1/pictures/real_world.png)

**说明**: 在 10 个真实世界任务上的评估结果。(a)-(c) 分别报告三个难度级别（简单/中等/困难）的成功率；(d) 展示泛化评估的扰动变体；(e) 展示使用的双臂硬件平台。HiMem-WAM 在困难任务上比 π₀.₅ 提升约 +25%（标准设置）。

### Figure 4: 10 个真实世界任务可视化

![[HiMem-WAM_fig4.png|600]]

**说明**: 三个难度级别的 10 个真实操作任务。第一行（简单）：叠碗、挂杯、将水果放入篮子、按按钮；第二行（中等）：叠三个碗、折叠毛巾、放置盘子、按两个按钮；第三行（困难）：放置两个盘子、制作早餐。

### Figure 5（附录）: RMBench 任务可视化

![Figure 5 - RMBench Tasks](https://arxiv.org/html/2606.10363v1/pictures/appendix_RMBench.png)

**说明**: RMBench 任务的执行轨迹可视化，以及 DPFlow 光流预测的可视化展示。

### Figure 6（附录）: LIBERO-PLUS 扰动可视化

![Figure 6 - LIBERO-PLUS](https://arxiv.org/html/2606.10363v1/pictures/libero_plus_4suites.png)

**说明**: LIBERO-PLUS 基准中的 7 种扰动类型可视化，包括相机扰动、初始状态扰动、语言扰动、光照变化、背景变化、观测噪声和布局变化，用于评估模型鲁棒性。

---

### Table 1: RMBench 结果（记忆依赖任务）

| 任务 | 类型 | TMC | DP | ACT | π₀.₅ | X-VLA | **HiMem-WAM** |
|------|------|-----|----|-----|------|-------|---------------|
| Observe and Pick Up | M(1) | 1% | 1% | 9% | 9% | 28% | — |
| Rearrange Blocks | M(1) | 0% | 29% | 13% | 13% | 33% | — |
| Put Back Block | M(1) | 0% | 0% | 11% | 18% | 32% | — |
| Swap Blocks | M(1) | 11% | 2% | 24% | 16% | 38% | — |
| Swap T | M(1) | 2% | 2% | 15% | 3% | 27% | — |
| **Average M(1)** | | 2.8% | 6.8% | 14.4% | 11.8% | — | **31.6%** |
| Battery Try | M(n) | 10% | 19% | 16% | 26% | 28% | — |
| Blocks Ranking Try | M(n) | 10% | 0% | 6% | 1% | 24% | — |
| Cover Blocks | M(n) | 0% | 0% | 2% | 0% | 19% | — |
| Press Button | M(n) | 0% | 0% | 1% | 2% | 8% | — |
| **Average M(n)** | | 5.0% | 4.8% | 6.3% | 7.3% | — | **19.8%** |
| **Total Average** | | 3.8% | 5.9% | 10.8% | 9.8% | — | **26.3%** |

**说明**: HiMem-WAM 在单次记忆任务 M(1) 上以 31.6% 的成功率大幅领先所有基线；在多次试探任务 M(n) 上也达到 19.8%，显示记忆机制的有效性。

### Table 2: LIBERO 基准结果（仿真）

| 方法 | Spatial | Object | Goal | Long | **Avg.** |
|------|---------|--------|------|------|---------|
| **VLA 类** | | | | | |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| NORA-1.5 | 97.3 | 96.4 | 94.5 | 89.6 | 94.5 |
| AtomVLA | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |
| **WAM 类** | | | | | |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **Ours (w/o Stage II)** | 96.0 | 99.6 | 97.1 | 93.8 | 96.6 |
| **HiMem-WAM** | **98.2** | **99.8** | **98.4** | **94.5** | **97.7** |

**说明**: HiMem-WAM 以 97.7% 的平均成功率与 Fast-WAM（97.6%）持平并略优，验证了分层潜动作在标准任务上不损失性能。

### Table 3: LIBERO-PLUS 零样本鲁棒性结果（7 种扰动）

| 方法 | 相机 | 初始状态 | 语言 | 光照 | 背景 | 噪声 | 布局 | **平均** |
|------|------|---------|------|------|------|------|------|---------|
| **VLA 类** | | | | | | | | |
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 16.3 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 71.4 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 56.1 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 45.8 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 70.2 |
| **WAM 类** | | | | | | | | |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.6 |
| HoloBrain-0 | 65.5 | 58.2 | 78.7 | 88.1 | 90.3 | 66.9 | 79.5 | 75.3 |
| **Ours (w/o Stage II)** | 77.2 | 37.9 | 71.7 | 91.1 | 83.6 | 73.2 | 70.7 | 72.2 |
| **HiMem-WAM** | **78.2** | **38.1** | **76.6** | **92.2** | **91.0** | **80.7** | **74.9** | **76.0** |

**说明**: Stage II 潜动作预训练贡献 +3.8% 鲁棒性提升（76.0% vs 72.2%），说明学习到的运动先验能捕捉任务完成模式而非视觉伪影，在各类扰动下均更鲁棒。

### Table 4: 真实世界评估——动作表示对比（含/不含 Stage II）

| 任务难度 | 关节位置 (w/o II) | 关节位置 (w/ II) | 末端执行器 (w/o II) | 末端执行器 (w/ II) |
|---------|-----------------|-----------------|-------------------|-------------------|
| 简单 | 100.0% | 100.0% | 90.0% | 100.0% |
| 中等 | 80.0% | **82.5%** | 67.5% | **75.0%** |
| 困难 | 15.0% | **35.0%** | 10.0% | **30.0%** |

**说明**: Stage II 潜动作预训练在困难任务上带来约 +20% 的提升（关节位置和末端执行器均如此），验证了分层表示学习对复杂任务的关键作用。

### Table 5: 训练/推理信号使用情况

| 信号 | Stage I | Stage II | Stage III | 推理 |
|------|---------|----------|-----------|------|
| RGB 观测 | ✓ | ✓ | ✓ | ✓ |
| 本体感知 | ✓ | ✓ | ✓ | ✓ |
| 语言指令 | ✓ | ✓ | ✓ | ✓ |
| 光流（监督） | ✓ | — | — | — |
| 动作标注 | 可选 | — | ✓ | — |
| 低级潜变量 $Z^l$ | 输出 | 监督 | 辅助监督 | 预测 |
| 高级技能 $Z^h$ | — | 监督 | 辅助监督 | 预测 |
| 边界标签 | — | 监督 | 门控监督 | 预测 |
| 外部记忆 | — | 禁用 | ✓ | ✓ |

**说明**: 丰富的训练时监督信号（光流、未来帧）引导学习；推理时完全因果，无需光流或视频生成。

---

## 实验

### 数据集

| 数据集/基准 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| LIBERO | 4 套任务 | 仿真操作，标准基准 | 标准性能测试 |
| LIBERO-PLUS | 7 种扰动 | 相机/光照/语言等扰动 | 零样本鲁棒性评估 |
| RMBench | M(1)+M(n) 任务 | 记忆依赖操作 | 记忆能力评估 |
| 真实世界 | 10 个任务，3 难度 | 双臂平台，标准+泛化设置 | 真实部署验证 |

### 实现细节

- **规划器 Backbone**: Qwen3-VL（大型视觉语言模型）
- **动作表示**: 支持关节位置和末端执行器位姿两种
- **硬件**: 双臂机器人平台
- **训练**: 三阶段渐进式训练，Stage III 使用教师强制记忆预热稳定训练

### 可视化结果

真实世界评估表明，HiMem-WAM 在"制作早餐"等需要长时域规划的复杂任务上表现最为突出，记忆门控能有效追踪任务进度（如哪些食材已经放置）。在 LIBERO-PLUS 中，相机扰动（78.2%）和光照变化（92.2%）的鲁棒性优于所有 WAM 基线。

---

## 批判性思考

### 优点
1. **因果推理**: 推理时无需光流或未来帧预测，完全因果，适合实时部署
2. **稀疏记忆更新**: 边界触发的写门避免了密集历史聚合，记忆更新可解释，与任务结构对齐
3. **分层监督解耦**: 三阶段训练使复杂的分层表示学习在各阶段都有明确监督，降低了端到端训练难度

### 局限性
1. **计算开销大**: 光流提取和三阶段训练引入了显著的计算成本，训练流水线工程复杂
2. **记忆性能依赖边界质量**: M(n) 类型任务（多次试探）的性能明显弱于 M(1)（31.6% vs 19.8%），说明记忆机制对于需要多次尝试才能成功的任务效果有限
3. **评估平台有限**: 真实世界评估仅限于一种双臂平台的 10 个任务，泛化性有待在更多平台验证

### 潜在改进方向
1. **改进 M(n) 性能**: 针对需要多次试探记忆的任务设计专门的记忆检索策略（如置信度加权的多条目记忆）
2. **轻量化训练**: 探索减少对光流监督依赖的方法，如自监督运动表示或直接从视频学习

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（三阶段损失函数、超参数详细描述）
- [x] 数据集可获取（LIBERO、RMBench 均为公开基准）

---

## 关联笔记

### 基于
- [[WAM]]: World Action Model 基础框架，HiMem-WAM 直接在其上扩展
- [[Fast-WAM]]: 主要对比基线，WAM 的高效版本
- [[Qwen3-VL]]: 用作高级规划器的视觉语言模型骨干

### 对比
- [[WorldVLA]]: WAM 类方法，LIBERO-PLUS 上仅 25.6% vs HiMem-WAM 76.0%
- [[π0.5]]: 真实世界对比基线，困难任务上 HiMem-WAM 提升 +25%
- [[HoloBrain-0]]: LIBERO-PLUS 上另一 WAM 类 SOTA（75.3%）

### 方法相关
- [[Hierarchical Policy|分层策略]]: 核心方法论框架
- [[Latent Action|潜动作]]: 低级运动表示的基础概念
- [[Memory-Gated Module|记忆门控]]: 关键技术创新
- [[Dynamic Chunking|动态分块]]: 技能边界发现的核心机制
- [[Optical Flow|光流]]: Stage I 训练的监督信号

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评估基准
- [[RMBench]]: 记忆依赖任务专用基准

---

## 速查卡片

> [!summary] HiMem-WAM (arXiv 2026)
> - **核心**: 分层潜动作 + 边界触发记忆门控，解决长时域操作中的历史状态追踪问题
> - **方法**: 低级运动原语 + 高级技能潜变量 + 外部记忆，三阶段训练，推理完全因果
> - **结果**: LIBERO 97.7%、LIBERO-PLUS 76.0%（鲁棒性 SOTA）、RMBench 26.3%、真实困难任务超 π₀.₅ +25%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-10*
