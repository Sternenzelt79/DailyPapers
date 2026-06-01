---
title: "HARP-VLA: Human-Robot Aligned Representation Learning for Vision-Language-Action Model"
method_name: "HARP-VLA"
authors: [Xiang Zhu, Puzhen Yuan, Yichen Liu, Jianyu Chen]
year: 2026
venue: arXiv
tags: [cross-embodiment, vision-language-action, representation-alignment, latent-action-model, human-robot-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.31234
created: 2026-06-01
---

# 论文笔记：HARP-VLA: Human-Robot Aligned Representation Learning for Vision-Language-Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（匿名提交） |
| 日期 | May 2026 |
| 项目主页 | [GitHub](https://github.com/anonymity35/HARP-VLA) |
| 对比基线 | [[UniVLA]], [[Pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.31234) / [Code](https://github.com/anonymity35/HARP-VLA) |

---

## 一句话总结

> HARP-VLA 通过联合对齐人类与机器人的视觉表征和潜在动作，将大规模人类视频有效用于 VLA 预训练，在 CALVIN ABC→D 上达到 4.481 平均长度，并比最强基线提升 7.1% 真实世界成功率。

---

## 核心贡献

1. **三阶段人机对齐框架**: 利用少量配对人机演示作为跨具身桥梁，结合大规模无配对视频进行动态学习，联合对齐视觉表征和潜在动作
2. **Source-Relative Pair-Discriminative (SRPD) 对齐损失**: 设计了源相对损失（SR）和对对识别损失（PD）的组合，比标准 L2/HR 对齐损失效果显著更好
3. **操作中心辅助提示**: 引入物体关键点轨迹和腕部/末端执行器轨迹作为辅助监督信号，增强跨具身动态对齐

---

## 问题背景

### 要解决的问题

从大规模人类视频中学习[[Vision-Language-Action Model|VLA 模型]]具有巨大潜力，但人类和机器人之间存在严重的**跨具身差异**（cross-embodiment discrepancy），主要体现在两个维度：
1. **动作执行差距**（Action Execution Gap）：人手运动轨迹无法直接转化为可执行的机器人动作
2. **视觉表征差距**（Visual Representation Gap）：相似的操作动态在特征流形中被分别编码，导致人类视频的监督信号无法有效迁移

### 现有方法的局限

- **LAPA / UniVLA** 等[[潜在动作模型|Latent Action Model]]方法将不同具身的动作映射到统一的离散潜在空间，但视觉表征仍然分离
- **RoVi-Aug / EgoMimic** 等方法尝试对齐表征，但仅使用简单的 L2 损失或 HR 损失，未考虑跨具身对的区分性
- 缺乏同时对齐视觉表征**和**潜在动作的统一框架

### 本文的动机

少量配对的人机演示（paired demonstrations）能够作为"跨具身桥梁"——通过这些桥梁同时对齐视觉编码器和潜在动作模型，再结合大量无标注的无配对视频提供动态监督，可以实现高效的跨具身知识迁移。

---

## 方法详解

### 模型架构

HARP-VLA 采用**三阶段训练**架构：

- **Stage 1**：联合视觉与潜在动作对齐（Joint Visual & Latent-Action Alignment）
- **Stage 2**：使用对齐后的 LAM 标注帧，进行 VLA 预训练（Pretraining with Latent Actions）
- **Stage 3**：用真实动作微调，映射潜在动作嵌入到可执行动作（Finetuning with Real Actions）

**训练数据**：$\mathcal{D} = \mathcal{D}_p \cup \mathcal{D}_u$（配对 + 无配对演示）

**共享任务提示**：
- 物体关键点轨迹 $K_x$（通过关键点检测提取）
- 腕部/末端执行器轨迹 $E_x$
- 使用[[动态时间规整|Dynamic Time Warping (DTW)]]进行时序对齐

### 核心模块

#### 模块 1：具身感知视觉编码器（Embodiment-Aware Visual Encoder）

**设计动机**: 人类与机器人使用不同的视觉编码路径，避免对已对齐的人类表征进行干扰

**具体实现**:
- 人类帧使用**冻结**的原始编码器 $F$（如 [[SigLIP]] 等预训练 ViT）
- 机器人帧使用可训练的适配编码器 $T_\theta$（在原始编码器基础上微调）

$$
\Phi_\theta(X, e_X) = \begin{cases} F(X) & \text{if } e_X = h \\ T_\theta(X) & \text{if } e_X = r \end{cases}
$$

#### 模块 2：潜在动作模型（Latent Action Model, LAM）

**设计动机**: 通过自预测（self-prediction）和交叉预测（cross-prediction）同时建立同具身动态一致性和跨具身动作对齐

**潜在动作推断**：

$$
a_x^t = E_\theta(z_x^t, z_x^{t+\Delta t}, l_X)
$$

**码本量化**：

$$
q_x^t = Q_\theta(a_x^t)
$$

**解码器预测**：

$$
\hat{Y}_x^t = D_\theta(\tilde{z}_x^t, q_x^t, l_X)
$$

其中 $\tilde{z}_x^t$ 在**自预测**时为自身帧特征，在**交叉预测**（配对数据）时替换为对方具身的帧特征。

#### 模块 3：SRPD 对齐损失（Source-Relative Pair-Discriminative Loss）

这是 HARP 的核心创新。对齐损失由两部分组成：

**Source-Relative（SR）损失**：以初始配对距离作为基准，鼓励减小配对人机特征间的距离

**Pair-Discriminative（PD）损失**：要求配对的人机特征距离小于与其他配对样本的平均距离，强化对对区分性

---

## 关键公式

### 公式 1：[[潜在动作模型|潜在动作预测损失]]

$$
\mathcal{L}_{lam} = \mathbb{E}_t\left[\|\hat{Y}_x^t - Y_x^t\|^2\right] + \mathcal{L}_{vq}
$$

**含义**: 重建预测帧特征的 MSE 损失加上[[向量量化|VQ（向量量化）]]码本损失，监督潜在动作模型学习帧间动态

**符号说明**:
- $\hat{Y}_x^t$：解码器预测的目标帧特征
- $Y_x^t$：真实目标帧特征（$z_x^{t+\Delta t}$）
- $\mathcal{L}_{vq}$：VQ 码本 commitment loss

### 公式 2：[[余弦距离]]

$$
d(u, v) = 1 - \cos(u, v)
$$

**含义**: 衡量两个特征向量之间的相似度，距离越小表示特征越接近

**符号说明**:
- $u, v$：特征向量
- $\cos(u, v)$：余弦相似度

### 公式 3：[[Source-Relative 对齐损失|Source-Relative 损失]]

$$
\mathcal{L}_{SR} = \mathbb{E}_{(H,R)\sim\mathcal{D}_p}\left[m_s + d(f^R, f^H) - d(f^{R_0}, f^H)\right]_+
$$

**含义**: 以初始（未适配）机器人特征 $f^{R_0}$ 与人类特征的距离作为动态基准，鼓励适配后的机器人特征与对应人类特征更接近

**符号说明**:
- $f^R$：适配后机器人帧特征
- $f^H$：人类帧特征
- $f^{R_0}$：未适配（原始）机器人帧特征（动态基准）
- $m_s$：margin 超参数
- $[\cdot]_+$：hinge 函数（ReLU）

### 公式 4：[[Pair-Discriminative 对齐损失|Pair-Discriminative 损失]]

$$
\mathcal{L}_{PD} = \mathbb{E}_{(H,R)\sim\mathcal{D}_p} \sum_{\alpha\in\{R\to H,\, H\to R\}} \lambda_\alpha \left[m_t + d(f^R, f^H) - \bar{d}^\alpha\right]_+
$$

**含义**: 要求配对人机特征距离小于该样本与批次内其他样本的平均距离，增强配对的区分性

**符号说明**:
- $\alpha$：对比方向（机器人→人类 或 人类→机器人）
- $\lambda_\alpha$：各方向权重
- $m_t$：margin 超参数
- $\bar{d}^\alpha$：当前样本与批次内其他配对样本的平均距离

### 公式 5：[[阶段一总损失|Stage 1 总损失]]

$$
\mathcal{L}_{stage1} = \mathcal{L}_{lam} + \lambda_{aux}\mathcal{L}_{aux} + \lambda_{align}\mathcal{L}_{align}
$$

其中辅助损失 $\mathcal{L}_{aux} = \lambda_K \mathcal{L}_K + \lambda_E \mathcal{L}_E$（关键点 + 末端执行器轨迹预测）

对齐损失 $\mathcal{L}_{align} = \mathcal{L}_{SR} + \mathcal{L}_{PD}$

**含义**: Stage 1 的联合训练目标，同时优化潜在动作预测、操作中心辅助监督和跨具身对齐

### 公式 6：[[VLA 微调动作损失|Stage 3 动作回归损失]]

$$
\mathcal{L}_{finetune} = \mathcal{L}_1(a_{pred}, a_{gt})
$$

**含义**: Stage 3 中轻量动作头将潜在动作嵌入映射到真实可执行动作，使用 L1 损失监督，backbone 通过 [[LoRA]] 微调

---

## 关键图表

### Figure 1：HARP-VLA 动机与框架概览

![Figure 1](https://arxiv.org/html/2605.31234v1/x1.png)

**说明**: 上方展示现有 VLA 预训练的问题——人类和机器人演示被编码到分离的视觉表征空间，存在较大的动作差距。下方展示 HARP 的解决方案：以少量配对演示为桥梁，联合对齐视觉表征和[[潜在动作模型|潜在动作]]，同时利用大规模无配对视频提供动态监督。

### Figure 2：Stage 1 联合视觉与潜在动作对齐

![Figure 2](https://arxiv.org/html/2605.31234v1/x2.png)

**说明**: 左图展示人→机器人交叉预测示例；右图对比自预测（同具身）和交叉预测（跨具身）两种训练模式。[[动态时间规整|DTW]] 用于将配对的人机序列进行时序对齐后再进行交叉预测。

### Figure 3：Stage 2 预训练 + Stage 3 微调

![Figure 3](https://arxiv.org/html/2605.31234v1/x3.png)

**说明**: Stage 2 使用对齐后的 LAM 为帧序列生成统一的潜在动作标签 $q_{xt}$，以交叉熵损失预训练 VLA backbone；Stage 3 引入轻量动作头，通过 L1 损失将潜在动作嵌入映射到真实关节动作，仅最小化微调代价。

### Figure 4：UMAP 视觉对齐可视化

![Figure 4](https://arxiv.org/html/2605.31234v1/x4.png)

**说明**: 左图展示视觉表征在对齐前（蓝色/橙色点分离）和对齐后（混合聚集）的 [[UMAP]] 分布变化；右图展示不带/带 HARP-LAM 对齐的潜在动作分布，验证了人机表征在特征流形上的有效融合。

### Figure 5：配对人机余弦距离（Box Plot）

![Figure 5](https://arxiv.org/html/2605.31234v1/x5.png)

**说明**: 箱线图比较不同方法下配对人机特征的余弦距离分布。HARP-SRPD 的中值距离最小，且分布最集中，定量验证了 SRPD 损失在减小跨具身特征差距方面的优越性。

### Figure A1：配对数据策划流程（附录）

![Figure A1](https://arxiv.org/html/2605.31234v1/x6.png)

**说明**: 蓝色点为物体关键点轨迹，红色点为腕部位置，红色箭头展示基于 L2 距离的 DTW 时序对齐过程。该流程将人类与机器人演示对齐为配对样本，是 HARP 框架的数据基础。

### Table 1：跨具身检索性能（Cross-Embodiment Retrieval R@1）

| Method | H2R R@1 ↑ | R2H R@1 ↑ | Avg. R@1 ↑ |
|--------|:---------:|:---------:|:---------:|
| Unadapted | 44.09 | 43.01 | 43.55 |
| HR | 45.16 | 45.16 | 45.16 |
| HARP-HR | 46.24 | 60.22 | 53.23 |
| HARP-L2 | 70.97 | 52.69 | 61.83 |
| HARP-SR | 84.95 | 64.52 | 74.74 |
| **HARP-SRPD** | **87.10** | **69.89** | **78.50** |

**说明**: HARP-SRPD 在 H2R 和 R2H 两个方向均取得最高 R@1，平均分比无适配基线提升 34.95 个点，验证了 SRPD 损失的有效性。

### Table 2：RLBench 成功率

| Method | Avg. ↑ |
|--------|:------:|
| Unadapted | 37.56 |
| HR | 39.70 |
| HARP-SR | 43.41 |
| **HARP-SRPD** | **46.59** |

**说明**: 在 RLBench 仿真基准上，HARP-SRPD 比无适配基线高 9.03 个点（24%相对提升）。

### Table 3：真实世界操作成功率

| Model | Pick ↑ | Push ↑ | Press ↑ | Flip ↑ | Avg. ↑ |
|-------|:------:|:------:|:-------:|:------:|:------:|
| π₀ | 58.3 | 75.0 | 56.7 | 35.0 | 56.3 |
| π₀.₅ | 71.7 | 83.3 | 68.3 | 53.3 | 69.2 |
| OpenVLA | 0.0 | 23.3 | 18.3 | 0.0 | 10.4 |
| UniVLA | 38.3 | 61.7 | 31.7 | 21.7 | 38.4 |
| **HARP-VLA** | **76.7** | **81.7** | **85.0** | **61.7** | **76.3** |

**说明**: HARP-VLA 超过最强基线 [[Pi0.5]] 7.1 个百分点（76.3% vs 69.2%），在最难的 Press 和 Flip 任务上优势尤为显著。

### Table 4：CALVIN ABC→D 基准

| Model | Task1 ↑ | Task2 ↑ | Task3 ↑ | Task4 ↑ | Task5 ↑ | Avg. Len. ↑ |
|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-----------:|
| π₀ | 92.3 | 82.4 | 72.1 | 62.2 | 53.7 | 3.627 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.270 |
| UniVLA | 95.4 | 85.5 | 75.4 | 66.9 | 56.5 | 3.800 |
| **HARP-VLA** | **99.8** | **96.7** | **91.3** | **84.4** | **75.9** | **4.481** |

**说明**: HARP-VLA 在 CALVIN ABC→D 上以 4.481 平均任务长度大幅超越所有基线，5 个连续任务的成功率均排名第一，Task5 成功率从 56.5% 提升到 75.9%（+19.4pp）。

### Table A1：训练数据汇总

| 数据集 | 具身 | 数据类型 | 规模 | Stage1 | Stage2 | Stage3 |
|--------|------|----------|------|:------:|:------:|:------:|
| OpenEgo | Human | Unpaired | 36.4M | ✓ | ✓ | ✗ |
| Bridge-V2 | Robot | Unpaired | 8.6M | ✓ | ✓ | ✗ |
| RH20T | Human-Robot | Paired | 8.16M | ✓ | ✓ | ✗ |
| Human2Robot | Human-Robot | Paired | 9.9M | ✓ | ✓ | ✗ |
| CALVIN | Robot/Sim | Real-action | 1.1M | ✗ | ✗ | ✓ |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OpenEgo | 36.4M 帧 | 人类第一视角，无配对 | Stage 1-2 预训练 |
| Bridge-V2 | 8.6M 帧 | 机器人操作，无配对 | Stage 1-2 预训练 |
| RH20T | 8.16M 帧 | 人机同步配对数据 | Stage 1-2 对齐 |
| Human2Robot | 9.9M 帧 | 人机配对，多任务 | Stage 1-2 对齐 |
| CALVIN | 1.1M 帧 | 仿真桌面操作 | Stage 3 微调 |

### 评估基准

- **跨具身检索**：H2R 和 R2H 方向的 Recall@1
- **RLBench**：多任务操作仿真成功率
- **CALVIN ABC→D**：零样本泛化，5 个连续任务平均长度
- **真实世界**：Pick / Push / Press / Flip 四项操作任务

### 实现细节

- **视觉编码器**：[[SigLIP]]（冻结人类编码路径，机器人路径可训练 $T_\theta$）
- **VLA Backbone**：大型预训练语言-视觉模型（具体未披露，匿名）
- **微调方法**：[[LoRA]]（backbone）+ 轻量动作头（Stage 3）
- **优化**：三阶段分别训练，Stage 1 使用[[动态时间规整|DTW]]进行配对对齐

### 消融研究关键发现

1. **损失函数消融**：SRPD > SR > L2 > HR，每个组件均有贡献
2. **视觉编码器冻结**：Stage 2 预训练时冻结视觉编码器相比不冻结在 CALVIN 上提升 0.231 平均长度

---

## 批判性思考

### 优点

1. **方法扎实**：SRPD 损失设计合理，SR 避免了固定 margin 问题（用初始距离作为动态基准），PD 增加配对识别性
2. **多维验证**：从表征对齐（检索 + UMAP）到策略性能（仿真 + 真实），验证链条完整
3. **数据效率**：只需少量配对数据作为桥梁，大量利用廉价无配对数据

### 局限性

1. **配对数据依赖**：性能依赖配对数据的多样性和时序对齐/共享提示的鲁棒性，对噪声敏感
2. **评估范围受限**：仅在单臂桌面操作上验证，未涉及双臂、移动底座或更复杂场景
3. **匿名代码**：代码为匿名仓库，可复现性待公开正式版

### 潜在改进方向

1. 扩展到更多样化的人类视频（室外、非结构化环境）
2. 探索对噪声共享提示的鲁棒性
3. 扩展到双臂操作和更多机器人形态

### 可复现性评估

- [ ] 代码开源（匿名仓库，待正式开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（三阶段流程描述较详细）
- [x] 数据集可获取（使用公开数据集 BridgeV2, RH20T, OpenEgo 等）

---

## 关联笔记

### 基于

- [[UniVLA]]: 潜在动作模型预训练框架，HARP 在此基础上增加跨具身对齐
- [[LAPA]]: 早期潜在动作预训练方法，提出离散化潜在动作空间
- [[跨具身学习]]: 跨具身迁移的核心问题

### 对比

- [[Pi0.5]]: 真实世界最强基线，HARP-VLA 超越 7.1%
- [[OpenVLA]]: 开源 VLA 基线，在本文场景中表现较差
- [[UniVLA]]: CALVIN 上的最强竞争对手

### 方法相关

- [[潜在动作模型]]: HARP Stage 1 的核心模块
- [[向量量化]]: 潜在动作的离散化方法
- [[LoRA]]: Stage 3 微调策略
- [[动态时间规整]]: 配对数据时序对齐
- [[UMAP]]: 表征可视化分析

### 硬件/数据相关

- [[CALVIN]]: 主要仿真评估基准
- [[SigLIP]]: 视觉编码器骨干

---

## 速查卡片

> [!summary] HARP-VLA (arXiv 2026)
> - **核心**: 联合对齐人类与机器人的视觉表征和潜在动作，使大规模人类视频能有效用于 VLA 预训练
> - **方法**: 三阶段框架 + SRPD 对齐损失 + 具身感知编码器 + 操作中心辅助提示
> - **结果**: CALVIN ABC→D 4.481 avg. len，真实世界 76.3%（+7.1% vs 最强基线）
> - **代码**: [GitHub](https://github.com/anonymity35/HARP-VLA)

---

*笔记创建时间: 2026-06-01*
