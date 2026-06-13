---
title: "AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding"
method_name: "AffordanceVLA"
authors: [Qize Yu, Jiadi You, Yuran Wang, Jiaqi Liang, Bowen Ping, Yang Tian, Yue Chen, Minghong Cai, Zeying Gong, Ruihai Wu, Yinchuan Li, Junwei Liang, Yingcong Chen]
year: 2026
venue: arXiv
tags: [vla, affordance, robot-manipulation, mixture-of-experts, imitation-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.06155
created: 2026-06-05
---

# 论文笔记：AffordanceVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University / HKUST (Guangzhou) / CUHK / Knowin AI |
| 日期 | April 2026 |
| 项目主页 | [AffordanceVLA](https://skywalker-yqz.github.io/AffordanceVLA/) |
| 对比基线 | [[Pi0]] / [[F1-VLA]] / [[Seer]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06155) / [Code](https://github.com/Skywalker-yqz/AffordanceVLA) |

---

## 一句话总结

> AffordanceVLA 将结构化 [[Affordance]]（可供性）预测作为中间表示，通过 Which2Act / Where2Act / How2Act 三模块解耦感知理解与动作生成，在 LIBERO 和 CALVIN 等基准上大幅超越现有 VLA 方法。

---

## 核心贡献

1. **结构化 Affordance 中间表示**: 提出将 affordance 预测分解为"选什么物体 → 在哪里交互 → 如何交互"的三层结构，作为视觉语言理解与机器人控制之间的语义桥梁
2. **Mixture-of-Transformer (MoT) 架构**: 设计三个专家模块（理解 / affordance 生成 / 动作），通过单向注意力机制防止信息泄露与表示坍塌
3. **自动化数据标注流水线 InternData-A1**: 利用 VLM + SAM 构建 100K+ 条带 affordance 标注的机器人操作数据，支撑端到端训练

---

## 问题背景

### 要解决的问题

现有 [[VLA]]（Vision-Language-Action）模型直接将视觉与语言映射到动作，缺乏对**"应该操作哪个物体"**和**"在哪里/如何操作"**的显式中间推理，导致在精细操作任务中泛化能力不足。

### 现有方法的局限

- [[Pi0]]、[[OpenVLA]] 等方法直接回归动作，语义理解与运动规划混杂在同一表示空间中
- 单纯扩大数据规模（"blindly scaling up data"）无法充分激发模型内在能力
- 已有 affordance 方法（如 Where2Act）仅关注单一维度的交互点，无法端到端融入 VLA 框架

### 本文的动机

结构化 affordance 可作为**"任务导向的中间表示"**：先明确操作对象和操作位置，再指导低层动作生成，同时保留视觉语言模型的理解能力，防止精调时的灾难性遗忘。

---

## 方法详解

### 模型架构

AffordanceVLA 采用 **[[Mixture-of-Experts|Mixture-of-Transformer (MoT)]]** 架构，包含三个专家 Transformer：

- **输入**: 语言指令 $l$ + 视觉观测 $o_t$
- **理解专家** $\mathcal{M}_{und}$: 融合视觉观测与语言指令，生成高层语义 token
- **Affordance 生成专家** $\mathcal{M}_{gen}$: 接收语义 token，预测结构化 affordance token（Which / Where / How）
- **动作专家** $\mathcal{M}_{act}$: 以 affordance token 为条件，生成最终机器人动作
- **注意力方向**: Understanding → Affordance → Action（单向），防止动作信息反向干扰感知理解

### 核心模块

#### 模块 1: Which2Act（物体定位）

**设计动机**: 明确"要操作哪个物体"，通过视觉潜变量提供对象级语义锚点

**具体实现**:
- 使用冻结的 [[Flux VAE]] 编码器将目标物体图像编码为连续视觉潜变量 $z_q$
- [[Transformer]] 解码器预测重建潜变量 $\hat{z}_q$
- 选择连续 [[VAE]] 而非离散 [[VQ-VAE]]，避免量化误差对 token 投影的干扰

#### 模块 2: Where2Act（2D 交互定位）

**设计动机**: 明确"在物体的哪个位置交互"，提供像素级空间引导

**具体实现**:
- 输出 2D [[Affordance Map]]（热力图），分辨率为 $H_t \times W_t$
- [[Transformer]] 解码器 + [[Binary Cross-Entropy]] 监督
- 标注来源：RexOmni 模型 fine-tune 后基于关键帧自动标注

#### 模块 3: How2Act（3D 几何推理）

**设计动机**: 明确"如何执行操作"，提供三维空间的几何约束

**具体实现**:
- **3D 形状生成**: 条件 [[扩散模型]] 生成目标物体的 3D 形状（点云），建模接触面几何
- **空间布局回归**: 预测物体的位置、尺度等 10 维布局参数，使用 [[Smooth-L1 Loss]] 监督
- 两路输出共同为动作专家提供三维约束

---

## 关键公式

### 公式 1: [[MSE Loss|Which2Act 重建损失]]

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \|\hat{z}_{q,(c,h,w)} - z_{q,(c,h,w)}\|^2
$$

**含义**: 在通道 $C$、高度 $H$、宽度 $W$ 的空间维度上对视觉潜变量进行均方误差重建

**符号说明**:
- $\hat{z}_q$: 模型预测的视觉潜变量
- $z_q$: Flux VAE 编码器输出的目标潜变量
- $C, H, W$: 潜变量的通道数、高度、宽度

### 公式 2: [[Binary Cross-Entropy|Where2Act 二元交叉熵损失]]

$$
\mathcal{L}_{where} = -\frac{1}{H_t \cdot W_t} \sum_i \left[ M_i \cdot \log(\sigma(\hat{y}_i)) + (1 - M_i) \cdot \log(1 - \sigma(\hat{y}_i)) \right]
$$

**含义**: 对预测的 affordance 热力图与标注 affordance mask 进行逐像素二元交叉熵监督

**符号说明**:
- $M_i \in \{0, 1\}$: 第 $i$ 个像素的真实 affordance 标签
- $\hat{y}_i$: 模型预测的 logit
- $\sigma(\cdot)$: Sigmoid 函数
- $H_t, W_t$: affordance 图分辨率

### 公式 3: [[Diffusion Model|How2Act 形状扩散损失]]

$$
\mathcal{L}_{shape} = \mathbb{E}\left[\|\epsilon - \hat{\epsilon}_\theta(x_t, t, \bar{h}_{shape})\|^2\right]
$$

**含义**: 条件扩散模型去噪目标，以 affordance 特征 $\bar{h}_{shape}$ 为条件重建目标物体 3D 形状

**符号说明**:
- $\epsilon$: 真实噪声
- $\hat{\epsilon}_\theta$: 去噪网络预测的噪声
- $x_t$: 第 $t$ 步加噪后的点云
- $\bar{h}_{shape}$: affordance 生成专家输出的形状条件特征

### 公式 4: [[Smooth-L1 Loss|How2Act 布局回归损失]]

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \text{SmoothL}_1\!\left(\hat{y}_{layout}^{(j)},\, y_{layout}^{(j)}\right)
$$

**含义**: 对 10 维物体布局参数（位置、尺度等）进行 Smooth-L1 回归，鲁棒于异常值

**符号说明**:
- $\hat{y}_{layout}^{(j)}$: 第 $j$ 维预测布局参数
- $y_{layout}^{(j)}$: 第 $j$ 维真实布局参数

### 公式 5: [[Multi-Task Learning|Stage I 多任务 Affordance 预训练损失]]

$$
\mathcal{L}_{Stage1} = \lambda_{which} \mathcal{L}_{which} + \lambda_{where} \mathcal{L}_{where} + \lambda_{shape} \mathcal{L}_{shape} + \lambda_{layout} \mathcal{L}_{layout}
$$

**含义**: 第一阶段仅训练 Affordance 生成专家，联合优化四个 affordance 预测目标

**符号说明**:
- $\lambda_{which}, \lambda_{where}, \lambda_{shape}, \lambda_{layout}$: 各子任务损失权重

### 公式 6: [[Imitation Learning|Stage II 联合训练损失]]

$$
\mathcal{L}_{Stage2} = \lambda_{act} \mathcal{L}_{act} + \lambda_{afd} \mathcal{L}_{afd}
$$

**含义**: 第二阶段解冻理解专家与动作专家，在动作监督与 affordance 监督间平衡训练

**符号说明**:
- $\mathcal{L}_{act}$: 动作生成损失（流匹配 / 扩散目标）
- $\mathcal{L}_{afd}$: affordance 联合损失（$\mathcal{L}_{Stage1}$）
- $\lambda_{afd}$ 在 Stage III 退火至 0.15

---

## 关键图表

### Figure 1: AffordanceVLA 系统概览

![Figure 1 Overview](https://arxiv.org/html/2606.06155v1/x1.png)

**说明**: AffordanceVLA 整体框架。左侧输入视觉观测与语言指令，通过 [[Mixture-of-Experts|MoT]] 三专家依次处理，Which2Act 定位操作物体（视觉潜变量），Where2Act 输出 2D 热力图，How2Act 输出 3D 形状与布局，最终由动作专家生成机器人动作序列。

### Figure 2: 数据标注流水线与模型架构

![Figure 2 Pipeline](https://arxiv.org/html/2606.06155v1/x2.png)

**说明**: 左侧展示 InternData-A1 自动标注四步流程（RexOmni fine-tune → 关键帧检测 → LLM 指令分解 + VLM 标注 → SAM 视觉 grounding）；右侧展示 MoT 架构中三个 [[Transformer]] 专家的单向注意力连接方式。

### Figure 3: 数据效率曲线

![Figure 3 Data Efficiency](https://arxiv.org/html/2606.06155v1/x3.png)

**说明**: 在不同训练数据比例下对比 AffordanceVLA 与 Pi0 的 LIBERO 成功率。AffordanceVLA 在仅使用 40% 数据时即超越使用 100% 数据的 Pi0，说明结构化 affordance 表示带来的架构优势而非数据优势。

### Figure 4: 真实世界实验可视化

![Figure 4 Real World](https://arxiv.org/html/2606.06155v1/x4.png)

**说明**: 真实机器人执行多项操作任务的关键帧。展示模型在基础任务（拾取杯子、香蕉等）和复杂任务（抽屉开合、烤箱操作）上的执行效果，Where2Act 热力图直观展示模型的注意区域。

### Figure 5: Subgoal Token 定量评估

![Figure 5 Subgoal Tokens](https://arxiv.org/html/2606.06155v1/x5.png)

**说明**: 对比不同 affordance token 设计的性能差异，验证结构化三层 affordance（Which/Where/How）相比 block-wise token 和无 affordance 设计的优越性。

### Figure 6: How2Act 布局 Token 可视化

![Figure 6 Layout Tokens](https://arxiv.org/html/2606.06155v1/x6.png)

**说明**: How2Act 模块预测的 3D 布局 token 在不同任务场景下的可视化，展示空间定位精度。

### Figure 7: Backbone 表示评估

![Figure 7 Backbone](https://arxiv.org/html/2606.06155v1/x7.png)

**说明**: 评估不同 backbone 特征在 affordance 预测上的表现，验证 affordance 中间表示在保留视觉语言能力方面的效果。

### Figure 8: Where2Act Affordance Map 可视化

![Figure 8 Where2Act](https://arxiv.org/html/2606.06155v1/x8.png)

**说明**: Where2Act 输出的 2D affordance 热力图（如抽屉把手、杯子边缘等）与真实标注的对比，展示模型在多样操作场景下的空间定位准确性。

### Table 1: LIBERO 基准对比

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| RoboVLMs | 90.2% | 91.5% | 87.3% | 68.1% | 84.3% |
| Pi0 | 98.0% | 96.8% | 94.4% | 88.4% | 94.4% |
| F1-VLA | 98.2% | 97.8% | 95.4% | 91.3% | 95.7% |
| **AffordanceVLA** | **98.6%** | **98.4%** | **96.2%** | **89.8%** | **95.8%** |

**关键发现**: AffordanceVLA 在 LIBERO 四个子集上均达到或超过 SOTA，Long-horizon 任务的 89.8% 相比 OpenVLA 的 53.7% 有大幅提升。

### Table 2: CALVIN ABC→D 基准对比

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg Length |
|--------|-----|-----|-----|-----|-----|-----------|
| VPP | 88.2% | 73.6% | 61.4% | 48.8% | 27.5% | 4.01 |
| Seer-Large | 91.1% | 80.7% | 70.4% | 58.0% | 38.5% | 4.28 |
| **AffordanceVLA** | **93.1%** | **83.2%** | **74.1%** | **62.0%** | **43.8%** | **4.33** |

**关键发现**: 在需要跨场景泛化的 CALVIN 基准上，AffordanceVLA 平均完成 4.33 步连续任务，超过此前 SOTA Seer-Large 的 4.28 步。

### Table 3: 消融实验（LIBERO-Long）

| 配置 | LIBERO Avg | 说明 |
|------|-----------|------|
| No-Afd (Pi0 架构) | 92.4% | 仅换数据，不加 affordance |
| Frozen-Affordance | 67.1% | affordance 专家不参与联合训练 |
| Block-wise Tokens | 90.3% | 非结构化 affordance token |
| **Full AffordanceVLA** | **95.8%** | 完整三层结构化 affordance |

**关键发现**: Frozen-Affordance 严重坍塌（67.1%）证明联合优化至关重要；Block-wise Token 相比全模型下降 5.5%，说明结构化表示而非单纯监督密度是关键。

### Table 4: 真实世界实验结果

| 任务类型 | Tasks | AffordanceVLA | Pi0 | 提升 |
|----------|-------|--------------|-----|------|
| 基础操作 | 拾取 6 种物体 | 88.3% | 70.8% | +17.5% |
| 复杂操作 | 抽屉/烤箱 | 82.9% | 44.8% | +38.1% |

**关键发现**: 在需要精细操作的复杂任务上，AffordanceVLA 以 38.1% 的绝对优势超越 Pi0，验证 How2Act 3D 几何推理的实用价值。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[AGD20K]] | 20K 图像 | 人-物交互 affordance 标注 | Stage I 预训练 |
| [[RefSpatial]] | — | 空间参考 affordance | Stage I 预训练 |
| [[PRISM]] | — | 机器人操作 affordance | Stage I 预训练 + RexOmni fine-tune |
| InternData-A1 | 100K+ 条 | 自动标注机器人操作 + affordance | Stage II 联合训练 |
| [[LIBERO]] | 500 任务 | 4 子集 (Spatial/Object/Goal/Long) | 测试 |
| [[CALVIN]] | ABC→D 设置 | 长序列跨场景泛化 | 测试 |

### 实现细节

- **Backbone**: [[Pi0]] 为基础（[[PaliGemma]] 视觉语言模型 + 流匹配动作头）
- **Affordance 专家**: Transformer 解码器
- **扩散模型**: 条件扩散（How2Act 形状生成）
- **Stage I**: 冻结视觉编码器 + 理解专家 + 动作专家，仅训练 affordance 生成专家
- **Stage II**: 解冻理解专家与动作专家，$\lambda_{act} : \lambda_{afd} = $ 平衡调节
- **Stage III**: $\lambda_{afd}$ 退火至 0.15，适配目标任务
- **数据标注**: RexOmni + [[SAM]] + [[SAM-3D]] + LLM 指令分解，100% bbox 内点验证 + 3,000 样本人工审核

### 可视化结果

- Where2Act 热力图精准定位抽屉把手、杯子边缘等精细交互区域
- How2Act 3D 布局预测在烤箱开关等需要深度感知的任务上提供显著几何约束
- 数据效率曲线显示结构化表示在低数据场景下优势更为突出

---

## 批判性思考

### 优点

1. **三层结构化中间表示**: Which/Where/How 分层设计直觉清晰，每层解决特定维度的歧义性
2. **防遗忘机制**: 单向注意力 + 专家解耦有效缓解 VLA 微调中的灾难性遗忘问题
3. **数据效率**: 40% 数据超越 Pi0 100% 数据，架构优势独立于数据规模
4. **实用性验证**: 同时在模拟器基准和真实机器人上验证，说服力强

### 局限性

1. **标注流水线依赖**: InternData-A1 的质量取决于 RexOmni、SAM 等上游模型的精度，存在错误传播风险
2. **How2Act 复杂度**: 条件扩散生成 3D 形状引入额外推理延迟，实时控制场景下可能受限
3. **场景泛化**: 主要在桌面操作任务验证，足式机器人或非接触式任务的适用性未探讨

### 潜在改进方向

1. 在线学习机制：利用执行反馈动态更新 affordance 预测，形成闭环
2. 轻量化 How2Act：用隐式表示替代显式 3D 点云生成，降低推理开销

### 可复现性评估

- [x] 代码开源（GitHub）
- [ ] 预训练模型（未明确提供权重）
- [x] 训练细节完整（三阶段策略描述详细）
- [ ] 数据集可获取（InternData-A1 未公开）

---

## 关联笔记

### 基于

- [[Pi0]]: 使用 Pi0 的 PaliGemma + 流匹配动作头作为基础架构
- [[Flux VAE]]: Which2Act 视觉潜变量编码器
- [[SAM]]: 数据标注流水线中的视觉 grounding 工具

### 对比

- [[Pi0]]: 主要基线，同为基于 PaliGemma 的 VLA
- [[F1-VLA]]: LIBERO 上最接近的竞争者
- [[Seer]]: CALVIN 上最接近的竞争者

### 方法相关

- [[Affordance]]: 核心概念，可供性感知
- [[Mixture-of-Experts]]: MoT 架构基础
- [[Diffusion Model]]: How2Act 3D 形状生成
- [[Imitation Learning]]: 整体训练范式

### 数据/环境相关

- [[LIBERO]]: 主要评测基准
- [[CALVIN]]: 跨场景泛化基准
- [[AGD20K]]: 预训练 affordance 数据集

---

## 速查卡片

> [!summary] AffordanceVLA
> - **核心**: 将结构化 affordance（Which/Where/How）作为 VLA 中间表示，桥接语义理解与动作生成
> - **方法**: Mixture-of-Transformer 三专家架构 + 三阶段训练 + 自动化 affordance 标注流水线
> - **结果**: LIBERO 95.8% / CALVIN 4.33 avg length，真实机器人复杂任务超 Pi0 38.1%
> - **代码**: [GitHub](https://github.com/Skywalker-yqz/AffordanceVLA)

---

*笔记创建时间: 2026-06-05*
