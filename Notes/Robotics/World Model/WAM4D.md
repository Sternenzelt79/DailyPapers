---
title: "WAM4D: Fast 4D World Action Model via Spatial Register Tokens"
method_name: "WAM4D"
authors: [Ying Li, Xiaobao Wei, Jiajun Cao, Hao Wang, Xiaowei Chi, Chengyu Bai, Qianpu Sun, Jiajun Li, Xiaojie Zhang, Jian Tang, Sirui Han, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, 4d-representation, depth-estimation, robot-manipulation, knowledge-distillation, flow-matching, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.14048v1
created: 2026-06-15
---

# 论文笔记：WAM4D: Fast 4D World Action Model via Spatial Register Tokens

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未在摘要页公开（通讯: Shanghang Zhang） |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[LingBot]] / [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.14048) / [Code](https://github.com/myendless1/wam4d) |

---

## 一句话总结

> WAM4D 在训练时引入轻量级空间寄存器 Token 蒸馏几何先验，推理时移除几何分支，以接近 2D WAM 的计算代价实现 4D 空间感知的机器人操作策略。

---

## 核心贡献

1. **空间寄存器蒸馏（Spatial Register Distillation）**: 训练时插入可学习寄存器 Token 作为未来深度的读出接口，从 [[Depth Anything 3|Depth Anything V3]] 蒸馏几何先验到视频-动作 Transformer，推理时完全移除寄存器分支，无额外开销。
2. **因果混合注意力（Causal Mixture Attention）**: 为 [[Mixture-of-Transformers]] 主干设计模态专属的因果可见性掩码，防止未来动作 Token 访问未来视频 Token，消除非因果捷径。
3. **训练-推理不对称范式**: 通过训练时几何监督 + 推理时纯观测-动作路径，在 RoboTwin 2.0 上以 525ms 延迟和 9.71 GiB 显存达到 91.8% 平均成功率，真实机器人任务 0.90 平均子动作成功率。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]]（WAM）主要在 2D 视频空间或潜空间建模，生成的视频虽然视觉合理，但缺少精确操作所需的 3D 空间约束（遮挡接触几何、物体动力学）。机器人操作不是简单的观测到动作映射，需要对场景几何、接触点和物体动力学进行推理。

### 现有方法的局限

- 2D WAM（π₀、π₀.₅）无几何先验，空间推理能力弱
- 直接预测 4D 表示（RGBD / 点云输出）引入大量计算开销
- 将几何基础模型接入 WAM 会显著增加推理延迟和显存
- [[Motus]] / [[LingBot-VA]] 延迟分别达 1516ms 和 844ms，难以实时部署

### 本文的动机

几何基础模型（如 [[Depth Anything 3|Depth Anything V3]]）已经具备强大的深度先验，但仅在训练阶段使用这些先验来塑造视频-动作特征即可——不需要在推理时保留几何分支。"寄存器 Token"作为训练时的几何查询探针，蒸馏完成后推理阶段可安全丢弃。

---

## 方法详解

### 模型架构

WAM4D 采用 **[[Mixture-of-Transformers]]** 多模态 Transformer 架构，基于 [[LingBot-VA]] 预训练权重初始化：

- **输入**: 语言指令 $l$ + 多视角 RGB 历史帧 $O_t^{hist}$ + 历史动作队列 $a_{t-L_a:t-1}$
- **VAE Backbone**: [[Wan2.2]] VAE 编解码视频流
- **核心模块**: [[Register Tokens|空间寄存器蒸馏]] + [[Mixture-of-Transformers|因果混合注意力]]
- **输出**: 双臂 16 维动作块 $\tilde{a}_{t:t+H_a-1}$（绝对末端执行器位姿）
- **总参数**: 5.089B–5.841B（不同消融变体）

### 核心模块

#### 模块 1: WAM4D Backbone（视频-动作主干）

**设计动机**: 基于 [[Flow Matching]] 的因果视频-动作联合建模，延续 LingBot-VA 的 [[Mixture-of-Transformers]] 设计，视频流和动作流共享 Transformer 层但使用独立的预测头。

**具体实现**:
- 历史和未来 RGB 帧经 VAE 编码得到潜变量 Token 序列
- 未来帧潜变量和未来动作在训练时加噪，作为扩散去噪目标
- 历史动作和历史视频 Token 构成因果上下文（Eq. 1）
- Token 序列送入 VABlock 堆叠，使用因果可见性掩码 $M_{VA}$（Eq. 5）

#### 模块 2: Spatial Register Distillation（空间寄存器蒸馏）

**设计动机**: 不在推理路径上保留几何模块，而是在训练时用寄存器 Token 作为"临时探针"，把几何先验塑入视频-动作特征。

**具体实现**:
- 初始化一个可学习寄存器网格 $R_\star$，沿未来时间步展开（Eq. 6）
- 在 Transformer 选定层（默认 12/14/16/18 层）插入 DepthBlock，寄存器只能自注意力 + 交叉注意力历史视频 Token（Eq. 7）
- 寄存器输出经轻量投影层 + [[Depth Anything 3|Depth Anything V3]] 的 DualDPT 几何头，预测未来深度图（Eq. 8）
- 深度损失用 SmoothL1 对有效深度像素 $\Omega_\tau$ 监督（Eq. 9）
- **推理时**: 寄存器分支、DepthBlock、几何头全部移除，模型退化为纯观测-动作路径

**视角配置（三视角默认）**:
- 主视角：256×320，8×10 寄存器网格
- 每个腕部视角：128×160，4×5 寄存器网格
- 三视角合计：12×10 网格 × 8 未来帧 = **960 寄存器 Token**

#### 模块 3: Causal Mixture Attention（因果混合注意力）

**设计动机**: 原始 MoT 的注意力掩码未明确限制跨模态信息流，可能出现未来动作直接"看到"未来视频的非因果捷径。

**可见性规则**:

| Query Token 组 | 允许 Key/Value 组 |
|---|---|
| 未来动作噪声 | 历史视频、历史动作、未来动作噪声 |
| 寄存器 | 寄存器、历史视频 |
| 未来视频噪声 | 历史视频、未来视频噪声 |
| 历史视频 | 历史视频 |
| 历史动作 | 历史动作 |

**关键安全约束**: 未来动作 Token 被遮蔽于未来视频 Token 和寄存器之外；寄存器只访问历史视频，确保深度目标不污染动作预测路径。

---

## 关键公式

### 公式 1: [[Causal Context|因果上下文]]

$$
\mathcal{C}_t = \{l,\; O_t^{hist},\; a_{t-L_a:t-1}\}
$$

**含义**: 决策时刻 $t$ 的条件信息，由语言指令、历史多视角观测和历史动作队列构成。

**符号说明**:
- $l$: 自然语言任务指令
- $O_t^{hist}$: 历史 RGB 帧序列（最多 17 帧）
- $a_{t-L_a:t-1}$: 长度为 $L_a$ 的历史动作队列

### 公式 2: [[VAE Encoding|VAE 编码]]

$$
[\mathbf{Z}_t^{hist},\; \mathbf{Z}_t^{fut}] = E_{vae}([O_t^{hist},\; O_t^{fut}])
$$

**含义**: VAE 编码器将历史和未来 RGB 帧序列压缩为潜变量 Token，训练时未来 Token 加噪为 $\tilde{\mathbf{Z}}_t^{fut}$。

**符号说明**:
- $E_{vae}$: [[Wan2.2]] VAE 编码器
- $\mathbf{Z}_t^{hist}$: 历史视频潜变量
- $\mathbf{Z}_t^{fut}$: 未来视频潜变量（去噪目标）

### 公式 3: [[Action Embedding|动作流嵌入]]

$$
\mathbf{A}_t^{hist} = E_a(a_{t-L_a:t-1}), \quad \tilde{\mathbf{A}}_t^{fut} = E_a(\tilde{a}_{t:t+H_a-1})
$$

**含义**: 轻量动作嵌入层将历史和未来（已加噪）动作块映射为 Transformer 可处理的 Token。

**符号说明**:
- $E_a$: 动作嵌入层
- $H_a$: 未来动作预测步长（Action Chunk 大小 = 32）
- $\tilde{a}$: 扩散过程中的加噪动作

### 公式 4: [[Token Sequence|Token 序列构造]]

$$
\mathbf{X}_t^{(0)} = [\mathbf{Z}_t^{hist},\; \tilde{\mathbf{Z}}_t^{fut},\; \mathbf{A}_t^{hist},\; \tilde{\mathbf{A}}_t^{fut}]
$$

**含义**: 将四类 Token 拼接为 Transformer 输入序列，历史信息提供因果上下文，未来信息作为去噪目标。

### 公式 5: [[Backbone Forward|主干前向传播]]

$$
\mathbf{X}_t^{(\ell+1)} = \text{VABlock}_\ell(\mathbf{X}_t^{(\ell)};\; M_{VA})
$$

**含义**: 第 $\ell$ 层 VABlock 在因果可见性掩码 $M_{VA}$ 约束下更新 Token 序列。

**符号说明**:
- $M_{VA}$: 因果混合注意力掩码矩阵
- $\ell$: Transformer 层索引

### 公式 6: [[Register Initialization|寄存器初始化]]

$$
\mathbf{R}_t^0 = \underset{\tau \in \mathcal{T}_t}{\text{Repeat}}(\mathbf{R}_\star)
$$

**含义**: 将可学习寄存器网格 $R_\star$ 沿未来时间步 $\mathcal{T}_t$ 展开复制，生成初始寄存器 Token 集合。

**符号说明**:
- $\mathbf{R}_\star$: 共享的可学习寄存器网格（12×10 格，三视角）
- $\mathcal{T}_t = \{t+1,...,t+H_v\}$: 未来深度监督时间步集合

### 公式 7: [[Register Update|寄存器更新]]

$$
\mathbf{R}_t^{(\ell+1)} = \text{DepthBlock}_\ell\!\left(Q=\mathbf{R}_t^\ell,\; K{,}V=[\mathbf{R}_t^\ell,\; \mathbf{Z}_t^{hist,\ell}]\right)
$$

**含义**: 在选定层 $\ell \in \mathcal{L}_r$，寄存器 Token 以自注意力 + 对历史视频 Token 的交叉注意力方式更新，提取未来几何信息。

**符号说明**:
- $\text{DepthBlock}_\ell$: 独立于主干的深度提取注意力块
- $\mathbf{Z}_t^{hist,\ell}$: 第 $\ell$ 层历史视频 Token（只读，不反向传播到此路径）
- $\mathcal{L}_r = \{12, 14, 16, 18\}$: 默认寄存器插入层

### 公式 8: [[Geometric Head|几何头输出]]

$$
\mathbf{G}_t = \mathcal{P}_g\!\left(\{\mathbf{R}_t^{(\ell+1)}\}_{\ell \in \mathcal{L}_r}\right), \quad \hat{\mathbf{D}}_t^{fut} = \mathcal{G}_\phi(\mathbf{G}_t)
$$

**含义**: 多层寄存器输出经轻量投影层汇聚，再送入预训练几何头预测未来深度序列。

**符号说明**:
- $\mathcal{P}_g$: 轻量线性投影层（4 个适配器）
- $\mathcal{G}_\phi$: [[Depth Anything 3|Depth Anything V3]] DualDPT 几何头（DA3-GIANT-1.1）
- $\hat{\mathbf{D}}_t^{fut}$: 预测的未来深度图序列

### 公式 9: [[Depth Loss|深度监督损失]]

$$
\mathcal{L}_{depth} = \frac{1}{\sum_{\tau \in \mathcal{T}_t}|\Omega_\tau|} \sum_{\tau \in \mathcal{T}_t} \sum_{p \in \Omega_\tau} \text{SmoothL1}\!\left(\hat{D}_{\tau,p},\; D_{\tau,p}\right)
$$

**含义**: 对未来所有有效深度像素 $\Omega_\tau$ 计算 SmoothL1 损失，监督寄存器对几何信息的提取质量。

**符号说明**:
- $\Omega_\tau$: 时刻 $\tau$ 的有效深度像素集合（忽略无效区域）
- $D_{\tau,p}$: 真实深度值（来自仿真器或深度传感器）
- $\hat{D}_{\tau,p}$: 几何头预测深度值

### 公式 10: [[Training Objective|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{video} + \lambda_{act}\,\mathcal{L}_{action} + \lambda_{depth}\,\mathcal{L}_{depth}
$$

**含义**: 三路损失联合优化：视频生成质量、动作预测精度、深度几何先验蒸馏。

**符号说明**:
- $\mathcal{L}_{video}$: 视频流 [[Flow Matching]] 损失
- $\mathcal{L}_{action}$: 动作流 [[Flow Matching]] 损失
- $\lambda_{act} = \lambda_{depth} = 1$: 损失权重（均为 1）

---

## 关键图表

### Figure 1: 4D 世界动作模型设计对比

![Figure 1](https://arxiv.org/html/2606.14048v1/x1.png)

**说明**: 对比几种 4D WAM 设计范式。现有方案或将 4D 几何作为额外输入（增加推理复杂度），或直接预测 RGBD 输出（增加输出维度），或外接几何模块（增加延迟）。WAM4D 仅在训练时使用寄存器蒸馏几何先验，推理路径与 2D WAM 相同。

### Figure 2: WAM4D 架构与因果可见性模式

![Figure 2](https://arxiv.org/html/2606.14048v1/x2.png)

**说明**: 展示 [[Mixture-of-Transformers]] 主干的 Token 组成及 [[Mixture-of-Transformers|因果混合注意力]] 的掩码矩阵。寄存器 Token 仅在训练时存在，只访问历史视频 Token，与动作流完全隔离。

### Figure 3: 训练路径 vs 推理路径

![Figure 3](https://arxiv.org/html/2606.14048v1/x3.png)

**说明**: 清晰展示训练-推理不对称设计。训练时三路损失同时反向传播；推理时寄存器分支（灰色虚框）完全移除，仅保留视频-动作生成路径，实现零额外开销。

### Figure 4: AstriBot S1 真实机器人任务

![Figure 4](https://arxiv.org/html/2606.14048v1/x4.png)

**说明**: 展示四类真实世界任务（盘子取放、瓶子摆放、LEGO 积木分拣、钢笔装帽），均要求精确的接触几何感知。WAM4D 平均子动作成功率 0.90，优于所有基线。

### Figure 5: 空间寄存器注意力可视化

![Figure 5](https://arxiv.org/html/2606.14048v1/x5.png)

**说明**: 在 RoboTwin 随机化样本上可视化寄存器的注意力权重分布，验证寄存器确实关注了场景中的几何关键区域（物体边缘、接触点）。

### Figure 6: RGB-D 和点云 4D Rollout 可视化

![Figure 6](https://arxiv.org/html/2606.14048v1/x6.png)

**说明**: 展示 WAM4D 生成的多步未来预测，同时包含 RGB 视频帧、深度图和反投影点云，验证 4D 时空一致性。深度图与 RGB 在空间上精确对应。

### Figure 7: 长程自回归失败案例

![Figure 7](https://arxiv.org/html/2606.14048v1/x7.png)

**说明**: 展示长程自回归 Rollout 中的主要失败模式——严重遮挡后物体身份不一致。注：此问题仅影响生成可视化，真实控制时模型持续接收真实观测，不影响策略成功率。

### Table 1: RoboTwin 2.0 全任务成功率对比

| 方法 | Clean (%) | Randomized (%) | Avg. (%) | 延迟 (ms) | VRAM (GiB) |
|------|-----------|----------------|----------|-----------|-----------|
| π₀ | 65.9 | 58.4 | 62.2 | 64.16±0.06 | 8.45 |
| π₀.₅ | 82.7 | 76.8 | 79.8 | 72.03±0.06 | 8.45 |
| Motus | 88.7 | 87.0 | 87.9 | 1516.30±10.64 | 11.55 |
| LingBot-VA | 92.9 | 91.6 | 92.3 | 843.57±11.55 | 12.97 |
| Fast-WAM | 91.9 | 91.8 | 91.8 | **425.53±6.01** | 11.55 |
| **WAM4D** | **93.8** | 89.9 | 91.8 | 525.43±5.64 | **9.71** |

**关键发现**: WAM4D 在 Clean 设置取得最高成功率（93.8%），以 91.8% 平均成功率与 Fast-WAM 持平，但 VRAM 少 1.84 GiB，真实世界表现更优。

### Table 2: AstriBot S1 真实机器人子动作成功率

| 方法 | Plate S1 | Bottle S1 | Blocks S1 | Blocks S2 | Blocks S3 | Pen S1 | Pen S2 | Avg. |
|------|----------|-----------|-----------|-----------|-----------|--------|--------|------|
| π₀.₅ | 1.0 | 0.8 | 0.7 | 0.6 | 0.5 | 0.8 | 0.8 | 0.74 |
| LingBot-VA | 1.0 | 1.0 | 1.0 | 0.7 | 0.4 | 0.9 | 0.9 | 0.84 |
| Fast-WAM | 0.9 | 1.0 | 0.8 | 0.7 | 0.5 | 0.9 | 0.8 | 0.80 |
| **WAM4D** | 0.9 | 0.9 | **1.0** | **0.9** | **0.8** | 0.9 | 0.9 | **0.90** |

**关键发现**: WAM4D 在接触密集型任务（Blocks S2/S3）上显著优于 Fast-WAM（0.9/0.8 vs 0.7/0.5），体现了几何先验的实际价值。

### Table 5-8: 消融实验结果

**几何头设计消融（Table 8 核心结果）**:

| 变体 | Clean SR (%) | FVD ↓ | AbsRel ↓ | F-score ↑ |
|------|-------------|-------|---------|----------|
| 无深度监督 | 71.7 | 181.2 | – | – |
| 随机初始化深度头 | 70.0 | 189.8 | 0.059 | 0.589 |
| 固定预训练深度头 | 75.2 | 179.8 | 0.053 | 0.685 |
| **可训练预训练深度头** | **80.1** | **164.5** | **0.049** | **0.710** |

**寄存器层位置消融（Table 7 核心结果）**:

| 层位置 | Clean SR (%) | 结论 |
|--------|-------------|------|
| 浅层 (2,4,6,8) | 72.5 | 有利于视频生成但几何控制弱 |
| **中层 (12,14,16,18)** | **75.2** | 最优平衡，为默认配置 |
| 深层 (22,24,26,28) | 74.5 | 特征过于专用化 |
| 均匀 (6,12,18,24) | 70.6 | 混合抽象层级效果差 |
| 双向注意力 | 76.6 | SR 最高但破坏几何指标且不因果 |

**关键发现**: 预训练几何头 + 可训练微调 是关键，随机初始化甚至比无深度监督还差（70.0% vs 71.7%），证明强几何先验是有效蒸馏的前提。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 50 任务 × (50 clean + 500 randomized) 轨迹 | 仿真，包含随机化设置 | 主要评测 |
| AstriBot S1 | 4 任务 × 100 示例 | 真实双臂机器人 | 真实世界评测 |

### 实现细节

- **Backbone**: 基于 [[LingBot-VA]] 预训练权重初始化，[[Wan2.2]] VAE
- **优化器**: AdamW，学习率 $2 \times 10^{-5}\sqrt{N}$（$N$ = 机器数量），10 步 warmup
- **Batch Size**: 未明确说明
- **训练步数**: 主实验 50k 步，消融 10k 步
- **硬件**: 多机并行（具体 GPU 型号未公开）
- **梯度裁剪**: 2.0
- **精度**: bf16 参数
- **帧采样**: 最多 17 帧，历史 1/5/9 帧，未来 8 帧（每 4 步采一帧）
- **动作块**: 32 步，16 维（双臂绝对末端位姿）

### 可视化结果

Figure 6 展示了 WAM4D 生成的多步 RGB-D 预测，深度图与 RGB 在空间上高度对齐，点云时序一致性良好。Figure 5 的寄存器注意力热图表明模型自发学习关注接触关键区域。

---

## 批判性思考

### 优点

1. **训练-推理不对称设计优雅**: 将几何先验完全内化为特征表示，推理路径零开销，工程实用性强
2. **几何先验有效性验证充分**: 消融实验覆盖头部初始化、层位置、注意力模式等多个维度，结论可信
3. **真实机器人验证有说服力**: 在接触密集型任务上的提升（Blocks S2 0.9 vs 0.5）直接证明了空间几何感知的价值

### 局限性

1. **长程自回归遮挡失忆**: 模型缺少显式长期物体记忆，严重遮挡后无法保持物体身份一致性（尽管不影响控制）
2. **速度仍慢于 VLA**: 525ms 延迟虽比 LingBot-VA（844ms）更快，但远慢于 π₀（64ms），不适合快速动态任务
3. **几何监督来源受限**: 当前依赖仿真器深度真值，真实场景需要深度传感器，部署条件受约束
4. **机构和训练数据未公开**: 复现细节不完整

### 潜在改进方向

1. 引入持久物体记忆模块，解决长程遮挡失忆问题
2. 探索寄存器 Token 在推理时的轻量版本（如稀疏化），进一步降低延迟
3. 结合 [[Monocular Depth Estimation]] 替代深度传感器，扩展真实场景适用性

### 可复现性评估

- [x] 代码开源（GitHub: myendless1/wam4d）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、帧采样、归一化均有说明）
- [x] 数据集可获取（RoboTwin 2.0 公开）

---

## 关联笔记

### 基于

- [[LingBot]]: WAM4D 的基础架构和预训练权重来源
- [[Depth Anything 3|Depth Anything V3]]: 预训练几何头来源（DA3-GIANT-1.1 DualDPT）
- [[Wan2.2]]: VAE 编解码器

### 对比

- [[Fast-WAM]]: 速度优化方向的竞争方法（425ms vs 525ms）
- [[LingBot]]: 父模型，WAM4D 在其基础上添加几何蒸馏
- [[Motus]]: 另一 4D WAM 方法，延迟更高（1516ms）

### 方法相关

- [[Flow Matching]]: 视频和动作的去噪框架
- [[Mixture-of-Transformers]]: 多模态主干架构
- [[Knowledge Distillation]]: 寄存器蒸馏的理论基础
- [[Register Tokens]]: ViT 中 register token 概念在此扩展到几何蒸馏

### 硬件/数据相关

- [[AstriBot S1]]: 真实双臂机器人平台
- [[RoboTwin 2.0]]: 主要仿真评测 Benchmark

---

## 速查卡片

> [!summary] WAM4D: Fast 4D World Action Model via Spatial Register Tokens
> - **核心**: 训练时蒸馏几何先验，推理时移除几何分支，实现零开销 4D 世界动作建模
> - **方法**: 空间寄存器 Token 蒸馏（Depth Anything V3）+ 因果混合注意力
> - **结果**: RoboTwin 2.0 平均 91.8%（Clean 93.8%），真实机器人 0.90，延迟 525ms / 9.71 GiB
> - **代码**: https://github.com/myendless1/wam4d

---

*笔记创建时间: 2026-06-15*
