---
title: "Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning"
method_name: "AdaWAM"
authors: [Yinzhou Tang, Jingbo Xu, Yu Shang, Zihao Song, Chen Gao, Wei Wu, Yong Li]
year: 2026
venue: arXiv
tags: [world-action-model, adaptive-reasoning, embodied-ai, diffusion-transformer, multimodal-routing]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.07089
created: 2026-06-08
---

# 论文笔记：Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University; Manifold AI |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07089) / Code: N/A |

---

## 一句话总结

> AdaWAM 提出自适应多模态路由机制，让 [[WAM|世界动作模型]] 能在文本推理、视觉预测和纯动作解码三种模式间动态切换，在保持推理效率的同时显著提升长时程和精细操作任务的成功率。

---

## 核心贡献

1. **自适应多模态推理框架**: 提出 AdaWAM，包含动态路由器，在推理时自主决定是否触发文本推理（[[CoT|语言链式推理]]）或视觉推理（视频预测），而非固定使用所有模块
2. **多模态推理数据标注流程**: 设计自动化标注 pipeline，利用轨迹运动特征识别精细操作区间，利用 [[Qwen3-VL|Qwen3-VL-8B]] 验证子任务切换点
3. **高效推理与强泛化**: 推理效率与纯动作解码的 [[Fast-WAM]] 相当，但在 LIBERO-Long 上达到 99.1% 成功率，未见任务组合泛化成功率达 61%

---

## 问题背景

### 要解决的问题

现有 [[WAM|World Action Model]] 方法依赖视频预测作为动作先验，但缺乏自适应的多模态推理能力：在所有时间步都进行视频预测导致推理效率低，同时语言仅被用于静态任务描述而未被充分利用于动态高层决策。

### 现有方法的局限

- **[[Fast-WAM]]**: 仅做纯动作解码，高层语义推理能力弱，长时程任务成功率低
- **CoT-VLA / MM-ACT 等**: 固定在每步触发语言推理，推理延迟大
- **所有 WAM**: 缺乏感知"何时需要视觉想象"的机制，要么过度预测浪费算力，要么完全不预测损失精度

### 本文的动机

不同操作阶段需要不同推理模式：任务转换点需要文本推理做高层规划，精细接触操作需要视觉推理（想象未来状态以精准控制），中间动作执行阶段只需直接解码动作。因此，关键在于动态判断"当下是否需要做梦（视觉预测）"。

---

## 方法详解

### 模型架构

AdaWAM 采用 **[[Diffusion Transformer|扩散 Transformer]] + [[CoT|语言推理]] + [[MoE|动态路由]]** 混合架构：

- **输入**: 语言指令 $l$ + 历史视觉观测 $z_{\leq t}$ + 当前文本上下文嵌入 $e_l$、$e_{ct}$
- **视觉骨干**: [[Wan2.2|Wan2.2-5B]]（视频扩散 Transformer）作为 VideoDiT $\mathcal{M}_\theta$
- **动作解码**: 压缩 1B ActionDiT $\pi_\phi$（隐藏维度 1024），基于视觉与语言条件生成 [[Action Chunking|动作块]]
- **文本推理**: [[Qwen3-VL|Qwen3-VL-4B]] 作为紧凑 VLM $\mathcal{V}_\omega$
- **动态路由器**: 轻量级路由网络 $\mathcal{R}_\psi$，独立预测 `<TR>` 和 `<VR>` 两个路由 token
- **总参数**: ~6B（VideoDiT 5B + 文本推理 4B，推理时按需激活）

### 核心模块

#### 模块 1: 多模态推理数据标注 Pipeline

**设计动机**: 端到端训练需要监督信号来指导路由器何时触发哪种推理，但人工标注成本极高，需要自动化标注。

**轨迹引导子任务标注**:
- 解析末端执行器运动、夹爪开合状态、运动模式等机器人状态轨迹
- 为每个预定义子任务生成候选时间窗口
- 使用 [[Qwen3-VL]] 8B 作为语义验证器，确认子任务完成的时刻
- 施加单调时序约束确保子任务序列符合物理逻辑

**运动驱动精细操作标注**:
- 将机器人状态转化为运动模式特征：位移、姿态、局部变化、夹爪活动度
- 区分精细操作（抓取、释放、微调）与粗糙运动
- 对检测到的区间做时序细化以去噪

#### 模块 2: 动态路由器

**设计动机**: 需要一个轻量级决策模块，能根据当前执行上下文决定是否激活代价高昂的文本/视觉推理模块，在效率与精度间取得最优平衡。

**具体实现**:
- 路由器输入上下文 $C_t = [v_t \| e_l \| e_{ct}]$（当前视觉帧 + 语言嵌入 + 当前文本条件嵌入）
- 独立输出两个二值 token：`<TR>` 控制是否触发文本推理，`<VR>` 控制是否触发视觉推理
- 使用 [[BCE 损失|二元交叉熵损失]] 训练，标注来自标注 pipeline

#### 模块 3: 视频-动作扩散 Transformer（核心生成模型）

**设计动机**: 以 [[Flow Matching|流匹配]] 目标联合训练视觉世界模型和动作策略，使动作生成能以预测的未来视觉状态为条件。

**具体实现**:
- **VideoDiT** $\mathcal{M}_\theta$: 在潜在空间中预测未来视觉潜变量 $\tilde{z}_f$
- **ActionDiT** $\pi_\phi$: 基于集合状态 $\mathcal{S}_t = \{z_{\leq t}, \tilde{c}_t, \tilde{z}_f\}$ 生成动作块 $a_{t:t+H}$
- 两个模块共享联合训练目标，互相条件化

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}\left[\left\|v_\theta(z_\tau, \ldots) - (z_1 - z_0)\right\|^2 + \left\|v_\phi(a_\tau, \ldots) - (a_1 - a_0)\right\|^2\right]
$$

**含义**: 对视频潜变量 $z$ 和动作 $a$ 同时施加流匹配损失，联合优化世界模型和动作策略。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 插值时间步，均匀采样
- $z_\tau = \tau z_1 + (1-\tau) z_0$: 噪声插值视频潜变量（$z_0$ 为噪声，$z_1$ 为真实）
- $a_\tau = \tau a_1 + (1-\tau) a_0$: 噪声插值动作（$a_0$ 为噪声，$a_1$ 为真实）
- $v_\theta$: VideoDiT 预测的速度场
- $v_\phi$: ActionDiT 预测的速度场

### 公式 2: [[MoE|联合训练损失]]（带路由器监督）

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \sum_{k \in \{\text{text}, \text{video}\}} \mathcal{L}_{\text{BCE}}(y_k, \hat{y}_k)
$$

**含义**: 在流匹配损失基础上加入路由器的二元分类损失，$\lambda$ 控制路由监督权重。

**符号说明**:
- $\mathcal{L}_{\text{FM}}$: 视频+动作联合流匹配损失
- $\lambda$: 路由器损失权重系数
- $\mathcal{L}_{\text{BCE}}$: 二元交叉熵损失
- $y_k$: 路由标签（来自标注 pipeline，$k \in \{\text{text}, \text{video}\}$）
- $\hat{y}_k$: 路由器预测的路由概率

### 公式 3: 测试时文本推理（Eq. 3）

$$
\tilde{c}_t = \begin{cases} \mathcal{V}_\omega(z_{\leq t},\, l) & \text{if } \langle TR \rangle = 1 \\ c_t & \text{if } \langle TR \rangle = 0 \end{cases}
$$

**含义**: 当路由器判断处于子任务转换点时，激活文本推理 VLM 更新语言条件嵌入；否则沿用上一时间步的条件嵌入。

**符号说明**:
- $\tilde{c}_t$: 更新后的文本条件嵌入
- $\mathcal{V}_\omega$: Qwen3-VL-4B 文本推理模块
- $c_t$: 上一时间步的文本条件嵌入
- $\langle TR \rangle$: 路由器输出的文本推理 token（0/1）

### 公式 4: 测试时视觉推理（Eq. 4）

$$
\tilde{z}_f = \begin{cases} \mathcal{M}_\theta(z_{\leq t},\, \tilde{c}_t) & \text{if } \langle VR \rangle = 1 \\ \emptyset & \text{if } \langle VR \rangle = 0 \end{cases}
$$

**含义**: 当路由器判断处于需要视觉想象的精细操作阶段时，激活 VideoDiT 预测未来帧；否则不进行视频预测（动作直接以 $\emptyset$ 作为视觉输入）。

**符号说明**:
- $\tilde{z}_f$: 预测的未来视觉潜变量
- $\mathcal{M}_\theta$: VideoDiT（Wan2.2-5B）世界模型
- $\langle VR \rangle$: 路由器输出的视觉推理 token（0/1）

### 公式 5: [[Action Chunking|动作采样]]

$$
a_{t:t+H} \sim \pi_\phi(\cdot \mid \mathcal{S}_t), \quad \mathcal{S}_t = \{z_{\leq t},\, \tilde{c}_t,\, \tilde{z}_f\}
$$

**含义**: ActionDiT 基于历史视觉观测、更新的文本条件和可选的未来视觉预测，生成长度为 $H$ 的动作块。

**符号说明**:
- $a_{t:t+H}$: 从当前帧 $t$ 起长度为 $H$ 的动作序列（动作块）
- $\mathcal{S}_t$: 完整状态集合，整合三类信息
- $H$: 动作块长度（chunk size）

---

## 关键图表

### Figure 1: 范式对比

![[AdaWAM_fig1.png]]

**说明**: 三种范式对比：(a) 视频-动作联合预测（传统 [[WAM]]，每步都做视觉预测，效率低）；(b) 纯动作预测（[[Fast-WAM]]，无视觉推理，泛化差）；(c) AdaWAM 自适应多模态推理（按需触发文本推理和视觉预测）。核心洞察是不同操作阶段需要不同推理模式。

### Figure 2: 标注 Pipeline 概览

![[AdaWAM_fig2.png]]

**说明**: 自动化数据标注流水线。轨迹线索（运动特征）负责定位精细操作区间和候选子任务窗口，[[Qwen3-VL]] VLM 验证模块进行语义确认并生成子任务标签。输出为路由器训练所需的 `<TR>` / `<VR>` 标注。

### Figure 3: AdaWAM 模型架构

![[AdaWAM_fig3.png]]

**说明**: AdaWAM 完整架构图。动态路由器 $\mathcal{R}_\psi$ 接收上下文 $C_t$，独立输出文本推理和视觉推理的激活信号；文本推理模块 $\mathcal{V}_\omega$（Qwen3-VL-4B）更新语言条件；视觉推理模块 $\mathcal{M}_\theta$（Wan2.2-5B）在需要时预测未来潜变量；动作预测模块 $\pi_\phi$ 最终生成动作块。

### Figure 4: 真实场景任务可视化

![[AdaWAM_fig4.png]]

![[AdaWAM_fig5.png]]

**说明**: 真实机器人平台（[[ALOHA]] 双臂）上的长时程任务（Clean Table）和精细操作任务（Wipe Table）的执行过程可视化。展示了 AdaWAM 在多阶段任务执行中的鲁棒性。

### Figure 5: HangingMug 精细操作案例对比

![[AdaWAM_fig6.png]]

**说明**: 纯动作 WAM（Fast-WAM）与 AdaWAM 在 HangingMug 精细任务上的轨迹质量对比。AdaWAM 在抓取和悬挂阶段激活视觉推理，产生更精准的末端执行器轨迹，而 Fast-WAM 在需要接触的阶段出现明显漂移。

### Figure 6: 推理效率对比

![Figure 6a](https://arxiv.org/html/2606.07089/2606.07089v1/x7.png)

![Figure 6b](https://arxiv.org/html/2606.07089/2606.07089v1/x8.png)

**说明**: 气泡图对比推理时延（x 轴）、成功率（y 轴）与任务时长（气泡大小）。AdaWAM 每步推理时延与 [[Fast-WAM]] 相当，但任务总时长更短（成功率更高、少走弯路）；显著优于需要每步运行完整 CoT 的方法（如 MM-ACT），后者推理延迟极大。

### Table 1: LIBERO 基准测试结果

| 方法 | Spatial | Object | Goal | Long | Overall |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| ACoT-VLA | 98.6 | 99.0 | 99.4 | 97.0 | 98.5 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 88.0 | 95.0 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| AdaWAM w/o V.R. | 97.5 | 99.4 | 96.8 | 96.6 | 97.6 |
| **AdaWAM** | **98.0** | **99.6** | **97.1** | **99.1** | **98.5** |

**关键发现**: AdaWAM 在 LIBERO-Long 上达到 99.1%，超越所有基线（包括同分的 ACoT-VLA 和 LingBot-VA），而 Overall 98.5% 与最佳方法持平。文本推理对长时程任务提升最显著（v.s. AdaWAM w/o V.R. 96.6% → 99.1%）。

### Table 2: RoboTwin 2.0 仿真结果（选取代表性任务）

| 任务 | Fast-WAM | AdaWAM w/o V.R. | AdaWAM |
|------|----------|-----------------|--------|
| HangingMug | 58 | 56 | **59** |
| PickDiverseBottles | 80 | 79 | **87** |
| PutObjectCabinet | 94 | 92 | **96** |
| RotateQRCode | 93 | 95 | 94 |
| Hard SR (整体) | 83.43 | 83.29 | **88.43** |
| Overall SR | 91.88 | 91.31 | **93.11** |

**关键发现**: 视觉推理模块（V.R.）对精细操作任务帮助显著（HangingMug, PickDiverseBottles），Hard SR 从 83.43% 提升到 88.43%（+5 个百分点）。

### Table 3: 真实机器人任务成功率

| 任务 | π0.5 | X-VLA | Motus | FastWAM | AdaWAM w/o V.R. | AdaWAM |
|------|------|-------|-------|---------|-----------------|--------|
| Clean Up Trash | 60 | 60 | 50 | 30 | 50 | **70** |
| Wipe Table Clean | 50 | 20 | 20 | 50 | 60 | **60** |

**关键发现**: AdaWAM 在两个真实任务上均排名第一。值得注意 FastWAM 在垃圾清理任务上仅 30%，而添加文本/视觉推理后提升至 70%，说明自适应推理对真实世界多阶段任务至关重要。

### Table 4: 未见任务组合泛化能力

| 环境 | 任务 | FastWAM | MM-ACT | AdaWAM w/o T.R. | AdaWAM w/o V.R. | AdaWAM |
|------|------|---------|--------|-----------------|-----------------|--------|
| 已见 | soup&cheese | 98 | 84 | 96 | 95 | **98** |
| 已见 | cheese&butter | 93 | 80 | 94 | 92 | **96** |
| 未见 | soup&butter | 0 | 24 | 3 | 38 | **61** |

**关键发现**: 无文本推理的方法（FastWAM, AdaWAM w/o T.R.）在未见任务组合上完全失败（0-3%），说明语言推理是实现组合泛化的关键。AdaWAM 达到 61%，远超仅有视觉推理的 38%，证明两种推理模式的协同效果。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 130 任务套件 | 多子集（Spatial/Object/Goal/Long） | 仿真训练+测试 |
| [[RoboTwin]] 2.0 | 50 任务场景 | 双臂高保真仿真，域随机化 | 仿真测试（7 难任务子集） |
| 真实机器人数据 | 2 任务 | AgileX Split-Type ALOHA 平台 | 真实机器人测试 |

### 实现细节

- **视觉推理骨干**: Wan2.2-5B（VideoDiT）
- **动作预测**: 压缩 ActionDiT，1B 参数，隐藏维度 1024
- **文本推理**: Qwen3-VL-4B
- **优化器**: AdamW
- **Stage 1**（视频+动作联合训练）: 50,000 步，学习率 $3 \times 10^{-5}$，权重衰减 0.005
- **Stage 2**（路由器微调）: 10,000 步，学习率 $1 \times 10^{-5}$，权重衰减 0.01
- **硬件**: 8 × NVIDIA A100（80GB）
- **真实机器人**: PiPER 6-DoF 机械臂，主从遥操作数据采集

### 可视化结果

Figure 5 的案例分析显示，在 HangingMug 任务中，AdaWAM 在接近挂钩阶段自动激活视觉推理，预测出精确的末端路径，而 Fast-WAM 在相同阶段出现方向偏差，最终导致任务失败。Figure 6 进一步证明 AdaWAM 的推理延迟（每步时间）与 Fast-WAM 相当，总时长更短，说明更高的成功率直接减少了重试次数。

---

## 批判性思考

### 优点

1. **思路清晰**: "按需推理"的思路在工程上很务实，不是所有任务都需要视频预测，动态激活是正确的设计方向
2. **标注 pipeline 可扩展**: 基于运动特征 + VLM 验证的自动化标注避免了人工密集标注，方法论上值得借鉴
3. **组合泛化证明价值**: Table 4 的 soup&butter 实验是全文最有说服力的结果，0% → 61% 的跨越说明语言推理是实质性能力提升而非微调过拟合

### 局限性

1. **视觉路由器依赖启发式标注**: 路由器训练标签来自轨迹运动特征的启发式规则，存在噪声；且路由阈值的超参数选择敏感性未讨论
2. **感知限制**: 仅使用 RGB 图像，在遮挡严重或几何复杂的场景（如 RotateQRCode）性能不如 Fast-WAM w/o V.R.，说明视觉预测并非万能
3. **消融不够彻底**: 论文未消融路由器本身的贡献（总是激活 V.R. vs 路由激活）；训练计算量（2 阶段，共 60K 步）对资源需求较高

### 潜在改进方向

1. 用强化学习替代监督学习训练路由器，基于环境反馈自适应优化路由策略
2. 引入触觉/深度等多模态感知信号补充 RGB 的不足
3. 探索 self-supervised 路由器训练，减少对标注 pipeline 的依赖

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（提供了充分的超参数）
- [x] 数据集可获取（LIBERO/RoboTwin 公开）

---

## 关联笔记

### 基于

- [[WAM]]: AdaWAM 是 WAM 范式的扩展，在联合视频-动作建模基础上加入自适应路由
- [[Fast-WAM]]: 主要基线，AdaWAM 以其效率为对标，在精度上大幅超越
- [[Wan2.2]]: 视觉推理模块的骨干视频扩散模型

### 对比

- [[Fast-WAM]]: 纯动作解码 WAM，效率高但语义推理弱
- [[LingBot]]: 语言增强 WAM，固定使用语言推理，无视觉自适应
- [[Flow Matching|CoT-VLA]]: 固定 CoT 推理，延迟大

### 方法相关

- [[Flow Matching]]: AdaWAM 的训练目标，流匹配统一视频与动作生成
- [[Diffusion Transformer]]: VideoDiT 和 ActionDiT 的基础架构
- [[Action Chunking]]: AdaWAM 的动作输出形式，每步预测 $H$ 步动作
- [[CoT]]: 文本推理模块的工作原理，自回归生成下一子任务描述
- [[MoE]]: 动态路由器的设计哲学与 MoE 的条件激活思路相通
- [[Qwen3-VL]]: 文本推理和标注验证模块

### 硬件/数据相关

- [[LIBERO]]: 主要仿真评测基准
- [[RoboTwin]]: 精细操作仿真评测
- [[ALOHA]]: 真实机器人实验平台（AgileX Split-Type）

---

## 速查卡片

> [!summary] AdaWAM: Dreaming when Necessary
> - **核心**: 自适应路由器按需激活文本推理和视觉预测，避免无效"做梦"
> - **方法**: Dynamic Router + Qwen3-VL-4B（文本）+ Wan2.2-5B（视觉）+ 1B ActionDiT，两阶段流匹配训练
> - **结果**: LIBERO-Long 99.1%（SOTA），未见任务组合泛化 61%，推理效率与 Fast-WAM 相当
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-08*
