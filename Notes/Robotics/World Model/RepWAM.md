---
title: "RepWAM: World Action Modeling with Representation Visual-Action Tokenizers"
method_name: "RepWAM"
authors: [Junke Wang, Qihang Zhang, Shuai Yang, Yiming Luo, Yujun Shen, Zuxuan Wu, Yu-Gang Jiang, Yinghao Xu]
year: 2026
venue: arXiv
tags: [world-action-model, visual-tokenization, latent-action, robot-manipulation, representation-learning, semantic-alignment]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.13674
created: 2026-06-12
---

# 论文笔记：RepWAM: World Action Modeling with Representation Visual-Action Tokenizers

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University, Ant Group |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[π0.5]]、[[Lingbot-VA]]、[[Motus]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13674) |

---

## 一句话总结

> RepWAM 用语义对齐的视觉-动作表征 tokenizer 替换重建导向的 tokenizer，将视觉隐变量与基础模型对齐，同时将潜在动作建模为语义状态间的迁移，大幅提升 World Action Model 的操作性能。

---

## 核心贡献

1. **RepViTok（表征视觉-动作 Tokenizer）**: 在共享语义空间内联合学习视觉潜变量和潜在动作 token，突破重建导向 tokenizer 的表征瓶颈
2. **因果扩散 Transformer + Flow Matching**: 将语言条件下的未来观测生成与对应潜在动作预测联合建模，两阶段训练策略（视频预训练→机器人适配）
3. **无重建导向预训练的强泛化**: 不依赖视频生成预训练骨干（如 WAN），仅凭语义表征在 RoboTwin 2.0 上超越 π₀.₅，说明 tokenizer 设计的核心地位

---

## 问题背景

### 要解决的问题

现有 [[World-Action Model|世界动作模型 (WAM)]] 直接沿用视频生成器的 tokenizer（重建导向），导致两个关键瓶颈：

1. 视觉隐变量侧重像素外观重建，忽视与任务相关的语义（物体身份、交互关系）
2. 动作表征与视觉隐变量处于不同空间，逆向动态模型需要反复跨模态桥接

### 现有方法的局限

- **重建导向 VAE**（如 WAN2.2 VAE）：像素级保真度高，但任务相关语义信息稀薄
- **LAPA 等潜在动作方法**：将视觉变化当动作替代，但动作 token 不在语义视觉空间内
- **联合训练**（joint prediction）：直接在同一阶段预测视频 + 动作，容易优化冲突

### 本文的动机

通过将视觉隐变量与冻结基础视觉模型对齐（特征对齐损失），并将潜在动作建模为语义状态之间的运输算子 + 残差，把感知与控制的模态差距缩至最小，使 [[Inverse Dynamics Model|逆向动态模型]] 解码机器人动作时更容易迁移。

---

## 方法详解

### 模型架构

RepWAM 采用**两阶段因果生成**架构：

- **输入**: 语言指令 $s$ + 观测帧序列 $o_{t:t+k}$
- **Backbone**: [[因果扩散 Transformer]] (Causal DiT) 联合预测视觉 chunk $z_{t:t+k}$ 和潜在动作 chunk $\ell_{t:t+k-1}$
- **核心模块**: [[RepViTok]] 负责视觉-动作 tokenization；[[Flow Matching]] 作为生成目标
- **输出**: 通过 [[Action Expert|动作专家]] 解码机器人连续运动指令 $a_t$
- **总参数**: 1.3B / 5B 两个规模

### 核心模块

#### 模块 1: 表征视觉 Tokenizer（Visual Tokenization）

**设计动机**: 用 [[Feature Alignment|特征对齐]] 将视觉隐变量拉向冻结基础模型，使其携带丰富语义信息

**具体实现**:
- 初始帧采用 $16\times16$ patch；后续帧采用 $4\times16\times16$ spatio-temporal tubelet
- [[Vision Transformer|视觉 Transformer]] 编码器使用时序因果 masking + 帧内全空间注意力
- 解码器通过转置卷积重建像素

**双目标训练**:
1. 重建目标（保持像素保真度）
2. 特征对齐目标（语义化）

---

#### 模块 2: 潜在动作 Tokenizer（Latent Action Tokenization）

**设计动机**: 在已语义对齐的视觉空间内学习动作 token，防止内容泄漏，且动作捕捉的是语义状态间的迁移

**具体实现**:
- [[Inverse Dynamics Model|逆向动态模型 (IDM)]] $q_\phi$：将相邻帧 $(z_t, z_{t+1})$ 压缩为紧凑潜在动作 $\ell_t$
- [[Forward Dynamics Model|前向动态模型 (FDM)]] $f_\psi$：将 $(z_t, \ell_t)$ 映射为运输算子 $K_t$（软空间路由，类比光流）和残差 $\delta_t$
- 运输算子 $K_t$ 作用于语义 token 空间，残差 $\delta_t$ 捕捉无法用运输解释的变化

---

#### 模块 3: 因果扩散 Transformer（Causal World Action Model）

**设计动机**: 以块因果方式对视觉-动作 chunk 进行自回归生成，语言作为条件信号

**具体实现**:
- 语言通过冻结文本编码器嵌入为条件 token
- 视觉与动作 token 共享注意力权重，但使用**模态独立的前馈网络**
- 块因果 masking：当前块只注意过去块
- [[Flow Matching]] 联合回归视觉和动作速度场

---

## 关键公式

### 公式 1: [[Visual Tokenizer|视觉重建目标]]

$$
\mathcal{L}_{rec} = \lambda_1 \|o - \hat{o}\|_1 + \lambda_{perc} \mathcal{L}_{perc}(o, \hat{o}) + \lambda_{gan} \mathcal{L}_{gan}(\hat{o})
$$

**含义**: 视觉自编码器的重建损失，兼顾像素 $\ell_1$ 误差、感知相似性和对抗真实性

**符号说明**:
- $o$: 原始观测帧
- $\hat{o}$: 解码器重建帧
- $\lambda_1, \lambda_{perc}, \lambda_{gan}$: 各项权重系数
- $\mathcal{L}_{perc}$: 感知损失（VGG 特征距离）
- $\mathcal{L}_{gan}$: GAN 对抗损失

---

### 公式 2: [[Feature Alignment|特征对齐损失]]

$$
\mathcal{L}_{align} = \|avg(W_{align}\, z) - avg(G(o))\|_2^2
$$

**含义**: 将视觉隐变量（经线性投影）与冻结基础模型特征拉近，使视觉 token 具备语义表征能力

**符号说明**:
- $z$: 视觉编码器输出的隐变量
- $W_{align}$: 可学习线性投影矩阵
- $G(o)$: 冻结视觉基础模型提取的特征
- $avg(\cdot)$: 序列平均池化

---

### 公式 3: [[Latent Action Tokenizer|潜在动作前向动态]]

$$
\ell_t = q_\phi(z_t, z_{t+1}), \quad K_t, \delta_t = f_\psi(z_t, \ell_t), \quad \hat{z}_{t+1} = K_t z_t + \delta_t
$$

**含义**: IDM 将相邻视觉状态压缩为潜在动作 $\ell_t$；FDM 将其分解为运输算子 $K_t$ 和残差 $\delta_t$，重建下一状态

**符号说明**:
- $q_\phi$: 逆向动态模型（4 层 MLP，hidden 256）
- $f_\psi$: 前向动态模型（4 层 MLP，hidden 256）
- $K_t$: 运输算子，在语义 token 空间做软空间路由
- $\delta_t$: 残差，捕捉无法由运输解释的语义变化
- $\ell_t \in \mathbb{R}^{d_\ell}$，$d_\ell = 4$

---

### 公式 4: [[Latent Action Tokenizer|前向预测与反向一致性损失]]

$$
\mathcal{L}_{fwd} = \sum_t \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \mathcal{L}_{cons} = \sum_t \|\hat{z}_t - z_t\|_2^2
$$

**含义**: 前向预测损失约束 FDM 准确重建下一视觉状态；反向一致性损失防止内容泄漏

**符号说明**:
- $\hat{z}_{t+1}$: FDM 预测的下一帧隐变量
- $z_{t+1}$: 编码器提取的真实下一帧隐变量

---

### 公式 5: [[Flow Matching|Flow Matching 线性插值]]

$$
x_\alpha = (1 - \alpha)\epsilon_{t:t+k} + \alpha\, u_{t:t+k}, \quad \dot{x}_\alpha = u_{t:t+k} - \epsilon_{t:t+k}
$$

**含义**: 条件流匹配框架中的线性插值路径；$\dot{x}_\alpha$ 为目标速度场，模型学习回归该速度

**符号说明**:
- $\alpha \in [0,1]$: 插值参数
- $\epsilon_{t:t+k}$: 高斯噪声
- $u_{t:t+k} = [z_{t:t+k}, \ell_{t:t+k-1}]$: 视觉-动作 chunk（真实数据）

---

### 公式 6: [[Flow Matching|联合流匹配训练损失]]

$$
\mathcal{L}_{FM} = \mathbb{E}\left[\|F_\theta^v(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^v\|_2^2 + \lambda_a \|F_\theta^a(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^a\|_2^2\right]
$$

**含义**: 联合对视觉分支和动作分支的速度场进行回归，$\lambda_a$ 平衡两个模态

**符号说明**:
- $F_\theta^v, F_\theta^a$: 因果 DiT 的视觉/动作预测头
- $s_{<t}$: 语言条件及历史上下文
- $\dot{x}_\alpha^v, \dot{x}_\alpha^a$: 视觉/动作目标速度
- $\lambda_a$: 动作权重系数

---

## 关键图表

### Figure 1: RepViTok 整体架构

![[RepWAM_fig1_overview.png]]

**说明**: 表征视觉-动作 tokenizer 的整体流程。左侧视觉分支将观测帧编码为语义对齐的视觉隐变量（蓝色路径，特征对齐到冻结基础模型）；右侧动作分支通过耦合的 IDM 和 FDM 在共享语义空间内学习潜在动作（绿色路径）。

---

### Figure 2: 真实机器人实验结果对比

![[RepWAM_fig2_realworld.png]]

**说明**: Franka 双臂机器人上 3 个操作任务（pick-the-fruit、push-the-drawer、insert-the-tube）的成功率（10 次 rollout）。RepWAM-5B 在全部任务上超越 π₀.₅ 和 Lingbot-VA；1.3B 模型在长时序任务（push-the-drawer）上有明显差距，体现模型容量对多步推理的重要性。

---

### Figure 3: 真实机器人执行示例

![[RepWAM_fig3_executions.png]]

**说明**: 从左到右依次展示 picking-fruit、pushing-drawer、inserting-tube 三项任务的代表性 rollout 截图，定性验证 RepWAM 在杂乱环境和精细操作中的执行能力。

---

### Figure 4: 潜在动作可视化 & IDM 迁移性对比

![[RepWAM_fig4_latent_action.png]]

**说明**: **左**: LAPA vs RepViTok 的动作隐变量激活可视化——RepViTok 响应集中在操作相关区域（目标物体周围），而 LAPA 响应较分散。**右**: 用冻结动作 latent 的 IDM 损失曲线——RepViTok 达到更低损失，说明其潜在动作更易迁移到机器人动作解码。

---

### Figure 5: 视频 CFG scale 消融

![[RepWAM_fig5_cfg.png]]

**说明**: 在 RoboTwin 2.0 上测试视频 CFG scale（1.0、1.25、2.0）的影响。RepViTok-based WAM 在 CFG scale=1.0（不做无条件外插）时达到最高成功率，表明语言对齐更强，推理无需额外 unconditional 分支，降低延迟和显存。

---

### Figure 6: 重建质量定性示例

![[RepWAM_fig6_reconstruction.png]]

**说明**: ImageNet 和 UCF101 上的图像/视频重建结果。RepViTok 在保留语义细节（人脸可识别、文字清晰、物体边界锐利）和保持视频时序一致性方面均优于对比方案。

---

### Table 1: RoboTwin 2.0 基准测试（50 任务平均）

| 方法 | Easy 成功率 | Hard 成功率 |
|------|------------|------------|
| π₀.₅ | 82.7 | 76.8 |
| Motus | 88.7 | 87.0 |
| Lingbot-VA | **92.9** | **91.6** |
| RepWAM-1.3B | 86.6 | 83.1 |
| **RepWAM-5B** | **89.3** | **88.4** |

**关键发现**: RepWAM-5B 不依赖 WAN 视频生成预训练骨干，仍超越 π₀.₅ 约 7 个点（Easy）/ 12 个点（Hard）；与 Lingbot-VA 的差距主要来自缺少 WAN 预训练，而非 tokenizer 劣势。

---

### Table 2: 语义视觉 Tokenizer 消融（AgiBot 评估集）

| Tokenizer | gFVD↓ (Seen) | gFVD↓ (Unseen) | PSNR↑ (Seen) | PSNR↑ (Unseen) | OLS↑ (Seen) | OLS↑ (Unseen) | Pick-Fruit |
|-----------|-------------|----------------|-------------|----------------|-------------|----------------|------------|
| WAN2.2 VAE | — | — | — | — | 13.68 | 11.21 | 20% |
| ViTok（无语义对齐）| — | — | — | — | — | — | 10% |
| **RepViTok** | **61.01** | **72.91** | **18.47** | **17.72** | **18.82** | **14.15** | **30%** |

**关键发现**: 加入特征对齐后，gFVD、PSNR、Open-Loop Score（OLS）及真实机器人成功率全面提升，语义对齐是核心增益来源。

---

### Table 3: RepViTok 替换 WAN2.2 VAE 在 1.3B WAM 上的效果

| Tokenizer | RoboTwin Easy | RoboTwin Hard |
|-----------|--------------|--------------|
| WAN2.2 VAE | 78.0 | 76.0 |
| **RepViTok** | **86.6** | **83.1** |

**关键发现**: 仅替换 tokenizer，不改变 WAM 架构，RoboTwin 成功率提升约 8 个点（Easy）/ 7 个点（Hard），证明 tokenizer 设计是决定 WAM 性能的关键因素。

---

### Table 4: 潜在动作训练策略消融

| 配置 | gFVD↓ (Seen/Unseen) | PSNR↑ (Seen/Unseen) | OLS↑ (Seen/Unseen) | Pick-Fruit |
|------|---------------------|---------------------|---------------------|------------|
| 无潜在动作 | — / — | — / — | 19.87 / 16.98 基线 | 30% |
| Joint Prediction | — / — | — / — | — / — | 20% |
| **Two-Stage Training** | **48.23 / 58.83** | **22.86 / 19.93** | **19.87 / 16.98** | **50%** |

**关键发现**: 联合预测（joint prediction）因优化冲突性能下滑至 20%；两阶段训练（先视频预训练再机器人适配）显著提升 gFVD 和 PSNR，成功率达 50%，为最优策略。

---

### Table 5: 重建质量对比（ImageNet & UCF101）

| 模型 | ImageNet PSNR@256 | ImageNet SSIM@256 | ImageNet PSNR@512 | ImageNet SSIM@512 |
|------|-------------------|-------------------|-------------------|-------------------|
| 对比方法 | — | — | — | — |
| **RepViTok** | **28.90** | **0.89** | **31.00** | **0.92** |

**关键发现**: RepViTok 在标准图像/视频重建 benchmark 上保持竞争力，证明其在追求语义对齐的同时没有牺牲重建质量。

---

## 实验

### 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| AgiBot | ~100G video-action tokens | 潜在动作预训练 |
| AgiBot + RoboMIND + RoboCOIN + InternA1 | ~300G tokens | 机器人 Embodiment 适配 |
| RoboTwin 2.0 | 50 任务 | 仿真 benchmark 评估 |
| 真实 Franka 双臂 | 50 demos × 3 任务 | 真实机器人评估 |

### 实现细节

- **视觉自编码器**: 12 层 Transformer，hidden 768
- **IDM/FDM**: 4 层 MLP，hidden 256；$d_v=96$，$d_\ell=4$
- **WAM 规模**: 1.3B（30 层，hidden 1536/768）；5B（hidden 3072/768）
- **优化器**: Muon，峰值 lr $1\times10^{-2}$，weight decay 0.01，cosine annealing，bfloat16，gradient clipping 2.0
- **硬件**: 1.3B → 64×H20；5B → 128×H20
- **推理配置**: chunk size 2，attention window 32，5 视频步 × 10 动作步
- **真实机器人微调**: 500 步，lr $1\times10^{-4}$，sequence length 150K

### 可视化结果

- **动作 latent 激活**（Figure 4 左）: RepViTok 的潜在动作响应集中在物体交互区域，LAPA 的响应较分散，定性验证语义空间中的动作表征更具任务相关性
- **重建质量**（Figure 6）: 人脸可识别、边界锐利、文字保真，视频帧间时序连贯

---

## 批判性思考

### 优点

1. **表征空间统一化视角新颖**: 将视觉和动作都拉入同一语义空间，从根本上解决模态对齐问题
2. **两阶段训练策略务实有效**: 先视频预训练捕捉动态模式，再机器人适配，避免联合训练的优化冲突
3. **CFG scale=1.0 推理加速**: 不需要无条件分支，直接降低推理延迟和显存占用，工程价值明显
4. **严格消融设计**: 逐步验证 tokenizer 设计（Table 2→3）和训练策略（Table 4）的独立贡献

### 局限性

1. **缺少 WAN 视频生成预训练**: 与 Lingbot-VA 相比仍有 3-4 个点差距，互联网视频预训练的益处尚未充分利用
2. **真实机器人评估规模有限**: 每任务仅 10 次 rollout、50 个演示样本，统计显著性有限
3. **动作维度 $d_\ell=4$ 较小**: 4 维潜在动作是否能捕捉复杂高自由度操作的全部动态尚未验证

### 潜在改进方向

1. 引入互联网视频（尤其是自我中心人类操作演示）扩大预训练数据
2. 探索更大的潜在动作维度 $d_\ell$ 对灵巧手等高维操作的影响
3. 将 RepViTok 设计迁移到其他 WAM 框架（如 Motus、Being-H0.7）验证通用性

### 可复现性评估

- [ ] 代码开源（承诺公开，尚未发布）
- [ ] 预训练模型（承诺公开，尚未发布）
- [x] 训练细节完整（超参数、硬件、数据来源均已给出）
- [x] 数据集可获取（AgiBot 等开放数据集）

---

## 关联笔记

### 基于

- [[World-Action Model]]: WAM 范式——将操作分解为世界模型预测 + 动作专家两阶段
- [[Latent Action]]: 从观测序列中无监督学习动作变量的核心思路
- [[Flow Matching]]: 生成目标，替代 DDPM 提升训练和推理效率

### 对比

- [[π0.5]]: 所有任务均被 RepWAM-5B 以大幅度超越（+50% 真实机器人）
- [[Lingbot-VA]]: 使用 WAN 视频预训练骨干，RepWAM 在 tokenizer 层面与之比较

### 方法相关

- [[RepViTok]]: 本文提出的核心表征视觉-动作 tokenizer
- [[Inverse Dynamics Model]]: IDM 将相邻帧压缩为潜在动作的核心组件
- [[Forward Dynamics Model]]: FDM 负责从潜在动作重建下一视觉状态
- [[Feature Alignment]]: 语义对齐的核心技术——拉近视觉隐变量与基础模型特征
- [[Causal Diffusion Transformer]]: 块因果自回归生成视觉-动作序列的模型架构

### 硬件/数据相关

- [[AgiBot]]: 主要预训练数据来源（~100G video-action tokens）
- [[RoboTwin 2.0]]: 50 任务仿真 benchmark

---

## 速查卡片

> [!summary] RepWAM: World Action Modeling with Representation Visual-Action Tokenizers
> - **核心**: 用语义对齐的 RepViTok 替换重建导向 tokenizer，统一视觉-动作表征空间
> - **方法**: RepViTok（视觉特征对齐 + IDM/FDM 潜在动作）+ 因果 DiT + 两阶段 Flow Matching 训练
> - **结果**: RoboTwin 2.0 (89.3/88.4)，真实机器人超越 π₀.₅ +50%
> - **代码**: 承诺开源，尚未发布

---

*笔记创建时间: 2026-06-12*
