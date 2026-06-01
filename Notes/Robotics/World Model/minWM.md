---
title: "minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models"
method_name: "minWM"
authors: [Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, Jun Zhu]
year: 2026
venue: arXiv
tags: [world-model, video-generation, diffusion-distillation, camera-control, autoregressive-generation, real-time-inference, open-source]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.30263
created: 2026-06-01
---

# 论文笔记：minWM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ShengShu / Tsinghua University / Renmin University of China / HKUST / UT-Austin |
| 日期 | May 2026 |
| 项目主页 | [GitHub](https://github.com/shengshu-ai/minWM) |
| 对比基线 | [[WorldPlay]]（HY-WorldPlay）, Matrix-Game |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30263) / [Code](https://github.com/shengshu-ai/minWM) |

---

## 一句话总结

> minWM 是一个全栈开源框架，通过 PRoPE 相机控制微调 + Causal Forcing 蒸馏流水线，将现有双向视频扩散模型转化为可实时交互的相机可控少步自回归世界模型，推理速度提升超 200×。

---

## 核心贡献

1. **全栈开源流水线**: 覆盖数据构建→相机可控微调→AR 训练→蒸馏→流式低延迟推理的完整端到端框架，支持 Wan2.1 和 HY1.5 两种架构
2. **PRoPE 相机注入**: 提出基于射影矩阵的旋转位置编码 [[PRoPE|Projectivity-aware Rotary Position Encoding]]，将相机内外参信息注入 self-attention，实现精确的轨迹跟随生成
3. **Causal Forcing++ 三阶段蒸馏**: 设计 AR 扩散训练 + 因果 ODE/CD 初始化 + 非对称 DMD 后训练的蒸馏链，在保持生成质量的同时实现 223-237× 的首帧延迟降低

---

## 问题背景

### 要解决的问题

现有交互式视频世界模型（如 [[WorldPlay]]）使用双向视频扩散模型，推理时需要对完整视频序列去噪，造成极高延迟，无法满足实时交互需求（游戏仿真、具身 AI 训练环境等）。

### 现有方法的局限

- 双向（bidirectional）扩散模型计算整段视频的注意力，首帧延迟高达数秒
- 多步推理（如 50 步 DDPM）进一步放大延迟
- 闭源系统（如 Matrix-Game）缺乏可复现性，研究社区难以参与改进
- 直接使用感知估计相机位姿（如 SpatialVid）训练存在轨迹不一致问题

### 本文的动机

通过将双向模型转化为因果自回归（[[CausVid|Causal AR]]）模型，并配合 few-step 蒸馏，可以同时解决"等待完整序列"和"多步去噪"两个延迟瓶颈；同时提供完整开源实现降低研究门槛。

---

## 方法详解

### 模型架构

minWM 采用**两阶段训练流水线**架构：

- **阶段一（相机控制微调）**: 在双向视频扩散基础模型上，用 [[PRoPE]] 注入相机参数，学习跟随相机轨迹
- **阶段二（AR 扩散蒸馏）**: 三步蒸馏将双向多步模型转化为因果少步 AR 模型
- **支持模型**: Wan2.1-T2V-1.3B（cross-attention 条件）和 HY1.5-TI2V-8B（[[MM-DiT|MMDiT]] 架构）
- **推理模式**: 流式自回归，每次只处理当前帧，无需未来帧信息

### 核心模块

#### 模块1: PRoPE（Projectivity-aware Rotary Position Encoding）

**设计动机**: 利用[[射影变换|射影矩阵]]的群结构，将相机内参 $K$ 和外参 $T_i^{cw}$ 统一编码进 [[RoPE]] 旋转变换，使相机信息在 self-attention 中以乘法方式自然注入。

**具体实现**:

- 构造提升射影矩阵（lifted projective matrix）$\tilde{P}_i$，融合内参和外参
- 将其嵌入块对角旋转矩阵 $D_t^{PRoPE}$，与标准 2D RoPE 坐标编码组合
- 修改 attention 的 Q/K 计算方式，使注意力得分中包含相机间的射影关系

**与标准 RoPE 对比**: 标准 RoPE 只编码像素坐标 $(x, y)$，PRoPE 额外将相机视角信息编码进旋转相位，使不同相机帧之间的 token 关系具有几何意义。

#### 模块2: Causal Forcing（因果强制训练，Stage 1）

**设计动机**: 通过 teacher forcing 将双向模型转化为[[自回归扩散模型|因果 AR 扩散模型]]，为后续 few-step 蒸馏奠定基础。

**具体实现**:
- 对双向 transformer 施加因果注意力掩码（causal attention masking）
- 使用 ground-truth 历史帧 $x_{<i}^{gt}$ 作为上下文训练当前帧生成
- 将双向多步扩散模型 $p_\theta(\mathbf{x})$ 转化为因果条件分解形式

#### 模块3: 因果初始化（Stage 2，两选项）

**Option A — Causal ODE 初始化**:
- 从噪声中间帧回归到干净帧，通过 ODE 轨迹监督训练 few-step 模型
- 需要离线预先生成 ODE 数据集（overhead 较高）

**Option B — Causal CD 初始化（推荐）**:
- 用[[一致性蒸馏|Consistency Distillation]]替代 ODE，通过对相邻去噪步骤的一致性约束训练
- 无需离线数据策划，训练效率更高
- 用指数移动平均（EMA）参数 $\theta_-$ 作为目标网络

#### 模块4: 非对称 DMD 后训练（Stage 3）

**设计动机**: Causal ODE/CD 初始化后的 few-step AR 模型质量仍有提升空间；[[DMD|非对称 DMD]] 通过让学生模型的自回归 rollout 对齐高质量双向教师分布来进一步提升质量。

**具体实现**:
- 学生模型自回归生成视频帧序列（self-rollout）
- 分别用冻结教师网络（real score）和在线判别网络（fake score）估计得分
- 通过梯度引导对齐学生分布与真实数据分布

---

## 关键公式

### 公式1: [[PRoPE|PRoPE 提升射影矩阵]]

$$
\tilde{P}_i = \begin{bmatrix} K_i & 0 \\ (T_i^{cw})^\top & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

**含义**: 将相机内参矩阵 $K_i$（3×3）和世界到相机的外参变换 $T_i^{cw}$ 融合为齐次坐标下的 4×4 射影矩阵

**符号说明**:
- $K_i$: 第 $i$ 帧的相机内参矩阵
- $T_i^{cw}$: 世界坐标系到相机坐标系的外参变换
- $\tilde{P}_i$: 提升的射影矩阵（lifted projective matrix）

### 公式2: [[PRoPE|PRoPE 块对角旋转矩阵]]

$$
D_t^{PRoPE} = \begin{bmatrix} I_{a/8} \otimes \tilde{P}_{i(t)} & 0 \\ 0 & \begin{bmatrix} RoPE_{a/4}(x_t) & 0 \\ 0 & RoPE_{a/4}(y_t) \end{bmatrix} \end{bmatrix}
$$

**含义**: 将相机射影矩阵（占 head 维度 1/8）与标准 2D RoPE 坐标编码（占 1/4）拼接为块对角旋转矩阵，注入 attention

**符号说明**:
- $a$: attention head 的特征维度
- $\otimes$: Kronecker 积
- $RoPE_{a/4}(x_t), RoPE_{a/4}(y_t)$: 沿 $x$, $y$ 坐标的标准旋转位置编码

### 公式3: [[PRoPE|PRoPE Attention 计算]]

$$
\text{Attn}^{PRoPE}(Q,K,V) = D^{PRoPE} \odot \text{Attn}\!\left((D^{PRoPE})^\top \odot Q,\; (D^{PRoPE})^{-1} \odot K,\; (D^{PRoPE})^{-1} \odot V\right)
$$

**含义**: 将 PRoPE 旋转矩阵以乘法方式应用于 Query、Key、Value，使注意力得分中隐含相机视角间的射影几何关系

**符号说明**:
- $\odot$: 逐元素乘（旋转操作）
- $D^{PRoPE}$: 块对角旋转矩阵

### 公式4: [[CausVid|因果 ODE 损失]]（Stage 2a）

$$
\theta^* = \arg\min_\theta \;\mathbb{E}\!\left[\left\| G_\theta\!\left(\mathbf{x}_t^i,\, \mathbf{x}_{<i}^{gt},\, t\right) - \mathbf{x}_0^i \right\|^2\right]
$$

**含义**: 训练 few-step 生成函数 $G_\theta$，使其从噪声帧 $\mathbf{x}_t^i$ 和历史 GT 帧条件下直接预测干净帧 $\mathbf{x}_0^i$

**符号说明**:
- $G_\theta$: few-step 生成函数（学生模型）
- $\mathbf{x}_t^i$: 第 $i$ 帧在时间步 $t$ 的噪声版本
- $\mathbf{x}_{<i}^{gt}$: 第 $i$ 帧之前的 ground-truth 历史帧
- $\mathbf{x}_0^i$: 第 $i$ 帧的干净目标

### 公式5: [[一致性蒸馏|因果 CD 损失]]（Stage 2b）

$$
\theta^* = \arg\min_\theta \;\mathbb{E}\!\left[w(t)\, d\!\left(G_\theta\!\left(\mathbf{x}_t^i,\, \mathbf{x}_{<i}^{gt},\, t\right),\; G_{\theta_-}\!\left(\hat{\mathbf{x}}_{t-\Delta t}^i,\, \mathbf{x}_{<i}^{gt},\, t-\Delta t\right)\right)\right]
$$

**含义**: 通过一致性约束训练，使相邻去噪步骤的输出一致，无需离线 ODE 数据集

**符号说明**:
- $w(t)$: 时间步权重函数
- $d(\cdot,\cdot)$: 距离函数（如 $\ell_2$）
- $\theta_-$: EMA 参数（停止梯度，作为目标网络）
- $\hat{\mathbf{x}}_{t-\Delta t}^i$: 一步去噪后的中间帧
- $\Delta t$: 相邻时间步间隔

### 公式6: [[DMD|非对称 DMD 梯度]]（Stage 3）

$$
\nabla_\theta \mathbb{E}_t\!\left[D_{KL}\!\left(p_{\theta,t}(\tilde{\mathbf{x}}_t) \,\|\, p_{data,t}(\tilde{\mathbf{x}}_t)\right)\right] = -\mathbb{E}\!\left[\left(s_{real}(\tilde{\mathbf{x}}_t, t) - s_{fake}(\tilde{\mathbf{x}}_t, t)\right) \frac{\partial \tilde{\mathbf{x}}}{\partial \theta}\right]
$$

**含义**: 通过最小化学生 rollout 分布与真实数据分布的 KL 散度，用真实/伪造得分估计之差作为梯度信号对齐质量

**符号说明**:
- $\tilde{\mathbf{x}}_t$: 学生模型自回归 rollout 得到的视频帧（加噪后）
- $s_{real}$: 冻结真实扩散模型的得分估计（real score）
- $s_{fake}$: 在线训练的判别网络得分估计（fake score）
- $p_{\theta,t}$: 学生模型在时间步 $t$ 的边际分布
- $p_{data,t}$: 真实数据在时间步 $t$ 的边际分布

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.30263v1/x1.png)

**说明**: minWM 的完整流水线概览。从左到右：数据构建（DL3DV 三维重建 + WorldPlay 生成）→ PRoPE 相机可控微调 → Causal Forcing++ 三阶段蒸馏 → 流式低延迟推理。输出为可交互的相机可控视频世界模型。

### Figure 2: 蒸馏后相机可控生成效果

![Figure 2](https://arxiv.org/html/2605.30263v1/x2.png)

**说明**: 蒸馏后 few-step AR 模型在不同相机动作下的生成结果，验证 Causal Forcing++ 蒸馏有效保留了基础模型的相机可控性，生成视频在不同轨迹下均保持视觉一致性。

### Figure 3: 训练数据对相机可控生成的影响

#### Figure 3(a): SpatialVid 直接训练

![Figure 3a](https://arxiv.org/html/2605.30263v1/x3.png)

#### Figure 3(b): 3D 重建+重渲染数据训练

![Figure 3b](https://arxiv.org/html/2605.30263v1/x4.png)

#### Figure 3(c): WorldPlay 生成数据训练

![Figure 3c](https://arxiv.org/html/2605.30263v1/x5.png)

**说明**: 直接使用 SpatialVid（感知估计位姿）训练无法实现可靠的相机控制（图 a）；使用 3D 重建得到的精确 GT 轨迹训练可学到有效相机可控性（图 b）；[[WorldPlay]] 生成的视频同样提供精确轨迹，效果类似（图 c）。**关键发现：ground-truth 相机位姿对训练至关重要**。

### Figure 4: 训练步数对相机可控性的影响（HY1.5）

#### Figure 4(a): 1k-2k 步

![Figure 4a](https://arxiv.org/html/2605.30263v1/x6.png)

#### Figure 4(b): 5k 步

![Figure 4b](https://arxiv.org/html/2605.30263v1/x7.png)

#### Figure 4(c): 8k 步

![Figure 4c](https://arxiv.org/html/2605.30263v1/x8.png)

**说明**: 以 HY1.5 为例，相机可控性随训练步数渐进涌现：1k-2k 步基本不可控（图 a）；5k 步开始获得可控性（图 b）；8k 步达到强可控性（图 c）。

### Figure 5: Batch Size 对相机可控训练的影响（Wan2.1）

#### Figure 5(a): Batch Size < 4

![Figure 5a](https://arxiv.org/html/2605.30263v1/x9.png)

#### Figure 5(b): Batch Size = 8

![Figure 5b](https://arxiv.org/html/2605.30263v1/x10.png)

#### Figure 5(c): Batch Size = 16

![Figure 5c](https://arxiv.org/html/2605.30263v1/x11.png)

**说明**: 以 Wan2.1 为例，batch size 对相机控制训练有决定性影响：<4 通常失败（图 a）；8 有改善但不稳定（图 b）；16 可稳定训练并达到高可控性（图 c）。**关键发现：batch size ≥ 16 是稳定训练的必要条件**。

### Table 1: 首帧延迟对比

| 基础模型 | 模型类型 | 首帧延迟 (s) | 相对多步双向的加速比 |
|---------|---------|------------|------------------|
| HY1.5 | Multi-step bidirectional | 1.041 | 1.00× |
| HY1.5 | Multi-step AR | 0.014 | 9.52× |
| HY1.5 | **Few-step AR（Ours）** | **0.0046** | **223.75×** |
| Wan2.1 | Multi-step bidirectional | 0.055 | 1.00× |
| Wan2.1 | Multi-step AR | 0.651 | 9.39× |
| Wan2.1 | **Few-step AR（Ours）** | **0.137** | **236.64×** |

**关键发现**: Few-step AR 相比多步双向基线实现超 200× 加速；HY1.5 绝对延迟仅 4.6ms，Wan2.1 因模型更大为 137ms，但均满足实时交互要求。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DL3DV | 大规模 | 真实场景 3D 重建，GT 相机轨迹 | 阶段一相机控制微调 |
| [[WorldPlay]] 生成数据 | 按需生成 | 由 OpenVid 图像驱动 WorldPlay 生成，精确轨迹 | 阶段一相机控制微调 |
| SpatialVid | 大规模 | 感知估计位姿（实验失败，位姿噪声过大） | 消融对比 |

### 实现细节

**HY1.5-TI2V-8B 配置**:
- Batch size: 32，Learning rate: $1 \times 10^{-5}$
- 阶段一（双向相机微调）: 8K 步
- 阶段二 Stage 1（AR 扩散训练）: 4K 步
- 阶段二 Stage 2（因果 ODE/CD 初始化）: 1.5K 步
- 阶段三（非对称 DMD）: 500 步

**Wan2.1-T2V-1.3B 配置**:
- Batch size: 32，Learning rate: $2 \times 10^{-6}$
- 阶段一（双向相机微调）: 5K 步
- 阶段二 Stage 1: 4K 步
- 阶段二 Stage 2: 2K 步
- 阶段三: 200 步

**硬件**: A800 GPU（推理延迟在单卡 A800 上测量）

### 消融研究关键结论

1. **数据质量 > 数据规模**: GT 轨迹数据（3D 重建 / WorldPlay 生成）显著优于感知估计位姿（SpatialVid）
2. **训练步数门槛**: HY1.5 需要至少 5K 步才开始出现可控性，8K 步达到稳定
3. **Batch Size 关键**: Wan2.1 训练需要 batch size ≥ 16，否则梯度信号不足，无法学到相机控制

---

## 批判性思考

### 优点

1. **全栈开源**: 覆盖数据→训练→推理全链路，是目前少有的交互式视频世界模型开源实现
2. **模块化设计**: 支持灵活替换基础模型（已支持 Wan2.1 和 HY1.5），便于社区扩展
3. **极低推理延迟**: 200×+ 的加速比使实时交互成为可能，具有实际应用价值
4. **详细工程经验**: batch size、训练步数等关键超参数的系统性消融分析对复现有重要价值

### 局限性

1. **SpatialVid 数据问题未解决**: 论文承认当前无法可靠使用感知估计位姿，限制了数据规模扩展
2. **数据构建成本高**: 依赖 3D 重建（DL3DV）或 WorldPlay 生成，无法直接使用互联网视频
3. **缺乏定量质量评估**: 论文主要报告延迟指标，缺少 FID/FVD 等生成质量的系统定量对比
4. **相机控制信号局限**: 目前仅支持相机运动控制，计划中的姿态控制等尚未实现

### 潜在改进方向

1. 研究如何利用感知估计位姿（噪声鲁棒训练），以利用 SpatialVid 等大规模数据集
2. 扩展到更多控制条件（机器人关节姿态、物体交互等）用于具身 AI
3. 结合在线 RL（如 [[RAVEN]] 的方法）解决 AR 推理时的 distribution shift 问题

### 可复现性评估

- [x] 代码开源（https://github.com/shengshu-ai/minWM）
- [ ] 预训练模型（暂未明确提供）
- [x] 训练细节完整（batch size、learning rate、步数均有记录）
- [x] 数据集可获取（DL3DV 公开，WorldPlay 需自行生成）

---

## 关联笔记

### 基于

- [[CausVid]]: 因果 AR 视频扩散蒸馏框架，minWM 的 Causal Forcing 思路来源
- [[DMD]]: 分布匹配蒸馏，Stage 3 非对称 DMD 后训练的基础方法
- [[RoPE]]: 旋转位置编码，PRoPE 的技术基础
- [[WorldPlay]]: 交互式视频世界模型，minWM 的主要对比/对齐系统

### 对比

- [[WorldPlay]]: 闭源交互式世界模型，minWM 提供其开源替代实现
- [[RAVEN]]: 同样基于 CausVid 框架，用在线 RL 解决 distribution gap 问题

### 方法相关

- [[PRoPE]]: 本文提出的核心相机注入方法
- [[一致性蒸馏]]: Causal CD 初始化的基础方法
- [[自回归扩散模型]]: minWM 目标架构

### 硬件/数据相关

- [[DL3DV]]: 3D 重建数据集，提供 GT 相机轨迹

---

## 速查卡片

> [!summary] minWM (2026)
> - **核心**: 将双向视频扩散模型转化为可实时交互的相机可控 AR 世界模型
> - **方法**: PRoPE 相机注入 + Causal Forcing++ 三阶段蒸馏（AR 训练→ODE/CD 初始化→非对称 DMD）
> - **结果**: HY1.5 加速 223×（4.6ms 首帧），Wan2.1 加速 236×
> - **代码**: https://github.com/shengshu-ai/minWM

---

*笔记创建时间: 2026-06-01*
