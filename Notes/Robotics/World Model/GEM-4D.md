---
title: "GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation"
method_name: "GEM-4D"
authors: [Kaichen Zhou, Yuzhen Chen, Fangneng Zhan, Hang Hua, Grace Chen, Xinhai Chang, Ao Qu, Yilun Du, Zhuang Liu, Paul Pu Liang, Mengyu Wang]
year: 2026
venue: arXiv
tags: [world-model, video-generation, robot-manipulation, geometry, flow-matching, inverse-dynamics, 4d-correspondence]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.22882
created: 2026-06-08
---

# 论文笔记：GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Harvard AI and Robotics Lab; Media Lab and EECS, MIT; Computer Science, Princeton University; MIT-IBM Watson AI Lab |
| 日期 | June 2026 |
| 项目主页 | [gem-4d.github.io](https://gem-4d.github.io/) |
| 对比基线 | [[TesserACT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22882) / Code N/A |

---

## 一句话总结

> GEM-4D 通过在训练期间向视频生成骨干注入来自预训练几何基础模型的密集 4D 对应监督，使视频世界模型具备物理一致的帧间对应能力，从而将真实世界机械臂操作成功率从 61% 提升至 81%。

---

## 核心贡献

1. **几何-速度对齐原理（Geometry-Enhanced Velocity Alignment）**: 形式化了几何基础模型表示与帧间对应之间的关系，证明几何监督等价于对视频骨干内部表示施加对应约束。
2. **双流 Flow Matching 架构**: 引入平行的 Geometry DiT，以视频 DiT 的中间特征 $\mathbf{m}_t$ 为条件预测几何速度场；推理时完全丢弃几何分支，实现零额外推理代价。
3. **自适应逆动力学系统（AIDS）**: 无需任务特定训练，将对应一致的视频 rollout 转换为可执行的 6-DoF 末端执行器轨迹，在真实 Droid 任务上超越最强基线 +20 个百分点。

---

## 问题背景

### 要解决的问题

[[视频世界模型]]（Video World Models）能从单条指令生成逼真的未来帧序列，但普遍无法在跨帧间持续追踪同一物理点。生成的视频视觉上合理，却缺乏执行机器人动作所需的物理基础——即**帧间对应一致性**（Inter-frame Correspondence Consistency）。

### 现有方法的局限

- **纯像素 / 隐空间重建损失**（如标准 [[Diffusion Transformer]] 训练）无法约束帧间对应：像素损失可以接近零而对应完全错误（many-to-one 映射）。
- **显式 4D 监督方法**（如 [[TesserACT]]）在输出空间额外预测 RGB、深度、法向量，需要大量几何标注，且没有统一的对应信号。
- **Geometry Forcing** 等表示对齐方法的几何引导能力有限，增量改善不如 GEM-4D。

### 本文的动机

关键观察：**帧间对应的几何因子（深度 $D$、相机旋转 $\mathbf{R}$、平移 $\mathbf{T}$、场景流 $\Delta\mathbf{X}$）已被现代 4D 几何基础模型（如 [[PAGE-4D]]、[[Depth Anything]] V3、[[VGGT]]）的中间特征所编码**。监督视频骨干去预测这些表示，等价于监督它产生对应一致的生成结果，且无需显式对应 loss 或额外输出头。

---

## 方法详解

### 模型架构

GEM-4D 采用**双流 [[Flow Matching]] + [[Diffusion Transformer]]** 架构：

- **输入**: 语言指令 $c$ + 初始帧 $\mathbf{I}_0$（通过 Image VAE 编码）
- **视频骨干（Video DiT）**: [[CogVideoX]] 微调版，[[VAE]] 编码视频隐变量 $\mathbf{z}_0$；中层特征 $\mathbf{m}_t$ 作为对应蒸馏的媒介
- **几何分支（Geometry DiT）**: 以 $\mathbf{m}_t$ 为唯一场景条件，预测几何表示的速度场；**仅在训练时使用，推理时完全丢弃**
- **输出（训练）**: 视频速度 $\mathbf{v}_\theta^{\text{vid}}$ + 几何速度 $\mathbf{v}_\psi^{\text{geo}}$
- **输出（推理）**: 仅视频分支生成未来帧序列 $\{\mathbf{I}_t\}$，零额外推理代价

### 核心模块

#### 模块1: Geometry-Enhanced Velocity Alignment（几何速度对齐）

**设计动机**: 利用 [[PAGE-4D]] / [[VGGT]] 等几何基础模型已内化深度、位姿、运动结构的特性，通过特征对齐将对应一致性蒸馏到视频骨干内部表示。

**具体实现**:
- 冻结几何基础模型 $G$ 对视频序列提取密集几何表示 $\mathbf{g}_0$（见公式4）
- Geometry DiT $\psi$ 只接收视频特征 $\mathbf{m}_t$，无法直接访问像素或相机参数
- 联合损失梯度通过 $\mathbf{m}_t$ 反传到视频骨干参数 $\theta$，迫使 $\mathbf{m}_t$ 编码几何因子（见公式7）

#### 模块2: 自适应逆动力学系统（Adaptive Inverse Dynamic System, AIDS）

**设计动机**: 将对应一致的 rollout 视频转换为可执行机器人轨迹，同时鲁棒处理生成视频中的漂移/崩溃伪影，无需任务特定训练。

**四步流程**:

1. **3D 场景定位（3D Scene Grounding）**: 给定指令、深度图和相机内参，[[Qwen3.5-VL]] + [[SAM2]] 生成目标物体和末端执行器（EE）的 mask；用 [[FoundationPose]] 将 EE CAD 模型对齐到 EE 点云，恢复初始 EE 姿态 $(\mathbf{R}^0_{\text{ee}}, \mathbf{T}^0_{\text{ee}}) \in SE(3)$。

2. **双准则置信门控追踪器（Dual-Criterion Confidence-Gated Tracker）**: 用 [[CoTracker3]] 传播从 EE mask 采样的密集关键点，通过锚点保留率 $s_t$ 和帧间变化 $\Delta s_t$ 区分两类失败模式，分别用重采样或 VLM 重定位处理（见公式8-9）。

3. **几何-运动学姿态后备（Geometry-Kinematics Pose Fallback）**: [[FoundationPose]] 对每帧预测 EE 姿态和置信度 $\kappa_t$；当 $\kappa_t < \kappa^*$ 时，检测位移或旋转跳变并拒绝估计，用深度反投影恢复平移，[[SLERP]] 插值恢复旋转（见公式10-11）。

4. **抓取插入与动作合成（Grasp Insertion & Action Synthesis）**: [[GraspGen]] 在目标点云上预测抓取候选集；按距离参考姿态的加权位姿偏差排序选最优抓取；插入轨迹后平滑，再用逆运动学（IK）转换为关节动作序列（见公式12）。

---

## 关键公式

### 公式1: [[帧间对应|帧间对应投影方程]]

$$
\mathbf{p}_{t+1} \sim \mathbf{K} \left[ \mathbf{R}_{t \rightarrow t+1} \, \mathbf{D}(\mathbf{p}_t) \, \mathbf{K}^{-1} \, \mathbf{p}_t \;+\; \mathbf{T}_{t \rightarrow t+1} \;+\; \Delta\mathbf{X}_t \right]
$$

**含义**: 帧 $t$ 中像素 $\mathbf{p}_t$ 对应的物理点在帧 $t+1$ 的投影位置，由深度、相机运动和场景流共同决定。

**符号说明**:
- $\mathbf{p}_t, \mathbf{p}_{t+1}$: 帧 $t$ 和 $t+1$ 中的像素坐标（齐次）
- $\mathbf{K}$: 相机内参矩阵
- $\mathbf{R}_{t \rightarrow t+1}, \mathbf{T}_{t \rightarrow t+1}$: 帧间相机旋转和平移
- $\mathbf{D}(\mathbf{p}_t)$: 像素 $\mathbf{p}_t$ 处的深度值
- $\Delta\mathbf{X}_t$: 场景流（动态物体的 3D 运动偏移）

### 公式2: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{\mathrm{FM}}^{\text{vid}} = \mathbb{E}_{\mathbf{z}_0, \mathbf{z}_1, t} \left[ \| \mathbf{v}_\theta^{\text{vid}}(\mathbf{z}_t, t, c) - \mathbf{v}^*(\mathbf{z}_t, t) \|_2^2 \right]
$$

**含义**: 训练视频 DiT 的速度场预测目标，最小化预测速度与解析目标速度之差。

**符号说明**:
- $\mathbf{z}_0$: VAE 编码的视频隐变量（干净样本）
- $\mathbf{z}_1 \sim \mathcal{N}(0, \mathbf{I})$: 噪声
- $\mathbf{z}_t$: 时刻 $t$ 的插值隐变量
- $\mathbf{v}_\theta^{\text{vid}}$: 视频 DiT 预测的速度场
- $\mathbf{v}^*$: 解析目标速度

### 公式3: [[Diffusion Transformer|视频 DiT 参数化]]

$$
\mathbf{m}_t = E_\theta^{\text{vid}}(\mathbf{z}_t, t, c), \qquad \mathbf{v}_\theta^{\text{vid}} = U_\theta^{\text{vid}}(\mathbf{m}_t)
$$

**含义**: 视频 DiT 分解为特征提取器 $E$ 和输出头 $U$，中间特征 $\mathbf{m}_t$ 是几何蒸馏的媒介。

**符号说明**:
- $E_\theta^{\text{vid}}$: 视频 DiT 骨干（特征提取部分）
- $U_\theta^{\text{vid}}$: 视频 DiT 输出头（速度预测部分）
- $\mathbf{m}_t$: 中层特征，同时用于几何分支条件

### 公式4: [[几何基础模型|几何表示提取]]

$$
\mathbf{g}_0 = G\!\left(\{\mathbf{I}_t\}_{t=0}^{T}\right) \in \mathbb{R}^{T \times \frac{H}{P} \times \frac{W}{P} \times C}
$$

**含义**: 冻结的几何基础模型 $G$ 从视频序列提取密集几何表示，作为几何分支的学习目标。

**符号说明**:
- $G$: 冻结的几何基础模型（PAGE-4D / VGGT 等）
- $P$: patch size
- $C$: 特征通道数

### 公式5: [[Flow Matching|几何流匹配损失]]

$$
\mathcal{L}_{\mathrm{FM}}^{\text{geo}} = \mathbb{E}_{\mathbf{g}_0, \mathbf{g}_1, t} \left[ \| \mathbf{v}_\psi^{\text{geo}}(\mathbf{g}_t, t, \mathbf{m}_t) - \mathbf{v}_{\text{geo}}^*(\mathbf{g}_t, t) \|_2^2 \right]
$$

**含义**: 训练 Geometry DiT 以 $\mathbf{m}_t$ 为唯一条件预测几何速度场，强制 $\mathbf{m}_t$ 编码完整的几何因子。

**符号说明**:
- $\mathbf{g}_1 \sim \mathcal{N}(0, \mathbf{I})$: 几何噪声
- $\mathbf{v}_\psi^{\text{geo}}$: Geometry DiT 预测的几何速度场
- $\psi$: Geometry DiT 参数

### 公式6: [[联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\mathrm{FM}}^{\text{vid}} + \alpha \cdot \mathcal{L}_{\mathrm{FM}}^{\text{geo}}
$$

**含义**: 联合训练目标，$\alpha$ 平衡外观生成和几何蒸馏。

**符号说明**:
- $\alpha$: 几何损失权重系数

### 公式7: 梯度分解

$$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{\mathrm{FM}}^{\text{vid}} + \alpha \cdot \frac{\partial \mathcal{L}_{\mathrm{FM}}^{\text{geo}}}{\partial \mathbf{m}_t} \cdot \frac{\partial \mathbf{m}_t}{\partial \theta}
$$

**含义**: 几何损失通过链式法则将梯度反传到视频骨干 $\theta$，驱动 $\mathbf{m}_t$ 编码几何结构。

### 公式8: [[点追踪|双准则追踪器指标]]

$$
s_t = \frac{|\mathcal{V}_t|}{|\mathcal{V}_{t_0}|} \in [0, 1], \qquad \Delta s_t = s_t - s_{t-1}
$$

**含义**: $s_t$ 为锚点保留率（渐进漂移的信号），$\Delta s_t$ 为帧间变化（突变崩溃的信号）。

**符号说明**:
- $\mathcal{V}_{t_0}$: 初始帧的锚点关键点集合
- $\mathcal{V}_t$: 帧 $t$ 时仍可靠追踪的锚点子集
- $\tau$: 保留率阈值；$\delta$: 下降阈值

### 公式9: 自适应 EE Mask 更新策略

$$
\hat{\mathcal{M}}_{\text{ee}}^{\,t} = \begin{cases}
\text{re-anchor tracker at } t, & \text{if } s_t < \tau \\
\mathrm{Qwen3.5\text{-}VL}(I_t, c), & \text{if } \Delta s_t < -\delta \\
\mathcal{M}_{\text{ee}}^{\,t}, & \text{otherwise}
\end{cases}
$$

**含义**: 根据两类失败模式分别采用不同修复策略：渐进漂移时重新采样关键点，突变崩溃时用 VLM 重定位 EE mask。

### 公式10: [[FoundationPose|FoundationPose 姿态估计]]

$$
(\mathbf{R}_{\text{ee}}^t, \mathbf{T}_{\text{ee}}^t, \kappa_t) = \mathrm{FoundationPose}(\mathbf{I}_t, \mathbf{D}_t, \mathcal{M}_{\text{ee}}^{\,t}, \mathrm{CAD})
$$

**含义**: 对每帧预测 EE 的 6-DoF 姿态及置信度，用于轨迹恢复。

### 公式11: 时序一致性跳变检测

$$
\|\mathbf{T}_{\text{ee}}^t - \mathbf{T}_{\text{ee}}^{t-1}\|_2 > \epsilon_t, \quad d_{\text{geo}}(\mathbf{R}_{\text{ee}}^t, \mathbf{R}_{\text{ee}}^{t-1}) > \epsilon_R
$$

**含义**: 当 FoundationPose 置信度低时，额外检查位移跳变和旋转跳变；超阈值则拒绝该帧估计并回退到几何后备。

**符号说明**:
- $d_{\text{geo}}(\cdot, \cdot)$: SO(3) 上的测地距离
- $\epsilon_t, \epsilon_R$: 位移和旋转跳变阈值

### 公式12: [[GraspGen|最优抓取选择]]

$$
\mathbf{T}_{\text{grasp}}^* = \arg\min_{\mathbf{T}_{\text{grasp}}^{(i)}} \left( \lambda_t \, \|\mathbf{t}_{\text{grasp}}^{(i)} - \mathbf{t}_{\text{ref}}\|_2 + \lambda_R \, d_{\text{geo}}\!\left(\mathbf{R}_{\text{grasp}}^{(i)}, \mathbf{R}_{\text{ref}}\right) \right)
$$

**含义**: 从 GraspGen 候选集中选择距参考姿态加权位姿偏差最小的抓取作为最终执行姿态。

**符号说明**:
- $\lambda_t, \lambda_R$: 平移和旋转一致性权重
- $(\mathbf{R}_{\text{ref}}, \mathbf{T}_{\text{ref}})$: 轨迹中最接近目标物体的参考姿态

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1 - Overview](https://gem-4d.github.io/Overview/teaser.jpg)

**说明**: GEM-4D 与 [[TesserACT]] 基线的对比。给定指令"Franka Emika Panda Touch the Mouse"，Tesseract（右）生成的视频出现机器人身体形变和几何不一致；GEM-4D（左）保持跨帧的结构一致性，生成的 3D 点云更加连贯。

### Figure 2: GEM-4D 训练架构

![Figure 2 - Architecture](https://arxiv.org/html/2605.22882v3/x2.png)

**说明**: 训练期间，视频 DiT 预测噪声视频隐变量的速度，其中间特征 $\mathbf{m}_t$ 引导 Geometry DiT 预测几何速度。几何分支（蓝色）读取视频特征但不写回，保持视频分支（橙色）单流架构。**推理时只使用视频分支**。

### Figure 3: Adaptive Inverse Dynamic System

![Figure 3 - AIDS](https://gem-4d.github.io/Overview/fig4.jpg)

**说明**: 自适应逆动力学系统的四步流程：3D 场景定位 → 双准则追踪 → 几何-运动学姿态后备 → 抓取插入与动作合成。Grasp Pose 选择示意了"Wrong Grasp Pose"（红）与"Right Grasp Pose"（绿）的对比。

### Figure 4: Generated Frames to Arm Action

**说明**: 从初始观测出发，通过 GEM-4D 预测的未来帧序列，到 UF 机械臂实际执行的动作。展示了系统从 world model rollout 到物理执行的完整闭环。

### Figure 5: 定性 4D 场景生成结果

**说明**: 在 Droid（真实）和 RLBench（合成）数据集上对比 GT、TesserAct 和 GEM-4D 的 RGB + 深度输出。TesserAct 在机械臂区域深度出现翘曲和噪声条纹；GEM-4D 跨帧保持几何结构，深度更干净、物体边界更清晰。

### Figure 6: 真实机器人 Rollout

![Figure 6 - Real Robot](https://gem-4d.github.io/Overview/fig6.jpg)

**说明**: 左至右：GT 视频、GEM-4D 生成 RGB、反投影 3D 点云。GEM-4D 在未见过背景下生成真实且几何一致的 rollout，支持迁移到 UF Arm 执行。

### Table 1: 4D 场景生成定量对比

| 域 | 方法 | FVD↓ | SSIM↑ | PSNR↑ | AbsRel↓ | δ₁↑ | δ₂↑ | Chamfer↓ | δ_avg^vis↑ |
|----|------|------|-------|-------|---------|-----|-----|---------|-----------|
| Real | CogVideoX | 35.56 | 75.91 | 20.18 | 22.33 | 68.32 | 83.17 | 0.2670 | 66.22 |
| Real | Wan 2.2-14B | 33.43 | 76.24 | 20.70 | 21.39 | 71.18 | 84.35 | 0.2349 | 68.18 |
| Real | TesserAct | 33.28 | 75.66 | 20.08 | 22.07 | 66.80 | 82.60 | 0.2630 | 67.14 |
| Real | Geometry-Forcing | 33.17 | 76.12 | 20.53 | 21.96 | 69.74 | 83.83 | 0.2443 | 67.97 |
| Real | **GEM-4D** | **31.82** | **82.05** | **21.11** | **20.13** | **78.19** | **88.21** | **0.2001** | **71.23** |
| Sim | CogVideoX | 40.21 | 75.51 | 20.03 | 15.41 | 70.99 | 92.90 | 0.2913 | 58.32 |
| Sim | Wan 2.2-14B | 49.20 | 73.01 | 19.87 | 17.81 | 67.07 | 90.16 | 0.1762 | 61.99 |
| Sim | TesserAct | 41.97 | 76.72 | 19.71 | 16.02 | 69.26 | 93.03 | 0.1813 | 61.15 |
| Sim | Geometry-Forcing | 34.06 | 77.92 | 19.48 | 15.34 | 68.96 | 92.80 | 0.1488 | 60.84 |
| Sim | **GEM-4D** | **27.94** | **80.27** | **23.36** | **14.11** | **74.13** | **95.01** | **0.0702** | **68.18** |

**说明**: GEM-4D 在真实和合成域均取得全面最优。尤其是合成域的 Chamfer Distance 降低约 60%（0.1813→0.0702），证明几何蒸馏显著提升了 3D 重建一致性。指标标准差约 1-2%（基于 20 次生成）。

### Table 2: 操作任务成功率对比

| 方法 | Droid-AUTOLab | Droid-CLVR | Droid-RAIL | Lift | Numbered Block Put | Rubbish InBin | Reach Target | Lamp On | Pick Up Cup | Slide Block |
|------|--------------|-----------|-----------|------|-------------------|---------------|-------------|---------|------------|------------|
| CogVideoX | 49 | 64 | 39 | - | - | - | - | - | - | - |
| Tesseract | 58 | 65 | 59 | 21 | 0 | 2 | 36 | 49 | 18 | 33 |
| **GEM-4D** | **75** | **83** | **87** | **78** | **75** | **82** | **67** | **81** | **80** | **63** |

**说明**: 真实 Droid 任务由人类评估（15 名参与者平均），RLBench 任务通过在仿真器中重放生成轨迹评估。GEM-4D 在所有任务上均大幅超越基线，RLBench 上成功率 63-82%；Droid AUTOLab/CLVR/RAIL 分别提升 +17/+18/+28 个百分点。

### Table 3: 消融实验（4D 场景生成，真实域）

| 方法 | FVD↓ | SSIM↑ | PSNR↑ | AbsRel↓ | δ₁↑ | δ₂↑ | Chamfer↓ |
|------|------|-------|-------|---------|-----|-----|---------|
| CogVideoX（无几何引导） | 35.56 | 75.91 | 20.18 | 22.33 | 68.32 | 83.17 | 0.2670 |
| Wan 2.2-14B（无几何引导） | 33.43 | 76.24 | 20.70 | 21.39 | 71.18 | 84.35 | 0.2349 |
| GEM-4D (Dep)（深度监督代替几何特征） | 32.91 | 78.58 | 20.75 | 20.89 | 74.60 | 86.67 | 0.2229 |
| GEM-4D (VGGT)（VGGT 几何先验） | 33.68 | 75.89 | 20.64 | 21.73 | 71.03 | 83.80 | 0.2370 |
| **GEM-4D（PAGE-4D，完整）** | **31.82** | **82.05** | **21.11** | **20.13** | **78.19** | **88.21** | **0.2001** |

**关键发现**:
- VGGT 直接用作几何先验略微降低性能——VGGT 主要针对静态/准静态场景训练，与机器人操纵的动态场景需求不匹配。
- 深度监督（GEM-4D(Dep)）有竞争力，验证了几何约束的有效性，但不如完整的 4D 特征蒸馏。
- PAGE-4D 特征蒸馏效果最佳，因为其专门针对动态场景的运动分解。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ManiSkill3 | N/A | GPU 并行仿真 | 训练 |
| RLBench | 780 个未见样本（评估） | 合成仿真，有 GT 深度 | 训练 + 测试 |
| Bridge V2 | N/A | 真实机器人多任务 | 训练 |
| RT-1 | N/A | 大规模真实机器人 | 训练 |
| Droid | 400 个未见样本（评估） | 大规模真实场景，无 GT 深度 | 测试（人类评估） |

### 实现细节

- **视频骨干**: 基于 [[CogVideoX]] 的 [[Diffusion Transformer]]，在训练集上微调
- **几何基础模型**: PAGE-4D（冻结），提供密集 4D 几何表示
- **深度估计**（Droid 评估）: Depth Anything V3（Droid 无 GT 深度，需估计）
- **点追踪**: CoTracker3（真实和合成域均使用）
- **生成采样**: 每个样本生成 20 次，取平均结果报告

### 可视化结果

- GEM-4D 在未见背景下仍能生成真实且几何一致的 rollout
- 生成的 3D 点云反映合理的场景结构，支持向 UF Arm 迁移
- 与 Tesseract 对比，GEM-4D 的深度图更干净，物体边界更清晰，无明显翘曲伪影

---

## 批判性思考

### 优点

1. **零推理代价的几何一致性**: 几何分支只在训练时存在，推理时完全丢弃，实现了"表示级别"的几何约束而非输出空间的几何预测，效率极高。
2. **无需几何标注**: 通过蒸馏几何基础模型的中间特征而非预测显式几何量，避免了大规模深度/法向量标注的需求。
3. **端到端闭环**: 从语言指令到真实机器人执行完整闭环，AIDS 模块无需任务特定训练，通用性强。
4. **全面的消融验证**: 消融实验系统验证了不同几何先验（PAGE-4D vs VGGT）和不同监督方式（特征对齐 vs 深度预测）的影响。

### 局限性

1. **依赖高质量几何基础模型**: PAGE-4D 冻结用作 teacher，若几何基础模型对某类场景泛化差（如强运动模糊、透明物体），蒸馏质量难以保证。
2. **AIDS 中的 VLM 依赖**: 逆动力学系统依赖 Qwen3.5-VL、SAM-2、FoundationPose、GraspGen 等多个外部模型，部署复杂度高，推理延迟可能较大。
3. **单视角限制**: 训练和评估均基于单视角 RGB 输入，多视角或立体输入可能进一步提升几何一致性（作者明确未与多视角方法对比）。
4. **CogVideoX 骨干固定**: 方法原理上与视频骨干无关，但实验仅在 CogVideoX 上验证，对更大规模骨干的有效性未知。

### 潜在改进方向

1. 探索将 GEM-4D 原理迁移到更大规模视频生成模型（如 Wan 2.2-14B），验证可扩展性。
2. 在线几何基础模型更新（联合微调 teacher），避免 teacher-student gap。
3. 将 AIDS 中的多模型流水线压缩为单一端到端模型，降低部署复杂度。

### 可复现性评估

- [ ] 代码开源（项目主页暂无代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节部分完整（数据集、指标、骨干选择）
- [x] 数据集可获取（Droid、RLBench 均公开）

---

## 关联笔记

### 基于

- [[CogVideoX]]: 视频 DiT 骨干，GEM-4D 在其基础上微调
- [[Flow Matching]]: 核心生成框架，视频分支和几何分支均采用 flow matching
- [[PAGE-4D]]: 主要几何 teacher 模型，由同一研究组开发
- [[VGGT]]: 备选几何 teacher，消融实验中表现略差

### 对比

- [[TesserACT]]: 最近的显式 4D 监督 baseline，输出空间包含 RGB+深度+法向量
- [[Geometry Forcing]]: 同类表示对齐方法，效果弱于 GEM-4D

### 方法相关

- [[帧间对应]]: 核心解决目标
- [[Diffusion Transformer]]: 模型架构基础
- [[CoTracker3]]: AIDS 追踪器组件
- [[FoundationPose]]: AIDS 姿态估计组件
- [[SAM2]]: AIDS 分割组件
- [[GraspGen]]: AIDS 抓取生成组件
- [[SLERP]]: 旋转插值后备策略
- [[Depth Anything]]: 真实域深度估计

### 硬件/数据相关

- [[Droid]]: 真实世界评估数据集
- [[RLBench]]: 仿真评估数据集
- [[ManiSkill3]]: 仿真训练环境

---

## 速查卡片

> [!summary] GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation
> - **核心**: 通过蒸馏 4D 几何基础模型的中间特征，赋予视频 DiT 帧间对应一致性，零额外推理代价
> - **方法**: 双流 Flow Matching（Video DiT + Geometry DiT），推理时丢弃几何分支；自适应逆动力学系统完成视频→动作转换
> - **结果**: 真实 Droid 操作成功率 61%→81%（+20pp），RLBench 仿真 63-82%；4D 场景生成全面超越 TesserACT
> - **代码**: [gem-4d.github.io](https://gem-4d.github.io/)

---

*笔记创建时间: 2026-06-08*
