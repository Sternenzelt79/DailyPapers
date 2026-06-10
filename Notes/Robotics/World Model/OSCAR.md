---
title: "OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics"
method_name: "OSCAR"
authors: [Zhuoyuan Wu, Jun Gao]
year: 2026
venue: arXiv
tags: [world-model, robot-policy-evaluation, cross-embodiment, skeleton-conditioning, video-generation, diffusion-transformer, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.04463
created: 2026-06-04
---

# 论文笔记：OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University / University of Michigan / NVIDIA |
| 日期 | June 2026 |
| 项目主页 | [OSCAR Project Page](https://wuzy2115.github.io/oscar-project-page/) |
| 对比基线 | [[Genie Envisioner]], [[Kinema4D]], [[EnerVerse-AC]], [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04463) |

---

## 一句话总结

> OSCAR 用 2D 骨架渲染作为统一条件表示，在单张 GH200 上训练出跨本体的动作条件视频世界模型，并以此作为机器人策略真实评估的高效代理。

---

## 核心贡献

1. **骨架统一条件表示**: 用 2D 运动学骨架渲染替代 latent action 或 mesh 渲染，自然泛化到机械臂、人形机器人、人手等不同运动链结构
2. **大规模多本体数据流水线**: 从 2.16M 集数据中筛选出 180,657 个高质量片段，覆盖 4 种机器人 + 人手
3. **虚拟策略评估代理**: 在 OSCAR 生成视频上的策略评估与真实机器人测试强相关（Spearman ρ=0.852），可大幅节省真实评估成本

---

## 问题背景

### 要解决的问题

动作条件视频世界模型（[[Action-Conditioned Video World Model]]）需要在给定初始帧和动作序列时预测未来帧，现有方法存在三大痛点：场景多样性不足、动作跟随精度差、跨本体泛化差。

### 现有方法的局限

- **文本条件方法**（如 [[Cosmos-Predict2.5]]）：无法精确跟随连续动作
- **Latent action 方法**（如 [[IRASim]]、[[Ctrl-World]]）：动作空间不对齐，跨本体迁移困难
- **Mesh 渲染方法**（如 [[Kinema4D]]，14B 参数）：对相机标定依赖强，计算开销大
- **已有数据**：单一场景/本体，多样性受限

### 本文的动机

2D 骨架渲染是一种介于"过度简化"（latent action）和"过度复杂"（mesh 表面）之间的表示：它保留了运动学约束（关节连接关系），同时避免了纹理过拟合，且天然适用于不同形态的机械臂和人手（通过 [[MANO]] 手部模型）。

---

## 方法详解

### 模型架构

OSCAR 采用 [[Diffusion Transformer]]（[[Cosmos-Predict2.5]]-2B）作为骨干，通过三个组件实现动作条件视频生成：

- **输入**: 第一帧 $I_0$ + 渲染好的骨架序列 $S_{1:T}$
- **Backbone**: [[Cosmos-Predict2.5]]-2B（[[Rectified Flow]] 目标，WAN 2.1 [[VAE]]）
- **核心模块**: [[骨架条件编码]] → [[条件注入]] → [[DiT]] 去噪
- **输出**: 预测的未来帧视频
- **总参数**: 2B

### 核心模块

#### 模块1：骨架条件编码（Skeleton Condition Encoding）

**设计动机**: 利用 [[正向运动学|Forward Kinematics]] 从关节角度 $q_t$ 和 URDF 模型 $M$ 计算每个关节的 SE(3) 位姿，再投影到图像空间形成 2D 骨架图。

**具体实现**:
- 使用 [[正向运动学]] 计算 $K$ 个连杆的 SE(3) 位姿序列
- 通过相机内外参将 3D 关节点投影到 2D 像素坐标
- 在黑色画布上光栅化骨架（仅连线，无表面纹理）
- 人手部分通过 [[MANO]] 模型的等效运动链完成相同流程

#### 模块2：条件注入（Conditioning Injection）

**设计动机**: 将骨架潜变量与含噪视频潜变量融合，在 [[DiT]] patch 嵌入阶段直接注入条件信息。

**具体实现**:
- 用 WAN 2.1 [[VAE]] 将 $I_0$ 和 $S_{1:T}$ 编码为潜变量
- 将骨架潜变量直接与含噪视频潜变量在通道维度拼接
- 通过改动 [[Patch Embedding]] 层接受额外输入通道

#### 模块3：数据流水线（Four-Stage Data Pipeline）

**阶段1：数据收集**
- 机器人来源：RH20T、InternData-A1、DROID、AgiBot World、AIROA-MoMa（共 1.74M 片段）
- 人手来源：EgoDex、EPIC-Kitchens（共 428K 片段）

**阶段2：过滤**
- 最少 70 帧/片段
- 静态相机约束
- 有效动作检测（排除静止场景）
- 骨架可见度阈值

**阶段3：语义去重**
- 用 [[SigLIP]] 嵌入聚类（相似度阈值 0.95）
- 用 RMS 距离验证轨迹多样性（64 步重采样动作）
- 2.16M → 180,657 片段

**阶段4：文本标注**
- 使用 Qwen3-VL-30B 生成 80-100 词描述

---

## 关键公式

### 公式1：[[正向运动学|Forward Kinematics]]

$$
\{T_{k,t}\}_{k=1}^{K} = \text{FK}(q_t, M)
$$

**含义**: 给定关节角度 $q_t$ 和 URDF 运动学模型 $M$，正向运动学输出 $K$ 个连杆的 SE(3) 位姿

**符号说明**:
- $q_t \in \mathbb{R}^{n_q}$: $t$ 时刻的关节角度向量
- $M$: URDF 运动学树（包含连杆和关节参数）
- $T_{k,t} \in SE(3)$: 第 $k$ 个连杆在 $t$ 时刻的位姿矩阵
- $K$: 连杆总数

### 公式2：[[骨架投影|Skeleton Projection]]

$$
(u_{k,t}, v_{k,t}) = \pi\!\left(K_{\text{cam}},\; T^{\text{cam}}_{\text{world}}\, T_{k,t}\, o_k\right)
$$

$$
S_t = \text{Rasterise}\!\left(\{(u_{k,t}, v_{k,t})\}_{k=1}^{K},\; \mathcal{E}(M)\right)
$$

**含义**: 将 3D 关节点投影到 2D 像素坐标，再按运动链边集 $\mathcal{E}(M)$ 光栅化为骨架图像

**符号说明**:
- $K_{\text{cam}}$: 相机内参矩阵
- $T^{\text{cam}}_{\text{world}}$: 世界坐标到相机坐标的外参变换
- $o_k$: 第 $k$ 个关节在本体坐标系中的原点位置
- $\mathcal{E}(M)$: URDF 运动学树的边集（关节连接关系）
- $S_t$: 第 $t$ 帧的 2D 骨架渲染图

### 公式3：[[MANO|Human Hand Extension]]

$$
S^{\text{human}}_t = \text{Rasterise}\!\left(\left\{\pi\!\left(K_{\text{cam}},\; T^{\text{cam}}_{\text{world}}\, T^{\text{MANO}}_{k,t}\, o^{\text{MANO}}_k\right)\right\}_k,\; \mathcal{E}(M^{\text{MANO}})\right)
$$

$$
\{T^{\text{MANO}}_{k,t}\}_k = \text{FK}(q^{\text{MANO}}_t, M^{\text{MANO}})
$$

**含义**: 用 MANO 手部模型的等效运动链，将人手也转化为相同格式的 2D 骨架，实现跨本体统一

**符号说明**:
- $M^{\text{MANO}}$: MANO 手部的运动学模型
- $q^{\text{MANO}}_t$: 手部关节角度参数

### 公式4：[[Rectified Flow]] 训练目标

$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{t,z_0,\epsilon}\left[\| v_\theta(z_t, t, c) - (\epsilon - z_0) \|_2^2\right]
$$

$$
z_t = (1-t)z_0 + t\epsilon
$$

**含义**: 在加权插值路径上预测速度场，引导含噪潜变量从噪声 $\epsilon$ 恢复到干净潜变量 $z_0$

**符号说明**:
- $v_\theta$: DiT 网络预测的速度场
- $z_t$: $t$ 时刻的含噪潜变量（线性插值）
- $z_0$: 干净视频的 VAE 潜变量
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $c$: 条件信号（骨架 + 第一帧 + 文本描述）

---

## 关键图表

### Figure 1：策略评估代理效果

![Figure 1a](https://arxiv.org/html/2606.04463v1/x1.png)

![Figure 1b](https://arxiv.org/html/2606.04463v1/x2.png)

**说明**: 左图对比 OSCAR 生成的 rollout（上）与真实机器人执行（下），三帧均匀采样；右图展示 RoboArena 上 7 个策略的平均成功率，世界模型评估与真实评估强相关（Spearman ρ=0.852）。

### Figure 2：方法总览

![Figure 2](https://arxiv.org/html/2606.04463v1/x3.png)

**说明**: OSCAR 的三个组件——(1) 条件编码：用 [[VAE]] 将首帧 $I_0$ 和骨架序列 $S_{1:T}$ 编码为潜变量；(2) 条件注入：骨架潜变量与含噪视频潜变量在 patch embedder 处融合；(3) 视频生成：[[DiT]] 去噪后用 VAE decoder 解码输出视频。

### Figure 3：8 个训练来源的骨架叠加效果

![Figure 3a](https://arxiv.org/html/2606.04463v1/x4.png)
![Figure 3b](https://arxiv.org/html/2606.04463v1/x5.png)
![Figure 3c](https://arxiv.org/html/2606.04463v1/x6.png)
![Figure 3d](https://arxiv.org/html/2606.04463v1/x7.png)
![Figure 3e](https://arxiv.org/html/2606.04463v1/x8.png)
![Figure 3f](https://arxiv.org/html/2606.04463v1/x9.png)
![Figure 3g](https://arxiv.org/html/2606.04463v1/x10.png)
![Figure 3h](https://arxiv.org/html/2606.04463v1/x11.png)

**说明**: 每个块展示一个数据源的 4 个片段。上排：DROID、RH20T-cfg5、RH20T-cfg7、InternData（机器人）；下排：AgiBot G1、AIROA-MoMa、EgoDex、EPIC-Kitchens（类人/人手）。骨架渲染在所有本体上保持一致的视觉格式。

### Figure 4：定性对比（两个本体）

![Figure 4a](https://arxiv.org/html/2606.04463v1/x14.png)
![Figure 4b](https://arxiv.org/html/2606.04463v1/x15.png)

**说明**: 与 5 个 baseline 对比，OSCAR 在视觉质量和动作精确跟随上均有显著优势。

### Figure 5：定性对比（其余 4 个本体）

![Figure 5a](https://arxiv.org/html/2606.04463v1/x16.png)
![Figure 5b](https://arxiv.org/html/2606.04463v1/x17.png)

**说明**: 跨本体泛化测试，单一模型在 AgiBot G1（人形）、Toyota HSR（移动操作）等不同运动结构上均表现良好。

### Figure 6：AgiBot G1 和 DROID 补充样本

![Figure 6a](https://arxiv.org/html/2606.04463v1/x18.png)
![Figure 6b](https://arxiv.org/html/2606.04463v1/x19.png)
![Figure 6c](https://arxiv.org/html/2606.04463v1/x20.png)
![Figure 6d](https://arxiv.org/html/2606.04463v1/x21.png)

**说明**: 更多定性样本，展示生成视频与真实执行的视觉一致性。

### Figure 7：条件表示消融（每个本体一个样本）

![Figure 7a](https://arxiv.org/html/2606.04463v1/x22.png)
![Figure 7b](https://arxiv.org/html/2606.04463v1/x23.png)
![Figure 7c](https://arxiv.org/html/2606.04463v1/x24.png)
![Figure 7d](https://arxiv.org/html/2606.04463v1/x25.png)
![Figure 7e](https://arxiv.org/html/2606.04463v1/x26.png)
![Figure 7f](https://arxiv.org/html/2606.04463v1/x27.png)

**说明**: 行对比：Ground Truth、骨架条件（规范，加粗）、URDF mesh 渲染、latent action。骨架条件在视觉质量和动作跟随上均优于其他。

### Figure 8：数据组合消融（每个本体一个样本）

![Figure 8a](https://arxiv.org/html/2606.04463v1/x28.png)
![Figure 8b](https://arxiv.org/html/2606.04463v1/x29.png)
![Figure 8c](https://arxiv.org/html/2606.04463v1/x30.png)
![Figure 8d](https://arxiv.org/html/2606.04463v1/x31.png)
![Figure 8e](https://arxiv.org/html/2606.04463v1/x32.png)
![Figure 8f](https://arxiv.org/html/2606.04463v1/x33.png)

**说明**: 行对比：GT、仅机器人数据、+人手（从头训练）、+人手（warm-start，规范，加粗）。Warm-start 策略效果最佳。

### Figure 9：人手定性样本

![Figure 9a](https://arxiv.org/html/2606.04463v1/x34.png)
![Figure 9b](https://arxiv.org/html/2606.04463v1/x35.png)
![Figure 9c](https://arxiv.org/html/2606.04463v1/x36.png)
![Figure 9d](https://arxiv.org/html/2606.04463v1/x37.png)
![Figure 9e](https://arxiv.org/html/2606.04463v1/x38.png)
![Figure 9f](https://arxiv.org/html/2606.04463v1/x39.png)
![Figure 9g](https://arxiv.org/html/2606.04463v1/x40.png)
![Figure 9h](https://arxiv.org/html/2606.04463v1/x41.png)

**说明**: 每个面板将 GT（上）和 OSCAR 输出（下，附 MANO 骨架叠加）并排展示，三个时间步。

### Figure 10：人手补充样本

![Figure 10a](https://arxiv.org/html/2606.04463v1/x42.png)
![Figure 10b](https://arxiv.org/html/2606.04463v1/x43.png)
![Figure 10c](https://arxiv.org/html/2606.04463v1/x44.png)
![Figure 10d](https://arxiv.org/html/2606.04463v1/x45.png)
![Figure 10e](https://arxiv.org/html/2606.04463v1/x46.png)
![Figure 10f](https://arxiv.org/html/2606.04463v1/x47.png)
![Figure 10g](https://arxiv.org/html/2606.04463v1/x48.png)
![Figure 10h](https://arxiv.org/html/2606.04463v1/x49.png)

**说明**: 更多人手生成样本。

### Figure 11：策略评估对比——"把食物放到盘子上"

![Figure 11a](https://arxiv.org/html/2606.04463v1/x50.png)
![Figure 11b](https://arxiv.org/html/2606.04463v1/x51.png)
![Figure 11c](https://arxiv.org/html/2606.04463v1/x52.png)
![Figure 11d](https://arxiv.org/html/2606.04463v1/x53.png)
![Figure 11e](https://arxiv.org/html/2606.04463v1/x54.png)
![Figure 11f](https://arxiv.org/html/2606.04463v1/x55.png)
![Figure 11g](https://arxiv.org/html/2606.04463v1/x56.png)

**说明**: 每条对比 OSCAR rollout（上）与真实机器人视频（下），6 个时间步。任务：把食物放到盘子上。

### Figure 12：策略评估对比——"按手机按钮"

![Figure 12a](https://arxiv.org/html/2606.04463v1/x57.png)
![Figure 12b](https://arxiv.org/html/2606.04463v1/x58.png)
![Figure 12c](https://arxiv.org/html/2606.04463v1/x59.png)
![Figure 12d](https://arxiv.org/html/2606.04463v1/x60.png)
![Figure 12e](https://arxiv.org/html/2606.04463v1/x61.png)
![Figure 12f](https://arxiv.org/html/2606.04463v1/x62.png)
![Figure 12g](https://arxiv.org/html/2606.04463v1/x63.png)

**说明**: 任务：按手机上的按钮。OSCAR 生成的 rollout 与真实机器人动作高度吻合。

### Figure 13：策略评估对比——"把面包移到盘子上"

![Figure 13a](https://arxiv.org/html/2606.04463v1/x64.png)
![Figure 13b](https://arxiv.org/html/2606.04463v1/x65.png)
![Figure 13c](https://arxiv.org/html/2606.04463v1/x66.png)
![Figure 13d](https://arxiv.org/html/2606.04463v1/x67.png)
![Figure 13e](https://arxiv.org/html/2606.04463v1/x68.png)
![Figure 13f](https://arxiv.org/html/2606.04463v1/x69.png)
![Figure 13g](https://arxiv.org/html/2606.04463v1/x70.png)

**说明**: 任务：把面包移到盘子上。

### Table 1：数据统计

| 来源 | 本体 | 公开片段数 | 筛选后片段数 |
|------|------|-----------|-------------|
| RH20T (cfg5) | Franka Panda | 2,241 | 1,261 |
| InternData-A1 | Franka Panda | 630,000 | 2,233 |
| DROID | Franka Panda | 76,000 | 21,904 |
| AgiBot-Beta | AgiBot G1 | 1,003,672 | 65,720 |
| AIROA-MoMa | Toyota HSR | 25,469 | 3,712 |
| EgoDex | 人手 | 338,000 | 78,273 |
| EPIC-Kitchens | 人手 | 89,977 | 7,554 |
| **合计** | | **2,165,359** | **180,657** |

**说明**: 四阶段过滤将数据从 216 万减至 18 万，压缩比约 12:1，保留高质量多样片段。

### Table 2：与 Baseline 定量对比（4 个本体平均）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | tLPIPS↓ | FVD↓ | FID↓ | L2-latent↓ | FPS↑ |
|------|-------|-------|--------|---------|------|------|-----------|------|
| Cosmos-Predict2.5 | 14.78 | 0.563 | 0.370 | 0.022 | 18.01 | 47.59 | 0.435 | 0.292 |
| TesserAct | 16.26 | 0.730 | 0.277 | 0.055 | 24.50 | 51.90 | 0.364 | 0.343 |
| IRASim | 6.48 | 0.088 | 0.909 | 0.606 | 411.42 | 394.10 | 2.453 | 2.330 |
| Ctrl-World | 19.06 | 0.705 | 0.321 | 0.042 | 28.90 | 53.33 | 0.292 | 1.631 |
| EnerVerse-AC | 20.47 | 0.746 | 0.223 | 0.021 | 33.70 | 38.23 | 0.197 | 1.900 |
| Genie Envisioner (2B) | 23.29 | 0.838 | 0.140 | 0.007 | 15.37 | 22.92 | 0.129 | 1.382 |
| Kinema4D (14B) | 17.68 | 0.741 | 0.198 | 0.021 | 17.07 | 37.16 | 0.233 | 0.089 |
| **OSCAR (Ours, 2B)** | **24.24** | **0.846** | **0.094** | **0.015** | **7.08** | **15.07** | **0.096** | **2.214** |

**说明**: OSCAR 在 PSNR、SSIM、LPIPS、FVD、FID、L2-latent 上全面领先，且仅用 2B 参数、单 GPU，比 Kinema4D（14B）快 25 倍。

### Table 3：消融实验

| 消融维度 | 变体 | PSNR↑ | SSIM↑ | LPIPS↓ | tLPIPS↓ | FVD↓ | FID↓ | L2-latent↓ |
|----------|------|-------|-------|--------|---------|------|------|-----------|
| 条件表示 | Latent action | 19.22 | 0.784 | 0.170 | 0.018 | 12.03 | 26.11 | 0.205 |
| 条件表示 | Mesh 渲染 | 23.11 | 0.831 | 0.106 | 0.013 | 7.89 | 16.38 | 0.109 |
| 条件表示 | **骨架（规范）** | **23.48** | **0.832** | **0.106** | **0.015** | **7.69** | **16.37** | **0.117** |
| 数据组合 | +人手（从头训练） | 23.87 | 0.842 | 0.097 | 0.014 | 7.65 | 15.72 | 0.100 |
| 数据组合 | **+人手（warm-start）** | **24.24** | **0.846** | **0.094** | **0.015** | **7.08** | **15.07** | **0.096** |

**关键发现**: 骨架条件优于 latent action（PSNR +4.26 dB），与 mesh 渲染相当但泛化更好；Warm-start 训练策略比从头训练 +人手更优（FVD 7.08 vs 7.65）。

### Table 4：策略评估相关性（RoboArena，65 个 session × 7 个策略）

| 条件表示 | MMRV↓ | Pearson ρ↑ | Spearman r↑ | SISR_Δ↓ |
|----------|-------|-----------|-------------|---------|
| Latent action | 1.429 | +0.643 | +0.867 | 1.98 |
| Mesh 渲染 | 0.714 | +0.679 | +0.781 | 3.04 |
| **骨架** | **0.571** | **+0.750** | **+0.852** | **1.73** |

**说明**: 骨架条件在策略排序相关性（MMRV 最低）和成功率误差（SISR_Δ 最小）上均最优，表明其生成质量最接近真实机器人行为。

---

## 实验

### 数据集

| 数据集 | 规模（公开/筛选） | 本体 | 用途 |
|--------|-----------------|------|------|
| RH20T (cfg5) | 2,241 / 1,261 | Franka Panda | 训练 |
| InternData-A1 | 630,000 / 2,233 | Franka Panda | 训练 |
| DROID | 76,000 / 21,904 | Franka Panda | 训练 + 评估 |
| AgiBot-Beta | 1,003,672 / 65,720 | AgiBot G1 | 训练 |
| AIROA-MoMa | 25,469 / 3,712 | Toyota HSR | 训练 |
| EgoDex | 338,000 / 78,273 | 人手 | 训练 |
| EPIC-Kitchens | 89,977 / 7,554 | 人手 | 训练 |
| RoboArena | 65 session × 7 策略 | Franka Panda | 策略评估测试 |

### 实现细节

- **Backbone**: [[Cosmos-Predict2.5]]-2B（[[Diffusion Transformer]]）
- **VAE**: WAN 2.1（时空编码）
- **条件注入**: Patch embedder 通道拼接
- **数据增强**: 语义去重（[[SigLIP]] 嵌入，0.95 相似度阈值）
- **标注模型**: Qwen3-VL-30B-A3B-Instruct（80-100 词描述）
- **采样率**: 大多数来源 15fps，DROID 1-2fps
- **硬件**: 单张 NVIDIA GH200 GPU
- **推理速度**: 2.214 FPS（GH200）
- **策略评估**: GPT-5 作为 VLM 评分器，[[Bradley-Terry]] 模型做排名

### 可视化结果

定性对比（Figure 4-6）表明 OSCAR 生成的视频在视觉保真度和动作精确跟随上远超所有 baseline，尤其是在 Franka Panda 和 AgiBot G1 上效果突出。策略评估对比（Figure 11-13）进一步验证世界模型能捕捉策略成功/失败的关键差异。

---

## 批判性思考

### 优点

1. **极致计算效率**: 2B 参数单卡训练，比 Kinema4D（14B）小 7 倍、快 25 倍，降低了世界模型的使用门槛
2. **优雅的跨本体设计**: 骨架表示无需改变模型结构即可统一处理机械臂、类人机器人、人手，设计简洁但效果卓越
3. **端到端验证闭环**: 不仅展示了视频生成质量，还直接验证了策略评估代理的有效性（强相关性），说明了实际应用价值
4. **大规模数据贡献**: 180K+ 多本体过滤数据集是目前最大规模的此类资源

### 局限性

1. **相机标定依赖**: 骨架投影精度完全依赖相机内外参的质量，标定误差会直接降低骨架-RGB 对齐效果，限制了可用训练数据的范围
2. **2B 参数上限**: 骨干网络规模有限，作者也承认"scaling may require more compute"，目前未探索更大模型的潜力
3. **静态相机假设**: 数据过滤要求静态相机，排除了大量移动机器人（移动操作、导航）场景的数据
4. **评估协议成本**: 策略评估需要 65 个真实 session 数据作为基准，初始校准成本仍较高

### 潜在改进方向

1. **自适应相机标定**: 引入在线相机参数估计，放宽对精确预标定的依赖
2. **动态相机支持**: 扩展到移动机器人场景，需要将相机运动从骨架条件中解耦
3. **模型缩放探索**: 尝试 7B/14B 骨架，验证 scaling law 在跨本体世界模型上的适用性

### 可复现性评估

- [ ] 代码开源（项目主页有链接，但代码尚未确认开放）
- [ ] 预训练模型（待确认）
- [x] 训练细节完整（论文中有详细描述）
- [x] 数据集可获取（所有来源均为公开数据集）

---

## 关联笔记

### 基于

- [[Cosmos-Predict2.5]]: 使用其 2B DiT 作为骨干网络，在此基础上微调
- [[MANO]]: 人手骨架渲染的基础模型
- [[Rectified Flow]]: 训练目标的数学框架

### 对比

- [[Genie Envisioner]]: 同为 2B 参数的显式几何条件方法，OSCAR 在 FVD 上从 15.37 优化到 7.08
- [[Kinema4D]]: 14B mesh 渲染方法，OSCAR 用 1/7 参数取得更好效果
- [[EnerVerse-AC]]: latent action 方法代表，骨架条件在 PSNR 上领先 3.77 dB
- [[Ctrl-World]]: latent action 方法，OSCAR PSNR 高出 5.18 dB
- [[IRASim]]: 早期 latent action 方法，OSCAR 全面领先

### 方法相关

- [[Action-Conditioned Video World Model]]: 本文所属的核心任务类别
- [[正向运动学]]: 骨架渲染的数学基础
- [[SigLIP]]: 语义去重中用于视觉嵌入的模型
- [[Diffusion Transformer]]: 视频生成的核心架构
- [[Bradley-Terry]]: 策略评估中用于排名的统计模型

### 硬件/数据相关

- [[DROID]]: 主要评估数据集，来自 Franka Panda 操作
- [[AgiBot World]]: 最大规模训练来源（65 万集数据）
- [[RoboArena]]: 策略评估 benchmark

---

## 速查卡片

> [!summary] OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model
> - **核心**: 用 2D 骨架渲染统一条件，跨本体视频世界模型
> - **方法**: FK → 骨架投影 → DiT 条件注入，180K+ 多本体数据训练
> - **结果**: PSNR 24.24（超越所有 2B/14B baseline），策略评估 Spearman ρ=0.852
> - **代码**: [项目主页](https://wuzy2115.github.io/oscar-project-page/)

---

*笔记创建时间: 2026-06-04*
