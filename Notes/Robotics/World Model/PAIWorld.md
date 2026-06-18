---
title: "PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation"
method_name: "PAIWorld"
authors: [Yuhang Huang, Jiazhao Zhang, Xuan Lv, Junyan Xu, Zhiyuan Yu, Ruizhen Hu, Kai Xu, Wancheng Feng, Shilong Zou, Hewen Xiao, Ziqiao Zhou, Kaiyun Huang, Zhiyu Peng, Juzhan Xu, Hang Zhao, Chenyang Zhu, Renjiao Yi, Yifei Huang, Douhui Wu, Yan Zhang, Kexu Cheng, Chunhe Song, Yunzhi Xue, Xiuhong Zhang, Leitao Guo, Yunji Chen, Bin Wu, Haibin Yu]
year: 2026
venue: arXiv
tags: [world-model, 3d-consistency, multi-view, robotic-manipulation, diffusion-transformer, geometric-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.18375
created: 2026-06-18
---

# 论文笔记：PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Institute of AI for Industries, Chinese Academy of Sciences |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[Cosmos Predict2]]、[[Genie-Envisioner]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.18375) |

---

## 一句话总结

> PAIWorld 通过几何感知跨视图注意力与潜在 3D 特征蒸馏两大技术柱，在机器人操作世界模型中实现多视图 3D 一致性，在 WorldArena 排名第一。

---

## 核心贡献

1. **Geometry-Aware Cross-View Attention**: 引入 [[Geometric RoPE|Geo-RoPE]] 编码相机几何，在 [[Diffusion Transformer]] 层间添加跨视图注意力块，实现视角间显式信息交互
2. **Latent 3D-REPA**: 使用冻结的 [[Depth Anything 3]] 编码器进行 token 关系蒸馏，提供 3D 几何监督信号，避免模型退化为纹理复制捷径
3. **超加性协同效应**: 两组件联合使用带来 MEt3R 改善 2.64，远超各自独立贡献之和（0.93 + 0.72），验证了架构与训练目标须同步改进

---

## 问题背景

### 要解决的问题

当前面向机器人操作的世界模型在多视图生成时缺乏 **3D 一致性**，导致跨相机视角的物体漂移（object drift）和纹理错位（texture misalignment），无法满足操作任务对空间准确性的高要求。

### 现有方法的局限

- **单视图 WFM（World Foundation Models）**: 无法处理多摄像头机器人平台
- **多视图 flat concatenation**: 简单地将多视图 token 沿序列维度拼接后送入标准 [[Diffusion Transformer]]，**没有专用的视角间信息交换通路**
- **两个关键缺陷同时存在**: 1) 缺乏视角间通信，2) 缺乏 3D 几何先验——二者必须同时解决

### 本文的动机

仅有通信通路而无 3D 目标时，模型会退化为"纹理复制捷径"（texture-copying shortcuts）；仅有几何先验而无通信通路时，单视图几何信号无法在视角间传播。因此作者设计了互相强化的两组件：Geo-RoPE 建立共享 3D 坐标系，[[Latent 3D-REPA]] 确保交换内容具有几何含义。

---

## 方法详解

### 模型架构

PAIWorld 采用 **[[Flow Matching]] DiT Backbone** 架构，基于 [[Cosmos Predict2]].5 构建，在预训练视频 [[VAE]] 的潜在空间中运行：

- **输入**: 多视图历史帧 $\{I^v_{1:t_0}\}_{v=1}^V$ + 相机参数 $\{K^v, R^v, t^v\}$ + 条件信号 $c$（动作图或文本）
- **Backbone**: ~14B 参数的 [[Diffusion Transformer|DiT]] 模型
- **核心模块**: [[Geometry-Aware Cross-View Attention]] + [[Latent 3D-REPA|Latent 3D Relation Alignment]]
- **输出**: 未来多视图帧 $\{I^v_{t_0+1:T}\}_{v=1}^V$
- **总参数**: 约 14B

#### 动作条件编码

与传统抽象动作向量不同，PAIWorld 将机器人动作渲染为**动作图（action maps）**——在空间上显式可视化末端执行器轨迹，并拼接至输入帧，避免了动作表示维度失匹配问题。

### 核心模块

#### 模块 1: Geometric Rotary Position Embedding (Geo-RoPE)

**设计动机**: 标准 [[RoPE|Rotary Position Encoding]] 只编码 2D 位置，无法感知相机姿态和 3D 空间关系。Geo-RoPE 将 token 的物理相机几何编码进注意力机制。

**具体实现**: 将注意力头拆分为两个子空间，分别编码不同粒度的几何信息：

**Ray 子空间（像素级）**: 每个像素点 $(h, w)$ 的世界坐标系射线方向，捕获视角内空间变化

**Pose 子空间（视图级）**: 12 维相机姿态特征向量，对整个视角均匀，捕获视角间关系

Split-RoPE 拆分计算后拼接，防止两类信号干扰：

$$
\tilde{q}_{\text{ray}} = \text{RoPE}(q_{\text{ray}},\ d^v(h,w)), \quad \tilde{q}_{\text{pose}} = \text{RoPE}(q_{\text{pose}},\ e^v), \quad \tilde{q} = [\tilde{q}_{\text{ray}};\, \tilde{q}_{\text{pose}}]
$$

---

#### 模块 2: Geometry-Aware Cross-View Attention

**设计动机**: 在选定的 DiT 层插入跨视图注意力块，使各视图的 query 经 Geo-RoPE 旋转后，可 attend 到所有其他视图的 key/value，实现几何偏置的视角间通信。

**具体实现**:
- 使用 [[AdaLN-Zero]] 将 gate 初始化为零，保护预训练权重
- 周期性地将视图维度和空间维度一起 flatten，执行跨视图 + 跨空间的联合注意力（处理 $V \cdot H \cdot W$ 个 token）

---

#### 模块 3: Latent 3D-REPA

**设计动机**: 基于 [[REPA]] 框架的改进，使用冻结的 [[Depth Anything 3]] 作为 3D 感知教师，通过 **token 关系蒸馏**（而非特征值蒸馏）避免特征空间不对齐问题。

**具体实现**:
- **Anchor 采样**: 从 $N$ 个 token 中采样 $M$ 个 anchor，每个 anchor 再采样 $K$ 个邻居，将复杂度从 $O(N^2)$ 降至 $O(M \cdot K)$
- **两级蒸馏**: 空间级（帧内 token 关系）+ 时间级（跨帧 token 关系），确保时序一致性

---

## 关键公式

### 公式 1: [[条件视频生成|多视图条件分布]]

$$
p_\theta\!\left(\{I^v_{t_0+1:T}\}_{v=1}^V \;\middle|\; \{I^v_{1:t_0}\}_{v=1}^V,\, \{K^v, R^v, t^v\}_{v=1}^V,\, c\right)
$$

**含义**: PAIWorld 建模的核心概率分布——给定历史多视图帧、相机内外参及条件信号，预测未来多视图帧序列

**符号说明**:
- $I^v_{1:t_0}$: 视图 $v$ 的历史帧（从 1 到条件帧 $t_0$）
- $K^v, R^v, t^v$: 视图 $v$ 的相机内参矩阵、旋转矩阵、平移向量
- $c$: 条件信号（动作图或文本）

### 公式 2: [[Flow Matching|流匹配插值]]

$$
z_s = (1 - s)\,z_0 + s\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: Flow Matching 的线性插值路径，在数据 $z_0$ 和噪声 $\varepsilon$ 之间构造概率流

**符号说明**:
- $s \in [0, 1]$: 流匹配时间步
- $z_0$: 真实数据的潜在表示
- $\varepsilon$: 高斯噪声

### 公式 3: [[Diffusion Loss|扩散训练损失]]

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{s,\varepsilon}\!\left[\left\|\, u_\theta(z_s, s) - (\varepsilon - z_0)\,\right\|_2^2\right]
$$

**含义**: 模型 $u_\theta$ 预测流匹配速度场，使噪声潜变量朝数据方向演化

**符号说明**:
- $u_\theta$: 网络预测的速度场
- $\varepsilon - z_0$: 目标速度（从噪声到数据的方向）

### 公式 4: [[Geometric RoPE|射线方向计算]]

$$
d^v(h, w) = \text{normalize}\!\left((R^v)^\top (K^v)^{-1} \begin{bmatrix} h + 0.5 \\ w + 0.5 \\ 1 \end{bmatrix}\right)
$$

**含义**: 将像素坐标 $(h, w)$ 通过相机内参逆投影，再经相机旋转变换到世界坐标系下的射线方向

**符号说明**:
- $R^v$: 相机旋转矩阵（3×3）
- $K^v$: 相机内参矩阵（3×3）
- $h, w$: 像素行列坐标（加 0.5 取像素中心）

### 公式 5: [[Geometric RoPE|相机姿态特征向量]]

$$
e^v = \left[\text{yaw},\, \text{pitch},\, \text{roll},\, t^v,\, -(R^v)^\top t^v,\, (R^v)^\top e_z\right]
$$

**含义**: 12 维相机姿态编码向量，同时包含旋转（欧拉角）、平移和光轴方向，对整个视角均匀

**符号说明**:
- $\text{yaw, pitch, roll}$: 相机旋转的欧拉角（3 维）
- $t^v$: 相机平移（3 维）
- $-(R^v)^\top t^v$: 相机光心在世界坐标系中的位置（3 维）
- $(R^v)^\top e_z$: 相机光轴方向（3 维）

### 公式 6: [[Split-RoPE|分裂 RoPE 编码]]

$$
\tilde{q}_{\text{ray}} = \text{RoPE}(q_{\text{ray}},\, d^v(h,w)), \quad \tilde{q}_{\text{pose}} = \text{RoPE}(q_{\text{pose}},\, e^v), \quad \tilde{q} = [\tilde{q}_{\text{ray}};\, \tilde{q}_{\text{pose}}]
$$

**含义**: 将 query 向量按头维度拆分为 ray 和 pose 两半，分别施加不同粒度的几何旋转编码后拼接，同理用于 key

**符号说明**:
- $q_{\text{ray}}, q_{\text{pose}}$: query 的 ray 子空间和 pose 子空间部分（各 $d/2$ 维）
- $d^v(h,w)$: 像素级射线方向
- $e^v$: 视图级姿态特征

### 公式 7: [[Cross-View Attention|几何感知跨视图注意力]]

$$
\hat{Z}^v_t = Z^v_t + \text{gate} \cdot \text{softmax}\!\left(\frac{\tilde{Q}^v_t\,[\tilde{K}^1_t;\cdots;\tilde{K}^V_t]^\top}{\sqrt{d}}\right) [V^1_t;\cdots;V^V_t]
$$

**含义**: 视图 $v$ 的 token $Z^v_t$ 通过 softmax 注意力 attend 到所有视图的 key/value，gate 由 [[AdaLN-Zero]] 初始化为 0 保留预训练能力

**符号说明**:
- $\tilde{Q}^v_t, \tilde{K}^v_t$: 经 Geo-RoPE 旋转后的 query 和 key
- $V^v_t$: 视图 $v$ 的 value（不旋转）
- $\text{gate}$: 可学习标量，初始为 0
- $d$: 注意力头维度

### 公式 8: [[Token Relation Distillation|采样相似度矩阵]]

$$
S(F)_{i,a} = \frac{f_i^\top f_a}{\|f_i\| \cdot \|f_a\|}, \quad a \in \mathcal{A}
$$

**含义**: 计算 token $i$ 与 anchor 集合 $\mathcal{A}$ 中各 anchor $a$ 之间的余弦相似度，构成关系矩阵，避免特征空间对齐的约束

**符号说明**:
- $f_i$: 第 $i$ 个 token 的特征向量
- $\mathcal{A}$: 采样的 anchor 集合（$|\mathcal{A}| = M$）
- $S(F)_{i,a}$: token $i$ 与 anchor $a$ 的相似度

### 公式 9: [[Latent 3D-REPA|3D REPA 蒸馏损失]]

$$
\mathcal{L}_{\text{REPA}} = \mathcal{L}_{\text{spatial}} + \mathcal{L}_{\text{temporal}}
$$

**含义**: 两级几何监督：空间级约束帧内 token 关系，时间级约束跨帧 token 关系

**符号说明**:
- $\mathcal{L}_{\text{spatial}}$: 帧内（intra-frame）token 关系蒸馏损失
- $\mathcal{L}_{\text{temporal}}$: 跨帧（inter-frame）token 关系蒸馏损失

$$
\mathcal{L}_{\text{spatial}} = \text{SmoothL1}(S^{\text{DiT}}_{\text{intra}},\ S^{\text{DA3}}_{\text{intra}})
$$

$$
\mathcal{L}_{\text{temporal}} = \text{SmoothL1}(S^{\text{DiT}}_{\text{inter}},\ S^{\text{DA3}}_{\text{inter}})
$$

**符号说明**:
- $S^{\text{DiT}}_{\text{intra/inter}}$: DiT 学生模型的帧内/跨帧相似度矩阵
- $S^{\text{DA3}}_{\text{intra/inter}}$: [[Depth Anything 3]] 教师模型的帧内/跨帧相似度矩阵

### 公式 10: [[Training Objective|联合训练目标]]

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \cdot \mathcal{L}_{\text{REPA}}
$$

**含义**: 生成质量损失与 3D 几何对齐损失的加权组合，平衡视频生成能力与几何一致性

**符号说明**:
- $\lambda = 0.5$: REPA 权重，平衡生成质量与几何对齐
- $\mathcal{L}_{\text{diff}}$: Flow Matching 速度场预测损失
- $\mathcal{L}_{\text{REPA}}$: 3D token 关系蒸馏损失

---

## 关键图表

### Figure 1: PAIWorld 系统概览

![Figure 1](https://arxiv.org/html/2606.18375/2606.18375v1/x1.png)

**说明**: PAIWorld 在 2.5M 多视图机器人操作视频片段上预训练，作为通用 backbone 支持三类下游应用：基于模型的规划（Model-Based Planning）、动作模型学习（Action Model）、多视图策略训练（Multi-View Policy Training）。

### Figure 2: PAIWorld 框架总览

![Figure 2](https://arxiv.org/html/2606.18375/2606.18375v1/x2.png)

**说明**: 展示两大技术柱——左侧为**视角间通路**（Geometry-Aware Cross-View Attention + Geo-RoPE），右侧为**几何训练目标**（Latent 3D-REPA，使用冻结 [[Depth Anything 3]] 作教师）。两者通过共享 3D 坐标系互相强化。

### Figure 3: WorldArena 定性结果（一）

![Figure 3](https://arxiv.org/html/2606.18375/2606.18375v1/x3.png)

**说明**: 在 WorldArena benchmark 上的未来帧预测结果，展示物理合理的动态效果和跨多视角的稳定场景布局。

### Figure 4: WorldArena 定性结果（二）

![Figure 4](https://arxiv.org/html/2606.18375/2606.18375v1/x4.png)

**说明**: WorldArena benchmark 上更多场景的定性结果，体现跨时间步的时序一致性。

### Figure 5: AgiBot-Challenge2026 定性结果（一）

![Figure 5](https://arxiv.org/html/2606.18375/2606.18375v1/x5.png)

**说明**: 在 AgiBot-Challenge2026 benchmark 上的动作条件生成结果，展示末端执行器轨迹跟踪和操作对象的运动预测。

### Figure 6: AgiBot-Challenge2026 定性结果（二）

![Figure 6](https://arxiv.org/html/2606.18375/2606.18375v1/x6.png)

**说明**: 更多操作任务场景下的动作条件生成定性结果。

### Figure 7: 多视图生成一致性对比

![Figure 7](https://arxiv.org/html/2606.18375/2606.18375v1/x7.png)

**说明**: 与 baseline 的定性对比，展示 PAIWorld 在多视图几何一致性上的优势（无物体漂移和纹理错位）。

---

### Table 1: WorldArena Benchmark 结果

| Method | EWMScore | Visual Quality | Motion Quality | Content Consistency | Physics Adherence | 3D Accuracy | Controllability |
|--------|----------|----------------|----------------|---------------------|-------------------|-------------|-----------------|
| MAI | 66.17 | 63.28 | 62.10 | 59.60 | 66.16 | 94.29 | 60.98 |
| Hoseley | 66.17 | 62.60 | 67.96 | 58.75 | 63.90 | 92.84 | 59.11 |
| MWM | 66.47 | 59.30 | 58.35 | 63.14 | 55.46 | 89.06 | 82.85 |
| Pelican-Unify | 66.61 | 63.60 | 61.73 | 60.41 | 63.98 | 97.65 | 61.77 |
| FlowWAM-FiveAges | 66.87 | 64.07 | 61.70 | 60.20 | 65.17 | 97.78 | 62.05 |
| BetaBWM | 67.00 | 62.85 | 68.83 | 61.36 | 63.78 | 92.04 | 60.39 |
| SparkWorld | 67.29 | 65.04 | 59.71 | 59.59 | 58.59 | 92.54 | 73.76 |
| BWM-Fast | 67.87 | 62.79 | 78.79 | 58.30 | 61.18 | 91.53 | 60.27 |
| GenieEnvisioner-Sim2.0-2B | 68.26 | 60.99 | 62.16 | 60.12 | 66.11 | 95.06 | 73.33 |
| **PAIWorld (Ours)** | **70.67** | 63.05 | **79.66** | 57.97 | 61.20 | 91.58 | 74.40 |

**关键发现**: PAIWorld 在 EWMScore 上以 70.67 排名第一，超过亚军 2.41 分；Motion Quality 79.66 全场最优，体现了跨视图动态建模的优势。

---

### Table 2: AgiBot-Challenge2026 结果

| Team | EWMScore | PSNR | Scene Consistency | nDTW |
|------|----------|------|-------------------|------|
| NeoVerse-ABot | **0.829** | **0.6246** | 0.8974 | **0.9651** |
| Loop | 0.8241 | 0.6207 | 0.9022 | 0.9492 |
| Wild Path | 0.8232 | — | — | — |
| VIPL-GENUN | 0.8195 | — | — | — |
| **PAIWorld (Ours)** | 0.8245 | 0.6161 | **0.9041** | 0.9531 |

**关键发现**: PAIWorld 总分第二（0.8245），但 Scene Consistency 指标（0.9041）全场最优，直接反映了 3D 一致性建模的成效。

---

### Table 3: 文本条件多视图生成（AgiBot-World）

| Method | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Scene Cons. ↑ | Geometric ↓ | MEt3R ↓ |
|--------|--------|---------|-------|-------|---------------|-------------|---------|
| [[Genie-Envisioner]] | 0.7445 | 0.3345 | 83.78 | 207.20 | **0.9231** | 0.5327 | 15.75 |
| [[Cosmos Predict2]] | 0.5870 | 0.3251 | 58.28 | 188.64 | 0.8456 | 0.4824 | 17.47 |
| Wan2.1 | 0.5715 | 0.3354 | 56.47 | 184.22 | 0.8617 | 0.4716 | 16.59 |
| **PAIWorld (Ours)** | **0.7683** | **0.1844** | **45.04** | **175.78** | 0.9041 | **0.4056** | **14.20** |

**关键发现**: PAIWorld 在 SSIM、LPIPS、FID、FVD、Geometric、MEt3R 六项指标全面领先；Scene Consistency 略低于 Genie-Envisioner（0.9041 vs 0.9231），但 3D 几何一致性（MEt3R=14.20）显著更优。

---

### Table 4: 消融实验（AgiBot-World，MEt3R ↓）

| Cross-View Attention | Latent 3D-REPA | SSIM ↑ | LPIPS ↓ | FID ↓ | MEt3R ↓ | ΔREPA |
|----------------------|----------------|--------|---------|-------|---------|-------|
| ✗ | ✗ | 0.6912 | 0.2783 | 53.17 | 16.84 | — |
| ✓ | ✗ | 0.7204 | 0.2361 | 50.02 | 15.91 | 0.93 |
| ✗ | ✓ | 0.7156 | 0.2447 | 49.88 | 16.12 | 0.72 |
| ✓ | ✓ | **0.7683** | **0.1844** | **45.04** | **14.20** | **2.64** |

**关键发现**: 联合使用两组件带来 2.64 的 MEt3R 改善，超过各自单独贡献之和（0.93 + 0.72 = 1.65），验证了**超加性协同**（super-additive synergy）效应。

---

## 实验

### 数据集

| 数据集 | 占比 | 用途 |
|--------|------|------|
| AgiBot-World | 35% | 训练 |
| RoboMIND | 20% | 训练 |
| Galaxea | 15% | 训练 |
| RoboCOIN | 15% | 训练 |
| RoboTwin | 15% | 训练 |

总计：2.5M 多视图机器人操作视频片段

### 实现细节

- **Backbone**: Cosmos-Predict2.5 DiT（~14B 参数）
- **文本编码器**: Cosmos-Reason1
- **优化器**: AdamW
- **峰值学习率**: $3 \times 10^{-5}$（线性 warmup 3000 步 + cosine 衰减）
- **训练步数**: 30,000 次迭代
- **训练时间**: ~7 天
- **硬件**: 200 NVIDIA H200 GPU
- **REPA 权重**: $\lambda = 0.5$
- **Gate 初始化**: AdaLN-Zero（gate = 0）
- **头维度拆分**: $d_r = d_p = d/2$（ray 和 pose 各占一半）
- **冻结教师模型**: Depth Anything 3

### 评估 Benchmark

- **WorldArena**: 动作条件生成，评估 EWMScore（视觉质量、运动质量、内容一致性、物理合理性、3D 精度、可控性的综合得分）
- **AgiBot-Challenge2026**: 动作条件生成，评估 EWMScore、PSNR、Scene Consistency、nDTW
- **AgiBot-World（文本条件）**: 评估 SSIM、LPIPS、FID、FVD、Scene Consistency、Geometric、MEt3R

---

## 批判性思考

### 优点

1. **问题诊断精准**: 清晰指出 multi-view flat concatenation 的两个同步缺陷，逻辑自洽
2. **超加性验证严格**: 消融实验设计规范，2.64 > 0.93 + 0.72 的超加性结果有力支撑了设计动机
3. **工程扩展性**: 基于 [[Cosmos Predict2]].5 这一强力基模，gate 零初始化设计保留了预训练能力，scalable 性好

### 局限性

1. **计算成本极高**: 200 张 H200 GPU 训练 7 天，学术界难以复现
2. **MEt3R 非全场最优不公平**: Table 3 中 Scene Consistency 落后于 Genie-Envisioner，说明 3D 一致性提升和纹理相似度有 tradeoff
3. **物理交互建模缺失**: 作者自述无法处理接触、可形变物体、流体等复杂物理交互
4. **数据集闭源**: 部分训练数据来源（RoboCOIN、Galaxea）不公开，影响可复现性

### 潜在改进方向

1. **物理交互建模**: 引入接触动力学先验，提升对可形变物体和流体的预测能力
2. **长时规划一致性**: 当前只验证了短时 rollout，长时程规划的一致性有待探索
3. **蒸馏效率优化**: Anchor 采样策略的超参 $M, K$ 的自适应选择
4. **轻量化版本**: 类似 Genie-Envisioner-2B 提供小参数量版本，降低使用门槛

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数完整列出）
- [ ] 全部数据集可获取（部分闭源）

---

## 关联笔记

### 基于

- [[Cosmos Predict2]]: PAIWorld 的基础骨干模型（Cosmos-Predict2.5 DiT）
- [[REPA]]: Latent 3D-REPA 的方法基础（表示对齐框架）
- [[Depth Anything 3]]: 用作冻结的 3D 感知教师编码器
- [[Flow Matching]]: PAIWorld 采用的生成范式

### 对比

- [[Genie-Envisioner]]: WorldArena 和 AgiBot-World 上的主要竞争对手
- [[Cosmos Predict2]]: 基础模型，多视图 flat-concat baseline

### 方法相关

- [[RoPE|Rotary Position Encoding]]: Geo-RoPE 的基础位置编码方法
- [[Diffusion Transformer]]: PAIWorld 的核心架构类型
- [[AdaLN-Zero]]: Gate 零初始化策略，保护预训练权重
- [[Cross-View Attention]]: 跨视图注意力机制
- [[Token Relation Distillation]]: Latent 3D-REPA 的核心蒸馏策略

### 数据集相关

- [[AgiBot-World]]: 主要训练数据集（35%），同时作为评估 benchmark
- [[RoboMIND]]: 训练数据集（20%）
- [[WorldArena]]: 评估 benchmark（动作条件生成）

---

## 速查卡片

> [!summary] PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation
> - **核心**: 通过几何感知跨视图注意力 + 3D token 关系蒸馏解决机器人操作世界模型的多视图 3D 一致性问题
> - **方法**: Geo-RoPE 编码相机几何 + Cross-View Attention 打通视角通路 + Latent 3D-REPA 提供几何监督
> - **结果**: WorldArena 第 1（EWMScore 70.67），AgiBot-Challenge2026 第 2（Scene Consistency 最优）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-18*
