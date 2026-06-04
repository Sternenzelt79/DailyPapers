---
title: "3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training"
method_name: "3DThinkVLA"
authors: [Jiaxin Shi, Xidong Zhang, Fucai Zhu, Zhe Li, Siyu Zhu, Weihao Yuan]
year: 2026
venue: arXiv
tags: [vla, 3d-spatial-reasoning, knowledge-distillation, robot-manipulation, embodied-ai]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.04436
created: 2026-06-04
---

# 论文笔记：3DThinkVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University、Harbin Institute of Technology、Nanyang Technological University、Fudan University、Nanjing University、Daimon Robotics、Great Bay University |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[OpenVLA-OFT]]、[[SpatialVLA]]、[[Spatial Forcing]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04436) / Code（待公开） |

---

## 一句话总结

> 通过潜在空间的 3D 几何感知 + 在线推理蒸馏，让 VLA 模型在仅用 2D 图像的情况下隐式完成 3D 空间推理，在 LIBERO 上达到 98.7% 平均成功率。

---

## 核心贡献

1. **发现 Prompt-Induced Reasoning Gap**: 揭示动作预测 prompt 会让模型绕过已学习的空间先验，退化为"动作捷径"，导致 3D 推理能力无法迁移到动作预测路径。
2. **三模块联合框架**: 提出潜在 3D 几何感知、在线 3D 推理蒸馏、空间增强动作集成三个互补组件，通过共同训练协同提升 [[VLA|VLA 模型]] 的空间推理能力。
3. **推理阶段零开销**: 推理时只需 2D 图像，不依赖外部 3D 传感器，仅添加轻量适配器（训练开销仅 1.5×）。

---

## 问题背景

### 要解决的问题

[[Vision-Language-Action Model|VLA 模型]] 在真实 3D 环境中操作时，仍依赖 2D 图像，缺乏显式的空间深度理解。现有"注入 3D 信息"方案要么依赖深度传感器/点云，要么在推理阶段产生冗余文本生成，推理效率低。

### 现有方法的局限

- **显式 3D 方法**（如 [[VGGT]] 对齐、深度图注入）：需要额外传感器，部署受限。
- **隐式 3D 方法**（如 [[SpatialVLA]]）：仅靠特征对齐，没有解决推理通道的空间感知缺失。
- **CoT-VLA 等思维链方法**：动作 prompt 引发"推理旁路"问题，文本中间步骤拖慢推理速度。

### 本文的动机

作者发现：用显式 3D 推理 prompt 时模型能正确激活空间先验，但换成动作预测 prompt 后空间先验失效（**prompt-induced reasoning gap**）。因此，他们设计了一个在线蒸馏机制，将 3D 思考能力从"老师路径"（显式推理 prompt）无声地迁移到"学生路径"（动作 prompt），且无需生成任何中间文字。

---

## 方法详解

### 模型架构

3DThinkVLA 以 **Qwen3-VL-2B** 为 VLM 骨干，采用 [[OpenVLA-OFT|OFT 风格]] 动作头，通过三个轻量适配模块增强空间感知能力：

- **输入**: 语言指令 $L$ + 观测图像 $I_t$ + 动作 prompt $\tau_A$
- **Backbone**: [[VLM]] Qwen3-VL-2B
- **核心模块**: [[知识蒸馏|在线推理蒸馏]] + [[潜在空间对齐|几何适配器]]
- **输出**: 连续动作块 $\hat{A}_t$（OFT 动作头）
- **总参数**: ~2B（轻量适配器额外增加）

整体推理流程：

- **Geometry Adapter**（MLP）将 VLM 中间视觉特征 $\mathcal{F}^{\text{Geo}}$ 与 [[VGGT]] 提取的 3D 特征 $\mathcal{F}^{3D}$ 在潜在空间中对齐
- **Reasoning Distillation**：共享 reasoning anchor token，将老师路径（显式 3D 推理 prompt）的空间知识蒸馏到学生路径（动作 prompt）
- **Spatially Augmented Action Integration**：将几何特征 $H_{\text{geo}}^A$ 和推理特征 $H_{\text{reasoning}}^A$ 注入动作头作为层级条件

### 核心模块

#### 模块 1：Latent 3D Geometry Perception（潜在 3D 几何感知）

**设计动机**: 利用 [[VGGT]]（3D 基础模型）提供低层级几何线索，无需推理阶段加载 3D 模型。

**具体实现**:
- MLP geometry adapter 提取 VLM 骨干的中间视觉层特征，记为 $\mathcal{F}^{\text{Geo}}$
- 冻结 VGGT，提取 3D 特征 $\mathcal{F}^{3D}$
- 用[[余弦相似度]]损失 $\mathcal{L}_{\text{geo}}$ 拉近两者在潜在空间的距离
- 推理阶段：VGGT 卸载，adapter 输出 $H_{\text{geo}}^A$ 注入动作头

#### 模块 2：Online 3D Reasoning Distillation（在线 3D 推理蒸馏）

**设计动机**: 使用[[知识蒸馏]]机制，将显式 3D 推理 prompt 激活的空间先验无声迁移到动作 prompt 路径，消除 prompt-induced reasoning gap。

**具体实现**:
- **老师路径（Teacher）**: 使用带 3D 推理 prompt（$L^{\text{teacher}}$）的前向传播，提取 reasoning anchor token 的隐层表示 $H_{\text{teacher}}^R$，并用 stop-gradient（`sg`）防止梯度回传
- **学生路径（Student）**: 同一骨干，使用动作 prompt（$\tau_A$），提取相同位置的 anchor token 表示 $\hat{H}_{\text{student}}^R$
- 通过轻量映射网络 $\mathcal{R}(\cdot)$ 将学生表示变换后，与老师表示用余弦相似度对齐
- 推理阶段：仅用学生路径，无额外文字生成

#### 模块 3：Spatially Augmented Action Integration（空间增强动作集成）

**设计动机**: 将两路空间特征以层级条件方式注入动作头，实现几何感知与推理感知的联合增强。

**具体实现**:
- $H_{\text{geo}}^A$（来自 geometry adapter）提供低层几何条件
- $H_{\text{reasoning}}^A$（来自 reasoning distillation）提供高层空间推理条件
- 与动作 token $H_A$ 求和后送入 OFT 动作头

---

## 关键公式

### 公式 1：[[VLA|VLA 基础动作预测]]

$$
\hat{A}_{t} = \pi_{\Theta}(I_t, L, \tau_A)
$$

**含义**: VLA 模型以图像观测 $I_t$、语言指令 $L$ 和动作 prompt $\tau_A$ 为输入，预测动作 $\hat{A}_t$。

**符号说明**:
- $\hat{A}_t$: 预测的动作块
- $I_t$: 时刻 $t$ 的 2D 图像观测
- $L$: 自然语言任务指令
- $\tau_A$: 动作预测触发 prompt
- $\pi_\Theta$: 参数为 $\Theta$ 的 VLA 策略

---

### 公式 2：[[余弦相似度|几何对齐损失]]

$$
\mathcal{L}_{\text{geo}} = 1 - \mathcal{S}(\mathcal{F}^{3D}, \mathcal{F}^{\text{Geo}})
$$

**含义**: 用余弦相似度最大化 VLM 中间视觉特征与 VGGT 3D 特征的对齐程度。

**符号说明**:
- $\mathcal{S}(\cdot, \cdot)$: 余弦相似度
- $\mathcal{F}^{3D}$: VGGT 提取的 3D 潜在特征
- $\mathcal{F}^{\text{Geo}}$: MLP geometry adapter 输出的几何特征

---

### 公式 3：[[知识蒸馏|老师推理锚点]]

$$
H_{\text{teacher}}^{R} = \text{sg}(f_{\theta}(I_t, L_{\text{task}}, L^{\text{teacher}}, \tau_R))
$$

**含义**: 用带 3D 推理 prompt 的老师路径前向传播提取 reasoning anchor token 表示，stop-gradient 阻断梯度。

**符号说明**:
- $\text{sg}(\cdot)$: stop-gradient 操作
- $f_\theta$: VLM 骨干
- $L^{\text{teacher}}$: 显式 3D 推理 prompt
- $\tau_R$: reasoning anchor token prompt

---

### 公式 4：[[知识蒸馏|学生推理锚点]]

$$
\hat{H}_{\text{student}}^{R} = f_{\theta}(I_t, L_{\text{task}}, \tau_R, L_{\text{action}}, \tau_A)
$$

**含义**: 同一骨干在动作预测路径下提取 reasoning anchor token 的学生表示，目标是对齐老师表示。

**符号说明**:
- $L_{\text{action}}$: 动作预测 prompt（非 3D 推理 prompt）
- 其余符号同上

---

### 公式 5：[[知识蒸馏|推理蒸馏损失]]

$$
\mathcal{L}_{\text{reasoning}} = 1 - \mathcal{S}(H_{\text{teacher}}^{R}, \mathcal{R}(\hat{H}_{\text{student}}^{R}))
$$

**含义**: 通过余弦相似度最大化，将老师路径的 3D 空间推理知识蒸馏到学生路径。

**符号说明**:
- $\mathcal{R}(\cdot)$: 轻量映射网络（用于特征空间适配）
- $\mathcal{S}(\cdot,\cdot)$: 余弦相似度

---

### 公式 6：空间增强动作集成

$$
\hat{A}_{t} = \text{Action Head}(H_A + H_{\text{geo}}^A + H_{\text{reasoning}}^A)
$$

**含义**: 将几何感知特征和推理感知特征作为层级条件叠加在动作 token 上，再通过动作头解码。

**符号说明**:
- $H_A$: 原始动作 token 隐层表示
- $H_{\text{geo}}^A$: geometry adapter 注入的几何条件特征
- $H_{\text{reasoning}}^A$: reasoning distillation 注入的推理条件特征

---

### 公式 7：[[VLA|VLA 步训练目标]]

$$
\mathcal{L}_{\text{vla}} = \mathcal{L}_{\text{action}} + \lambda_a \mathcal{L}_{\text{geo}} + \lambda_d \mathcal{L}_{\text{reasoning}}
$$

**含义**: VLA 步总损失 = 动作预测损失 + 几何对齐损失 + 推理蒸馏损失。

**符号说明**:
- $\mathcal{L}_{\text{action}}$: 动作预测损失（行为克隆）
- $\lambda_a, \lambda_d$: 几何对齐和推理蒸馏的权重系数（均为 0.5）

---

### 公式 8：[[知识蒸馏|VLM 共训练损失]]

$$
\mathcal{L}_{\text{vlm}} = \lambda_{3D} \mathcal{L}_{\text{CE}}
$$

**含义**: VLM 分支的交叉熵损失，用于在 3D 推理数据上保持语言理解能力。

**符号说明**:
- $\lambda_{3D}$: VLM 损失权重（0.1）
- $\mathcal{L}_{\text{CE}}$: 交叉熵损失

---

### 公式 9：总损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vla}} + \mathcal{L}_{\text{vlm}}
$$

**含义**: 整体训练目标，联合优化 VLA 动作预测能力与 VLM 的 3D 空间语言理解能力。

---

## 关键图表

### Figure 1: 框架总览

![Figure 1 Overview](https://arxiv.org/html/2606.04436/2606.04436v1/x1.png)

**说明**: (a) 3DThinkVLA 整体共训练框架：VLA 数据路径与 3D 推理数据路径并行训练，VLM 骨干共享；(b) 识别出的 prompt-induced reasoning gap：显式 3D 推理 prompt 激活了空间先验，但动作 prompt 绕过了这些先验。

---

### Figure 2: 方法架构详图

![Figure 2 Architecture](https://arxiv.org/html/2606.04436/2606.04436v1/x2.png)

**说明**: 展示三个核心模块的连接关系。Geometry Adapter（左下）对齐 VLM 中间特征与 VGGT 特征；Reasoning Distillation（中）通过 teacher-student 共享 anchor token 传递空间知识；两路输出合并注入 OFT 动作头（右）。

---

### Figure 3: LIBERO-Plus 定性结果

![Figure 3a Failure 1](https://arxiv.org/html/2606.04436/2606.04436v1/fig/failure1.png)
![Figure 3b Success 1](https://arxiv.org/html/2606.04436/2606.04436v1/fig/success1.png)
![Figure 3c Failure 2](https://arxiv.org/html/2606.04436/2606.04436v1/fig/failure2.png)
![Figure 3d Success 2](https://arxiv.org/html/2606.04436/2606.04436v1/fig/success2.png)

**说明**: 在 LIBERO-Plus 零样本迁移任务中的成功/失败案例。任务 1："Put the black bowl in the bottom drawer of the cabinet and close it"；任务 2："Put the black bowl in the top drawer of the cabinet and close it"。展示了 3DThinkVLA 对高度变化和透明容器挑战的鲁棒性。

---

### Figure 4: 真实机器人实验平台

![Figure 4a Robot Setup](https://arxiv.org/html/2606.04436/2606.04436v1/fig/robotsetup_a.png)
![Figure 4b Robot Tasks](https://arxiv.org/html/2606.04436/2606.04436v1/fig/robotsetup_b.png)

**说明**: (a) Realman 7-DoF 机械臂 + 1-DoF 夹爪，配备顶部摄像头和腕部摄像头；(b) 真实场景任务设置，包含高度变化和透明容器两类挑战，成功率 88.0%~93.3%。

---

### Figure 5: 训练分析

![Figure 5 Training Analysis](https://arxiv.org/html/2606.04436/2606.04436v1/x3.png)

**说明**: (a) 动作损失训练曲线：蓝线（使用在线 3D 推理蒸馏）收敛更快更低，红线（仅 3D 共训练无蒸馏）收敛较慢；(b) Projection-Space Similarity 分析：量化验证 reasoning distillation 有效缩短了老师/学生路径的特征空间距离。

---

### Table 1: LIBERO 基准性能对比

| 方法 | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|---------|
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| SpatialForcing | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| VITA | 95.9 | 98.9 | 95.1 | 96.8 | 96.7 |
| **3DThinkVLA（Ours）** | **100.0** | **100.0** | **98.8** | **95.8** | **98.7** |

**关键发现**: 3DThinkVLA 在 Spatial 和 Object 两个子集上达到满分 100%，整体平均 98.7% 超越所有基线，包括 SpatialForcing（98.5%）。

---

### Table 2: LIBERO-Plus 零样本迁移对比

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|------|--------|-------|----------|-------|------------|-------|--------|-----|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.9 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| ABot-M0 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| Qwen3-VL-OFT | 47.0 | 60.1 | 87.0 | 96.3 | 95.3 | 73.1 | 79.2 | 75.0 |
| **3DThinkVLA（Ours）** | **73.8** | 64.5 | 78.0 | **98.4** | 94.8 | **84.7** | **81.5** | **81.0** |

**关键发现**: 在摄像头视角、灯光和噪声变化上表现突出，Camera 扰动（73.8%）和 Noise（84.7%）均大幅领先其他方法，平均 81.0%。

---

### Table 3: SimplerEnv WidowX 基准对比

| 方法 | Carrot | Eggplant | Spoon | Block | Avg |
|------|--------|----------|-------|-------|-----|
| Octo | 8.3 | 43.1 | 12.5 | 0.0 | 16.0 |
| OpenVLA | 0.0 | 4.1 | 0.0 | 0.0 | 1.0 |
| RoboVLM | 20.8 | 79.2 | 45.8 | 4.2 | 37.5 |
| SpatialVLA | 25.0 | 100.0 | 16.7 | 29.2 | 42.7 |
| Open π₀ | 61.3 | 89.6 | 73.7 | 15.8 | 60.0 |
| QDepth-VLA | 57.5 | 95.0 | 82.0 | 39.6 | 68.5 |
| FALCON | 41.7 | 100.0 | 62.5 | 20.8 | 56.3 |
| UniVLA | 83.3 | 66.7 | 33.3 | 95.8 | 69.8 |
| VITA | 68.8 | 95.6 | 84.2 | 37.5 | 71.5 |
| **3DThinkVLA（Ours）** | **75.0** | 95.8 | **87.5** | 33.3 | **72.9** |

**关键发现**: 在 SimplerEnv 上达到 72.9%，略超 VITA（71.5%）和 UniVLA（69.8%），Spoon 任务（87.5%）表现突出。

---

### Table 4: 消融实验（LIBERO 各子集）

| ID | Co-train 数据 | Geometry Adapter | Reasoning Distillation（蒸馏） | Spatial Aug. | Spatial | Object | Goal | Long | Avg |
|----|---------------|-----------------|-------------------------------|--------------|---------|--------|------|------|-----|
| R1 | — | ✗ | ✗ | ✗ | 93.6 | 99.6 | 97.4 | 92.6 | 95.8 |
| R2 | 2D Data | ✗ | ✗ | ✗ | 96.8 | 99.8 | 98.0 | 95.0 | 97.4 |
| R3 | 3D Data | ✗ | ✗ | ✗ | 99.8 | 100.0 | 98.8 | 93.0 | 97.9 |
| R4 | 3D Data | ✓ | ✗ | ✗ | 99.0 | 100.0 | 98.2 | 95.8 | 98.3 |
| R5 | 3D Data | ✓ | ✓（蒸馏） | ✗ | 99.6 | 100.0 | 98.8 | 95.8 | 98.6 |
| R6 | 3D Data | ✓ | ✗ | ✓ | 99.2 | 100.0 | 99.4 | 94.4 | 98.3 |
| R7（Full） | 3D Data | ✓ | ✓ | ✓ | **100.0** | **100.0** | **98.8** | **95.8** | **98.7** |

**关键发现**: (1) 3D 共训练数据（R3 vs R1）比 2D 数据（R2）更有效；(2) 三个模块均互补，缺任何一个均降低性能；(3) Geometry Adapter 对长程任务（Long）提升最显著（93.0→95.8）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 个子集（Spatial/Object/Goal/Long）| 仿真操纵基准，多样任务指令 | 主要训练 + 测试 |
| LIBERO-Plus | 7 类扰动 | 零样本迁移，相机/机器人/灯光/背景等扰动 | 泛化测试 |
| SimplerEnv WidowX | 4 个操纵任务 | 仿真跨平台评估 | 跨域迁移测试 |
| SpatialReasoning-CoT | 24,000 样本 | 来自 SpatialReasoner 数据集，带 CoT 标注 | 3D 推理共训练 |
| LLaVA | 24,000 样本 | 通用视觉语言数据 | VLM 通用能力维护 |

### 实现细节

- **Backbone**: Qwen3-VL-2B（冻结 ViT，微调 LLM）
- **动作头**: OFT 风格（连续动作预测）
- **优化器**: AdamW（参数组独立学习率）
  - Base LR: $2.5 \times 10^{-5}$
  - Backbone LR: $1.0 \times 10^{-5}$
  - Action Head LR: $2.0 \times 10^{-4}$
- **调度器**: Cosine decay，最小 LR $1.0 \times 10^{-6}$
- **Batch Size**: 32
- **精度**: bfloat16 混合精度
- **梯度裁剪**: max norm 1.0
- **损失权重**: VLA=1.0，VLM=0.1，几何对齐/推理蒸馏=0.5
- **硬件**: 8× NVIDIA A100 80GB
- **训练开销**: 约 1.5× 基线 VLA

### 真实机器人部署

- **平台**: Realman 7-DoF 机械臂 + 1-DoF 夹爪
- **传感器**: 顶部固定摄像头 + 腕部摄像头（均为 2D RGB，无深度传感器）
- **任务**: 高度变化抓取、透明容器操纵
- **成功率**: 88.0%~93.3%

---

## 批判性思考

### 优点

1. **无传感器依赖**: 推理时纯 2D 输入，部署门槛低，适合工业落地。
2. **清晰的问题诊断**: 首次明确提出 prompt-induced reasoning gap 并给出可量化验证（Figure 5b）。
3. **组件正交互补**: 消融实验表明三模块提升独立且叠加，设计合理。
4. **训练效率合理**: 1.5× 开销换取显著性能提升，工程上可接受。

### 局限性

1. **长程任务提升有限**: LIBERO-Long（95.8%）相比 SpatialForcing（96.0%）未见明显优势，复杂序列任务仍有瓶颈。
2. **共训练数据来源受限**: 依赖 SpatialReasoner 数据集，若数据分布与真实场景差异大，迁移性存疑。
3. **SimplerEnv Block 任务弱**: 仅 33.3%（低于 UniVLA 的 95.8%），说明对某些特定物体类型的泛化仍不稳定。
4. **LIBERO-Plus Robot 扰动一般**: Camera 和 Robot 扰动下均弱于 ABot-M0（60.4% vs 73.8%；64.5% vs 67.9%），说明形态/视角鲁棒性有待加强。

### 潜在改进方向

1. 引入动态 3D 数据（视频 + 点云序列）提升长程任务的时序空间理解。
2. 将 reasoning anchor token 扩展为多 token 推理链，可能进一步压缩推理 gap。
3. 探索 Geometry Adapter 对其他 3D 基础模型（如 Depth Anything）的迁移性。

### 可复现性评估

- [ ] 代码开源（论文称"将开源"，暂未发布）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（超参数、数据配比均有说明）
- [x] 数据集可获取（LIBERO、SpatialReasoner 均公开）

---

## 关联笔记

### 基于

- [[OpenVLA-OFT]]: 动作头架构（OFT 风格）和 LIBERO 基线
- [[VGGT]]: 提供 3D 特征监督信号（训练阶段教师）
- [[VLM]]: Qwen3-VL-2B 作为语言视觉骨干

### 对比

- [[SpatialVLA]]: 同样做隐式 3D 感知的 VLA，但无推理蒸馏
- [[Spatial Forcing]]: LIBERO 上最接近的竞争对手（98.5% vs 98.7%）
- [[OpenVLA-OFT]]: 直接基线，本文在其上添加空间感知能力

### 方法相关

- [[知识蒸馏]]: Online 3D Reasoning Distillation 的核心机制
- [[余弦相似度]]: 几何对齐损失与推理蒸馏损失的度量函数
- [[VGGT]]: 3D 基础模型，提供几何先验
- [[Action Chunking]]: OFT 动作头的输出形式

### 硬件/数据相关

- [[Realman Robot]]: 真实机器人部署平台

---

## 速查卡片

> [!summary] 3DThinkVLA
> - **核心**: 通过潜在 3D 几何感知 + 在线推理蒸馏，让 VLA 模型以纯 2D 输入实现隐式 3D 空间推理
> - **方法**: Geometry Adapter（VGGT 对齐）+ Reasoning Distillation（teacher-student anchor token）+ Spatially Augmented Action Head
> - **结果**: LIBERO 98.7%、LIBERO-Plus 81.0%、SimplerEnv 72.9%、真实机器人 88~93%
> - **代码**: 待公开（https://arxiv.org/abs/2606.04436）

---

*笔记创建时间: 2026-06-04*
