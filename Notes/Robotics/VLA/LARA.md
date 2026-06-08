---
title: "LARA: Latent Action Representation Alignment for Vision-Language-Action Models"
method_name: "LARA"
authors: [Mengya Liu, Baoxiong Jia, Jiangyong Huang, Jingze Zhang, Siyuan Huang]
year: 2026
venue: ICML 2026
tags: [vla, latent-action-model, representation-alignment, diffusion-policy, robot-manipulation, bimanual-manipulation, flow-matching]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.07100
created: 2026-06-08
---

# 论文笔记：LARA: Latent Action Representation Alignment for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未显式列出（通讯作者 Siyuan Huang） |
| 日期 | June 2026 |
| 项目主页 | [https://lmy1001.github.io/ICML26_LARA/](https://lmy1001.github.io/ICML26_LARA/) |
| 对比基线 | [[GR00T-N1.6]], [[OpenVLA]], [[Octo]], [[LAPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07100) / [Code](https://github.com/lmy1001/LARA) |

---

## 一句话总结

> LARA 通过在训练时对齐 [[潜在动作模型|LAM]] 的隐变量与 [[Diffusion Transformer|DiT]] 中间特征，实现 LAM 与 [[VLA]] 的联合优化，在 3 个仿真和 1 个真实世界基准上分别提升约 10%、5%、15%。

---

## 核心贡献

1. **双向表示对齐框架**: 将 [[潜在动作模型|LAM]] 的量化潜变量 $\mathbf{z}_t^\varphi$ 与 [[Diffusion Transformer|DiT]] 的中间层特征 $\mathbf{h}_t^\theta$ 通过余弦相似度损失对齐，打破 LAM 与 VLA 分离训练的瓶颈。
2. **三模式通用部署**: 同一框架支持预训练（pre-training）、后训练增强（post-training enhancement）和潜在动作精炼（latent action refinement），对现有 [[VLA]] 架构侵入性极小。
3. **全面实验验证**: 在 LIBERO、SIMPLER-ENV、GR1-Sim-24、G1-Real 四个基准上验证，覆盖仿真和真实世界双臂人形机器人。

---

## 问题背景

### 要解决的问题

[[VLA|视觉-语言-动作模型]]在数据稀缺场景下需借助大量无标注视频训练 [[潜在动作模型|LAM]]，再将 LAM 编码的伪动作标签用于 VLA 预训练。但 LAM 与 VLA **始终独立训练**，导致两个关键问题：

1. LAM 学习的潜在动作表示在 VLA 训练中**未得到接地（ungrounded）**，无法从 VLA 的语言/视觉语义中获益。
2. VLA 被 **冻结的 LAM 表示** 所约束，无法反向优化 LAM 以获得更好的特征。

### 现有方法的局限

- [[LAPA]]、[[LAPO]]、Moto-GPT 等方法均将 LAM 作为**静态伪标签生成器**使用，两阶段训练缺乏信息回流。
- [[OpenVLA]]、[[Octo]] 等方法不使用无标注视频，数据利用率低。
- 即使是 [[GR00T-N1.6]] 这样的大模型，在低数据量（如 30 条演示）的任务上表现仍有显著提升空间。

### 本文的动机

将 LAM 的**中间潜变量空间**与 VLA（[[Diffusion Transformer|DiT]]）的**中间特征空间**显式对齐——两者都在处理"从观测预测动作"的中间表示，在语义层面天然匹配。联合训练使：
- **LAM** 从 VLA 的语言 / 视觉语义中获益，学到更具语义意义的动作表示（[[逆向动力学模型|IDM]] 抑制幻觉特征）；
- **VLA** 受 LAM 的[[前向动力学模型|FDM]]约束，减少错误轨迹预测（前向动力学正则化）。

---

## 方法详解

### 模型架构

LARA 采用**联合训练 + 表示对齐**架构：

- **输入**: 语言指令 $l$ + 图像观测 $\mathbf{o}_t$ + 机器人本体状态 $\mathbf{s}_t$
- **VLA Backbone**: 基于 [[Diffusion Transformer|DiT]] 的扩散策略（如 [[GR00T-N1.6]]、[[π₀]] 系列），使用 Eagle-2 提取视觉-语言特征
- **LAM**: [[Vision Transformer|ViT]] 编码器构成的 [[逆向动力学模型|IDM]] + VQ 量化器 + [[前向动力学模型|FDM]]，codebook 大小 128
- **对齐注入层**: DiT 倒数第 2 层（Layer L-2）的隐特征 $\mathbf{h}_t^\theta$，经投影头 $f_\psi$ 后与 LAM 潜变量对齐
- **输出**: [[Action Chunking|动作块]] $\bm{A}_{t:t+k}$

```
无标注视频 ──► LAM (IDM + Quantizer + FDM) ──► z_t^φ
                                                    ↑ CosSim 对齐
机器人数据 ──► VLA (DiT) ──► h_t^θ ──► f_ψ(h_t^θ)
```

### 核心模块

#### 模块1: 潜在动作表示对齐（LARA Alignment）

**设计动机**: 利用 [[VQ-VAE]] 框架中的量化潜变量 $\mathbf{z}_t^\varphi$ 作为"动作语义锚"，通过余弦相似度将其拉近 [[Diffusion Transformer|DiT]] 的中间特征，实现双向知识流动。

**具体实现**:
- [[逆向动力学模型|IDM]] 从连续观测帧对 $(\mathbf{o}_t, \mathbf{o}_{t+C})$ 中编码潜在动作 $\mathbf{z}_t$，经 [[VQ-VAE|向量量化]] 后得 $\mathbf{z}_t^\varphi$
- DiT 编码器在 Layer L-2 输出隐状态 $\mathbf{h}_t^\theta$，经可学习投影头 $f_\psi$ 映射到与 $\mathbf{z}_t^\varphi$ 同维的空间
- 对齐损失最大化二者余弦相似度

#### 模块2: LAM（潜在动作模型）三组件

**设计动机**: 从无动作标注的视频中提取结构化动作表示，供 VLA 联合训练使用。

**具体实现**:
- **[[逆向动力学模型|IDM]]（Inverse Dynamics Model）**: 输入 $(o_t, o_{t+C})$，输出连续潜变量 $\mathbf{z}_t$
- **向量量化（[[VQ-VAE]]）**: 将 $\mathbf{z}_t$ 映射到 128-token codebook 中得到 $\mathbf{z}_t^q$（= $\mathbf{z}_t^\varphi$）
- **[[前向动力学模型|FDM]]（Forward Dynamics Model）**: 以 $\mathbf{z}_t^q$ 和 $o_t$ 为输入，重建 $\hat{\mathbf{I}}_{t+C}$，提供前向动力学监督

#### 模块3: 表示对齐的两种退化形式（消融对比）

- **预训练对齐（Eq. 5）**: 使用预训练 LAM 的冻结表示 $\mathbf{y}_t^\text{pretrain}$ 与 DiT 特征对齐，不联合更新 LAM
- **LARA 联合对齐（Eq. 6）**: LAM 和 VLA 同步更新，$\mathbf{z}_t^\varphi$ 在训练中动态演化——**性能显著优于冻结变体**（见消融实验）

---

## 关键公式

### 公式1: [[Flow Matching|流匹配]]动作生成损失

$$
\mathcal{L}_{\text{ACT}}(\theta) = \mathbb{E}_{\tau, \bm{\epsilon}}\left[\|v_{\theta}(\bm{A}_t^{\tau}, \mathbf{c}_t) - (\bm{A}_t - \bm{\epsilon})\|^2\right]
$$

**含义**: 训练速度场网络 $v_\theta$ 预测从噪声到真实动作的去噪方向。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 插值时间步
- $\bm{\epsilon} \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $\bm{A}_t^\tau = (1-\tau)\bm{\epsilon} + \tau\bm{A}_t$: 线性插值噪声动作
- $\mathbf{c}_t$: 条件上下文（语言 + 视觉特征）
- $v_\theta$: VLA 的速度场网络（[[Diffusion Transformer|DiT]] 参数化）

### 公式2: [[Flow Matching|前向欧拉积分]]推理

$$
\bm{A}_t^{\tau + \frac{1}{K}} = \bm{A}_t^{\tau} + \frac{1}{K} v_{\theta}(\bm{A}^{\tau}, \mathbf{c}_t)
$$

**含义**: 推理时用 $K$ 步欧拉积分从纯噪声 $\bm{A}_t^0$ 逐步恢复动作 $\bm{A}_t^1$。

**符号说明**:
- $K$: 积分步数（推理时的去噪步数）
- $\bm{A}_t^0 \sim \mathcal{N}(0, I)$: 初始噪声

### 公式3: [[潜在动作模型|LAM]] 重建损失

$$
\mathcal{L}_{\text{LAM}}(\varphi) = \|\bm{I}_{t+C} - \hat{\bm{I}}_{t+C}\|_2^2 + \|\text{sg}[\mathbf{z}_t^q] - \mathbf{z}_t\|_2^2 + \beta\|\mathbf{z}_t^q - \text{sg}[\mathbf{z}_t]\|_2^2
$$

**含义**: VQ-VAE 框架下联合训练 [[逆向动力学模型|IDM]]、量化器和 [[前向动力学模型|FDM]] 的三项复合损失。

**符号说明**:
- $\bm{I}_{t+C}$: 未来帧真值，$\hat{\bm{I}}_{t+C}$: FDM 重建的未来帧
- $\mathbf{z}_t^q$: 量化后的 codebook 向量（= $\mathbf{z}_t^\varphi$）
- $\mathbf{z}_t$: IDM 编码的连续潜变量
- $\text{sg}[\cdot]$: stop-gradient 算子
- $\beta$: commitment loss 系数

### 公式4: DiT 编解码中间状态

$$
\mathbf{h}_t^{\theta} = E_{\theta}(\bm{A}_t^{\tau}, \mathbf{c}_t), \quad \hat{\mathbf{v}}_t = D_{\theta}(\mathbf{h}_t^{\theta}, \mathbf{c}_t)
$$

**含义**: 将 DiT 显式分为编码器 $E_\theta$（取 Layer L-2 输出）和解码器 $D_\theta$，以便提取中间特征用于对齐。

**符号说明**:
- $\mathbf{h}_t^\theta$: DiT Layer L-2 的隐状态，维度与 $\mathbf{z}_t^\varphi$ 对齐后通过 $f_\psi$ 投影
- $\hat{\mathbf{v}}_t$: 预测速度场

### 公式5: 预训练 LAM 表示对齐损失（退化形式）

$$
\mathcal{L}_{\text{RA}}(\theta, \psi) = -\mathbb{E}_{\bm{A}_t, \bm{\epsilon}, \tau}\left[\texttt{CosSim}\left(\mathbf{y}_t^{\text{pretrain}}, f_{\psi}(\mathbf{h}_t^{\theta})\right)\right]
$$

**含义**: 使用**冻结的**预训练 LAM 表示 $\mathbf{y}_t^\text{pretrain}$ 作为监督信号，仅更新 DiT 和投影头。消融实验表明此变体不如联合训练。

**符号说明**:
- $\mathbf{y}_t^\text{pretrain}$: 预训练 LAM 冻结输出的潜变量
- $f_\psi$: 可学习投影头

### 公式6: [[VQ-VAE|LARA]] 对齐损失（完整联合形式）

$$
\mathcal{L}_{\text{LARA}}(\theta, \varphi, \psi) = -\mathbb{E}_{\bm{A}_t, \bm{\epsilon}, \tau}\left[\texttt{CosSim}\left(\mathbf{z}_t^{\varphi}, f_{\psi}(h_t^{\theta})\right)\right]
$$

**含义**: 联合优化 VLA（$\theta$）、LAM（$\varphi$）和投影头（$\psi$），最大化 LAM 量化潜变量与 DiT 中间特征的余弦相似度。

**符号说明**:
- $\mathbf{z}_t^\varphi$: 联合训练过程中**动态更新**的 LAM 量化潜变量
- $h_t^\theta$: DiT Layer L-2 隐状态
- $f_\psi(\cdot)$: 投影头，将 $\mathbf{h}_t^\theta$ 映射到 $\mathbf{z}_t^\varphi$ 的特征空间

### 公式7: 联合训练总目标

$$
\mathcal{L}(\theta, \varphi, \psi) = \mathcal{L}_{\text{ACT}}(\theta) + w_1 \mathcal{L}_{\text{LARA}}(\theta, \varphi, \psi) + w_2 \mathcal{L}_{\text{LAM}}(\varphi)
$$

**含义**: 三项损失加权求和，同时优化动作生成质量、LAM-VLA 表示对齐和 LAM 的视频重建能力。

**符号说明**:
- $w_1, w_2$: 可学习损失权重
- $\mathcal{L}_{\text{ACT}}$: [[Flow Matching|流匹配]]动作损失（Eq. 1）
- $\mathcal{L}_{\text{LARA}}$: 表示对齐损失（Eq. 6）
- $\mathcal{L}_{\text{LAM}}$: LAM VQ-VAE 重建损失（Eq. 3）

---

## 关键图表

### Figure 1: 框架总览

![Figure 1](https://arxiv.org/html/2606.07100v1/x1.png)

**说明**: LARA 如何桥接无标注视频数据与动作标注机器人数据。左侧大量无标注视频用于训练 [[潜在动作模型|LAM]]，右侧机器人演示数据用于 VLA 训练，LARA 通过**表示对齐**联合优化二者。框架支持三种部署模式：预训练、后训练增强、潜在动作精炼。

### Figure 2: 传统 LAM-VLA vs. LARA 训练范式对比

![Figure 2](https://arxiv.org/html/2606.07100v1/x2.png)

**说明**: 对比两种训练范式。左图：传统方法中 LAM 仅作伪标签生成器，与 VLA 训练解耦；右图：LARA 显式对齐 LAM 潜变量与 DiT 中间特征，实现双向信息流。

### Figure 3: LARA 方法架构

![Figure 3](https://arxiv.org/html/2606.07100v1/x3.png)

**说明**: 展示 [[潜在动作模型|LAM]] 的三组件结构（[[逆向动力学模型|IDM]]、[[VQ-VAE|向量量化器]]、[[前向动力学模型|FDM]]）以及 LARA 如何在 [[Diffusion Transformer|DiT]] 的 Layer L-2 提取隐状态并通过投影头 $f_\psi$ 与 $\mathbf{z}_t^\varphi$ 对齐。

### Figure 4: 任务可视化

![Figure 4](https://arxiv.org/html/2606.07100v1/x4.png)

**说明**: 展示 GR1-Sim-24(30) 仿真中的代表性双臂任务，以及 Unitree G1 人形机器人上的两个真实世界任务：**Pick-n-Place**（拾取并放置）和 **Grasp-n-Pour**（抓取并倒水）。

### Figure 5: 注意力图可视化

![Figure 5](https://arxiv.org/html/2606.07100v1/x5.png)

**说明**: 对比 LAM 与 LARA-LAM 的注意力热图。LARA 精炼后的 LAM 注意力集中在**机器人末端执行器和交互目标**上，而基线 LAM 注意力分散至背景区域，说明对齐损失有效抑制了无关视觉特征。

### Figure 6: 消融实验

![Figure 6](https://arxiv.org/html/2606.07100v1/x6.png)

**说明**: 在 LIBERO-Long 上的消融。左：不同对齐深度（Layer 选择）的影响——Layer L-2 对 GR00T-N1.6 为最优；右：联合训练 vs. 冻结 LAM 对比——联合训练显著超越冻结变体，验证 LAM 动态更新的重要性。

### Table 1: 主对比实验（OXE-Constrained）

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | LIBERO-Avg | SIMPLER-Pick | SIMPLER-Move | SIMPLER-Drawer | SIMPLER-Avg |
|------|----------------|---------------|-------------|-------------|------------|--------------|--------------|----------------|-------------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 16.3 | 46.2 | 35.6 | 32.7 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 | 17.0 | 4.2 | 22.7 | 14.6 |
| Moto-GPT | — | — | — | — | — | 74.0 | 60.4 | 43.1 | 61.4 |
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 | — | — | — | — |
| LARA (DiT-only) | 84.5 | 90.0 | 86.5 | 76.5 | 84.4 | 62.3 | 84.0 | 21.0 | 55.8 |
| **LARA (full)** | **88.0** | **92.0** | **88.5** | **86.0** | **88.6** | **82.3** | **83.7** | **29.5** | **65.2** |
| 提升（vs DiT-only） | +4.1% | +2.2% | +2.3% | +12.4% | +5.0% | +32.1% | −0.4% | +40.5% | +16.8% |

**关键发现**: LARA 在 LIBERO-Long（长程任务）和 SIMPLER-Drawer 上提升最显著（分别 +12.4% / +40.5%），说明表示对齐对**需要长时序规划**的任务帮助最大。

### Table 1: 主对比实验（Unconstrained / Post-training）

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | LIBERO-Avg | SIMPLER-Pick | SIMPLER-Move | SIMPLER-Drawer | SIMPLER-Avg |
|------|----------------|---------------|-------------|-------------|------------|--------------|--------------|----------------|-------------|
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 | 88.0 | 72.7 | 41.8 | 70.7 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 | — | — | — | — |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | 75.3 | 67.5 | 42.9 | 61.9 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 | — | — | — | — |
| TraceVLA | — | — | — | — | — | 45.0 | 63.8 | 63.1 | 57.3 |
| Magma | — | — | — | — | — | 75.0 | 53.0 | 58.9 | 62.3 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 | 98.7 | 75.0 | 59.3 | 77.7 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 | — | — | — | — |
| GR00T-N1.6 | 97.5 | 96.0 | 95.5 | 91.0 | 95.0 | 97.3 | 87.0 | 52.3 | 78.9 |
| **GR00T-N1.6-LARA** | **96.5** | **97.5** | **96.0** | **92.5** | **95.6** | **98.0** | **89.0** | **52.8** | **79.9** |
| 后训练提升 | −1.0% | +1.6% | +0.5% | +1.6% | +0.6% | +0.7% | +2.3% | +1.0% | +1.3% |

### Table 2: GR1-Sim 仿真与 G1-Real 真实世界结果

| 方法 | GR1-Sim-24 Avg | Pick-n-Place Pick | Pick-n-Place Place | Pick-n-Place Full | Grasp-n-Pour Grasp-L | Grasp-n-Pour Grasp-R | Grasp-n-Pour Pour | Grasp-n-Pour Full | G1-Real Avg |
|------|----------------|-------------------|--------------------|-------------------|----------------------|----------------------|-------------------|-------------------|-------------|
| *OXE-Constrained* | | | | | | | | | |
| LARA (DiT-only) | 6.4 | 74.0 | 78.4 | 58.0 | 58.0 | 78.0 | 93.1 | 54.0 | 56.0 |
| **LARA (full)** | **11.4** | **90.0** | **88.9** | **80.0** | **80.0** | **84.0** | **100.0** | **68.0** | **74.0** |
| 提升 | +78.1% | +21.6% | +13.4% | +37.9% | +37.9% | +7.7% | +7.4% | +25.9% | +32.1% |
| *Unconstrained* | | | | | | | | | |
| GR00T-N1.6 | 47.0 | 90.0 | 84.4 | 76.0 | 78.0 | 80.0 | 87.2 | 68.0 | 72.0 |
| **GR00T-N1.6-LARA** | **48.5** | **92.0** | **91.3** | **84.0** | **86.0** | **76.0** | **94.4** | **68.0** | **76.0** |
| 后训练提升 | +3.2% | +2.2% | +8.2% | +10.5% | +10.3% | −4.0% | +7.2% | 0.0% | +5.6% |

**关键发现**: GR1-Sim-24 下 LARA 完整版相比 DiT-only 提升 **78.1%**（6.4%→11.4%），真实世界 G1 平均提升 **32.1%**，显示对齐框架在低数据量人形任务上效果尤为突出。

### Table 3: LAM 精炼实验（SIMPLER-ENV）

| 方法 | Pick Object | Move Near | Open Drawer | Close Drawer | Pick Coke Can | 平均 |
|------|-------------|-----------|-------------|--------------|---------------|------|
| LAM（基线） | 36.3 | 61.0 | 25.7 | 38.0 | 53.0 | 42.8 |
| **LARA-LAM** | **41.0** | **63.7** | **29.3** | **53.7** | **59.7** | **49.5** |
| 提升 | +12.9% | +4.4% | +14.0% | +41.3% | +12.6% | +15.7% |

**关键发现**: LARA 精炼后的 LAM 在所有子任务上均优于基线，Close Drawer 提升最大（+41.3%），与注意力图可视化结论一致——精炼后 LAM 更关注交互目标。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OXE (Open X-Embodiment) | 大规模 | 多具身、多任务机器人数据 | 预训练 |
| LIBERO | 四套任务（Spatial/Object/Goal/Long） | 桌面操作，语言条件 | 训练/测试 |
| SIMPLER-ENV | Pick/Move/Drawer 三类 | Google 机器人仿真 | 测试 |
| GR1-Sim-24(30) | 24 任务各 30 条演示 | 双臂人形仿真，低数据量 | 训练/测试 |
| G1-Real(50) | 2 任务各 50 条演示 | Unitree G1 真实世界 | 训练/测试（各 50 次试验） |

### 实现细节

- **Backbone**: GR00T-N1.6（DiT 架构）、[[π₀]] 系列（后训练消融）
- **LAM**: ViT 编码器，128-token VQ codebook，使用无标注视频预训练
- **对齐注入层**: GR00T-N1.6 用 Layer L-2，[[π₀|π₀.₅]] 用 Layer L（架构差异导致最优层不同）
- **损失权重**: $w_1, w_2$ 可学习
- **训练模式**: OXE-Constrained（仅用 OXE 预训练）/ Unconstrained（使用额外数据）

### 可视化结果

注意力图分析（Figure 5）：LARA-LAM 的注意力热图高度集中在末端执行器和操作目标上，而基线 LAM 注意力弥散至背景墙壁、家具等无关区域，定性解释了 LAM 精炼带来的性能提升。

---

## 批判性思考

### 优点
1. **框架简洁高效**: 仅需在现有 VLA 架构上添加一个投影头和余弦相似度损失，额外计算开销极小，对架构侵入性低。
2. **三模式通用**: 同一对齐框架适用于预训练、后训练增强和 LAM 精炼，灵活性强，可直接提升已有大模型（如 GR00T-N1.6）。
3. **低数据量场景效果突出**: GR1-Sim-24(30) 仅 30 条演示即获 78.1% 相对提升，说明对齐正则化在数据稀缺时有效抑制了过拟合。

### 局限性
1. **对齐层选择依赖架构**: Layer L-2 对 GR00T-N1.6 最优，但 π₀.₅ 需用 Layer L，说明最优对齐深度需要针对不同 DiT 架构进行超参调整，缺乏统一的理论指导。
2. **LAM 的视频数据质量未讨论**: 无标注视频的领域差距（domain gap）对 LAM 学习质量的影响未系统分析；真实世界部署中视频来源的多样性是否足够是个开放问题。
3. **仿真到真实的迁移差距**: GR1-Sim-24 绝对成功率仍较低（11.4%），真实世界 G1 虽有改善但个别子任务（Grasp-R）出现负迁移（−4.0%）。

### 潜在改进方向
1. 设计自适应层选择机制，根据梯度信号自动确定最优对齐深度，避免手动超参搜索。
2. 探索多层对齐或分层对齐策略，捕获 DiT 不同深度的语义层次。
3. 研究视频数据来源对 LAM 质量的影响，建立视频数据筛选准则以提升真实世界迁移性。

### 可复现性评估
- [x] 代码开源（https://github.com/lmy1001/LARA）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（损失权重、对齐层、codebook 大小等均有报告）
- [x] 数据集可获取（LIBERO/SIMPLER-ENV/OXE 均公开）

---

## 关联笔记

### 基于
- [[潜在动作模型|LAM]]: LARA 的 LAM 组件直接沿用 VQ-VAE 框架的三组件设计
- [[LAPA]]: 同样使用 LAM 伪标签，但 LARA 通过联合训练克服其静态标签局限
- [[LAPO]]: LAM 伪动作标签预训练的早期工作
- [[Flow Matching]]: LARA VLA 部分的动作生成基础

### 对比
- [[GR00T-N1.6]]: 被 LARA 作为后训练增强基座，GR00T-N1.6-LARA 在多基准上超越原模型
- [[OpenVLA]]: 经典 VLA 基线，OXE-Constrained 设置下 LARA 在 LIBERO 平均超越 12.1%
- [[Octo]]: 多具身预训练基线
- [[LAPA]]: 同赛道最直接对比，LARA 在 LIBERO 平均超越 22.9%（88.6% vs 65.7%）

### 方法相关
- [[Diffusion Transformer|DiT]]: VLA 的骨干网络，LARA 在其 Layer L-2 提取对齐特征
- [[VQ-VAE]]: LAM 量化器的理论基础
- [[逆向动力学模型|IDM]]: LAM 的核心编码器组件
- [[前向动力学模型|FDM]]: 提供前向动力学正则化的 LAM 解码器组件
- [[Flow Matching]]: 动作生成的扩散框架
- [[Action Chunking]]: VLA 输出的动作序列表示

### 硬件/数据相关
- [[GR1-Sim-24]]: 双臂人形仿真基准（24 任务）
- [[LIBERO]]: 桌面操作基准，语言条件任务集
- [[SIMPLER-ENV]]: Google 机器人仿真评测平台

---

## 速查卡片

> [!summary] LARA: Latent Action Representation Alignment for VLA
> - **核心**: 通过对齐 LAM 量化潜变量与 DiT 中间特征，实现 LAM 与 VLA 联合优化
> - **方法**: 余弦相似度对齐损失 $\mathcal{L}_\text{LARA}$，联合优化三项目标（Eq. 7）
> - **结果**: LIBERO +10%, SIMPLER +17%, GR1-Sim +78%, G1-Real +32%（OXE-Constrained）
> - **代码**: https://github.com/lmy1001/LARA

---

*笔记创建时间: 2026-06-08*
