---
title: "World Pilot: Steering Vision-Language-Action Models with World-Action Priors"
method_name: "WorldPilot"
authors: [Zefu Lin, Rongxu Cui, Junjia Xu, Xiaojuan Jin, Wenling Li, Lue Fan, Zhaoxiang Zhang]
year: 2026
venue: arXiv
tags: [vla, world-model, world-action-model, robot-manipulation, out-of-distribution, flow-matching, latent-steering]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.12403
created: 2026-06-11
---

# 论文笔记：World Pilot: Steering Vision-Language-Action Models with World-Action Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CASIA、南京大学、北京航空航天大学 |
| 日期 | June 2026 |
| 项目主页 | [world-pilot.github.io](https://world-pilot.github.io/) |
| 对比基线 | [[ABot-M0]]、[[pi0]]、[[Cosmos Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12403) |

---

## 一句话总结

> World Pilot 通过两条互补路径（Latent Steering 和 Action Steering）将 [[World-Action Model|WAM]] 的预测先验注入 VLA 策略，使其在零样本 OOD 基准上实现 84.7% 总成功率的 SOTA 表现。

---

## 核心贡献

1. **Latent Steering**: 将 WAM 的场景演化潜变量 $Z^w_t$ 通过 [[Cross-Attention]] 注入 VLM 隐层状态，为感知阶段提供动态场景先验
2. **Action Steering**: 将 WAM 预期轨迹 $\tilde{A}^w_t$ 编码为单个 prefix token，插入基于 [[Flow Matching|流匹配]] 的动作生成器，提供轨迹级运动先验
3. **模块化设计**: WAM 全程冻结，VLA 与 WAM 仅通过动作损失耦合，两者均可独立替换为更强版本

---

## 问题背景

### 要解决的问题

[[VLA|视觉-语言-动作（VLA）模型]] 直接从 VLM 的图像-指令编码生成动作块，缺乏对场景动态演化的理解。当测试场景出现视角变化、几何变形、姿态偏移等分布外扰动时，策略鲁棒性显著下降。

### 现有方法的局限

- 标准 VLA（如 [[pi0]]、ABot-M0）仅依赖当前帧语义编码，无法预测未来场景动态
- [[World-Action Model|WAM]] 虽然具备场景预测能力，但将其解码为像素级未来帧会引入纹理、光照等行动无关的噪声，稀释控制相关结构
- 直接用 WAM 输出初始化动作或逐步注入 WAM 轨迹会使生成器过度依赖有噪 WAM 信号

### 本文的动机

WAM 的潜空间（latent space）比解码图像更紧凑，保留了场景演化的动态结构而过滤了行动无关细节；将预期轨迹压缩为单个 token 则给动作生成器保留了充分自由度，同时仍能获得轨迹级方向引导。

---

## 方法详解

### 模型架构

**World Pilot** 采用双路径增强 VLA 架构：

- **输入**: 语言指令 $\ell$ + 观测 $O_t$ + 本体状态 $q_t$
- **Backbone**: [[Qwen3-VL]]（VLM 编码器）+ [[DiT|扩散变换器]]（动作生成器）
- **核心模块**: [[Latent Steering]] 用于感知增强，[[Action Steering]] 用于运动引导
- **外部组件**: [[World-Action Model|WAM]]（基于 Cosmos Policy，全程冻结）
- **输出**: [[Action Chunking|动作块]] $\hat{A}_{\theta,t}$

整体框架包含两条信息流：
- **语义路径**: 图像 + 语言 → VLM 隐层状态 $H_t$
- **世界先验路径**: WAM 输出 $(Z^w_t, \tilde{A}^w_t)$ → 经 Latent Steering 和 Action Steering 分别注入

### 核心模块

#### 模块1：Latent Steering（潜变量引导）

**设计动机**: 利用 [[VAE|变分自编码器]] 的紧凑潜空间表示场景动态，通过 [[Cross-Attention|交叉注意力]] 将预期场景演化注入 VLM 感知，而非使用解码后的像素图像（会引入行动无关的纹理/光照噪声）。

**具体实现**:
1. WAM 对当前观测 $O_t$ 用 VAE 编码后，通过 [[Diffusion Transformer|扩散变换器]] 去噪生成场景演化潜变量 $Z^w_t$
2. 经动态编码器 $f_{\text{dyn}}$ 投影并添加未来时刻嵌入 $\rho_{\text{fut}}$：

$$
D^w_t = f_{\text{dyn}}(Z^w_t) + \rho_{\text{fut}}
$$

3. VLM 隐层状态 $H_t$ 通过交叉注意力从 $D^w_t$ 中提取动态先验，残差相加保留原始语义：

$$
\bar{H}_t = H_t + \text{CrossAttn}(H_t, D^w_t)
$$

**关键选择**: 使用潜变量而非解码图像，因为"像素内容携带纹理、光照、背景等行动无关细节，稀释控制相关结构"。

#### 模块2：Action Steering（动作引导）

**设计动机**: 为流匹配动作生成器提供轨迹级方向先验，同时避免逐步绑定到 WAM 噪声信号，保留生成器的决策自由度。

**具体实现**:
1. 将 WAM 输出的预期轨迹 $\tilde{A}^w_t$ 重采样到 VLA 的动作 horizon $K$
2. 通过动作编码器 $f_{\text{act}}$ 压缩为单个先验 token：

$$
s^w_t = f_{\text{act}}(\text{Align}_K(\tilde{A}^w_t))
$$

3. 将 $s^w_t$ 作为 prefix token 插入流匹配生成器输入序列：

$$
[u_t;\ s^w_t;\ Q_t;\ X_{\tau,t}]
$$

其中 $u_t$ 为可选状态 token，$Q_t$ 为可学习的未来查询 token，$X_{\tau,t}$ 为流时刻 $\tau$ 处的含噪轨迹。

---

## 关键公式

### 公式1：[[World-Action Model|World Pilot 整体框架]]

$$
(Z^w_t,\ \tilde{A}^w_t) = W_\phi(O_t, \ell, q_t)
$$

$$
\hat{A}_{\theta,t} = \pi_\theta(O_t, \ell, q_t;\ Z^w_t, \tilde{A}^w_t)
$$

**含义**: WAM $W_\phi$ 基于当前观测、语言指令和状态生成场景演化潜变量与预期轨迹，VLA 策略 $\pi_\theta$ 在此基础上生成可执行动作块。

**符号说明**:
- $Z^w_t$: WAM 输出的场景演化潜变量
- $\tilde{A}^w_t$: WAM 预期的动作轨迹
- $\hat{A}_{\theta,t}$: VLA 最终输出的可执行动作块
- $\phi, \theta$: WAM 和 VLA 的参数（$\phi$ 冻结，$\theta$ 训练）

### 公式2：[[Latent Steering|Latent Steering 动态编码]]

$$
D^w_t = f_{\text{dyn}}(Z^w_t) + \rho_{\text{fut}}
$$

**含义**: 将 WAM 场景演化潜变量通过动态编码器投影，并叠加未来时刻位置嵌入，构建供 VLM 查询的动态表示。

**符号说明**:
- $f_{\text{dyn}}$: 动态编码器（可训练）
- $Z^w_t$: WAM 场景演化潜变量
- $\rho_{\text{fut}}$: 未来场景时刻的可学习时序嵌入

### 公式3：[[Cross-Attention|Latent Steering 交叉注意力融合]]

$$
\bar{H}_t = H_t + \text{CrossAttn}(H_t, D^w_t)
$$

**含义**: VLM 隐层状态通过残差交叉注意力从动态场景表示中提取先验，保留原始语义的同时引入场景演化信息。

**符号说明**:
- $H_t$: 原始 VLM 隐层状态（Query 来源）
- $D^w_t$: WAM 动态场景表示（Key/Value 来源）
- $\bar{H}_t$: 融合后的增强隐层状态

### 公式4：[[Action Steering|Action Steering 轨迹编码]]

$$
s^w_t = f_{\text{act}}(\text{Align}_K(\tilde{A}^w_t))
$$

**含义**: 将 WAM 预期轨迹重采样对齐到 VLA horizon，再通过动作编码器压缩为单个轨迹先验 token。

**符号说明**:
- $f_{\text{act}}$: 动作编码器（可训练）
- $\text{Align}_K$: 将 WAM 轨迹重采样到 VLA 动作 horizon $K$ 的对齐操作
- $\tilde{A}^w_t$: WAM 预期动作轨迹
- $s^w_t$: 单个轨迹先验 prefix token

### 公式5：[[Flow Matching|World Pilot 训练损失]]

$$
\mathcal{L}_{\text{World Pilot}} = \mathbb{E}_{\tau, \epsilon}\left[w(\tau)\|\hat{A}_{\theta,t} - A^*_t\|^2_2\right]
$$

$$
w(\tau) = \frac{1}{(1-\tau)^2}
$$

**含义**: 在清洁动作参数化下实现速度空间目标的加权 [[Flow Matching|流匹配]] 损失，权重 $w(\tau)$ 在流时刻接近 1 时赋予更高权重。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 流匹配时刻，均匀采样
- $\epsilon$: 采样噪声
- $w(\tau) = 1/(1-\tau)^2$: 时刻依赖的加权函数
- $\hat{A}_{\theta,t}$: 模型预测的动作
- $A^*_t$: 清洁目标动作

---

## 关键图表

### Figure 1: World Pilot 整体框架

![Figure 1](https://arxiv.org/html/2606.12403v1/x1.png)

**说明**: World Pilot 以 [[World-Action Model|WAM]] 为先验源，通过两条路径增强 VLA 决策链：Latent Steering 将场景演化潜变量路由到 VLM 隐层状态，Action Steering 将预期轨迹以运动先验形式输入动作生成器。

### Figure 2: World Pilot 完整架构

![Figure 2](https://arxiv.org/html/2606.12403v1/x2.png)

**说明**: 语义路径（图像+语言 → VLM → 隐层状态）与 WAM 的两条先验路径并行，Latent Steering 通过 [[Cross-Attention|交叉注意力]] 注入 VLM，Action Steering 通过 prefix token 注入基于 [[DiT|DiT]] 的流匹配动作生成器。

### Figure 3: 真实机器人评估场景

![Figure 3](https://arxiv.org/html/2606.12403v1/x3.png)

**说明**: 机器人平台（左）、与训练分布匹配的域内场景（中）、以及出现外观变化、几何变形、形变状态或姿态偏移的 OOD 场景（右）。评估覆盖堆叠积木、折叠毛巾、放置水果、对齐容器盖四类任务。

### Table 1: LIBERO-Plus 零样本 OOD 基准结果

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|------|--------|-------|----------|-------|------------|-------|--------|-----------|
| ABot-M0 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| **World Pilot** | **82.8** | 60.6 | 87.2 | **98.6** | **96.4** | **93.6** | 80.5 | **84.7** |

**关键发现**: World Pilot 在 Total 指标上超越最强基线 2.6 个百分点，Camera 轴提升幅度最大（+13.2），显示出对视角变化的显著鲁棒性；在 Robot（机器人几何变形）和 Layout（布局变化）轴上提升有限。

### Table 2: 真实机器人评估结果（成功率 %）

| 任务 | 方法 | ID 成功率 | OOD 成功率 | ID→OOD 下降 |
|------|------|-----------|------------|-------------|
| 堆叠积木 | World Pilot | 70 | 50 | -20 |
| 折叠毛巾 | World Pilot | 85 | 70 | -15 |
| 放置水果 | World Pilot | 90 | 70 | -20 |
| 对齐容器盖 | World Pilot | 80 | 65 | -15 |

**关键发现**: World Pilot 的 ID→OOD 下降幅度控制在 20 个百分点以内，而其他基线（[[pi0]]、ABot-M0、Cosmos Policy）下降幅度为 25~50 个百分点，体现出显著的 OOD 鲁棒性优势。

### Table 3: 消融实验——路径贡献

| 变体 | LIBERO-Plus 成功率 |
|------|-------------------|
| ABot-M0（基线） | 80.5% |
| 仅 Latent Steering | 83.7%（+3.2） |
| 仅 Action Steering | 83.1%（+2.6） |
| 完整 World Pilot | **84.7%（+4.2）** |

**关键发现**: 两条路径贡献互补——Latent Steering 和 Action Steering 单独各自提升约 3 个点，组合后取得 4.2 点提升，证明两者捕捉了不同维度的先验信息。

### Table 4: 世界模型预训练需求消融

| 基准 | ABot-M0 | + Latent Steering（Cosmos-Predict，仅场景预测） |
|------|---------|------------------------------------------------|
| LIBERO-Plus | 80.5 | 82.6（+2.1） |
| RoboCasa | 54.0 | 62.7（+8.7） |
| RoboTwin2.0 | 81.2 | 85.3（+4.1） |

**关键发现**: 即使使用未经动作后训练的纯场景预测 WAM（Cosmos-Predict），场景演化先验仍然有效，在 RoboCasa 上带来 8.7 点的显著提升，说明动作先验并非必须条件。

### Table 5: 潜空间 vs 像素空间条件消融

| 表示形式 | LIBERO-Plus 成功率 |
|----------|-------------------|
| 潜变量（1步去噪） | 84.6% |
| 潜变量（3步去噪） | 84.5% |
| 潜变量（5步去噪） | **84.7%** |
| 解码未来图像 | 83.5% |

**关键发现**: 潜变量表示在不同去噪步数下稳定（84.5~84.7%），而解码为像素图像后下降 1.2 个点，因图像引入"视觉伪影和被稀释的动态结构"。

### Table 6: 动作先验集成方式消融

| 集成方式 | LIBERO-Plus 成功率 |
|----------|-------------------|
| 单个编码 token（World Pilot） | **84.7%** |
| 逐步编码 token | 83.6% |
| 流匹配初始化 | 84.1% |
| 原始轨迹 token | 83.0% |

**关键发现**: 单 token 形式允许生成器在保持轨迹方向引导的同时自由优化具体动作；逐步 token 和原始轨迹方式使输出过度绑定到有噪 WAM 信号，性能下降。

---

## 实验

### 数据集与基准

| 数据集/基准 | 规模 | 特点 | 用途 |
|-------------|------|------|------|
| [[LIBERO]]-Plus | 7 轴 OOD 扰动 | 视角/几何/光照/布局等零样本 OOD 评估 | 仿真主基准 |
| [[RoboCasa]] | 长时序厨房任务 | 多步操作，场景复杂 | 仿真迁移验证 |
| RoboTwin2.0 | 双臂操作 | 精密操作任务 | 仿真泛化验证 |
| 真实机器人 | 4 类任务 × 每类 100 条示教 | 堆叠/折叠/放置/对齐，20 次评估/设置 | 真实世界验证 |

### 实现细节

- **VLM Backbone**: [[Qwen3-VL]]（基于 ABot-M0）
- **动作生成器**: [[DiT|DiT]]-based [[Flow Matching|流匹配]]
- **WAM**: Cosmos Policy（5步去噪，全程冻结）
- **Dropout**: WAM 条件上施加 0.3 dropout，防止过度依赖先验
- **硬件**: 8× RTX PRO 6000 GPU
- **可训练组件**: VLM backbone、动态编码器 $f_{\text{dyn}}$、Latent Steering 交叉注意力、动作编码器 $f_{\text{act}}$、流匹配动作生成器

---

## 批判性思考

### 优点

1. **设计优雅**: 双路径互补，分别在感知和动作生成阶段引入先验，无需修改 WAM 结构
2. **模块化强**: WAM 冻结且可替换，VLA backbone 也可独立升级，工程灵活性高
3. **充分验证**: 仿真 + 真实机器人评估完备，消融实验覆盖各设计决策，结论可信
4. **OOD 鲁棒性突出**: 真实机器人 OOD 下降幅度显著优于竞品，实用价值高

### 局限性

1. **WAM 覆盖依赖**: 测试场景超出 WAM 视频预训练分布时性能退化
2. **提升不均匀**: LIBERO-Plus 的 Robot 和 Layout 轴提升有限，说明几何/布局变化还需专项设计
3. **计算开销**: 每步决策需额外 WAM 前向传播，不适合高频实时控制
4. **解耦代价**: 模块化设计阻止了 WAM 与 VLA 的联合优化，存在先验-策略协同适配的上限

### 潜在改进方向

1. **不确定性感知先验门控**: 当 WAM 覆盖度低时自动降低先验权重，减少分布外场景的负面影响
2. **联合 WAM-VLA 协同微调**: 通过联合训练关闭先验-策略回路，挖掘更深层协同
3. **先验蒸馏或自适应查询**: 将额外 WAM 推理开销压缩，使方法适用于高频控制场景

### 可复现性评估

- [ ] 代码开源（项目主页存在，但代码未确认开源）
- [ ] 预训练模型（未确认）
- [x] 训练细节完整（论文提供了充分的实现描述）
- [x] 数据集可获取（LIBERO-Plus、RoboCasa 均公开）

---

## 关联笔记

### 基于

- [[World-Action Model]]: WAM 框架，提供场景演化潜变量和预期轨迹
- [[Cosmos-Predict]]: World Pilot 使用的具体 WAM 实现（冻结）
- [[Qwen3-VL]]: VLM backbone
- [[Action Chunking]]: VLA 动作输出形式

### 对比

- [[pi0]]: 主要竞品，真实机器人 OOD 下降 25~45 点（World Pilot 控制在 20 点内）
- [[ABot-M0]]: World Pilot 基座 VLA，LIBERO-Plus 基线 80.5%

### 方法相关

- [[Latent Steering]]: 感知阶段世界先验注入的核心机制（本文提出）
- [[Action Steering]]: 动作生成阶段轨迹先验注入机制（本文提出）
- [[Cross-Attention]]: Latent Steering 的实现基础
- [[Flow Matching]]: 动作生成器的训练范式
- [[DiT]]: 动作生成器和 WAM 的骨干架构
- [[VAE]]: WAM 编码器，将观测映射到紧凑潜空间

### 数据集相关

- [[LIBERO]]: 主评估基准（LIBERO-Plus 为其 OOD 扩展版）
- [[RoboCasa]]: 长时序操作迁移基准

---

## 速查卡片

> [!summary] World Pilot (arXiv 2606.12403)
> - **核心**: 用 WAM 的两类先验（场景演化潜变量 + 预期轨迹）双路增强 VLA 策略
> - **方法**: Latent Steering（CrossAttn 注入 VLM）+ Action Steering（prefix token 注入 DiT）
> - **结果**: LIBERO-Plus OOD 84.7% SOTA；真实机器人 OOD 下降 ≤20pt vs 竞品 25~50pt
> - **代码**: [world-pilot.github.io](https://world-pilot.github.io/)

---

*笔记创建时间: 2026-06-11*
