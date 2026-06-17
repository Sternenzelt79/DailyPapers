---
title: "WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT"
method_name: "WAM-RL"
authors: [Zezhong Qian, Xiaowei Chi, Yu Qi, Haozhan Li, Zhi Yang Chen, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, reinforcement-learning, robot-manipulation, flow-matching, video-generation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.17906
created: 2026-06-17
---

# 论文笔记：WAM-RL

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 北京大学（主力）、东北大学、清华大学 |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Genie Envisioner]]、[[πRL]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17906) / Code: N/A |

---

## 一句话总结

> WAM-RL 首次将强化学习引入 [[World-Action Model]] 范式，通过在线视频 SFT + [[KL Divergence|KL 正则]] 稳定世界模型训练，并以重建奖励优化动作模型，实现世界模型与动作模型的联合进化。

---

## 核心贡献

1. **首个 WAM 强化学习框架**: 将 [[Reinforcement Learning|RL]] 引入 [[World-Action Model]] 范式，是该领域的首次尝试
2. **在线视频 SFT + KL 正则**: 通过 [[KL Divergence|KL 散度]] 约束隐空间分布偏移，使世界模型在在线交互中稳定更新
3. **重建奖励设计**: 用想象轨迹与真实观测之间的像素级 MSE 相似度作为奖励信号，避免稀疏奖励问题；并通过消融实验揭示"判别力强的奖励不一定带来更好的训练效果"

---

## 问题背景

### 要解决的问题

[[World-Action Model]]（WAM）展现出强泛化能力，但完全依赖专家示范轨迹进行监督学习，无法通过与真实环境交互自主提升能力。特别是在**长时程任务**中，模型无法从错误中恢复。

### 现有方法的局限

- 针对 [[VLA]] 的 RL 方法（如 [[RIPT-VLA]]、[[πRL]]）仅优化动作模型，保持视觉表示固定
- 直接对 WAM 的世界模型进行在线微调会导致隐空间表示严重漂移（distribution shift），使动作模型失效
- 仅优化动作模型（actor-only）在短时程任务上有改善，但**对长时程任务几乎无效**

### 本文的动机

WAM 的核心能力来自世界模型，动作模型主要充当"翻译器"。要实现长时程改进，必须让世界模型也参与在线更新。关键挑战在于如何在更新世界模型的同时保持隐空间的稳定性。

---

## 方法详解

### 模型架构

WAM-RL 基于 **[[Genie Envisioner]]-ACT** 架构（DiT-based 视频生成器 + ACT 动作头）：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$
- **世界模型**: 在隐空间生成想象的未来观测帧 $\hat{x}_{t+1:t+H}$
- **动作模型（Actor）**: 将隐空间预测翻译为可执行动作 $a_{t:t+k}$（[[Action Chunking|动作块]]）
- **环境反馈**: 真实观测 $x_{t+1:t+H}$ 用于计算重建奖励
- **训练框架**: 世界模型用在线视频 SFT + [[KL Divergence|KL 正则]]；动作模型用 [[Policy Gradient|策略梯度]] RL

**设计哲学**：

> "The primary capability of WA models originates from the world model, while the action model mainly serves as a translator."

### 核心模块

#### 模块 1：Online Video SFT with KL Regularization（在线视频 SFT + KL 约束）

**设计动机**：直接在线微调世界模型会导致隐空间几何结构崩塌，动作模型依赖固定的隐空间表示，分布偏移后动作模型完全失效。

**具体实现**：
- 用在线交互数据对世界模型做自监督视频预测微调
- 用 [[KL Divergence|KL 散度]] 约束当前模型与旧模型隐层分布的距离
- 用高斯分布近似隐层分布，使 KL 计算可解析

#### 模块 2：Action Model RL with Reconstruction-Based Reward（重建奖励 RL）

**设计动机**：避免稀疏奖励（sparse reward）问题，用世界模型的"想象质量"作为稠密奖励信号。

**具体实现**：
- 世界模型根据当前动作预测未来 $H$ 帧的想象轨迹 $\hat{x}_{t+1:t+H}$
- 真实执行后获得真实观测 $x_{t+1:t+H}$
- 计算两者相似度作为奖励：成功时世界模型能准确预测，失败时预测与现实偏差大
- 用 [[Policy Gradient|策略梯度]] 对动作模型参数 $\phi$ 进行更新

---

## 关键公式

### 公式 1：[[Flow Matching|Flow Matching ODE]]

$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

**含义**：Flow Matching 学习一个时间依赖的向量场 $v_\theta$，将简单分布（如噪声）传输到数据分布。

**符号说明**：
- $x_t$：$t$ 时刻的状态（在流场中插值）
- $v_\theta(x_t, t)$：由参数 $\theta$ 建模的速度场

### 公式 2：[[Flow Matching|Flow Matching 损失]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{x_0, x_1, t}\left[\|v_\theta(x_t, t) - v^*(x_t, t)\|^2\right]
$$

**含义**：最小化预测速度场与目标速度场之间的均方误差。

**符号说明**：
- $x_0$：噪声样本（源分布）
- $x_1$：真实数据样本（目标分布）
- $v^*(x_t, t)$：最优速度场（连接 $x_0$ 到 $x_1$ 的直线）

### 公式 3：[[Flow Matching|Flow-SDE（随机微分方程版 Flow Matching）]]

$$
dx_t = v_\theta(x_t, t)\,dt + \sigma\,dW_t
$$

**含义**：在确定性 Flow Matching ODE 中引入随机性，使策略对 RL 中标准策略梯度方法可解析（可以计算对数似然）。

**符号说明**：
- $\sigma$：噪声强度
- $dW_t$：维纳过程增量（布朗运动）

### 公式 4：动作对数似然（用于策略梯度）

$$
\log \pi_\theta(a|s) = \sum_t \log p(x_{t-1}|x_t)
$$

**含义**：通过逐步去噪过程的对数条件概率之和，计算动作序列的策略对数似然，作为策略梯度的基础。

**符号说明**：
- $\pi_\theta(a|s)$：给定状态 $s$ 输出动作 $a$ 的策略
- $p(x_{t-1}|x_t)$：反向去噪转移概率

### 公式 5：[[KL Divergence|世界模型视频损失]]

$$
\mathcal{L}_{video} = \mathbb{E}_{x_{1:T}}\left[\ell\left(f_\theta(x_{<t}), x_t\right)\right]
$$

**含义**：世界模型以历史帧 $x_{<t}$ 为条件，预测当前帧 $x_t$，最小化重建误差（自监督）。

**符号说明**：
- $f_\theta$：世界模型（DiT-based 视频预测网络）
- $\ell(\cdot, \cdot)$：重建损失函数（如 MSE 或感知损失）

### 公式 6：隐空间分布建模（高斯近似）

$$
p_\theta(z_t | x_{<t}) = \mathcal{N}(z_t,\, \Sigma_\theta), \quad p_{old}(z_t | x_{<t}) = \mathcal{N}(z_t^{old},\, \Sigma_{old})
$$

**含义**：用高斯分布近似当前模型与旧模型的隐空间分布，从而使 KL 散度可解析计算。

**符号说明**：
- $z_t$：当前世界模型的隐层特征均值
- $z_t^{old}$：旧模型（更新前）的隐层特征均值
- $\Sigma_\theta, \Sigma_{old}$：对应的协方差矩阵

### 公式 7：[[KL Divergence|KL 正则项]]

$$
\mathcal{L}_{KL} = \mathbb{E}_t\left[D_{KL}\left(\mathcal{N}(z_t,\, \Sigma_\theta) \;\|\; \mathcal{N}(z_t^{old},\, \Sigma_{old})\right)\right]
$$

**含义**：约束当前世界模型的隐空间分布不要偏离旧模型太远，防止分布漂移导致动作模型失效。

**符号说明**：
- $D_{KL}(\cdot \| \cdot)$：KL 散度

### 公式 8：[[KL Divergence|世界模型总目标]]

$$
\mathcal{L}_{WM} = \mathcal{L}_{video} + \lambda_{KL}\,\mathcal{L}_{KL}
$$

**含义**：世界模型的最终训练目标，在视频预测质量与隐空间稳定性之间权衡。

**符号说明**：
- $\lambda_{KL}$：KL 正则项权重系数

### 公式 9：重建奖励（Reconstruction-Based Reward）

$$
r_t = \text{sim}\left(\hat{x}_{t+1:t+H},\; x_{t+1:t+H}\right)
$$

**含义**：用世界模型的想象轨迹与真实观测轨迹的相似度作为奖励，想象越准确说明动作越合理。

**符号说明**：
- $\hat{x}_{t+1:t+H}$：世界模型根据当前动作想象的未来 $H$ 帧
- $x_{t+1:t+H}$：真实执行动作后的真实观测帧
- $\text{sim}(\cdot, \cdot)$：相似度函数（可选像素 MSE / 光流一致性 / DINOv2 特征相似度 / V-JEPA2 特征相似度）

### 公式 10：[[Policy Gradient|动作模型策略梯度]]

$$
\nabla_\phi J = \mathbb{E}\left[\nabla_\phi \log \pi_\phi(a_t|s_t)\, A_t\right]
$$

**含义**：用优势函数加权的策略梯度更新动作模型参数 $\phi$，引导策略朝高奖励方向改进。

**符号说明**：
- $\phi$：动作模型（Actor）参数
- $\pi_\phi(a_t|s_t)$：动作策略
- $A_t$：优势函数（Advantage），衡量当前动作比基线好多少

---

## 关键图表

### Figure 1：WAM-RL 整体框架

![Figure 1 - WAM-RL Overview](https://arxiv.org/html/2606.17906v1/x1.png)

**说明**：WAM-RL 的整体框架图。世界模型在隐空间生成想象的未来观测，动作模型翻译为可执行动作；真实环境反馈作为重建奖励信号，同时触发世界模型的在线视频 SFT 更新。两个模块通过 [[KL Divergence|KL 正则]] 保持稳定联合优化。

### Figure 2：不同重建目标的奖励分布对比

![Figure 2 - Reward Distributions](https://arxiv.org/html/2606.17906v1/x2.png)

**说明**：归一化后的成功/失败奖励分布（成功固定为 1，失败按比例缩放）。**光流一致性**的成功/失败分布最为分离（判别力最强），但在实验中效果反而最差。**像素 MSE** 判别力相对弱，却取得最好结果，揭示"奖励判别力强不等于训练效果好"。

### Figure 3：Video SFT 对单 Action Chunk 行为的影响

![Figure 3 - Video SFT Effect](https://arxiv.org/html/2606.17906v1/x3.png)

**说明**：对比有无 Video SFT 时模型的行为。**无 Video SFT**：初始错误后无法恢复，迅速偏向 OOD 状态。**有 Video SFT**：模型能在单个 [[Action Chunking|Action Chunk]] 内预判失败并生成纠正动作，体现了改进的失败动态建模能力。

### Table 1：主要结果对比

| 方法 | LIBERO-Object | RLBench (Water Plants) |
|------|---------------|------------------------|
| Base | 68% | 19% |
| [[πRL]] | 78% | 18% |
| **WAM-RL（Ours）** | **82%** | **22%** |

**关键发现**：WAM-RL 在 LIBERO-Object 上将成功率从 68% 提升至 82%，显著超过仅优化动作模型的 πRL（78%）。在长时程任务 RLBench Water Plants 上，πRL 相对 Base 几乎无提升（19%→18%），而 WAM-RL 实现了有效提升（19%→22%），验证了世界模型联合优化对长时程任务的必要性。

### Table 2：消融实验——重建目标选择

| 方法 | 成功率 |
|------|--------|
| Base | 19% |
| [[πRL]] | 18% |
| Pixel MSE（Ours） | 21% |
| 光流 MSE | 19% |
| [[DINOv2]] MSE | 16% |
| [[V-JEPA2]] | 17% |

**关键发现**：
- 像素 MSE 取得最佳效果，与世界模型的训练目标对齐，提供稳定学习信号
- 光流一致性的奖励分布最有判别力，但实际训练效果却最差
- 高层语义特征（[[DINOv2]]、[[V-JEPA2]]）效果反而低于 Base，可能因特征空间与世界模型隐空间不兼容

---

## 实验

### 数据集与基准

| 数据集/基准 | 特点 | 用途 |
|------------|------|------|
| [[LIBERO]]-Object | 短时程对象操作任务 | 主要评估基准 |
| [[RLBench]] Water Plants | 多步长时程机器人技能 | 长时程能力评估 |

### 实现细节

- **架构**: Genie Envisioner-ACT（DiT-based 视频生成器 + ACT 动作头）
- **硬件**: 8 × NVIDIA A800 GPU
- **训练时长**: 约 8 小时
- **奖励函数**: 像素级 MSE（最佳消融结果）
- **KL 正则系数**: $\lambda_{KL}$（具体值未披露）

### 可视化结果

Video SFT 效果可视化（Figure 3）展示了世界模型更新带来的关键能力：**在单个 Action Chunk 内生成恢复行为**。无 Video SFT 时，模型初次失败后会进入 OOD 的"茫然"状态并持续漂移；有 Video SFT 后，模型能预见失败并在同一 chunk 内切换到纠正动作。

---

## 批判性思考

### 优点

1. **问题定义清晰**：明确指出 actor-only RL 对长时程任务无效，并提出可信的原因分析
2. **稳定性方案合理**：KL 正则约束隐空间漂移是针对 WAM 特点的精准设计
3. **奖励消融有洞察力**：揭示"奖励判别力 ≠ 训练效果"的反直觉发现，具有重要实践价值

### 局限性

1. **KL 正则限制适应速度**：过强的约束导致世界模型适应真实分布的速度有限，长期可能成为性能瓶颈
2. **重建奖励成功/失败分离度不足**：像素 MSE 的奖励分布重叠较多（见 Figure 2），学习信号噪声较大
3. **实验规模较小**：仅在 2 个基准上评估，且长时程任务提升幅度有限（19%→22%）
4. **奖励设计任务无关**：纯粹基于重建相似度，缺乏任务理解，难以扩展到需要复杂推理的任务

### 潜在改进方向

1. **更具表达力的任务感知奖励**：结合任务目标的语义奖励（如 VLM 打分）
2. **判别式重建奖励**：不仅比较像素，还引入对成功/失败有区分力的特征
3. **动态 KL 系数调度**：训练早期宽松、后期收紧，平衡适应速度与稳定性

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节较完整（GPU 数量、训练时长、基准明确）
- [x] 数据集可获取（LIBERO、RLBench 均为公开基准）

---

## 关联笔记

### 基于

- [[Genie Envisioner]]: 本文使用的基础 WAM 架构（DiT-based 视频生成器）
- [[Flow Matching]]: 视频生成和动作生成的核心训练框架
- [[World-Action Model]]: 本文扩展的范式

### 对比

- [[πRL]]: 代表性 actor-only RL baseline，本文对比对象
- [[DreamZero]]: 首先形式化 World-Action Model 范式的工作

### 方法相关

- [[Flow Matching]]: 视频和动作的生成框架
- [[KL Divergence]]: 隐空间稳定性约束的核心工具
- [[Reinforcement Learning]]: 动作模型优化框架
- [[Policy Gradient]]: 动作模型的具体更新算法
- [[Action Chunking]]: 动作模型输出形式
- [[V-JEPA2]]: 消融中测试的特征相似度方案
- [[DINOv2]]: 消融中测试的视觉特征提取器
- [[Optical Flow]]: 消融中测试的重建目标

### 硬件/数据相关

- [[LIBERO]]: 短时程操作基准
- [[RLBench]]: 长时程多步骤操作基准

---

## 速查卡片

> [!summary] WAM-RL
> - **核心**: 首次将 RL 引入 World-Action Model，联合优化世界模型与动作模型
> - **方法**: 在线视频 SFT + KL 正则（稳定世界模型）+ 像素重建奖励（训练动作模型）
> - **结果**: LIBERO-Object 68%→82%；长时程任务 actor-only RL 无效，WAM-RL 有效
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-17*
