---
title: "RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models"
method_name: "RoVLA"
authors: [Jingzhou Luo, Yifan Wen, Yongjie Bai, Xinshuai Song, Yang Liu, Liang Lin]
year: 2026
venue: arXiv
tags: [vla, robustness, consistency-learning, diffusion-policy, imitation-learning, embodied-ai]
zotero_collection: Robotics/VLA
image_source: mixed
arxiv_html: https://arxiv.org/html/2605.19678
created: 2026-05-21
---

# 论文笔记：RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Sun Yat-sen University (HCP Lab) |
| 日期 | May 2026 |
| 项目主页 | [GitHub](https://github.com/HCPLab-SYSU/RoVLA) |
| 对比基线 | [[OpenVLA-OFT]], [[Pi0]], [[Pi0.5]], [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19678) / [Code](https://github.com/HCPLab-SYSU/RoVLA) |

---

## 一句话总结

> RoVLA 通过三种互补的一致性约束（指令一致性、演化一致性、观测一致性），显著提升 VLA 模型在语言改写和视觉扰动下的鲁棒性。

---

## 核心贡献

1. **指令一致性（IC）**: 利用 Qwen3-8B 生成多样化改写指令，隐式正则化策略网络对语义等价指令的响应
2. **演化一致性（EC）**: 在扩散去噪过程的两个不同时间步之间施加速度预测一致性损失，抑制轨迹波动
3. **观测一致性（OC）**: 对视觉特征和机器人状态施加对抗扰动，强制干净输入与扰动输入之间的预测一致性

---

## 问题背景

### 要解决的问题

现有 [[Vision-Language-Action Model|VLA]] 模型在训练与测试分布一致时表现良好，但在以下三种场景下会失败：
- **伪视觉理解**：视角变化、背景更换、光照变化导致策略崩溃
- **伪语言遵循**：语义等价但表达不同的指令产生不一致动作
- **伪组合泛化**：多个轻微扰动叠加导致严重性能退化

### 现有方法的局限

现有改进鲁棒性的方向（扩大训练数据、更强的适应方法、世界模型方法）将鲁棒性视为"涌现的副产品"，而非显式的策略学习目标，导致模型对训练数据中的浅层相关性产生依赖。

### 本文的动机

通过在训练过程中施加多种一致性约束，让模型主动学习任务语义、环境状态与动作生成之间的稳定关系，而非依赖训练集中的伪相关性。三种约束相互互补，分别针对语言、演化动态和视觉观测三个维度的脆弱性。

---

## 方法详解

### 模型架构

RoVLA 采用**双系统（Dual-System）** 架构，基于 [[Conditional Flow Matching|条件流匹配]] 框架：

- **输入**: 语言指令 $T$ + 观测图像 $\mathbf{I}_t$ + 本体状态 $\mathbf{q}_t$
- **语义编码器（高层系统）**: [[InternVL|InternVL3.5-2B]]，提取第 16 层 decoder 特征，包含语言 token $l_t$ 和视觉 token $v_t$
- **动作生成器（低层系统）**: 32 层 [[Diffusion Transformer|DiT]]，基于 [[Flow Matching]] 生成连续 [[Action Chunking|动作块]]
- **输出**: 动作块 $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+L-1}] \in \mathbb{R}^{L \times d_a}$，$L=16$
- **训练时**：在三种一致性约束下联合优化

### 核心模块

#### 模块 1：动作块与流匹配基础（SFT Loss）

**设计动机**：用 [[Conditional Flow Matching]] 替代传统扩散，在连续动作空间中生成高质量多步动作块。

**具体实现**：
- 线性概率路径：$\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t^{gt} + (1 - \tau) \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \mathbf{I})$，$\tau \in [0, 1]$
- 真实速度场：$\mathbf{v}^{gt} = \mathbf{A}_t^{gt} - \varepsilon$
- DiT 网络预测速度：$\hat{\mathbf{v}}^{\tau} = \text{DiT}_{\theta_2}(l_t, v_t, \mathbf{A}_t^{\tau}, \mathbf{q}_t, \tau)$
- 流匹配损失：$\mathcal{L}_{\text{FM}} = \mathbb{E}_t[\|\hat{\mathbf{v}}^{\tau} - \mathbf{v}^{gt}\|_2^2]$
- 推理时：从 $\mathcal{N}(0, \mathbf{I})$ 采样初始噪声，用前向 Euler 积分 8 步得到动作

#### 模块 2：指令一致性（IC, Instructional Consistency）

**设计动机**：强制策略在语义等价但表面形式不同的指令下产生一致动作，解决"伪语言遵循"问题。

**具体实现**：
- 使用 Qwen3-8B 为每条轨迹生成约 15 个多样改写指令，覆盖 7 种模板风格（用户意图式、功能目标式、礼貌请求式、简洁命令式、教学式、抽象目标式等）
- 后处理：过滤长度 < 8 字符的输出、归一化短语、去除拒绝回复、去重
- 训练时从改写集合 $\mathcal{D}_T = \{T^{(1)}, \ldots, T^{(N_{\text{lang}})}\}$ 均匀采样一条改写指令
- 无额外显式损失项，以隐式数据增强方式正则化模型

#### 模块 3：演化一致性（EC, Evolutionary Consistency）

**设计动机**：在去噪生成过程中抑制表示波动，保证不同演化时刻的速度预测一致，减少轨迹抖动。

**具体实现**：
- 训练时采样两个演化时间步 $\tau_1, \tau_2 \sim \text{Beta}$
- 时间步采样分布：$p(\tau) = \text{Beta}\left(\frac{s - \tau}{s}; 1.5, 1\right)$，$s = 0.999$（非均匀采样，偏向较小时间步）
- 对干净输入分别计算两个时间步下的速度预测 $\hat{\mathbf{v}}_{\text{clean}}^{\tau_1}$ 和 $\hat{\mathbf{v}}_{\text{clean}}^{\tau_2}$
- 施加 $\ell_2$ 一致性损失

#### 模块 4：观测一致性（OC, Observational Consistency）

**设计动机**：通过对抗扰动增强策略对视觉和本体状态噪声的鲁棒性，解决"伪视觉理解"问题。

**具体实现**：
- 沿 $\mathcal{L}_{\text{EC}}$ 的梯度方向对视觉特征 $v_t$ 和本体状态 $\mathbf{q}_t$ 施加有界扰动：

$$
\tilde{v}_t = v_t + \eta_v \frac{\nabla_{v_t} \mathcal{L}_{\text{EC}}}{\|\nabla_{v_t} \mathcal{L}_{\text{EC}}\|_2}, \quad \eta_v = \min(\alpha, \varepsilon_{\text{adv}} \|v_t\|_2)
$$

$$
\tilde{\mathbf{q}}_t = \mathbf{q}_t + \eta_q \frac{\nabla_{\mathbf{q}_t} \mathcal{L}_{\text{EC}}}{\|\nabla_{\mathbf{q}_t} \mathcal{L}_{\text{EC}}\|_2}, \quad \eta_q = \min(\alpha, \varepsilon_{\text{adv}} \|\mathbf{q}_t\|_2)
$$

- 扰动参数：$\alpha = 0.01$，$\varepsilon_{\text{adv}} = 0.03$
- 强制扰动预测与干净预测之间的一致性（使用 stop-gradient）

---

## 关键公式

### 公式 1：[[Action Chunking|动作块]]定义

$$
\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+L-1}] \in \mathbb{R}^{L \times d_a}
$$

**含义**：策略每步预测长度为 $L$ 的动作序列，减少推理频率并提升时序一致性。

**符号说明**：
- $L = 16$：动作块长度
- $d_a$：单步动作维度

### 公式 2：[[Conditional Flow Matching|线性概率路径]]

$$
\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t^{gt} + (1 - \tau) \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I}), \quad \tau \in [0, 1]
$$

**含义**：在噪声 $\varepsilon$ 和目标动作 $\mathbf{A}_t^{gt}$ 之间插值，构造训练用的中间状态。

**符号说明**：
- $\tau$：流匹配时间步，$\tau = 0$ 为纯噪声，$\tau = 1$ 为目标动作
- $\mathbf{A}_t^{gt}$：真实动作块

### 公式 3：[[Conditional Flow Matching|速度场]]与损失

$$
\mathbf{v}^{gt} = \mathbf{A}_t^{gt} - \varepsilon
$$

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_t\left[\left\|\mathbf{V}_\theta(T, \mathbf{I}_t, \mathbf{A}_t^{\tau}, \mathbf{q}_t, \tau) - \mathbf{v}^{gt}\right\|_2^2\right]
$$

**含义**：监督 DiT 网络预测线性插值的速度场，即从噪声到目标动作的方向。

### 公式 4：[[Evolutionary Consistency|演化一致性损失]]

$$
\mathcal{L}_{\text{EC}} = \left\|\hat{\mathbf{v}}_{\text{clean}}^{\tau_1} - \hat{\mathbf{v}}_{\text{clean}}^{\tau_2}\right\|_2^2
$$

**含义**：强制模型在两个不同去噪时间步下产生一致的速度预测，抑制演化过程中的表示波动。

**符号说明**：
- $\tau_1, \tau_2$：两个采样的演化时间步
- $\hat{\mathbf{v}}_{\text{clean}}^{\tau_i}$：干净输入下时间步 $\tau_i$ 的速度预测

### 公式 5：[[Evolutionary Consistency|非均匀时间步采样]]

$$
p(\tau) = \text{Beta}\left(\frac{s - \tau}{s}; 1.5, 1\right), \quad s = 0.999
$$

**含义**：偏向采样较小时间步（接近纯噪声），因为这些阶段的预测噪声更大、更需要一致性约束。

### 公式 6：[[Adversarial Perturbation|观测对抗扰动]]

$$
\tilde{v}_t = v_t + \eta_v \frac{\nabla_{v_t} \mathcal{L}_{\text{EC}}}{\|\nabla_{v_t} \mathcal{L}_{\text{EC}}\|_2}, \quad \eta_v = \min(\alpha, \varepsilon_{\text{adv}} \|v_t\|_2)
$$

$$
\tilde{\mathbf{q}}_t = \mathbf{q}_t + \eta_q \frac{\nabla_{\mathbf{q}_t} \mathcal{L}_{\text{EC}}}{\|\nabla_{\mathbf{q}_t} \mathcal{L}_{\text{EC}}\|_2}, \quad \eta_q = \min(\alpha, \varepsilon_{\text{adv}} \|\mathbf{q}_t\|_2)
$$

**含义**：沿最大化演化一致性损失的梯度方向对视觉特征和本体状态施加有界对抗扰动。

**符号说明**：
- $\alpha = 0.01$：绝对扰动上界
- $\varepsilon_{\text{adv}} = 0.03$：相对扰动预算系数
- $\tilde{v}_t, \tilde{\mathbf{q}}_t$：扰动后的视觉特征和本体状态

### 公式 7：[[Observational Consistency|观测一致性损失]]

$$
\mathcal{L}_{\text{OC}} = \frac{1}{2} \sum_{i=1}^{2} \left\|\hat{\mathbf{v}}_{\text{pert}}^{\tau_i} - \text{sg}\left(\hat{\mathbf{v}}_{\text{clean}}^{\tau_i}\right)\right\|_2^2
$$

**含义**：强制扰动输入下的速度预测向干净输入预测对齐，`sg` 表示 stop-gradient，防止梯度反传污染干净预测。

### 公式 8：[[Adaptive Weighting|自适应权重调度]]

$$
\lambda_j = \frac{1}{1 + \mathcal{L}_j^{ema}}
$$

$$
\mathcal{L}_j^{ema} = \gamma \mathcal{L}_{j-1}^{ema} + (1 - \gamma) \mathcal{L}_{\text{SFT}}^{(j)}
$$

**含义**：用监督损失的指数滑动平均动态调节一致性约束的权重，训练初期优先动作预测，后期逐渐加强一致性约束。

**符号说明**：
- $\gamma = 0.95$：EMA 衰减系数
- $\mathcal{L}_j^{ema}$：第 $j$ 步的监督损失 EMA
- 初始 $\mathcal{L}_0^{ema} = 100$

### 公式 9：总训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda_j \mathcal{L}_C
$$

$$
\mathcal{L}_C = \mathcal{L}_{\text{EC}} + \mathcal{L}_{\text{OC}}
$$

**含义**：将监督流匹配损失与一致性正则化损失加权相加，联合优化策略。

---

## 关键图表

### Figure 1：VLA 脆弱性示意

![[RoVLA_fig1.png|600]]

**说明**：展示现有 VLA 模型在视觉观测变化（视角、光照、背景）和语义等价语言指令改写下的脆弱性。相同任务在不同表达或视角下产生截然不同的动作。

---

### Figure 2：RoVLA 整体架构

![[RoVLA_fig2.png|600]]

**说明**：(a) 训练时，RoVLA 在双系统骨干网络基础上施加三种一致性约束（IC / EC / OC）；(b) 推理时，策略仅使用标准双系统骨干，执行 $K = 8$ 步去噪生成动作块。

---

### Figure 3：真实环境评测任务可视化

![[RoVLA_fig3.png|600]]

**说明**：五个桌面操作任务：拾取香蕉、拾取苹果、将香蕉放入碗、将苹果放入碗、将苹果放入抽屉。使用腕部摄像头，评估感知、指令理解和操作能力。

---

### Figure 4：LIBERO-Plus 定性结果

![[RoVLA_fig4.png|600]]

**说明**：在 (a) 增加物体数量、(b) 语言指令改写、(c) 光照变化、(d) 相机模糊、(e) 视角变化等扰动条件下，RoVLA 仍能维持相对稳定的动作决策。

---

### Figure 5：RoboTwin 2.0 定性结果

![[RoVLA_fig5.png|600]]

**说明**：在随机化环境下（物体姿态、场景布局、视觉外观随机变化），RoVLA 的目标物体定位和执行轨迹仍保持相对稳定。

---

### Figure 6：真实环境腕部相机视角可视化

![[RoVLA_fig6.png|600]]

**说明**：从腕部摄像头视角展示五个真实操作任务的执行轨迹快照，即使初始物体配置在不同试次间有变化，RoVLA 仍能维持稳定执行。

---

### Table 1：LIBERO-Plus 评测结果

**Zero-shot 扰动泛化**（在 LIBERO 上训练，在 LIBERO-Plus 上测试）：

| Method | Pretrained | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|-----------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA | ✓ | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| OpenVLA-OFT_m | ✓ | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| WorldVLA | ✗ | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | ✓ | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π₀ | ✓ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | ✓ | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| InternVL-3.5 + DiT | ✗ | 58.3 | 37.2 | 76.3 | 95.1 | 94.8 | 79.1 | 74.0 | 71.6 |
| **RoVLA (Ours)** | ✗ | **58.4** | 36.3 | **92.9** | **95.6** | **95.0** | **80.9** | 73.0 | **74.3** |

**扰动暴露泛化**（在 LIBERO-Plus 上微调）：

| Method | Pretrained | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|-----------|--------|-------|----------|-------|------------|-------|--------|-------|
| π₀ | ✓ | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| π₀.₅ | ✓ | 70.3 | 41.7 | 81.1 | 97.3 | 94.6 | 71.8 | 84.9 | 75.7 |
| GR00T-N1.6 | ✓ | 92.6 | 33.5 | 80.1 | 93.6 | 95.4 | 93.6 | 75.0 | 79.4 |
| OpenVLA-OFT | ✓ | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.5 |
| InternVL-3.5 + DiT | ✗ | 94.2 | 32.4 | 64.5 | 94.7 | 93.0 | 94.7 | 74.8 | 77.1 |
| **RoVLA (Ours)** | ✗ | **96.6** | 32.0 | **91.5** | **95.9** | **96.1** | **95.1** | 74.1 | **82.0** |

**关键发现**：RoVLA 在语言扰动上优势最为突出（+16.6 pt zero-shot，+27.0 pt fine-tuned），无需大规模预训练即可超越需要预训练的方法。

---

### Table 2：RoboTwin 2.0 评测结果（部分任务）

| 任务 | GO-1 Clean | GO-1 Rand | π₀.₅ Clean | π₀.₅ Rand | InternVL+DiT Clean | InternVL+DiT Rand | RoVLA Clean | RoVLA Rand |
|------|-----------|-----------|-----------|-----------|------|------|------|------|
| Click Alarmclock | 95.0 | 90.0 | 97.0 | 93.0 | 92.0 | 99.0 | 90.0 | 94.0 |
| Click Bell | 98.0 | 95.0 | 75.0 | 76.0 | 89.0 | 85.0 | 97.0 | 94.0 |
| Dump Bin Bigbin | 57.0 | 45.0 | 30.0 | 42.0 | 83.0 | 71.0 | 82.0 | 76.0 |
| Grab Roller | 99.0 | 99.0 | 90.0 | 89.0 | 88.0 | 96.0 | 80.0 | 87.0 |
| Handover Mic | 12.0 | 8.0 | 28.0 | 18.0 | 25.0 | 26.0 | 34.0 | 34.0 |
| Move Can Pot | 16.0 | 4.0 | 29.0 | 27.0 | 62.0 | 64.0 | 63.0 | 47.0 |
| **总计（50任务）** | **37.8** | **36.2** | **43.0** | **43.8** | **44.9** | **45.4** | **48.2** | **50.0** |

**关键发现**：RoVLA 在随机化环境下（50.0%）高于干净环境（48.2%），说明一致性约束减少了对训练分布的过拟合。

---

### Table 3：真实环境评测

| 任务 | GR00T-N1.6 | InternVL3.5+DiT | RoVLA (Ours) |
|------|------------|-----------------|--------------|
| 拾取香蕉 | 80 | 60 | 80 |
| 拾取苹果 | 50 | 60 | **70** |
| 将香蕉放入碗 | 70 | 50 | **80** |
| 将苹果放入碗 | 40 | 20 | **50** |
| 将苹果放入抽屉 | 10 | 0 | **20** |
| **总计** | 50 | 38 | **60** |

**关键发现**：RoVLA 在所有真实任务上均优于 baseline，且无需大规模预训练数据。

---

### Table 4：消融实验（LIBERO-Plus 扰动暴露设置）

| IC | EC | OC | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|----|----|----|--------|-------|----------|-------|------------|-------|--------|-------|
| ✗ | ✗ | ✗ | 94.2 | 32.4 | 64.5 | 94.7 | 93.0 | 94.7 | 74.8 | 77.1 |
| ✗ | ✓ | ✗ | 94.6 | 32.5 | 68.4 | 94.0 | 94.5 | 94.2 | 77.0 | 78.2 |
| ✓ | ✓ | ✗ | 93.7 | 30.6 | 89.5 | 94.7 | 93.5 | 94.6 | 73.8 | 80.5 |
| ✗ | ✓ | ✓ | 95.3 | 35.6 | 69.6 | 94.5 | 95.4 | 95.6 | 74.5 | 79.0 |
| ✓ | ✓ | ✓ | **96.6** | 32.0 | **91.5** | **95.9** | **96.1** | **95.1** | 74.1 | **82.0** |

**关键发现**：
- EC 是 OC 的前置条件（OC 使用 EC 梯度生成扰动），必须同时启用
- IC 主要提升语言鲁棒性（64.5% → 89.5%）
- OC 主要提升机器人状态和噪声鲁棒性（Robot: +5.2 pt，Noise: +1.4 pt）
- 三者协同使用效果最佳，不存在负干扰

---

## 实验

### 数据集/基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 标准 | 7 种任务扰动类型（相机/机器人/语言/光照/背景/噪声/布局） | 零样本泛化训练集 |
| LIBERO-Plus | 扩展 | 对 LIBERO 添加多维扰动 | 主要评测集 |
| RoboTwin 2.0 | 50 任务 | 双臂操作，干净/随机化两种环境 | 泛化性评测 |
| 真实环境 | 5 任务 | Franka Research 3，腕部摄像头 | 真实世界验证 |

### 实现细节

- **语义编码器**: InternVL3.5-2B，解冻最后 4 层，提取第 16 层特征
- **动作生成器**: 32 层 DiT，基于条件流匹配
- **优化器**: AdamW，线性 warmup（前 5%）至 $1 \times 10^{-4}$，余弦衰减至 0
- **训练步数**: LIBERO-Plus / 真实环境 60k 步；RoboTwin 2.0 120k 步
- **推理步数**: 8 步前向 Euler 积分
- **硬件**: NVIDIA H20 GPU
- **扰动参数**: $\alpha = 0.01$，$\varepsilon_{\text{adv}} = 0.03$
- **EMA 衰减**: $\gamma = 0.95$，初始 $\mathcal{L}_0^{ema} = 100$

---

## 批判性思考

### 优点

1. **三维正交覆盖**：三种一致性约束分别针对语言、动态演化、视觉观测三个独立维度，设计互补，消融实验验证无负干扰
2. **无需预训练数据**：RoVLA 不依赖大规模机器人预训练数据，仅用目标域数据微调即可超越预训练方法（GR00T-N1.6、π₀.₅）
3. **即插即用**：一致性约束作为训练时的正则化损失，推理时不增加任何计算开销
4. **零样本鲁棒性**：即使在训练时未见扰动（LIBERO-Plus zero-shot），语言鲁棒性也提升 16.6 个百分点
5. **随机化环境更优**：RoboTwin 2.0 随机化下（50.0%）优于干净环境（48.2%），罕见地证明了鲁棒性训练减少过拟合

### 局限性

1. **精细接触动力学**：对于需要精确力控制的操作（如精密装配），方法改善有限
2. **复杂双臂协调**：复杂双臂任务的性能提升有限，可能需要更强的几何和动力学约束
3. **布局扰动瓶颈**：Layout 维度的提升最小，说明大规模预训练赋予的空间先验仍难以被替代
4. **Qwen3-8B 依赖**：IC 的指令改写依赖 Qwen3-8B，在资源受限场景下增加数据准备成本

### 潜在改进方向

1. 引入几何感知约束或接触力建模，改善精细接触动力学场景
2. 探索 IC 与在线强化学习结合，动态生成更难的语言扰动
3. 将 EC/OC 扩展到空间/布局维度，如对 3D 坐标施加等变性约束

### 可复现性评估

- [x] 代码开源（承诺发布，GitHub 链接已给出）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（超参数、步数、硬件均详细说明）
- [x] 数据集可获取（LIBERO-Plus、RoboTwin 2.0 均为公开基准）

---

## 关联笔记

### 基于

- [[InternVL|InternVL3.5-2B]]: 语义编码骨干
- [[Diffusion Transformer]]: 动作生成骨干（DiT）
- [[Conditional Flow Matching]]: 动作生成的核心训练框架

### 对比

- [[OpenVLA-OFT]]: 需要大规模预训练的对比基线，LIBERO-Plus 总分 79.5%
- [[Pi0]]: π₀ 系列，需要预训练，总分 67.4%
- [[Pi0.5]]: π₀.₅，总分 75.7%
- [[GR00T N1.5]]: GR00T-N1.6，需要预训练，总分 79.4%
- [[RIPT-VLA]]: 通过强化学习改进的 VLA，总分 68.4%

### 方法相关

- [[Evolutionary Consistency]]: 核心方法之一
- [[Observational Consistency]]: 核心方法之一
- [[Adversarial Perturbation]]: OC 使用的扰动技术
- [[Adaptive Weighting]]: 训练中的自适应权重调度
- [[Conditional Flow Matching]]: 底层生成框架

### 数据集相关

- [[LIBERO]]: 主要训练与评测基准
- [[RoboTwin 2.0]]: 双臂操作泛化评测基准

---

## 速查卡片

> [!summary] RoVLA: Multi-Consistency Constraints for Robust VLA
> - **核心**: 三种互补一致性约束（IC/EC/OC）显著提升 VLA 鲁棒性
> - **方法**: 指令改写增强 + 演化时间步一致性 + 对抗扰动一致性
> - **结果**: LIBERO-Plus 82.0%（+4.9），语言维度 91.5%（+27.0），无需预训练
> - **代码**: https://github.com/HCPLab-SYSU/RoVLA

---

*笔记创建时间: 2026-05-21*
