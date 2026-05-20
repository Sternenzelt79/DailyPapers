---
title: "RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models"
method_name: "RoVLA"
authors: [Jingzhou Luo, Yifan Wen, Yongjie Bai, Xinshuai Song, Yang Liu, Liang Lin]
year: 2026
venue: arXiv
tags: [vla, robustness, consistency-regularization, flow-matching, diffusion-policy, imitation-learning]
zotero_collection: Notes/Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.19678v1
created: 2026-05-20
---

# 论文笔记：RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | SYSU (Sun Yat-sen University) - HCP Lab |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[GR00T-N1]], [[OpenVLA-OFT]], [[pi0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19678) / [Code](https://github.com/HCPLab-SYSU/RoVLA) |

---

## 一句话总结

> RoVLA 通过指令一致性、演化一致性和观测一致性三重约束，系统增强 VLA 模型对视觉扰动和语言变体的鲁棒性。

---

## 核心贡献

1. **Instructional Consistency (IC)**: 利用 LLM 生成每条轨迹的多种语义等价指令变体，训练时随机采样以隐式正则化策略，使模型对指令措辞变化保持稳定。
2. **Evolutionary Consistency (EC)**: 在扩散去噪过程中强制不同时间步的速度场预测一致，抑制动作生成中的表示波动，无需额外推理开销。
3. **Observational Consistency (OC)**: 利用 EC 梯度生成对抗性扰动，施加到视觉特征和机器人状态上，再约束扰动前后预测一致，提升对视觉/本体感知噪声的鲁棒性。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在训练分布之外表现脆弱：面对视角变化、光照变化、背景变化等视觉扰动，以及语义等价但措辞不同的语言指令时，策略性能大幅下降。

### 现有方法的局限

- [[Diffusion Policy]] 等流匹配策略对视觉扰动缺乏显式鲁棒性设计
- [[VLA]] 模型（如 [[pi0]]、GR00T-N1.6）依赖语义编码器但没有一致性约束
- 数据增强方法只覆盖已知扰动类型，对组合扰动泛化不足

### 本文的动机

单纯数据增强无法覆盖所有分布外场景；三种一致性约束从语言语义、动作生成过程、视觉感知三个维度分别施加正则化，形成系统性防御机制。

---

## 方法详解

### 模型架构

RoVLA 采用**双系统骨干（dual-system backbone）**架构：
- **输入**: 语言指令 $l_t$ + 视觉观测 $o_t$ + 机器人状态 $q_t$ + 噪声动作块 $A_t^\tau$
- **高层语义编码器**: [[InternVL|InternVL3.5-2B]] 的第 16 层 decoder 输出，提取语义视觉特征 $v_t$
- **低层动作生成器**: [[Diffusion Transformer|DiT]] (32 层 Transformer)，基于[[条件流匹配|Conditional Flow Matching]]建模动作
- **输出**: 动作块 $a_{t:t+k}$（Action Chunk）
- **一致性约束**: IC + EC + OC 在训练阶段注入，推理阶段不增加任何开销

```
语言指令 l_t ──┐
视觉观测 o_t ──┤→ InternVL3.5-2B (第16层) → v_t ──┐
机器人状态 q_t ──────────────────────────────────────┤→ DiT (32层) → v̂_clean → 动作块
噪声动作 A_t^τ ──────────────────────────────────────┘
                         ↑
              IC / EC / OC 训练时约束
```

### 核心模块

#### 模块 1: Instructional Consistency (IC)

**设计动机**: 利用 [[LLM]] 的语言生成能力，提前构建多样化指令空间，强迫策略对措辞变化免疫。

**具体实现**:
- 使用 [[Qwen3|Qwen3-8B]] 为每条训练轨迹生成约 15 条语义等价的指令变体
- 7 种 prompt 模板覆盖不同表达风格：用户意图式、功能目标式、功能参照式、礼貌请求式、简洁命令式、教导式、抽象目标式
- 训练时均匀采样一条指令（而非固定原始指令），通过数据多样性**隐式**实现正则化，无需额外损失项

#### 模块 2: Evolutionary Consistency (EC)

**设计动机**: 流匹配去噪过程中速度场在不同时间步应预测一致的"清洁"动作意图；不一致代表表示抖动，是过拟合信号。

**具体实现**:
- 训练时采样两个去噪时间步 $\tau_1, \tau_2$，各自前向计算速度场预测 $\hat{v}_{clean}^{\tau_1}$、$\hat{v}_{clean}^{\tau_2}$
- 用 L2 损失约束两者一致
- 时间步采样使用非均匀 Beta 分布：$p(\tau) = \mathrm{Beta}\!\left(\frac{s-\tau}{s};\,1.5,\,1\right)$，$s=0.999$，对早期去噪阶段（低噪声）采样更密集

#### 模块 3: Observational Consistency (OC)

**设计动机**: 利用 EC 梯度信号生成有针对性的对抗扰动，再在扰动样本上施加一致性约束，实现对抗训练效果而无需双倍前向开销。

**具体实现**:
- 对语义视觉特征 $v_t$ 施加梯度攻击得到 $\tilde{v}_t$，对机器人状态 $q_t$ 得到 $\tilde{q}_t$
- 扰动样本的速度场预测 $\hat{v}_{pert}^{\tau_i}$ 与原始预测 $\hat{v}_{clean}^{\tau_i}$ 之间用 L2 + stop-gradient 约束
- stop-gradient 防止两侧互相塌缩（collapse）

---

## 关键公式

### 公式 1: [[条件流匹配|SFT 监督损失]]

$$
\mathcal{L}_{SFT} = \frac{1}{2}\left(\mathcal{L}_{clean} + \mathcal{L}_{pert}\right)
$$

**含义**: 在原始样本和扰动样本上均做流匹配监督，兼顾正常训练与对抗鲁棒性。

**符号说明**:
- $\mathcal{L}_{clean}$: 原始样本上的流匹配速度场回归损失
- $\mathcal{L}_{pert}$: 扰动样本上的流匹配速度场回归损失

---

### 公式 2: [[Evolutionary Consistency|演化一致性损失 EC]]

$$
\mathcal{L}_{EC} = \left\|\hat{v}_{clean}^{\tau_1} - \hat{v}_{clean}^{\tau_2}\right\|_2^2
$$

**含义**: 强制模型在两个随机去噪时间步上预测出一致的清洁速度场，抑制去噪过程中的表示波动。

**符号说明**:
- $\hat{v}_{clean}^{\tau_i}$: 在去噪时间步 $\tau_i$ 预测的清洁速度场
- $\tau_1, \tau_2 \sim \mathrm{Beta}(\frac{s-\tau}{s}; 1.5, 1)$: 非均匀采样的两个去噪时间步，$s=0.999$

---

### 公式 3: [[Observational Consistency|观测一致性损失 OC]]

$$
\mathcal{L}_{OC} = \frac{1}{2}\sum_{i=1}^{2}\left\|\hat{v}_{pert}^{\tau_i} - \mathrm{sg}\left(\hat{v}_{clean}^{\tau_i}\right)\right\|_2^2
$$

**含义**: 强制扰动样本的速度场预测与原始样本一致，stop-gradient 防止双侧塌缩。

**符号说明**:
- $\hat{v}_{pert}^{\tau_i}$: 扰动样本在时间步 $\tau_i$ 的速度场预测
- $\mathrm{sg}(\cdot)$: stop-gradient 算子，阻断梯度回传
- $\hat{v}_{clean}^{\tau_i}$: 原始样本的速度场预测（作为固定目标）

---

### 公式 4: [[对抗扰动生成|OC 对抗扰动]]

$$
\tilde{v}_t = v_t + \eta_v \cdot \frac{\nabla_{v_t}\mathcal{L}_{EC}}{\left\|\nabla_{v_t}\mathcal{L}_{EC}\right\|_2}, \quad \eta_v = \min\!\left(\alpha,\, \varepsilon_{adv}\left\|v_t\right\|_2\right)
$$

**含义**: 利用 EC 损失对视觉特征的 L2 归一化梯度方向生成对抗扰动，扰动幅度与特征范数成比例，模拟视觉噪声；机器人状态 $q_t$ 用相同方式生成 $\tilde{q}_t$。

**符号说明**:
- $\eta_v$: 自适应扰动步长，上界为 $\alpha$ 和特征范数缩放 $\varepsilon_{adv}\|v_t\|_2$ 的最小值
- $\alpha = 0.01$: 绝对扰动上界
- $\varepsilon_{adv} = 0.03$: 相对扰动比例系数
- $\nabla_{v_t}\mathcal{L}_{EC}$: EC 损失关于视觉特征的梯度

---

### 公式 5: [[自适应加权|总训练损失]]

$$
\mathcal{L}_{total} = \mathcal{L}_{SFT} + \lambda_j \mathcal{L}_{C}, \quad \mathcal{L}_{C} = \frac{1}{2}\left(\mathcal{L}_{EC} + \mathcal{L}_{OC}\right)
$$

$$
\lambda_j = \frac{1}{1 + \mathcal{L}^{ema}_j}
$$

其中 $\mathcal{L}^{ema}_j$ 按指数移动平均更新：$\mathcal{L}^{ema}_j = \gamma \mathcal{L}^{ema}_{j-1} + (1-\gamma)\mathcal{L}_C$，$\gamma=0.95$，$\mathcal{L}^{ema}_0=100$。

**含义**: 通过 EMA 跟踪一致性损失幅度，自动平衡 SFT 与一致性正则化，训练初期（$\mathcal{L}_C$ 大）权重小，损失收敛后权重趋近 1，避免手动调参。

**符号说明**:
- $\mathcal{L}_{SFT}$: 监督微调损失（干净样本 + 扰动样本均值）
- $\mathcal{L}_{C}$: 一致性正则化损失（EC + OC 均值）
- $\lambda_j$: 第 $j$ 步的自适应权重
- $\mathcal{L}^{ema}_j$: 一致性损失的指数移动平均，$\gamma=0.95$

---

## 关键图表

### Figure 1: 问题动机

![Figure 1](https://arxiv.org/html/2605.19678v1/x1.png)

**说明**: 现有 VLA 模型在视觉观测变化（视角、光照、背景）和语义等价但措辞不同的语言指令下表现脆弱，说明缺乏跨分布鲁棒性。

### Figure 2: RoVLA 整体架构

![Figure 2](https://arxiv.org/html/2605.19678v1/x2.png)

**说明**: (a) 训练阶段：[[InternVL|InternVL3.5-2B]] 作为高层语义编码器，[[Diffusion Transformer|DiT]] 作为低层动作生成器，三种一致性约束（IC/EC/OC）联合施加。(b) 推理阶段：仅执行标准双系统前向传播 + K 步去噪，无额外开销。

### Figure 3: 真实世界评估任务

![Figure 3](https://arxiv.org/html/2605.19678v1/x3.png)

**说明**: 在 Franka Research 3 机器人上搭建五个桌面操作任务，机器人手腕装有相机，测试感知、指令理解和操作能力。

### Figure 4: LIBERO-Plus 定性结果

![Figure 4](https://arxiv.org/html/2605.19678v1/x4.png)

**说明**: 在多种扰动条件下（物体数量增加、指令表达变化、光照变化、相机模糊、视角变化），RoVLA 保持相对稳定的动作决策。

### Figure 5: RoboTwin 2.0 定性结果

![Figure 5](https://arxiv.org/html/2605.19678v1/x5.png)

**说明**: 在随机化环境下（物体姿态、场景布局、视觉外观变化），RoVLA 保持稳定的目标定位和执行轨迹。

### Figure 6: 真实世界定性结果

![Figure 6](https://arxiv.org/html/2605.19678v1/x6.png)

**说明**: 腕部相机视角下五个真实操作任务的执行快照，即使初始物体配置跨试次变化，RoVLA 仍保持相对稳定的执行轨迹。

---

### Table 1: LIBERO-Plus 零样本扰动泛化（Fine-tuned on LIBERO，Zero-shot Disturbance）

| Method | Pretrained | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|-----------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA | ✓ | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | **69.6** |
| OpenVLA-OFT_m | ✓ | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| WorldVLA | ✗ | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | ✓ | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π₀ | ✓ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | ✓ | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| InternVL-3.5+DiT | ✗ | 58.3 | 37.2 | 76.3 | 95.1 | 94.8 | 79.1 | 74.0 | 71.6 |
| **RoVLA (Ours)** | ✗ | **58.4** | 36.3 | **92.9** | **95.6** | **95.0** | **80.9** | 73.0 | **74.3** |

**说明**: 语言扰动分项较 backbone 提升 16.6 个百分点，整体超越所有未预训练方法，接近大规模预训练模型。

---

### Table 2: LIBERO-Plus 扰动暴露泛化（Fine-tuned on LIBERO-Plus）

| Method | Pretrained | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|-----------|--------|-------|----------|-------|------------|-------|--------|-------|
| π₀ | ✓ | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| π₀.₅ | ✓ | 70.3 | 41.7 | 81.1 | 97.3 | 94.6 | 71.8 | 84.9 | 75.7 |
| GR00T-N1.6 | ✓ | 92.6 | 33.5 | 80.1 | 93.6 | 95.4 | 93.6 | 75.0 | 79.4 |
| OpenVLA-OFT | ✓ | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.5 |
| InternVL-3.5+DiT | ✗ | 94.2 | 32.4 | 64.5 | 94.7 | 93.0 | 94.7 | 74.8 | 77.1 |
| **RoVLA (Ours)** | ✗ | **96.6** | 32.0 | **91.5** | **95.9** | **96.1** | **95.1** | 74.1 | **82.0** |

**说明**: 语言分项提升 27.0 个百分点（最大增益），各视觉扰动分项均有改善，整体超越 OpenVLA-OFT 和 GR00T-N1.6（均为大规模预训练模型）。

---

### Table 3: RoboTwin 2.0 结果（部分任务）

| Task | GO-1 Clean | GO-1 Rand. | π₀.₅ Clean | π₀.₅ Rand. | InternVL+DiT Clean | InternVL+DiT Rand. | **RoVLA Clean** | **RoVLA Rand.** |
|------|-----------|-----------|-----------|-----------|-------------------|-------------------|-----------------|-----------------|
| Click Alarmclock | 95.0 | 90.0 | 97.0 | 93.0 | 92.0 | 99.0 | 90.0 | 94.0 |
| Click Bell | 98.0 | 95.0 | 75.0 | 76.0 | 89.0 | 85.0 | **97.0** | **94.0** |
| Dump Bin Bigbin | 57.0 | 45.0 | 30.0 | 42.0 | 83.0 | 71.0 | **82.0** | 76.0 |
| Grab Roller | 99.0 | 99.0 | 90.0 | 89.0 | 88.0 | 96.0 | 80.0 | 87.0 |
| Handover Mic | 12.0 | 8.0 | 28.0 | 18.0 | 25.0 | 26.0 | **34.0** | **34.0** |
| Move Can Pot | 16.0 | 4.0 | 29.0 | 27.0 | 62.0 | 64.0 | **63.0** | 47.0 |
| Move Pillbottle Pad | 9.0 | 11.0 | 33.0 | 29.0 | 39.0 | 47.0 | **42.0** | **43.0** |
| Move Playingcard Away | 37.0 | 24.0 | 59.0 | 67.0 | 70.0 | 75.0 | 56.0 | **72.0** |
| Move Stapler Pad | 3.0 | 4.0 | 16.0 | 18.0 | 20.0 | 14.0 | **22.0** | 19.0 |
| Open Laptop | 65.0 | 60.0 | 19.0 | 35.0 | 49.0 | 46.0 | **75.0** | 60.0 |
| Place A2b Left | 33.0 | 36.0 | 62.0 | 60.0 | 65.0 | 63.0 | **69.0** | 65.0 |
| Place A2b Right | 31.0 | 22.0 | 62.0 | 57.0 | 52.0 | 68.0 | **64.0** | **74.0** |
| Place Bread Basket | 47.0 | 52.0 | 48.0 | 56.0 | 62.0 | 57.0 | **62.0** | **61.0** |
| Place Bread Skillet | 2.0 | 1.0 | 38.0 | 46.0 | 55.0 | 58.0 | **69.0** | 56.0 |
| Place Container Plate | 73.0 | 70.0 | 71.0 | 78.0 | **95.0** | 88.0 | 92.0 | 89.0 |
| Place Dual Shoes | 6.0 | 10.0 | 12.0 | 7.0 | 8.0 | 14.0 | **32.0** | **28.0** |
| Place Fan | 1.0 | 0.0 | 25.0 | 36.0 | 19.0 | 28.0 | **44.0** | **51.0** |
| Place Mouse Pad | 15.0 | 10.0 | 21.0 | 26.0 | 27.0 | 24.0 | **27.0** | **28.0** |
| Place Object Scale | 26.0 | 27.0 | 40.0 | 49.0 | 41.0 | 44.0 | **47.0** | **53.0** |
| Place Object Stand | 56.0 | 63.0 | 74.0 | 65.0 | 68.0 | 68.0 | 72.0 | **76.0** |
| Place Phone Stand | 30.0 | 37.0 | 49.0 | 53.0 | 55.0 | 57.0 | **59.0** | **66.0** |
| Place Shoe | 15.0 | 13.0 | 57.0 | 61.0 | 72.0 | 68.0 | **80.0** | 72.0 |
| Rotate Qrcode | 22.0 | 9.0 | 47.0 | 56.0 | 51.0 | 56.0 | 48.0 | **64.0** |
| Scan Object | 1.0 | 2.0 | 42.0 | 38.0 | 52.0 | 29.0 | **52.0** | **44.0** |
| Shake Bottle Horizontally | 97.0 | 92.0 | 96.0 | **100.0** | 90.0 | 89.0 | 89.0 | 94.0 |
| **Total (25 tasks)** | 37.8 | 36.2 | 43.0 | 43.8 | 44.9 | 45.4 | **48.2** | **50.0** |

**说明**: RoVLA 在随机化环境中表现反而优于清洁环境（50.0% vs 48.2%），表明减少了对特定场景配置的过拟合。较 backbone 提升 3.3–4.6 点，较 π₀.₅ 提升 5.2–6.2 点。

---

### Table 4: 真实世界任务成功率

| 任务 | GR00T-N1.6 | InternVL-3.5+DiT | **RoVLA (Ours)** |
|------|-----------|-----------------|-----------------|
| Pick Up Banana | 80 | 60 | **80** |
| Pick Up Apple | 50 | 60 | **70** |
| Put Banana in Bowl | 70 | 50 | **80** |
| Put Apple in Bowl | 40 | 20 | **50** |
| Put Apple in Drawer | 10 | 0 | **20** |
| **Total** | 50 | 38 | **60** |

**说明**: RoVLA 整体成功率 60%，比 GR00T-N1.6 高 10 个百分点，比 backbone 高 22 个百分点。4/5 任务取得最佳，Pick Up Banana 与 GR00T-N1.6 持平（80%）。

---

### Table 5: 消融实验（Disturbance-exposed，LIBERO-Plus）

| IC | EC | OC | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|----|----|----|--------|-------|----------|-------|------------|-------|--------|-------|
| ✗ | ✗ | ✗ | 94.2 | 32.4 | 64.5 | 94.7 | 93.0 | 94.7 | 74.8 | 77.1 |
| ✗ | ✓ | ✗ | 94.6 | 32.5 | 68.4 | 94.0 | 94.5 | 94.2 | 77.0 | 78.2 |
| ✓ | ✓ | ✗ | 93.7 | 30.6 | **89.5** | 94.7 | 93.5 | 94.6 | 73.8 | 80.5 |
| ✗ | ✓ | ✓ | 95.3 | 35.6 | 69.6 | 94.5 | 95.4 | **95.6** | 74.5 | 79.0 |
| ✓ | ✓ | ✓ | **96.6** | 32.0 | **91.5** | **95.9** | **96.1** | **95.1** | 74.1 | **82.0** |

**关键发现**: IC 对语言鲁棒性贡献主导（从 68.4% 跳至 89.5%）；EC+OC 对噪声扰动最有效（95.6%）；Camera 和 Background 分项在三者组合下均达最优；Layout 分项提升有限（74.8% → 74.1%，略有下降），是当前主要瓶颈。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO-Plus | 15,874 条扰动增强演示 | 涵盖语言/视觉/本体感知扰动 | 训练 + 测试 |
| LIBERO (原始) | 1,693 条演示，4 个任务套件 | 标准操作基准 | 预训练基础 |
| RoboTwin 2.0 | 2,500 清洁 + 25,000 随机化演示 | 域随机化，多场景 | 训练 + 测试 |
| 真实世界 | 125 条轨迹 | Franka Research 3，腕部相机 | 真实评估 |

### 实现细节

- **语义编码器**: InternVL3.5-2B（取第 16 层输出）
- **动作生成器**: DiT（32 层 Transformer）
- **动作建模**: Conditional Flow Matching（条件流匹配）
- **优化器**: AdamW，学习率 $1\times10^{-4}$
- **学习率调度**: 线性 warmup（前 5% 步）+ 余弦衰减
- **训练步数**: LIBERO-Plus / 真实世界 60k 步；RoboTwin 2.0 120k 步
- **OC 超参数**: $\alpha=0.01$, $\varepsilon_{adv}=0.03$
- **推理**: 8 步去噪（前向欧拉积分）
- **指令增强**: Qwen3-8B 生成约 15 条变体 / 轨迹

### 可视化结果

LIBERO-Plus 定性结果显示 RoVLA 在物体数量增加、光照变化、相机模糊、视角变化和语言变体五种扰动下均保持稳定动作序列。RoboTwin 2.0 随机化环境下目标定位和轨迹执行稳定。真实世界任务中，抓取香蕉/苹果和放入碗中的成功率较高，放入抽屉任务难度最大（20%）。

---

## 批判性思考

### 优点

1. **零推理开销**: 三种一致性约束全部在训练阶段施加，推理时与标准 VLA 无差异
2. **正交互补**: IC 针对语言，EC 针对生成过程，OC 针对视觉/本体感知，三者分工明确且协同增益
3. **自适应权重**: EMA 自动平衡 SFT 和一致性损失，避免手动调参
4. **通用性强**: 框架不依赖特定 backbone，理论上可迁移到其他 VLA

### 局限性

1. **细粒度接触和双臂协调改善有限**: 对需要精确力控或复杂双臂协调任务效果不显著
2. **真实世界任务偏简单**: 五个任务仍属于拾放范畴，缺乏更复杂的操作评估
3. **IC 依赖 LLM 离线生成**: 需要预先为所有轨迹生成指令变体，增加数据准备成本

### 潜在改进方向

1. 引入更强的几何与动力学约束，改善接触精度
2. 将 OC 扩展到任务级/语义级一致性，而非仅速度场层面
3. 在线/实时指令增强，减少对离线 LLM 预处理的依赖

### 可复现性评估

- [x] 代码开源（GitHub: HCPLab-SYSU/RoVLA）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（优化器、步数、超参数均已报告）
- [x] 数据集可获取（LIBERO-Plus、RoboTwin 2.0 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[pi0]]: 流匹配动作生成骨干，RoVLA 基于类似双系统设计
- [[Diffusion Transformer]]: DiT 作为低层动作生成器
- [[InternVL]]: InternVL3.5-2B 作为语义编码器

### 对比

- [[GR00T-N1]]: NVIDIA 双系统 VLA，在 LIBERO-Plus 达到 79.4%，被 RoVLA (82.0%) 超越
- [[OpenVLA-OFT]]: 对比基线，LIBERO-Plus 达到 79.5%

### 方法相关

- [[条件流匹配]]: 核心动作建模方式
- [[对抗训练]]: OC 模块的设计思路
- [[一致性正则化]]: EC 和 OC 的共同理论基础
- [[Qwen3]]: 用于生成指令变体的 LLM

### 硬件/数据相关

- [[Franka Research 3]]: 真实世界实验使用的机器人平台
- [[LIBERO-Plus]]: 主要评估 benchmark（扰动增强版）
- [[RoboTwin 2.0]]: 域随机化 benchmark

---

## 速查卡片

> [!summary] RoVLA: Multi-Consistency Constraints for Robust VLA
> - **核心**: 三重一致性约束（IC/EC/OC）系统增强 VLA 对语言变体和视觉扰动的鲁棒性
> - **方法**: 离线 LLM 指令增强 + 去噪过程一致性 + 对抗扰动一致性，全部在训练阶段施加
> - **结果**: LIBERO-Plus 语言分项 +27pp，整体 82.0%（超越 GR00T-N1.6 和 OpenVLA-OFT）
> - **代码**: https://github.com/HCPLab-SYSU/RoVLA

---

*笔记创建时间: 2026-05-20*
