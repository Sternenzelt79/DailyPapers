---
title: "Flash-WAM: Modality-Aware Distillation for World Action Models"
method_name: "Flash-WAM"
authors: [Arman Akbari, Ci Zhang, Arash Akbari, Lin Zhao, Yixiao Chen, Weiwei Chen, Xuan Zhang, Geng Yuan, Yanzhi Wang]
year: 2026
venue: arXiv
tags: [world-action-model, consistency-distillation, diffusion-policy, robot-manipulation, model-compression, flow-matching, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.05254v1
created: 2026-06-05
---

# 论文笔记：Flash-WAM: Modality-Aware Distillation for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Northeastern University, Qualcomm AI Research |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[LingBot]]（LingBot-VA teacher） |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05254) / Code: 未公开 |

---

## 一句话总结

> Flash-WAM 通过模态感知的一致性蒸馏框架，将视频-动作联合扩散模型的推理速度提升 23×，在不显著损失性能的前提下将单次推理延迟从 8.1 秒压缩至 348 毫秒。

---

## 核心贡献

1. **模态不相容问题诊断**: 识别出视频流与动作流因 [[SNR-Shifted Scheduler|SNR偏移调度器]] 差异导致训练质量集中区间不同，标准 [[一致性蒸馏]] 在低噪声区间（动作域）产生梯度消失问题
2. **Proposition 1 理论保证**: 证明任何满足边界条件的一致性函数在 σ→0 时梯度缩放达到线性 O(σ) 当且仅当 b'(0)≠0；标准 LCM 仅达到二次 O(σ²)
3. **Flash-WAM 模态感知蒸馏**: 为动作流选用线性梯度缩放参数化（linear-gradient-scaling），为视频流保留方差保持 LCM 参数化，实现 23× 加速同时保留近教师级别性能

---

## 问题背景

### 要解决的问题

[[World Action Model|WAM]]（如 LingBot-VA）联合生成未来视频和机器人动作，依赖多步迭代扩散。LingBot-VA 需要 25 次视频去噪 + 50 次动作去噪，单控制块在 NVIDIA L40S 上耗时 8.1 秒，远超实时控制需求（通常 <1 秒）。

### 现有方法的局限

直接对联合视频-动作模型施加 [[一致性蒸馏]]（如 LCM）会导致任务成功率从 91.25% 崩溃至 23.97%。根本原因：

- 视频流使用 SNR 偏移调度器，训练质量集中在**高噪声**区间（大 σ）
- 动作流也使用 SNR 偏移调度器，但维度低，训练质量集中在**低噪声**区间（小 σ）
- 标准 LCM 的一致性函数 $f(x_\sigma, \sigma) = c_\text{skip}(\sigma) x_\sigma + c_\text{out}(\sigma) \hat{x}_0$ 在 σ→0 时梯度缩放为 O(σ²)，即**动作流几乎无梯度信号**

[[DMD2]] 在 1v/2a 时达到 78.74% 但无法做到 1v/1a（仅 50.56%）。

### 本文的动机

不同模态有不同的训练噪声制度（noise regime），应为每个模态选择**最适合其噪声制度**的一致性函数参数化形式，而非统一使用同一参数化。视频流在高噪声区稳定，适合方差保持参数化；动作流在低噪声区训练，需要线性梯度缩放参数化。

---

## 方法详解

### 模型架构

Flash-WAM 沿用 [[LingBot]]（LingBot-VA）的 **[[Transformer]]** 骨干架构（[[Flex Attention]]），在推理时学生模型与教师共享结构，差异仅在损失函数层面：

- **输入**: 语言指令 + 视觉观测 + 状态
- **Backbone**: 共享 Transformer + [[Flex Attention]]
- **核心模块**: 模态感知一致性函数（per-stream loss head）
- **输出**: 视频帧块 + [[Action Chunking|动作块]]
- **教师**: LingBot-VA（25v/50a 步）；**学生**: 1v/1a 或 1v/2a 步

### 核心模块

#### 模块1: SNR 偏移调度器分析

**设计动机**: 理解视频与动作噪声训练分布差异

LingBot-VA 使用 SNR 偏移 [[Flow Matching]] 调度器。对于偏移量 $s$，有效噪声水平 $\sigma$ 与原始采样 $\tilde{\sigma} \sim \mathcal{U}[0,1]$ 的关系：

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}
$$

视频维度高（$d_v \gg d_a$），其最优偏移 $s^*$ 更大，导致训练质量集中于高 σ（高噪声）。动作维度低，偏移小，训练质量集中于低 σ（低噪声，即接近真实数据）。

#### 模块2: 线性梯度缩放参数化（动作流）

**设计动机**: 在低 σ 区间提供线性而非二次的梯度信号

**具体实现**:
- 动作流一致性函数使用 [[v-prediction]] 形式：$f^a(x^a_\sigma, \sigma) = x^a_\sigma - \sigma \cdot v_\theta(x^a_\sigma, \sigma)$
- 该形式等价于 $a(\sigma) = 1, b(\sigma) = -\sigma$，满足 $b'(0) = -1 \neq 0$
- 由 Proposition 1 保证，梯度缩放为线性 O(σ)，在低 σ 区间不消失

#### 模块3: 方差保持参数化（视频流）

**设计动机**: 在高 σ 区间保持输出范围有界，数值稳定

**具体实现**:
- 视频流保留标准 LCM 参数化：$f^v(x^v_\sigma, \sigma) = c_\text{skip}(\sigma) x^v_\sigma + c_\text{out}(\sigma) \hat{x}^v_0$
- Karras 系数 $c_\text{skip}, c_\text{out}$ 在高 σ 区间输出范围受控
- 视频生成质量（FVD 等）不受影响

---

## 关键公式

### 公式1: [[Flow Matching|Flow Matching 训练目标]]

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{x_0, \epsilon, \sigma} \left\| v_\theta(x_\sigma, \sigma) - (\epsilon - x_0) \right\|^2
$$

**含义**: 基础 [[Flow Matching]] 目标，预测速度场 $v_\theta$，教师模型用此目标训练。

**符号说明**:
- $x_0$: 清洁数据
- $\epsilon$: 噪声
- $\sigma$: 噪声水平
- $x_\sigma = (1-\sigma)x_0 + \sigma\epsilon$: 加噪数据

### 公式2: [[SNR-Shifted Scheduler|SNR 偏移采样器]]

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0,1]
$$

**含义**: 通过偏移量 $s$ 控制训练质量在噪声谱上的集中区间，$s$ 越大训练越集中于高噪声。

**符号说明**:
- $s$: SNR 偏移量（视频 $s_v > s_a$ 动作）
- $\tilde{\sigma}$: 均匀采样的原始噪声

### 公式3: [[一致性蒸馏|一致性函数通用形式]]

$$
f(x_\sigma, \sigma) = a(\sigma) x_\sigma + b(\sigma) v_\theta(x_\sigma, \sigma)
$$

**含义**: 通用一致性函数参数化，映射任意噪声水平的样本到干净数据估计 $\hat{x}_0$。

**符号说明**:
- $a(\sigma), b(\sigma)$: 标量系数函数，需满足边界条件 $a(0)=1, b(0)=0$
- $v_\theta$: velocity 预测网络

### 公式4: [[一致性蒸馏|Consistency Distillation 损失]]

$$
\mathcal{L}_\text{CD} = d\!\left(f_{\theta_S}(x_{\sigma_s}, \sigma_s),\; f_{\theta_S'}(\tilde{x}_{\sigma_e}, \sigma_e)\right)
$$

**含义**: 要求去噪轨迹上相邻两点 $(\sigma_s, \sigma_e)$ 的一致性函数输出尽量接近，$d$ 为 Huber 距离。

**符号说明**:
- $\theta_S$: 学生网络参数
- $\theta_S'$: EMA 目标网络参数
- $\sigma_s > \sigma_e$: 起点与终点噪声水平
- $\tilde{x}_{\sigma_e}$: 用教师 ODE 从 $x_{\sigma_s}$ 步进一步到 $\sigma_e$ 的样本

### 公式5: [[Proposition 1|Proposition 1 梯度缩放定理]]

**定理**: 对任意满足边界条件的一致性函数 $f$，在 σ→0 时有 $|b(\sigma)| = O(\sigma)$，且该界当且仅当 $b'(0) \neq 0$ 时可达（最优缩放为线性 O(σ)）。

**推论**: 标准 LCM（Karras 系数）满足 $b(\sigma) = O(\sigma^2)$（二次缩放），在低 σ 区间梯度信号二次消失。

### 公式6: [[Flash-WAM|动作流一致性函数（线性梯度缩放）]]

$$
f^a(x^a_\sigma, \sigma) = x^a_\sigma - \sigma \cdot v_\theta(x^a_\sigma, \sigma)
$$

**含义**: 动作流专用参数化，$b(\sigma) = -\sigma$ 满足 $b'(0) = -1 \neq 0$，实现最优线性梯度缩放。

**符号说明**:
- $x^a_\sigma$: 动作流的加噪样本
- $v_\theta$: 共享 Transformer 的 velocity 预测头（动作部分）

### 公式7: [[Flash-WAM|视频流一致性函数（方差保持）]]

$$
f^v(x^v_\sigma, \sigma) = c_\text{skip}(\sigma) x^v_\sigma + c_\text{out}(\sigma) \hat{x}^v_0
$$

**含义**: 视频流保留标准 LCM 方差保持参数化，在高 σ 区间输出有界，数值稳定。

**符号说明**:
- $c_\text{skip}(\sigma)$: Karras skip 系数
- $c_\text{out}(\sigma)$: Karras output 系数
- $\hat{x}^v_0$: 网络对清洁视频的估计

### 公式8: [[Flash-WAM|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}^v + \lambda_a \cdot \mathcal{L}^a
$$

**含义**: 视频与动作蒸馏损失加权求和，通过 $\lambda_a$ 平衡两路学习信号。

**符号说明**:
- $\mathcal{L}^v$: 视频流一致性蒸馏损失
- $\mathcal{L}^a$: 动作流一致性蒸馏损失
- $\lambda_a = 1.0$: 动作损失权重

---

## 关键图表

### Figure 1: 推理延迟与成功率对比

![Figure 1a - 推理延迟](https://arxiv.org/html/2606.05254v1/x2.png)

![Figure 1b - 成功率对比](https://arxiv.org/html/2606.05254v1/x3.png)

**说明**: (a) 各方法单块推理延迟，Flash-WAM 在 NVIDIA L40S 上达到 348ms，低于实时控制预算；(b) RoboTwin 2.0 平均成功率，[[DMD2]] 和朴素 LCM 均显著下降，Flash-WAM 保持接近教师的性能。

### Figure 2: Flash-WAM 框架总览

![Figure 2 - Flash-WAM Overview](https://arxiv.org/html/2606.05254v1/x4.png)

**说明**: 左：诊断——朴素 [[一致性蒸馏]] 为何在联合视频-动作模型上失败（动作流低 σ 区间梯度消失）；中：Flash-WAM 训练流水线，视频和动作各用适配的一致性函数；右：部署时单步去噪自回归生成视频与动作。

### Figure 3: 真实世界 Unitree G1 评测套件

![Figure 3 - Real World Evaluation](https://arxiv.org/html/2606.05254v1/x5.png)

**说明**: 在 Unitree G1 人形机器人上评测三个操作任务（T1/T2/T3），Flash-WAM (1v/2a) 平均成功率 60%，显著优于 LingBot-VA 降低步数基线（40%）和 Video-only LCM（43.3%）。

### Figure 4: RoboTwin 定性对比

![Figure 4 - Qualitative Comparison](https://arxiv.org/html/2606.05254v1/x6.png)

**说明**: RoboTwin 任务"pick_diverse_bottles"的开环定性对比，Flash-WAM 生成的视频与动作质量接近教师模型。

### Table 1: RoboTwin 2.0 综合结果

| 方法 | $N^v$ | $N^a$ | Clean | Rand. | Average | Speedup |
|------|-------|-------|-------|-------|---------|---------|
| π₀ | – | – | 65.92 | 58.40 | 62.2 | – |
| π₀.₅ | – | – | 82.74 | 76.76 | 79.8 | – |
| X-VLA | – | – | 72.9 | 72.8 | 72.8 | – |
| Motus | – | – | 88.66 | 87.02 | 87.8 | – |
| LingBot-VA (Teacher) | 25 | 50 | 91.64 | 90.86 | 91.25 | 1.0× |
| LingBot-VA + DMD2 | 1 | 2 | 85.08 | 72.36 | 78.74 | 19.0× |
| LingBot-VA + Video-only LCM | 1 | 2 | 80.66 | 76.92 | 78.79 | – |
| LingBot-VA + Naive Joint LCM | 1 | 2 | 25.88 | 22.07 | 23.97 | – |
| **Flash-WAM (Ours)** | **1** | **2** | **88.42** | **82.66** | **85.54** | **–** |
| LingBot-VA + DMD2 | 1 | 1 | 52.66 | 48.46 | 50.56 | 23.3× |
| LingBot-VA + Video-only LCM | 1 | 1 | 77.90 | 69.46 | 73.68 | – |
| LingBot-VA + Naive Joint LCM | 1 | 1 | 39.68 | 32.96 | 36.32 | – |
| **Flash-WAM (Ours)** | **1** | **1** | **82.56** | **80.26** | **81.41** | **23.3×** |

**关键发现**: Naive Joint LCM 崩溃至 23.97%，Flash-WAM 在 1v/1a 下仍达 81.41%（教师 91.25%），超越 DMD2（50.56%）约 30 个百分点。

### Table 2: LIBERO 基准结果

| 方法 | $N^v$ | $N^a$ | Spatial | Object | Goal | Long | Average | Speedup |
|------|-------|-------|---------|--------|------|------|---------|---------|
| π₀ | – | – | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | – |
| X-VLA | – | – | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | – |
| LingBot-VA (Teacher) | 20 | 50 | 98.5 | 99.8 | 98.0 | 98.3 | 98.6 | 1.0× |
| Video-only LCM | 1 | 2 | 95.1 | 92.0 | 96.0 | 97.8 | 95.2 | 13.7× |
| **Flash-WAM (Ours)** | **1** | **2** | **97.0** | **92.8** | **96.4** | **98.0** | **95.7** | **13.7×** |
| Video-only LCM | 1 | 1 | 95.0 | 91.5 | 95.0 | 95.4 | 94.2 | 16.3× |
| **Flash-WAM (Ours)** | **1** | **1** | **96.0** | **92.6** | **96.0** | **95.8** | **95.1** | **16.3×** |

**关键发现**: LIBERO 上 Flash-WAM 与 Video-only LCM 差距更小（95.7% vs 95.2%），说明 LIBERO 动作难度相对较低；但在长时程任务（Long）Flash-WAM 更佳（98.0% vs 97.8%）。

### Table 3: 真实世界 Unitree G1 人形机器人结果

| 方法 | $N^v / N^a$ | T1 | T2 | T3 | Average |
|------|-------------|----|----|----|---------| 
| LingBot-VA (Teacher) | 33 / 10 | 50% | 70% | 80% | 66.7% |
| LingBot-VA (reduced NFE) | 1 / 2 | 30% | 30% | 60% | 40.0% |
| LingBot-VA + Video-only LCM | 1 / 2 | 30% | 50% | 50% | 43.3% |
| **Flash-WAM (Ours)** | **1 / 2** | **50%** | **60%** | **70%** | **60.0%** |
| LingBot-VA (reduced NFE) | 1 / 1 | 10% | 30% | 30% | 23.3% |
| LingBot-VA + Video-only LCM | 1 / 1 | 20% | 40% | 40% | 33.3% |
| **Flash-WAM (Ours)** | **1 / 1** | **40%** | **50%** | **60%** | **50.0%** |

**关键发现**: 真实世界任务中 Flash-WAM (1v/2a) 以 60% 成功率最接近教师（66.7%），物理部署中优势更明显。

### Table 4: 消融实验（RoboTwin 不同 Horizon）

| 方法 | $N^v$ | $N^a$ | H=1 Clean | H=1 Rand. | H=2 Clean | H=2 Rand. | H=3 Clean | H=3 Rand. | Avg Clean | Avg Rand. |
|------|-------|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| LingBot-VA (Teacher) | 25 | 50 | 94.18 | 93.56 | 90.35 | 86.95 | 93.22 | 93.28 | 92.93 | 91.55 |
| Video-only LCM | 1 | 2 | 87.10 | 82.73 | 73.13 | 68.19 | 62.50 | 68.25 | 80.66 | 76.92 |
| Video-only LCM + reg | 1 | 2 | 91.53 | 88.50 | 83.00 | 74.69 | 68.00 | 62.75 | 86.92 | 82.02 |
| Naive Joint LCM | 1 | 2 | 41.00 | 35.13 | 4.00 | 3.13 | 0.00 | 0.00 | 25.88 | 20.08 |
| **Flash-WAM** | **1** | **2** | **92.30** | **88.47** | **84.88** | **76.63** | **73.50** | **63.25** | **88.42** | **82.66** |
| Video-only LCM | 1 | 1 | 85.57 | 78.17 | 72.06 | 61.81 | 43.75 | 34.75 | 77.90 | 69.46 |
| Video-only LCM + reg | 1 | 1 | 66.87 | 61.07 | 39.19 | 35.56 | 10.25 | 4.75 | 53.48 | 48.40 |
| Naive Joint LCM | 1 | 1 | 54.63 | 46.00 | 21.56 | 15.63 | 0.00 | 0.00 | 39.68 | 32.96 |
| **Flash-WAM** | **1** | **1** | **87.30** | **86.93** | **78.44** | **72.63** | **63.50** | **60.75** | **82.56** | **80.26** |

**关键发现**:
1. Naive Joint LCM 在 Horizon=3 时完全崩溃（0%），而 Flash-WAM 仍保持 73.5%
2. Video-only LCM + reg（辅助正则化）在 1v/1a 下性能骤降（53.48% vs 88.42%），说明辅助正则化无法替代模态感知参数化

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 多任务仿真 | Clean/Randomized 两种环境，3 个操作 Horizon | 主要仿真基准 |
| LIBERO | 4 个任务套件 | Spatial/Object/Goal/Long 四个子集 | 泛化性验证 |
| Unitree G1 Real | 3 个真实任务 (T1/T2/T3) | 人形机器人真实操作 | 物理部署验证 |

### 实现细节

- **Backbone**: LingBot-VA（共享 Transformer + Flex Attention）
- **优化器**: AdamW（β₁=0.9, β₂=0.999），学习率 5×10⁻⁶
- **梯度裁剪**: 2.0
- **Warmup**: 100 步
- **Batch Size**: 48（4× H100）
- **EMA 衰减**: α = 0.995
- **动作损失权重**: λ_a = 1.0
- **动作正则权重**: λ_r = 0.2
- **CFG 范围**: [2.0, 10.0]
- **图像分辨率**: 128×128
- **动作维度**: 30
- **每视频帧动作数**: 4
- **帧块大小 K**: 4
- **硬件**: NVIDIA L40S（推理）、4× H100（训练）

### 可视化结果

Figure 4 展示 RoboTwin "pick_diverse_bottles" 任务的开环视频生成对比：Flash-WAM 生成的视频帧在语义连贯性和运动轨迹准确性上均接近教师模型，优于朴素蒸馏方法。

---

## 批判性思考

### 优点
1. **理论严谨**: Proposition 1 从梯度缩放角度为模态感知参数化提供了充分必要条件，理论基础扎实
2. **工程实用**: 修改仅限于损失头，无需改动架构，可直接嫁接到任意 WAM 类模型
3. **真实验证**: 在 Unitree G1 人形机器人上的真实部署结果有说服力

### 局限性
1. **依赖教师模型**: 需要 LingBot-VA 作为教师，蒸馏质量上限受教师限制；真实世界教师仅 66.7%
2. **单教师适配**: 目前仅验证 LingBot-VA，对不同 SNR 偏移系数的 WAM 是否普适需验证
3. **图像分辨率低**: 仅 128×128，复杂场景泛化能力存疑

### 潜在改进方向
1. 探索 flow-based distillation（而非 consistency distillation）在多模态场景的适配
2. 将模态感知参数化扩展到三模态以上（如语音 + 视觉 + 动作）
3. 结合 [[Action Chunking]] 的时序结构进一步提升压缩比

### 可复现性评估
- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（附录有详细超参数）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均公开）

---

## 关联笔记

### 基于
- [[LingBot]]: Flash-WAM 的教师模型（LingBot-VA），提供联合视频-动作生成框架
- [[一致性蒸馏]]: Flash-WAM 的核心蒸馏方法论基础（LCM 框架）
- [[Flow Matching]]: 教师模型的生成框架

### 对比
- [[Distribution Matching Distillation|DMD2]]: 另一类扩散蒸馏方法，1v/1a 下性能大幅下降（50.56%）
- [[WAM]]: World Action Model 的通用框架概念

### 方法相关
- [[一致性蒸馏]]: 核心蒸馏范式，Flash-WAM 对其进行模态感知扩展
- [[SNR-Shifted Scheduler]]: 导致视频/动作训练制度不一致的根本原因
- [[v-prediction]]: 动作流采用的参数化形式

### 硬件/数据相关
- [[RoboTwin]]: 主要仿真评测基准
- [[LIBERO]]: 泛化性评测基准
- [[Unitree G1]]: 真实部署人形机器人平台

---

## 速查卡片

> [!summary] Flash-WAM
> - **核心**: 模态感知一致性蒸馏，解决联合视频-动作 WAM 的 23× 加速难题
> - **方法**: 动作流用线性梯度缩放参数化（v-prediction），视频流用方差保持 LCM；联合损失 $\mathcal{L} = \mathcal{L}^v + \lambda_a \mathcal{L}^a$
> - **结果**: RoboTwin 2.0 从 8.1s → 348ms，成功率 91.25% → 81.41%（1v/1a）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-05*
