---
title: "CapVector: Learning Transferable Capability Vectors in Parametric Space for Vision-Language-Action Models"
method_name: "CapVector"
authors: [Wenxuan Song, Han Zhao, Fuhao Li, Ziyang Zhou, Xi Wang, Jing Lyu, Pengxiang Ding, Yan Wang, Donglin Wang, Haoang Li]
year: 2026
venue: arXiv
tags: [vla, model-merging, parameter-space, capability-transfer, fine-tuning, sim-to-real, cross-embodiment]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.10903
created: 2026-05-13
---

# 论文笔记：CapVector

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确列出（综合多机构） |
| 日期 | May 2026 |
| 项目主页 | 未知 |
| 对比基线 | [[OpenVLA]] / Spatial Forcing / [[Pi05\|π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.10903) |

---

## 一句话总结

> CapVector 从参数空间中提取"能力向量"（capability vectors），将辅助训练目标带来的能力增益与任务特异性参数解耦，使能力在不同 VLA 任务和机器人平台上可迁移。

---

## 核心贡献

1. **能力向量提取**: 通过对比标准 SFT 与辅助目标 SFT 的参数差，从 [[Model Merging|模型参数空间]] 中解耦出"能力向量"（capability vectors），将辅助目标（如空间感知）内化为可复用的参数偏移
2. **正交正则化**: 提出 [[Orthogonal Regularization|正交正则化]] 损失，强制下游微调的参数更新方向与能力向量正交，防止微调过程中能力退化
3. **通用迁移框架**: 在 [[OpenVLA]]、StarVLA、[[Pi05|π0.5]] 三种 VLA 架构上验证，能力向量可跨仿真环境（LIBERO→RoboTwin）和跨机器人平台迁移，同时将计算开销降低至 <0.8% 内存增量

---

## 问题背景

### 要解决的问题

预训练 [[VLA]] 模型在下游任务微调时，往往需要引入辅助训练目标（如空间感知约束、链式思维推理）来提升泛化性能。然而，这些辅助目标在每次新任务微调时都要重新计算，带来显著的计算和内存开销（+17% 内存、+28% FLOPs）。

### 现有方法的局限

- **辅助目标 SFT**（如 Spatial Forcing、LaRA-VLA）：每次微调都需在线运行辅助目标，无法复用已学得的能力
- **[[Model Merging|模型合并]]** 方法（如 MergeVLA、RETAIN）：关注多技能组合或防止遗忘，但未专门研究辅助目标能力的迁移性
- **[[LoRA]]** 等参数高效微调：减少参数规模但未解决辅助目标的计算开销

### 本文的动机

辅助训练目标带来的能力增益（如空间理解能力）本质上是编码在参数变化中的。如果能将这部分参数偏移与任务特异性更新解耦，就可以"一次提取、多次复用"，在推理和下游微调中均不再需要运行辅助目标。

---

## 方法详解

### 整体框架

CapVector 分为**提取阶段**（离线，一次性）和**使用阶段**（每次下游微调时），核心思想是将能力向量 $\gamma_{ao}$ 视为可插拔的参数偏移。

- **输入**: 能力提取数据集 $\mathcal{D}_{ext}$（小规模）+ 预训练 [[VLA]] 模型 $\theta_{pt}$
- **离线提取**: 对 $\mathcal{D}_{ext}$ 分别训练标准 SFT 和辅助目标 SFT，做差得到 $\gamma_{ao}$
- **合并初始化**: $\theta_{meta} = \theta_{pt} + \alpha \cdot \gamma_{ao}$，作为下游微调的起点
- **在线防护**: 下游微调时加入 [[Orthogonal Regularization|正交正则化]] 损失，阻止 $\gamma_{ao}$ 被覆盖

### 核心模块

#### 模块1: 能力向量提取（Capability Vector Extraction）

**设计动机**: 利用 [[Model Merging|参数差分]] 思想，将辅助目标引入的参数变化中的"能力成分"单独分离

**关键假设**: 辅助目标 SFT 和标准 SFT 在同一 $\mathcal{D}_{ext}$ 上训练时，任务特异性参数变化近似相同（$\Delta_{ft} \approx \delta_{ao}$），因此两者差值近似为纯能力增益 $\gamma_{ao}$。

**具体实现**:
- 在 $\mathcal{D}_{ext}$ 上用标准 [[Supervised Fine-Tuning|SFT]] 训练 $\theta_{ft}$
- 在 $\mathcal{D}_{ext}$ 上用辅助目标 SFT（如 Spatial Forcing）训练 $\theta_{ao}$
- 计算差值：$\gamma_{ao} = \theta_{ao} - \theta_{ft}$
- 加权合并：$\theta_{meta} = \theta_{pt} + \alpha \cdot \gamma_{ao}$（$\alpha=1.1$ 为最优值）

**视觉丰富度要求**: $\mathcal{D}_{ext}$ 的视觉多样性至关重要——使用仅单一背景的数据会导致 shortcut learning，能力向量质量显著下降（见 Figure 3 和 Table 4）

#### 模块2: 正交正则化（Orthogonal Regularization）

**设计动机**: 下游微调产生的参数更新 $\Delta'_{ft}$ 若与 $\gamma_{ao}$ 方向相近，会逐渐覆盖已内化的能力，导致性能退化

**数学约束**: 要求每层参数矩阵中，能力向量与任务更新方向正交：

$$
\langle \gamma_{ao}^{(p)}, \Delta_{ft}^{\prime(p)} \rangle = 0
$$

**实现**：对所有参数矩阵逐元素计算乘积之和作为正则损失（见公式7），权重 $\lambda = 10^{-4}$ 时效果最佳

**[[LoRA]] 下的处理**: 仅在矩阵 $A$（更新方向矩阵）上计算正交性，与 $B$ 无关

---

## 关键公式

### 公式1: [[Supervised Fine-Tuning|标准 SFT 参数分解]]

$$
\theta_{ft} = \theta_{pt} + \Delta_{ft}
$$

**含义**: 标准微调将预训练模型参数 $\theta_{pt}$ 偏移 $\Delta_{ft}$，得到任务特化模型

**符号说明**:
- $\theta_{pt}$: 预训练 VLA 模型参数
- $\Delta_{ft}$: 标准 SFT 引入的参数偏移（含任务特异性更新）

---

### 公式2: [[Model Merging|辅助目标 SFT 分解]]

$$
\theta_{ao} = \theta_{pt} + \Delta_{ao} = \theta_{pt} + \delta_{ao} + \gamma_{ao}
$$

**含义**: 辅助目标 SFT 的参数偏移可分解为任务特异部分 $\delta_{ao}$ 和能力增益 $\gamma_{ao}$

**符号说明**:
- $\delta_{ao}$: 与任务相关的参数变化（与 $\Delta_{ft}$ 近似相同）
- $\gamma_{ao}$: 辅助目标带来的"能力向量"——这是本文要提取的关键

---

### 公式3: [[Model Merging|能力向量提取公式]]

$$
\gamma_{ao} = \theta_{ao} - \theta_{ft}
$$

**含义**: 在同一 $\mathcal{D}_{ext}$ 上，辅助目标 SFT 与标准 SFT 的参数差近似等于纯能力增益

**符号说明**:
- $\theta_{ao}$: 辅助目标 SFT 得到的模型参数
- $\theta_{ft}$: 标准 SFT 得到的模型参数

---

### 公式4: [[Model Merging|能力增强元模型]]

$$
\theta_{meta} = \theta_{pt} + \alpha \cdot \gamma_{ao}
$$

**含义**: 将能力向量加权合并进预训练模型，得到能力增强的元模型作为下游微调起点

**符号说明**:
- $\alpha$: 合并权重，消融实验得最优值 $\alpha = 1.1$
- $\gamma_{ao}$: 提取的能力向量

---

### 公式5: 下游微调

$$
\theta'_{ft} = \theta_{meta} + \Delta'_{ft}
$$

**含义**: 从能力增强的元模型出发进行下游任务微调

---

### 公式6: [[Orthogonal Regularization|正交约束条件]]

$$
\langle \gamma_{ao}^{(p)}, \Delta_{ft}^{\prime(p)} \rangle = 0
$$

**含义**: 要求每层 $p$ 的参数矩阵中，能力向量与任务更新向量内积为零（方向正交），防止任务更新覆盖已有能力

---

### 公式7: [[Orthogonal Regularization|正交正则化损失]]

$$
\mathcal{L}_{orth}(\gamma_{ao}, \Delta'_{ft}) = \sum_p \sum_{i,j} \left| \gamma_{ao,ij}^{(p)} \cdot \Delta_{ft,ij}^{\prime(p)} \right|
$$

**含义**: 对所有参数矩阵逐元素乘积取绝对值后求和，近似衡量正交约束违反程度

**符号说明**:
- $\gamma_{ao,ij}^{(p)}$: 第 $p$ 层能力向量在位置 $(i,j)$ 的元素
- $\Delta_{ft,ij}^{\prime(p)}$: 第 $p$ 层任务更新在位置 $(i,j)$ 的元素

---

### 公式8: [[Supervised Fine-Tuning|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{action} + \lambda \cdot \mathcal{L}_{orth}
$$

**含义**: 下游微调时联合优化动作预测损失和正交正则化损失，$\lambda = 10^{-4}$ 时效果最优

**符号说明**:
- $\mathcal{L}_{action}$: 标准行为克隆（BC）动作预测损失
- $\lambda$: 正则化权重，消融得最优值 $10^{-4}$

---

## 关键图表

### Figure 1: CapVector 整体框架示意

![Figure 1](https://arxiv.org/html/2605.10903v1/x1.png)

**说明**: 灰线表示标准微调路径（$\theta_{pt} \to \theta_{ft}$），黄线表示辅助目标 SFT 路径（$\theta_{pt} \to \theta_{ao}$），粉色虚线为提取的能力向量 $\gamma_{ao}$。能力向量合并进预训练模型后，下游微调从 $\theta_{meta}$ 出发，正交正则化确保 $\Delta'_{ft}$ 不覆盖 $\gamma_{ao}$。

---

### Figure 2: LIBERO-Long 训练曲线与正则化权重消融

![Figure 2](https://arxiv.org/html/2605.10903v1/x2.png)

**说明**:
- (a) 在 LIBERO-Long 上，CapVector 收敛速度明显快于 Spatial Forcing 和 OpenVLA-OFT，早期训练（5k steps）即显示显著优势
- (b) 正交正则化权重 $\lambda$ 的消融：$\lambda = 10^{-4}$ 为最优，过大过小均损害性能

---

### Figure 3: 提取数据集视觉丰富度的影响

![Figure 3](https://arxiv.org/html/2605.10903v1/x3.png)

**说明**: 使用 LIBERO-Spatial（单一背景）提取的能力向量性能不如 LIBERO-Long（多背景），说明 $\mathcal{D}_{ext}$ 的视觉多样性是能力向量质量的关键——单一视觉环境导致 shortcut learning，能力向量无法泛化。

---

### Figure 4: UR3 工业任务真实机器人环境设置

![Figure 4](https://arxiv.org/html/2605.10903v1/x4.png)

**说明**: UR3 机器人的工业任务真实环境配置，用于验证 sim-to-real 迁移，任务包括真实工厂场景中的操作任务。

---

### Figure 5: UR3 真实机器人实验结果

![Figure 5](https://arxiv.org/html/2605.10903v1/x5.png)

**说明**: CapVector 在 UR3 机器人上的真实世界工业任务实验结果，验证了从仿真到真实机器人的能力向量迁移有效性。

---

### Figure 6: 跨机器人平台部署（开箱即用）

![Figure 6](https://arxiv.org/html/2605.10903v1/x6.png)

**说明**: 在 ARX Lift 2（6 自由度双臂）和 AgileX Cobot 上的跨机器人平台部署，能力向量无需额外训练即可直接迁移，体现了"开箱即用"的跨机器人泛化能力。

---

### Figure S1: 跨机器人平台环境设置（附录）

![Figure S1](https://arxiv.org/html/2605.10903v1/x7.png)

**说明**: ARX Lift 2 和 AgileX Cobot 机器人的详细实验环境配置说明（附录图）。

---

### Table 1: LIBERO 域内对比实验

| 训练阶段 | 方法 | Spatial | Object | Goal | Long | Average |
|---------|------|---------|--------|------|------|---------|
| 5k Steps | Spatial Forcing ($\theta_{ao}$) | 93.8% | 94.8% | 94.6% | 66.6% | 87.5% |
| | OpenVLA-OFT ($\theta_{ft}$) | 87.0% | 99.8% | 92.8% | 48.8% | 82.1% |
| | CapVector w/o orth. loss | 96.0% | 99.0% | 97.4% | 68.0% | 90.1% |
| | **CapVector (ours)** | **98.0%** | **99.2%** | **96.6%** | **73.0%** | **91.7%** |
| 1 Epoch | Spatial Forcing | 98.4% | 99.6% | 97.8% | 84.8% | 95.2% |
| | OpenVLA-OFT | 97.0% | 99.8% | 96.4% | 70.4% | 90.9% |
| | CapVector w/o orth. loss | 98.4% | 100.0% | 98.0% | 86.0% | 95.6% |
| | **CapVector (ours)** | **98.6%** | **99.8%** | **97.6%** | **90.0%** | **96.5%** |
| 8 Epochs | Spatial Forcing | 93.0% | 98.2% | 98.4% | 87.2% | 94.2% |
| | OpenVLA-OFT | 92.8% | 98.2% | 97.8% | 87.0% | 93.9% |
| | CapVector w/o orth. loss | 97.6% | 97.8% | 96.6% | 92.2% | 96.1% |
| | **CapVector (ours)** | **98.0%** | **98.0%** | **96.8%** | **93.6%** | **96.6%** |
| 150k Steps | Spatial Forcing | 97.2% | 99.2% | 96.8% | 94.2% | 96.9% |
| | OpenVLA-OFT | 96.8% | 94.8% | 92.8% | 86.2% | 92.7% |
| | CapVector w/o orth. loss | 97.4% | 99.0% | 97.2% | 91.2% | 96.2% |
| | **CapVector (ours)** | **98.4%** | **98.4%** | **96.8%** | **94.8%** | **97.1%** |

**关键发现**: CapVector 在各训练阶段均优于或持平 Spatial Forcing，且不需要在训练时运行辅助目标；正交损失在早期训练（5k steps、1 Epoch）提升明显。

---

### Table 2: RoboTwin 2.0 跨域迁移实验

| 方法 | Turn switch | Handover block | Handover mic | Place shoe | Pick dual bottles | Place object basket | Put bottles dustbins | Place phone stand | Put object cabinet | Stack bowls two | Avg. |
|-----|-------------|----------------|--------------|-----------|-------------------|---------------------|----------------------|-------------------|-------------------|-----------------|------|
| OpenVLA-OFT | 33.0% | 1.0% | 4.0% | 2.0% | 1.0% | 7.0% | 1.0% | 1.0% | 9.0% | 8.0% | 6.7% |
| + Spatial Forcing | 47.0% | 23.0% | **100.0%** | 2.0% | 17.0% | 24.0% | 15.0% | 17.0% | 23.0% | 63.0% | 33.1% |
| CapVector (Dext=Spatial) | 33.0% | 1.0% | **99.0%** | **13.0%** | **23.0%** | **39.0%** | **22.0%** | 12.0% | 17.0% | **59.0%** | 31.8% |
| CapVector (Dext=Long) | 18.0% | 12.0% | **99.0%** | **13.0%** | 6.0% | 33.0% | 4.0% | 5.0% | 21.0% | 26.0% | 23.7% |
| CapVector (Dext=90) | 36.0% | 1.0% | 0.0% | 2.0% | 1.0% | 11.0% | 3.0% | 0.0% | 9.0% | 7.0% | 9.4% |
| π0.5 | 3.0% | 44.0% | 9.0% | 3.0% | 27.0% | 7.0% | 12.0% | 1.0% | 34.0% | 11.0% | 15.1% |
| + Spatial Forcing | 5.0% | 0.0% | 15.0% | **19.0%** | **29.0%** | **18.0%** | **32.0%** | 27.0% | **72.0%** | 19.0% | 23.6% |
| CapVector (Merged VLM) | 4.0% | 1.0% | 14.0% | **19.0%** | 27.0% | **18.0%** | **32.0%** | 19.0% | **73.0%** | **22.0%** | 22.9% |
| CapVector (VLM + Expert) | 5.0% | 0.0% | 12.0% | **20.0%** | **32.0%** | 15.0% | 31.0% | **34.0%** | 64.0% | **22.0%** | 23.5% |

**关键发现**: 在 LIBERO 提取的能力向量迁移至 RoboTwin（跨仿真环境），CapVector 接近甚至部分超越在 RoboTwin 上从头运行 Spatial Forcing 的效果，验证了跨域迁移能力。

---

### Table 3: LaRA-VLA 辅助目标验证（StarVLA 骨干）

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| LaRA-VLA ($\theta_{ao}$) | 96.4% | 98.6% | 99.8% | 96.6% | 97.9% |
| StarVLA ($\theta_{ft}$) | 96.8% | 86.6% | 98.2% | 96.4% | 94.5% |
| **CapVector (ours)** | **96.6%** | **98.2%** | **99.2%** | **94.4%** | **97.1%** |

**关键发现**: CapVector 在 StarVLA 骨干上同样有效，接近使用 LaRA-VLA 辅助目标的完整训练效果，验证了方法的通用性。

---

### Table 4: 提取数据集视觉丰富度特征

| 基准 | LIBERO-Spatial | LIBERO-Long | LIBERO-90 | RoboTwin (Clean) | RoboTwin (Random) |
|-----|----------------|-------------|-----------|------------------|-------------------|
| 任务数 | 10 | 10 | 90 | 5 | 5 |
| 背景数 | 1 | 3 | 3 | 1 | 10k |
| 背景×目标对 | 10 | 10 | 90 | 5 | 50k |
| 每任务对数 | 1 | 1 | 1 | 1 | 10k |

**关键发现**: 视觉丰富度（背景多样性）是能力向量质量的决定性因素，LIBERO-Long（3背景）优于 LIBERO-Spatial（1背景）。

---

### Table S1: 合并权重 α 消融（OpenVLA-OFT，LIBERO 平均）

| α=0.5 | α=0.7 | α=0.9 | α=1.1 | α=1.3 | α=1.5 |
|-------|-------|-------|-------|-------|-------|
| 91.2% | 91.8% | 92.8% | **94.8%** | 90.6% | 92.4% |

**关键发现**: $\alpha = 1.1$ 为最优，轻微超过单位1的权重有助于放大能力增益信号。

---

### Table S2: 正交损失计算开销

| | FLOPs | GPU Memory |
|--|-------|-----------|
| Baseline | 17.9T | 62.8G |
| + Orthogonal Loss | 17.9T + 0.3G | 62.8G + 0.5G |

**关键发现**: 正交损失新增 FLOPs 仅约 0.002%，内存增量 <0.8%，可忽略不计。

---

### Table S3: 辅助目标 SFT 计算开销对比

| 方法 | FLOPs | GPU Memory |
|------|-------|-----------|
| OpenVLA-OFT | 17.9T | 62.8G |
| + Spatial Forcing | 17.9T + 5.0T | 62.8G + 10.9G |
| + CapVector (Ours) | 17.9T + 0.3G | 62.8G + 0.5G |

**关键发现**: CapVector 将辅助目标开销从 +28% FLOPs / +17% Memory 压缩至可忽略水平，核心收益是"一次提取、多次复用"的设计。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO-Spatial | 10 任务 | 单背景，空间排列变化 | 能力提取 / 域内测试 |
| LIBERO-Object | 10 任务 | 不同目标物体 | 域内测试 |
| LIBERO-Goal | 10 任务 | 不同目标状态 | 域内测试 |
| LIBERO-Long | 10 任务 | 3 背景，长程操作 | 能力提取 / 测试 |
| LIBERO-90 | 90 任务 | 多样化任务 | 能力提取对比 |
| RoboTwin 2.0 | 10 任务，100 rollouts/task | 双臂操作，跨域测试 | 跨域迁移测试 |

### 实现细节

- **骨干网络**: [[OpenVLA]]（自回归，LoRA）/ StarVLA（Flow Matching）/ [[Pi05|π0.5]]
- **OpenVLA-OFT 训练**: 1 GPU，batch size 8，150k steps
- **StarVLA 训练**: 8 GPUs，batch size 16，20k steps
- **π0.5 训练**: 4 GPUs，batch size 32，60k steps
- **合并权重**: $\alpha = 1.1$（最优）
- **正则化权重**: $\lambda = 10^{-4}$（最优）
- **评估**: 每条件 500 rollouts（LIBERO）/ 100 rollouts（RoboTwin）

### 可视化结果

- 跨机器人平台（UR3 工业、ARX Lift 2 双臂、AgileX Cobot）均验证了能力向量的 sim-to-real 和跨机器人迁移
- 视觉丰富度不足的 $\mathcal{D}_{ext}$ 会导致能力向量在新场景中出现明显性能下降（Figure 3）

---

## 批判性思考

### 优点

1. **理论动机清晰**: 能力向量提取基于合理的参数分解假设，公式推导自然
2. **通用性强**: 跨三种 VLA 架构（自回归、Flow Matching、混合）和多种辅助目标均有效
3. **计算效率显著**: 推理和下游微调均无需运行辅助目标，仅需 <0.8% 内存增量
4. **真实机器人验证**: 在 UR3、ARX Lift 2、AgileX Cobot 三种真实机器人上验证，增强可信度

### 局限性

1. **数据质量依赖强**: 提取数据集 $\mathcal{D}_{ext}$ 的视觉丰富度对能力向量质量有决定性影响，实际部署中难以保证
2. **假设近似性**: $\Delta_{ft} \approx \delta_{ao}$ 的假设在数据分布差异大时可能失效，理论边界未分析
3. **能力向量单一**: 当前框架每次只能提取一种辅助目标能力，多能力组合未充分探索
4. **跨域表现有限**: RoboTwin 上部分任务（如 Handover block）相比 Spatial Forcing 仍有差距

### 潜在改进方向

1. 研究多个能力向量的线性组合或自适应权重，实现多能力叠加
2. 分析 $\Delta_{ft} \approx \delta_{ao}$ 假设成立的充分条件，提供理论保证
3. 将能力向量提取自动化，设计无监督的能力发现框架

### 可复现性评估

- [ ] 代码开源（未提及）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（完整超参数已报告）
- [x] 数据集可获取（LIBERO、RoboTwin 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[OpenVLA]]: 核心骨干之一，CapVector 在其上验证并提取能力向量
- [[Pi05]]: 另一骨干，验证方法通用性
- [[LoRA]]: OpenVLA-OFT 使用 LoRA，正交正则化针对 LoRA 的 A 矩阵设计

### 对比

- [[Spatial Forcing]]: 主要对比辅助目标，CapVector 复用其能力但省去在线开销
- [[Model Merging]]: CapVector 属于参数空间合并的新视角，针对能力向量而非多技能合并

### 方法相关

- [[Model Merging]]: 核心技术基础——参数差分提取能力
- [[Orthogonal Regularization]]: 关键正则化手段，防止能力退化
- [[Supervised Fine-Tuning]]: SFT 是提取和使用能力向量的基础操作

### 硬件/数据相关

- [[LIBERO]]: 主要仿真 benchmark
- [[RoboTwin]]: 跨域迁移测试 benchmark

---

## 速查卡片

> [!summary] CapVector (2026)
> - **核心**: 从参数空间提取辅助目标能力向量，实现一次提取、多次复用
> - **方法**: $\gamma_{ao} = \theta_{ao} - \theta_{ft}$ 差分提取 + 正交正则化防退化
> - **结果**: LIBERO avg 97.1%，跨域迁移至 RoboTwin，真实机器人 sim-to-real 验证
> - **代码**: 未公开

---

*笔记创建时间: 2026-05-13*
