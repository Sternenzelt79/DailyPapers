---
title: "MolmoAct2: Action Reasoning Models for Real-world Deployment"
method_name: "MolmoAct2"
authors: [Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, Ranjay Krishna]
year: 2026
venue: arXiv
tags: [vla, embodied-ai, flow-matching, action-tokenizer, world-model, bimanual-manipulation, fully-open]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.02881
created: 2026-05-09
---

# 论文笔记：MolmoAct2: Action Reasoning Models for Real-world Deployment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Allen Institute for AI (AI2), University of Washington, University of Michigan, NVIDIA |
| 日期 | May 2026 |
| 项目主页 | https://allenai.org/blog/molmoact2 |
| 对比基线 | [[Pi05]], [[Pi0]], [[GR-ER]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.02881) / [Code](https://github.com/allenai/molmoact2) |

---

## 一句话总结

> 完全开源的 [[VLA|视觉-语言-动作模型]]，通过 [[MolmoER|具身推理 VLM]] 主干、[[OpenFAST|频域动作分词器]] 与 [[Flow Matching|流匹配]] 动作专家，在 720 小时双臂数据上实现多本体（DROID/SO-100/双臂 YAM）部署。

---

## 核心贡献

1. **全栈开源 [[VLA]] 系统**: 开放权重、训练代码、约 12.5M 样本的训练数据与最大公开 [[Bimanual Manipulation|双臂]] 数据集 ([[BimanualYAM]] 720 小时, 34.5k 演示, 28 任务)
2. **[[MolmoER]] 主干**: 在 13 个具身推理基准上达到 63.8% 平均分，超越 GPT-5 (57.9%) 与 GR-ER 1.5 Thinking (61.3%)；相比 Molmo2 baseline 提升 17 个百分点
3. **[[OpenFAST]] 动作分词器**: 频域 + BPE 的离散动作 tokenizer，2048 token 词表，统一 32 维动作空间
4. **[[Flow Matching]] 动作专家 + KV 注入**: 借助 [[Cross-Attention|VLM KV 缓存]] 的 [[Action Expert|动作专家]]，无需重算 VLM
5. **[[MolmoAct2-Think]] 自适应深度推理**: 利用时序冗余跳过未变化场景区域的深度更新（cos 相似度阈 0.996）
6. **多本体真机部署**: DROID 单臂 87.1%、SO-100 平台 56.7%、LIBERO 97.2% 平均成功率，全面优于 [[Pi05]]/[[Pi0]] 基线

---

## 问题背景

### 要解决的问题

当前 [[VLA]] 模型作为机器人通用控制器存在三大短板：(1) 对真实世界部署关键的指标（鲁棒性、长程任务、多本体迁移）表现不佳；(2) 大多数 SOTA 模型（[[Pi05]]、GR-2 等）权重/数据/代码部分或全部闭源，社区难以复现；(3) 双臂高维动作空间缺乏开放规模数据。

### 现有方法的局限

- [[Pi05]] / [[Pi0]] 等闭源系统不释放训练代码与训练数据，难以二次开发
- 现有 [[Action Tokenizer|动作分词]] 方案（[[FAST]]、原始 quantization）领域受限或词表臃肿
- 主流 [[VLM]] 主干（Qwen-VL、LLaVA）在具身常识/空间指代/3D 理解上不专门
- [[Bimanual Manipulation|双臂]] 公开数据集体量小（如 ALOHA <100 小时），不足以训练通用策略

### 本文的动机

构建"真正可部署"的开放 [[VLA]]：用具身专用 [[VLM]] (Molmo2-ER) 提供更好的视觉推理基础，用 [[Flow Matching]] 动作专家保留连续控制精度，用 [[OpenFAST]] 在 token 化与连续生成之间灵活切换，并通过自建大规模双臂数据填补开放生态空缺。

---

## 方法详解

### 模型架构

[[MolmoAct2]] 采用 **VLM + Action Expert 解耦** 架构：

- **输入**: 语言指令 $l$ + 多视角观测 $\{o_t^{cam_i}\}$ + 本体状态 $s_t$
- **Backbone**: [[MolmoER|Molmo2-ER]] (4B)，在 12.51M 样本上训练（其中 3.26M 为新增具身 QA / pointing / 视频 QA）
- **核心模块**:
  - [[Action Expert]] 通过 [[Cross-Attention]] 从 VLM 各层 KV 中读取条件，公式见 [[#公式 4 与 5：KV 投影与交叉注意力]]
  - [[OpenFAST]] 离散 tokenizer 用于预训练；推理时切换为 [[Flow Matching]] 连续动作生成
- **输出**: 连续 [[Action Chunking|动作块]] $a_{t:t+H} \in \mathbb{R}^{H \times 32}$，并补 padding 到 32 维以兼容多本体
- **总参数**: ≈4B (VLM) + 动作专家若干百 M

### 核心模块

#### 模块 1: [[MolmoER|Molmo2-ER 主干]]

**设计动机**: 通用 [[VLM]] 在具身任务上缺乏空间推理能力 ([[Pointing]]、[[Embodied QA]]、第一/三人称对齐)。

**Specialize-then-Rehearse 训练方案**:
- 第一阶段：在具身专用数据 (1.33M Embodied QA + 780K Pointing + 700K Ego-Exo 多图等) 上专门化
- 第二阶段：与原 Molmo2 通用语料混合演练 (rehearse)，保持通用 VL 能力
- 数据配方见 [[#Table 1：Molmo2-ER 训练数据混合]]

#### 模块 2: [[OpenFAST]] 动作分词器

**设计动机**: 让连续动作可被 [[LLM Backbone|LLM]] 端预训练。

**三阶段流程**:
1. 把 1 秒动作轨迹做 [[Frequency-Domain Transform|频域变换]]（DCT 类）
2. 系数按 1–99 percentile 统计 normalize 后量化
3. [[Byte-Pair Encoding|BPE]] 压缩为 2048 词表的离散 token；夹爪指令单独通道

混合训练分布：BimanualYAM 30% / SO-100/101 30% / DROID 30% / Fractal+BC-Z+Bridge 10%（详见 [[#Table 2：OpenFAST 训练混合]]）

#### 模块 3: [[Action Expert]] + [[Flow Matching]]

**设计动机**: 保留连续控制精度 + 复用 VLM 计算。

**KV 投影 + 交叉注意力**：动作专家在每层将 VLM 的 $K^{vlm}_\ell, V^{vlm}_\ell$ 投影到自己的隐藏维 $d_h$，再与动作 token $h_\ell$ 做 [[Cross-Attention]]。详细公式见下节。

#### 模块 4: [[MolmoAct2-Think]] 自适应深度推理

**设计动机**: 真实操作场景大多数时间帧间冗余，重复计算深度浪费算力。

**关键机制**:
- 把深度图量化为 $10 \times 10$ 网格 = 100 空间码位 + 128 个学习深度码值
- 每帧 RGB patch 与上一帧做余弦相似度，低于 0.996 才更新对应格子
- 通过 mask $M_t$ 控制哪些 token 参与重新计算，再以 sigmoid 门 $g_\ell$ 调制
- 公式参见 [[#公式 9 与 11：自适应深度更新与门控]]

---

## 关键公式

### 公式 1: [[Flow Matching|流插值]]

$$
x_t = (1-t)\,\varepsilon + t\,a, \qquad u = a - \varepsilon
$$

**含义**: 在动作目标 $a$ 与高斯噪声 $\varepsilon$ 之间做线性插值，$u$ 为 [[Velocity Field|条件速度场]]的目标。

**符号说明**:
- $a \in \mathbb{R}^{H \times 32}$: 真实动作块
- $\varepsilon \sim \mathcal{N}(0, I)$: 噪声起点
- $t \sim \mathcal{U}(0, 1)$: 时间步
- $u$: 速度目标

### 公式 2: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}\left[\,\| m \odot (f_\theta(x_t, t, c) - u) \|_2^2\,\right]
$$

**含义**: 训练动作专家 $f_\theta$ 拟合速度场，$m$ 是动作维度有效掩码（不同本体维度不同）。

**符号说明**:
- $f_\theta$: [[Action Expert]]
- $c$: 来自 [[MolmoER]] 的语言-视觉条件
- $m$: 动作维度掩码（处理 32 维 padding）

### 公式 3: [[Action Expert|动作自注意力块]]

$$
h'_\ell = h_\ell + g^{sa}_\ell \cdot \text{SA}(h_\ell)
$$

**含义**: 动作 token 内部自注意力，门控 $g^{sa}_\ell$ 控制更新强度。

### 公式 4 与 5：KV 投影与交叉注意力

$$
\tilde{K}_\ell = \text{reshape}(P_K K^{vlm}_\ell), \qquad \tilde{V}_\ell = \text{reshape}(P_V V^{vlm}_\ell)
$$

$$
\text{CA}(Q_\ell, \tilde{K}_\ell, \tilde{V}_\ell) = \text{softmax}\!\left(\frac{Q_\ell \tilde{K}_\ell^\top}{\sqrt{d_h}}\right)\tilde{V}_\ell
$$

**含义**: 动作专家**复用** VLM 已经算好的 $K/V$，只需用投影矩阵 $P_K, P_V$ 把 VLM 的隐藏维投影到动作专家维度，从而在不重算 [[VLM Backbone|VLM]] 的前提下条件化动作生成。

**符号说明**:
- $K^{vlm}_\ell, V^{vlm}_\ell$: 第 $\ell$ 层 VLM 的 KV
- $P_K, P_V$: 可学习线性投影
- $Q_\ell$: 动作 token 经过自注意力后的 query
- $d_h$: 动作专家头维度

### 公式 6: 跨注意力 + MLP 残差

$$
\bar{h}_\ell = h'_\ell + g^{ca}_\ell \cdot \text{CA}(\cdot), \qquad h_{\ell+1} = \bar{h}_\ell + g^{ff}_\ell \cdot \text{MLP}(\bar{h}_\ell)
$$

**含义**: 标准 Transformer block，三处加门控 $g^{sa}, g^{ca}, g^{ff}$，便于稳定训练 + 允许"跳过"某些层。

### 公式 7: [[Multi-Sample Flow Matching|多样本流匹配损失]]

$$
\mathcal{L}_{\text{flow}}(a, c) = \frac{1}{K}\sum_{i=1}^{K}\| m \odot (f_\theta(x_{t_i}, t_i, c) - (a - \varepsilon_i)) \|_2^2
$$

**含义**: 对同一动作样本采 $K$ 个 $(t_i, \varepsilon_i)$ 计算损失，降方差。预训练 $K=4$，微调升至 $K=8$。

### 公式 8 与 9：[[Adaptive Depth Reasoning|自适应深度更新]] 与门控

$$
m_{t,i} = \mathbb{1}\!\left[\,\cos(x_{t,i},\, x_{t-1,i}) < 0.996\,\right]
$$

$$
c_\ell = \frac{\sum_t A_t (1-M_t) V^{vlm}_{\ell,t}}{\sum_t A_t (1-M_t)}, \qquad g_\ell = \sigma(w_\ell c_\ell + b_\ell)
$$

**含义**: 第一式确定哪些 patch 需要重新计算（低相似度的格子置 1）；第二式聚合"未更新"区域的 VLM value 得到上下文向量 $c_\ell$，再过 sigmoid 得到深度门 $g_\ell$。

**符号说明**:
- $x_{t,i}$: 第 $t$ 帧第 $i$ 格的 RGB patch embedding
- $A_t$: 注意力权重
- $M_t$: 更新 mask（0 表示保持旧值）
- $\sigma$: sigmoid

### 公式 10: 门控 KV 条件化

$$
\bar{K}^{vlm}_{\ell,t} = (1 - M_t + M_t g_\ell)\, K^{vlm}_{\ell,t}
$$

**含义**: 对未更新区域 ($M_t=0$) 直接保留旧 KV，对更新区域用门控值 $g_\ell$ 缩放，节省深度推理算力。

### 公式 11: [[Joint Training|联合训练目标]]

$$
\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{flow}}
$$

**含义**: post-training 同时优化 [[Cross-Entropy|语言建模损失]]（保住 VLM 的语言 / [[OpenFAST]] token 预测能力）与流匹配动作损失，防止灾难性遗忘。

---

## 关键图表

### Figure 1: Multi-platform Deployment Overview

![Figure 1](https://arxiv.org/html/2605.02881v1/x5.png)

**说明**: [[MolmoAct2]] 在 DROID Franka 单臂、SO-100/101 桌面双臂、BimanualYAM 双臂 YAM 三种本体上统一部署，展示"一个权重训练 → 多个本体微调"的能力矩阵。

### Figure 2: Training Data Composition

![Figure 2](https://arxiv.org/html/2605.02881v1/x6.png)

**说明**: 训练语料结构图。底层是 12.51M 样本的 [[MolmoER]] 数据，上层是按 30/30/30/10 比例的 [[OpenFAST]] 与机器人轨迹数据，覆盖具身 QA、pointing、ego-exo 视频、抽象推理等多种来源。

### Figure 3: BimanualYAM Collection Setup

![Figure 3](https://arxiv.org/html/2605.02881v1/x7.png)

**说明**: 自建数据集 [[BimanualYAM]] 的硬件 (<$6,000)、双臂 YAM 工作台、专家遥操作流程、严格的失败重试协议。两个月采集 720 小时 / 34.5k demo / 28 任务。

### Figure 4: MolmoAct2 Architecture

![Figure 4](https://arxiv.org/html/2605.02881v1/x8.png)

**说明**: 核心架构图。左：[[MolmoER]] 处理多视角图像 + 语言；右：[[Action Expert]] 通过 [[Cross-Attention]] 读取 VLM 每层投影后的 KV，输出 [[Flow Matching]] 速度场。门控 $g^{sa}, g^{ca}, g^{ff}$ 实现稳定训练。

### Figure 5: MolmoAct2-Think Adaptive Depth

![Figure 5](https://arxiv.org/html/2605.02881v1/figures/MAF44.png)

**说明**: [[MolmoAct2-Think]] 流程：RGB patch 与上一帧比较 → cos<0.996 的格子置 mask → 仅这些区域执行深度推理 → 门控融合回主干。利用时序冗余降低实时部署算力。

### Figure 6: RoboEval Benchmark Comparison

![Figure 6](https://arxiv.org/html/2605.02881v1/figures/MAF6.png)

**说明**: [[RoboEval]] 多任务对比柱状图。MolmoAct2 在长程任务（Pack Box、Rotate Valve）上提升最显著，并在 efficiency / stability / trajectory quality 上同样领先 [[Pi05]]。

### Figure 7: Efficient Fine-tuning on Real Robots

![Figure 7](https://arxiv.org/html/2605.02881v1/figures/MAF5.png)

**说明**: 展示用 64–128 H100 在 100K updates 内即可在新本体上微调到收敛。多 flow 样本数从 $K=4$ 升到 $K=8$ 显著加速收敛。

### Table 1：[[MolmoER]] 训练数据混合

| 子集 | 样本数 | 权重 |
|------|--------|------|
| Image Embodied QA | 1.33M | 11% |
| Image Pointing | 780K | 11% |
| Image Detection | 100K | 1% |
| Video Embodied QA | 703K | 10% |
| Multi-image / Ego-Exo | 700K | 9% |
| Abstract Reasoning | 150K | 4% |
| **Molmo2-ER 子总和** | **3.26M** | **46%** |
| 通用 Molmo2 语料（rehearse） | 9.25M | 54% |
| **总计** | **12.51M** | **100%** |

**说明**: 验证 specialize-then-rehearse 配方：约一半算力用于具身专用数据，另一半保通用 VL 能力。

### Table 2：[[OpenFAST]] 训练混合

| 数据源 | 占比 |
|--------|------|
| [[BimanualYAM]] | 30% |
| SO-100/101 | 30% |
| [[DROID]] | 30% |
| Fractal + BC-Z + Bridge | 10% |

### Table 3：具身推理基准 (13 项平均)

| 模型 | 平均分 |
|------|--------|
| Molmo2 baseline | 46.8% |
| GR-ER 1.5 Thinking | 61.3% |
| GPT-5 | 57.9% |
| **Molmo2-ER (ours)** | **63.8%** |

**说明**: 比基线 Molmo2 提升 17 pp，超越 GPT-5 5.9 pp。

### Table 4：MolmoSpace 任务

| 模型 | 平均成功率 |
|------|-----------|
| [[Pi05]] | 34.5% |
| **MolmoAct2-DROID** | **37.7%** |

### Table 5：LIBERO 仿真 held-out

| 模型 | 平均成功率 |
|------|-----------|
| Pi0.5-DROID | 10.0% |
| **MolmoAct2-DROID** | **20.6%** |

**说明**: 跨 embodiment 泛化优势显著（2 倍）。

### Table 6：DROID 真机操作

| 模型 | 平均成功率 |
|------|-----------|
| Pi0.5-DROID | 45.2% |
| **MolmoAct2-DROID** | **87.1%** |

**说明**: 真机上接近 2 倍提升，验证开放数据 + 具身 VLM 的协同。

### Table 7：SO-100 平台

| 模型 | 平均成功率 |
|------|-----------|
| π0-SO100/101 | 45.3% |
| **MolmoAct2-SO100/101** | **56.7%** |

### Table 8：[[LIBERO]] 全套

| 模型 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| [[Pi05]] | — | — | — | — | 96.9% |
| **MolmoAct2** | 97.8 | 100.0 | 97.8 | 93.2 | **97.2** |
| **MolmoAct2-Think** | — | — | — | — | **98.1** |

**说明**: Think 变体在 Long-horizon 任务上额外提升，验证 [[Adaptive Depth Reasoning]] 的有效性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[BimanualYAM]] | 720 h / 34.5k demo / 28 任务 | 最大开放双臂；两月专家采集；硬件 <$6k | 预训练 + 微调 |
| [[DROID]] | ~76k 轨迹 | 单臂 Franka 多场景 | 预训练 + 微调 |
| SO-100/101 | 自采 | 桌面双臂低成本 | 预训练 + 微调 |
| Fractal / BC-Z / Bridge | 公开 | 跨本体补充 | 预训练 |
| [[LIBERO]] | 4 套 130 任务 | 仿真泛化基准 | 评估 |
| [[RoboEval]] | 多任务 | 真机评估协议 | 评估 |

### 实现细节

- **VLM 主干**: [[MolmoER|Molmo2-ER-4B]]
- **Pre-training**: 200K steps, seq=4,200, global batch=128, 64×H100, ≈5,760 GPU-h
  - VLM lr: $5 \times 10^{-6}$；LM lr: $1 \times 10^{-5}$
- **Post-training**: 100K updates, seq=2,100 (robot) / 4,200 (VLM), batch=128, ≈2,300 GPU-h
  - Action expert lr: $5 \times 10^{-5}$
- **Fine-tuning**: 100K updates, batch=64–128, 32–64 H100/embodiment
- **Multi-sample flow**: 预训练 $K=4$，微调 $K=8$
- **Action 维度**: 统一 padding 到 32

### 可视化结果

- 长程任务 Pack Box / Rotate Valve 上轨迹平滑度显著优于 Pi0.5
- BimanualYAM 双臂协调任务（穿绳、双手分拣）成功率领先
- MolmoAct2-Think 在场景静态阶段近乎不更新深度，部署延迟降低（论文未给具体加速比）

---

## 批判性思考

### 优点

1. **真正的开放生态**: 数据 + 权重 + 训练代码 + 自建大规模双臂数据集，远超 [[Pi05]]、GR-2 等"半开"系统
2. **统一动作空间**: 32 维 padding + [[OpenFAST]] 让多本体共享词表，跨本体迁移 (DROID → LIBERO) 提升 2 倍
3. **VLM 复用 KV**: [[Action Expert]] 不重算 VLM，部署时算力友好
4. **具身专用 [[VLM]]**: Molmo2-ER 在 13 个基准上以 4B 参数超越 GPT-5，证明具身 + rehearse 配方有效
5. **自适应深度**: [[MolmoAct2-Think]] 提供了"动态计算"思路，结合时序冗余非常自然

### 局限性

1. **算力门槛仍高**: 5,760 + 2,300 + 微调 GPU-h，社区难以从零复现
2. **MolmoAct2-Think 加速量化缺失**: 论文未给出延迟/能耗具体数字（"temporal redundancy" 是定性的）
3. **OpenFAST 与 Flow Matching 切换**: 预训练用 token，推理用 flow，理论一致性需更细分析
4. **真机评估仍偏 [[Manipulation|操作]]**: 缺导航 / 全身 / 移动操作场景
5. **依赖 Allen 定制双臂硬件**: BimanualYAM 虽便宜（<$6k）但仍非通用平台

### 潜在改进方向

1. 把 [[Adaptive Depth Reasoning]] 推广到完整 KV 缓存（不仅深度）
2. 引入 [[World Model|世界模型]] 做长程规划（当前仍是反应式）
3. [[Diffusion Model|扩散]]/[[Flow Matching|流]] 蒸馏到一步推理，降低实时延迟
4. 与 [[VGGT]]/3D 表征结合，弥补单纯 2D RGB 的空间感

### 可复现性评估

- [x] 代码开源 (`github.com/allenai/molmoact2`)
- [x] 预训练模型 (HuggingFace: MolmoAct2 / MolmoAct2-Think / MolmoAct2-Pretrain / Molmo2-ER-4B)
- [x] 训练细节完整 (lr / batch / GPU-h 全披露)
- [x] 数据集可获取 (BimanualYAM、SO-100/101、DROID、Molmo2-ER 全在 HF)

---

## 关联笔记

### 基于

- [[Molmo]]: 通用 VLM 主干，被 specialize 为 Molmo2-ER
- [[FAST]]: 频域动作分词的前身，[[OpenFAST]] 在其上扩 BPE
- [[Pi0]] / [[Pi05]]: 直接对照基线，验证 VLM+Flow 的范式

### 对比

- [[Pi05]]: 闭源对照，本工作在 DROID 真机超出近 2 倍
- [[GR-ER]]: 闭源具身推理 VLM，Molmo2-ER 以 4B 超越其 1.5 Thinking 版
- [[OpenVLA]]: 早期完全开源 VLA，但无双臂、无 Flow

### 方法相关

- [[Flow Matching]]: 动作专家核心
- [[Action Expert]]: VLA 解耦设计
- [[Cross-Attention]]: KV 复用机制
- [[OpenFAST]]: 动作分词
- [[Adaptive Depth Reasoning]]: Think 变体核心

### 硬件 / 数据相关

- [[BimanualYAM]]: 自建双臂数据集
- [[DROID]] / [[LIBERO]] / [[RoboEval]]: 评估基准

---

## 速查卡片

> [!summary] MolmoAct2: Action Reasoning Models for Real-world Deployment
> - **核心**: 完全开源的多本体 VLA，VLM 主干 + Flow 动作专家 + 自适应深度
> - **方法**: Molmo2-ER (4B) + OpenFAST (2048 词表) + Action Expert (KV 复用) + Adaptive Depth Think
> - **结果**: DROID 真机 87.1%（+42 pp）/ LIBERO 97.2% / Embodied 推理 63.8%（>GPT-5）
> - **数据**: BimanualYAM 720h 双臂 + 12.5M VLM 样本，全部开放
> - **代码**: https://github.com/allenai/molmoact2

---

*笔记创建时间: 2026-05-09*
