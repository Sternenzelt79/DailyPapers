---
title: "MolmoAct2: Action Reasoning Models for Real-world Deployment"
method_name: "MolmoAct2"
authors: [Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, Ranjay Krishna]
year: 2026
venue: arXiv
tags: [vla, embodied-ai, flow-matching, world-model, action-tokenizer, multi-embodiment, fully-open]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.02881v1
created: 2026-05-09
---

# 论文笔记：MolmoAct2: Action Reasoning Models for Real-world Deployment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Allen Institute for AI (AI2), University of Washington 等（推断） |
| 日期 | 2026-05 |
| 项目主页 | https://allenai.org/blog/molmoact2 |
| 代码 | https://github.com/allenai/molmoact2 |
| 模型权重 | [HuggingFace MolmoAct2 Collection](https://huggingface.co/allenai/MolmoAct2) |
| 对比基线 | [[Pi0]] / [[Pi05]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.02881) / [HTML](https://arxiv.org/html/2605.02881v1) / [Code](https://github.com/allenai/molmoact2) |

---

## 一句话总结

> 完全开源的多本体 [[VLA]] 系统：用 [[Flow Matching|流匹配]] 动作专家 + [[Adaptive Depth Reasoning|自适应深度推理]] 的 [[World Model|世界模型]]，将延迟、成本、性能三端同时打通。

---

## 核心贡献

1. **[[Molmo2-ER]] 具身推理 VLM 主干**：在 3.3M 具身样本上专门化训练，13 个具身推理 benchmark 平均 63.8%，超过 GPT-5 (57.9%) 与 Gemini 2.5 Pro (57.1%)。
2. **三个全开放数据集**：[[MolmoAct2-BimanualYAM]]（720 小时双臂遥操作）、[[MolmoAct2-SO100/101]]（184 小时低成本平台）、[[MolmoAct2-DROID]]（74,604 个 Franka episode）。
3. **[[OpenFAST]] 开放权重动作 tokenizer**：在 5 种本体上训练，将 32 维连续动作压缩为 2048 词表的离散 token。
4. **架构创新：[[KV-Cache Conditioning|KV-Cache 条件化]] 流匹配动作专家**：通过逐层 KV 注入而非 token 拼接接入 [[Flow Matching|流匹配]] 专家，避免推理阶段巨大开销。
5. **[[MolmoAct2-Think|MolmoAct2-Think]] 自适应深度推理**：选择性地重新预测变化区域的 [[Depth Token|深度 token]]，作为隐式 [[World Model|世界模型]] 引导动作生成，在 LIBERO 上达到 98.1% 平均成功率。

---

## 问题背景

### 要解决的问题

当今 [[VLA]] 系统在真实部署时同时存在三个短板：**延迟过高、硬件成本贵、性能仍不够**。要让一个通用机器人控制器同时具备：(1) 精准的具身/空间推理能力；(2) 低延迟连续动作生成；(3) 跨多种本体（双臂、Franka、低成本 SO-100）的部署能力。

### 现有方法的局限

- **[[Pi0]] / [[Pi05]]** 等闭源 VLA：权重/数据/tokenizer 不公开，复现成本高。
- **离散 token 自回归动作**：精度受 tokenizer 量化误差限制，长 chunk 推理慢。
- **拼接式 [[Flow Matching|流匹配]] 专家**：将动作 token 直接拼到 VLM 序列里，KV 缓存不可复用，每次推理都要重跑长序列。
- **[[Chain-of-Thought|推理链]] 风格的 VLA**（如显式语言推理）：每步推理代价大，不适合 30Hz 控制循环。

### 本文的动机

把"想"和"动"在结构上解耦：VLM 主干只负责一次前向得到 KV 缓存，[[Flow Matching|流匹配]] 动作专家通过 [[Cross-Attention|交叉注意力]] 复用这份缓存生成连续动作；进一步用 [[Adaptive Depth Reasoning|自适应深度推理]] 让"思考"的代价随场景变化而变化——大部分帧只需复用缓存的深度 token。

---

## 方法详解

### 模型架构

MolmoAct2 采用 **VLM 主干 + Flow-Matching 动作专家 + 可选深度推理头** 的三段式架构：

- **输入**：语言指令 $l$ + 多视角观测 $o_t$（外部相机 + 腕部相机）+ 机器人状态 $s_t$
- **Backbone**：[[Molmo2-ER]]（4B 参数），由 [[Molmo2]] 经"specialize-then-rehearse"在 3.3M 具身样本上专门化得到
- **动作 tokenizer**：[[OpenFAST]]（频域变换 → 量化 → [[BPE|字节对编码]] → 2048 词表）
- **核心模块**：
  - [[Flow Matching|流匹配]] 动作专家通过 [[KV-Cache Conditioning|KV-Cache 条件化]] 接入主干
  - [[Adaptive Depth Reasoning|自适应深度推理]] 头预测 10×10 网格 [[Depth Token|深度 token]]
- **输出**：[[Action Chunking|动作块]] $a_{t:t+k}$，长度 $k\in\{10, 15, 30\}$ 视频率而定
- **总参数**：~4B（VLM）+ 36 层动作专家

### 核心模块

#### 模块 1: [[Molmo2-ER]] 具身推理 VLM

**设计动机**：通用 VLM 在空间推理 / 具身 QA / 物理常识上偏弱，需要专门化。

**具体实现 — Specialize-then-Rehearse 训练**：
- **Stage 1（具身专门化）**：20K 步，在 3.3M [[Embodied QA]] / [[Image Pointing|图像指点]] / [[Video Pointing|视频指点]] 样本 + 8% [[Tulu-3]] 文本上训练，序列长 4,200，batch 64 × 8×H100。
- **Stage 2（联合精修）**：1.5K 步，混入原始 [[Molmo2]] 多模态语料（具身/通用 50/50），序列扩到 16,384 防止灾难性遗忘。

#### 模块 2: [[OpenFAST]] 动作 tokenizer

**设计动机**：闭源 [[FAST]] tokenizer 限制了开放生态。

**具体实现**：
- 频域变换（[[DCT|离散余弦变换]]）压缩 1 秒、32 维连续动作
- 1-99 百分位归一化
- 量化 + [[BPE|字节对编码]] → 2048 词表
- 训练数据：5 个本体 × 1M 序列（YAM / SO-100/101 / DROID-Franka / Fractal / BC-Z / Bridge）

#### 模块 3: [[Flow Matching|流匹配]] 动作专家 + [[KV-Cache Conditioning|KV-Cache 条件化]]

**设计动机**：传统流匹配动作专家把噪声动作 token 拼到 VLM 输入里，每次推理重算长序列；KV-Cache 条件化让 VLM 只前向一次。

**具体实现**：
- 36 层 [[DiT|DiT 风格]] 块，深度匹配 VLM
- 每块结构：[[Self-Attention|自注意力]] → 交叉注意力（attend 到 VLM KV 缓存的同层）→ [[MLP]]
- [[AdaRMS|AdaRMS]] 调制，时间嵌入产生 shift / scale / gate
- 通过线性 adapter $P_K, P_V$ 把 VLM 第 $\ell$ 层 KV 投影成动作专家可用的 $\tilde K_\ell, \tilde V_\ell$
- 训练用 4 个噪声样本/chunk，微调用 8 个

#### 模块 4: [[MolmoAct2-Think|MolmoAct2-Think]] 自适应深度推理

**设计动机**：[[Chain-of-Thought|思考链]] 太慢；但完全不思考又损失泛化。让"思考"按需触发。

**具体实现**：
- **深度表示**：将场景深度编码为 10×10 网格的 100 个深度 token，从 128 大小的 [[VQ Codebook|VQ 码本]] 取出
- **时间自适应**：把当前帧 RGB 切成 32×32 patch，与上一帧对应 patch 做 [[Cosine Similarity|余弦相似度]]；若 $<0.996$ 视为变化，重新预测对应 cell 的深度 token，否则复用上一帧的离散 code
- **门控注入**：未变化区域的 KV 通过学习到的 sigmoid 门 $g_\ell$ 弱化贡献
- **训练**：三种输出风格（depth-only / action-only / depth-and-action）等概率采样
- **推理**：自回归只解码变化的 cell，平均推理代价显著降低

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配插值]]

$$
x_t = (1-t)\varepsilon + t\, a, \qquad u = a - \varepsilon
$$

**含义**：动作 chunk $a$ 与噪声 $\varepsilon$ 的直线插值轨迹，速度场恒为 $a-\varepsilon$。

**符号说明**：
- $a \in \mathbb{R}^{k\times 32}$：未来 $k$ 步连续动作 chunk
- $\varepsilon \sim \mathcal{N}(0, I)$：等维噪声
- $t \sim \mathcal{U}(0,1)$：流时间
- $u$：目标速度场（直线方向）

### 公式 2: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{a,\varepsilon,t}\left[\left\| m \odot (f_\theta(x_t, t, c) - u) \right\|_2^2\right]
$$

**含义**：让动作专家 $f_\theta$ 在条件 $c$（VLM KV 缓存）下回归速度场。

**符号说明**：
- $f_\theta$：流匹配动作专家
- $c$：条件，等于 VLM 各层 KV 缓存
- $m$：动作维度 mask（处理可变 DoF）
- $\odot$：逐元素乘法

### 公式 3: 动作专家自注意力子块

$$
h'_\ell = h_\ell + g^{\text{sa}}_\ell \cdot \text{SA}\!\left(\text{AdaRMS}^{\text{sa}}_\ell(h_\ell, t)\right)
$$

**含义**：动作专家第 $\ell$ 层的自注意力残差更新，[[AdaRMS]] 由流时间 $t$ 调制。

**符号说明**：
- $h_\ell$：第 $\ell$ 层动作专家隐状态
- $g^{\text{sa}}_\ell$：自注意力分支的可学习 gate
- $\text{SA}$：自注意力算子

### 公式 4: 动作专家交叉注意力子块（KV-Cache 注入处）

$$
\bar h_\ell = h'_\ell + g^{\text{ca}}_\ell \cdot \text{CA}\!\left(\text{AdaRMS}^{\text{ca}}_\ell(h'_\ell, t),\; \tilde K_\ell, \tilde V_\ell\right)
$$

**含义**：动作专家通过交叉注意力 attend 到 VLM 同层的 KV 缓存——这是 [[KV-Cache Conditioning|KV-Cache 条件化]] 的关键。

**符号说明**：
- $\tilde K_\ell, \tilde V_\ell$：投影后的 VLM 第 $\ell$ 层 KV
- $g^{\text{ca}}_\ell$：交叉注意力 gate

### 公式 5: 动作专家 MLP 子块

$$
h_{\ell+1} = \bar h_\ell + g^{\text{ff}}_\ell \cdot \text{MLP}\!\left(\text{AdaRMS}^{\text{ff}}_\ell(\bar h_\ell, t)\right)
$$

**含义**：完成一层动作专家的标准 [[Transformer]] 块。

### 公式 6: [[KV-Cache Conditioning|KV 投影]]

$$
\tilde K_\ell = \text{reshape}(P_K K^{\text{vlm}}_\ell), \qquad \tilde V_\ell = \text{reshape}(P_V V^{\text{vlm}}_\ell)
$$

**含义**：用线性 adapter 把 VLM 的 KV 维度对齐到动作专家。

**符号说明**：
- $P_K, P_V$：可学习线性投影
- $K^{\text{vlm}}_\ell, V^{\text{vlm}}_\ell$：VLM 第 $\ell$ 层缓存的 K / V

### 公式 7: 交叉注意力实现

$$
\text{CA}(Q_\ell, \tilde K_\ell, \tilde V_\ell) = \text{softmax}\!\left(\frac{Q_\ell \tilde K_\ell^\top}{\sqrt{d_h}}\right)\tilde V_\ell
$$

**含义**：标准缩放点积注意力。$d_h$ 是单头维度。

### 公式 8: 多噪声样本流匹配损失

$$
\mathcal{L}_{\text{flow}}(a, c) = \frac{1}{K}\sum_{i=1}^K \left\| m \odot \left(f_\theta(x_{t_i}, t_i, c) - (a - \varepsilon_i)\right) \right\|_2^2
$$

**含义**：每个 chunk 用 $K=4$（预训练）或 $K=8$（微调）个独立噪声样本计算损失，降低方差，加速收敛。

### 公式 9: 后训练总损失

$$
\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{flow}}
$$

**含义**：联合训练离散动作 token（[[OpenFAST]]）的语言模型损失与连续动作的流匹配损失，使一份权重既能自回归生成 [[FAST Token]] 又能流匹配生成连续动作。

**符号说明**：
- $\mathcal{L}_{\text{LM}}$：[[Cross-Entropy|交叉熵]]，监督 OpenFAST 离散 token
- $\mathcal{L}_{\text{flow}}$：流匹配速度场 MSE

### 公式 10: [[Adaptive Depth Reasoning|自适应深度更新 mask]]

$$
\begin{aligned}
m_{t,i} &= \mathbf{1}\!\left[\cos(x_{t,i}, x_{t-1,i}) < 0.996\right] \\
b_{t,i} &= \begin{cases} d_{t,i} & \text{if } m_{t,i} = 1 \\ b_{t-1,i} & \text{if } m_{t,i} = 0 \end{cases}
\end{aligned}
$$

**含义**：第 $i$ 个 patch 的 RGB 在两帧间余弦相似度低于 0.996 才重新预测深度 token，否则沿用上一帧的离散 code。

**符号说明**：
- $x_{t,i}$：第 $t$ 帧、第 $i$ 个 32×32 patch 的特征
- $d_{t,i}$：本帧新预测的深度 code
- $b_{t,i}$：写入缓存的最终深度 code
- $\mathbf{1}[\cdot]$：指示函数

### 公式 11: 深度 KV 门控注入

$$
c_\ell = \frac{\sum_t A_t (1 - M_t) V^{\text{vlm}}_{\ell, t}}{\sum_t A_t (1 - M_t)}, \qquad g_\ell = \sigma(w_\ell c_\ell + b_\ell)
$$

**含义**：把"未发生变化"的位置对应的 VLM V 缓存做加权平均，再过 sigmoid 得到层级 gate $g_\ell$，弱化静止区域对动作专家的影响。

**符号说明**：
- $A_t$：注意力权重聚合
- $M_t$：变化 mask（与公式 10 一致，反向使用）
- $w_\ell, b_\ell$：可学习 gate 参数
- $\sigma$：[[Sigmoid|sigmoid]]

---

## 关键图表

### Figure 1: MolmoAct2 总览

![Figure 1](https://arxiv.org/html/2605.02881v1/x5.png)

**说明**：从数据集（MolmoAct2-BimanualYAM / SO-100/101 / DROID）到模型（Molmo2-ER + 流匹配动作专家 + MolmoAct2-Think）再到真实部署的整体训练 pipeline。这是论文的"全家福"图，说明全栈开放：数据、tokenizer、权重、训练代码全部开源。

### Figure 2: 训练数据组成

![Figure 2](https://arxiv.org/html/2605.02881v1/x6.png)

**说明**：MolmoAct2 的训练数据混合饼图，跨三大类：(1) 多模态网络数据（46% Molmo2-ER 语料 + 46% Molmo2 通用 + 8% Tulu-3 文本）；(2) 三个新发布的机器人数据集；(3) 学术机器人数据（BC-Z / Bridge / RT-1 / MolmoAct）。

### Figure 3: MolmoAct2-BimanualYAM 数据采集装置

![Figure 3](https://arxiv.org/html/2605.02881v1/x7.png)

**说明**：自研双臂遥操作平台，整机成本 < $6,000，包含 2 个 YAM 机械臂、外部相机和腕部相机。共采集 720 小时、34.5K demo、28 个真实任务（折叠、解缠、收餐桌、扫码、装箱）。低成本设计是论文 "real-world deployment" 主题的硬件基础。

### Figure 4: MolmoAct2 架构详图

![Figure 4](https://arxiv.org/html/2605.02881v1/x8.png)

**说明**：核心架构图。左侧 [[Molmo2-ER]] 处理多视角图像 + 语言指令，逐层产生 KV 缓存。右侧 36 层流匹配动作专家通过 [[Cross-Attention|交叉注意力]] 复用同层 KV，[[AdaRMS]] 由流时间 $t$ 调制。关键设计：**KV 缓存共享而非 token 拼接**，VLM 只前向一次。

### Figure 5: MolmoAct2-Think 自适应深度推理

![Figure 5](https://arxiv.org/html/2605.02881v1/figures/MAF44.png)

**说明**：MolmoAct2-Think 的 pipeline。当前帧 RGB 切成 32×32 patch，与上一帧逐 patch 算余弦相似度；变化 patch 才重新预测对应 cell 的深度 token。深度 token 作为 [[World Model|世界模型]] 隐式表示注入动作专家。这种"时间局部性 + 选择性思考"是核心创新。

### Figure 6: RoboEval 基准性能对比

![Figure 6](https://arxiv.org/html/2605.02881v1/figures/MAF6.png)

**说明**：RoboEval benchmark 上各任务成功率对比与雷达图。MolmoAct2 / MolmoAct2-Think 在多任务、长程任务上均显著领先 [[Pi0]] / [[Pi05]]。

### Figure 7: 高效微调结果

![Figure 7](https://arxiv.org/html/2605.02881v1/figures/MAF5.png)

**说明**：在每个本体上做"高效微调"（仅 50K-100K 步、16-128 batch）后真实任务的成功率柱状图，证明 MolmoAct2 在新本体上能快速适配。

### Table 1: 多模态网络数据训练语料

| Pillar | Molmo2 样本 | Molmo2-ER 样本 | 权重 |
|--------|-------------|----------------|------|
| Image QA | 2.4M | - | 0.115 |
| Video QA | 2.4M | - | 0.092 |
| Image Pointing | 1.1M | 780K | 0.046 / 0.11 |
| Video Pointing | 370K | - | 0.069 |
| Video Tracking | 800K | - | 0.069 |
| Image Embodied QA | - | 1.33M | 0.11 |
| Image Detection | - | 100K | 0.01 |
| Video Embodied QA | - | 703K | 0.1 |
| Multi-image / Ego-Exo | 700K | 700K | 0.09 |
| Captions / Long QA | 1.2M | - | 0.069 |
| Abstract Reasoning | - | 150K | 0.04 |
| NLP | 980K | - | 0.08 |
| **小计** | **8.25M** | **3.26M** | 总计 12.51M |

**说明**：多模态训练混合，凸显"具身推理"专用 pillar（Embodied QA、Pointing、Tracking）权重明显高于通用 QA。

### Table 2: OpenFAST tokenizer 训练混合

| 数据集 | 比例 | 机器人 | 动作表示 |
|--------|------|--------|----------|
| MolmoAct2-BimanualYAM | 30% | YAM | 绝对关节 |
| MolmoAct2-SO100/101 | 30% | SO-100/101 | 绝对关节 |
| MolmoAct2-DROID | 30% | Franka | 绝对关节 |
| Fractal | 3.33% | Google Robot | Delta 末端 |
| BC-Z | 3.33% | Google Robot | Delta 末端 |
| Bridge | 3.33% | WidowX | Delta 末端 |

**说明**：5 种本体混合，3 种主流 + 3 种学术。同时覆盖绝对关节与 delta 末端两种主流控制接口。

### Table 3: 具身推理结果（13 benchmark 平均）

| 模型 | 类型 | 平均 |
|------|------|------|
| GR-ER 1.5 Thinking | 闭源 | 61.3% |
| Gemini 2.5 Pro | 闭源 | 57.1% |
| GPT-5 | 闭源 | 57.9% |
| Molmo2 (baseline) | 开源 | 46.8% |
| **Molmo2-ER (ours)** | **开源** | **63.8%** |

**说明**：13 个具身/空间推理 benchmark：Point-Bench、RefSpatial、RoboSpatial-Poi、Where2Place、BLINK、CV-Bench、ERQA、EmbSpatial、MindCube、RoboSpatial-VQ、SAT、OpenEQA、VSI-Bench。Molmo2-ER 不仅超越同尺寸开源模型 17 点，还首次让开源 4B VLM 在该综合指标上击败 GPT-5 / Gemini 2.5 Pro。

### Table 4: MolmoSpace 任务结果

| Model | Pick | Pick & Place | Open | Close | Average |
|-------|------|--------------|------|-------|---------|
| π0.5-DROID | 36.4±2.9 | 13.6±2.2 | 22.7±2.6 | 65.1±3.1 | 34.5 |
| **MolmoAct2-DROID** | **43.7±3.1** | **26.7±2.8** | 9.5±2.0 | **70.8±3.0** | **37.7** |

**说明**：MolmoSpace 仿真任务上整体优于 [[Pi05]]，但 Open 任务（开抽屉/柜门）显著落后，提示门控/铰链类任务上仍有改进空间。

### Table 5: 仿真留出环境

| Model | Pick MSProc | Pick Classic | Pick | Pick Rand.-Cam. | Avg. |
|-------|-------------|--------------|------|-----------------|------|
| π0.5-DROID | 18.1±2.4 | 6.4±1.5 | 7.0±1.6 | 8.0±1.9 | 10.0 |
| **MolmoAct2-DROID** | **35.6±3.0** | **18.9±2.6** | **20.5±2.6** | **15.4±2.4** | **20.6** |

**说明**：Held-out 环境（包括随机相机视角）上 MolmoAct2 平均成功率 2 倍于 [[Pi05]]，显示更强的视觉/空间泛化能力——证明 [[Molmo2-ER]] 的具身推理能力对未知环境鲁棒。

### Table 6: 真实世界 DROID 操作任务

| Model | Apple on Plate | Pipette in Tray | Red Cube | Knife in Box | Objects in Bowl | Avg |
|-------|-----------------|-----------------|----------|--------------|-----------------|-----|
| π0.5-DROID | 66.7% | 33.3% | 53.3% | 26.7% | 46.2% | 45.2% |
| **MolmoAct2-DROID** | **100.0%** | **86.7%** | **93.3%** | **93.3%** | **62.0%** | **87.1%** |

**说明**：真实 Franka 环境，5 个任务平均 87.1% vs 45.2%，将真实部署成功率几乎翻倍。Pipette / Knife / Cube 等需要细抓握的任务提升 40+ 点。

### Table 7: SO-100 平台任务

| Model | Fork on Plate | Stack Blocks | Tissues | Pen on Notebook | Block in Box | Avg |
|-------|---------------|--------------|---------|-----------------|--------------|-----|
| π0-SO100/101 | 30.0 | 6.7 | 20.0 | 80.0 | 90.0 | 45.3 |
| **MolmoAct2-SO100/101** | **70.0** | **20.0** | **73.3** | **86.7** | 33.3 | **56.7** |

**说明**：低成本 SO-100/101 平台上整体领先 [[Pi0]]。但 "Block in Box"（90→33.3）严重退化，作者归因于 SO-100 数据中存在大量低质量遥操作 demo（即使经四阶段过滤后仍有噪声）。

### Table 8: LIBERO 仿真 benchmark

| Model | Spatial | Object | Goal | Long | Average |
|-------|---------|--------|------|------|---------|
| π0 | 96.8% | 98.8% | 95.8% | 85.2% | 94.2% |
| π0.5 | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| **MolmoAct2** | 97.8% | **100.0%** | 97.8% | 93.2% | 97.2% |
| **MolmoAct2-Think** | **98.8%** | 99.8% | **98.5%** | **95.4%** | **98.1%** |

**说明**：LIBERO 四个套件均 SOTA，Long-horizon 上 [[MolmoAct2-Think|MolmoAct2-Think]] 比 [[Pi05]] 高 3 点，验证自适应深度推理对长程任务有效。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[MolmoAct2-BimanualYAM]] | 720 小时 / 34.5K demo / 28 任务 | 双臂遥操作，自研 < $6K 平台 | 训练 + tokenizer |
| [[MolmoAct2-SO100/101]] | 184 小时 / 38K episode / 19.8M 帧 | 低成本平台，源自 1,222 个 LeRobot 公共集 | 训练 + tokenizer |
| [[MolmoAct2-DROID]] | 74,604 episode / 17.8M 帧 | Franka 子集，含语言重新标注 | 训练 + tokenizer |
| [[Molmo2-ER Corpus]] | 3.3M | 13 类具身/空间任务 | VLM 专门化 |
| BC-Z / BridgeData V2 / RT-1 / MolmoAct | 学术开放 | delta 末端 | tokenizer 多本体 |
| LIBERO | 4 套件 | 仿真 long-horizon | 评测 |

四阶段数据过滤（结构有效性 → 评估集去重 → license 检查 → [[TOPReward]] 质量门）值得 VLA 数据工程参考。

### 实现细节

- **Backbone**：[[Molmo2-ER]] 4B
- **优化器**：AdamW（默认）
- **训练阶段**：
  - 预训练 [[MolmoAct2-Pretrain]]：200K 步，仅离散动作自回归，序列 4,200，batch 128 × 64×H100，约 5,760 GPU 小时
  - 后训练 [[MolmoAct2]]：100K 步，离散+连续联合训练，序列 2,100，batch 128 × 64×H100，约 2,300 GPU 小时
  - 本体微调：50K-100K 步，仅机器人数据，8 个流匹配样本，batch 16-128
- **硬件**：64× H100（预训练/后训练），32× H100（本体微调），8× H100（评测微调）
- **控制频率**：YAM/SO-100 30Hz，DROID 15Hz，LIBERO 10Hz
- **chunk 长度**：与控制频率成正比（10 / 15 / 30 步）

### 可视化结果

定性上 MolmoAct2-Think 在长程任务（多步装箱、解缠绳子）上展现更稳定的子目标分解，且对工具使用类任务（pipette、knife）抓握精度显著提升。

---

## 批判性思考

### 优点

1. **真正的 fully-open**：权重、数据、tokenizer、训练代码全部公开，包括三个新数据集和 OpenFAST，几乎是当前 VLA 领域开放度最高的工作。
2. **架构设计有"工程美感"**：[[KV-Cache Conditioning|KV-Cache 条件化]] 让 VLM 单次前向、动作专家轻量化，比 [[Pi0]] / [[Pi05]] 的拼接式设计更适合实时控制；自适应深度推理把"思考"转化为时间增量更新，是把 [[World Model|世界模型]] 思想嵌入 VLA 的优雅方案。
3. **多本体覆盖**：从 < $6K 双臂到 Franka 到 SO-100，硬件适配广，对学术和教育场景友好。
4. **具身推理刷新开源 SOTA**：4B 模型在 13 个具身 benchmark 上击败 GPT-5 / Gemini 2.5 Pro，让中等规模开源 VLM 第一次"有面子"。

### 局限性

1. **arXiv ID 异常**：本文标注为 2605.02881，超出当前合理范围（疑似预印本编号问题或本文为虚构占位）；这不影响方法分析，但影响时间线判断。
2. **Open 任务退化**（Table 4）与 SO-100 "Block in Box"（90→33）退化（Table 7）说明：(a) 铰链类任务对模型先验仍是难点；(b) 公开数据集的质量参差，过滤策略仍可改进。
3. **MolmoAct2-Think 阈值固定（0.996）**：cosine 阈值是手工设置，没有针对动态场景自适应；快速运动 / 光照变化可能导致大面积误触发，反而比不思考更慢。
4. **流匹配采样步数未充分消融**：论文未给出推理 NFE 与成功率的 trade-off 曲线；30Hz 控制下端到端延迟仍需更细粒度的报告。
5. **未显式与 RT-2 / Octo / OpenVLA 等对比**：基线主要是 Pi0 / Pi05，对开源生态的全景对比不够。

### 潜在改进方向

1. **可学习的变化阈值**：把 0.996 cosine 阈值替换为输入相关的可学习门，结合任务难度自适应。
2. **更深的世界模型集成**：当前深度只是 10×10 离散网格，可拓展到 [[3D Gaussian Splatting|3DGS]] 或 [[Voxel World Model|体素世界模型]] 作为更精细的 [[World Model|世界模型]]。
3. **流匹配 → [[Consistency Model|一致性模型]]**：减少推理步数到 1-2 步，进一步降延迟。
4. **跨本体迁移分析**：训练好的 [[OpenFAST]] tokenizer 在新本体上的零样本迁移效果应单独评估。

### 可复现性评估

- [x] 代码开源（github.com/allenai/molmoact2）
- [x] 预训练模型（HuggingFace 多个 checkpoint）
- [x] 训练细节完整（步数、序列长、batch、GPU、超参齐全）
- [x] 数据集可获取（三个数据集全部 HuggingFace 公开）

可复现性接近满分，是 VLA 领域罕见的"全栈可复现"工作。

---

## 关联笔记

### 基于

- [[Molmo2]]：本文 VLM 主干的前身。
- [[Pi0]]：首个流匹配 VLA，本文动作专家思路的源头。
- [[Pi05]]：拼接式 KV 注入路线，本文用 KV-Cache 条件化路线对比。
- [[FAST]]：动作 tokenizer 思想前身，本文开源化为 [[OpenFAST]]。
- [[DROID]]：Franka 数据来源。

### 对比

- [[Pi0]] / [[Pi05]]：主要基线，多个 benchmark 上 MolmoAct2 显著领先。
- [[OpenVLA]] / [[Octo]]（建议补充实验对比）。

### 方法相关

- [[Flow Matching]]：动作专家训练目标。
- [[KV-Cache Conditioning]]：核心架构创新。
- [[Adaptive Depth Reasoning]]：MolmoAct2-Think 的核心机制。
- [[OpenFAST]]：开源动作 tokenizer。
- [[Action Chunking]]：动作输出形式。
- [[World Model]]：深度 token 作为隐式世界模型。
- [[AdaRMS]]：流时间调制方法。
- [[DiT]]：动作专家块结构来源。
- [[BPE]]：tokenizer 词表压缩。
- [[Cross-Attention]] / [[Self-Attention]] / [[Sigmoid]] / [[Cosine Similarity]]

### 硬件/数据相关

- [[MolmoAct2-BimanualYAM]] / [[MolmoAct2-SO100/101]] / [[MolmoAct2-DROID]]：本文新发布的三个数据集。
- [[YAM Robot]] / [[SO-100]] / [[Franka Panda]]：支持的本体硬件。
- [[LIBERO]]：仿真评测套件。

---

## 速查卡片

> [!summary] MolmoAct2: Action Reasoning Models for Real-world Deployment
> - **核心**：fully-open 多本体 VLA，KV-Cache 条件化的流匹配动作专家 + 自适应深度推理。
> - **方法**：[[Molmo2-ER]] (4B) + [[OpenFAST]] tokenizer + 36 层流匹配专家（attend VLM KV 缓存）+ MolmoAct2-Think（按需重预测 10×10 深度 token）。
> - **结果**：13 个具身推理 benchmark 平均 63.8%（超 GPT-5 / Gemini 2.5 Pro）；DROID 真实任务 87.1% vs π0.5 45.2%；LIBERO 平均 98.1% (Think)。
> - **开放度**：权重 / 数据（720h 双臂 + SO-100 + DROID）/ tokenizer / 代码全部开源。
> - **代码**：https://github.com/allenai/molmoact2

---

*笔记创建时间: 2026-05-09*
