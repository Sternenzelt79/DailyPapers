---
title: "World Action Models: The Next Frontier in Embodied AI"
method_name: "WAM"
authors: [Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, Mike Zheng Shou, Xuanjing Huang, Xipeng Qiu, Yu-Gang Jiang]
year: 2026
venue: arXiv
tags: [world-action-model, embodied-ai, survey, vla, world-model, robot-manipulation, diffusion-policy]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/abs/2605.12090
created: 2026-05-14
---

# 论文笔记：World Action Models: The Next Frontier in Embodied AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University, Shanghai Innovation Institute, National University of Singapore |
| 日期 | May 2026 |
| 项目主页 | [openmoss.github.io/Awesome-WAM](https://openmoss.github.io/Awesome-WAM) |
| 对比基线 | [[VLA]]、[[World Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.12090) / [GitHub](https://github.com/OpenMOSS/Awesome-WAM) |

---

## 一句话总结

> 首篇系统性综述 World Action Models（WAMs）——将世界状态预测与动作生成统一建模的具身基础模型，建立分类体系、分析数据生态与评测协议，指出 VLA 走向 WAM 是具身 AI 的必然前沿。

---

## 核心贡献

1. **正式定义 WAM**: 给出 WAM 的概率形式 $p(o', a \mid o, l)$，将其与 VLA、Video Policy、AWM 等概念严格区分，消除文献术语碎片化问题。
2. **结构化分类体系**: 将现有方法组织为 **Cascaded WAM**（显式/隐式规划）和 **Joint WAM**（自回归/扩散式生成）两大范式，并进一步细分生成模态、条件机制和动作解码策略。
3. **数据与评测综述**: 系统梳理四类训练数据源（机器人遥操作、便携式人类演示、仿真、互联网自中心视频）以及三维评测框架（视觉保真度、物理常识、动作可信度）。

---

## 问题背景

### 要解决的问题
标准 [[VLA]] 模型学习的是从观测到动作的反应式映射（$p(a \mid o, l)$），**不显式建模动作对物理世界的影响**，导致在需要预见未来状态的操作任务中泛化能力受限。

### 现有方法的局限
- VLA 无法利用海量无动作标注的人类视频（$o_t, o_{t+1}$ pairs）
- 世界模型研究（Action-Conditioned WM、Language-Conditioned WM）与动作生成相互割裂，缺乏统一框架
- 文献在架构、学习目标、应用场景上碎片化，缺少统一概念体系

### 本文的动机
将世界建模的预测能力（物理前瞻）与动作生成能力整合为单一框架，既能从精标注机器人数据中学习精确控制，又能从互联网规模视频中汲取丰富的物理先验——这是传统 VLA 所无法实现的。

---

## 方法详解：WAM 分类体系

### 核心定义与公式

[[WAM]] 定义为满足以下两个核心标准的具身基础模型：
1. **前向预测建模（Forward Predictive Modeling）**: 必须对未来状态 $o'$ 生成可量化的表示（显式如像素视频，或隐式如物理潜空间）
2. **耦合动作生成（Coupled Action Generation）**: 动作 $a$ 必须严格与预测的未来状态 $o'$ 对齐，而非独立预测

### WAM 的两大范式

#### Cascaded WAM（级联式）

**设计理念**: 显式分解目标分布，先合成未来状态表示，再据此解码动作。

$$
p(o', a \mid o, l) = p(a \mid o', o, l) \cdot p(o' \mid o, l)
$$

**两类实现路径**:
- **显式规划（Explicit Planning）**: 中间表示为像素空间视频帧；动作由学习式逆动力学模型（IDM）或几何提取（光流、位姿跟踪）恢复
- **隐式规划（Implicit Planning）**: 中间表示为[[LDM|潜变量]]序列，避免解码回像素空间以降低推理延迟

| 方法 | 中间表示 | Stage-1 Backbone | Stage-2 Model |
|------|---------|-----------------|---------------|
| UniPi | Pixel RGB | Video U-Net | Lightweight IDM |
| AVDC | Pixel → Optical Flow | U-Net | 几何管线（无需动作标注）|
| NovaFlow | Optical Flow | Wan 2.1 | Flow-based executor |
| Dream2Flow | Optical Flow | off-the-shelf I2V | CoTrackerV3 + RL |
| TesserAct | Pixel RGB-DN | CogVideoX (DiT) | PointNet++ |
| VPP | Latent video | Stable Video Diffusion | VideoFormer + DiT |
| LAPA | Discrete latent action tokens | C-ViViT tokenizer | LWM-Chat-1M |
| S-VAM | Latent features | Stable Video Diffusion | Uni-Perceiver |
| MWM | Latent features | DiT | Diffusion Policy |

#### Joint WAM（联合式）

**设计理念**: 在单一统一模型内同时预测未来世界状态和动作，世界建模与动作生成共同作为训练监督目标。

$$
p(o', a \mid o, l) \quad \text{直接建模联合分布}
$$

**4.2.1 自回归生成（Autoregressive Generation）**

三种表示范式：
1. **显式解耦表示（Explicit Decoupled）**: 保持模态异构格式，通过结构分离的输出头解码（GR-1、GR-2 系列）
2. **统一离散表示（Unified Discrete）**: 将动作和视觉帧全部量化到同一词表，用同一预测头生成（CoT-VLA、WorldVLA、F1）
3. **预测潜表示（Predictive Latent）**: 在连续潜嵌入空间上自回归，避免像素重建开销（[[JEPA|VLA-JEPA]]）

| 模型 | 参数量 | Backbone | 代表创新 |
|------|--------|---------|---------|
| GR-1 | 195M | GPT-style Causal Transformer | 视频重建作为动作正则化先驱 |
| GR-2 | 30-719M | GPT-style | VQGAN tokens + CVAE 动作块 |
| CoT-VLA | 7B | VILA-U | 视觉思维链（CoT）→动作 |
| WorldVLA | 7B | Chameleon-based MLLM | 模态特异因果掩码 |
| F1 | 4.2B | Mixture-of-Transformer (MoT) | Generation专家+Action专家 |
| VLA-JEPA | 2B | Qwen3-VL-2B | JEPA潜空间预测，结构无泄漏 |

**4.2.2 扩散式生成（Diffusion-based Generation）**

分为两大结构：

**Unified Stream（单流）**:
- *显式未来预测*: 将未来帧潜变量与动作 token 拼接，在同一 DiT 内联合去噪
  - PAD: 从头训练，多模态输入联合去噪
  - UWM: 对世界和动作分配独立噪声水平，单模型覆盖 policy/forward dynamics/inverse dynamics/纯视频生成四种模式
  - DreamZero (14B, Wan2.1-I2V): 联合去噪 + KV-Cache 观测替换保持闭环
  - Cosmos Policy (2B): 引入潜帧注入（proprioception、action chunk、value function），支持 best-of-N 规划
- *隐式未来预测*: 可学习的 future token 在内部被监督匹配未来观测的编码器嵌入
  - FLARE: learnable future tokens + 冻结 teacher encoder 对齐
  - FRAPPE: MoP+LoRA 多专家后训练配方

**Multi-Stream（多流）**:
- *Cross-Attention Coupled*: Video DiT 与 Action DiT 通过显式交叉注意力耦合（CoVAR Bridge Attention、LDA-1B MM-DiT、DUST 独立噪声时间步）
- *Hidden-State Coupling*: Video DiT 的中间隐状态传递给 Action DiT 作为条件信号（DiT4DiT、Fast-WAM、WAV）
- *Shared Representation*: 观测和动作先融合进共享 Transformer，再由各自解码头恢复（UVA、PhysGen）

---

## 关键公式

### 公式1: [[VLA|VLA 目标函数]]

$$
\mathcal{L}_{\text{VLA}} = \mathbb{E}_{(o,l,a) \sim \mathcal{D}} \left[ -\log p(a \mid o, l) \right]
$$

**含义**: VLA 只学习动作条件概率，不对环境动态建模。

**符号说明**:
- $o$: 当前观测（视觉输入、本体感知等）
- $l$: 语言指令
- $a$: 动作序列

---

### 公式2: [[World Model|世界模型目标函数]]

$$
\mathcal{L}_{\text{WM}} = \mathbb{E}_{(o,a,o') \sim \mathcal{D}} \left[ -\log p(o' \mid o, a) \right]
$$

**含义**: 世界模型预测在动作干预下的未来状态演化，不生成动作。

**符号说明**:
- $o'$: 下一时刻观测
- $a$: 智能体执行的动作（干预信号）

---

### 公式3: [[WAM|WAM 统一目标函数]]

$$
\mathcal{L}_{\text{WAM}} = \mathbb{E}_{(o,l,o',a) \sim \mathcal{D}} \left[ -\log p(o', a \mid o, l) \right]
$$

**含义**: WAM 对未来状态和动作的联合分布进行建模，是 VLA 和 WM 的统一推广。

**符号说明**:
- $p(o', a \mid o, l)$: 给定当前观测和语言指令，未来状态与动作的联合条件概率
- $\mathcal{D}$: 训练数据集（可包含 $(o, l, o', a)$ 四元组或仅含视频的 $(o, o')$ 对）

---

### 公式4: [[Cascaded WAM|Cascaded WAM 分解]]

$$
p(o', a \mid o, l) = p(a \mid o', o, l) \cdot p(o' \mid o, l)
$$

**含义**: Cascaded WAM 显式分解联合分布，先生成未来状态预测，再基于预测状态推导动作。

---

### 公式5: [[Action-Conditioned World Model|动作条件世界模型]]

$$
P(o' \mid o, a)
$$

**含义**: 给定当前状态和执行动作，预测下一观测状态。

---

### 公式6: [[Language-Conditioned World Model|语言条件世界模型]]

$$
P(o' \mid o, l)
$$

**含义**: 给定当前状态和语言指令，预测语义相符的未来状态，无需低级动作信号。

---

### 公式7-11: [[Visual Fidelity|视觉保真度评估指标]]

**PSNR（峰值信噪比）**:

$$
\text{PSNR}(x, y) = 10 \log \frac{\text{MAX}^2}{\text{MSE}(x, y)}
$$

**SSIM（结构相似性）**:

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

**LPIPS（学习式感知相似性）**:

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} w_l \odot \left\| \hat{f}_l(x)_{hw} - \hat{f}_l(y)_{hw} \right\|_2^2
$$

**DreamSim（人类对齐感知相似性）**:

$$
\text{DreamSim}(x, y) = 1 - \|E(x) - E(y)\|_2
$$

**DINO（语义对齐）**:

$$
\text{DINO}(g_t, r_t) = \frac{\langle f(g_t), f(r_t) \rangle}{\|f(g_t)\|_2 \|f(r_t)\|_2}
$$

**FVD（视频分布距离）**:

$$
\text{FVD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2} \right)
$$

**含义**: 分别度量像素级重建精度、结构相似性、感知相似性、人类对齐相似性、语义对齐和分布级视频真实性。

---

## 关键图表

### Figure 1: WAM 时间演化与分类体系

![Figure 1 - WAM Taxonomy](https://openmoss.github.io/Awesome-WAM/figs/roadmap.svg)

**说明**: 论文中 Figure 1 展示了 WAM 代表工作的时间演化树。左侧为 **Joint WAM** 发展脉络（自回归 vs 扩散式，扩散又分 Unified Stream 和 Multi-Stream）；右侧为 **Cascaded WAM** 的演进（显式 Explicit vs 隐式 Implicit 表示对齐）。涵盖 GR-1、GR-2、WorldVLA、DreamZero、Cosmos Policy、FLARE、UniPi、AVDC、LAPA 等代表性工作。

---

### Figure 2: WAM 综述路线图

![Figure 2 - Roadmap](https://openmoss.github.io/Awesome-WAM/figs/roadmap.svg)

**说明**: 四维综述结构——背景（VLA + World Model + WM for VLA）、架构（Cascaded + Joint）、训练数据（4类数据源）、评测（世界建模 + 动作策略评估）。

---

### Figure 3: WAM 概念定义与对比

![Figure 3 - Definition](https://openmoss.github.io/Awesome-WAM/figs/definition.svg)

**说明**: 左侧对比 VLA（$O_t \to A$）、WAM（$O_t \to A + O_{t+1}$）、WM（$O_t + A \to O_{t+1}$）三种输入输出范式；右侧界定 WAM 与 VAM（Video Action Model）、Video Policy 的概念边界。

---

### Figure 4: 世界模型在 VLA 学习与评估中的角色

![Figure 4 - WM for VLA](https://openmoss.github.io/Awesome-WAM/figs/wm4vla.svg)

**说明**: 四种应用模式：(a) 模仿学习——生成或过滤训练轨迹；(b) 强化学习——在想象环境中执行虚拟交互；(c) 奖励建模——从预测动态中推导奖励信号；(d) 策略评估——作为数据驱动模拟器进行虚拟 rollout 测试。

---

### Figure 5: Cascaded WAM 架构模式对比

![Figure 5 - Cascaded](https://openmoss.github.io/Awesome-WAM/figs/cascaded.svg)

**说明**: 三种模式——1(a) 学习式动作提取（World Model 生成像素规划图 → IDM 回归动作）；1(b) 几何提取（视频 → 光流/位姿解析 → 解析动作）；2(a) 隐式潜变量表示（跳过像素解码，直接从潜特征解码动作）。

---

### Figure 6: Joint WAM 扩散式架构模式

![Figure 6 - Joint](https://openmoss.github.io/Awesome-WAM/figs/joint.svg)

**说明**: 1(a) Unified Stream：世界和动作在单一 DiT 主干内联合去噪；2(a) Multi-Stream Cross-Attention：Video DiT 和 Action DiT 通过显式交叉注意力耦合；2(b) Hidden-State Coupling：Video DiT 中间隐状态条件化 Action DiT；2(c) Shared Representation：视觉和动作先融合再分流解码。

---

### Figure 7: 具身数据格局

![Figure 7 - Data](https://openmoss.github.io/Awesome-WAM/figs/data.svg)

**说明**: 以"迁移难度（Transfer Difficulty）"为 Y 轴、"规模化难度（Scaling Difficulty）"为 X 轴，将四类数据源定位：机器人遥操作（高质量高成本）、便携人类演示（中等质量可扩展）、仿真数据（可大规模生成但存在 sim-to-real gap）、人类自中心视频（海量规模但高度 gap）。

---

### Table 1: Cascaded WAM 方法对比

| 方法 | 中间表示 | Stage-1 | Stage-2 | 需要动作标注 | 零样本 |
|------|---------|---------|---------|------------|--------|
| UniPi | Pixel RGB | Video U-Net | IDM (CNN+MLP) | ✓ | ✗ |
| AVDC | Pixel→光流 | U-Net | 几何管线 | ✗ | ✗ |
| NovaFlow | 光流 | Wan 2.1 | Flow executor | ✗ | ✓ |
| Dream2Flow | 光流 | I2V | CoTracker+RL | ✗ | ✓ |
| TesserAct | Pixel RGB-DN | CogVideoX | PointNet++ | ✓ | ✗ |
| VPP | Latent video | SVD | VideoFormer+DiT | ✓ | ✗ |
| LAPA | Discrete latent | C-ViViT | LWM-Chat-1M | ✓ | ✗ |
| MWM | Latent features | DiT | Diffusion Policy | ✓ | ✗ |

---

### Table 2: 自回归 Joint WAM 方法对比

| 模型 | 参数 | Backbone | 输入输出 |
|------|------|---------|---------|
| GR-1 | 195M | GPT-style | obs, text, proprio → patches, actions |
| GR-2 | 30-719M | GPT-style | obs, text, proprio → VQ tokens, actions |
| CoT-VLA | 7B | VILA-U | obs, text → visual CoT tokens, discrete actions |
| WorldVLA | 7B | Chameleon MLLM | obs, text → VQ tokens, discrete actions |
| F1 | 4.2B | MoT | obs, text → VQ tokens, continuous actions |
| VLA-JEPA | 2B | Qwen3-VL-2B | obs, text → future latents, actions |

---

### Table 3: 扩散式 Joint WAM 方法对比（关键方法）

| 模型 | 参数 | Backbone | 评测 |
|------|------|---------|------|
| PAD | N/A | ImageNet-init DiT | Sim: MetaWorld; Real: Panda |
| UWM | N/A | Single DiT | Sim: LIBERO; Real: Franka |
| DreamZero | 14B | Wan2.1-I2V-14B | Sim: PolaRiS; Real: AgiBot-G1 |
| Cosmos Policy | 2B | Cosmos-Predict2-2B | Sim: LIBERO; Real: ALOHA |
| FLARE | N/A | GR00T-style DiT | Sim: RoboCasa; Real: GR1 |
| LDA-1B | 1.6B | Qwen3-VL-4B + DINOv3 | Sim: RoboCasa-GR1; Real: Galbot G1 |
| Motus | 8B | Wan2.2-2B + Qwen3-VL-2B | Sim: RoboTwin 2.0; Real: AC-One |
| DiT4DiT | N/A | Cosmos-Predict2.5-2B | Sim: LIBERO; Real: Unitree G1 |
| Fast-WAM | 6B | Wan2.2-5B | Sim: RoboTwin 2.0; Real: Galaxea R1 |

---

### Table 8: 世界建模评测指标汇总

| 名称 | 年份 | 评估重点 | 实现方式 |
|------|------|---------|---------|
| PSNR | - | 像素级重建保真度 | MSE 对数比 |
| SSIM | 2004 | 结构相似性 | 亮度/对比度/结构对比 |
| LPIPS | 2018 | 深度感知相似性 | 深度特征空间加权距离 |
| DreamSim | 2023 | 人类对齐感知相似性 | 人类三元组判断训练的嵌入空间 |
| FVD | 2018 | 分布级真实性和时序质量 | 预训练视频特征空间 Fréchet 距离 |
| VideoPhy | 2024 | 固-固/固-液/液-液物理交互 | 人工标注二分类 |
| PhyGenBench | 2024 | 物理常识对齐 | VLM+LLM 自动评估 |
| WorldModelBench | 2025 | 五项物理定律检查 | 人工标注 + 微调 VLM 判别器 |
| Physics-IQ | 2026 | 真实物理事件未来预测 | Spatial IoU / MSE |
| EWMBench | 2025 | 末端执行器轨迹一致性 | HSD、nDTW、DYN 指标 |
| WorldSimBench | 2024 | 生成视频是否保留控制相关信息 | Implicit Manipulative Evaluation |
| Wow, wo, val! | 2026 | IDM 图灵测试 | IDM 推断动作 → 真实机器人执行成功率 |

---

## 训练数据生态

### 四大数据范式

| 数据类型 | 迁移难度 | 规模化难度 | 代表数据集 |
|---------|---------|-----------|-----------|
| 机器人遥操作 | 低（精确运动学对齐）| 高（物理硬件限制）| OXE (1M+ traj)、ARIO (3M+)、AgiBot World (1M+) |
| 便携人类演示 | 中（跨形态重定向）| 中 | UMI、FastUMI-100K、DexUMI |
| 仿真数据 | 中（存在 sim-to-real gap）| 低（无限程序生成）| RoboTwin 2.0、SynGrasp-1B (10M traj)、InternData-A1 |
| 人类自中心视频 | 高（跨形态迁移难）| 极低（互联网规模）| Ego4D (3670h)、DreamDojo (43827h)、EgoScale (20854h) |

### WAM 的独特数据优势

传统 VLA 只能使用 $(o_t, a_t)$ 配对数据，纯世界模型只能使用 $(o_t, o_{t+1})$ 视频数据。**WAM 能同时消化两种类型**：用高质量 $(o_t, a_t, o_{t+1})$ 三元组紧密耦合内部表示，同时通过联合训练策略（动作位置补零 + 注意力掩码）吸收海量无动作标注视频。

---

## 评测体系

### 世界建模评测（三维）

1. **视觉保真度（Visual Fidelity）**: PSNR/SSIM 低级保真度 + LPIPS/DreamSim/DINO 感知语义相似性 + FVD 分布级真实性
2. **物理常识（Physical Commonsense）**: 对象动态（VideoPhy、PhyGenBench、WorldModelBench、Physics-IQ）+ 运动轨迹合理性（WorldScore、EWMBench）
3. **动作可信度（Action Plausibility）**: 生成视频是否保留足够控制信息（WorldSimBench、IDM Turing Test）

> 关键发现（Wow, wo, val!）：**视觉上令人信服的视频在 IDM Turing Test 下成功率接近零**，视觉合理性与可执行机器人行为之间存在根本性鸿沟。

### 动作策略评测（五类 Benchmark）

1. **通用操作**: MetaWorld、RLBench、LIBERO、ManiSkill 系列、RoboCasa、SimplerEnv
2. **双臂/人形**: RoboTwin、BiGym、HumanoidBench、HumanoidGen
3. **移动操作**: ManipulaTHOR、HomeRobot、BEHAVIOR-1K
4. **接触/变形操作**: SoftGym、PlasticineLab、TacSL、ManiFeel
5. **真实机器人**: RoboArena、RoboChallenge、Maniparena

---

## 开放挑战与机遇

### 1. 架构耦合（Architectural Coupling）
现有范式（级联/联合扩散/离散 tokenization/隐式表示对齐）在匹配条件下缺乏系统对比。核心问题：**显式像素预测是否对物理基础起到必要作用？** 若 WAM 优势主要来自训练时的辅助梯度而非推理时的未来帧生成，则潜变量预测（JEPA 风格）可能是更高效的路径。

### 2. 多模态物理状态表示（Multimodal Physical State Representation）
现有 WAM 几乎全部预测 RGB 视觉模态，但接触丰富操作最关键的物理信息（触觉分布、接触力、声学特征、材料顺应性）在像素空间不可见。**模态自适应预测（modality-adaptive prediction）** 是重要未探索前沿。

### 3. 数据利用与混合设计（Data Utilization）
最优数据混合原则不清楚：人类视频预训练的收益是语义性的还是动态性的？训练课程如何从互联网规模先验过渡到精标注机器人演示？需要**信息论视角**区分物理先验的三层价值：低级物理先验、中级因果动态、高级任务逻辑。

### 4. 长时程规划与时序抽象（Long-Horizon Planning）
WAM 主要在短时程任务上评测，长时程推理面临：预测误差累积、缺乏纠偏重规划机制、连续生成计算开销。三条路径：**模块化层次**（外部高级规划器）、**内在层次 WAM**（多分辨率未来预测）、**时序上下文扩展**。

### 5. 推理延迟与计算效率（Inference Latency）
DreamZero 通过算法加速 + CUDA 优化达到 7Hz，但现代非生成式 VLA 策略标准是 50Hz。核心问题：**下游控制究竟需要多高的预测保真度？** 指向任务自适应预测保真度（task-adaptive predictive fidelity）。

### 6. 评测方法论（Evaluation Methodology）
当前评测严重解耦：世界建模用像素指标（PSNR/FVD），动作生成用任务成功率。**缺少联合评测指标**量化想象未来与生成动作之间的因果一致性。建议引入：Counterfactual Consistency（动作如何响应想象未来的扰动）、Foresight-Conditioned Success（执行轨迹是否严格遵循生成的视觉计划）。

### 7. 安全与可靠物理部署（Safety）
WAM 的预测能力使其失败模式比反应式策略更严重——模型可能基于错误的物理想象执行延伸动作序列。**预测集成安全（prediction-integrated safety）**: 将未来想象上的不确定性估计作为安全监控的一类输入，在执行前对预测动作进行验证。

---

## 批判性思考

### 优点
1. **首篇系统性 WAM 综述**: 建立清晰分类体系，消除文献术语碎片化，为快速发展的领域提供路线图
2. **数据视角独到**: 明确指出 WAM 的独特数据优势（能消化有标注和无标注数据），以迁移难度/规模化难度两维度框架分析四类数据源
3. **评测多维覆盖**: 视觉保真度/物理常识/动作可信度三维框架，特别是 IDM Turing Test 的引入揭示了视觉合理性与可执行性之间的根本性鸿沟

### 局限性
1. **基准不一致**: 综述涵盖的方法在不同 benchmark 上评测，缺乏匹配控制条件下的系统对比，结论依赖各论文自报告结果
2. **级联 vs 联合的权衡未定论**: 论文指出两类范式但未给出在不同场景下选择原则，"architectureal fashion"问题有待解决
3. **推理延迟 gap 未解决**: 当前最优 WAM 7Hz vs VLA 标准 50Hz，差距显著，实时闭环控制可行性仍存疑

### 潜在改进方向
1. 建立联合评测框架（世界预测质量与动作质量联合衡量），替代分离式排行榜
2. 系统性消融研究：匹配数据量/计算量下比较 Cascaded vs Joint、显式 vs 隐式预测
3. 探索触觉/力感 WAM，突破视觉模态局限

### 可复现性评估
- [ ] 代码开源（综述性论文，资源列表在 GitHub）
- [x] 预训练模型（综述引用的各原始论文有不同开放程度）
- [x] 训练细节完整（综述对各方法有详细描述）
- [x] 数据集可获取（综述覆盖的大量数据集已开放）

---

## 关联笔记

### 基于
- [[VLA]]: WAM 的直接前驱，标准 VLA 学习反应式 obs→action 映射
- [[World Model]]: WAM 的另一基础，预测环境动态
- [[Dreamer]]: 经典 RSSM 世界模型，WAM 的理论基础之一

### 对比
- [[UniPi]]: 最早的 Cascaded WAM 基础工作（文本条件视频生成 + IDM）
- [[Diffusion Policy]]: 不包含世界建模，作为纯动作生成基线

### 方法相关
- [[JEPA]]: VLA-JEPA 等方法的预测表示学习框架
- [[DiT]]: Joint WAM 扩散式方法的核心主干架构
- [[Action Chunking]]: WAM 动作解码的常用策略
- [[LDM|Latent Diffusion Model]]: 隐式规划 WAM 的关键组件
- [[Inverse Dynamics Model]]: Cascaded WAM 第二阶段的核心组件
- [[Cosmos-Predict2]]: 多个 Joint WAM（Cosmos Policy、DiT4DiT、VAG、AdaWorldPolicy）的预训练 backbone
- [[V-JEPA2]]: 视频 JEPA 方法，WAM 背景技术

### 硬件/数据相关
- [[Ego4D]]: 最重要的人类自中心数据集（3670h）
- [[OXE]]: Open X-Embodiment，跨机器人遥操作聚合数据集

---

## 速查卡片

> [!summary] World Action Models Survey (WAM)
> - **核心**: 首篇 WAM 系统综述，将世界建模与动作生成统一为 $p(o', a \mid o, l)$
> - **分类**: Cascaded（显式/隐式规划）vs Joint（自回归/扩散式）两大范式
> - **数据**: 四类数据源；WAM 独特优势在于能同时消化有标注和无标注视频
> - **评测关键发现**: 视觉合理性与可执行性存在根本鸿沟（IDM Turing Test）
> - **资源**: [GitHub](https://github.com/OpenMOSS/Awesome-WAM) / [主页](https://openmoss.github.io/Awesome-WAM)

---

*笔记创建时间: 2026-05-14*
