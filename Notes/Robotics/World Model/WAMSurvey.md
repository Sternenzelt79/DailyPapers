---
title: "World Action Models: The Next Frontier in Embodied AI"
method_name: "WAMSurvey"
authors: [Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, Mike Zheng Shou, Xuanjing Huang, Xipeng Qiu, Yu-Gang Jiang]
year: 2026
venue: arXiv
tags: [world-action-model, embodied-ai, vla, world-model, survey, robot-manipulation, diffusion-policy]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: ""
created: 2026-05-14
---

# 论文笔记：World Action Models: The Next Frontier in Embodied AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University, Shanghai Innovation Institute, National University of Singapore |
| 日期 | May 2026 |
| 项目主页 | [https://openmoss.github.io/Awesome-WAM](https://openmoss.github.io/Awesome-WAM) |
| 对比基线 | [[VLA]], [[World Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.12090) / [Code](https://github.com/OpenMOSS/Awesome-WAM) |

---

## 一句话总结

> 首篇系统性综述 World Action Models（WAMs）的论文，将统一预测状态建模与动作生成的具身基础模型归纳为 Cascaded WAM 和 Joint WAM 两大范式，并梳理其数据生态与评估体系。

---

## 核心贡献

1. **形式化定义 WAM 范式**: 提出 World Action Models 的正式定义 $p(o', a | o, l)$，与 VLA、World Model、Video Policy 等相关概念明确区分
2. **首个 WAM 结构化分类法**: 将现有方法分为 Cascaded WAM（显式/隐式规划）和 Joint WAM（自回归/扩散）两大范式及其子类
3. **系统化数据与评估综述**: 分析四类训练数据源（机器人遥操作、人类演示、仿真、自我中心视频）以及视觉保真度/物理常识/动作可信度三维评估框架

---

## 问题背景

### 要解决的问题

[[VLA|Vision-Language-Action (VLA)]] 模型已实现强大的语义泛化，但它们学习的是"观测→动作"的反应式映射，没有显式建模机器人干预下物理世界如何演变。这一缺陷限制了其在需要预测未来状态的场景中的泛化能力。

### 现有方法的局限

- **VLA 模型**：学习直接的 $p(a|o,l)$ 映射，缺少前向预测能力，无法进行物理推理
- **World Model**：仅建模 $p(o'|o,a)$，缺少与动作生成的耦合
- **文献碎片化**：跨越 Cascaded/Joint 架构、不同学习目标与应用场景，缺乏统一概念框架

### 本文的动机

将预测状态建模与动作生成统一于同一框架（WAM），使模型能从物理预见性出发生成动作，从而获得更深的物理理解和更强的零样本泛化能力，同时可充分利用海量无动作标注的视频数据。

---

## 方法详解

### WAM 形式化定义

[[World Action Model|World Action Models (WAMs)]] 是一类统一了环境动力学建模（世界建模）与运动控制（动作生成）的**具身基础模型**。其必须满足两个核心条件：

- **前向预测建模**：模型必须通过生成或利用未来状态 $o'$ 的可量化表示，来预测物理环境的未来演变。建模可以是显式的（如像素级视频帧、稠密光流）或隐式的（如物理约束潜在空间）。
- **耦合动作生成**：模型必须严格与预测的未来状态 $o'$ 对齐，推导运动指令 $a$。这种耦合可以是联合概率输出，也可以是 Cascaded 或统一潜在架构中的策略条件化。

### 关键分类：Cascaded WAM vs. Joint WAM

**Cascaded WAM**：显式分解目标

$$
p(o', a | o, l) = p(a | o', o, l) \cdot p(o' | o, l)
$$

先预测未来状态表示，再从中推导动作。分为：
- **显式规划（像素空间）**：用视频生成模型预测 RGB 帧，再通过学习式 IDM 或几何提取获得动作
- **隐式规划（潜在空间）**：跳过像素解码，用潜在特征直接推导动作，效率更高

**Joint WAM**：直接建模联合分布

$$
p(o', a | o, l)
$$

在共享表示空间中联合优化状态预测与动作生成。分为：
- **自回归生成**：将未来状态和动作序列化为 token 流，因果预测
- **基于扩散的生成**：通过扩散/流匹配并发生成未来状态和动作序列

---

## 关键公式

### 公式1: [[VLA|VLA 训练目标]]

$$
\mathcal{L}_{\text{VLA}} = \mathbb{E}_{(o,l,a) \sim \mathcal{D}} \left[ -\log p(a | o, l) \right]
$$

**含义**: VLA 学习在给定观测和语言指令下动作的条件概率，不涉及未来状态预测

**符号说明**:
- $o$: 当前观测（视觉输入、本体感受信号等）
- $l$: 语言指令
- $a$: 动作序列
- $\mathcal{D}$: 训练数据集

---

### 公式2: [[World Model|World Model 训练目标]]

$$
\mathcal{L}_{\text{WM}} = \mathbb{E}_{(o,a,o') \sim \mathcal{D}} \left[ -\log p(o' | o, a) \right]
$$

**含义**: World Model 学习在给定当前观测和动作下，下一观测的条件概率，建模环境动力学

**符号说明**:
- $o'$: 下一时刻观测
- $a$: 动作（低层控制信号）

---

### 公式3: [[World Action Model|WAM 训练目标]]

$$
\mathcal{L}_{\text{WAM}} = \mathbb{E}_{(o,l,o',a) \sim \mathcal{D}} \left[ -\log p(o', a | o, l) \right]
$$

**含义**: WAM 同时建模未来状态和动作的联合分布，超越"观测→动作"映射，获取更强的物理理解

**符号说明**:
- $p(o', a | o, l)$: 给定当前观测 $o$ 和语言 $l$，未来状态 $o'$ 与动作 $a$ 的联合分布

---

### 公式4: [[Action-Conditioned World Model|Action-Conditioned World Model 条件分布]]

$$
P(o' | o, a)
$$

**含义**: 动作条件化世界模型给定当前状态和动作预测下一状态，建模动作对环境的因果效应

---

### 公式5: [[Language-Conditioned World Model|Language-Conditioned World Model]]

$$
P(o' | o, l)
$$

**含义**: 语言条件化世界模型使用自然语言作为高层语义条件，生成符合语言描述的未来视觉状态

---

### 公式6: [[PSNR|峰值信噪比 (PSNR)]]

$$
\text{PSNR}(x, y) = 10 \log \frac{\text{MAX}^2}{\text{MSE}(x, y)}
$$

**含义**: 衡量像素级重建保真度，通过最大信号值与均方误差之比的对数比衡量

**符号说明**:
- $\text{MAX}$: 像素最大值（如255）
- $\text{MSE}(x,y)$: 参考帧与生成帧之间的均方误差

---

### 公式7: [[SSIM|结构相似度 (SSIM)]]

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

**含义**: 从亮度、对比度和结构三个维度评估图像相似性，比 PSNR 更接近人类感知

**符号说明**:
- $\mu_x, \mu_y$: 图像 $x$, $y$ 的均值
- $\sigma_x^2, \sigma_y^2$: 方差
- $\sigma_{xy}$: 协方差
- $C_1, C_2$: 稳定常数

---

### 公式8: [[LPIPS|感知相似度 (LPIPS)]]

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \hat{f}^l(x)_{hw} - \hat{f}^l(y)_{hw} \right\|_2^2
$$

**含义**: 在深度特征空间中度量感知相似性，比像素级指标更好地捕捉人类感知

**符号说明**:
- $\hat{f}^l(\cdot)_{hw}$: 第 $l$ 层位置 $(h,w)$ 处的通道归一化深度特征
- $w_l$: 学习到的逐通道权重向量

---

### 公式9: [[DreamSim|人类对齐感知相似度 (DreamSim)]]

$$
\text{DreamSim}(x, y) = 1 - \| E(x) - E(y) \|_2
$$

**含义**: 基于人类判断三元组训练的融合嵌入相似度，评估对象布局、身份和场景语义的保留程度

**符号说明**:
- $E(\cdot)$: 融合嵌入编码器

---

### 公式10: [[DINO|DINO 语义特征相似度]]

$$
\text{DINO}(g_t, r_t) = \frac{\langle f(g_t), f(r_t) \rangle}{\| f(g_t) \|_2 \| f(r_t) \|_2}
$$

**含义**: 在 DINOv2 特征空间中计算余弦相似度，衡量语义/实例级对齐

**符号说明**:
- $f(\cdot)$: DINOv2 编码器
- $g_t, r_t$: 生成帧与参考帧

---

### 公式11: [[FVD|视频生成质量 (FVD)]]

$$
\text{FVD} = \| \mu_r - \mu_g \|_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2} \right)
$$

**含义**: 在预训练视频特征空间中计算真实视频与生成视频分布之间的 Fréchet 距离，反映整体逼真度和时序动态

**符号说明**:
- $\mu_r, \mu_g$: 真实/生成视频分布的均值
- $\Sigma_r, \Sigma_g$: 对应协方差矩阵

---

## 关键图表

### Figure 1: WAM 时序演化与分类法（Roadmap）

![[WAMSurvey_fig1_roadmap.svg]]

**说明**: WAM 代表性工作的时序演化与分类。左侧分支展示 Joint WAM 的发展（自回归/扩散，单流/多流），右侧分支展示 Cascaded WAM（显式/隐式表示对齐）。时间跨越 2024-2026 年，涵盖 GR-1、UniPi、PAD、DreamZero、Cosmos Policy、UWM、FLARE 等代表性工作。

---

### Figure 2: WAM 概念定义与比较

![[WAMSurvey_fig2_definition.svg]]

**说明**: 左图对比 VLA、WAM 和 WM 的输入-输出形式化——WAM 需同时预测动作 $A$ 和下一观测 $O_{\text{next}}$；右图说明 WAM 相对于 Video Action Model (VAM) 和 Video Policy 的概念范围，WAM 是模态无关的更广义超集。

---

### Figure 3: World Model for VLA 的四种应用模式

![[WAMSurvey_fig3_wm4vla.svg]]

**说明**: World Model 为 VLA 学习和评估提供支持的四种方式：(a) **模仿学习**——生成或过滤训练轨迹 $\mathcal{T}$；(b) **强化学习**——作为虚拟环境进行想象交互；(c) **奖励建模**——从预测动态中产生奖励信号；(d) **策略评估**——作为数据驱动的模拟器进行虚拟回滚测试。

---

### Figure 4: Cascaded WAM 结构比较

![[WAMSurvey_fig4_cascaded.svg]]

**说明**: Cascaded WAM 的三种模式对比：
- **1(a) Learned Action**：世界模型生成显式像素空间未来规划，由学习式逆动力学模型（IDM）映射为动作
- **1(b) Geometric Extraction**：通过几何提取（光流/姿态追踪）将显式视觉规划转化为动作
- **2(a) Latent Representation**：规划载体为潜在未来表示而非 RGB 帧，动作模型直接从中解码

---

### Figure 5: Joint WAM（扩散式）主要架构模式

![[WAMSurvey_fig5_joint.svg]]

**说明**: 扩散式 Joint WAM 的四种耦合架构：
- **1(a) Unified Stream**：世界与动作变量整合于单一 DiT 骨干（显式或隐式未来预测）
- **2(a) Cross-Attention Coupled**：独立的 Video DiT 和 Action DiT 通过显式跨注意力耦合
- **2(b) Hidden-State Coupling**：视频 DiT 的中间隐状态用于条件化 Action DiT
- **2(c) Shared Representation**：视频和动作先通过统一编码器融合，再分别解码

---

### Figure 6 / Figure 7: WAM 训练数据全景图

![[WAMSurvey_fig6_data.svg]]

**说明**: WAM 训练的具身数据全景，以"迁移难度（Y 轴）"和"规模扩展难度（X 轴）"为坐标展开：
- **机器人遥操作数据**：高质量、复杂设置，迁移容易但规模有限
- **便携式人类演示（UMI 式）**：良好保真度、低设置成本
- **仿真数据**：容易规模化，但存在领域差距
- **人类/自我中心视频**：大规模互联网数据，迁移难度最高

---

### Table 1: Cascaded WAM 方法对比

| 方法 | 中间表示 | Stage-1 骨干 | Stage-2 模型 | 动作标注 | 零样本 | 评估平台 |
|------|----------|-------------|------------|----------|--------|----------|
| UniPi | Pixel RGB | Video U-Net | Lightweight IDM (CNN+MLP) | ✓ | × | PDSketch, CLIPort, WidowX |
| VLP | Pixel RGB | Video U-Net + PaLM-E 12B | LAVA | ✓ | × | Language Table |
| RoboEnvision | Pixel RGB | OpenSora (DiT) | OpenSora DiT | ✓ | × | LanguageTable, LHMM |
| Say, Dream & Act | Pixel RGB | COSMOS-PREDICT2 | Transformer (ACT) + Qwen2.5 | ✓ | × | LIBERO, Franka |
| TesserAct | Pixel RGB-DN | CogVideoX (DiT) | PointNet++, MLP | ✓ | × | RLBench |
| MVISTA-4D | Pixel RGB-D | WAN2.2 TI2V | PointNet | ✓ | × | RLBench, RoboTwin2 |
| Vidar | Pixel RGB | Wan2.2/Vidu 2.0 | Masked IDM (MIDM) | ✓ | × | RoboTwin 2.0, Aloha |
| Gen2Act | Pixel RGB | VideoPoet | Custom closed-loop policy | ✓ | × | Mobile manipulators |
| Veo-Act | Pixel RGB | Veo-3 | Multi-head IDM + VLA (π0.5) | ✓ | × | IsaacLab |
| VAG | Pixel RGB | Cosmos-Predict2 | 1D U-Net | ✓ | × | LIBERO, AgiBot |
| π0.7 | Pixel RGB | BAGEL | VLA | ✓ | × | — |
| AVDC | Pixel RGB → Optical flow | U-Net | Off-the-shelf geometric pipeline | × | × | Meta-World, Franka |
| Im2Flow2Act | Optical flow | AnimateDiff + SD (latent) | Flow-conditioned IL policy | ✓ | × | CLIPort, WidowX |
| 3DFlowAction | 3-D optical flow | AnimateDiff + SD | Flow-constrained opt. policy | × | × | Custom, UR5e |
| NovaFlow | Optical flow | Wan 2.1 + off-the-shelf | Flow-based executor | × | ✓ | RoboCasa, Libero10 |
| Dream2Flow | Optical flow | Off-the-shelf I2V model | Flow-based executor | × | ✓ | — |
| Dreamitate | Pixel RGB | U-Net | CoTrackerV3 + traj. opt./RL | × | × | Franka, Spot |
| 4DGen | Pixel RGB-D + 4D point maps | U-Net | FoundationPose + SAM2 | × | × | OmniGibson, Franka |
| RIGVid | Pixel RGB | Kling v1.6 | FoundationPose + SAM2 | × | ✓ | Franka, Unitree G1 |
| LVP | Pixel RGB | WAN 2.1 I2V 14B | HaMeR + DexRetargeting | × | ✓ | Franka, Unitree G1 |
| Video Policy | Latent features | U-Net | Action U-Net | ✓ | × | SIMPLER-Bridge, Franka |
| ARDuP | Latent video | U-Net | Lightweight CNN | ✓ | × | CALVIN, MetaWorld |
| mimic-video | Latent video | Cosmos-Predict2 | Action Decoder DiT | ✓ | × | SIMPLER, LIBERO |
| VPP | Latent video | Stable Video Diffusion | VideoFormer + DiT | ✓ | × | Language Table, SIMPLER |
| VILP | Latent video | U-Net + 3D conv | Two CNN + MLP | ✓ | × | SIMPLER, LIBERO |
| LAPA | Discrete latent tokens | C-ViViT tokenizer | LWM-Chat-1M | ✓ | × | CALVIN, MetaWorld |
| villa-X | Latent features | Stable Video Diffusion | PaliGemma VLM | ✓ | × | Custom, Franka |
| S-VAM | Latent features | Uni-Perceiver | Diffusion Policy + MLP | ✓ | × | LIBERO, RLBench |
| OmniVTA | Latent tactile features | Two-stream DiT | Diffusion Policy | ✓ | × | CALVIN, MetaWorld |
| MWM | Latent features | DiT | Diffusion Policy | ✓ | × | LIBERO, RLBench |

---

### Table 2: Autoregressive Joint WAM 方法摘要

| 模型 | 参数量 | 骨干 | I/O 形式 | 评估平台 |
|------|--------|------|----------|----------|
| GR-1 | 195M | GPT-style Causal Transformer | Obs, text, proprio → future patches, continuous actions | CALVIN, Kinova Gen2 |
| GR-MG | N/A | GPT-style + InstructPix2Pix | Obs, text, history → goal image, task progress, actions | CALVIN, Kinova Gen3 |
| GR-2 | 30-719M | GPT-style Causal Transformer | Obs, text, proprio → future VQ tokens, continuous actions | CALVIN, Kinova Gen3 |
| CoT-VLA | 7B | VILA-U | Obs, text → future visual tokens, discrete actions | LIBERO, Bridge-V2 |
| WorldVLA | 7B | Chameleon-based MLLM | Obs, text → future VQ tokens, discrete actions | LIBERO |
| RynnVLA-002 | 5B | Chameleon + Action Head | Obs, text, state → future VQ tokens, continuous actions | LIBERO, SO100 |
| F1 | 4.2B | Mixture-of-Transformer (MoT) | Obs, text → future VQ tokens, continuous actions | LIBERO, SimplerEnv, Genie-1 |
| VLA-JEPA | 2B | Qwen3-VL-2B | Obs, text → future latents, continuous actions | LIBERO, SimplerEnv, LIBERO-Plus |

---

### Table 3: Diffusion-based Joint WAM 方法摘要

| 模型 | 参数量 | 骨干 | I/O 形式 | 评估平台 |
|------|--------|------|----------|----------|
| **Unified Stream: Explicit** | | | | |
| PAD | N/A | ImageNet-initialized DiT | Obs, pose, depth, text → future image+depth, actions | MetaWorld, Panda |
| VideoVLA | 5B | CogVideoX-5B | Obs, text → future video latents, actions | SIMPLER, Realman |
| UWM | N/A | Single DiT | Obs → future image latents, actions | LIBERO, Franka |
| DreamZero | 14B | Wan2.1-I2V-14B | Obs, text, proprio → future video, actions | PolaRiS, GenieSim3.0, AgiBot |
| Cosmos Policy | 2B | Cosmos-Predict2-2B | Obs, text, proprio → future video+values | LIBERO, ALOHA |
| GigaWorld-Policy | 5B | Wan 2.2-5B | Obs, text, proprio → future video, actions | RoboTwin 2.0, AgileX |
| X-WAM | 5B | Wan2.2-TI2V-5B | Obs, text, proprio → future RGB-D+states+actions | RoboCasa, RoboTwin 2.0, AC One |
| **Unified Stream: Implicit** | | | | |
| FLARE | N/A | GR00T-style policy DiT | Obs, text, proprio → future latents, actions | RoboCasa, GR1 humanoid |
| FRAPPE | 1B | RDT-1B | Obs, text, proprio → future representations, actions | RoboTwin 2.0, AgileX |
| **Multi-Stream: Cross-Attention** | | | | |
| CoVAR | N/A | Open-Sora 1.2 | Obs, text, proprio → future video, actions | CALVIN, Libero90, UR5 |
| LDA-1B | 1.6B | Qwen3-VL-4B + DINOv3-ViT-s | Obs, text → future observation, actions | RoboCasa-GR1, Galbot/Unitree |
| DUST | N/A | Eagle-2 | Obs, text, proprio → future obs, actions | RoboCasa, GR-1, Franka |
| LingBot-VA | 5.3B | Wan2.2-5B | Obs, text, action history → future video+actions | Robotwin 2.0, LIBERO, Franka |
| DexWorldModel | N/A | DINOv3 + Wan2.2-5B | Obs, text, action history → future latents+actions | RoboTwin, AgiLex |
| AIM | N/A | Wan2.2-TI2V-5B | Obs, text, action history → future RGB+spatial values+actions | RoboTwin 2.0 |
| Motus | 8B | Wan2.2-2B + Qwen3-VL-2B | Obs, text → future video+actions | Robotwin 2.0, LIBERO-Long |
| MotuBrain | N/A | Vidu | Obs, text → future video+actions | RoboTwin 2.0, WorldArena |
| AdaWorldPolicy | 2.8B | Cosmos-Predict2 | Obs, text, proprio → future video+force+actions | LIBERO-10, Variant PushT |
| UD-VLA | 8B | Emu3 | Obs, text → future visual tokens, discrete actions | CALVIN, LIBERO, SimplerEnv |
| **Multi-Stream: Hidden-State** | | | | |
| DiT4DiT | N/A | Cosmos-Predict2.5-2B | Obs, text, proprio → future video latents, actions | LIBERO, RoboCasa-GR1, G1 |
| Fast-WAM | 6B | Wan2.2-5B | Obs, text → latent world features, actions | RoboTwin 2.0, LIBERO, Galaxea |
| WAV | 2.2B | Genie Envisioner | Obs, text → future visual traj+values+actions | LIBERO, Piper |
| Act2Goal | 1.76B | Genie Envisioner | Obs, goal image, proprio → future video latents, actions | Robotwin 2.0, AgiBot |
| **Multi-Stream: Shared Repr.** | | | | |
| UVA | 0.5B | Shared Transformer | Obs, text, action history → future image latents, actions | PushT, ToolHang, LIBERO-10 |
| PhysGen | 0.73B | NOVA | Obs, text, action history → future video+actions | LIBERO, ManiSkill, Franka |

---

### Table 4: 机器人遥操作数据集摘要（部分重要条目）

| 名称 | 年份 | 规模 | 机器人 | 收集方式 | 观测模态 |
|------|------|------|--------|----------|----------|
| QT-Opt | 2018 | 580k traj. | KUKA iiwa | script | RGB |
| RoboNet | 2019 | ~162k traj. | 7 robots | script | RGB, P |
| OXE | 2023 | 1M+ traj. | 22 robots | teleop/script | RGB, D, text, PC |
| DROID | 2024 | 76k traj. | Franka | teleop | RGB, D, text |
| ARIO | 2024 | 3M+ traj. | 35 robots | aggregation/teleop | RGB, D, A, T, text |
| AgiBot World | 2025 | — | AgiBot | teleop | RGB, text |

---

### Table 8: 世界建模评估指标与基准摘要

| 名称 | 年份 | 评估重点 | 实现方式 |
|------|------|----------|----------|
| PSNR | — | 像素级重建保真度 | log(MAX²/MSE) |
| SSIM | 2004 | 亮度/对比度/结构相似性 | 结构相似度指数 |
| LPIPS | 2018 | 深度感知相似度 | 深度特征加权距离 |
| DreamSim | 2023 | 人类对齐感知相似度 | 融合嵌入距离 |
| DINO | 2023 | 语义/实例级对齐 | DINOv2 余弦相似度 |
| FVD | 2018 | 分布级逼真度和时序质量 | 视频特征空间 Fréchet 距离 |
| VideoPhy | 2024 | 物理交互场景（固-固、固-液、液-液）| 人类二值标注 |
| PhyGenBench | 2024 | 生成视频的物理常识对齐 | PhyGenEval（VLM/LLM 自动评估）|
| VBench-2.0 | 2025 | 物理原理遵守 | 视频多问答 + 异常实体检测 |
| WorldModelBench | 2025 | 物理定律遵守 | 五条二值物理定律检查 + VLM 评判者 |
| Physics-IQ | 2026 | 真实物理事件的未来演化预测 | Spatial IoU/Spatiotemporal IoU/MSE |
| WorldScore | 2025 | 可控性、质量、动态 | 光流运动幅度 + 平滑度 |
| EWMBench | 2025 | 具身世界模型运动正确性 | EEF 轨迹指标（HSD, nDTW, DYN）|
| WorldSimBench | 2024 | 动作可信度（隐式操控评估）| IDM 提取动作后的任务成功率 |
| Wow, wo, val! | 2026 | 生成视频能否被转化为可执行动作 | IDM 图灵测试 + 真实世界执行成功率 |

---

### Table 9: 动作策略评估基准摘要（部分重要条目）

| 名称 | 年份 | 任务数 | 仿真平台 | 评估重点 |
|------|------|--------|----------|----------|
| MetaWorld | 2019 | 50 | MuJoCo | Multi-task |
| RLBench | 2020 | 100 | CoppeliaSim | Generalization |
| LIBERO | 2023 | 130 | MuJoCo | LH, Lang, LL |
| ManiSkill3 | 2025 | 62 | SAPIEN | Data Scaling |
| RoboCasa | 2024 | 100 | MuJoCo | DS, MT |
| RoboVerse | 2025 | 276 | MetaSim | S2R, DS, MT |
| COLOSSEUM | 2024 | 60 | CoppeliaSim | Generalization |
| SimplerEnv | 2024 | — | SAPIEN | Sim-to-Real |
| AGNOSTOS | 2025 | — | CoppeliaSim | Zero-shot G |
| GemBench | 2025 | 41 | CoppeliaSim | Generalization |
| RoboTwin | 2025 | 40 | SAPIEN/ManiSkill3 | LH, MT, Dex |
| HumanoidBench | 2024 | 27 | MuJoCo | Dex |
| RoboArena | 2025 | 30 | ManiSkill3 | Real-world G |

---

## 架构深度解析

### Cascaded WAM：显式规划（像素空间）

**代表工作路径**:
- **UniPi** (2023): 文生视频 DiT → CNN+MLP IDM，奠定两阶段蓝图
- **VLP**: 引入 VLM 语义干预 + 树搜索值评分
- **Gen2Act**: 零样本调用 VideoPoet，用点追踪作为辅助损失
- **Veo-Act**: Veo-3 生成粗导航阶段，接触检测后切换 π0.5 精细控制

**光流几何提取路线**:
- **AVDC**: 光流作为中间表示，SE(3) 变换解析导出动作，无需动作标注
- **Im2Flow2Act**: 在潜在空间估计光流，跳过像素合成
- **3DFlowAction**: 提升为三维光流，捕获旋转和深度位移分量
- **NovaFlow/Dream2Flow**: 零训练操作，预训练视频生成模型直接用

### Cascaded WAM：隐式规划（潜在空间）

- **VPP**: 预训练 VAE 编码观测帧，单步预测潜在序列，首次达到实时推理速度
- **S-VAM**: 自蒸馏压缩多步生成为单步前向传播，QFormer 聚合后条件化 DiT
- **MWM**: 用未来语义掩码潜在替代 RGB 预测，几何信息瓶颈过滤光度干扰

### Joint WAM：自回归生成

**显式解耦表示路线** (GR 系列):
- **GR-1**: 视频重建预训练 → 双分支头并行解码未来视觉 patch 和连续动作
- **GR-MG**: 引入 [PROG] token，将世界回滚解耦为宏/微步骤层次
- **GR-2**: 完全离散视觉管线 + CVAE 动作分块

**统一离散表示路线**:
- **CoT-VLA**: 因果注意力生成视觉 CoT，再用全注意力并行预测动作 token
- **WorldVLA**: 模态专用因果掩码，禁止当前动作 token 注意先前动作
- **F1**: MoT 框架，Generation Expert（VQ token）→ Action Expert（反向动力学）

**预测潜在表示路线**:
- **VLA-JEPA**: 基于 [[JEPA|Joint-Embedding Predictive Architecture]]，连续潜在动作 token 条件化自回归世界模型，流匹配执行头

### Joint WAM：扩散式生成（单流）

**显式未来预测**:
- **PAD**: 多模态输入编码为统一潜在序列，联合去噪未来帧和动作块
- **UWM**: 对世界和动作变量分配独立噪声时间步，单模型支持策略推理/前向动力学/逆动力学/纯视频生成四种模式
- **DreamZero**: 基于 Wan2.1-14B，KV 缓存观测替换避免生成漂移，异步执行+DiT 缓存+量化实现实时推理（7Hz）
- **Cosmos Policy**: 本体感受/动作/未来状态/值函数编码为潜在帧并交错插入，单一 checkpoint 同时作为策略/世界模型/值函数

**隐式未来预测**:
- **FLARE**: 可学习的未来 token 附加到动作 token 序列，在中间层被监督匹配真实未来观测的冻结教师编码器嵌入
- **FRAPPE**: 混合前缀与 LoRA 多专家，每个专家对应不同教师表示，用于 RDT 骨干的后训练适配

### Joint WAM：扩散式生成（多流）

**跨注意力耦合**:
- **CoVAR**: Bridge Attention 模块拼接视频/动作特征 → 联合注意力 → 分离
- **LDA-1B**: 共享 MM-DiT 注意力层，DINO 潜在空间预测未来状态，任务嵌入+寄存器 token 切换策略/动力学/预测模式
- **Motus**: 世界/动作/语义理解三专家 Tri-modal Joint Attention，UniDiffuser 式独立噪声调度

**隐状态耦合**:
- **DiT4DiT**: Video DiT 的中间隐激活由 hook 截取，通过跨注意力条件化 Action DiT，三时间步设计
- **Fast-WAM**: 训练时保留视频分支，推理时完全移除，Video DiT 仅用单次前向传播编码当前视觉上下文
- **WAV**: 视频模块生成潜在未来回滚 → 值模块评估 → 动作解码器条件化视觉+值嵌入

**共享表示**:
- **UVA**: 历史观测+动作块+掩码未来 token → 统一 Transformer，视频扩散头和动作扩散头分离解码
- **PhysGen**: 帧 token 和动作 token 拼接为共享物理 token，因果 Transformer 建模，Action-DiT 通过跨注意力解码

---

## 训练数据生态

### 四类数据源的权衡

| 数据类型 | 代表数据集 | 优势 | 劣势 |
|---------|-----------|------|------|
| 机器人遥操作 | OXE, DROID, ARIO, AgiBot World | 精确运动学对齐，零模拟-真实差距 | 收集成本高，规模受限 |
| 便携式人类演示（UMI） | UMI, DexUMI, FastUMI-100K | 保真度好，设置成本低 | 需要形态迁移 |
| 仿真数据 | MimicGen, RoboTwin 2.0, RoboCasa | 无限程序化变体，可扩展 | 领域差距，物理近似 |
| 人类/自我中心视频 | Ego4D, EgoVid-5M, EgoDex, HumanNet | 互联网级规模，丰富物理先验 | 最高迁移难度，无动作标注 |

WAM 的独特优势在于**统一数据消化**：可同时处理高质量 $(o_t, a_t, o_{t+1})$ 三元组与大规模无配对数据 $(o_t, o_{t+1})$（通过联合训练策略中的掩码/补零处理）。

### 数据规模趋势

- OXE（2023）：100 万+轨迹，22 个机器人，统一化
- ARIO（2024）：300 万+轨迹，35 个机器人，多模态（RGB/D/A/T）
- HumanNet（2026）：100 万小时人类视频
- EgoDex（2026）：829 小时，194 任务，高保真 3D 手指追踪

---

## 评估框架

### 世界建模能力评估（三维）

1. **视觉保真度（Visual Fidelity）**
   - 像素级：PSNR、SSIM
   - 感知级：LPIPS、DreamSim
   - 语义级：DINO 特征相似度
   - 分布级：FVD

2. **物理常识（Physical Commonsense）**
   - 对象动力学：VideoPhy、PhyGenBench、VBench-2.0、WorldModelBench
   - 运动轨迹：WorldScore、EWMBench（HSD/nDTW/DYN 指标）
   - 物理原理：Physics-IQ（Spatial IoU、Spatiotemporal IoU）

3. **动作可信度（Action Plausibility）**
   - WorldSimBench：隐式操控评估（生成视频是否保留可操控信息）
   - Wow, wo, val!：IDM 图灵测试（生成视频经 IDM 推断动作后的真实执行成功率）

### 动作策略能力评估

覆盖 40+ 主流基准，分五类：
1. 通用操作（MetaWorld, RLBench, LIBERO, ManiSkill 系列, RoboCasa）
2. 双臂与类人形（RoboTwin, BiGym, HumanoidBench, HumanoidGen）
3. 移动操作（ManipulaTHOR, HomeRobot, BEHAVIOR-1K）
4. 接触与可变形物体操作（SoftGym, PlasticineLab, TacSL, ManiFeel）
5. 真实设备（RoboArena, RoboChallenge, Maniparena）

---

## 开放挑战与未来方向

### 1. 架构耦合（Architectural Coupling）
现有各类耦合策略（Cascaded、Joint 扩散、离散 Tokenization、隐式对齐）缺乏在相同规模和数据条件下的系统对比研究。未来需回答：显式像素空间预测是否必要？隐式预测是否可通过训练时辅助梯度达到同等效果？

### 2. 多模态物理状态表示（Multimodal Physical State）
现有 WAM 几乎普遍预测 RGB 视觉未来状态，但接触丰富操作中最关键的物理信息（触觉分布、接触力、声学特征）在像素空间不可见。需要扩展到联合预测触觉/力/本体感知的多模态世界建模。

### 3. 数据利用与混合设计（Data Mixture Design）
最优数据混合原则仍不清楚——人类视频预训练的收益主要来自语义还是动力学？训练课程如何从互联网规模先验过渡到精确动作标注演示？需要从信息论角度理解数据混合。

### 4. 长时规划与时序抽象（Long-Horizon Planning）
现有 WAM 主要在短时操作任务上评估。三条互补路径：
- **模块化层次**：VLM 分解高层任务→WAM 执行低层物理
- **内在层次 WAM**：多分辨率未来预测（粗粒度战略规划+细粒度物理细节）
- **扩展时序上下文**：解决标准注意力的二次开销

### 5. 推理延迟与计算效率（Inference Latency）
DreamZero 通过算法优化和 CUDA 低层优化实现 7Hz，但仍远低于非生成式 VLA 策略的 50Hz 标准。核心问题：给定任务，最小充分世界模型是什么？

### 6. 评估方法论（Evaluation Methodology）
当前评估严重解耦：世界建模用像素指标（忽略物理正确性），动作生成用下游成功率。缺失关键的**联合评估指标**：
- *Counterfactual Consistency*：动作如何响应想象未来的扰动
- *Foresight-Conditioned Success*：执行轨迹是否严格遵循生成的视觉规划

### 7. 安全与可靠物理部署（Safety）
WAM 可能提交于扩展动作序列，其错误更难中断。但预测能力也提供了安全强制的原则性机会：在执行前对预测动作进行物理约束验证。需要将**预测集成安全**（prediction-integrated safety）纳入推理管线。

---

## 批判性思考

### 优点
1. **首篇系统性综述**：填补了 WAM 领域缺乏统一概念框架的空白，Cascaded/Joint 二分法清晰实用
2. **全面的数据生态分析**：四类数据源的权衡矩阵（迁移难度 vs. 规模难度）为实践者提供了直观指导
3. **多维评估框架**：将视觉保真度/物理常识/动作可信度三维评估综合，指出了现有评估解耦问题

### 局限性
1. **缺乏定量对比**：作为综述，未提供各方法在统一 benchmark 下的性能数字，各表格主要是定性描述
2. **部分方法细节依赖引用**：对某些 2026 年最新方法的描述较为简略，读者需查阅原文
3. **"WAM"命名是否会被广泛接受**：与 AWM（Action World Model）等既有命名竞争，术语标准化仍需社区共识

### 潜在改进方向
1. 建立统一的 WAM 基准测试套件，在相同条件下比较 Cascaded vs. Joint 范式
2. 开发联合评估指标（Counterfactual Consistency、Foresight-Conditioned Success）
3. 探索多模态物理状态表示（视觉+触觉+力）的 WAM 架构

### 可复现性评估
- [x] 代码开源（GitHub Repo: https://github.com/OpenMOSS/Awesome-WAM）
- [ ] 预训练模型（综述论文，无模型）
- [x] 训练细节完整（引用各方法原文）
- [x] 数据集可获取（引用公开数据集）

---

## 关联笔记

### 基于
- [[VLA|Vision-Language-Action Model]]: WAM 的直接前身范式
- [[World Model]]: WAM 的另一核心组件
- [[Diffusion Model|扩散模型]]: Joint WAM 的主流生成骨干
- [[DiT|Diffusion Transformer]]: 大多数现代 Joint WAM 的骨干架构

### 对比
- [[VLA]]: WAM = VLA + 前向预测建模
- [[Video Policy]]: WAM 是架构无关的更广义范式，Video Policy 专指继承视频生成骨干的方法

### 方法相关
- [[Action Chunking]]: WAM 动作输出的标准形式
- [[JEPA|Joint-Embedding Predictive Architecture]]: 隐式预测潜在表示的理论基础（VLA-JEPA）
- [[Inverse Dynamics Model]]: Cascaded WAM 第二阶段的核心组件
- [[Flow Matching|Flow Matching]]: DreamZero、Cosmos Policy 等的生成框架

### 硬件/数据相关
- [[OXE Dataset]]: 最重要的机器人遥操作数据集聚合
- [[LIBERO]]: 最常用的 WAM 评估基准

---

## 速查卡片

> [!summary] World Action Models Survey (2026)
> - **核心**: WAM = 统一预测状态建模（World Model）+ 动作生成（VLA）的具身基础模型，目标是联合分布 $p(o', a | o, l)$
> - **分类**: Cascaded WAM（先预测未来状态再推导动作）vs. Joint WAM（联合生成）
> - **代表**: DreamZero (14B Wan2.1)、Cosmos Policy (2B)、UWM、FLARE、VLA-JEPA、GR-2
> - **GitHub**: https://github.com/OpenMOSS/Awesome-WAM

---

*笔记创建时间: 2026-05-14*
