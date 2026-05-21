---
title: "SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution"
method_name: "SWEET"
authors: [Yiren Song, Yihan Wang, Xiyao Deng, Zhuoran Yan, Mike Zheng Shou]
year: 2026
venue: arXiv
tags: [sparse-world-model, image-editing, keyframe-planning, robot-manipulation, diffusion-policy]
zotero_collection: Robotics/World Model
image_source: local
created: 2026-05-21
---

# 论文笔记：SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Show Lab, National University of Singapore; Central South University |
| 日期 | May 2026 |
| 项目主页 | https://github.com/showlab/SWEET |
| 对比基线 | [[Wan2.2]]、[[FLUX-Kontext]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.19319) / [Code](https://github.com/showlab/SWEET) |

---

## 一句话总结

> 用图像编辑模型替代视频生成模型作为稀疏视觉世界模型，以更低成本生成任务级关键帧，再通过目标条件扩散策略执行机器人动作。

---

## 核心贡献

1. **图像编辑 vs. 视频生成的受控对比**: 在相同机器人数据设置下，证明图像编辑（FLUX-Kontext）在关键帧预测质量和推理效率上均优于密集视频生成（Wan2.2）。
2. **SWEET 稀疏视觉规划框架**: 通过逐步图像编辑生成任务级关键帧序列，条件为语言指令和可选的箭头空间引导，无需密集视频滚出。
3. **混合训练策略**: 引入真实与编辑关键帧的 1:1 混合训练，有效减少动作预测器在真实-编辑子目标之间的域差距。

---

## 问题背景

### 要解决的问题

密集视频生成作为机器人视觉预测的中间表征计算代价高昂，而大量操作任务只需少量关键状态（接近、接触、抓取、搬运、放置）即可描述，密集帧包含大量冗余信息。

### 现有方法的局限

- [[WAM|World Action Model]] 类方法联合预测视频帧与动作，但密集视频推理在 H20 GPU 上耗时 400+ 秒（81帧 / 任务）
- 现有图像子目标方法通常只预测单步目标或最终状态，未探索基于现代图像编辑模型的多步稀疏关键帧序列生成
- 生成的视觉子目标与真实观测存在域差距，直接用于动作预测器会导致性能下降

### 本文的动机

[[FLUX-Kontext]] 等现代图像编辑模型在 in-context 图像变换方面表现优异：保留源场景结构（背景、物体身份）同时按语义修改任务相关区域——这与机器人关键帧预测的需求天然对齐。

---

## 方法详解

### 模型架构

SWEET 采用**两阶段解耦**架构：

- **高层规划器**（Image Editing Planner）：基于 [[FLUX-Kontext]] + [[LoRA]] 微调，生成稀疏关键帧序列
- **低层执行器**（Goal-conditioned Action Predictor）：基于 [[Diffusion Policy]] 架构，将相邻关键帧转化为可执行动作块
- **输入**: 初始观测 $o_0$ + 子任务文本 $l_m$ + 可选箭头空间提示 $r_m$
- **输出**: 稀疏关键帧计划 $\hat{K}$ + 可执行动作块 $\hat{A}_{t:t+H-1}$

### 核心模块

#### 模块1: 关键帧数据集构建

**设计动机**: 为[[稀疏视觉规划]]提供高质量监督信号。

**具体实现**:
- 对每条轨迹手动标注任务级关键帧（接近/接触/抓取/搬运/放置/释放）
- 每个关键帧转换包含：$(k_m, k_m^{arr}, l_m, k_{m+1}, A_m)$
  - $k_m$：当前关键帧
  - $k_m^{arr}$：叠加箭头的关键帧（箭头颜色编码夹爪状态变化：开→关/关→开）
  - $l_m$：子任务文本描述
  - $k_{m+1}$：目标关键帧
  - $A_m$：两帧之间的动作序列
- 箭头标注经测试可用 Gemini-3 等 VLM 自动生成，但关键帧本身仍需人工标注

#### 模块2: 图像编辑规划器（Image Editing Planner）

**设计动机**: 利用 [[FLUX-Kontext]] 的 in-context 图像编辑能力实现稀疏关键帧预测。

**具体实现**:
- 基于 FLUX.1-Kontext-dev，对 DiT backbone 做 rank=32 的 [[LoRA]] 微调
- 训练 6K steps，单张 NVIDIA H20 GPU，AdamW lr=1e-4
- 训练时随机 drop 箭头条件（保留文本），使模型同时支持纯语言和语言+箭头两种规划模式
- 推理时从初始观测出发，逐步生成关键帧序列，每步以当前关键帧+文本+箭头为条件

#### 模块3: 目标条件扩散动作预测器（Goal-conditioned Diffusion Action Predictor）

**设计动机**: 将稀疏视觉关键帧转化为可执行机器人动作，采用闭环滚动时域方式执行。

**具体实现**:
- 基于 [[Diffusion Policy]] 架构，扩展为起始帧+目标帧双条件策略
- 当前观测和预测子目标图像分别由视觉编码器编码后拼接，作为动作扩散模型的条件
- 动作块长度 $H = 16$，DDPM 去噪目标，AdamW lr=1e-4，batch size=32
- **闭环执行**：高层关键帧计划一次性生成（open-loop），低层动作执行为滚动时域闭环控制

#### 模块4: 混合训练策略（Mixed Training with Edited Transitions）

**设计动机**: 减少训练（真实关键帧）与推理（编辑生成关键帧）之间的域差距。

**具体实现**:
- 用图像编辑规划器为每条真实转换生成编辑目标帧 $\tilde{k}_{m+1}$
- 过滤低质量编辑样本（背景扭曲、物体消失、机器人不一致、错误任务状态）
- 将过滤后的编辑转换与原始真实转换以 **1:1** 比例混合训练
- 动作标签不变，仅改变目标帧的视觉分布

---

## 关键公式

### 公式1: [[稀疏视觉规划|问题形式化]]

操作轨迹 $\tau = \{(x_t, a_t)\}_{t=1}^T$ 用稀疏关键帧序列表示：

$$
K = \{k_0, k_1, \ldots, k_M\}, \quad k_m = x_{t_m}, \quad t_0 < t_1 < \cdots < t_M
$$

**含义**: 将整条轨迹压缩为 $M+1$ 个任务关键状态，两帧间动作序列为 $A_m = (a_{t_m}, a_{t_m+1}, \ldots, a_{t_{m+1}-1})$

**符号说明**:
- $K$: 稀疏关键帧序列
- $k_m$: 第 $m$ 个关键帧对应的 RGB 观测
- $t_m$: 关键帧时间步
- $A_m$: 两关键帧之间的动作序列

### 公式2: [[SWEET|SWEET 规划目标]]

$$
\hat{K} = E_\theta(o_0, C), \quad \hat{A}_{t:t+H-1} = \pi_\phi(o_t, \hat{k}_{m+1})
$$

**含义**: 规划器 $E_\theta$ 从初始观测和子任务条件集合 $C = \{c_m\}_{m=0}^{M-1}$ 生成关键帧计划；动作预测器 $\pi_\phi$ 以当前观测和下一关键帧为条件输出动作块。

**符号说明**:
- $E_\theta$: 图像编辑规划器（参数 $\theta$）
- $\pi_\phi$: 目标条件动作预测器（参数 $\phi$）
- $c_m = (l_m, r_m)$: 子任务条件（语言 + 可选箭头）
- $H$: 动作预测时域（$H=16$）

### 公式3: [[FLUX-Kontext|图像编辑规划器推理]]

$$
\hat{k}_{m+1} = E_\theta(k_m, l_m, r_m), \quad m = 0, \ldots, M-1
$$

**含义**: 规划器递归地将当前关键帧 $k_m$ 和条件 $(l_m, r_m)$ 转化为下一关键帧预测 $\hat{k}_{m+1}$，从 $\hat{k}_0 = o_0$ 出发逐步构建稀疏关键帧计划。

**符号说明**:
- $k_m$: 当前关键帧（推理时为上一步预测帧）
- $l_m$: 子任务文本描述
- $r_m$: 可选箭头空间提示

### 公式4: [[Diffusion Policy|动作预测器训练目标]]

$$
\phi^* = \arg\min_\phi \mathcal{L}_{act}(o_t, k_{m+1}, A_t^H)
$$

其中 $A_t^H = (a_t, \ldots, a_{t+H-1})$ 为固定时域动作块。

**含义**: 用标准 DDPM 去噪目标训练动作预测器，以当前观测和目标关键帧为条件，预测固定长度的动作块。

**符号说明**:
- $\mathcal{L}_{act}$: 扩散动作生成损失（DDPM 去噪目标）
- $A_t^H$: 从时间步 $t$ 开始的 $H$ 步动作块

### 公式5: [[SWEET|闭环执行]]

$$
\hat{A}_{t:t+H-1} = \pi_\phi(o_t, \hat{k}_{m+1})
$$

**含义**: 推理时，动作预测器以实时观测 $o_t$ 和预规划的目标关键帧 $\hat{k}_{m+1}$ 为条件，以滚动时域方式输出动作块并执行，直至到达目标帧对应状态。

---

## 关键图表

### Figure 1: SWEET Teaser — 稀疏关键帧规划示意

![[SWEET_fig1_teaser_initial.png]]
![[SWEET_fig1_teaser_kf2.png]]
![[SWEET_fig1_teaser_kf4.png]]

**说明**: SWEET 将语言引导的操作指令转化为稀疏视觉关键帧序列（通过逐步图像编辑），再由目标条件动作预测器执行。上图展示了两类任务（DROID 真实机器人场景）：第一类将易拉罐放到金属部分，第二类将零食袋移入上层柜子。每个任务有 3 个关键帧对应"抓取 → 搬运 → 放置释放"。

### Figure 2: SWEET 系统架构总览

![[SWEET_fig2_architecture_a.png]]

**说明**: SWEET 包含三个训练和推理组件：(a) **图像编辑规划器训练**：以当前关键帧+文本提示为条件，目标为下一关键帧；(b) **动作预测器训练**：以当前帧+目标关键帧为视觉条件，预测动作块；(c) **推理流水线**：规划器一次性生成完整关键帧计划，动作预测器按序闭环执行各子任务。图中 🔥 = 可学习参数，❄ = 冻结参数。

### Figure 3: 定性比较 — FLUX-Kontext vs. Wan2.2

![[SWEET_fig3_qualitative_a.png]]
![[SWEET_fig3_qualitative_b.png]]
![[SWEET_fig3_qualitative_c.png]]
![[SWEET_fig3_qualitative_d.png]]

**说明**: 在 RoboMimic（仿真）和 DROID（真实机器人）上对比关键帧规划质量。FLUX-Kontext（Kontext 行）更好地保持了机器人具身一致性，物体交互更合理。Wan2.2 存在四类典型失败：**Implausible Grasp**（夹爪未形成有效接触）、**Object Deformation**（物体变形/身份改变）、**Arm Deformation**（机械臂/夹爪结构扭曲）、**Object Teleportation**（物体位置不连续跳变）。

### Figure 4: RoboMimic 完整流水线可视化

![[SWEET_fig4_pipeline_a.png]]
![[SWEET_fig4_pipeline_b.png]]

**说明**: SWEET 在 RoboMimic 仿真基准三个任务（Lift、Can、Square）上的完整流水线可视化，包含关键帧规划、动作预测和执行三个阶段。

### Figure 5: 动作预测器消融可视化

![[SWEET_fig5_ablation.png]]

**说明**: 对比 Real-trained、Gen-trained、Mix-trained 三种动作预测器配置在 RoboMimic 上的轨迹差异，Mix-trained 在面对编辑生成子目标时表现最稳定。

---

### Table 1: 关键帧预测定量对比（FLUX-Kontext vs. Wan2.2）

| Dataset | Method | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|--------|--------|---------|
| RoboMimic | Wan 2.2 TI2V 5B | 0.01777 | 19.00 | 0.8563 | 0.1303 |
| RoboMimic | **FLUX-Kontext** | **0.01114** | **20.71** | **0.9119** | **0.07832** |
| DROID-seen | Wan 2.2 TI2V 5B | 0.02180 | 17.35 | 0.8030 | 0.2032 |
| DROID-seen | **FLUX-Kontext** | **0.01896** | **18.14** | **0.8366** | **0.1570** |
| DROID-unseen | Wan 2.2 TI2V 5B | 0.02844 | 15.64 | 0.7337 | 0.2625 |
| DROID-unseen | **FLUX-Kontext** | **0.02682** | **16.02** | **0.7608** | **0.2343** |

**说明**: FLUX-Kontext 在全部三个数据集上的全部四个指标上均优于 Wan2.2。效率对比：H20 GPU 上，FLUX-Kontext 生成 3 帧稀疏关键帧约需 **10 秒**，Wan2.2 生成 81 帧密集视频需 **400+ 秒**（速度比 >40x）。

---

### Table 2: 动作预测器消融——混合训练策略（RoboMimic 任务成功率 %）

| Task | Test Target | Real-trained SR↑ | Real-trained MSE↓ | Gen-trained SR↑ | Gen-trained MSE↓ | Mix-trained SR↑ | Mix-trained MSE↓ |
|------|-------------|-----------------|------------------|-----------------|------------------|-----------------|------------------|
| Lift | Generated keyframe | 58.0 | 0.0159 | 91.5 | 0.0171 | **92.0** | **0.0141** |
| Lift | Real keyframe | 87.0 | 0.0165 | 78.5 | 0.0144 | **92.0** | 0.0145 |
| Can | Generated keyframe | 45.0 | 0.0560 | 66.0 | 0.0578 | **81.0** | **0.0534** |
| Can | Real keyframe | 67.0 | 0.0653 | 34.0 | 0.0598 | **79.0** | **0.0527** |
| Square | Generated keyframe | 4.5 | 0.0335 | 13.5 | 0.0339 | **29.5** | **0.0333** |
| Square | Real keyframe | 14.0 | 0.0332 | 10.5 | 0.0335 | **31.5** | 0.0331 |

**关键发现**: 
- Real-trained 和 Gen-trained 均表现出明显的域专一性：在对应域测试时表现好，跨域测试时退化严重（如 Real-trained 在 Lift Generated 仅 58%，Gen-trained 在 Can Real 仅 34%）
- **Mix-trained 在两种目标类型下均表现最佳**，且 Lift/Can 任务上还降低了夹爪位置预测 MSE，证明混合训练能有效桥接真实-编辑子目标域差距
- Square 任务（精确插孔）整体成功率低，但 Mix-trained 仍显著优于单域变体（29.5% vs 4.5%/13.5%）

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] (子集) | 700 样本，~2100 标注子任务 | 真实机器人，10+ 场景，覆盖物体/背景/布局/光照变化 | 真实机器人关键帧预测，seen/unseen 场景各 50 样本测试集 |
| [[Robomimic\|RoboMimic]] | 600 样本，1600 子任务 | 仿真，Lift / Can / Square 三任务 | 关键帧质量 + 下游执行评估，每任务 200 次评估试验 |

### 实现细节

- **高层规划器**: FLUX.1-Kontext-dev，DiT backbone LoRA rank=32，6K steps，AdamW lr=1e-4，单张 H20 GPU
- **低层动作预测器**: Diffusion Policy 架构（start-goal 双条件），动作块 $H=16$，DDPM 目标，AdamW lr=1e-4，batch=32
- **推理关键帧数**: RoboMimic Lift=2 步（接触+提升），Can/Square=3 步；DROID 全部任务=3 步
- **混合训练**: 1:1 真实帧与编辑帧混合，编辑帧需通过视觉一致性和任务可信度过滤

### 可视化结果

- Wan2.2 在 DROID 复杂场景下表现明显劣于 FLUX-Kontext，物体变形和机器人臂结构扭曲问题突出
- SWEET 能在 RoboMimic Lift 任务上达到 **92%** 成功率，在 Can 上达到 **81%**，Square 精确操作任务上达到 **29.5%**（对比 Real-trained 的 14%，提升约 2x）

---

## 批判性思考

### 优点
1. **推理效率极高**: 比密集视频生成快 40x 以上（10s vs 400s），对实时机器人控制更友好
2. **视觉保真度更好**: 图像编辑天然保留场景结构，避免视频模型的时域不一致问题
3. **混合训练策略简洁有效**: 不需要额外模型，只需将规划器生成样本加入训练即可大幅提升鲁棒性
4. **可扩展的规划-执行接口**: 规划器和动作预测器解耦，两者可独立升级

### 局限性
1. **关键帧标注依赖人工**: VLM（如 Gemini-3）尚无法可靠识别机器人视频中的接触/状态转换帧，限制数据规模扩展
2. **动作预测器精度有限**: 尤其在精确操作任务（Square 插孔）中，视觉子目标正确但动作执行仍不够精确，部分原因是训练数据量不足
3. **当前为 one-shot 开环规划**: 高层计划一次性生成后不再更新，对于长时域或需要反馈的任务鲁棒性不足
4. **与 end-to-end 方法的公平对比缺失**: 论文主要对比 Wan2.2 vs. FLUX-Kontext，未与 OpenVLA、π0 等直接端到端 VLA 方法做全面基准对比

### 潜在改进方向
1. 微调视频语言模型用于关键帧自动标注，提升数据规模
2. 联合训练视觉预测和动作预测（类 World Action Model），对齐视觉子目标和动作
3. 引入反馈重规划机制（planner replans after each executed chunk），提升长时域鲁棒性

### 可复现性评估
- [x] 代码开源（https://github.com/showlab/SWEET）
- [ ] 预训练模型（代码库尚未公开，论文中未提供模型权重链接）
- [x] 训练细节完整（LoRA rank、steps、lr、batch size 均列出）
- [ ] 数据集可获取（DROID 子集的关键帧标注未公开）

---

## 关联笔记

### 基于
- [[FLUX-Kontext]]: 图像编辑规划器的基础模型，基于 Flow Matching 的 in-context 图像编辑
- [[Diffusion Policy]]: 目标条件动作预测器的架构基础，DDPM 去噪策略
- [[LoRA]]: 规划器微调方式，rank=32 LoRA 只更新 DiT backbone

### 对比
- [[WAM|World Action Model]]: 联合预测视觉和动作，SWEET 选择解耦稀疏关键帧而非密集视频
- [[Wan2.2]]: 视频生成对比基线，TI2V 5B 模型，关键帧质量和推理效率均低于 SWEET

### 方法相关
- [[World Model]]: SWEET 是一种稀疏视觉世界模型
- [[稀疏视觉规划]]: 论文提出的核心范式，关键帧级别的视觉规划
- [[Action Chunking]]: 动作预测器输出固定长度动作块（H=16）

### 硬件/数据相关
- [[DROID]]: 大规模真实机器人数据集，SWEET 用其子集（700样本）评估真实场景关键帧预测
- [[Robomimic|RoboMimic]]: 仿真基准，用于评估完整流水线执行（Lift/Can/Square）

---

## 速查卡片

> [!summary] SWEET (2026, Show Lab NUS)
> - **核心**: 用图像编辑替代视频生成做稀疏关键帧视觉规划，推理效率提升 40x+
> - **方法**: FLUX-Kontext LoRA 微调 → 关键帧序列 → 目标条件 Diffusion Policy 执行，混合训练减少域差距
> - **结果**: RoboMimic Lift 92%、Can 81% 成功率；DROID seen/unseen 场景关键帧 MSE/PSNR/SSIM/LPIPS 全面优于 Wan2.2
> - **代码**: https://github.com/showlab/SWEET

---

*笔记创建时间: 2026-05-21*
