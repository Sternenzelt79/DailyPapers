---
title: "MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
method_name: "MotionWAM"
authors: [Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang]
year: 2026
venue: arXiv
tags: [world-action-model, humanoid-robot, loco-manipulation, video-diffusion, flow-matching, whole-body-control, real-time]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.09215v1
created: 2026-06-09
---

# 论文笔记：MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mondo Robotics; HKUST (GZ); HKUST |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[GR00T-N1.6\|GR00T-N1.7]]、[[Pi0.5\|π₀.5]]、[[Diffusion Policy]]、ACT |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09215) / Code 暂未公开 |

---

## 一句话总结

> MotionWAM 通过将视频扩散 Transformer 的中间去噪特征直接条件化运动策略，实现实时全身 loco-manipulation，在 9 项 Unitree G1 任务上以 76.1% 成功率大幅超越 VLA 基线（最强基线 43.9%）。

---

## 核心贡献

1. **实时 WAM 架构**: 视频 DiT 仅运行单次前向传播（"one-shot imagination"），将固定噪声时间步 $\tau_f \approx 1$ 下的中间隐状态传递给运动 DiT，彻底消除迭代去噪瓶颈，同时保留时序场景预测信息。
2. **统一全身运动表征**: 以 [[FSQ|Finite Scalar Quantization]] 构建 22 token × 32 级离散运动 token，覆盖行走、躯干姿态、高度调节、脚步交互与手部抓取，取代上下体解耦的层级控制。
3. **三阶段课程训练**: Stage 1 用 ~2,136 小时人类/类人机器人自我中心视频预训练视觉先验；Stage 2 跨具身动作后训练将先验接地到动作空间；Stage 3 任务级遥操作微调，防止各阶段梯度相互污染。

---

## 问题背景

### 要解决的问题

类人机器人的 [[Loco-Manipulation]] 任务（如踢球、推购物车、擦白板）要求腿部与手臂**协同完成任务**，而非仅用腿部维持平衡。现有方法无法做到这一点。

### 现有方法的局限

- **层级解耦控制**: 上身操作策略输出末端执行器目标，下身仅接收粗粒度速度/高度指令，腿的行为完全由平衡保持驱动，无法执行任务驱动的脚步动作。
- **[[World Action Model|WAM]] 推理慢**: Cosmos Policy 等方法需要完整迭代去噪，推理频率仅 0.7 Hz，无法部署在实时控制闭环中。
- **[[VLA|Vision-Language-Action]] 缺乏时序场景预测**: 现有 VLA 模型直接从图像映射到动作，缺少对未来场景的显式建模。

### 本文的动机

若能从视频世界模型中提取**中间去噪特征**（单次前向传播即可获得），则可在保持实时推理速率的同时，为运动策略提供丰富的时序预测上下文，再结合统一全身运动 token，自然实现上下体协同。

---

## 方法详解

### 模型架构

MotionWAM 采用 **Dual-DiT** 架构：
- **输入**: 语言指令 $l$ + 自我中心 RGB 观测 $\mathbf{o}_t$ + 本体感受状态 $\mathbf{p}_t$
- **视频 DiT** (`Video DiT`): 从 [[Cosmos-Predict2.5]] (2B) 初始化，在固定噪声时间步 $\tau_f \approx 1$（纯噪声初始化）运行**单次前向传播**，提取隐状态 $\mathbf{h}_t^{\tau_f}$
- **运动 DiT** (`Motion DiT`): DiT-B 配置，通过交叉注意力（interleaved cross-attention）接收 $\mathbf{h}_t^{\tau_f}$，预测全身运动 latent $\mathbf{m}_t$
- **输出**: 离散运动 token $\hat{k}_t$（经 [[SONIC]] 控制器解码为关节指令）+ 连续末端执行器指令 $\hat{\mathbf{m}}_t^{cont}$
- **总参数**: 2.5B（可训练）

### 核心模块

#### 模块 1: One-Shot Imagination（单次想象）

**设计动机**: 利用视频扩散模型对场景动态的强大先验，同时规避迭代去噪的巨大延迟。

**具体实现**:
- 将 $\tau_f$ 固定为接近 1 的值（纯噪声），此时 $\mathbf{z}_{t+1}^{\tau_f} \sim \mathcal{N}(0,I)$
- 运行视频 DiT 一次前向传播：$\mathbf{h}_t^{\tau_f} = \mathcal{H}[v_\theta^{video}](\mathbf{z}_{t+1}^{\tau_f}, \tau_f \mid \mathbf{z}_t^0, l)$
- 中间 Transformer 层的 attention 隐状态已编码未来场景的结构信息，无需完成去噪即可使用

#### 模块 2: 统一全身运动 Token（Unified Whole-Body Motion Token）

**设计动机**: 将连续末端轨迹与离散运步指令编码进同一 latent 空间，使策略可以同时学习手部操作与任务驱动脚步。

**具体实现**:
- 离散部分：22 个 token，每 token 32 级，通过 [[FSQ]] 量化
- 连续部分：灵巧手 / ALOHA2 夹爪命令 $\mathbf{m}_t^{cont}$
- 运动 latent 编解码：$\mathbf{m}_t = (\mathbf{m}_t^{cont}, \tilde{k}_t) \to \hat{\mathbf{m}}_t = (\hat{\mathbf{m}}_t^{cont}, \hat{\tilde{k}}_t) \to \hat{k}_t = \text{round}(\hat{\tilde{k}}_t) \to \text{SONIC} \to \mathbf{a}_t$

#### 模块 3: 三阶段训练框架（Three-Stage Training）

| 阶段 | 目标 | 数据 | 损失 |
|------|------|------|------|
| Stage 1 | 自我中心视频预训练 | ~2,136h（人类 30%、G1 类人 50%、其他机器人 20%） | $\mathcal{L}_{video}$ |
| Stage 2 | 跨具身动作后训练 | 异构 Unitree G1 数据集 + 各具身投影头 | $\mathcal{L}_{Stage2} = \mathcal{L}_{motion} + \mathcal{L}_{video}$ |
| Stage 3 | 任务级全身微调 | 每任务 200 条遥操作演示（共 1,800 条） | $\mathcal{L}_{motion}$ |

---

## 关键公式

### 公式 1: [[World Action Model|问题形式化]]

$$
\mathbf{o}_{t+1} \sim p_v(\cdot \mid \mathbf{o}_t, l), \quad \mathbf{m}_t \sim p_a\!\left(\cdot \mid \mathbf{o}_t, \mathbf{p}_t, \mathcal{H}(\mathbf{o}_{t+1}^{\tau_v})\right)
$$

**含义**: 世界模型预测下一帧 $\mathbf{o}_{t+1}$，运动策略以当前观测 $\mathbf{o}_t$、本体状态 $\mathbf{p}_t$ 和从视频隐状态提取的信息 $\mathcal{H}(\mathbf{o}_{t+1}^{\tau_v})$ 为条件预测运动 latent。

**符号说明**:
- $\mathbf{o}_t$: 当前自我中心 RGB 观测
- $l$: 自然语言指令
- $\mathbf{p}_t$: 本体感受状态
- $\mathcal{H}$: 从视频 DiT 中提取隐状态的算子
- $\tau_v$: 视频去噪时间步（趋向 0 时为干净帧）

### 公式 2: [[Cosmos-Predict2.5|Video DiT 隐状态提取]]

$$
\mathbf{h}_t^{\tau_f} = \mathcal{H}[v_\theta^{video}](\mathbf{z}_{t+1}^{\tau_f},\, \tau_f \mid \mathbf{z}_t^0,\, l), \quad \mathbf{z}_{t+1}^{\tau_f}\big|_{\tau_f \to 1} \sim \mathcal{N}(0, I)
$$

**含义**: 在纯噪声初始化 $\tau_f = 1$ 处运行视频 DiT 单次前向传播，提取中间层 attention 隐状态 $\mathbf{h}_t^{\tau_f}$，作为运动 DiT 的条件输入。

**符号说明**:
- $\mathbf{z}_{t+1}^{\tau_f}$: 时间步 $\tau_f$ 处的含噪视频 latent
- $\mathbf{z}_t^0$: 当前帧干净 latent（编码器输出）
- $v_\theta^{video}$: 视频 DiT 速度场网络
- $\tau_f$: 固定提取时间步，$\tau_f \approx 1$

### 公式 3: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{video} = \mathbb{E}_{\tau_v,\, \mathbf{z}_{t+1}^0,\, \epsilon_v}\!\left[\left\|v_\theta^{video}\!\left(\mathbf{z}_{t+1}^{\tau_v}, \tau_v \mid \mathbf{z}_t^0, l\right) - \left(\epsilon_v - \mathbf{z}_{t+1}^0\right)\right\|_2^2\right]
$$

**含义**: 视频 DiT 的 [[Flow Matching]] 目标——预测从干净帧到噪声的速度场。

**符号说明**:
- $\tau_v \sim \mathcal{U}(0,1)$: 随机采样的去噪时间步
- $\epsilon_v$: 高斯噪声
- $\mathbf{z}_{t+1}^0$: 未来帧干净 latent（目标）

### 公式 4: [[Flow Matching|运动流匹配损失]]

$$
\mathcal{L}_{motion} = \mathbb{E}_{\tau_a,\, \mathbf{m}_t^0,\, \epsilon_m}\!\left[\left\|v_\phi^{motion}\!\left(\mathbf{m}_t^{\tau_a}, \tau_a \mid \mathbf{h}_t^{\tau_f}, \mathbf{p}_t, e\right) - \left(\epsilon_m - \mathbf{m}_t^0\right)\right\|_2^2\right]
$$

**含义**: 运动 DiT 的 [[Flow Matching]] 目标——以视频隐状态、本体状态和具身标识为条件，预测全身运动 latent 的速度场。

**符号说明**:
- $\mathbf{m}_t^0$: 干净运动 latent（训练目标）
- $\tau_a \sim \mathcal{U}(0,1)$: 运动去噪时间步
- $\epsilon_m$: 运动空间高斯噪声
- $e$: 具身标识符（用于跨具身训练）
- $v_\phi^{motion}$: 运动 DiT 速度场网络

### 公式 5: Stage 2 联合损失

$$
\mathcal{L}_{Stage2} = \mathcal{L}_{motion} + \mathcal{L}_{video}
$$

**含义**: 第二阶段同时优化运动预测与视频生成，视频损失作为表征正则项防止遗忘 Stage 1 学到的视觉先验。

### 公式 6: [[SONIC|全身运动 Latent 解码流程]]

$$
\mathbf{m}_t = \left(\mathbf{m}_t^{cont},\, \tilde{k}_t\right) \xrightarrow{\text{Motion DiT}} \hat{\mathbf{m}}_t = \left(\hat{\mathbf{m}}_t^{cont},\, \hat{\tilde{k}}_t\right) \xrightarrow{\text{round}} \hat{k}_t \xrightarrow{\text{SONIC}} \mathbf{a}_t
$$

**含义**: 运动 DiT 输出连续末端指令和量化前的离散运动 token，经取整后输入 SONIC 全身运动追踪控制器，最终映射为关节级动作 $\mathbf{a}_t$。

**符号说明**:
- $\mathbf{m}_t^{cont}$: 连续末端执行器命令
- $\tilde{k}_t$: 连续化的离散运动 token（[[FSQ]] 软编码）
- $\hat{k}_t = \text{round}(\hat{\tilde{k}}_t)$: 整数化运动 token
- $\mathbf{a}_t$: 最终关节级动作

---

## 关键图表

### Figure 1: MotionWAM 整体概览

![Figure 1 Overview](https://arxiv.org/html/2606.09215v1/x2.png)

**说明**: 展示传统层级架构（上身操作 + 下身平衡解耦）与 MotionWAM 统一全身控制的对比，以及 Unitree G1 执行多样化 loco-manipulation 任务的示例（踢球、推车、擦白板等）。

### Figure 2: Dual-DiT 三阶段训练架构

![Figure 2 Dual-DiT Architecture](https://arxiv.org/html/2606.09215v1/x3.png)

**说明**: 完整的 MotionWAM 双 DiT 模型架构与三阶段训练流程。视频 DiT（蓝色，冻结/微调）通过单次前向传播提取 $\mathbf{h}_t^{\tau_f}$，通过交叉注意力传递给运动 DiT（橙色）。三阶段逐步从通用视频先验收窄到任务级全身控制。

### Figure 3: 9 项真实世界任务套件

![Figure 3 Task Suite](https://arxiv.org/html/2606.09215v1/x4.png)

**说明**: Unitree G1 上的 9 项任务，均要求腿部主动参与任务执行而非仅维持平衡：取放瓶子、踢足球、取物品、装载推车、扔垃圾、提篮子、货架上货、擦白板、洗衣服。

### Figure 4: 与 SOTA 方法的逐任务对比

![Figure 4 SOTA Comparison](https://arxiv.org/html/2606.09215v1/x5.png)

**说明**: MotionWAM（76.1% 平均成功率）全面超越所有基线。协调性要求最高的任务增益最大：Wipe Board +45%，Kick Soccer +40%，Load Cart +40%，Retrieve Item +40%。

### Figure 5: 失败案例分析

![Figure 5 Failure Cases](https://arxiv.org/html/2606.09215v1/x6.png)

**说明**: 主要失败模式为被操作物体离开自我中心摄像头视野，或头部视角偏离训练分布。单目 RGB 输入在遮挡和视角变化情况下存在固有局限。

### Figure 6: 代表性成功 Rollout 展示

![Figure 6 Successful Rollouts](https://arxiv.org/html/2606.09215v1/x7.png)

**说明**: 展示多个任务中机器人腿臂协调执行全身运动的连续帧，包含非平衡行为（脚踩踏板、移步交互）。

### Table 1: 动作空间对比——层级解耦 vs. 统一全身

| 方面 | 层级解耦（传统） | MotionWAM（统一） |
|------|-----------------|-------------------|
| 上身 | 精细关节目标 | 统一运动 token |
| 下身 | 粗粒度基座指令（速度、高度、朝向） | 任务驱动运动控制 |
| 能力 | 仅平衡保持型运动 | 踩踏板、踢球等任务驱动脚步交互 |

**说明**: 统一运动 token 空间打破了上下体之间的显式解耦，使腿部行为可由任务目标直接驱动。

### Table 2: 三阶段训练消融实验

| 变体 | Stage 1 | Stage 2 | Lift Basket | Retrieve Item | Load Cart | Toss Garbage | Kick Soccer | 均值 |
|------|---------|---------|-------------|---------------|-----------|--------------|-------------|------|
| w/o Stage 2 | ✓ | — | 65 | 45 | 30 | 30 | 40 | 42.0 |
| w/o Stage 1 | — | ✓ | 70 | 75 | 60 | 35 | 55 | 59.0 |
| Full Model | ✓ | ✓ | **80** | **90** | **75** | **45** | **60** | **70.0** |

**关键发现**: Stage 2（跨具身动作后训练）对性能贡献更大（去除后均值降至 42.0%），Stage 1（视频预训练）也不可缺失（去除后均值降至 59.0%）。

### Table 3: 推理效率对比

| 模型 | 可训练参数 | 推理频率 (Hz) |
|------|-----------|--------------|
| GR00T-N1.7 | 1.6B | 6.5 |
| Qwen3DiT | 2.3B | 9.0 |
| Cosmos Policy | 2.0B | 0.7 |
| **MotionWAM（Ours）** | **2.5B** | **4.9** |

**关键发现**: MotionWAM 以 4.9 Hz 实现实时推理，比参数量相近的 Cosmos Policy 快 **7×**，根源在于省去了迭代去噪过程。

### Table 4: 每任务语言提示（Language Prompts）

| 任务 | 语言提示 |
|------|---------|
| PnP Bottle | "Pick the bottle and place it in the basket." |
| Kick Soccer | "Kick the soccer into the goal net." |
| Retrieve Item | "Put the bag on the table and then close the drawer." |
| Load Cart | "Push the cart forward and put the clothes on the table into the cart." |
| Toss Garbage | "Throw the garbage into the trash can." |
| Lift Basket | "Take out the clothes basket under the table and place it on the table." |
| Stock Shelves | "Place the drinks on the upper shelf and the vegetables on the lower shelf." |
| Wipe Board | "Clean the whiteboard thoroughly." |
| Do Laundry | "Throw the clothes into the washing machine." |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 人类自我中心视频 | ~641h（~30%） | 第一人称人类活动 | Stage 1 视频预训练 |
| G1 类人机器人视频 | ~1,068h（~50%） | Unitree G1 类平台 | Stage 1 视频预训练 |
| 其他机器人视频 | ~427h（~20%） | 跨具身多样性 | Stage 1 视频预训练 |
| 异构 Unitree G1 数据 | 未披露规模 | 跨具身、多投影头 | Stage 2 动作后训练 |
| 遥操作演示 | 1,800 条（200 条×9 任务） | 真实机器人全身演示 | Stage 3 微调 |

### 实现细节

- **视频 DiT Backbone**: Cosmos-Predict2.5-2B（Stage 1 全量训练，Stage 2/3 微调）
- **运动 DiT**: DiT-B 配置，从随机初始化开始
- **运动 Token**: [[FSQ]] 编码，22 token × 32 级（512 维离散空间）
- **硬件**: NVIDIA RTX 4090 工作站（推理）；训练硬件未披露
- **机器人平台**: Unitree G1 + 双 ALOHA2 夹爪
- **传感器**: Intel RealSense D435i（自我中心 RGB，头部安装）
- **全身控制器**: [[SONIC]] whole-body motion tracker

### 可视化结果

全身协调任务中，MotionWAM 展现出：(1) 踢球时腿部前摆与手臂平衡的自然协调；(2) 擦白板时沿墙侧步移动躯干；(3) 装载购物车时双臂推车同步前行。这些行为在层级解耦基线中均无法复现。

---

## 批判性思考

### 优点
1. **架构简洁高效**: 用单次前向传播代替迭代去噪，在保留视频先验的同时实现实时控制，工程设计思路清晰。
2. **统一动作空间突破**: FSQ 运动 token 将连续与离散控制融入单一 latent，是对传统层级控制的实质性突破。
3. **三阶段数据效率**: 通过廉价视频数据完成大规模预训练，任务微调仅需每任务 200 条演示，成本可控。
4. **任务多样性**: 9 项任务覆盖踢球、推车、整理货架等具有挑战性的 loco-manipulation 场景，评估较为全面。

### 局限性
1. **平台局限**: 所有实验均在 Unitree G1 上进行，跨平台迁移能力（如 H1、Digit）未经验证。
2. **单目视角依赖**: 单个自我中心 RGB 摄像头在目标遮挡或头部视角偏移时性能显著下降，缺乏鲁棒性研究。
3. **无 OOD 评估**: 未进行超出训练分布的物体/场景泛化测试，实际部署鲁棒性存疑。
4. **频率仍有限**: 4.9 Hz 虽优于竞争 WAM 方法，但仍低于基于 ACT 的方法（通常 > 10 Hz），对高频动态任务可能不够。

### 潜在改进方向
1. 引入多视角（立体/侧向）输入以缓解单目遮挡问题
2. 在 $\tau_f$ 上进行自适应搜索，探索不同去噪深度对策略质量的权衡
3. 将 Stage 2 扩展到更多机器人平台，验证 Foundation WAM 的跨具身通用性

### 可复现性评估
- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节（三阶段框架、数据构成、任务提示均有描述）
- [ ] 数据集可获取（遥操作数据集未公开）

---

## 关联笔记

### 基于
- [[Cosmos-Predict2.5]]: 视频 DiT backbone，提供场景动态先验
- [[SONIC]]: 全身运动追踪控制器，将运动 token 解码为关节级动作
- [[FSQ]]: Finite Scalar Quantization，构建离散运动 token

### 对比
- [[GR00T-N1.6\|GR00T-N1.7]]: NVIDIA 类人机器人 VLA 基线（43.9% vs 76.1%）
- [[Pi0.5]]: 物理智能双臂 VLA 基线（35.1%）
- [[Diffusion Policy]]: 扩散策略基线（15.2%）

### 方法相关
- [[World Action Model|WAM]]: 本文所属的大类别——统一世界预测与动作生成的模型
- [[Flow Matching]]: 视频 DiT 与运动 DiT 共用的生成目标
- [[Loco-Manipulation]]: 本文聚焦的任务类型——腿臂协同操控
- [[VLA]]: 本文相对的对比框架

### 硬件/数据相关
- [[Unitree\|Unitree G1]]: 实验平台

---

## 速查卡片

> [!summary] MotionWAM (arXiv 2606.09215)
> - **核心**: 视频 DiT 单次前向传播提取隐特征 → 条件化全身运动 DiT，实现实时 loco-manipulation
> - **方法**: Dual-DiT（Cosmos-Predict2.5 + DiT-B）+ FSQ 运动 token + 三阶段课程训练
> - **结果**: Unitree G1 九任务 76.1% 成功率，超最强基线 +32%，推理 4.9 Hz（比 Cosmos Policy 快 7×）
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-09*
