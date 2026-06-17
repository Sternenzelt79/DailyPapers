---
title: "iMaC: Translating Actions into Motion and Contact Images for Embodied World Models"
method_name: "iMaC"
authors: [Zhenyu Wu, Xiuwei Xu, Yukun Zhou, Yifan Li, Qiuping Deng, Xiaofeng Wang, Zheng Zhu, Bingyao Yu, Ziwei Wang, Jiwen Lu, Haibin Yan]
year: 2026
venue: CoRL 2026 (under review)
tags: [world-model, embodied-ai, robot-manipulation, action-representation, video-generation, contact-estimation, policy-evaluation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.09813
created: 2026-06-15
---

# 论文笔记：iMaC: Translating Actions into Motion and Contact Images for Embodied World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beijing University of Posts and Telecommunications, Tsinghua University, GigaAI, Nanyang Technological University |
| 日期 | June 2026 |
| 项目主页 | [imac-wm.github.io](https://imac-wm.github.io/) |
| 对比基线 | [[Ctrl-World]], [[ABot-PhysWorld]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09813) / [Code](https://github.com/imac-wm/iMac) |

---

## 一句话总结

> iMaC 将机器人动作转换为运动图像（Motion Images）和接触图像（Contact Images）作为 Embodied World Model 的视觉控制条件，从根本上替代低维动作向量表示，实现更精准的接触感知视频预测和策略评估。

---

## 核心贡献

1. **Motion Images（运动图像）**: 利用 [[URDF]] 和[[正向运动学]]将未来动作序列渲染为机器人外观视频，直接为 [[世界模型|World Model]] 提供像素级运动引导，无需模型自行推断运动学关系
2. **Contact Images（接触图像）**: 构造双流几何控制信号——机器人到场景距离图与场景到夹爪距离图，编码接触相关的空间几何关系，引导模型预测精准的物体交互
3. **Training-time Rollout（训练时滚动生成）**: 在训练阶段引入分块（chunk-wise）生成策略，让后续 chunk 以预测帧而非真实帧为参考，降低训练-测试 mismatch，支持分钟级长时视频生成
4. **Policy Evaluation Framework（策略评估框架）**: 在 8 个真实操作任务上验证 world model 成功率估计与真实世界策略性能之间的强正相关性（r=0.833~0.956）

---

## 问题背景

### 要解决的问题

[[具身智能|Embodied World Models]] 通常以低维动作向量（如关节角度、末端执行器位姿）为条件预测未来观测，但这种表示方式存在严重的信息瓶颈：模型必须从紧凑的向量中"推断"出机器人未来的外观和与环境的接触关系，这对精确的接触感知操作预测极为困难。

### 现有方法的局限

- **向量动作条件**（如 Ctrl-World、ABot-PhysWorld）：将动作向量直接注入生成模型，缺乏显式的运动先验，无法精确建模接触时刻的几何关系
- **泛化受限**：依赖手工定义的动作空间，难以迁移到不同机器人形态（heterogeneous embodied agents）
- **训练-测试 mismatch**：长时生成时，后续帧依赖预测帧，但训练时以真实帧为参考，导致误差积累

### 本文的动机

通过将动作转换为与视频 backbone 天然兼容的图像形式（运动图像 + 接触图像），让世界模型直接"看到"机器人未来的外观和空间关系，从而大幅降低预测难度并提升接触感知能力。这一范式消除了对手工动作空间的依赖，为异构机器人提供通用控制。

---

## 方法详解

### 模型架构

![Figure 1: iMaC Pipeline](https://arxiv.org/html/2606.09813v1/x1.png)

iMaC 采用 **图像到视频扩散变换器（IT2V DiT）** 架构，以 **WAN2.2** 为视频生成 backbone：

- **输入**: 语言指令 $l$ + 参考观测 $o_t$（多视角 RGB 拼接为单图 mosaic）+ 动作序列 $a_{t:t+H-1}$
- **Backbone**: [[WAN2.2]]（图像到视频 [[Diffusion Transformer]]）
- **核心模块**: [[Motion Images|运动图像]] + [[Contact Images|接触图像]] 双流控制
- **输出**: 未来视频帧 $\hat{o}_{t+1:t+H}$（含 RGB 和深度）

**Token 组合公式**（模型输入构造）：

$$
h_\tau = [P_v(z^r);\ P_v(x_\tau) + P_m(E(C^m)) + P_{s \to g}(E(C^{s \to g})) + P_{r \to s}(E(C^{r \to s}))]
$$

其中 $z^r$ 为参考帧 token，$x_\tau$ 为当前噪声帧，$C^m, C^{s \to g}, C^{r \to s}$ 分别为运动图像、场景到夹爪接触图像、机器人到场景接触图像，$P_*$ 为各流的投影层，$E$ 为编码器。

**核心参数与训练规模**：两阶段训练——阶段 1 在全部 8 个任务上共训练运动图像控制的共享模型，阶段 2 针对每个任务引入接触图像控制进行微调。

---

### 核心模块

#### 模块 1: Motion Images（运动图像）

**设计动机**: 利用 [[URDF]] 和 [[正向运动学]] 将动作序列"可视化"，给 world model 提供像素级的机器人运动先验，无需模型自行推断关节运动关系。

**具体实现**:

1. **关节状态计算**: 基于当前关节角 $q_t$ 和动作序列 $a_{t:t+k-1}$，递推计算未来关节状态：

$$
q_{t+k} = \phi(q_t,\ a_{t:t+k-1})
$$

2. **正向运动学**: 将关节角映射为机器人网格（mesh）：

$$
M_{t+k} = K_{\mathrm{URDF}}(q_{t+k})
$$

3. **渲染为图像**: 在各视角相机参数 $K_v, T^v_{t+k}$ 下渲染机器人外观：

$$
C^m_{t+k,v} = R(M_{t+k};\ K_v,\ T^v_{t+k})
$$

Motion Images 提供像素级运动引导，解决了向量动作条件无法显式表达机器人外观的根本问题。

#### 模块 2: Contact Images（接触图像）

**设计动机**: 操作任务的成功与否往往取决于夹爪与目标物体的精确接触。接触图像通过几何距离场显式编码这一关键空间关系。

**双流几何控制**:

**Stream 1 — 机器人到场景距离（Robot-to-Scene Distance）**:

$$
d^{r \to s}_{t+k}(r) = \min_{p \in P^s_t} \|r - p\|_2, \quad r \in P^r_{t+k}
$$

- $P^s_t$：场景点云（由 [[Depth Anything 3]] 估计深度后反投影）
- $P^r_{t+k}$：未来机器人关键点（末端执行器、手指关节等）
- 含义：机器人每个关键点距离最近场景点的距离，指示机器人"要去哪里"

**Stream 2 — 场景到夹爪距离（Scene-to-Gripper Distance）**:

$$
d^{s \to g}_{t+k}(p) = \min_{g \in P^g_{t+k}} \|p - g\|_2, \quad p \in P^s_t
$$

- $P^g_{t+k}$：未来夹爪点云
- 含义：场景每个点距离夹爪的距离，指示目标物体"离夹爪有多近"

两个距离图均以彩色深度图形式编码，与图像到视频 backbone 天然兼容（遵循 VisionBanana 方法论）。

#### 模块 3: Training-time Rollout（训练时滚动生成）

**设计动机**: 推理时长时生成需要以预测帧为参考，而训练时以真实帧为参考，导致 [[Exposure Bias|曝光偏差]] 问题。

**实现方式**:

前热身阶段（前 40 epochs）使用真实帧作为参考，之后切换为一步干净估计（one-step clean estimates）作为后续 chunk 的参考：

$$
\hat{x}^{(r)}_1 = x^{(r)}_\tau + (1 - \tau) v^x_\theta(\ldots)
$$

$$
\hat{d}^{(r)}_1 = d^{(r)}_\tau + (1 - \tau) v^d_\theta(\ldots)
$$

其中 $\tau \sim \mathcal{U}(0,1)$ 为 [[Flow Matching|流匹配]] 时间步，$v_\theta$ 为速度预测网络输出。

此设计在保持与记录数据的成对监督前提下，显著降低了训练-测试 mismatch。

---

## 关键公式

### 公式 1: [[Policy Mapping|策略映射]]

$$
a_{t:t+H-1} = \pi(o_t, l)
$$

**含义**: 策略 $\pi$ 基于当前观测 $o_t$ 和语言指令 $l$，预测未来 $H$ 步动作序列。

**符号说明**:
- $a_{t:t+H-1}$：从时刻 $t$ 到 $t+H-1$ 的动作序列
- $o_t$：当前时刻多视角观测
- $l$：语言任务指令
- $H$：预测地平线（horizon）

### 公式 2: [[World Model Prediction|世界模型预测]]

$$
\hat{o}_{t+1:t+H} = f_\theta(o_t, a_{t:t+H-1}, l)
$$

**含义**: World model $f_\theta$ 以当前观测、动作序列和语言指令为条件，预测未来 $H$ 帧观测。

**符号说明**:
- $\hat{o}_{t+1:t+H}$：预测的未来视频帧
- $f_\theta$：参数为 $\theta$ 的世界模型（WAN2.2 IT2V DiT）

### 公式 3: [[Flow Matching|流匹配]] 噪声过程

$$
x_\tau = (1 - \tau) x_0 + \tau x_1
$$

**含义**: 在时间步 $\tau$ 下，将真实数据 $x_0$ 与噪声 $x_1$ 进行线性插值，构造流匹配训练轨迹。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$：流匹配时间步，均匀采样
- $x_0$：真实视频帧
- $x_1$：纯噪声

### 公式 4: [[Flow Matching Loss|流匹配损失]]

$$
\mathcal{L}_{fm} = \mathbb{E}\left[\left\| v_\theta(h_\tau, \tau, l) - (x_1 - x_0) \right\|^2_2 \right]
$$

**含义**: 速度预测网络 $v_\theta$ 学习预测从 $x_0$ 到 $x_1$ 的速度场方向，即流匹配目标。

**符号说明**:
- $v_\theta$：速度预测网络（WAN2.2 DiT）
- $h_\tau$：当前时间步的 token 序列（含参考帧、噪声帧、控制图像）
- $x_1 - x_0$：目标速度方向

### 公式 5: 正向运动学与渲染

$$
q_{t+k} = \phi(q_t, a_{t:t+k-1}), \quad M_{t+k} = K_{\text{URDF}}(q_{t+k}), \quad C^m_{t+k,v} = R(M_{t+k}; K_v, T^v_{t+k})
$$

**含义**: 三步管线——递推关节状态 → 正向运动学得到 mesh → 渲染为运动图像。

**符号说明**:
- $\phi$：关节积分函数
- $K_{\text{URDF}}$：基于 URDF 的正向运动学函数
- $R$：可微渲染器
- $K_v, T^v_{t+k}$：视角 $v$ 的相机内参和外参

---

## 关键图表

### Figure 1: 整体框架

![iMaC Pipeline](https://arxiv.org/html/2606.09813v1/x1.png)

**说明**: iMaC 整体流水线。给定参考观测和未来动作，系统分两路生成控制信号：（1）通过 URDF + 正向运动学渲染[[运动图像]]；（2）通过场景点云与机器人未来位置计算双流[[接触图像]]。两路控制注入 WAN2.2 IT2V 世界模型，实现长时视频 chunk-wise 滚动生成，最终用于策略评估。

### Figure 2: 策略评估相关性

![Policy Evaluation Correlation](https://arxiv.org/html/2606.09813v1/x2.png)

**说明**: World model 预测成功率与真实世界策略成功率之间的 Pearson 相关系数分析。每个子图对应一个操作任务，评估 $\pi_{0.5}$ 和 GigaBrain-0.5 各三个 checkpoint。6/8 的任务相关系数 r=0.833~0.956，展现强正相关；Task 3（堆叠箱耳）和 Task 5（扫垃圾）因高度信息缺失导致相关性较低（r=0.678, 0.428）。

### Figure 3: 消融实验可视化

![Ablation Visualization](https://arxiv.org/html/2606.09813v1/x3.png)

**说明**: 接触图像/运动图像消融和深度来源对比可视化。移除运动图像：夹爪重复尝试动作但无法精确抓取；移除接触图像：缺乏接触感知引导，夹爪无法抓取布料；使用预测深度替代 Depth Anything 3：提供更一致的接触几何，生成质量更高。

### Figure 4: 8 个真实操作任务

![Task Visualization](https://arxiv.org/html/2606.09813v1/x4.png)

**说明**: 用于策略评估的 8 个真实操作任务可视化，每个任务展示初始帧和成功 rollout 的最终帧：
1. 把香蕉放入篮子
2. 把绿碗放入粉色盘子
3. 堆叠箱耳
4. 打开箱盖并倒薯片
5. 把垃圾扫进垃圾铲
6. 折叠衬衫
7. 撕开并粘贴胶带
8. 把熊猫玩偶放入粉色盘子

### Figure 5: 长时 Rollout 可视化（主要）

![Long-horizon Rollouts](https://arxiv.org/html/2606.09813v1/x5.png)

**说明**: 长时 iMaC rollout 结果，每个任务展示生成的 RGB 和深度帧，以及对应的运动图像和双流接触图像控制信号。

### Figure 6: 额外 Rollout 可视化

![Additional Rollouts](https://arxiv.org/html/2606.09813v1/x6.png)

**说明**: 补充 rollout 可视化，与 Figure 5 相同的布局，展示不同任务场景下生成视频与控制条件的对应关系。

### Figure 7: 失败案例分析

![Failure Cases](https://arxiv.org/html/2606.09813v1/x7.png)

**说明**: 缺失关键观测导致的失败案例可视化。即使生成的视频看起来合理，当所有可用视角都缺少决定任务成败的物理关系时（如 Task 3 中箱耳的高度关系），world model 会错误预测场景演化。

### Table 1: 主要结果对比

| Method | MSE ↓ | FID ↓ | PSNR ↑ | SSIM ↑ | FVD ↓ |
|--------|-------|-------|--------|--------|-------|
| Ctrl-World | 0.030±0.012 | 48.64±10.68 | 16.22±1.74 | 0.730±0.037 | 591.47±160.30 |
| ABot-PhysWorld | 0.041±0.017 | 74.23±22.50 | 14.41±1.62 | 0.630±0.071 | 642.98±105.27 |
| iMaC w/o contact | 0.028±0.009 | 38.81±9.89 | 16.34±1.39 | 0.735±0.039 | 523.94±156.84 |
| **iMaC** | **0.028±0.010** | **36.96±9.16** | **16.39±1.41** | **0.735±0.037** | **489.51±92.65** |

**说明**: iMaC 在所有指标上均优于基线。接触图像的引入（iMaC vs. iMaC w/o contact）在 FID（38.81→36.96）和 FVD（523.94→489.51）上有显著提升，说明接触感知对视频质量的实质性贡献。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| CVPR 2026 WorldModel Track Dataset | 8 个真实操作任务 | 多视角 RGB（头部+双腕）+ 深度，含策略 rollout 数据 | 训练与评估 |

### 实现细节

- **Backbone**: [[WAN2.2]] 图像到视频 [[Diffusion Transformer]]，多视角观测拼接为单图 mosaic 作为参考帧
- **训练策略**: 两阶段——Stage 1 在 8 任务共享模型上训练运动图像控制；Stage 2 针对各任务引入接触图像微调
- **Rollout Warmup**: 前 40 epochs 使用真实参考帧，之后切换为 one-step clean estimate 参考帧
- **深度估计**: 使用 [[Depth Anything 3]] 进行多视角深度估计
- **策略评估**: 对比 [[π0.5]] 和 [[GigaBrain-0.5]] 各三个 checkpoint，在匹配初始配置下评估

### 可视化结果

- 运动图像清晰呈现机器人关节运动轨迹，接触图像在夹爪接近目标时距离值显著下降
- 长时生成（分钟级）在 training-time rollout 策略下保持稳定，无明显误差积累
- 失败案例集中于高度信息缺失的任务（Task 3, 5），单纯增加 RGB 视角无法解决

---

## 批判性思考

### 优点
1. **范式新颖**：将动作表示从低维向量迁移到高维图像，与视频 backbone 天然对齐，是 embodied world model 领域的重要范式转变
2. **接触感知显式化**：双流接触图像设计优雅，将隐式几何关系显式化为可视化控制信号
3. **实用性强**：policy evaluation 框架在真实世界 8 个任务上展现了强相关性，具有实际应用价值

### 局限性
1. **深度依赖**：依赖 [[Depth Anything 3]] 的深度估计，厘米级误差会影响接触时机和碰撞定位精度
2. **观测覆盖局限**：无法推断所有可用视角都未覆盖的物理关系（如高度），Tasks 3/5 的低相关性暴露了这一系统性弱点
3. **URDF 依赖**：运动图像生成需要精确的机器人 URDF 模型，对无模型或形变机器人适用性受限

### 潜在改进方向
1. 引入触觉传感器或力矩信息补充高度等缺失的物理关系
2. 使用更高质量的深度传感器（如结构光、ToF）或操作专用深度模型替代通用深度估计
3. 探索无需 URDF 的运动图像生成方式，提升对异构机器人的适用性

### 可复现性评估
- [ ] 代码开源（GitHub 仓库存在：https://github.com/imac-wm/iMac，但处于 CoRL 2026 匿名投稿阶段）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（论文中详细描述了两阶段训练流程和 warmup 策略）
- [x] 数据集可获取（HuggingFace: open-gigaai/CVPR-2026-WorldModel-Track-Dataset）

---

## 关联笔记

### 基于
- [[WAN2.2]]: 视频生成 backbone（图像到视频 DiT）
- [[Flow Matching]]: 训练目标，流匹配损失
- [[Depth Anything 3]]: 多视角深度估计，用于构建接触图像的场景点云

### 对比
- [[Ctrl-World]]: 可控生成世界模型，使用向量动作条件（Guo et al. 2025）
- [[ABot-PhysWorld]]: 物理世界模型基线，向量动作条件（低于 iMaC）

### 方法相关
- [[URDF]]: 机器人运动学模型，用于渲染运动图像
- [[正向运动学]]: 关节角到 mesh 的映射，运动图像生成核心
- [[Contact Images]]: 本文提出的接触图像双流控制信号
- [[Motion Images]]: 本文提出的运动图像控制信号
- [[Exposure Bias]]: Training-time Rollout 所解决的核心问题
- [[Action Chunking]]: 动作分块预测策略

### 策略相关
- [[π0.5]]: 评估的策略之一（Physical Intelligence）
- [[GigaBrain-0.5]]: 评估的策略之一（GigaAI）

---

## 速查卡片

> [!summary] iMaC (2026)
> - **核心**: 将机器人动作转换为运动图像（URDF渲染）和接触图像（几何距离场），替代低维向量作为 world model 的视觉控制条件
> - **方法**: Motion Images（正向运动学渲染）+ Contact Images（双流距离图）+ Training-time Rollout
> - **结果**: 8任务上 FVD 489.5（vs. Ctrl-World 591.5），策略评估相关性 r=0.833~0.956
> - **代码**: https://github.com/imac-wm/iMac

---

*笔记创建时间: 2026-06-15*
