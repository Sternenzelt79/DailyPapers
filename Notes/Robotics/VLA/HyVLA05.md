---
title: "Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack"
method_name: "HyVLA-0.5"
authors: [He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, Zhengyou Zhang]
year: 2026
venue: arXiv
tags: [vla, flow-matching, embodied-ai, robot-manipulation, reinforcement-learning, cross-embodiment, imitation-learning]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2606.14409
created: 2026-06-15
---

# 论文笔记：Hy-Embodied-0.5-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent Robotics X × Tencent Hy Team |
| 日期 | June 2026 |
| 项目主页 | [tairos.tencent.com](https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla) |
| 对比基线 | [[π0]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.14409) / [Code](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA) / [Model](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI) |

---

## 一句话总结

> HyVLA-0.5 是一个覆盖数据采集、模型设计、预训练、SFT、RL后训练到真实部署的完整端到端VLA系统，在RoboTwin 2.0仿真中达到90.9%成功率，经FlowPRO后训练在真实机器人上达到99%成功率。

---

## 核心贡献

1. **Hy-UMI-10K 数据集**: 10,000+小时自我中心演示数据，含定制指尖UMI设备（亚毫米精度光学动捕 + 6轴力矩传感器），覆盖70种任务
2. **Hy-Embodied-0.5-MoT骨干 + Flow-Matching动作专家**: 4B参数[[混合专家模型|Mixture-of-Transformers]]骨干搭配[[条件流匹配|Flow Matching]]动作预测，支持高频连续动作输出
3. **FlowPRO RL后训练**: 无奖励函数的[[偏好优化|Preference Optimization]]算法，利用干预-回滚数据构造偏好对，配合近端正则化防止奖励劫持
4. **跨本体部署**: 端效器帧增量动作表示（delta-chunk）解耦策略与机器人运动学，无需目标机器人遥操作即可跨本体迁移
5. **异步推理 + Bézier平滑**: 生产者-消费者运行时使推理与伺服执行并行，Bézier曲线保证动作块之间C¹连续过渡

---

## 问题背景

### 要解决的问题

现有VLA系统通常只关注模型架构改进，忽略了从实验室到真实部署所涉及的完整技术栈。数据稀缺、架构瓶颈、部署约束三大挑战相互耦合，阻碍了VLA系统的实际落地。

### 现有方法的局限

- **数据收集**: 传统遥操作系统精度有限，难以获取亚毫米级精度的力觉信号
- **架构设计**: 多数VLM骨干对视觉和语言模态共享参数，计算效率低；内存编码器往往引入大量额外参数
- **部署约束**: 推理延迟与伺服频率不匹配，动作块切换产生抖动，跨机器人迁移需要重新采集数据

### 本文的动机

作者认为端到端机器人学习系统需要对数据、模型、训练策略和部署方案进行协同设计（co-design），而非单独优化某一环节。

---

## 方法详解

### 系统架构概览

![[HyVLA05_fig1_overview.jpeg]]

**说明**: HyVLA-0.5 的整体架构，展示从定制UMI数据采集装置到Hy-Embodied-0.5-MoT骨干、流匹配动作专家，再到实际机器人部署的完整流水线。

HyVLA-0.5 采用 **[[视觉语言动作模型|VLA]]** 架构，三大核心组件：

- **输入**: 语言指令 $\ell$ + 多视角多帧观测 $\mathcal{I}_t$ + 本体感知状态 $\mathbf{s}_t$
- **骨干**: [[混合专家模型|Hy-Embodied-0.5-MoT]]（4B参数，[[Mixture of Transformers]]）
- **动作专家**: [[条件流匹配|Flow Matching]]双塔结构，生成连续动作块
- **记忆编码器**: 无参数时序-空间注意力，处理K帧历史
- **输出**: [[Action Chunking|动作块]] $\mathbf{A}_t = (a_t, a_{t+1}, ..., a_{t+H-1})$，$H=50$，10Hz
- **总参数**: 约4B（骨干）+ 轻量动作专家

### 核心模块

#### 模块1：Hy-Embodied-0.5-MoT 骨干

![[HyVLA05_fig2_architecture.jpeg]]

**说明**: MoT架构细节，每4层插入一个时序注意力块，视觉流与文本流采用非共享QKV和FFN参数实现模态自适应计算。

**设计动机**: 视觉和语言信息具有不同的统计特性，共享参数会制约模型性能，利用[[Mixture of Transformers]]实现模态自适应计算。

**具体实现**:
- 使用原生分辨率[[Vision Transformer|ViT]]进行视觉编码
- 视觉流与语言流采用**非共享QKV和FFN参数**，每模态独立计算
- 每4层插入一个[[时序注意力|Temporal Attention]]块，用于跨帧信息聚合
- [[块因果注意力|Block-wise Causal Attention]]策略：VLM与动作专家双塔之间的信息流

#### 模块2：双塔流匹配动作专家

**设计动机**: 通过[[条件流匹配|Conditional Flow Matching]]在噪声分布与目标动作分布之间构建连续变换路径，实现高精度连续动作预测。

**具体实现**:
- **VLM塔**：处理视觉-语言条件信息
- **动作专家塔**：接受含噪动作输入 $\mathbf{A}_t^\tau$，输出速度场
- 推理时通过[[ODE求解器]]从高斯噪声迭代去噪生成动作

#### 模块3：紧凑记忆编码器

**设计动机**: 历史帧信息对精细操作任务至关重要，但全参数时序融合开销过大。

**具体实现**:
- **参数零增加**：不引入新参数，复用预训练权重
- **因子化时序-空间注意力**（Factorized Temporal-Spatial Attention）：先时序后空间分别注意
- 处理K帧多视角RGB历史（SFT阶段K=6）
- 在4个任务的消融中，去除后性能从90.9%降至88.8%（clean）/ 90.1%降至88.6%（randomized）

---

## 关键公式

### 公式1: [[条件动作预测|问题形式化]]

$$
o_t = (\mathcal{I}_t, \ell, \mathbf{s}_t), \quad \mathcal{I}_t = \{I^v_{t-k}\}_{v=1:n}^{k=0:K-1}
$$

**含义**: 将时间步 $t$ 的观测定义为多视角历史帧、语言指令、本体感知状态的三元组。

**符号说明**:
- $\mathcal{I}_t$：多视角多帧图像集合
- $v = 1:n$：$n$ 个摄像头视角
- $k = 0:K-1$：$K$ 帧历史（pre-training K=1，SFT K=6）
- $\ell$：语言任务指令
- $\mathbf{s}_t$：本体感知状态（关节角度等）

### 公式2: [[流匹配|Flow Matching 目标函数]]

$$
\mathcal{L}_{fm}(\theta) = \mathbb{E}_{\tau, \mathbf{A}_t, \varepsilon}\left[\left\|v_\theta(\mathbf{A}_t^\tau, o_t) - (\varepsilon - \mathbf{A}_t)\right\|_2^2\right]
$$

其中插值路径：

$$
\mathbf{A}_t^\tau = (1-\tau)\mathbf{A}_t + \tau\varepsilon, \quad \tau \sim \mathcal{U}(0, 1), \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 训练速度场网络 $v_\theta$ 使其匹配从噪声 $\varepsilon$ 到目标动作 $\mathbf{A}_t$ 的线性插值路径上的方向。

**符号说明**:
- $v_\theta(\cdot)$：待学习速度场网络
- $\mathbf{A}_t^\tau$：时间步 $\tau$ 处的插值动作
- $\tau \sim \mathcal{U}(0,1)$：均匀采样流时间
- $\varepsilon \sim \mathcal{N}(0, I)$：标准高斯噪声

### 公式3: [[FlowPRO|RPRO 偏好优化损失]]

$$
\mathcal{L}_{PRO}(\theta) = -\mathbb{E}\left[\log\sigma\left(r_\theta(s, a^w) - r_\theta(s, a^l)\right) + \sum_{a \in \{a^w, a^l\}} \frac{1}{2}\left[\log\sigma(r_\theta(s,a)) + \log\sigma(-r_\theta(s,a))\right]\right]
$$

隐式奖励定义：

$$
r_\theta(s, a) = \frac{\beta}{2}\left(\ell_{ref}(s, a) - \ell_\theta(s, a)\right)
$$

**含义**: 对比好坏动作对的流匹配负对数似然差，并加入锚定正则项防止奖励劫持。隐式奖励由当前策略与参考策略的负对数似然差定义（无需单独训练奖励模型）。

**符号说明**:
- $r_\theta(s, a)$：基于策略对数似然差的隐式奖励
- $a^w, a^l$：好（winning）和差（losing）动作，来自干预数据
- $\beta$：KL散度系数，控制与参考策略的偏离程度
- $\ell_{ref}(s,a)$：参考（SFT）策略的负对数似然
- $\ell_\theta(s,a)$：当前策略的负对数似然
- $\sigma(\cdot)$：Sigmoid函数

### 公式4: [[坐标变换|固定底座机械臂的动作变换]]

$$
{}^W\mathbf{T}_{G_{t+k}} = {}^W\mathbf{T}_{G_t} \cdot {}^{G_t}\mathbf{T}_{G_{t+k}}
$$

**含义**: 将增量动作块（相对末端执行器帧的位移）转换为世界坐标系下的绝对位姿。

### 公式5: [[坐标变换|浮动底座人形机器人的动作变换]]

$$
{}^C\mathbf{T}_{G_{t+k}} = \left({}^W\mathbf{T}_C\right)^{-1} \cdot {}^W\mathbf{T}_{G_t} \cdot {}^{G_t}\mathbf{T}_{G_{t+k}}
$$

**含义**: 对浮动底座机器人，先通过相机坐标系 $C$ 消除底座运动影响，再转换为末端执行器绝对位姿。

### 公式6: [[Bézier曲线|三次Bézier动作平滑]]

控制点：

$$
P_0 = h_0, \quad P_1 = P_0 + \lambda\hat{d}_{hist}, \quad P_2 = P_3 - \lambda\hat{d}_{fut}, \quad P_3 = f_c
$$

曲线方程：

$$
\mathbf{B}(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t)t^2 P_2 + t^3 P_3, \quad t \in [0, 1]
$$

**含义**: 用三次Bézier曲线在两个动作块的连接点处构造C¹连续轨迹，消除切换时的速度不连续抖动。

**符号说明**:
- $h_0$：当前块末尾（历史参考点）
- $f_c$：下一块的连接点（connection point）
- $\hat{d}_{hist}, \hat{d}_{fut}$：历史和未来轨迹在连接点处的切线方向
- $\lambda$：切线长度控制系数

---

## 关键图表

### Figure 3: UMI 数据采集装置

![[HyVLA05_fig3_umi_device.jpeg]]

**说明**: 定制化指尖UMI装置，集成外部光学动捕系统（亚毫米级追踪精度）、6轴力矩传感器（分辨率0.01N）、RGB-D相机。基于Changingtek CTAG2F90设计的人体工学夹爪，支持指尖附着操作。

### Figure 4: Hy-UMI-10K 数据集分布

![[HyVLA05_fig4_dataset_dist.png]]

**说明**: Hy-UMI-10K数据集的任务家族、对象类别和时长分布。六大场景类别及其占比：洗衣房（28.5%）、厨房（19.2%）、个人护理（13.8%）、灵巧/工具使用（10.4%）、收纳整理（10.0%）、清洁（5.7%）。

### Figure 5: FlowPRO 数据流水线

![[HyVLA05_fig5_flowpro_pipeline.png]]

**说明**: FlowPRO RL后训练数据流水线。操作员在策略Rollout过程中干预并回滚到前序状态，记录执行片段为负样本轨迹，记录纠正遥操作为正样本轨迹。再通过平滑插值构造稠密的每状态偏好对 $(s, a^w, a^l)$。

### Figure 7: 异步执行时序

![[HyVLA05_fig7_async_exec.png]]

**说明**: 生产者-消费者异步运行时架构。VLM推理（生产者）与伺服执行（消费者）并行进行，通过线程安全动作缓冲区解耦，使高延迟推理不阻塞低延迟伺服控制。

### Figure 8: Bézier 轨迹对比

![[HyVLA05_fig8_bezier.png]]

**说明**: 原始动作块（橙色，切换处有明显不连续跳变）与Bézier平滑后轨迹（蓝色，C¹连续）的对比。Bézier平滑有效消除了动作块连接处的速度不连续性。

### Figure 9: 真实机器人评测结果

![[HyVLA05_fig9_real_robot.png]]

**说明**: Dobot X-Trainer上六项双臂任务的成功率快照与逐任务成功率。任务包括：插入瓶子、折叠存放眼镜、摆台、拉链笔袋等精细操作。

### Figure 10: 力引导对象区分（Unitree G1）

![[HyVLA05_fig10_force.jpeg]]

**说明**: 在Unitree G1人形机器人上，通过TCN编码50步力矩信号窗口，成功区分轻重物体并选择较轻的一个。展示了力觉模态对VLA系统的扩展能力。

### Figure 12: FlowPRO 每轮迭代成功率

![[HyVLA05_fig12_flowpro_results.jpeg]]

**说明**: 四个真实机器人任务上RPRO、DAgger、π0.6*的逐迭代成功率对比。RPRO在瓶子/瓶盖/USB任务上3轮内收敛至99%，拉链任务达94%，且完成时间比DAgger快约40%。

### Table 1: RoboTwin 2.0 仿真评测

| 方法 | Clean | Randomized |
|------|-------|-----------|
| π₀ | 65.9% | 58.4% |
| π₀.₅ | 82.7% | 76.8% |
| Qwen-VLA | 86.1% | 87.2% |
| starVLA | 88.2% | 88.3% |
| JoyAI-RA | 90.5% | 89.3% |
| **HyVLA-0.5** | **90.9%** | **90.1%** |

**关键发现**: HyVLA-0.5在干净和随机化环境下均达到SOTA，超过π₀.₅约8个百分点。

### Table 2: 消融实验（RoboTwin 2.0）

| 配置 | Clean | Randomized | 说明 |
|------|-------|-----------|------|
| **HyVLA-0.5（完整）** | **90.9%** | **90.1%** | 含记忆编码器 + UMI预训练 |
| w/o 记忆编码器 | 88.8% | 88.6% | 去除时序历史 |
| w/o 记忆编码器 + UMI预训练 | 88.1% | 87.9% | 两者均去除 |

**关键发现**: 记忆编码器贡献2.1%（clean），UMI预训练额外贡献0.7%，两者均有正向效果。

### Table 3: FlowPRO vs. 基线（真实机器人）

| 任务 | DAgger | π₀.₆* | **RPRO** |
|------|--------|-------|---------|
| 瓶子插入 | 93±2.1% | 95±1.5% | **99±0.6%** |
| 瓶盖装配 | 88±1.8% | 95±1.2% | **99±0.7%** |
| USB插入 | 86±2.4% | 95±1.4% | **98±0.9%** |
| 拉链笔袋 | 83±2.0% | 89±1.6% | **94±1.1%** |

**完成时间（秒）**:

| 任务 | DAgger | π₀.₆* | RPRO |
|------|--------|-------|------|
| 瓶子插入 | 27s | 24s | **16s** |
| 瓶盖装配 | 29s | 27s | **21s** |

**关键发现**: RPRO在所有任务上均优于DAgger和π₀.₆*，且完成时间最短（瓶子任务快40%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Hy-UMI-10K | 10,000+小时 / 1.1M+集 | 光学动捕亚毫米精度，含力觉 | 预训练 |
| Dobot X-Trainer SFT | 4任务×300演示，18小时 | 目标机器人遥操作 | Track A SFT |
| JAKA K1 SFT | 1任务，300 UMI演示，1.2小时 | 无目标机器人数据 | Track B跨本体 |
| Astribot S1 SFT | 1任务，200 UMI演示，1.5小时 | 人形机器人跨本体 | Track B跨本体 |
| RoboTwin 2.0 | — | 仿真benchmark | 评测 |

### 实现细节

**预训练配置**:
- **优化器**: AdamW，学习率 $5 \times 10^{-5}$（线性warmup 1K步，decay至160K，hold至200K）
- **Batch Size**: 1024
- **训练步数**: 200K
- **历史帧数**: K=1，动作块H=50，采样频率10Hz
- **摄像头**: 3视角，224×320分辨率

**SFT配置**:
- **学习率**: $2.5 \times 10^{-5}$（decay至40K）
- **Batch Size**: 32
- **训练步数**: 60K
- **历史帧数**: K=6，动作频率50Hz，动作块H=50

**动作表示**:
- 20维双臂增量动作块（每臂10维）
- 每个末端执行器：3D平移 + 6D旋转（SO(3)表示）+ 1D夹爪

**FlowPRO批次混合**:
- Round 1: 80%当前偏好对 / 20% SFT数据
- Round k≥2: 70%当前 / 15%历史轮次 / 15% SFT数据

---

## 批判性思考

### 优点
1. **全栈系统**: 少有论文覆盖从数据采集到RL后训练的完整链路，工程价值极高
2. **FlowPRO创新**: 无奖励函数、无Critic模型的RL方案，将干预数据直接转化为偏好信号，采集效率高
3. **跨本体迁移**: 仅凭UMI数据（无目标机器人遥操数据）完成人形机器人部署，验证了delta-chunk表示的通用性
4. **力觉集成**: 将力矩传感器作为额外模态接入，展示了系统的可扩展性

### 局限性
1. **UMI设备门槛**: 定制光学动捕系统成本较高，难以在一般实验室中复现
2. **FlowPRO数据依赖人工干预**: 干预-回滚流程仍需操作员实时介入，尚不完全自动化
3. **仿真评测单一**: 仅在RoboTwin 2.0上评测，未与更多仿真baseline对比
4. **真实任务种类有限**: Track A/B各仅4-6种任务，泛化能力有待更广泛验证

### 潜在改进方向
1. 自动化干预检测（不需要人工实时介入）以进一步降低FlowPRO数据采集成本
2. 将力觉模态融入主干预训练（而非仅作后期扩展）
3. 探索更轻量的跨本体迁移方案（减少UMI演示数据需求）

### 可复现性评估
- [x] 代码开源（GitHub）
- [x] 预训练模型（HuggingFace）
- [x] 训练细节完整（论文附录）
- [x] 数据集可获取（HuggingFace）

---

## 关联笔记

### 基于
- [[π0]]: VLA基线，本文重要对比方法
- [[π0.5]]: 更新版VLA基线
- [[Diffusion Policy]]: 流匹配动作专家的前置工作
- [[Flow Matching]]: 动作专家核心技术基础

### 对比
- [[π0]]: RoboTwin仿真中对比，HyVLA-0.5高出约25个百分点
- [[DAgger]]: FlowPRO RL后训练的对比基线
- [[Qwen-VLA]]: 仿真benchmark对比方法
- [[starVLA]]: 仿真benchmark对比方法

### 方法相关
- [[条件流匹配|Flow Matching]]: 动作专家核心方法
- [[混合专家模型|Mixture of Transformers]]: 骨干架构
- [[Action Chunking]]: 动作块预测策略
- [[偏好优化|DPO/PRO]]: FlowPRO的理论基础
- [[Bézier曲线]]: 动作平滑方法

### 硬件/数据相关
- [[UMI|Universal Manipulation Interface]]: 数据采集装置设计基础
- [[Dobot X-Trainer]]: Track A部署平台
- [[Astribot S1]]: 人形机器人部署平台
- [[Unitree G1]]: 力觉实验平台

---

## 速查卡片

> [!summary] HyVLA-0.5
> - **核心**: 完整VLA系统全栈：数据→预训练→SFT→FlowPRO RL→部署
> - **方法**: MoT骨干 + Flow-Matching动作专家 + 无奖励RL后训练 + Bézier平滑异步部署
> - **结果**: 仿真90.9% SOTA；真实机器人经RL后训练达99%成功率，比DAgger快40%
> - **代码**: [github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)

---

*笔记创建时间: 2026-06-15*
