---
title: "NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation"
method_name: "NavWAM"
authors: [Daichi Azuma, Taiki Miyanishi, Koya Sakamoto, Shuhei Kurita, Yaonan Zhu, Petr Khrapchenkov, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo]
year: 2026
venue: arXiv
tags: [visual-navigation, world-action-model, diffusion-policy, goal-conditioned, robot-navigation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.13494
created: 2026-06-12
---

# 论文笔记：NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RIKEN AIP, University of Tokyo |
| 日期 | June 2026 |
| 项目主页 | [dachii-azm.github.io/navwam](https://dachii-azm.github.io/navwam/) |
| 对比基线 | [[ViNT]], [[GNM]], [[NoMaD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13494) / Code: N/A |

---

## 一句话总结

> NavWAM 将未来视觉预测、目标进度值和动作块统一到一个[[扩散策略|扩散-Transformer 策略]]的共享潜在序列中，无需外部规划器即可实现闭环导航。

---

## 核心贡献

1. **Navigation World Action Model（导航世界行动模型）**: 将直接策略与导航世界模型统一，在单次去噪链中联合预测动作、未来观测和目标进度值
2. **九帧潜在画布（Latent Canvas）**: 基于 [[Cosmos Predict2]] 骨干，将机器人状态、目标图像、当前观测、动作块、未来状态、未来观测和价值标量编排为结构化的九帧序列
3. **高效闭环部署**: 不依赖 CEM 等外部搜索方法，单次去噪即输出可执行动作，在真实机器人上达到 79.2% 成功率

---

## 问题背景

### 要解决的问题

目标条件视觉导航（goal-conditioned visual navigation）要求机器人根据目标图像，在复杂室内环境中规划并执行导航轨迹。核心挑战在于：机器人需要预判自身运动如何改变未来的自我中心观测，以及该变化是否使其更接近目标。

### 现有方法的局限

- **直接策略**（如 [[ViNT]]、[[GNM]]）：高效，但缺乏显式视觉预见（visual foresight），难以对复杂场景进行长程规划
- **导航世界模型**（Navigation World Models, NWM）：显式预测未来观测，但依赖外部规划器（如 [[CEM]]）在候选动作上搜索——这造成部署瓶颈：闭环行为依赖外部规划选择，而非学到的世界模型本身

### 本文的动机

将世界模型预测与动作生成解耦是多余的开销。如果一个策略能在同一次前向传播中同时完成"预测未来会看到什么"和"决定现在做什么"，就能兼具视觉预见和闭环效率。

---

## 方法详解

### 模型架构

NavWAM 采用 **[[Diffusion Transformer|扩散-Transformer 策略]]** 架构，基于 [[Cosmos Predict2]]（2B 参数）骨干：

- **输入**: 当前自我中心观测 $o_t$ + 目标图像 $g$ + 机器人状态 $s_t$
- **Backbone**: [[Cosmos Predict2]]（[[因果 VAE]] 编解码 + [[Diffusion Transformer|DiT]] 去噪）
- **核心机制**: [[Latent Canvas|九帧潜在画布]] — 将所有模态编排为视频格式的共享潜在序列
- **输出**: 动作块 $a_{t:t+H-1}$（可选：未来观测 $o_{t+H}$、进度值 $v_{t+H}$）
- **总参数**: ~2B

### 核心模块

#### 模块 1: 九帧潜在画布（Latent Canvas）

**设计动机**: 将导航所需的全部信号（感知、目标、状态、预测、动作、价值）统一编码到视频扩散模型的时序帧格式中，复用预训练的视频生成能力。

**帧结构**:

| 帧编号 | 内容 | 角色 |
|--------|------|------|
| 0 | 空白（VAE padding） | 条件帧（Observed） |
| 1 | 机器人状态 $[x, y, \psi]$ | 条件帧 |
| 2 | 目标图像 $g$ / 前一观测 | 条件帧 |
| 3 | 当前自我中心 RGB $o_t$ | 条件帧 |
| 4 | 动作块 $a_{t:t+H-1}$ | **生成帧** |
| 5 | 未来状态 $s_{t+H}$ | **生成帧** |
| 6 | 未来观测 $o_{t+H-1}$ | **生成帧** |
| 7 | 未来观测 $o_{t+H}$ | **生成帧** |
| 8 | 目标进度标量 $v_{t+H}$ | **生成帧** |

**具体实现**:
- 数值型帧（状态、动作、价值）通过平铺像素格式编码为图像，经[[因果 VAE]]压缩
- 每种模态均使用独立噪声掩码（noise mask），条件帧在去噪过程中固定不变

#### 模块 2: 三种条件化训练模式

**设计动机**: 使同一模型能在推理时切换"纯策略"和"世界模型辅助"两种模式，同时学习进度价值以提升轨迹质量。

| 模式 | 概率 | 条件帧 | 预测帧 | 用途 |
|------|------|--------|--------|------|
| **Policy Mode** | 50% | 0–3 | 4–8 | 标准闭环策略 |
| **World-Model Mode** | 25% | 0–4（含动作） | 5–8 | 预测未来观测 |
| **Value Mode** | 25% | 0–7 | 价值标量（帧8） | 估计目标进度 |

#### 模块 3: 动作上权重（Action Upweighting）

**设计动机**: 高维图像损失会淹没低维动作信号（动作 $\in \mathbb{R}^3$ vs. 图像 $\in \mathbb{R}^{H \times W \times 3}$），导致策略退化为图像预测器。

**具体实现**: 对动作帧（帧 4）的去噪损失施加 $\lambda = 5$ 的乘法权重。

---

## 关键公式

### 公式 1: [[直接策略|直接导航策略]]

$$
\pi_\theta(a_{t:t+H-1} \mid o_t, g)
$$

**含义**: 传统直接策略，仅从当前观测和目标图像预测动作块，无显式未来预测。

**符号说明**:
- $a_{t:t+H-1}$: 从时刻 $t$ 起的 $H$ 步动作块（locally normalized waypoints）
- $o_t$: 当前自我中心 RGB 观测
- $g$: 目标图像

---

### 公式 2: [[Navigation World Model|导航世界模型]]

$$
p_\theta(o_{t+H} \mid o_t, a_{t:t+H-1})
$$

**含义**: 导航世界模型，给定当前观测和动作序列，预测未来观测——需要外部规划器遍历候选动作。

**符号说明**:
- $o_{t+H}$: 预测的未来自我中心观测
- $a_{t:t+H-1}$: 输入的候选动作序列

---

### 公式 3: [[World-Action Model|NavWAM 联合生成分布]]

$$
p_\theta\!\left(a_{t:t+H-1},\, s_{t+H},\, o_{t+H-1:t+H},\, v_{t+H} \mid o_t, g\right)
$$

**含义**: NavWAM 的核心公式，将动作块、未来状态、未来观测序列和目标进度值统一建模为一个条件联合分布，单次前向传播完成所有预测。

**符号说明**:
- $s_{t+H}$: 预测的未来机器人状态 $[x, y, \psi]$
- $o_{t+H-1:t+H}$: 两帧未来自我中心观测
- $v_{t+H}$: 目标进度标量值（goal-progress value）

---

### 公式 4: [[扩散策略|扩散去噪训练目标]]

$$
\mathcal{L}_\text{diff} = \mathbb{E}_{\sigma, \epsilon}\!\left[w(\sigma)\,\left\|x_0 - F_\theta(x_\sigma, \sigma, c)\right\|_2^2\right]
$$

**含义**: 标准扩散去噪损失，训练去噪网络 $F_\theta$ 从噪声版本 $x_\sigma$ 中预测干净帧 $x_0$；动作帧额外乘以 $\lambda=5$。

**符号说明**:
- $\sigma$: 噪声强度（noise level）
- $w(\sigma)$: 噪声水平相关权重函数
- $x_0$: 干净目标帧（动作/状态/观测/价值）
- $x_\sigma = x_0 + \sigma\epsilon$: 加噪版本
- $c$: 条件信号（条件帧）
- $\lambda = 5$: 动作帧上权重系数

---

### 公式 5: [[Goal-Progress Value|目标进度值]]

$$
v_{t+H} = \text{clip}\!\left(1 - \frac{\left\|p_\text{end} - p_t\right\|_2}{d_\text{max}},\; 0,\; 1\right)
$$

**含义**: 将机器人到达终点的归一化距离转化为 $[0, 1]$ 的标量进度值，距终点越近值越大。

**符号说明**:
- $p_\text{end}$: 轨迹终点位置
- $p_t$: 机器人当前位置
- $d_\text{max}$: 最大归一化距离（轨迹全长）

---

### 公式 6: 机器人状态归一化

$$
s_t = \left[\frac{x_t}{100},\; \frac{y_t}{100},\; \frac{\psi_t}{\pi}\right] \in \mathbb{R}^3
$$

**含义**: 将机器人 2D 位置和航向角归一化到像素值域，以便编码为图像帧输入 VAE。

**符号说明**:
- $x_t, y_t$: 机器人在地图坐标系中的位置（米）
- $\psi_t$: 航向角（弧度）

---

### 公式 7: 动作块格式

$$
a_i = (\Delta x_i,\, \Delta y_i,\, \Delta\psi_i) \in \mathbb{R}^3,\quad i = t, \ldots, t+H-1
$$

**含义**: 每个动作为相对于当前机器人坐标系的局部归一化路径点（waypoint），$H=4$。

**符号说明**:
- $\Delta x_i, \Delta y_i$: 局部坐标增量
- $\Delta\psi_i$: 航向角增量

---

## 关键图表

### Figure 1: 方法对比概览

![Figure 1](https://arxiv.org/html/2606.13494v1/x1.png)

**说明**: 左侧：传统 [[Navigation World Model|NWM]] 需要对候选动作分别预测未来观测，再由外部规划器（如 CEM）评分选最优动作；右侧：NavWAM 在单次去噪中直接输出可执行动作，同时预测未来观测和目标进度值。

---

### Figure 2: NavWAM 架构总览（九帧潜在画布）

![Figure 2](https://arxiv.org/html/2606.13494v1/x2.png)

**说明**: 展示[[Latent Canvas|九帧潜在画布]]的完整结构。帧 0–3 作为条件输入，[[Cosmos Predict2]] 的 DiT 骨干通过联合去噪在帧 4–8 上生成动作块、未来状态、未来观测和价值标量。

---

### Figure 3: 视觉一致性对比（Consistency Bar Chart）

![Figure 3](https://arxiv.org/html/2606.13494v1/x3.png)

**说明**: 定量对比 NavWAM 与 NWM baseline 在 Subject Consistency 指标上的表现。NavWAM 达到 0.635–0.668，NWM baseline 仅 0.524，说明联合训练保留了可解释的视觉预见能力。

---

### Figure 4: go_stanford 数据集定性预测结果

![Figure 4](https://arxiv.org/html/2606.13494v1/x4.png)

**说明**: 展示 NavWAM 在 go_stanford 测试集上生成的未来视图预测。预测与实际执行动作语义一致，验证了动作-观测联合建模的有效性。

---

### Figure 5: 真实机器人闭环执行轨迹

![Figure 5](https://arxiv.org/html/2606.13494v1/x5.png)

**说明**: Diablo 机器人在四种室内环境（办公室、储藏室、会议室、走廊）上的闭环导航轨迹。NavWAM 成功率 79.2%（19/24 episodes），显著超越 OmniVLA 的 58.3%。

---

### Figure 6: 闭环执行时的实时未来视图预测

![Figure 6](https://arxiv.org/html/2606.13494v1/x6.png)

**说明**: 机器人闭环执行过程中，NavWAM 实时预测的未来视图。预测图像与实际后续观测高度吻合，表明世界模型预测在真实部署中有效。

---

### Figure S1: Diablo 机器人硬件平台

![Figure S1](https://arxiv.org/html/2606.13494v1/x7.png)

**说明**: Direct Drive Tech Diablo 轮式机器人，配备 3D 打印支架，搭载机载传感器和计算单元。

---

### Table 1: go_stanford 数据集导航性能对比

| 方法 | ATE↓ | RPE↓ |
|------|------|------|
| Cosmos Predict2 | 0.455 | 0.109 |
| NWM | 0.453 | 0.107 |
| **NavWAM** | **0.324** | **0.099** |
| **NavWAM w/ FT** | **0.192** | **0.070** |

**说明**: NavWAM zero-shot 下 ATE 从 0.453 降至 0.324（-28.5%）；经 per-dataset 微调后进一步降至 0.192，RPE 降至 0.070，全面超越 NWM baseline。ATE = Average Trajectory Error，RPE = Relative Pose Error。

---

### Table 2: 监督头消融实验（sit 数据集）

| 监督头配置 | 推理模式 | ATE(h=4)↓ | ATE(h=8)↓ | RPE(h=4)↓ | RPE(h=8)↓ |
|-----------|----------|----------|----------|----------|----------|
| Image only | Planning | 0.326 | 0.569 | 0.133 | 0.135 |
| Image + Action + State | Policy | 0.107 | 0.287 | 0.054 | 0.098 |
| Image + Action + State + Value | Policy | **0.076** | **0.192** | **0.037** | **0.070** |

**关键发现**: ① 仅图像预测（Planning 模式）性能最差，证明视觉预见单独并不足够；② 加入动作和状态监督大幅提升轨迹精度；③ 加入价值监督进一步改善，验证目标进度标量的正向贡献。

---

### Table 3: 与直接策略对比（sit 数据集）

| 方法 | ATE(h=4)↓ | ATE(h=8)↓ | SR%(h=4)↑ | SR%(h=8)↑ |
|------|----------|----------|----------|----------|
| OmniVLA | 0.086 | 0.162 | 45.4 | 12.1 |
| **NavWAM** | **0.077** | **0.144** | **46.3** | **15.9** |

**说明**: NavWAM 在短程（h=4）和长程（h=8）指标上均优于 OmniVLA，尤其长程成功率提升 +3.8pp（15.9% vs 12.1%）。

---

### Table 4: 真实世界部署结果（Diablo 机器人，24 episodes）

| 方法 | 办公室 | 储藏室 | 会议室 | 走廊 | 总成功率 |
|------|--------|--------|--------|------|----------|
| NWM | 1/8 | 0/6 | 1/6 | 2/4 | 16.7% |
| OmniVLA | 4/8 | 4/6 | 3/6 | 3/4 | 58.3% |
| **NavWAM** | **6/8** | **6/6** | **4/6** | **3/4** | **79.2%** |

**说明**: NavWAM 在所有四种室内环境中均获得最高成功率，相比 OmniVLA 总体提升 +20.9pp，在储藏室环境达到 100% 成功率。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| HM3D（模拟器） | 802 场景，185,000 轨迹 | Habitat 仿真，多样室内场景 | Phase 1 预训练 |
| recon + sacson + scand | 14,207 轨迹 | 真实室外/室内导航数据 | Phase 2 联合微调 |
| go_stanford | 3,544 轨迹 | 斯坦福校园室内 | Phase 3 可选微调 |
| sit | 未指定 | 室内桌椅场景 | 直接策略对比测试 |
| 真实环境（Diablo） | 24 episodes × 4 环境 | 办公室/储藏室/会议室/走廊 | 真实部署评估 |

### 实现细节

- **Backbone**: [[Cosmos Predict2]]（2B 参数，Causal VAE + DiT）
- **优化器**: AdamW，余弦学习率调度
- **学习率**: $10^{-4}$
- **Batch Size**: 32（每卡 8）
- **精度**: bfloat16 混合精度
- **硬件**: 4 × NVIDIA RTX PRO 6000
- **训练流程**: 三阶段课程（模拟预训练 → 真实数据微调 → 可选数据集微调）
- **动作块长度**: Phase 1 $H=16$，Phase 2/3 $H=4$

### 可视化结果

- NavWAM 预测的未来视图在语义和结构上与实际执行后的真实观测高度一致
- 真实机器人上，NavWAM 能在转弯、穿门等复杂场景中准确预测前方视角变化

---

## 批判性思考

### 优点

1. **统一建模消除规划瓶颈**: 将 NWM 与直接策略合并，避免了 CEM 等外部搜索的计算开销，推理效率提升数个量级
2. **多任务监督协同增强**: 图像、动作、状态、价值的联合监督互相约束，消融实验证明每个头均有正向贡献
3. **真实部署验证充分**: 79.2% 的真实机器人成功率和多环境测试，证明了方法的实用性

### 局限性

1. **评估范围有限**: 仅在图像目标室内导航上验证，未测试语言目标导航、室外场景或动态障碍物环境
2. **真实实验规模较小**: 仅 24 episodes，样本量不足以支撑强统计置信度
3. **依赖大型预训练骨干**: 基于 2B 参数的 [[Cosmos Predict2]]，对计算资源要求较高，轻量化版本尚未探索

### 潜在改进方向

1. 扩展到语言条件导航（text-conditioned）和 ObjectNav 任务
2. 引入更多帧的历史观测提升遮挡和长程规划能力
3. 探索基于价值函数的树搜索，在推理时选择最优动作块

### 可复现性评估

- [ ] 代码开源（项目主页存在，但代码未公开）
- [ ] 预训练模型（暂未提供）
- [x] 训练细节完整（三阶段、数据集规模、超参数均有描述）
- [x] 数据集可获取（HM3D, recon, sacson, scand 均为公开数据集）

---

## 关联笔记

### 基于

- [[Cosmos Predict2]]: 2B 参数扩散-Transformer 视频生成骨干，NavWAM 在此基础上微调
- [[World-Action Model]]: NavWAM 是 WAM 范式在导航领域的实例化

### 对比

- [[ViNT]]: 通用视觉导航 Transformer，NavWAM 的直接策略对比基线之一
- [[GNM]]: 跨平台通用导航模型，同属图像目标导航方向
- [[NoMaD]]: 无图扩散导航策略，相关导航基线
- [[OmniVLA]]: 直接策略基线，NavWAM 在成功率和 ATE 上均超越

### 方法相关

- [[扩散策略]]: NavWAM 使用扩散去噪作为核心推理机制
- [[Diffusion Transformer]]: 模型骨干的核心架构
- [[World-Action Model]]: NavWAM 所属的方法范式
- [[Goal-Progress Value]]: 目标进度标量，NavWAM 新引入的监督信号

### 硬件/数据相关

- [[Direct Drive Tech Diablo]]: 真实部署使用的轮式机器人平台
- [[HM3D]]: Habitat-Matterport 3D 仿真数据集，Phase 1 预训练

---

## 速查卡片

> [!summary] NavWAM: Navigation World Action Model
> - **核心**: 将导航世界模型与直接策略统一为单次扩散去噪，无需外部规划器
> - **方法**: 基于 Cosmos Predict2 的九帧潜在画布，联合预测动作块、未来观测和目标进度值
> - **结果**: 真实机器人成功率 79.2%，go_stanford ATE 0.192（w/ FT），优于 NWM+CEM baseline
> - **代码**: [项目主页](https://dachii-azm.github.io/navwam/)（代码暂未开源）

---

*笔记创建时间: 2026-06-12*
