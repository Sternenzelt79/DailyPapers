---
title: "RLDX-1 Technical Report"
method_name: "RLDX-1"
authors: [Dongyoung Kim, "...", Jinwoo Shin]
year: 2026
venue: arXiv
tags: [vla, dexterous-manipulation, world-model, diffusion-model, embodied-ai, flow-matching, humanoid, multi-stream-transformer]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.03269v2
created: 2026-05-09
---

# 论文笔记：RLDX-1 Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RLWRLD Inc. + KAIST (Jinwoo Shin) |
| 日期 | May 2026 |
| 项目主页 | https://rlwrld.ai/rldx-1 |
| 对比基线 | [[Pi05]], [[GR00T-N1.6]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.03269) / [HTML](https://arxiv.org/html/2605.03269v2) |

---

## 一句话总结

> RLDX-1 通过 [[Multi-Stream Action Transformer|MSAT]] 把 [[Motion Module|运动感知]]、[[Memory Module|长期记忆]] 和 [[Physics Stream|物理感知]] 三个能力以独立模态流方式注入 [[VLA]]，在灵巧操作任务上以约 87% 成功率显著超越 π0.5 和 GR00T N1.6。

---

## 核心贡献

1. **MSAT 架构**: 提出 [[Multi-Stream Action Transformer]]，扩展 [[MM-DiT]] 把 cognition / action / physics 三类异构模态用专用 stream + 跨模态 joint self-attention 统一处理。
2. **三大功能模块**: 在通用 [[VLA]] 之上引入 [[Motion Module]]（基于 [[Space-Time Self-Similarity]]）、[[Memory Module]]（cognition token 队列）、[[Physics Stream]]（torque + tactile），分别解决高速运动、长程任务、接触式操作。
3. **合成数据流水线**: 用 [[FLUX.2]] 做场景增强、[[Cosmos-Predict2]] 做 I2V、[[Inverse Dynamics Model|IDM]] 做动作标注，在 GR-1 Tabletop 上单凭合成数据相对真实数据带来 +9.1% 提升。
4. **三阶段训练 + RECAP RL**: Pre-train (1.5M episodes) → Mid-train (ALLEX/FR3) → Post-train，其中 RL 阶段用 [[VLM Critic|文本式 VLM Critic]]（让 VLM 直接输出整数 token 当 value）。
5. **推理优化**: 静态图 + 自定义算子融合，将 PyTorch Eager 71.2ms 推理压到 43.7ms，1.63× 加速，支持实机部署。

---

## 问题背景

### 要解决的问题
当前 [[VLA]] 模型在通用语义理解上很强，但在真实世界灵巧操作（dexterous manipulation）中仍缺乏三类底层能力：(1) 对场景中**高速运动物体**的感知；(2) 跨**长时序**任务的状态记忆；(3) 接触式操作所需的**物理量感知**（力 / 力矩 / 触觉）。

### 现有方法的局限
- [[Pi05]] / [[GR00T-N1.6]] 等模型主要靠 RGB + 语言，缺少触觉与力矩通道，遇到 plug insertion 这类接触任务退化严重。
- 基于单帧或短帧窗口的 [[Diffusion Policy|动作扩散策略]] 难以察觉传送带等场景中的物体速度。
- 把所有模态平铺成 token 序列与 self-attention 共享会导致**模态内统计被异质模态稀释**，physics 信号被淹没。

### 本文的动机
保持 MM-DiT 的双流跨模态思想，但把 cognition / action / physics 拆分为**专用 stream**，在早期块独立处理保留模态结构，再通过 joint self-attention 跨流交互，并用 stream masking 支持物理信号缺失时的回退。

---

## 方法详解

### 模型架构

RLDX-1 采用 **VLM + MSAT 双组件** 架构：

- **输入**: 语言指令 $l$ + 多帧观测 $o_{t-6}, o_{t-4}, o_{t-2}, o_t$ + 本体状态 $s_t$ + 物理信号 $p_t$（torque/tactile）
- **VLM Backbone**: [[Qwen3-VL]] 8B，加入 [[Cognition Token|64 个 learnable cognition tokens]] 聚合视觉-语言上下文
- **核心模块**:
  - [[Motion Module]]：在第 9/27 层视觉编码器中插入 [[Space-Time Self-Similarity|STSS]] encoder
  - [[Memory Module]]：维护 $n_{mem}=3$ 的 cognition feature 队列 $\mathbf{Q}_t$，过 causal-attention Transformer 得 $\mathbf{m}_t$
  - [[Physics Stream]]：独立处理 torque 与触觉，用 [[Stream Masking]] 在缺失时跳过
- **Action Head**: [[Multi-Stream Action Transformer|MSAT]]，[[Flow Matching|流匹配]] 生成 [[Action Chunking|动作块]] $\mathbf{a}_{t:t+H}$
- **总参数**: ~8B（VLM）+ MSAT

### 核心模块

#### 模块1: [[Motion Module|运动感知模块]]

**设计动机**: 普通 ViT 视觉编码器的 token 间相似度只反映外观，缺乏时序运动线索；要抓取传送带物体需感知物体速度。

**具体实现**:
- 在 27 层视觉 encoder 的第 9 层后注入 [[Space-Time Self-Similarity|STSS]] encoder
- 计算每个 spatio-temporal token 与其局部时空邻居的相关性矩阵
- 通过 residual 连接把运动感知特征加回主干
- 仅作用于中间层而非顶层，避免破坏语义对齐

#### 模块2: [[Memory Module|长期记忆模块]]

**设计动机**: 长程任务（如"从盒子里挑选物体"）需要记住此前已操作的目标。

**具体实现**:
- 维护一个 FIFO 队列 $\mathbf{Q}_t = [\mathbf{h}_{t-3\Delta}, \mathbf{h}_{t-2\Delta}, \mathbf{h}_{t-\Delta}]$ 缓存历史 [[Cognition Token]]
- 当前 cognition $\mathbf{h}_t$ 与队列拼接后送入一个 causal-attention Transformer
- 输出 memory feature $\mathbf{m}_t$ 与 $\mathbf{h}_t$ 一起进入 MSAT 的 cognition stream
- 时序覆盖：ALLEX 平台 120 timestep；FR3 平台 48 timestep

#### 模块3: [[Physics Stream|物理感知流]]

**设计动机**: 插拔、按压等接触任务需要 torque / tactile 反馈，但相机帧率高、物理信号低维高频，平铺成 token 会被视觉 token 淹没。

**具体实现**:
- 单独开辟 P-stream，在 MSAT 的早期 triple-stream 块中与 C / A 流并行
- 后期 block 中 C-A merged 流与 P 流通过 joint self-attention 跨流交互
- 用 [[Stream Masking]] 在不带物理传感器的平台上整条 P-stream 置 0
- 辅助目标：预测未来 $L$ 步物理信号，使用与动作同一套 [[Flow Matching]]

#### 模块4: MSAT 块结构

- **Double-Stream Block（早期）**：C-stream 处理 $[\mathbf{h}_t, \mathbf{m}_t]$，A-stream 处理 $[\mathbf{s}_t, \mathbf{a}^{\tau}_{t:t+H}]$
- **Single-Stream Block（晚期）**：把 C-A 序列合并联合处理
- **Triple-Stream Block（带 P）**：早期块拆为 C / A / P 三流，晚期合并为 C-A + P 两流
- **位置编码**：A-stream 上施加 [[RoPE|Rotary Position Embedding]] 表达时序结构
- **Timestep 编码**：将流匹配 $\tau$ 作为 in-context token 注入，而非 [[AdaLN|feature-wise modulation]]

---

## 关键公式

### 公式1: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}(\theta; t, \tau, \epsilon) = \left\| \mathbf{u}_\theta(\mathbf{a}^{\tau}_{t:t+H}, \tau, \mathbf{c}_t) - (\mathbf{a}_{t:t+H} - \epsilon) \right\|_2^2
$$

**含义**: 训练 MSAT 的速度场 $\mathbf{u}_\theta$ 去拟合从噪声 $\epsilon$ 指向真实动作块 $\mathbf{a}_{t:t+H}$ 的位移向量，等价于 [[Conditional Flow Matching]] 的标准回归目标。

**符号说明**:
- $\mathbf{u}_\theta$: 由 MSAT 参数化的条件速度场
- $\mathbf{a}^{\tau}_{t:t+H} = \tau \mathbf{a}_{t:t+H} + (1-\tau)\epsilon$: 在时间 $\tau \in [0,1]$ 处的中间样本
- $\mathbf{a}_{t:t+H}$: 长度 $H$ 的真实动作块
- $\mathbf{c}_t$: 条件上下文（cognition $\mathbf{h}_t$ + memory $\mathbf{m}_t$ + state $\mathbf{s}_t$ + physics tokens）
- $\epsilon \sim \mathcal{N}(0, I)$: 噪声样本
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间

### 公式2: [[Euler Method|Euler 推理]]

$$
\mathbf{a}^{\tau_{i+1}}_{t:t+H} = \mathbf{a}^{\tau_i}_{t:t+H} + (\tau_{i+1} - \tau_i) \cdot \mathbf{u}_\theta\!\left(\mathbf{a}^{\tau_i}_{t:t+H}, \tau_i, \mathbf{c}_t\right)
$$

**含义**: 部署时通过 Euler 法在 $\tau \in [0,1]$ 上对学好的速度场积分，从随机噪声推到动作块，少量步数（论文中常用 4 步）即可。

**符号说明**:
- $\tau_i$: 离散积分步，$0 = \tau_0 < \tau_1 < \dots < \tau_N = 1$
- 其余符号同公式 1

### 公式3: [[Memory Module|记忆队列更新]]（自定义形式化）

$$
\mathbf{m}_t = \mathrm{CausalTransformer}\!\left(\big[\mathbf{Q}_t,\; \mathbf{h}_t\big]\right), \qquad \mathbf{Q}_{t+\Delta} = \mathrm{enqueue}\!\left(\mathbf{Q}_t,\; \mathbf{h}_t\right)
$$

**含义**: 当前 cognition 与历史队列经因果注意力融合得到 memory feature $\mathbf{m}_t$，再以步长 $\Delta$ 把当前特征压入队列。

**符号说明**:
- $\mathbf{Q}_t$: 长度 $n_{mem}=3$ 的历史 cognition 缓存
- $\Delta$: 队列采样间隔（ALLEX=40、FR3=16 timestep 等）
- $\mathbf{h}_t$: 当前帧 cognition tokens

### 公式4: [[Physics Stream|物理信号辅助预测]]

$$
\mathcal{L}_{phys} = \left\| \mathbf{u}^{P}_\theta(\mathbf{p}^{\tau}_{t:t+L}, \tau, \mathbf{c}_t) - (\mathbf{p}_{t:t+L} - \epsilon^{P}) \right\|_2^2
$$

**含义**: 与动作流共用 flow-matching 框架，让 P-stream 同时学习预测未来 $L$ 步的力矩 / 触觉序列，作为 representation 自监督。

**符号说明**:
- $\mathbf{p}_{t:t+L}$: 未来 $L$ 步真实物理信号
- $\mathbf{u}^{P}_\theta$: P-stream 的速度头

### 公式5: [[VLM Critic|文本式价值预测]]（[[RECAP]] 中）

$$
V_\phi(s) = \mathbb{E}_{k \sim p_\phi(\cdot \mid s)}\big[k\big],\quad p_\phi(k\mid s) = \mathrm{softmax}\big(\mathrm{LMHead}(\mathrm{VLM}(s))\big)_k
$$

**含义**: 让 VLM 把 unnormalized 整数 $k \in \{0,\dots,K\}$ 当作文本 token 输出，其概率分布即是 value 估计；不引入额外回归头，省去对齐成本。

**符号说明**:
- $V_\phi$: critic 估值
- $p_\phi(k\mid s)$: VLM 在整数 token 上的概率分布
- $K$: 离散化粒度（论文里典型为 100）

---

## 关键图表

### Figure 1: RLDX-1 Overview / 一图看懂

![Figure 1](https://arxiv.org/html/2605.03269v2/x3.png)

**说明**: RLDX-1 作为通用 [[VLA]] 模型，集成 motion / memory / physics 三类功能能力，目标是真实世界灵巧操作。

### Figure 2: 三大功能能力总览

![Figure 2](https://arxiv.org/html/2605.03269v2/x4.png)

**说明**: 给定视频观测和语言指令，RLDX-1 通过 [[Motion Module]]、[[Memory Module]]、[[Physics Stream]] 三条路径感知运动 / 记忆 / 力矩与触觉，输出未来动作。

### Figure 3: RLDX-1 完整架构

![Figure 3](https://arxiv.org/html/2605.03269v2/x5.png)

**说明**: 由 [[VLM]]（Qwen3-VL 8B + cognition token）和 [[Multi-Stream Action Transformer|MSAT]] 两部分组成；triple-stream 块在早期分流，后期合并 C-A 与 P。

### Figure 4: 合成数据生成与过滤流水线

![Figure 4](https://arxiv.org/html/2605.03269v2/x6.png)

**说明**: (1) 数据生成：FLUX.2 做场景/任务增强，Cosmos-Predict2 做 I2V，[[Inverse Dynamics Model|IDM]] 做动作标注；(2) 过滤：VLM 视频质量过滤 + 基于 [[V-JEPA2]] 探针的 motion-consistency 过滤。

### Figure 5: 合成数据样例

![Figure 5](https://arxiv.org/html/2605.03269v2/x7.png)

**说明**: 同一段 ALLEX demo 经 task augmentation 与 scene augmentation 得到不同变体，扩展任务-场景多样性。

### Figure 6: 预训练数据集组成

![Figure 6](https://arxiv.org/html/2605.03269v2/x8.png)

**说明**: 共 1.5M episode，覆盖 OXE 870K、DROID 92K、Galaxea 114K、AgiBot World 275K、Fourier ActionNet 30K、Humanoid Everyday 9K、合成 150K。

### Figure 7: Mid-Training 数据组成

![Figure 7](https://arxiv.org/html/2605.03269v2/x9.png)

**说明**: 针对 ALLEX 与 FR3 两个目标平台的 in-house 数据切分。

### Figure 8: VLM-Critic 的 Value 轨迹

![Figure 8](https://arxiv.org/html/2605.03269v2/x10.png)

**说明**: 在 cube pick-and-place 任务上，文本预测式 critic 与 distributional critic 在不同 timestep 上的 value 单调上升，体现进度估计能力。

### Figure 9: Dynamic vs. Static Graph

![Figure 9](https://arxiv.org/html/2605.03269v2/x11.png)

**说明**: 静态图把 RoPE 与 attention mask 预计算后单次 CUDA Graph capture，把多次 kernel launch 合并，消除 launch 开销。

### Figure 10: 算子融合的内存访问

![Figure 10](https://arxiv.org/html/2605.03269v2/x12.png)

**说明**: 把 RMSNorm + RoPE + Attention 融合后，Q/K/V 留在 on-chip，仅写出最终输出，减少 HBM 流量。

### Figure 11: 仿真基准全景

![Figure 11](https://arxiv.org/html/2605.03269v2/x13.png)

**说明**: 涵盖 LIBERO、SIMPLER、RoboCasa、GR-1 Tabletop 四大仿真套件的整体表现。

### Figure 12: 真机平台

![Figure 12](https://arxiv.org/html/2605.03269v2/x14.png)

**说明**: OpenArm（Inspire 灵巧手）、ALLEX 48-DoF 人形、Franka Research 3 + AnySkin 触觉 + 力矩。

### Figure 13: OpenArm 评测任务

![Figure 13](https://arxiv.org/html/2605.03269v2/x15.png)

**说明**: 针对通用性的多任务评估集，覆盖 unseen object 与 unseen task。

### Figure 14: OpenArm 实测结果

![Figure 14](https://arxiv.org/html/2605.03269v2/x16.png)

**说明**: RLDX-1 在 unseen object / unseen task 上分别 54.2% / 54.2%，显著优于 [[Pi05|π0.5]]（37.5% / 45.8%）。

### Table 1: 仿真基准对比

| Benchmark | RLDX-1 | GR00T N1.6 | π0.5 |
|-----------|--------|-----------|------|
| LIBERO | **97.8%** | 96.7% | 96.9% |
| LIBERO-Plus | **86.7%** | 72.6% | 86.5% |
| SIMPLER Google-VM | **81.5%** | 76.1% | 72.7% |
| WidowX | **71.9%** | 57.1% | 46.9% |
| RoboCasa Kitchen | **70.6%** | 66.2% | – |
| GR-1 Tabletop | **58.7%** | 47.6% | – |
| RoboCasa365 Avg | **32.1%** | 26.9% | – |

**说明**: RLDX-1 在七大仿真基准上全部刷新 SOTA，复杂基准（GR-1 Tabletop / RoboCasa365）领先幅度尤其明显。

### Table 2: 功能能力实测（ALLEX 人形 / FR3）

| 任务 | 评估能力 | RLDX-1 | π0.5 |
|------|----------|--------|------|
| Object-in-Box Selection | 长期记忆 | **91.7%** | ~30% |
| Conveyor Belt Catch | 运动感知 | **87.5%** | 29.2% |
| Plug Insertion (FR3) | 物理感知 | 显著提升 | 接触失败 |
| ALLEX 总体 | 综合 | **86.8%** | ~40% |

**说明**: 三大功能模块分别在最相关任务上拉开 50+ 点的成功率差距，验证模块设计的针对性。

### Table 3: 推理优化收益

| 配置 | 延迟 | 加速 |
|------|------|------|
| PyTorch Eager | 71.2 ms | 1.00× |
| + 静态图（CUDA Graph） | 43.7 ms | **1.63×** |
| + 自定义算子融合 | 进一步压缩 | – |

**说明**: 静态图消除 launch 开销是主要收益来源，融合 RMSNorm+RoPE+Attention 进一步优化短 prefill workload。

### Table 4: 合成数据消融（GR-1 Tabletop）

| 训练数据 | 成功率 |
|----------|--------|
| 仅真实数据 | baseline |
| 仅合成数据 | baseline + **9.1%** |
| 真 + 合成 | 进一步提升 |

**关键发现**: 经过 IDM 标注 + V-JEPA2 motion-consistency 过滤的合成数据，在 GR-1 Tabletop 上比真实数据单独训练高 9.1%，证明合成数据流水线的有效性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Open-X-Embodiment\|OXE]] | 870K | 多机型、多任务公开集 | 预训练 |
| [[DROID]] | 92K | FR3 + Robotiq | 预训练 |
| Galaxea Open-World | 114K | 双臂 | 预训练 |
| [[AgiBot World]] | 275K | 人形 | 预训练 |
| Fourier ActionNet | 30K | 人形 | 预训练 |
| Humanoid Everyday | 9K | 日常人形任务 | 预训练 |
| 合成数据 | 150K | GR-1 humanoid I2V 合成 | 预训练 |
| In-house FR3 | – | AnySkin + 力矩 | Mid-training |
| In-house ALLEX | – | 48-DoF 人形 + 力矩 | Mid-training |

### 实现细节

- **Backbone**: Qwen3-VL 8B（顶部 4 层可训，其余冻结）
- **Action Head**: MSAT，flow matching 4 步采样
- **Pre-training**: 100K steps, batch 8192, lr=1e-4，5% 线性 warmup，64×H200，195 小时
- **Mid-training**: 25K steps, batch 1024, lr=5e-5, 64×H200, 15 小时；2K-step alignment warmup + 0.3 modality dropout
- **Post-training**: 自适应数据采集（基础 + 失败模式精修）+ [[RECAP]] RL（VLM critic）
- **硬件**: 64× NVIDIA H200

### 可视化结果

- ALLEX humanoid 上整体 86.8% 成功率，对比同尺寸 baseline ~40%。
- Conveyor Belt Catch（动态目标抓取）87.5%：[[Motion Module]] 的运动感知是关键。
- Object-in-Box Selection（长程记忆）91.7%：[[Memory Module]] 的 cognition 队列让模型记住已经被挑过的物体。
- FR3 Plug Insertion：[[Physics Stream]] 提供 torque + tactile 反馈，使插拔接触阶段从频繁失败变为稳定成功。

---

## 批判性思考

### 优点
1. **模块拆分思路清晰**：把 motion / memory / physics 视为正交能力，用专用 stream + stream masking 在不同硬件平台上灵活启用，避免"all-in-one token soup"。
2. **工程闭环完整**：从合成数据流水线、三阶段训练、RECAP RL 到推理静态图与算子融合，少见地把 VLA 的全部工程难点串起来。
3. **VLM-Critic 设计巧妙**：用文本 token 直接预测整数 value，复用 LM head，避免训练新的回归头与对齐误差。
4. **真机数据扎实**：ALLEX/FR3/OpenArm 三平台同时评估，且接触任务（plug insertion）有定性证据。

### 局限性
1. **算力门槛极高**：64×H200、195 小时预训练 + 15 小时 mid-train，社区难以复现，没有公开权重。
2. **物理模态依赖硬件**：P-stream 在 ALLEX/FR3 受益明显，但要落地到没有 AnySkin / 力矩传感器的平台仍需 stream masking 跳过，丧失能力。
3. **Memory Module 容量有限**：$n_{mem}=3$ 的 cognition 队列只覆盖 ALLEX 的 120 timestep，对真正长程任务（数分钟级）仍有限。
4. **VLM-Critic 规模化未知**：用 8B VLM 当 critic，在真实大规模 RL 训练中是否稳定（reward hacking、过拟合 prompt）尚缺消融。
5. **合成数据 9.1% 提升仅在 GR-1 Tabletop 验证**，泛化到其它平台与任务的边界没给。

### 潜在改进方向
1. 用 [[Hierarchical Memory]] 或可学习的压缩 token 替换 FIFO 队列，扩展到分钟级长程记忆。
2. 把 P-stream 替换为统一的 proprioception token，在缺触觉时由模型自己合成弱物理 prior。
3. 把合成数据流水线扩展到 [[3D Gaussian Splatting]] 表征上，做物理一致的场景增强。
4. 探索更小规模（1–3B）下的 MSAT，让社区可复现。

### 可复现性评估
- [ ] 代码开源（截至 v2 未见）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（步数、batch、lr、硬件均给出）
- [x] 公开数据集可获取（OXE / DROID / AgiBot World 等）
- [ ] In-house ALLEX/FR3 数据未公开

---

## 关联笔记

### 基于
- [[MM-DiT]]: MSAT 的双流 / 多流 self-attention 灵感来源
- [[Flow Matching]]: 动作头与 P-stream 共用的生成框架
- [[Pi05]]: 直接对标的通用 VLA baseline
- [[GR00T-N1.6]]: 同期对比 baseline

### 对比
- [[Pi05]]: 缺乏专用 physics / memory 模块，复杂任务下成功率约 40%
- [[GR00T-N1.6]]: 单流 transformer，无显式运动感知

### 方法相关
- [[Multi-Stream Action Transformer]]: 核心架构
- [[Motion Module]]: 运动感知组件
- [[Memory Module]]: 长程记忆组件
- [[Physics Stream]]: 物理感知组件
- [[Space-Time Self-Similarity]]: 运动模块内核
- [[Cognition Token]]: VLM 与 MSAT 的接口
- [[VLM Critic]]: RECAP RL 中的文本式 critic
- [[RECAP]]: 后训练 RL 框架
- [[Stream Masking]]: 模态缺失时的开关机制

### 硬件/数据相关
- ALLEX 48-DoF 人形 + 力矩反馈
- Franka Research 3 + AnySkin tactile + 力矩
- OpenArm + Inspire 灵巧手
- [[Inverse Dynamics Model]] 用于合成动作标注
- [[V-JEPA2]] 用于 motion-consistency 过滤
- [[Cosmos-Predict2]] / [[FLUX.2]] 用于合成数据生成

---

## 速查卡片

> [!summary] RLDX-1 Technical Report
> - **核心**: 通过 MSAT 多流架构把 motion / memory / physics 注入 VLA，实机灵巧操作 86.8%
> - **方法**: Qwen3-VL 8B + cognition tokens + MSAT + flow matching；运动靠 STSS、记忆靠 cognition queue、物理靠独立 P-stream + stream masking
> - **结果**: 七大仿真基准全 SOTA；ALLEX 86.8% vs π0.5 ~40%；推理 1.63× 加速
> - **代码**: 未开源；项目页 https://rlwrld.ai/rldx-1

---

*笔记创建时间: 2026-05-09*
