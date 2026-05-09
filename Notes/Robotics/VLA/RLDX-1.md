---
title: "RLDX-1 Technical Report"
method_name: "RLDX-1"
authors: [Dongyoung Kim, Huiwon Jang, Myungkyu Koo, et al.]
year: 2026
venue: arXiv
tags: [vla, robotics, dexterous-manipulation, world-model, flow-matching, humanoid]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.03269v1
created: 2026-05-09
---

# 论文笔记：RLDX-1 Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RLWRLD (68 位作者) |
| 日期 | May 2026 |
| 项目主页 | https://rlwrld.ai/rldx-1 |
| 对比基线 | [[Pi05]], [[GR00T-N1.6]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.03269) / [Code](https://github.com/RLWRLD/RLDX-1) |

---

## 一句话总结

> RLDX-1 通过 [[Multi-Stream Action Transformer|MSAT]] 架构融合运动感知、长期记忆和物理感知三大模块，把 ALLEX 人形机器人灵巧操作成功率从 ~40% 提升到 86.8%。

---

## 核心贡献

1. **MSAT 架构**: 提出 [[Multi-Stream Action Transformer]]，通过 modality-specific streams 与 cross-modal joint self-attention 统一异构模态。
2. **三大能力模块**: 集成 [[Space-Time Self-Similarity|STSS]] 运动感知、[[Memory Module|长期记忆队列]] 与 [[Physics Stream|物理感知流]]。
3. **合成数据流水线**: VLM 驱动的 task/scene augmentation + motion-consistency 过滤，覆盖罕见操作场景。
4. **推理优化**: Graph-level 静态图 + kernel-level 算子融合，单步从 71.2 ms 降到 43.7 ms（1.63× 加速）。
5. **三阶段训练**: Pre-training (1.5M episodes) → mid-training → post-training，构建多本体先验。

---

## 问题背景

### 要解决的问题
现有 [[Vision-Language-Action Model|VLA]] 在真实复杂任务中缺少三种关键能力：**运动感知**（动态物体跟踪）、**长期记忆**（跨多步任务的状态保持）、**物理感知**（接触力/触觉反馈）。

### 现有方法的局限
- [[Pi05]]、[[GR00T-N1.6]] 等单流 VLA 把所有模态压扁进同一序列，导致模态间相互干扰。
- 缺少显式的物理模态建模，灵巧抓取/插入类任务成功率长期 ~40%。
- 推理延迟过高，不适合实时高频控制。

### 本文的动机
通过 **多流（multi-stream）** 设计在保留模态独立表征的同时，用 [[Joint Self-Attention|联合自注意力]] 实现跨模态交互；并以 [[Flow Matching]] 同时预测动作和未来物理传感读数。

---

## 方法详解

### 模型架构

RLDX-1 采用 [[Multi-Stream Action Transformer|MSAT]] 架构：

- **输入**: 语言指令 $l$ + 4 帧视觉观测 $o_{t-6:t}$（offsets $\{-6,-4,-2,0\}$）+ 本体状态 $s_t$ + 历史物理读数。
- **Backbone**: 视觉编码器 + LLM（cognition 流）。
- **核心模块**:
  - [[Space-Time Self-Similarity|STSS]]：在视觉编码器第 9 层以残差方式注入运动特征。
  - [[Memory Module|记忆模块]]：在 LLM 第 4 层后维护长度 $n_{mem}=3$ 的 cognition 队列，按间隔 $H+1$ 采样。
  - [[Physics Stream|物理流]]：与 action 流并行，预测未来 $L=H+1$ 步传感读数。
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$，ALLEX horizon=40，FR3 horizon=16。
- **总流数**: Cognition (C) / Action (A) / Physics (P) 三条流。

### 核心模块

#### 模块1: 多流 Transformer Block

**设计动机**: 不同模态需要不同归一化与投影，但又需在同一注意力下交互。

**具体实现**:
- 三种 block 形态依次堆叠：
  - **Double-stream (early)**：仅 C + A，每条流独立 [[RMSNorm]] 与 QKV 投影，再 concat 做 [[Joint Self-Attention]]。
  - **Triple-stream (middle)**：加入 P 流。
  - **Single/Double-stream (late)**：合并 C-A，可选保留 P。
- 通过 [[RoPE]] 位置编码 + [[Causal Attention]] 维持时序因果。

#### 模块2: Space-Time Self-Similarity (STSS)

**设计动机**: 让视觉编码器显式感知**像素级运动**，而非只看单帧。

**具体实现**:
- 计算同一 patch 在相邻帧之间的 token 相似度矩阵。
- 残差注入到视觉编码器第 9 层，不破坏预训练特征。

#### 模块3: 长期记忆模块

**设计动机**: 解决 Object-in-Box 类需要"记得刚才放了什么"的任务。

**具体实现**:
- 在 LLM 第 4 层后抽取 cognition feature。
- 维护 FIFO 队列 $\{m_1, m_2, m_3\}$，sampling 间隔 $H+1$。
- 通过 [[Causal Attention|轻量级因果 Transformer]] 聚合历史。

#### 模块4: Physics Stream

**设计动机**: 触觉/力反馈对插入、抓取至关重要。

**具体实现**:
- 与 action 流共享 backbone，但有独立投影头。
- 用 [[Flow Matching]] 同时学习 action 与未来 $L$ 步物理传感预测。

#### 模块5: 合成数据流水线

- **Task augmentation**: [[VLM]] 通过 factorized composition + skill-primitive conditioning 生成新指令。
- **Scene augmentation**: 对 initial frame 用 [[FLUX.2-dev]] 做 I2I editing；对全 video 用 [[Cosmos-Transfer2.5]] 做 V2V style transfer。
- **Quality filter**: VLM 判断指令对齐 + 物理 plausibility。
- **Motion-consistency filter**: 训练 probe 比对生成视频与 simulator rollout。

---

## 关键公式

### 公式1: [[Flow Matching|动作流匹配训练目标]]

$$
a^{\tau} = \tau \, a + (1 - \tau) \, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \; \tau \sim \mathcal{U}(0, 1)
$$

**含义**: 在干净动作 $a$ 和高斯噪声 $\varepsilon$ 之间做线性插值，形成训练样本 $a^\tau$，用于学习速度场。

**符号说明**:
- $a \in \mathbb{R}^{H \times d_a}$: 真实动作块（ground-truth action chunk）。
- $\varepsilon$: 标准高斯噪声。
- $\tau$: 时间步，均匀采样自 $[0, 1]$。
- $a^\tau$: 噪声化动作。

### 公式2: [[Conditional Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau, \varepsilon, (a, c_t)} \left\| u_\theta(a^{\tau}, \tau, c_t) - (a - \varepsilon) \right\|_2^2
$$

**含义**: 网络 $u_\theta$ 在 condition $c_t$（cognition + memory + physics tokens）下预测从噪声指向真实动作的速度向量 $a-\varepsilon$。

**符号说明**:
- $u_\theta$: MSAT 的 action 流输出头。
- $c_t$: 当前时刻所有条件 token 的集合。
- $a - \varepsilon$: 目标速度场（直线路径）。

### 公式3: [[Physics Stream|物理流联合损失]]

$$
\mathcal{L}_{\text{phys}} = \mathbb{E}_{\tau, \varepsilon', (p, c_t)} \left\| v_\phi(p^{\tau}, \tau, c_t) - (p - \varepsilon') \right\|_2^2, \quad p \in \mathbb{R}^{L \times d_p}
$$

**含义**: 与动作流对偶，预测未来 $L = H+1$ 步物理传感读数 $p$（力/触觉）的速度场。

**符号说明**:
- $v_\phi$: physics 流输出头。
- $p$: 未来物理读数序列。
- $\varepsilon'$: 独立采样的物理噪声。

### 公式4: 总训练损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda_p \, \mathcal{L}_{\text{phys}}
$$

**含义**: 动作与物理两条流以加权和方式联合训练。

**符号说明**:
- $\lambda_p$: 物理损失权重。

### 公式5: 推理时的 [[Euler Integration|欧拉积分采样]]

$$
a^{\tau + \Delta\tau} = a^{\tau} + \Delta\tau \cdot u_\theta(a^{\tau}, \tau, c_t), \quad a^0 = \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 推理时从纯噪声出发，沿学到的速度场做欧拉积分若干步得到最终动作块 $a^1$。

**符号说明**:
- $\Delta\tau$: 积分步长（步数 = $1 / \Delta\tau$）。
- $a^0, a^1$: 起始噪声 / 终态动作。

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.03269v1/x1.png)

**说明**: RLDX-1 整体定位——以 [[Multi-Stream Action Transformer|MSAT]] 为核心，覆盖运动/记忆/物理三大能力，部署到 ALLEX 人形机器人和 FR3 机械臂。

### Figure 2: RLDX-1 Architecture Overview / 架构总览

![Figure 2](https://arxiv.org/html/2605.03269v1/x2.png)

**说明**: 输入语言 + 多帧视觉 + 状态 → cognition / action / physics 三条流 → 通过 MSAT 块逐层联合注意力 → 输出动作块和未来物理读数。

### Figure 3: Detailed Architecture / 细节结构

![Figure 3](https://arxiv.org/html/2605.03269v1/x3.png)

**说明**: 三类 block（double / triple / merged）的 RMSNorm + QKV 投影 + concat-attn 细节，[[STSS]] 注入位置（vision layer 9）和 memory 注入位置（LLM layer 4）。

### Figure 4: Synthetic Data Framework / 合成数据流水线

![Figure 4](https://arxiv.org/html/2605.03269v1/x4.png)

**说明**: Task aug ([[VLM]]) + Scene aug ([[FLUX.2-dev]] / [[Cosmos-Transfer2.5]]) + Quality / Motion-consistency 过滤的全链路。

### Figure 5: Synthetic Data Examples / 合成数据样例

![Figure 5](https://arxiv.org/html/2605.03269v1/x5.png)

**说明**: I2I 编辑生成多样化背景与物体外观；V2V 风格迁移在保持运动语义的前提下覆盖光照/材质域。

### Figure 6: Pre-training Dataset Composition

![Figure 6](https://arxiv.org/html/2605.03269v1/x6.png)

**说明**: 1.5M episodes 预训练数据来源——OXE 870K, DROID 92K, Galaxea 114K, AgiBot 275K, ActionNet 30K, Humanoid Everyday 9K, 合成 GR-1 150K。

### Figure 7: Mid-training Data Composition

![Figure 7](https://arxiv.org/html/2605.03269v1/x7.png)

**说明**: 针对单一本体（ALLEX / FR3）的功能数据混合比例。

### Figure 8: Value Prediction Comparison

![Figure 8](https://arxiv.org/html/2605.03269v1/x8.png)

**说明**: 物理流预测与真实传感读数曲线高度吻合，证明 [[Physics Stream]] 学到了有效物理动力学。

### Figure 9: Dynamic vs. Static Graph Execution

![Figure 9](https://arxiv.org/html/2605.03269v1/x9.png)

**说明**: 动态图导致 [[CUDA Graph]] 频繁重建；静态图把整个推理收敛到单图，消除碎片化开销。

### Figure 10: Operator Fusion Memory Effects

![Figure 10](https://arxiv.org/html/2605.03269v1/x10.png)

**说明**: 自定义 RMSNorm-RoPE-Attention 融合算子大幅减少显存读写，是 1.63× 加速的主要来源。

### Figure 11: Simulation Benchmarks

![Figure 11](https://arxiv.org/html/2605.03269v1/x11.png)

**说明**: LIBERO / GR-1 Tabletop / RoboCasa365 / SIMPLER WidowX 四个仿真环境上的成功率柱状图。

### Figure 12: Real-Robot Platforms

![Figure 12](https://arxiv.org/html/2605.03269v1/x12.png)

**说明**: ALLEX 人形（双臂 + 灵巧手）和 Franka FR3 机械臂的硬件配置。

### Figure 13: OpenArm Humanoid Benchmark

![Figure 13](https://arxiv.org/html/2605.03269v1/x13.png)

**说明**: ALLEX 在 conveyor belt / Object-in-Box 等记忆和动态任务上的设置。

### Figure 14: OpenArm Results

![Figure 14](https://arxiv.org/html/2605.03269v1/x14.png)

**说明**: RLDX-1 在 ALLEX 综合任务集 86.8% vs 基线 ~40%，传送带任务 87.5% vs π₀.₅ 29.2%。

### Table 1: Simulation Benchmark Results

| Benchmark | RLDX-1 | π₀.₅ | GR00T N1.6 | Δ vs best |
|-----------|--------|------|-----------|-----------|
| LIBERO | **97.8%** | 95.1% | 96.7% | +1.1 |
| GR-1 Tabletop | **58.7%** | 41.2% | 47.6% | +11.1 |
| RoboCasa365 (Avg) | **32.1%** | 22.4% | 26.9% | +5.2 |
| SIMPLER WidowX | **71.9%** | 49.8% | 57.1% | +14.8 |

**说明**: 在四个公开仿真 benchmark 上全面领先，越是动态/长程任务（GR-1, SIMPLER）领先越大。

### Table 2: Real-World Results

| Task | RLDX-1 | π₀.₅ | GR00T N1.6 |
|------|--------|------|-----------|
| ALLEX 综合（13 任务） | **86.8%** | ~40% | ~40% |
| Object-in-Box (memory) | **91.7%** | ~30% | ~30% |
| Conveyor Belt (motion) | **87.5%** | 29.2% | 31.0% |

**说明**: 真实世界差距更悬殊，证明 motion / memory 模块在 sim-to-real 后仍然有效。

### Table 3: 消融实验（基于论文叙述整理）

| 配置 | 综合成功率 | 说明 |
|------|-----------|------|
| w/o STSS | ~70% | 传送带等动态任务掉点显著 |
| w/o Memory | ~65% | Object-in-Box 几乎失败 |
| w/o Physics Stream | ~78% | 灵巧抓取插入掉点 |
| Full RLDX-1 | **86.8%** | - |

**关键发现**: 三个模块互补，移除任意一个都会在对应类型任务上明显退化。

### Table 4: 训练阶段对比

| Stage | Data Scale | Steps | Hours |
|-------|-----------|-------|-------|
| Pre-training | 1.5M episodes | 100K | 195 |
| Mid-training | embodiment-specific | 25K | 15 |
| Post-training | task-specific | varies | varies |

### Table 5: 推理性能

| 配置 | 单步延迟 (ms) | 加速比 |
|------|---------------|--------|
| Baseline (PyTorch eager) | 71.2 | 1.0× |
| + Static graph | 56.4 | 1.26× |
| + Fused kernels | **43.7** | **1.63×** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Open-X-Embodiment | 870K | 多本体跨机构 | 预训练 |
| DROID | 92K | 单臂大规模 | 预训练 |
| AgiBot | 275K | 中文双臂 | 预训练 |
| Galaxea | 114K | 双臂操作 | 预训练 |
| ActionNet | 30K | 第一视角 | 预训练 |
| Humanoid Everyday | 9K | 人形日常 | 预训练 |
| Synthetic GR-1 | 150K | 合成补充 | 预训练 |
| ALLEX 自采 | - | 13 任务真机 | mid/post |

### 实现细节

- **Backbone**: vision encoder + LLM (cognition stream)，参数规模未公开。
- **优化器**: AdamW，pre-train LR=1e-4，mid-train LR=5e-5。
- **Batch Size**: 8192 (pre) / 1024 (mid)。
- **Warmup**: 5%，mid-train 含 2K 步对齐阶段。
- **训练时长**: 195 小时 pre + 15 小时 mid。
- **硬件**: 推理在 NVIDIA RTX 5090，训练硬件未公开。
- **Action horizon**: ALLEX 40 / FR3 16。
- **Memory window**: ALLEX 120 / FR3 48 步历史。

### 可视化结果

- 在 conveyor belt 任务中，RLDX-1 能在物体被动移动的情况下持续跟踪并完成抓取，基线大概率扑空。
- Object-in-Box 任务：模型可在多个干扰物存在时记住"目标已放入哪个盒子"。
- 物理流预测曲线（Fig 8）显示模型已隐式学到接触动力学。

---

## 批判性思考

### 优点
1. **多流设计有理论合理性**：避免模态压扁导致的相互干扰，又通过 joint attention 保留交互。
2. **三模块对应三类长期未解的失败模式**（动态、记忆、接触），实验验证逐一收益。
3. **工程闭环完整**：从合成数据 → 训练 → 推理优化全链路打通，1.63× 加速对实时控制重要。
4. **真实世界差距悬殊**（86.8% vs ~40%），说明改进不是 benchmark hacking。

### 局限性
1. **作者数 68 人**，工程整合度极高但学术贡献难以归属，复现门槛极高。
2. **未公开模型规模和训练算力**，社区难以判断方法论 vs 算力红利的占比。
3. **消融来自正文叙述**，缺少完整 ablation table，部分结论存疑。
4. **物理流仅预测传感读数**，未真正做 [[World Model|可微物理建模]]，泛化到新接触类型存疑。
5. **arXiv ID 2605.03269 在 2026 年发布**，时间戳异常，需注意是 preprint 早期版本。

### 潜在改进方向
1. 把 physics stream 升级为显式 [[World Model]]，做 model-based RL fine-tune。
2. STSS 当前是手工注入到固定层，可探索 learnable injection schedule。
3. Memory 队列长度 $n_{mem}=3$ 偏短，可以引入压缩式长记忆（[[Mamba]] / [[Compressive Transformer]]）。
4. 合成数据 motion-consistency probe 是否会引入 simulator bias，值得 ablation。

### 可复现性评估
- [x] 代码开源（GitHub: RLWRLD/RLDX-1）
- [ ] 预训练模型（未明确）
- [x] 训练细节较完整
- [ ] 部分预训练数据（AgiBot / 自采 ALLEX）不可获取

---

## 关联笔记

### 基于
- [[Pi0]]: flow-matching VLA 的代表前作。
- [[GR00T-N1.6]]: NVIDIA 通用人形 VLA 基线。
- [[Flow Matching]]: 训练目标基础。

### 对比
- [[Pi05]]: 同期单流 VLA，被 RLDX-1 全面超越。
- [[GR00T-N1.6]]: 仿真上接近，真实差距大。
- [[OpenVLA]]: 早期 OXE-trained baseline。

### 方法相关
- [[Multi-Stream Action Transformer]]: 核心架构。
- [[Space-Time Self-Similarity]]: 运动模块。
- [[Memory Module]]: 长期记忆。
- [[Physics Stream]]: 物理感知。
- [[Flow Matching]]: 训练目标。
- [[Action Chunking]]: 输出形式。

### 硬件/数据相关
- [[ALLEX Humanoid]]: 主要测试平台。
- [[Franka FR3]]: 第二平台。
- [[Open-X-Embodiment]]: 预训练数据主源。

---

## 速查卡片

> [!summary] RLDX-1 Technical Report
> - **核心**: 多流 Transformer (cognition/action/physics) + 运动/记忆/物理三模块的 VLA。
> - **方法**: MSAT 架构 + flow matching 联合训练动作和物理读数 + VLM 合成数据 + 静态图&融合算子推理优化。
> - **结果**: ALLEX 真实任务 86.8%（基线 ~40%）；推理 1.63× 加速。
> - **代码**: https://github.com/RLWRLD/RLDX-1

---

*笔记创建时间: 2026-05-09*
