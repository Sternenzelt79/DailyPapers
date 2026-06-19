---
title: "MemoryWAM: Efficient World Action Modeling with Persistent Memory"
method_name: "MemoryWAM"
authors: [Sizhe Yang, Juncheng Mu, Tianming Wei, Chenhao Lu, Xiaofan Li, Linning Xu, Zhengrong Xue, Zhecheng Yuan, Dahua Lin, Jiangmiao Pang, Huazhe Xu]
year: 2026
venue: arXiv
tags: [world-action-model, memory-mechanism, robotic-manipulation, diffusion-policy, non-markovian]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.20562
created: 2026-06-19
---

# 论文笔记：MemoryWAM: Efficient World Action Modeling with Persistent Memory

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The Chinese University of Hong Kong / Tsinghua University / Zhejiang University |
| 日期 | June 2026 |
| 项目主页 | [MemoryWAM](https://yangsizhe.github.io/MemoryWAM/) |
| 对比基线 | [[LingBot-VA]], [[FastWAM]], [[π0.5\|π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.20562) / Code: 未公开 |

---

## 一句话总结

> MemoryWAM 通过混合记忆机制（短期滑动窗口 + 事件边界锚帧 + Gist 压缩 Token）解决 [[World Action Model|WAM]] 推理效率与长程历史上下文保留之间的根本矛盾，在机器人操控上超越现有 WAM 基线。

---

## 核心贡献

1. **混合记忆设计**: 提出三组件 Hybrid Memory（短期帧 + 锚帧 + Gist Token），将 KV Cache 复杂度从 $O(NL)$ 降至 $O(NL/d)$（压缩比 $d=15$）
2. **持久记忆机制**: Gist Token 永久保留，跨越整条轨迹传递压缩的长程历史语义，无需牺牲推理速度
3. **仿真与真实世界验证**: 在 [[RMBench]] 9 个记忆依赖任务上平均 83.0% 成功率，超越 LingBot-VA（78.2%），真实双臂机器人同样取得最优性能

---

## 问题背景

### 要解决的问题

机器人操控中大量任务是**非马尔可夫（Non-Markovian）**的——当前状态不足以决定下一步动作，智能体必须依赖历史观测（如"球被放在哪个杯子下面"）。[[World Action Model|WAM]] 通过联合建模视觉预测与动作预测来处理此类任务，但面临记忆效率困境。

### 现有方法的局限

- **滑动窗口记忆**（如 FastWAM）：推理高效，但丢失超出窗口的长程上下文，在记忆依赖任务中失效（RMBench 仅 5.9%）
- **全历史 KV Cache**（如 LingBot-VA）：保留完整上下文，但 KV Cache 规模随轨迹长度 $N$ 线性增长（$O(NL)$），推理延迟和显存占用不可接受
- **RNN/TTT（Test-Time Training）**：不能充分保留必要的历史语境

### 本文的动机

存在一个直觉：历史信息并非等价——**近期帧**提供精细的局部控制信息，**事件边界**（任务开始时刻）提供任务语义锚点，**中间历史**可以被高度压缩。三种不同粒度的记忆可以组合互补，兼顾效率与完整性。

---

## 方法详解

### 模型架构

![Figure 2: MemoryWAM Architecture](https://arxiv.org/html/2606.20562/x2.png)

**说明**: MemoryWAM 采用 [[Mixture-of-Transformers|MoT]] 架构，包含一个 [[Diffusion Transformer|Video DiT]]（5B 参数）和一个 [[Diffusion Transformer|Action DiT]]（1B 参数）。训练时视频预测提供密集监督；推理时不需要生成视频，直接利用持久记忆 KV Cache 生成动作。

MemoryWAM 采用 **[[Mixture-of-Transformers|MoT]] + [[Diffusion Transformer|DiT]]** 架构：

- **输入**: 语言指令 $l$ + 当前及历史观测 $o_{t-N:t}$ + 混合记忆 KV Cache $\mathcal{C}^v_{\leq t}$
- **Video DiT Backbone**: 基于 Wan2.2-TI2V，隐层维度 3072，30 个 Transformer Block，24 头（头维度 128）
- **Action DiT**: 隐层维度 1024，30 个 Transformer Block，24 头（头维度 128）
- **核心模块**: [[Hybrid Memory|混合记忆]] + [[Gist Token|Gist 压缩 Token]] + [[Causal Attention|因果注意力]]
- **输出**: [[Action Chunking|动作块]] $a_{t:t+h-1}$（动作视界 $h=16$）
- **总参数**: ~6B（Video DiT 5B + Action DiT 1B）

### 核心模块

#### 模块1: 混合记忆（Hybrid Memory）

**设计动机**: 利用[[KV Cache]]的持久化特性，将不同粒度的历史信息分别以最合适的方式保存，避免全量存储的显存爆炸。

**三组件结构**:

1. **短期记忆（Short-term Memory）**: 最近 $N_\text{recent}=4$ 帧的完整视觉 Token，保留精细的局部控制上下文
2. **锚帧记忆（Anchor Frame Memory）**: 任务开始时刻的 $N_\text{init}=2$ 帧，含完整视觉 Token，捕获任务初始场景语义（事件边界）
3. **Gist 记忆（Gist Memory）**: 每帧 $M_v=8$ 个可学习[[Gist Token]]，压缩长程中间历史；压缩比 $d = L/M = 120/8 = 15$

**具体实现**:
- 使用[[因果注意力|Causal Attention]]确保历史帧对当前帧可见
- Gist Token 在每帧生成后永久保留（不被驱逐）
- 推理时驱逐策略：旧的普通帧被移出短期窗口，其 Gist Token 转入 Gist 记忆池

#### 模块2: 注意力掩码设计（Attention Mask）

**设计动机**: 控制不同记忆组件之间的信息流向，使各组件发挥各自优势。

**具体实现**（以 Figure 3 为例，三帧场景）:

- 干净视频帧 $f$ 可看到所有历史帧（短期 + 锚帧 + Gist）
- Gist Token $g$ 通过[[交叉注意力|Cross-Attention]]聚合对应帧的信息
- 待去噪动作 $a$ 通过[[交叉注意力|Cross-Attention]]查询当前和历史 KV Cache

---

## 关键公式

### 公式1: [[Markov Policy|短视野策略]]

$$
a_t = \pi_\text{short}(o_{t-N:t},\, l)
$$

**含义**: 传统短视野策略仅依赖近 $N$ 帧观测，无法处理非马尔可夫任务。

**符号说明**:
- $a_t$: 时刻 $t$ 的动作
- $o_{t-N:t}$: 近 $N$ 帧观测序列
- $l$: 语言指令

---

### 公式2: [[Diffusion Transformer|Video DiT 前向传播]]

$$
\mathcal{C}_t^v = \Phi_v(z_t,\, l;\, \mathcal{C}_{<t})
$$

**含义**: Video DiT 以当前帧潜变量 $z_t$ 和语言条件 $l$ 为输入，利用历史 KV Cache $\mathcal{C}_{<t}$ 生成当前帧的视觉 KV Cache。

**符号说明**:
- $\mathcal{C}_t^v$: 第 $t$ 帧的视觉 KV Cache
- $\Phi_v$: Video DiT 参数
- $z_t$: 第 $t$ 帧的潜变量表示
- $\mathcal{C}_{<t}$: $t$ 时刻之前的所有 KV Cache

---

### 公式3: [[Action Chunking|Action DiT 预测]]

$$
a_{t:t+h-1} = \Phi_a(x_\tau^a,\, l;\, \mathcal{C}^v_{\leq t})
$$

**含义**: Action DiT 以带噪动作 $x_\tau^a$ 和完整记忆 KV Cache 为条件，去噪生成未来 $h$ 步动作块。

**符号说明**:
- $x_\tau^a$: 噪声水平 $\tau$ 下的动作噪声
- $\mathcal{C}^v_{\leq t}$: 包含当前帧在内的完整混合记忆 KV Cache
- $h=16$: 动作视界长度

---

### 公式4: [[Hybrid Memory|混合记忆 KV Cache]]

$$
\mathcal{C}^v_{\leq t} = \mathcal{C}^v_\text{short} \cup \mathcal{C}^v_\text{anchor} \cup \mathcal{C}^v_\text{gist}
$$

**含义**: 统一记忆 Cache 是三个子记忆的并集，分别负责短程、事件边界和长程历史。

**符号说明**:
- $\mathcal{C}^v_\text{short}$: 近期滑动窗口 KV Cache（$N_\text{recent}=4$ 帧）
- $\mathcal{C}^v_\text{anchor}$: 事件边界锚帧 KV Cache（$N_\text{init}=2$ 帧）
- $\mathcal{C}^v_\text{gist}$: Gist Token 压缩的长程历史 KV Cache

---

### 公式5: [[KV Cache|全历史记忆复杂度]]

$$
|\mathcal{C}^v_\text{full}| = O(N \cdot L)
$$

**含义**: 全历史 KV Cache 随轨迹帧数 $N$ 和每帧 Token 数 $L$ 线性增长，不可扩展。

**符号说明**:
- $N$: 轨迹总帧数
- $L=120$: 每帧视觉 Token 数

---

### 公式6: [[Gist Token|Gist 记忆复杂度]]

$$
|\mathcal{C}^v_\text{gist}| = O(N \cdot M) = O\!\left(\frac{NL}{d}\right)
$$

**含义**: Gist 记忆将每帧 $L$ 个 Token 压缩为 $M$ 个，实现 $d$ 倍压缩，使长程历史的存储代价可控。

**符号说明**:
- $M=8$: 每帧 Gist Token 数
- $d = L/M = 120/8 = 15$: 压缩比

---

### 公式7: [[Action Chunking|混合记忆条件动作生成]]

$$
a_{t:t+h-1} = \Phi_a\!\left(x_\tau^a,\, l;\; \mathcal{C}^v_\text{short} \cup \mathcal{C}^v_\text{anchor} \cup \mathcal{C}^v_\text{gist}\right)
$$

**含义**: 最终推理公式，动作生成同时依赖三种粒度的记忆，兼顾精细控制和长程语义。

---

### 公式8: [[Flow Matching|训练损失函数]]

$$
\mathcal{L} = \lambda_v \cdot \mathcal{L}_\text{video} + \lambda_a \cdot \mathcal{L}_\text{action}
$$

**含义**: 联合训练损失，视频预测损失提供视觉动力学的密集监督，动作损失直接优化策略质量。

**符号说明**:
- $\lambda_v = \lambda_a = 1.0$: 视频与动作损失的等权重系数
- $\mathcal{L}_\text{video}$: [[Flow Matching|Flow Matching]] 视频去噪损失
- $\mathcal{L}_\text{action}$: Flow Matching 动作去噪损失

---

## 关键图表

### Figure 1: Overview — WAM 记忆机制对比

![Figure 1](https://arxiv.org/html/2606.20562/x1.png)

**说明**: 展示三种 WAM 记忆策略的对比。滑动窗口高效但健忘；全历史 KV Cache 完整但代价线性增长；MemoryWAM 通过混合记忆（近期帧 + 锚帧 + Gist Token）实现效率与完整性的统一。

---

### Figure 2: MemoryWAM 模型架构

![Figure 2](https://arxiv.org/html/2606.20562/x2.png)

**说明**: [[Mixture-of-Transformers|MoT]] 架构细节。Video DiT 与 Action DiT 共享 KV Cache 接口；训练时视频预测提供密集监督，推理时跳过视频生成。Gist Token 作为可学习参数嵌入历史帧的压缩表示。

---

### Figure 3: 注意力掩码设计

![Figure 3](https://arxiv.org/html/2606.20562/x3.png)

**说明**: 以三帧（一个锚帧 + 一个近期帧）为例，展示 $f$（干净帧）、$g$（Gist Token）、$a$（动作）之间的[[因果注意力]]掩码结构。不同记忆组件的注意力范围差异化设计。

---

### Figure 4: 记忆机制对比（延迟 / 显存 / 成功率）

![Figure 4](https://arxiv.org/html/2606.20562/x4.png)

**说明**: 对比全注意力、[[Test-Time Training|TTT]]、[[Recurrent Neural Network|RNN]] 和混合记忆在推理延迟（单次推理时间）、GPU 显存和成功率三个维度的表现。MemoryWAM 在保持接近全注意力成功率的同时，显著降低延迟和显存开销。

---

### Figure 5: 真实世界任务

![Figure 5](https://arxiv.org/html/2606.20562/x5.png)

**说明**: Shell Game（魔术杯游戏，需记住球在哪个杯子下）和 Look and Press（需记忆指示物位置后按下对应按钮）两个非马尔可夫真实任务的演示。

---

### Figure 6: 硬件平台

![Figure 6](https://arxiv.org/html/2606.20562/x6.png)

**说明**: 真实实验平台为配备 RealSense D455 相机的双臂机器人系统（ARX 双臂）。

---

### Table 1: RMBench 仿真实验结果（成功率 %）

| 任务 | π₀.₅ | FastWAM | LingBot-VA | **MemoryWAM** |
|------|------|---------|-----------|---------------|
| Observe and Pick Up | 9% | 0% | 13% | **27%** |
| Rearrange Blocks | 13% | 0% | 100% | **100%** |
| Put Back Block | 11% | 0% | 100% | **100%** |
| Swap Blocks | 24% | 0% | 99% | **100%** |
| Swap T | 15% | 7% | 88% | **94%** |
| Battery Try | 16% | 20% | 41% | **41%** |
| Blocks Ranking Try | 6% | 26% | 100% | **100%** |
| Cover Blocks | 0% | 0% | 79% | **98%** |
| Press Button | 0% | 0% | 84% | **87%** |
| **平均** | **10.4%** | **5.9%** | **78.2%** | **83.0%** |

**关键发现**: MemoryWAM 在 9 个任务中均优于或持平于 LingBot-VA（全历史方法），同时推理效率远高于后者。FastWAM 因滑动窗口丢失长程上下文，在大多数任务上完全失败。

---

### Table 2: 真实世界实验（成功次数 / 总次数）

| 任务 | π₀.₅ | LingBot-VA | **MemoryWAM** |
|------|------|-----------|---------------|
| Shell Game | 5/20 | 13/20 | **18/20** |
| Look and Press | 0/20 | 14/20 | **15/20** |

**关键发现**: MemoryWAM 在两个真实世界非马尔可夫任务上均超越 LingBot-VA，Shell Game 提升显著（18 vs 13）。

---

### Table 3: 消融实验（Cover Blocks / Press Button / 平均）

| 配置 | Cover Blocks | Press Button | 平均 |
|------|-------------|-------------|------|
| w/o Anchor Frames | 58% | 90% | 74.0% |
| w/o Gist Tokens | 75% | 5% | **40.0%** |
| w/o Sliding Window | 96% | 69% | 82.5% |
| Full Attention（上界） | 96% | 87% | 91.5% |
| **MemoryWAM（完整）** | **98%** | **87%** | **92.5%** |

**关键发现**: Gist Token 是最关键组件——去掉后 Press Button 从 87% 骤降至 5%，说明长程历史压缩对记忆依赖任务不可缺少。MemoryWAM 完整模型甚至以 92.5% 超越全注意力基线（91.5%）。

---

## 实验

### 数据集 / 评测平台

| 平台 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[RMBench]] | 9 个任务 | 长视野、记忆依赖、双臂操控 | 仿真评测 |
| ARX 双臂机器人 | 2 个任务 | Shell Game、Look and Press | 真实世界评测 |

### 实现细节

- **基础模型**: Wan2.2-TI2V（Video DiT 5B 参数）
- **优化器**: AdamW，学习率 $2 \times 10^{-4}$，权重衰减 0.01
- **Batch Size**: 每 GPU 1 个样本，bfloat16 精度
- **硬件**: 8 × GPU（型号未指定）
- **噪声调度**: Shifted Logit-Normal，视频 shift=5.0，动作 shift=1.0，1000 时间步
- **数据增强**: 与[[Gaussian Noise|高斯噪声]]的线性混合，比例 $\in [0,1]$
- **推理步数**: 50 步（仿真），10 步（真实世界）
- **控制频率**: 10 Hz，帧间延迟约 0.3 秒
- **推理采样**: 从 16 步预测中下采样 4 帧，索引 $\{3, 7, 11, 15\}$

### 记忆超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| $N_\text{recent}$ | 4 | 短期窗口帧数 |
| $N_\text{init}$ | 2 | 锚帧数量 |
| $M_v$ | 8 | 每帧 Gist Token 数 |
| $L$ | 120 | 每帧视觉 Token 数 |
| $d$ | 15 | 压缩比（$L/M$） |
| $h$ | 16 | 动作视界 |

---

## 批判性思考

### 优点

1. **清晰的效率-性能平衡**: 三组件混合记忆设计思路简洁，在不牺牲性能的情况下将 KV Cache 从 $O(NL)$ 降至 $O(NL/d)$，工程可行性强
2. **强实验支撑**: 消融实验设计合理，精确定位了每个记忆组件的贡献；真实机器人实验验证了方法的实际可行性
3. **推理时不生成视频**: 训练时视频预测提供密集监督，推理时跳过，兼顾了训练质量和推理效率

### 局限性

1. **Gist Token 压缩信息损失**: 固定的 $M=8$ 个 Gist Token 能否充分表达任意长度轨迹的语义未经充分验证，极长轨迹场景下可能退化
2. **对初始帧（锚帧）的强依赖**: 任务初始状态被特殊保留，但对于开放世界任务，"任务开始"并不总是明确定义
3. **未开源代码**: 可复现性受限

### 潜在改进方向

1. **自适应 Gist Token 数量**: 根据历史帧的重要性动态分配压缩 Token 数，而非固定 $M=8$
2. **多事件边界检测**: 自动检测多个任务阶段边界，动态增加锚帧，处理更复杂的多阶段任务

### 可复现性评估

- [ ] 代码开源（未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、优化器均有描述）
- [ ] 数据集可获取（RMBench 未确认公开状态）

---

## 关联笔记

### 基于

- [[World Action Model|WAM]]: 联合视觉预测与动作预测的框架，MemoryWAM 在此基础上提出记忆增强
- [[Wan2.2]]: MemoryWAM 的 Video DiT 骨干网络
- [[Flow Matching]]: 训练时使用的生成模型框架

### 对比

- [[LingBot-VA]]: 全历史 KV Cache WAM，性能强但推理代价高；MemoryWAM 的主要对比基线
- [[FastWAM]]: 滑动窗口 WAM，推理高效但长程上下文丢失
- [[π0.5]]: VLA 基线，在记忆依赖任务上表现较差

### 方法相关

- [[Gist Token]]: 核心压缩机制，将长程历史压缩为少量可学习 Token
- [[KV Cache]]: 持久记忆的底层实现机制
- [[Mixture-of-Transformers]]: MoT 架构，Video DiT 与 Action DiT 解耦
- [[Causal Attention|因果注意力]]: 确保历史帧对当前帧的单向信息流
- [[Action Chunking]]: 动作块预测，每次预测 $h=16$ 步

### 硬件/数据相关

- [[RMBench]]: 记忆依赖机器人操控基准，9 个双臂任务
- [[ARX 双臂机器人]]: 真实实验平台

---

## 速查卡片

> [!summary] MemoryWAM: Efficient World Action Modeling with Persistent Memory
> - **核心**: 混合记忆（短期帧 + 锚帧 + Gist Token）解决 WAM 的效率-记忆困境
> - **方法**: KV Cache 分三级：近期 4 帧完整保留 + 2 个初始锚帧 + 每帧 8 个 Gist Token 压缩长程历史
> - **结果**: RMBench 83.0% vs LingBot-VA 78.2%；真实 Shell Game 18/20 vs 13/20
> - **代码**: 未公开（项目页: https://yangsizhe.github.io/MemoryWAM/）

---

*笔记创建时间: 2026-06-19*
