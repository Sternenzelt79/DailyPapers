---
title: "World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis"
method_name: "WLA"
authors: [Yi Yang, Zhihong Liu, Siqi Kou, Yiyang Chen, Yanzhe Hu, Jianbo Zhou, Boyuan Zhao, Zhijie Wei, Xiao Xia, Xueqi Li, Pengfei Liu, Zhijie Deng]
year: 2026
venue: arXiv
tags: [world-model, vla, autoregressive, long-horizon, test-time-scaling, cross-embodiment, robot-manipulation]
zotero_collection: 3-Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.05979v1
created: 2026-06-05
---

# 论文笔记：World-Language-Action Model (WLA)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | SJTU DENG Lab |
| 日期 | June 2026 |
| 项目主页 | - |
| 对比基线 | [[Fast-WAM]], [[LingBot]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05979) / [Code](https://github.com/SJTU-DENG-Lab/WLA) |

---

## 一句话总结

> WLA 是首个将世界建模、语言推理与动作生成统一于单一[[自回归策略|自回归]]框架的具身基础模型，仅 2B 激活参数即超越规模更大的基线，推理延迟低至 40ms。

---

## 核心贡献

1. **统一三流输出**: 同一个[[自回归Transformer]]骨干同时预测文本子任务、子目标图像和可执行机器人动作，无需独立规划模块
2. **隐式世界建模**: [[World Expert]]通过训练期共享 VAE 特征影响动作生成，推理时可完全移除，兼顾世界建模能力与低延迟推理
3. **测试时扩展（TTS）**: 采样多条动作候选 → 世界专家想象未来帧 → [[价值模型]]打分 → 执行最优轨迹，无需重新训练即可提升性能

---

## 问题背景

### 要解决的问题

长时程机器人操作需要高层语义规划（"放好杯子"由哪些步骤组成？）与低层精细动作控制的紧密配合，单纯的 [[Vision-Language-Action Model|VLA]] 缺乏显式世界建模，而 [[扩散世界模型|WAM]] 缺乏语言生成能力。

### 现有方法的局限

- **VLA 模型**：擅长语言条件控制，但无法预测未来视觉状态，难以做测试时推理验证
- **WAM 模型**（如 Fast-WAM）：使用双向扩散 Transformer 建模物理动态，但不具备语言生成能力，高层规划依赖外部模块
- **二阶段方法**：规划与执行分离，误差累积，实时性差

### 本文的动机

用统一的[[自回归策略|自回归]]框架将三种能力融合：语言推理（文本子任务）提供语义上下文，世界建模（子目标图像）验证动作合理性，动作生成直接输出可执行指令。训练时三者互相增强，推理时按需开关。

---

## 方法详解

### 模型架构

WLA 采用**三专家自回归架构**：

- **输入**: 语言指令 $\ell$ + 当前观测 $\mathbf{o}_t$ + 历史帧 $\mathbf{o}_{t-h}$ + 机器人状态 $\mathbf{q}_t$ + 记忆缓冲区 $\mathcal{M}$
- **Backbone**: RynnBrain-2B（2.1B 参数），[[因果注意力]]自回归 Transformer
- **核心模块**: [[Meta-Query|元查询]] $\mathbf{Q}$（64 个）聚合上下文，作为物理动态表示 $\mathbf{h}_t$ 传递给后续专家
- **World Expert**: SANA-600M（含 VAE 共 900M），预测未来 VAE 潜变量特征
- **Action Expert**: 390M [[Flow Matching|流匹配]]头，将 $\mathbf{h}_t$ 转化为可执行动作块
- **总激活参数**: ~2B（推理时 World Expert 可关闭）

```
文本指令 + 图像 + 状态
       ↓
  AR Backbone (2.1B)
  ├── 文本子任务预测 S_t
  ├── 元查询 → 物理动态 h_t
       ├── World Expert → 未来图像 o_{t+n}  [仅训练 / TTS]
       └── Action Expert → 动作块 a_{t:t+n}
```

### 核心模块

#### 模块1: 文本意图学习（Textual Intention Learning）

**设计动机**: 利用[[大语言模型|LLM]]的语义推理能力，将长时程任务分解为连续子任务窗口，为物理动态建模提供高层语境

**具体实现**:
- 维护记忆缓冲区 $\mathcal{M}$，存储历史子任务和观测，支持长时程上下文
- 预测与未来动作 horizon 对齐的子任务区间 $\mathcal{S}_t$
- 子任务文本通过[[因果注意力]]注入后续动态建模

#### 模块2: 物理动态建模（Physical Dynamics Modeling）

**设计动机**: 将感知信息压缩为紧凑的潜在动作表示 $\mathbf{h}_t$，同时携带足够的物理先验以引导动作生成和世界模型预测

**具体实现**:
- 64 个可学习[[Meta-Query|元查询]]通过[[因果注意力]]从所有上下文中聚合信息
- $\mathbf{h}_t$ 同时用于驱动 World Expert（训练期）和 Action Expert（推理期）
- 世界专家预测 VAE 潜变量而非像素，避免直接在高维像素空间建模

#### 模块3: 测试时扩展（Test-Time Scaling，TTS）

**设计动机**: 利用世界模型作为价值评估器，在推理阶段提升动作质量，无需额外训练数据

**具体实现**:
- 采样 $K$ 条动作候选轨迹
- World Expert 为每条候选预测想象的未来视觉帧
- [[价值模型]]（训练时以 $v_t = y \cdot \gamma^{T-t}$ 为标签）对想象结果打分
- 选择得分最高的动作序列执行

---

## 关键公式

### 公式1: [[自回归策略|文本意图预测]]

$$
\mathcal{S}_{t} = f(\mathbf{o}_{t-h},\, \mathbf{o}_{t},\, \ell,\, \mathcal{M})
$$

**含义**: 从历史观测、当前观测、语言指令和记忆缓冲区中预测当前时步的子任务区间

**符号说明**:
- $\mathcal{S}_t$: 时刻 $t$ 的子任务文本区间（与未来动作 horizon 对齐）
- $\mathbf{o}_{t-h}$: 历史观测帧
- $\mathbf{o}_t$: 当前观测
- $\ell$: 自然语言任务指令
- $\mathcal{M}$: 记忆缓冲区，维护长时程历史上下文

### 公式2: [[Meta-Query|物理动态聚合]]

$$
\mathbf{h}_{t} = f(\mathbf{o}_{t-h},\, \mathbf{o}_{t},\, \ell,\, \mathcal{M},\, \mathcal{S}_{t},\, \mathbf{Q})
$$

**含义**: 元查询 $\mathbf{Q}$ 通过因果注意力聚合所有上下文，生成物理动态表示 $\mathbf{h}_t$

**符号说明**:
- $\mathbf{h}_t$: 物理动态表示（潜在动作），用于驱动 World Expert 和 Action Expert
- $\mathbf{Q}$: 64 个可学习元查询（meta-queries）
- $\mathcal{S}_t$: 文本子任务，提供语义上下文

### 公式3: [[扩散世界模型|世界建模预测]]

$$
\mathbf{o}_{t+n} = f_{wm}(\mathbf{h}_{t},\, \mathbf{o}_{t})
$$

**含义**: World Expert 基于物理动态和当前帧预测未来帧（实际预测 VAE 潜变量，非像素）

**符号说明**:
- $\mathbf{o}_{t+n}$: 预测的未来视觉状态（第 $n$ 步）
- $f_{wm}$: SANA-600M 世界专家网络

### 公式4: [[Flow Matching|动作生成]]

$$
\mathbf{a}_{t:t+n} = f_{act}(\mathbf{h}_{t},\, \mathbf{q}_{t})
$$

**含义**: Action Expert 以物理动态和当前机器人关节状态为条件，通过流匹配生成动作块

**符号说明**:
- $\mathbf{a}_{t:t+n}$: 长度为 $n$ 的动作块（LIBERO: chunk=8，其余: chunk=32）
- $\mathbf{q}_t$: 当前机器人关节状态（本体感知）
- $f_{act}$: 390M 流匹配动作头

### 公式5: [[多任务学习|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{act} + \alpha\,\mathcal{L}_{wm} + \beta\,\mathcal{L}_{lang}
$$

**含义**: 三个损失联合优化，动作预测为主损失，世界建模和语言生成为辅助损失

**符号说明**:
- $\mathcal{L}_{act}$: 动作预测损失（流匹配回归）
- $\mathcal{L}_{wm}$: 世界建模损失（VAE 潜变量重建）
- $\mathcal{L}_{lang}$: 语言生成损失（子任务文本交叉熵）
- $\alpha = 0.1$: 世界建模损失权重
- $\beta = 0.005$: 语言损失权重

### 公式6: [[价值模型|TTS 价值标签]]

$$
v_t = y \cdot \gamma^{T - t}
$$

**含义**: 测试时扩展中，以折扣成功标签训练价值模型，用于对想象未来轨迹打分

**符号说明**:
- $v_t$: 时刻 $t$ 的价值标签
- $y \in \{0, 1\}$: 回合成功标志
- $\gamma$: 折扣因子
- $T$: 回合总时步数

---

## 关键图表

### Figure 1: 架构对比（VLA vs WAM vs WLA）

![[WLA_fig1.png]]

**说明**: 对比三类模型设计。VLA 直接从视觉语言输入预测动作；WAM 增加世界建模分支但缺乏语言生成；WLA 在单一[[自回归策略|自回归]]框架中同时输出文本子任务、子目标图像和动作，并标注了各方法的参数规模与性能。

### Figure 2: WLA 系统组件与 TTS 流程

![[WLA_fig2.png]]

**说明**: (a) 三专家架构详图：AR Backbone 通过[[Meta-Query|元查询]]产生 $\mathbf{h}_t$，分别送入 World Expert（SANA-600M）和 Action Expert（390M 流匹配头）；(b) TTS 模式执行流程：采样 $K$ 条候选 → 世界专家预测未来帧 → 价值模型打分 → 执行最优轨迹。

### Figure 3: 真实机器人实验

![[WLA_fig3.png]]

**说明**: (a) 四个长时程任务（Unscrew Cap、Pack Object、Stack Cup、Dispose Trash）示例；(b) OOD 物体与场景示例；(c) WLA-0 与 π₀.₅、Motus 成功次数对比（无需具身预训练即达到可比性能）；(d) 推理延迟和任务完成时间对比（WLA-0 推理速度约 40× 快于 Motus，Stack Cup 完成时间减半）。

### Figure 4: Beat Block Hammer 任务可视化

![[WLA_fig4.png]]

**说明**: 三种训练设置（Seen-Action / ++Video / ++Same-Emb.）下的执行序列对比，展示模型学到的"敲击"动作模式及跨体现迁移效果。

### Figure 5: 跨体现视频学习实验

![[WLA_fig5.png]]

**说明**: 五个未见 RoboTwin 2.0 任务与对应的人类视角视频示例，用于验证同体现与跨体现视频数据的迁移效果。

### Figure 6: RMBench — Battery Try 任务

![Figure 6](https://arxiv.org/html/2606.05979v1/2606.05979v1/figures/battery_try.png)

**说明**: Battery Try 长时程任务的代表性帧序列与子任务分解（WLA-0 在该任务达到 45% 成功率，远超 Fast-WAM 的 16%）。

### Figure 7: RMBench — Blocks Ranking Try 任务

![Figure 7](https://arxiv.org/html/2606.05979v1/2606.05979v1/figures/blocks_ranking_try.png)

**说明**: 积木排序任务，需理解颜色/数字语义，包含 9 个子任务指令，考验语言推理能力。

### Figure 8: RMBench — Cover Blocks 任务

![Figure 8](https://arxiv.org/html/2606.05979v1/2606.05979v1/figures/cover_blocks.png)

**说明**: 覆盖积木任务，6 步子任务序列，WLA-0 成功率 84%（消融去掉语言损失后骤降至 18%，证明语言推理的关键作用）。

### Figure 9: RMBench — Press Button 任务

![Figure 9](https://arxiv.org/html/2606.05979v1/2606.05979v1/figures/press_button.png)

**说明**: 计数按钮任务，10 步子任务分解，要求模型正确计数并顺序按压。WLA-0 成功率 74%。

### Figure 10: World Expert 预测质量

![Figure 10](https://arxiv.org/html/2606.05979v1/2606.05979v1/x6.png)

**说明**: 预测未来图像与真实图像的对比，验证 World Expert 能准确预测关键物体位置和场景变化，为 TTS 打分提供可靠依据。

### Table 1: 仿真基准对比（RoboTwin 2.0 & LIBERO）

| 方法 | 激活参数 | 具身预训练 | RoboTwin 2.0 Clean | LIBERO Avg |
|------|----------|-----------|-------------------|-----------|
| WLA-0 | **2B** | ✗ | **92.94%** | **98.6%** |
| WLA-0 −$\mathcal{L}_{wm}$ | 2B | ✗ | 90.98% | 97.9% |
| WLA-0 ++TTS | 2B | ✗ | — | 98.9% |
| Fast-WAM | 6B | ✗ | 91.88% | 97.6% |
| LingBot-VA | 5B | ✓ | 92.90% | 98.5% |

**关键发现**: WLA-0 仅 2B 激活参数，无需具身预训练，在两个基准均超越 3× 规模的基线，且去掉世界建模损失（$-\mathcal{L}_{wm}$）后性能明显下降，证明世界建模对动作质量的正向影响。

### Table 2: RMBench 长时程任务对比

| 方法 | Battery Try | Blocks Ranking | Cover Blocks | Press Button | 平均 |
|------|------------|----------------|--------------|--------------|------|
| WLA-0 | **45%** | 23% | **84%** | **74%** | **56.5%** |
| WLA-0 −$\mathcal{L}_{lang}$ | 38% | 12% | 18% | 1% | 17.3% |
| Fast-WAM | 16% | **37%** | 0% | 0% | 13.3% |
| Mem-0 | 28% | 18% | 68% | 0% | 28.5% |

**关键发现**: 去掉语言损失（$-\mathcal{L}_{lang}$）后 WLA-0 平均成功率从 56.5% 骤降至 17.3%，说明文本子任务预测对长时程推理至关重要。Fast-WAM 在需要精确计数/排序时完全失败（0%）。

### Table 3: 跨体现视频学习（五个未见任务）

| 未见任务 | Seen-Action | ++Video | ++Same-Emb. | ++Cross-Emb. |
|----------|------------|---------|-------------|--------------|
| Beat Block Hammer | 1/0 | 1/0 | 12/6 | 5/3 |
| Move Playingcard | 2/0 | 0/3 | 42/28 | 1/0 |
| Pick Diverse Bottles | 7/10 | 10/8 | 32/29 | **39/35** |
| Place Object Basket | 3/5 | 0/1 | 30/34 | **45/47** |
| Stack Bowls Three | 52/43 | 48/51 | 56/53 | 54/52 |

*格式：Clean/Randomized 成功率*

**关键发现**: 同体现视频（++Same-Emb.）整体最优；跨体现视频（++Cross-Emb.）在部分任务（Pick Diverse Bottles、Place Object Basket）超越同体现，表明跨体现学习具有选择性迁移潜力。人类视角视频（++Video）因 domain gap 几乎无效。

### Table 4: 消融实验——世界建模监督粒度

| 世界建模设置 | LIBERO Avg |
|------------|-----------|
| 单帧未来预测（默认） | **98.2%** |
| 多帧未来预测 | 94.2% |

**关键发现**: 单帧监督优于多帧，过多的世界建模监督反而干扰动作学习。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 50 任务 | 仿真，含 Clean/Randomized 设置 | 主要训练/测试 |
| LIBERO | 4 套件共 100 任务 | 仿真，语言条件操作 | 泛化测试 |
| RMBench | 4 长时程任务 | 仿真，需多步推理 | 长时程评估 |
| 真实机器人（AgilexRobotics Piper） | 4 任务 | 真实场景 + OOD 设置 | 真实世界评估 |

### 实现细节

- **Backbone**: RynnBrain-2B（2.1B 参数），因果注意力自回归 Transformer
- **World Expert**: SANA-600M（含 VAE 共 900M 参数）
- **Action Expert**: 390M 流匹配头
- **优化器**: AdamW，余弦调度，基础学习率 $5\times10^{-5}$，最小学习率 $5\times10^{-6}$
- **动作块大小**: 8（LIBERO），32（其余任务）
- **元查询数量**: 64
- **训练框架**: DeepSpeed
- **损失权重**: $\alpha=0.1$（世界建模），$\beta=0.005$（语言）

### 推理加速技术

三项技术联合将延迟从 ~116ms 压缩至 <40ms（NVIDIA RTX 5090）：

1. **CUDA Graph Capture**: 消除每步 Python dispatch 开销
2. **算子融合（Operator Fusion）**: 自定义 Triton kernel 融合相邻算子
3. **预计算与 KV 缓存**: 预计算 token embeddings、RoPE 表和[[交叉注意力]] K/V 张量

---

## 批判性思考

### 优点

1. **架构优雅**: 三专家设计实现了"训练时充分利用世界建模，推理时零代价移除"，无 trade-off
2. **语言损失价值显著**: 消融实验清晰证明语言子任务预测对长时程任务成功率的巨大贡献（+39.2%）
3. **参数效率突出**: 2B 激活参数超越 5-6B 基线，工程价值高
4. **TTS 无缝集成**: 世界模型天然可用于测试时扩展，设计自洽

### 局限性

1. **真实机器人平台单一**: 仅在 AgilexRobotics Piper 上验证，跨平台泛化性未知
2. **视频迁移效果不稳定**: 人类视角视频几乎无效，跨体现迁移仅在特定任务有效
3. **子任务标注依赖**: 训练需要子任务分解的文本标注，标注成本未详细讨论

### 潜在改进方向

1. 探索自动子任务发现，减少人工标注依赖
2. 扩展到双臂/移动操作等更复杂体现形式
3. 更大规模的跨体现预训练以提升零样本迁移

### 可复现性评估

- [x] 代码开源（GitHub: SJTU-DENG-Lab/WLA）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（Appendix 中有加速技术详述）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均为公开数据集）

---

## 关联笔记

### 基于

- [[Vision-Language-Action Model]]: WLA 继承 VLA 的语言条件控制框架
- [[扩散世界模型]]: World Expert 承自 WAM 系列的视觉动态建模思路
- [[Flow Matching]]: Action Expert 使用流匹配生成连续动作

### 对比

- [[Fast-WAM]]: 6B WAM 基线，缺乏语言生成，长时程任务失败
- [[Pi0.5]]: 真实机器人实验的主要竞争对手之一
- [[自回归策略]]: WLA 选择自回归而非扩散 Transformer 作为统一骨干

### 方法相关

- [[Meta-Query]]: 核心信息聚合机制，产生物理动态表示
- [[Action Chunking]]: 动作专家输出 chunk 动作，平衡响应速度与精度
- [[跨体现学习]]: Table 3 探索同/跨体现视频迁移
- [[价值模型]]: TTS 模式中对想象轨迹打分

### 硬件/数据相关

- [[AgilexRobotics Piper]]: 真实机器人实验平台
- [[RoboTwin 2.0]]: 主要仿真训练/测试环境

---

## 速查卡片

> [!summary] World-Language-Action Model (WLA)
> - **核心**: 统一自回归框架同时预测文本子任务、子目标图像和动作，三者互相增强
> - **方法**: AR Backbone（2.1B）+ World Expert（SANA-600M，推理可关闭）+ Action Expert（390M 流匹配头），联合损失 $\mathcal{L} = \mathcal{L}_{act} + 0.1\mathcal{L}_{wm} + 0.005\mathcal{L}_{lang}$
> - **结果**: RoboTwin 2.0 92.94%、LIBERO 98.6%（均超 5-6B 基线），RMBench 长时程 56.5% vs Fast-WAM 13.3%，推理延迟 <40ms
> - **代码**: https://github.com/SJTU-DENG-Lab/WLA

---

*笔记创建时间: 2026-06-05*
