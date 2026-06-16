---
title: "Qwen-RobotWorld Technical Report: Unifying Embodied World Modeling through Language-Conditioned Video Generation"
method_name: "QwenRobotWorld"
authors: [Jie Zhang, Xiaoyue Chen, Anzhe Chen, Chenxu Lv, Deqing Li, Gengze Zhou, Hang Yin, Haoqi Yuan, Haoyang Li, Jiahao Li, Jiazhao Zhang, Jingren Zhou, Kaiyuan Gao, Kun Yan, Lihan Jiang, Ningyuan Tang, Pei Lin, Qihang Peng, Shengming Yin, Tianhe Wu, Tianyi Yan, Xiao Xu, Yan Shu, Yanran Zhang, Ye Wang, Yi Wang, Yilei Chen, Yixian Xu, Yiyang Huang, Yuxiang Chen, Zekai Zhang, Zhendong Wang, Zhixing Lei, Zhixuan Liang, Zihao Liu, Zikai Zhou, Xiong-Hui Chen, Chenfei Wu]
year: 2026
venue: arXiv
tags: [world-model, video-generation, embodied-ai, language-conditioned, diffusion-transformer, robotic-manipulation]
zotero_collection: Robotics/World Model
image_source: online
created: 2026-06-16
---

# 论文笔记：Qwen-RobotWorld Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Alibaba Group (Qwen Team) |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[DreamGen]], [[EWMBench]], [[WorldModelBench]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17030) / Code 暂未公开 |

---

## 一句话总结

> Qwen-RobotWorld 以自然语言为统一动作接口，通过60层双流扩散变换器预测跨具身平台的未来视觉轨迹，在多个具身世界模型基准上达到最优性能。

---

## 核心贡献

1. **Double-Stream MMDiT 架构**: 提出60层双流[[多模态扩散变换器|MMDiT]]，通过逐层联合注意力机制将冻结的[[Qwen2.5-VL]]语义表征与[[视频VAE|Video-VAE]]潜在表示深度融合，实现语言条件的具身视频生成。
2. **具身世界知识语料库 (EWK)**: 构建包含860万视频-文本对（超过2亿帧）的大规模语料库，覆盖20+种具身形态和500+动作类别，建立统一的动作语言映射。
3. **通用+专家渐进课程训练策略**: 采用两阶段训练——先在通用视频数据上学习视觉先验，再注入具身专业化知识，实现跨领域泛化与任务精度的兼顾。

---

## 问题背景

### 要解决的问题

具身智能系统需要一种统一的世界模型，能够预测不同具身平台（机器人操作、自动驾驶、室内导航、人-机迁移）在语言指令下的未来视觉状态。现有方法通常针对单一领域设计，缺乏跨平台的通用性。

### 现有方法的局限

- **领域割裂**: 机器人操作、自动驾驶等领域各自维护独立的世界模型，无法共享知识；
- **动作接口不统一**: 不同平台使用关节角度、速度、航向角等异构动作表示，难以统一建模；
- **数据规模受限**: 单一领域的视频数据量有限，制约了模型的泛化能力；
- **语义理解薄弱**: 现有视频生成模型缺乏对自然语言动作指令的精细理解能力。

### 本文的动机

以**自然语言**作为统一的动作接口，不同具身形态的动作都可以描述为语言指令（如"向左转90度并抓取红色方块"），从而消除异构动作表示的障碍。结合大规模多领域视频数据和强大的[[多模态大语言模型|MLLM]]语义编码，实现具身世界建模的统一化。

---

## 方法详解

### 模型架构

Qwen-RobotWorld 采用**语言条件视频扩散**架构：

- **输入**: 自然语言指令 $l$ + 当前观测帧 $o_t$（视频条件帧）
- **语言编码器**: 冻结的 [[Qwen2.5-VL]]（MLLM），提取深度语义特征
- **视觉编码器**: [[视频VAE|Video-VAE]]，将视频编码为压缩潜在表示
- **核心主干**: 60层 [[双流MMDiT|Double-Stream MMDiT]]，通过逐层[[联合注意力|Joint Attention]]融合语言与视觉流
- **输出**: 预测的未来视觉轨迹（视频序列）

### 核心模块

#### 模块1：Double-Stream MMDiT with MLLM Action Encoding

**设计动机**: 利用 [[多模态扩散变换器|MMDiT]] 的双流结构实现语言语义与视觉动态的深度耦合，避免浅层融合导致的语义漂移。

**具体实现**:
- **语言流**: 将冻结的 [[Qwen2.5-VL]] 提取的动作语义特征作为独立的Token序列
- **视觉流**: 将 [[视频VAE|Video-VAE]] 编码的视频潜在表示作为另一个Token序列
- **逐层联合注意力**: 60层[[Transformer]]中，每层都对两个流的Token进行双向注意力交互（[[联合注意力|Joint Attention]]），相比简单的交叉注意力能实现更充分的模态对齐
- **冻结MLLM**: Qwen2.5-VL的权重在训练中保持冻结，保留其丰富的世界知识和语言理解能力

#### 模块2：具身世界知识语料库 (Embodied World Knowledge, EWK)

**设计动机**: 单一领域数据不足以训练泛化的具身世界模型，需要覆盖多具身形态、多任务类型的大规模语料库。

**具体实现**:
- **规模**: 860万个视频-文本对，超过2亿帧视频
- **多样性**: 覆盖20+种具身形态（工业机器人、移动底盘、人形机器人、自动驾驶车辆、室内导航智能体等）
- **动作标注**: 500+动作类别，统一映射为自然语言描述
- **质量保证**: 视频-文本对经过动作-语言对齐过滤，确保语义一致性

#### 模块3：General+Expert 渐进课程训练策略

**设计动机**: 直接在具身数据上训练容易过拟合特定领域；先学通用视觉先验再注入专业知识，能更好地平衡泛化性和任务精度。

**具体实现**:
- **阶段一（General）**: 在大规模通用视频数据上预训练，学习视觉物理先验（运动连贯性、光照一致性、物体永久性等）
- **阶段二（Expert）**: 在EWK具身数据上微调，在保留通用先验的基础上注入动作-视觉因果关系的专业知识
- **共享语言接口**: 两阶段均以自然语言为动作条件，确保知识迁移的一致性

---

## 关键公式

### 公式1：[[扩散去噪目标|视频扩散训练目标]]

$$
\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta\left(z_t, t, c_l, c_v\right)\right\|^2\right]
$$

**含义**: 训练扩散模型 $\epsilon_\theta$ 预测在时间步 $t$ 加入的噪声 $\epsilon$，同时以语言条件 $c_l$ 和视觉条件 $c_v$ 为引导。

**符号说明**:
- $z_0$: 干净视频的VAE潜在表示
- $\epsilon \sim \mathcal{N}(0, I)$: 采样的高斯噪声
- $t$: 扩散时间步
- $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$: 加噪后的潜在表示
- $c_l$: 来自Qwen2.5-VL的语言（动作指令）语义条件
- $c_v$: 来自视频VAE的视觉条件帧特征
- $\epsilon_\theta$: 参数化的60层Double-Stream MMDiT去噪网络

### 公式2：[[联合注意力|双流联合注意力机制]]

$$
\begin{aligned}
\left[\mathbf{Q}_v, \mathbf{K}_v, \mathbf{V}_v\right] &= \mathbf{X}_v \mathbf{W}_v^{QKV} \\
\left[\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l\right] &= \mathbf{X}_l \mathbf{W}_l^{QKV} \\
\mathbf{K} &= \left[\mathbf{K}_v \| \mathbf{K}_l\right],\quad \mathbf{V} = \left[\mathbf{V}_v \| \mathbf{V}_l\right] \\
\mathbf{X}_v' &= \operatorname{softmax}\!\left(\frac{\mathbf{Q}_v \mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
\end{aligned}
$$

**含义**: 视觉流Token $\mathbf{X}_v$ 与语言流Token $\mathbf{X}_l$ 在每层中通过拼接KV实现联合注意力，使视觉特征能够直接感知语言动作语义。

**符号说明**:
- $\mathbf{X}_v$: 视觉流Token（视频VAE潜在特征）
- $\mathbf{X}_l$: 语言流Token（Qwen2.5-VL语义特征）
- $\mathbf{W}^{QKV}$: 各自流的QKV投影矩阵
- $\left[\cdot \| \cdot\right]$: 沿序列维度拼接
- $d$: 注意力头维度
- $\mathbf{X}_v'$: 融合语言信息后的视觉流输出

---

## 关键图表

> 注：本论文于2026年6月15日发布，arXiv HTML版本暂未上线，图片暂时无法通过外链获取。以下图表说明基于论文描述重建。

### Figure 1: 系统总览 / System Overview

**说明**: Qwen-RobotWorld 以自然语言为统一动作接口，接收当前观测帧和语言动作指令，通过Double-Stream MMDiT生成未来视觉轨迹。支持四大场景：机器人操作（Robotic Manipulation）、自动驾驶（Autonomous Driving）、室内导航（Indoor Navigation）、人-机迁移（Human-to-Robot Transfer）。

### Figure 2: Double-Stream MMDiT 架构 / Architecture Diagram

**说明**: 展示60层双流扩散变换器的内部结构。左侧为语言流（冻结的Qwen2.5-VL编码器输出Token），右侧为视觉流（Video-VAE编码的视频潜在Token），每层通过联合注意力（Joint Attention）进行双向交互，最终由视觉流解码为预测视频帧。

### Figure 3: EWK 数据集分布 / Dataset Distribution

**说明**: 展示具身世界知识（EWK）语料库的组成，860万视频-文本对覆盖20+具身形态（工业臂、移动机器人、自动驾驶等）和500+动作类别的分布情况。

### Figure 4: 渐进课程训练流程 / Progressive Curriculum Training

**说明**: 两阶段训练流程示意。阶段一（General）在大规模通用视频上预训练；阶段二（Expert）在EWK具身数据上持续微调，语言接口贯穿始终。

### Figure 5: 定性结果对比 / Qualitative Results

**说明**: 在机器人操作、自动驾驶、室内导航等任务上与其他开源世界模型的定性对比，展示Qwen-RobotWorld在动作一致性和语义对齐上的优势。

### Figure 6: 零样本泛化分析 / Zero-Shot Generalization on RoboTwin-IF

**说明**: 在RoboTwin-IF基准上的零样本分析结果，验证模型对未见具身平台的鲁棒泛化能力和多视角一致性。

### Table 1: EWMBench 主要结果对比

| Method | 视觉场景一致性 | 动作正确性 | 语义对齐 | 综合排名 |
|--------|--------------|-----------|---------|---------|
| 开源基线1 | - | - | - | - |
| 开源基线2 | - | - | - | - |
| **Qwen-RobotWorld** | **最优** | **最优** | **最优** | **1st** |

**表格说明**: Qwen-RobotWorld 在EWMBench三个维度（视觉场景一致性、动作正确性、语义对齐）上均达到最优，总体排名第一。具体数值待arXiv HTML版本发布后补充。

### Table 2: DreamGen Bench 对比结果

| Method | DreamGen Score | 排名 |
|--------|---------------|------|
| 开源模型群 | - | 2nd+ |
| **Qwen-RobotWorld** | **最高** | **1st** |

**表格说明**: 在DreamGen Bench上综合排名第一，验证了方法在机器人学习泛化能力评估上的优越性。

### Table 3: WorldModelBench & PBench 开源模型对比

| Method | WorldModelBench | PBench |
|--------|----------------|--------|
| 其他开源模型 | 低于本方法 | 低于本方法 |
| **Qwen-RobotWorld** | **最优（开源最佳）** | **最优（开源最佳）** |

**表格说明**: 在WorldModelBench和PBench上超越所有开源模型，达到开源最优水平。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| EWK（具身世界知识） | 860万视频-文本对，200M+帧 | 20+具身形态，500+动作类别，统一语言标注 | 阶段二微调 |
| 通用视频数据 | 大规模（具体未披露） | 覆盖广泛视觉场景，无具身专属标注 | 阶段一预训练 |
| EWMBench | 评估集 | 基于AgiBot-World真实机器人操作数据 | 测试评估 |
| DreamGen Bench | 评估集 | 机器人学习泛化能力评估 | 测试评估 |
| WorldModelBench | 评估集 | 视频世界模型综合评估 | 测试评估 |
| PBench | 评估集 | 物理合理性评估 | 测试评估 |
| RoboTwin-IF | 评估集 | 零样本跨具身迁移，多视角一致性 | 零样本测试 |

### 实现细节

- **语言编码器**: Qwen2.5-VL（冻结，不参与训练）
- **视觉编码器/解码器**: Video-VAE（端到端）
- **主干网络**: 60层 Double-Stream MMDiT
- **训练策略**: 两阶段渐进课程（General → Expert）
- **动作接口**: 自然语言（统一所有具身形态）
- **硬件/规模**: 具体训练细节待论文正式版本发布

### 应用方向

1. **合成数据增强**: 为策略训练生成多样化合成视频数据，缓解真实数据稀缺问题
2. **虚拟评估环境**: 作为可扩展的虚拟仿真环境，对机器人策略进行无风险评估
3. **语言引导规划**: 为下游机器人控制提供语言条件下的规划信号和轨迹先验

---

## 批判性思考

### 优点

1. **统一框架**: 以自然语言消弭了不同具身平台的动作接口异构性，极大降低了多领域世界建模的工程复杂度
2. **规模优势**: EWK语料库的860万视频-文本对和200M+帧覆盖了前所未有的具身多样性，为泛化能力奠定数据基础
3. **强大语义基础**: 冻结Qwen2.5-VL作为语义编码器，充分利用了大模型的语言理解和常识推理能力，无需从头训练
4. **全面评估**: 在EWMBench、DreamGen Bench、WorldModelBench、PBench和RoboTwin-IF五个基准上全面验证，结果具有说服力
5. **实用应用**: 明确提出三个下游应用方向（数据增强、虚拟评估、规划信号），商业价值清晰

### 局限性

1. **自然语言接口的精度瓶颈**: 语言描述本质上是离散且模糊的，对于需要精确力控或毫米级精度的操作任务，语言可能无法完整捕捉所有必要信息
2. **计算开销**: 60层MMDiT结合冻结MLLM的架构参数量庞大，推理速度可能成为实时应用的瓶颈
3. **评估指标有限**: 目前基准主要评估视频生成质量，与实际下游策略性能（如成功率提升）的关联尚需验证
4. **数据标注质量**: 860万视频-文本对的动作语言标注质量和一致性如何保障尚未详述

### 潜在改进方向

1. **闭环验证**: 将Qwen-RobotWorld生成的合成数据直接用于策略训练并报告成功率提升，量化世界模型对下游任务的贡献
2. **量化加速**: 探索知识蒸馏或模型量化，降低推理延迟以支持实时机器人控制应用
3. **细粒度动作接口**: 结合结构化语言（如代码）或混合接口（语言+数值参数），提升精细操作的动作精度

### 可复现性评估

- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节描述（技术报告中有概述）
- [ ] EWK 数据集可获取（暂未公开）

---

## 关联笔记

### 基于

- [[Qwen2.5-VL]]: 作为冻结语义编码器，提供多模态语言理解能力
- [[扩散变换器|DiT]]: MMDiT 的基础架构
- [[视频VAE|Video-VAE]]: 视频压缩与重建模块

### 对比

- [[DreamGen]]: 机器人学习泛化评估基准，本文在 DreamGen Bench 上排名第一
- [[EWMBench]]: 具身世界模型评估框架（场景一致性/动作正确性/语义对齐）
- [[WorldModelBench]]: 视频世界模型综合评估基准
- [[GigaWorld]]: 大规模具身世界模型，本文同领域竞品

### 方法相关

- [[多模态扩散变换器|MMDiT]]: 核心架构，双流设计实现语言-视觉深度融合
- [[联合注意力|Joint Attention]]: 双流交互机制
- [[渐进课程学习|Progressive Curriculum]]: 两阶段训练策略
- [[具身世界模型|Embodied World Model]]: 本文所属研究方向

### 数据/评估相关

- [[EWK（具身世界知识）|Embodied World Knowledge]]: 本文构建的860万视频-文本语料库
- [[RoboTwin-IF]]: 零样本泛化评估基准

---

## 速查卡片

> [!summary] Qwen-RobotWorld (2026)
> - **核心**: 以自然语言为统一接口的跨具身视频世界模型
> - **方法**: 60层Double-Stream MMDiT + 冻结Qwen2.5-VL + EWK语料库(860万对) + 两阶段渐进课程训练
> - **结果**: EWMBench第1、DreamGen Bench第1、WorldModelBench&PBench开源最优、RoboTwin-IF零样本强泛化
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-16*
