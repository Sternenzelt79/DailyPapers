---
title: "RhinoVLA Technical Report"
method_name: "RhinoVLA"
authors: [Chen Zhang, Chenyang Zhou, Guanglei Ding, Guanghui He, Haibin Gao, Jiajia Chen, Jianyong Zhang, Lianyi Yu, Ningyi Xu, Ping Xu, Qingchen Li, Yingjun Hu, Yijia Zhang, Yuxi Liu]
year: 2026
venue: arXiv
tags: [vla, edge-deployment, flow-matching, cross-embodiment, token-efficiency, robot-manipulation]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2606.07383
created: 2026-06-08
---

# 论文笔记：RhinoVLA Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Huixi Intelligence |
| 日期 | June 2026 |
| 项目主页 | 待开源 |
| 对比基线 | [[Pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07383) / Code: 规划开源 |

---

## 一句话总结

> RhinoVLA 通过 token 高效的 Qwen3-VL 主干、统一跨机器人接口（View Registry + 72D slot 空间 + Robot-instance LoRA）与硬件协同优化，实现了在 Huixi R1 边缘设备上 11.69 Hz 实时闭环 VLA 控制。

---

## 核心贡献

1. **Token 高效主干**: 使用 [[Qwen3-VL]]（2.13B 参数）替代 PaliGemma，每张图像仅生成 64 个 merged visual tokens（PaliGemma 为 256），降低 4× 视觉 token 负担，直接减少 GEMM FLOPs。
2. **统一跨机器人接口**: 通过 View Registry、72D 物理 slot 空间和 Robot-instance [[LoRA]] 三机制统一异构机器人数据集，支持跨具身泛化部署。
3. **算法-系统协同优化**: 针对 Huixi R1 硬件（500 TOPS INT8）定制 W8A16 GEMM kernel、FlashAttention 平铺编译优化及并行视觉编码，实现从 5.84 Hz 到 11.69 Hz 的 2× 推理加速。

---

## 问题背景

### 要解决的问题

[[Vision-Language-Action Model|VLA]] 模型在边缘硬件上部署时，VLM 视觉 token 和上下文 token 数量大，造成严重推理延迟，无法满足机器人实时闭环控制（通常需要 ≥10 Hz）的要求。

### 现有方法的局限

- **π0.5（PaliGemma 主干）** 在 NVIDIA Jetson AGX Orin（43 TFLOPS FP16）上完整推理延迟达 858.3ms（约 1.17 Hz），其中 VLM 骨干占 528ms（61.5%）、动作专家占 257ms（29.9%），在 10 Hz 目标下已严重超出 roofline 极限。
- 不同数据集的摄像头命名、顺序、数量不一致，难以统一训练。
- 异构机器人同一向量索引指代不同物理量，跨机器人泛化困难。
- 独立机器人输出头部会导致部署图碎片化，难以统一管理。

### 本文的动机

通过选用 token 更紧凑的 [[Qwen3-VL]] 主干（64 tokens/图 vs 256）减少算力瓶颈，同时设计三层统一接口解决异构性，再通过硬件定制 kernel 进一步压榨推理效率，实现可实际落地的实时边缘 VLA。

---

## 方法详解

### 模型架构

RhinoVLA 采用 **双模块解耦** 架构：

- **输入**: 语言任务指令 + 多视角 RGB/Depth 图像 + 72D 物理状态 $s$
- **VLM Backbone**: [[Qwen3-VL]]（2.13B 参数），冻结主干，训练 [[LoRA]] 适配层，输出最后 18 层 KV 缓存 $c_{vlm}$
- **核心模块**: [[Flow Matching|连续动作专家]]（0.40B 参数），条件化 VLM 记忆，预测 72D 动作 slot 的 flow velocities
- **统一接口**: View Registry + 72D Slot 空间 + Robot-instance LoRA 三层跨机器人对齐机制
- **输出**: [[Action Chunking|动作块]] $\hat{a} \in \mathbb{R}^{H \times 72}$（H 为预测 horizon）
- **总参数**: ~2.53B（VLM 2.13B + Action Expert 0.40B）

### 核心模块

#### 模块1: View Registry（视图注册表）

**设计动机**: 解决不同数据集摄像头命名、顺序、数量不一致的问题，利用[[视觉语言模型|VLM]]的语言理解能力统一视角描述。

**具体实现**:
- 定义角色-模态词汇表，格式 `[role|modality]`，如 `[head|rgb]`、`[left_wrist|depth]`
- 训练提示模板结构化列举视图：
  ```
  Task: {task}
  Views:
  [{role_1}|{modality_1}] + image_1
  [{role_2}|{modality_2}] + image_2
  ...
  ```
- 支持的角色：`front`, `left_wrist`, `right_wrist`, `head` 等
- 支持的模态：`rgb`, `depth`, `rgbd`
- 视图数量和顺序可变，VLM 通过文本标记理解语义

#### 模块2: 统一 72D Slot 空间

**设计动机**: 不同数据集中同一向量索引可能对应不同物理量，固定物理语义映射可消歧义并实现跨具身训练。

**具体实现**:
- 固定 72 维连续向量，每维对应固定物理语义（如 D0–D6 为左臂关节，D58–D60 为底座速度）
- 引入二值掩码：**状态掩码** $m_s$ 指示当前机器人存在的维度，**动作掩码** $m_a$ 指示有效监督维度
- 仅 $m_a$ 激活的 slot 贡献 [[Flow Matching]] 训练目标，不同机器人自然"稀疏"使用不同 slot
- 灵巧手 slot 分配（每手 16D）采用 4-3-3-3-3 结构（拇指 4-DoF，其余四指各 3-DoF）

#### 模块3: Robot-instance LoRA

**设计动机**: 同一 nominal 动作 slot 在不同机器人实例间存在标定误差、关节极限和控制器差异，需轻量级适配而非独立头部。

**具体实现**:
- [[LoRA]] 适配器置于 Action Expert 前馈网络每层（注意力模块和最终动作投影保持共享）
- 通过 `instance_id` 硬路由选择对应 LoRA 权重
- 训练时基础 Action Expert 直接监督，instance LoRA 残差有正则化约束防止完全接管基础策略
- 优点：统一部署图、稀疏适配激活、低成本扩展新机器人

#### 模块4: Huixi R1 部署优化

**设计动机**: R1 硬件（500 TOPS INT8）的 SPM（on-chip scratchpad memory）特性可通过定制 kernel 极大减少 DDR 内存访问。

**具体实现**:
1. **编译优化**: FlashAttention 风格平铺，Q/K/V tiles 保持在 SPM 芯片上；激进算子融合（norm + linear + bias + activation + residual）
2. **混合精度 W8A16**: INT8 权重 + FP16 激活，自定义 GEMM kernel 融合权重加载、反量化、矩阵乘法；8 核并行执行；`up_proj` 延迟从 191μs 降至 113μs（1.69× 加速）
3. **并行视觉编码**: 3 路摄像头图像从串行改为批处理，延迟从 34.52ms 降至 24.31ms

---

## 关键公式

### 公式1: [[Flow Matching|线性插值轨迹构造]]

$$
x_t = (1-t)\,a + t\,z
$$

**含义**: 从高斯噪声 $a$ 到干净动作目标 $z$ 的线性插值，构造 flow matching 训练轨迹。

**符号说明**:
- $x_t \in \mathbb{R}^{H \times 72}$: 时间步 $t$ 的插值动作
- $a \sim \mathcal{N}(0, I)$: 高斯噪声起点
- $z \in \mathbb{R}^{H \times 72}$: 干净目标动作（72D slot 空间）
- $t \sim \mathcal{U}(0,1)$: 均匀采样时间步

### 公式2: [[Flow Matching|Action Expert 预测函数]]

$$
\hat{v}_\theta = f_\theta(x_t,\; t,\; s,\; m_s,\; m_a,\; c_{vlm},\; r)
$$

**含义**: Action Expert 以插值动作 $x_t$、时间 $t$、状态 $s$、掩码、VLM 记忆 $c_{vlm}$ 和机器人实例 $r$ 为条件，预测 flow velocity。

**符号说明**:
- $\hat{v}_\theta \in \mathbb{R}^{H \times 72}$: 预测的 flow velocity（即 $z - a$ 的估计）
- $s \in \mathbb{R}^{72}$: 当前 72D 物理状态
- $m_s$: 状态掩码（存在的物理维度）
- $m_a$: 动作掩码（有效监督维度）
- $c_{vlm}$: Qwen3-VL 最后 18 层 KV 缓存（VLM 视觉-语言记忆）
- $r$: 机器人实例 ID（选择对应的 robot-instance LoRA）

### 公式3: [[Flow Matching|掩码流匹配损失]]

$$
\mathcal{L}_{FM} = \frac{\displaystyle\sum_{h,d} m_a(d)\,w(h,d)\,\bigl\|\hat{v}_\theta(h,d) - \bigl(z(h,d) - a(h,d)\bigr)\bigr\|_2^2}{\displaystyle\sum_{h,d} m_a(d)\,w(h,d) + \varepsilon}
$$

**含义**: 仅对动作掩码 $m_a$ 激活的 slot 计算加权 flow velocity 预测误差，屏蔽当前机器人不使用的物理维度。

**符号说明**:
- $h \in [1, H]$: 预测 horizon 时间步索引
- $d \in [1, 72]$: 物理 slot 索引
- $m_a(d) \in \{0,1\}$: 第 $d$ 维动作掩码（1 为有效）
- $w(h,d)$: per-slot 权重（不同物理量量纲归一化）
- $\varepsilon$: 数值稳定小量

### 公式4: [[数据平衡采样|幂律数据采样概率]]

$$
p_i = \frac{N_i^{0.43}}{\displaystyle\sum_j N_j^{0.43}}
$$

**含义**: 对多数据源进行幂律平衡采样，指数 0.43 介于均匀采样（0）和按比例采样（1）之间，防止大数据集完全主导训练。

**符号说明**:
- $p_i$: 第 $i$ 个数据源的采样概率
- $N_i$: 第 $i$ 个数据源的样本数量
- $0.43$: 幂律指数（经验值）

### 公式5: [[模型效率分析|线性层 FLOPs 估算]]

$$
\text{FLOPs} = 2 B S d_{in} d_{out}
$$

**含义**: 估算 Transformer 线性层的浮点运算量，用于 roofline 分析确定 VLA 推理的计算瓶颈。

**符号说明**:
- $B$: batch size
- $S$: 输入 token 数量
- $d_{in}$: 输入特征维度
- $d_{out}$: 输出特征维度

---

## 关键图表

### Figure 1: 系统概览

![[RhinoVLA_fig1_overview.png]]

**说明**: RhinoVLA 系统概览。展示通过算法-系统协同设计在 Huixi R1 边缘硬件上实现 11.69 Hz 实时控制，支持跨多种机器人（Galbot G1、AgiBot G1/G2、Huixi R1）的无缝部署。

### Figure 2: 边缘硬件 Roofline 分析

![[RhinoVLA_fig2_roofline.png]]

**说明**: 在 NVIDIA Jetson AGX Orin（FP16）上对主流 VLA 模型的端到端 roofline 分析。显示 π0.5 在 5 Hz 目标频率已接近硬件极限，10 Hz 以上严重超出，凸显边缘部署瓶颈。

### Figure 3: RhinoVLA 整体架构

![[RhinoVLA_fig3_architecture.png]]

**说明**: RhinoVLA 完整架构图。展示三层跨机器人接口机制（View Registry → 多视角图像标准化输入、72D Slot 空间 + binary masks → 统一动作表示、Robot-instance LoRA → 实例适配），以及 [[Qwen3-VL]] 主干和 [[Flow Matching|连续动作专家]] 的信息流动。

### Figure 4: R1 推理加速累积效果

![[RhinoVLA_fig4_speedup.png]]

**说明**: Huixi R1 上三项优化的增量加速效果。从基础 5.84 Hz 经编译优化、W8A16 混合精度、并行视觉编码，逐步提升至 11.69 Hz，每项优化的增量贡献可见。

### Figure 5: 预训练损失诊断

![[RhinoVLA_fig5_training_loss.png]]

**说明**: 预训练阶段损失曲线（左：全局掩码 flow-matching 损失；右：完整目标损失及 AgiBot G1/G2 的 per-instance 损失）。显示不同机器人实例的损失收敛轨迹，验证 robot-instance LoRA 分别学习了具身特定残差。

### Figure 6: Instance LoRA 嵌入分析

![[RhinoVLA_fig6_lora_analysis.png]]

**说明**: 小规模诊断实验。左矩阵：不同机器人实例 LoRA 残差的相关性；右矩阵：对应实例动作掩码的 Hamming 距离。两者的对应关系验证了 instance LoRA 编码的是具身特定修正，而非仅仅区分数据集标识。

### Figure 7: AgiBot G1 毛巾折叠演示

![[RhinoVLA_fig7_towel_fold.png]]

**说明**: RhinoVLA 在 AgiBot G1 上执行双臂毛巾折叠任务（[[Bimanual Manipulation]]），展示可形变物体操纵的鲁棒性（Seen: 67%，Unseen: 43% 成功率）。

### Table 1: View Registry 词汇表与训练模板

| 字段 | 值/示例 |
|------|--------|
| 任务文本 | `pick up the cup` |
| 视图角色 | `front`, `left_wrist`, `right_wrist`, `head` |
| 视图模态 | `rgb`, `depth`, `rgbd` |
| 图像格式 | Qwen-VL structured image item |
| 训练提示结构 | `Task: {task}\nViews:\n[{role}|{modality}] + image\n...` |

**说明**: View Registry 统一了不同数据集的摄像头命名，通过角色-模态词汇表使 VLM 能够跨数据集理解视觉输入的语义。

### Table 2: 完整 72D Slot 空间定义

| Slots | 物理组 | 单位 | 说明 |
|-------|--------|------|------|
| D0–D6 | 臂0 规范关节 | rad | 左臂（上臂、前臂、腕部） |
| D7–D13 | 臂1 规范关节 | rad | 右臂（双臂机器人） |
| D14–D15 | 平行夹爪 | 闭合比 [0,1] | 0=开，1=闭 |
| D16–D31 | 手0 活跃 DoF | rad | 灵巧手0（16D，4-3-3-3-3） |
| D32–D47 | 手1 活跃 DoF | rad | 灵巧手1（16D，4-3-3-3-3） |
| D48–D50 | 头部/颈部 RPY | rad | roll, pitch, yaw |
| D51–D52 | 躯干 pitch / lift | rad, m | 躯干俯仰及升降机构 |
| D53–D54 | 折叠腿机制 | rad | 两执行器 |
| D55–D57 | 腰部 RPY | rad | 腰部旋转 |
| D58–D60 | 底座速度命令 | m/s, rad/s | $v_x$, $v_y$, yaw rate |
| D61–D71 | 保留辅助 slot | — | 未使用 |

**说明**: 固定物理语义的 72D 统一 slot 空间，不同机器人通过二值掩码稀疏激活对应维度，实现在同一向量空间内跨具身训练。

### Table 3: Instance LoRA 消融实验（动作预测）

| 方法 | 掩码 FM 损失 | 臂 MAE | 底座速度 MAE | Yaw 率 MAE | 夹爪 MAE |
|------|------------|--------|------------|-----------|---------|
| 基础（无 LoRA） | 0.0192 | 0.0446 | 0.0187 | 0.0195 | 0.1064 |
| + Instance LoRA | **0.0191** | **0.0440** | **0.0188** | **0.0194** | **0.1056** |

**关键发现**: Instance LoRA 在所有指标上一致带来小幅但稳定的提升，符合其作为"轻量具身适配"的设计定位，而非大幅改变基础策略。

### Table 4: LIBERO 仿真基准对比（成功率 %）

| 模型 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| [[Diffusion Policy]] | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| [[Octo]] | 78.9 | 85.7 | 84.6 | 51.5 | 75.1 |
| [[OpenVLA]] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[Pi0-FAST|π0-FAST]] | — | — | — | 60.2 | — |
| [[π₀|π0]] | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 |
| [[Pi0.5|π0.5]] | **98.8** | **98.2** | **98.0** | **92.4** | **96.9** |
| **RhinoVLA** | 93.0 | 91.0 | 93.4 | 82.4 | **90.0** |

**关键发现**: RhinoVLA 在 LIBERO 上平均 90.0%，显著超越 π0（86.0%）和 π0-FAST（LIBERO-Long +22.2pp），但略低于针对该基准深度优化的 π0.5（96.9%）。LIBERO-Long 上超越 π0 达 9.4pp，展示长 horizon 任务优势。

### Table 5: 真机任务成功率对比

| 机器人 | 任务 | 环境 | π0.5 SR | RhinoVLA SR |
|--------|------|------|---------|------------|
| Galbot G1 | 红袋→远 bin | Unseen | 100% | **100%** |
| Galbot G1 | 黑扇→中 bin | Unseen | — | 40% |
| Galbot G1 | 白沫→近 bin | Unseen | — | 20% |
| AgiBot G2 | 三步序列 | Seen | — | 58% |
| AgiBot G2 | 三步序列 | Unseen | 18% | **24%** |
| AgiBot G1 | 毛巾折叠 | Seen | — | 67% |
| AgiBot G1 | 毛巾折叠 | Unseen | — | 43% |

**关键发现**: Galbot G1（见过配置）完全可靠达到 100%；AgiBot G2 长 horizon 任务 Unseen 环境中超越 π0.5（24% vs 18%）；毛巾折叠证明对可形变物体的操纵鲁棒性。

### Table 6: Huixi R1 端到端延迟分解

| 阶段 | 延迟 (ms) | 占比 | 说明 |
|------|----------|------|------|
| 视觉编码器（3 视图并行） | 24.31 | 28.4% | 批处理 3 路摄像头 |
| VLM 骨干（Qwen3-VL） | 20.78 | 24.3% | KV 缓存生成 |
| Action Expert | 36.71 | 42.9% | Flow matching 推理 |
| 其他 | 3.74 | 4.4% | 通信、调度等 |
| **总计** | **85.54** | 100% | **11.69 Hz 闭环** |

**关键发现**: 优化后总延迟 85.54ms，满足 10 Hz 实时控制目标（实际达 11.69 Hz）。Action Expert 为最大延迟来源（42.9%），是未来进一步优化的重点。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| AgiBotWorld Beta | 2976.4 小时 | AgiBot G1 机器人数据 | 预训练 |
| AgiBotWorld 2026 | 数百小时 | AgiBot G2 机器人数据 | 预训练 |
| LIBERO | 标准仿真基准 | 4 个任务套件（Spatial/Object/Goal/Long） | 仿真评估 |
| Galbot G1 / AgiBot G1/G2 真机数据 | 若干任务 | 微调目标机器人 | 真机迁移 |

### 实现细节

- **VLM Backbone**: [[Qwen3-VL]]（2.13B 参数），主干冻结，训练 [[LoRA]] 适配层
- **Action Expert**: 0.40B 参数 Transformer，条件化 Qwen3-VL 最后 18 层 KV 缓存
- **预训练策略**: 冻结 Qwen3-VL 主干，优化 VLM LoRA + 共享 Action Expert + Robot-instance LoRA
- **微调策略**: 冻结 VLM 编码器和预训练 VLM LoRA，仅更新目标机器人的 instance LoRA
- **数据平衡**: 幂律采样（指数 0.43）
- **部署硬件**: Huixi R1（500 TOPS INT8），NVIDIA Jetson AGX Orin（43 TFLOPS FP16，对比分析）
- **量化方案**: W8A16（INT8 权重，FP16 激活）

### 可视化结果

- Instance LoRA 残差相关性与动作掩码 Hamming 距离的对应关系（Figure 6）验证了适配器编码具身特定修正的设计假设
- 毛巾折叠任务（Figure 7）展示了对可形变物体的双臂协同操纵能力
- R1 推理加速曲线（Figure 4）直观显示三项优化的叠加效果

---

## 批判性思考

### 优点

1. **工程实用性突出**: 直接面向边缘部署的实际痛点（推理延迟），提出算法-系统协同解决方案，11.69 Hz 已超越 10 Hz 实用门槛。
2. **统一接口设计优雅**: View Registry + 72D slot + instance LoRA 三层机制从不同维度解决跨具身异构性，扩展新机器人成本低。
3. **消融实验支撑充分**: Figure 6 的 LoRA 残差分析提供了较强的 mechanistic 证据，不仅是性能数字。

### 局限性

1. **仿真性能与顶级方法有差距**: LIBERO 上 90.0% vs π0.5 的 96.9%，尤其 Long 任务（82.4% vs 92.4%），说明跨机器人预训练对单任务基准的适配存在代价。
2. **真机评估任务较简单**: AgiBot G2 三步序列 Unseen 仅 24%，部分任务（黑扇、白沫）成功率偏低（20-40%），真机泛化能力仍有提升空间。
3. **Action Expert 仍是延迟主体**: 占 42.9%（36.71ms），未来若需进一步提速，需要针对该模块的专项优化。
4. **数据规模不对称**: G1 数据（2976h）远多于 G2（数百小时），可能导致 G2 任务性能受限。

### 潜在改进方向

1. 对 Action Expert 进行蒸馏或结构压缩，突破 Action Expert 延迟瓶颈
2. 探索更激进的 INT4 或稀疏量化方案进一步降低 VLM 骨干延迟
3. 增加更多机器人实例和任务类型的真机验证，完善跨具身泛化评估

### 可复现性评估

- [ ] 代码开源（规划中）
- [ ] 预训练模型（规划中）
- [x] 训练细节较完整（数据集、超参、架构描述充分）
- [x] 数据集可获取（AgiBotWorld 为公开数据集）

---

## 关联笔记

### 基于

- [[Pi0.5]]: 主要对比基线，PaliGemma 主干 VLA，RhinoVLA 解决其边缘部署问题
- [[Qwen3-VL]]: VLM 主干网络，64 tokens/图的 token 高效特性是 RhinoVLA 的核心选择依据
- [[Flow Matching]]: Action Expert 的生成建模方法，连续动作预测框架

### 对比

- [[π₀|π0]]: 同样使用 flow matching 的 VLA，LIBERO-Long 上 RhinoVLA 超越 9.4pp
- [[Pi0-FAST|π0-FAST]]: 自回归加速版 π0，LIBERO-Long 上 RhinoVLA 超越 22.2pp
- [[Diffusion Policy]]: 扩散策略基线，LIBERO 平均 72.4% vs RhinoVLA 90.0%
- [[Octo]]: 通用机器人策略，LIBERO 平均 75.1% vs RhinoVLA 90.0%

### 方法相关

- [[LoRA]]: Robot-instance LoRA 和 VLM LoRA 的基础技术
- [[Action Chunking]]: 预测多步动作块的策略
- [[Bimanual Manipulation]]: 双臂操纵任务（毛巾折叠评估）
- [[跨具身学习]]: 跨机器人统一训练的核心挑战

### 硬件/数据相关

- [[Unitree G1]]: 类似人形机器人参考
- AgiBotWorld: 预训练使用的大规模机器人数据集（2976h+）

---

## 速查卡片

> [!summary] RhinoVLA Technical Report
> - **核心**: Token 高效 VLA + 三层跨机器人接口 + R1 硬件协同优化，实现 11.69 Hz 边缘实时控制
> - **方法**: Qwen3-VL（64 tokens/图）+ View Registry + 72D Slot + Robot-instance LoRA + W8A16 量化
> - **结果**: LIBERO 平均 90.0%（超 π0 4pp），R1 上 11.69 Hz（超 10 Hz 门槛），真机多平台验证
> - **代码**: 规划开源（Huixi Intelligence）

---

*笔记创建时间: 2026-06-08*
