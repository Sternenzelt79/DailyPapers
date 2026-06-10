---
title: "Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination"
method_name: "Efficient-WAM"
authors: [Jiajun Li, Tiecheng Guo, Yifan Ye, Rongyu Zhang, Xiaowei Chi, Qianpu Sun, Ying Li, Yunfan Lou, Yan Huang, Zhihe Lu, Meng Guo, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, video-prediction, knowledge-distillation, flow-matching, robot-manipulation, asymmetric-denoising]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.10040
created: 2026-06-10
---

# 论文笔记：Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 香港大学、北京大学、木卡机器人、中科院自动化所、南京大学 |
| 日期 | June 2026 |
| 项目主页 | [efficientwam.github.io](https://efficientwam.github.io/) |
| 对比基线 | [[Motus]], [[GigaWorld-Policy]], [[UWM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.10040) / Code: N/A |

---

## 一句话总结

> 通过紧凑视频专家蒸馏、低分辨率未来潜变量和非对称去噪，将 WAM 压缩至 1B 参数，实现 32× 提速并维持竞争性控制性能。

---

## 核心贡献

1. **紧凑视频专家蒸馏**: 从 WAN-2.2-5B 通过层切片初始化 + 隐藏状态/时间运动蒸馏，得到 0.8B 视频专家，延迟降至 1/5
2. **多尺度视频潜变量布局**: 高分辨率编码当前观测（240 tokens），低分辨率编码未来帧（60 tokens），Token 减少 75%
3. **非对称去噪调度**: 视频分支仅用 2-5 步，行动分支用 10 步，利用扩散早期即可捕获结构信息的特性

---

## 问题背景

### 要解决的问题

[[World-Action Model|WAM]] 在机器人控制中引入未来视频预测来增强动作生成，但现有 WAM（如 WAN-5B、Motus-8B）参数量大、推理延迟高（>2s/chunk），无法满足闭环物理部署的实时性要求。

### 现有方法的局限

- [[UWM]]、[[GigaWorld-Policy]]（5B）和 [[Motus]]（8B）的每 chunk 推理延迟达 2000-3200ms
- 现有 WAM 追求高保真视频生成，但对动作控制而言，这是不必要的计算负担
- 简单缩小模型尺寸（随机初始化）会导致性能急剧下滑（Clean 从 86.4% 降至 69%）

### 本文的动机

**核心洞见**：未来想象对控制的有效性来源于对任务相关物体/机器人几何与运动的粗略捕捉，而非光照、纹理等外观细节。因此，可以通过三个正交维度降低视频侧计算成本，同时保留对行动预测有帮助的结构化时空信息。

---

## 方法详解

### 模型架构

![Figure 1: Efficient-WAM Overview](https://arxiv.org/html/2606.10040v1/x1.png)

**说明**：Efficient-WAM 采用 [[Video Prediction|视频预测]] + [[Flow Matching|流匹配]] 的双分支架构，以低成本未来想象为动作生成提供粗粒度时空线索，而非追求高保真视频重建。

Efficient-WAM 采用**视频-行动双专家**架构：
- **输入**: 语言指令 $l$ + 当前观测 $o$（多视角 RGB 图像）+ 机器人状态 $s$
- **视频分支 $\phi$**: 紧凑 [[Video Diffusion Model|视频扩散]] 专家（0.8B），生成未来帧潜变量 $\mathbf{z}^v$
- **行动分支 $\theta$**: 行动扩散专家，以 $\mathbf{z}^v$ 为条件生成动作块 $a_{1:H}$
- **输出**: [[Action Chunking|动作块]] $a_{1:H}$（$H=64$）
- **总参数**: ~1B

### 核心模块

#### 模块1: 紧凑视频专家（Compact Video Expert）

**设计动机**: 通过[[Knowledge Distillation|知识蒸馏]]保留教师 WAN-2.2-5B 的运动建模能力，同时大幅削减参数量

**具体实现**:
- **层切片初始化**: 从教师模型中均匀采样层来初始化 Transformer 深度，避免随机初始化的性能损失
- **隐藏状态蒸馏（$\mathcal{L}_{\text{hid}}$）**: 对切片层的中间隐藏状态做余弦对齐
- **时间运动蒸馏（$\mathcal{L}_{\text{mot}}$）**: 对帧间隐藏状态差分做余弦对齐，强化时序运动建模
- 效果：Clean 从随机初始化的 69% 提升至 87%（与 5B 教师模型 86.4% 持平）

#### 模块2: 多尺度视频潜变量（Token-Sparse Future Latents）

**设计动机**: 利用控制任务对视觉保真度不敏感的特性，通过[[Token Compression|Token 压缩]]降低视频侧计算

**具体实现**:
- 当前观测：高分辨率编码 384×320 → **240 tokens**（保留细节，助于精确行动预测）
- 未来帧：低分辨率编码 192×160 → **60 tokens**（减少 75%，捕获粗略几何/运动）
- 多尺度 token 在 Transformer 的注意力机制中拼接处理
- 效果：延迟从 430ms 降至 377ms，性能下降约 2%（可接受）

#### 模块3: 非对称去噪调度（Asymmetric Denoising）

**设计动机**: [[Diffusion Model|扩散模型]]早期去噪步骤即可捕获结构性信息，视频不需要与行动相同的步数

**具体实现**:
- **行动分支**：使用完整 10 步去噪（精确行动预测需要细粒度细化）
- **视频分支**：仅用 2-5 步（Figure 4 消融验证 4 步为最优平衡点）
- [[Decoupled Noise Scheduling|解耦噪声调度]]：视频与行动分别维护独立噪声时间步 $t_v, t_a$，联合训练时互不干扰
- 效果：延迟从 430ms 降至 139ms（单独应用时）

### 三阶段训练策略

![Figure 2: Efficient-WAM Architecture](https://arxiv.org/html/2606.10040v1/x2.png)

**说明**：展示多尺度视频潜变量布局与双专家架构的详细结构，高分辨率当前帧与低分辨率未来帧潜变量拼接后送入行动分支。

| 阶段 | 可训练模块 | 目标 |
|------|-----------|------|
| Stage 1 | 视频专家 | 流匹配 + 蒸馏损失 |
| Stage 2 | 行动专家（视频冻结）| 行动流匹配 + 轻量视频损失 |
| Stage 3 | 视频+行动专家（独立 LR）| 端到端联合精调 |

---

## 关键公式

### 公式1: [[Factorized Joint Distribution|联合分布分解]]

$$
p(\mathbf{z}^v, a_{1:H} \mid o, l, s) = \underbrace{p_\phi(\mathbf{z}^v \mid o, l)}_{\text{视频分支}} \cdot \underbrace{p_\theta(a_{1:H} \mid o, l, s, \mathbf{z}^v)}_{\text{行动分支}}
$$

**含义**: 将世界模型（视频预测）与策略（行动生成）分解为条件独立的两个分支，视频潜变量 $\mathbf{z}^v$ 作为行动生成的结构化条件

**符号说明**:
- $\mathbf{z}^v$: 未来视频帧的潜变量表示
- $o$: 当前多视角 RGB 观测
- $l$: 自然语言任务指令
- $s$: 机器人关节状态
- $a_{1:H}$: 长度为 $H$ 的动作块
- $\phi, \theta$: 视频专家和行动专家的参数

### 公式2: [[Flow Matching|流匹配]]目标

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|f(\mathbf{x}_t, t; c) - \mathbf{u}_t\|_2^2\right]
$$

其中插值路径和目标速度为：

$$
\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad \mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0
$$

**含义**: 通过回归目标速度场来训练扩散模型，使用线性插值轨迹；$\mathbf{x}_0$ 为噪声，$\mathbf{x}_1$ 为数据

**符号说明**:
- $t \in [0,1]$: 扩散时间步
- $\mathbf{x}_0$: 标准高斯噪声
- $\mathbf{x}_1$: 真实数据（视频潜变量或动作序列）
- $\mathbf{u}_t$: 目标速度（方向向量）
- $c$: 条件信息

### 公式3: [[Knowledge Distillation|隐藏状态蒸馏]]损失

$$
\mathcal{L}_{\text{hid}} = \frac{1}{|\mathcal{A}_{\text{hid}}|}\sum_{l \in \mathcal{A}_{\text{hid}}}\mathbb{E}_n\left[1 - \cos\left(\tilde{h}_{s,n}^l, \tilde{h}_{t,n}^{\tau(l)}\right)\right]
$$

**含义**: 对齐学生模型与教师模型对应层的归一化隐藏状态，通过余弦相似度最大化来传递特征表示能力

**符号说明**:
- $\mathcal{A}_{\text{hid}}$: 参与蒸馏的学生层集合
- $\tilde{h}_{s,n}^l$: 学生第 $l$ 层第 $n$ 个 token 的归一化隐藏状态
- $\tilde{h}_{t,n}^{\tau(l)}$: 教师对应层 $\tau(l)$ 的归一化隐藏状态
- $\tau(\cdot)$: 学生层到教师层的映射函数

### 公式4: [[Temporal Motion Distillation|时间运动蒸馏]]损失

$$
\mathcal{L}_{\text{mot}} = \frac{1}{|\mathcal{A}_{\text{mot}}|}\sum_{l \in \mathcal{A}_{\text{mot}}}\mathbb{E}_f\left[1 - \cos\left(\Delta\overline{h}_{s,f}^l, \Delta\overline{h}_{t,f}^{\tau(l)}\right)\right]
$$

**含义**: 对帧间隐藏状态差分做余弦对齐，专门强化模型对时序运动的建模能力（比单纯对齐静态特征更有效）

**符号说明**:
- $\Delta\overline{h}_{s,f}^l$: 学生第 $l$ 层相邻帧 $f$ 的隐藏状态差分（空间平均后）
- $\Delta\overline{h}_{t,f}^{\tau(l)}$: 教师对应层的帧间差分

### 公式5: [[Stage 1 Training|第一阶段训练]]目标

$$
\mathcal{L}_{\text{stage-1}} = \mathcal{L}_{\text{video-FM}} + \lambda_{\text{distill}}(\mathcal{L}_{\text{hid}} + \mathcal{L}_{\text{mot}})
$$

**含义**: 第一阶段同时优化视频生成能力（流匹配损失）和对教师的特征对齐（蒸馏损失），两者权重可调

**符号说明**:
- $\mathcal{L}_{\text{video-FM}}$: 视频侧流匹配损失
- $\lambda_{\text{distill}}$: 蒸馏损失权重

### 公式6: [[Stage 2 Training|第二阶段训练]]目标

$$
\mathcal{L}_{\text{stage-2}} = \mathcal{L}_{\text{action-FM}} + \lambda_v \mathcal{L}_{\text{video-FM}}
$$

**含义**: 冻结视频专家，训练行动专家；保留小权重视频损失（$\lambda_v = 0.01$）以防视频分支退化

**符号说明**:
- $\mathcal{L}_{\text{action-FM}}$: 行动侧流匹配损失
- $\lambda_v = 0.01$: 视频损失权重（轻量级保留）

### 公式7: [[Decoupled Noise Scheduling|解耦噪声调度]]前向过程

$$
\mathbf{x}_{t_a}^a = (1-t_a)\boldsymbol{\epsilon}_a + t_a\mathbf{a}, \quad \mathbf{u}^a = \mathbf{a} - \boldsymbol{\epsilon}_a
$$

$$
\mathbf{x}_{t_v}^v = (1-t_v)\boldsymbol{\epsilon}_v + t_v\mathbf{z}^v, \quad \mathbf{u}^v = \mathbf{z}^v - \boldsymbol{\epsilon}_v
$$

**含义**: 行动和视频各自维护独立噪声时间步 $t_a, t_v$，推理时可对两者分别设置不同的去噪步数

**符号说明**:
- $t_a, t_v$: 行动/视频各自的扩散时间步（独立采样）
- $\boldsymbol{\epsilon}_a, \boldsymbol{\epsilon}_v$: 标准高斯噪声
- $\mathbf{a}$: 真实动作序列
- $\mathbf{z}^v$: 真实视频潜变量

### 公式8: [[Joint Training Objective|联合训练]]目标

$$
\mathcal{L}_{\text{joint}} = \lambda_a\|f_a(\mathbf{x}_{t_a}^a) - \mathbf{u}^a\|_2^2 + \lambda_v\|f_v(\mathbf{x}_{t_v}^v) - \mathbf{u}^v\|_2^2
$$

**含义**: 第三阶段端到端联合训练，行动损失与视频损失的加权求和；$\lambda_a=1.0$，$\lambda_v=0.01$

**符号说明**:
- $\lambda_a = 1.0$: 行动损失权重
- $\lambda_v = 0.01$: 视频损失权重（视频已充分训练，轻量保持）
- $f_a, f_v$: 行动和视频网络的速度场函数

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2606.10040v1/x1.png)

**说明**: Efficient-WAM 的整体设计理念。以"低成本未来想象"替代高保真视频生成，捕获任务相关的物体和机器人几何与运动线索，作为行动生成的粗粒度条件。

### Figure 2: 模型架构

![Figure 2](https://arxiv.org/html/2606.10040v1/x2.png)

**说明**: 多尺度视频潜变量布局与双专家架构细节。当前帧高分辨率（384×320，240 tokens）与未来帧低分辨率（192×160，60 tokens）拼接后作为行动分支的条件输入。

### Figure 3: 真实世界操作任务

![Figure 3](https://arxiv.org/html/2606.10040v1/figure/result_real.png)

**说明**: 在 Astribot S1 机器人上评估的四项真实操作任务：精密抓取（移液管）、物体转移（试剂瓶）、语义分拣（LEGO 积木）、双手协调（笔盖拆卸）。

### Figure 4: 非对称去噪延迟-成功率权衡

![Figure 4](https://arxiv.org/html/2606.10040v1/x3.png)

**说明**: 视频去噪步数 $K_v$ 对延迟（柱状图）与成功率（折线图）的影响。$K_v=4$ 时达到最优平衡，继续增加步数带来的成功率增益不足以弥补延迟代价。

### Figure 5: 真实机器人部署环境

![Figure 5](https://arxiv.org/html/2606.10040v1/figure/robot_setup.png)

**说明**: Astribot S1 双臂机器人的实验场景设置，展示四类操作任务的物理布局。

### Table 1: RoboTwin 2.0 仿真基准对比

| 方法 | Clean (%) | Random (%) | 参数量 |
|------|-----------|------------|--------|
| **VLA 方法** | | | |
| [[Pi0\|π₀]] | 65.9 | 58.4 | 3.3B |
| StarVLA-α | 76.8 | 79.1 | 2B |
| [[Pi05\|π₀.₅]] | 82.7 | 76.8 | 3.3B |
| ABot-M0 | 86.1 | 85.1 | 4.2B |
| LingBot-VLA | 86.5 | 85.3 | 4B |
| **WAM 方法** | | | |
| [[UWM]] | 81.7 | 78.6 | 5B |
| [[GigaWorld-Policy]] | 86.4 | 85.0 | 5B |
| [[Motus]] | 88.7 | 87.0 | 8B |
| **Efficient-WAM** | **86.7** | **85.7** | **1B** |
| Efficient-WAM-RT | 83.1 | 82.0 | — |

**关键发现**: 1B 参数的 Efficient-WAM 以 1/5 参数量达到 5B WAM 相当的性能，仅比 8B 的 Motus 低约 2%。

### Table 2: 真实世界对比（Astribot S1）

| 任务 | π₀.₅ (%) | Motus (%) | Ours (%) |
|------|-----------|-----------|----------|
| 移液管抓取 | 100.0 | 85.0 | 95.0 |
| 试剂瓶转移 | 75.0 | 80.0 | 75.0 |
| LEGO 颜色分拣 | 30.0 | 65.0 | 65.0 |
| 笔盖拆卸 | 10.0 | 25.0 | 30.0 |
| **平均成功率 (%)** | **53.75** | **63.75** | **66.25** |
| **每 Chunk 延迟 (ms)** | **113.0** | **3215.0** | **98.0** |
| **每步延迟 (ms)** | **7.1** | **200.9** | **6.1** |

**关键发现**: Efficient-WAM 以 98ms/chunk 的延迟（vs Motus 的 3215ms，**32× 提速**）实现了更高的平均成功率（66.25% vs 63.75%），在精密操作任务上尤为突出。

### Table 3a: 消融——紧凑视频专家

| 变体 | 初始化 | 蒸馏 | Clean (%) | Rand. (%) | 参数量 | 延迟 (ms) |
|------|--------|------|-----------|-----------|--------|-----------|
| Full WAN（教师）| Full | — | 86.4 | 85.5 | 5B | 2013 |
| Compact-random | 随机 | 无 | 69 | 68 | 0.8B | 432 |
| Compact-sliced | 层切片 | 无 | 82 | 81 | 0.8B | 426 |
| **Efficient-WAM** | **层切片** | **有** | **87** | **86** | **0.8B** | **430** |

**关键发现**: 层切片初始化贡献 +13% Clean；蒸馏额外贡献 +5%，且不带来额外延迟开销。

### Table 3b: 消融——未来帧分辨率

| 分辨率 | Clean (%) | Rand. (%) | Tokens | 延迟 (ms) |
|--------|-----------|-----------|--------|-----------|
| 高（384×320）| 87 | 86 | 240 | 430 |
| 中（~）| 85 | 84 | 126 | 396 |
| **低（192×160）** | **83** | **82** | **60** | **377** |

**关键发现**: 低分辨率损失约 4% 性能，换取 53ms 延迟收益和 75% Token 压缩率，适用于 RT 部署场景。

### Table 4: 训练配置

| 阶段 | 可训练模块 | Batch Size | LR | 视频损失权重 | 行动损失权重 |
|------|-----------|-----------|-----|------------|------------|
| Stage 1 | 视频专家 | 16×8 | 5×10⁻⁵ | 1.0 | — |
| Stage 2 | 行动专家 | 16×8 | 5×10⁻⁵ | 0.01 | 1.0 |
| Stage 3 | 视频+行动 | 16×8 | 1×10⁻⁵/5×10⁻⁵ | 0.01 | 1.0 |

共同配置：AdamW、余弦 LR 调度、bf16 混合精度、权重衰减 1×10⁻³

### Table 5: 各配置延迟汇总

| 配置 | 紧凑 | 低分辨率 | 非对称 | GPU | 延迟 (ms) |
|------|------|---------|--------|-----|-----------|
| Full WAN | 否 | 否 | 否 | A800 | 2013 |
| Efficient-WAM | 是 | 否 | 否 | A800 | 430 |
| + 低分辨率未来 | 是 | 是 | 否 | A800 | 377 |
| + 非对称去噪 | 是 | 否 | 是 | A800 | 139 |
| **Efficient-WAM-RT** | **是** | **是** | **是** | **RTX 4090** | **98** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 2,500 清晰演示 + 25,000 随机化演示 | 多任务仿真基准，含干净/随机化两种评估设置 | 仿真训练 + 测试 |
| Astribot S1 真实数据 | 每任务 100 个人工演示 | 四类精密操作，双臂机器人 | 真实世界训练 + 测试 |

### 实现细节

- **视频骨干**: WAN-2.2-5B 的紧凑切片版本（0.8B）
- **优化器**: AdamW，余弦 LR 调度
- **Batch Size**: 16×8（每 GPU 16，共 8 GPU）
- **LR**: Stage 1/2 为 5×10⁻⁵，Stage 3 视频 1×10⁻⁵/行动 5×10⁻⁵
- **精度**: bf16 混合精度
- **权重衰减**: 1×10⁻³
- **硬件**: NVIDIA A800（训练）；RTX 4090（RT 部署推理）
- **动作块长度**: $H=64$，行动去噪步 10，视频去噪步 4（RT 模式）

### 可视化结果

附录中展示了 Full WAN（高保真）与 Efficient-WAM-RT（粗粒度）的未来预测对比（Figure 7/8）。Efficient-WAM-RT 预测图像明显模糊，但仍保留了物体位置、手臂轨迹等任务相关结构信息——验证了"粗粒度想象足以引导控制"的核心假设。

---

## 批判性思考

### 优点
1. **效率与性能的精妙平衡**: 不追求视觉保真度的核心洞见非常有价值，打破了 WAM 必须是大模型的思维定势
2. **三个正交优化维度**: 紧凑蒸馏、低分辨率、非对称去噪三者独立且可叠加，工程实用性强
3. **充分消融验证**: 每个设计选择都有详细消融，说服力强
4. **真实部署验证**: 不只是仿真结果，在真实机器人上证明了 98ms 闭环可行性

### 局限性
1. **视频质量损失明显**: Efficient-WAM-RT 的未来预测"明显粗糙"，在需要精密外观建模的任务上可能存在上限
2. **任务类型局限**: 仅在四类实验室操作任务上测试，泛化性（尤其是高动态、长时序任务）尚未验证
3. **与 Motus 仍有差距**: 在仿真基准的 Clean 设置下落后 2%（86.7% vs 88.7%），差距不大但真实场景可能被放大

### 潜在改进方向
1. 探索自适应分辨率策略（根据任务难度动态调整未来帧分辨率）
2. 将蒸馏方法推广到其他 WAM 架构（不限于 WAN 系列）
3. 研究更少步数（1-2 步）视频去噪是否仍能保持可用的结构信息

### 可复现性评估
- [ ] 代码开源（项目主页存在但代码未开源）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（Table 4 提供了完整超参数）
- [x] 数据集可获取（RoboTwin 2.0 公开，Astribot 真实数据为私有）

---

## 关联笔记

### 基于
- [[WAN]]: 教师模型 WAN-2.2-5B，提供视频生成先验
- [[GigaWorld-Policy]]: 直接基线对比（5B WAM）
- [[Flow Matching]]: 核心生成框架

### 对比
- [[Motus]]: 8B WAM，仿真最强基线，Efficient-WAM 以 1/8 参数量接近其性能
- [[UWM]]: 5B WAM，unified world model 方法
- [[Pi05]]: 3.3B VLA，真实世界重要对比基线

### 方法相关
- [[Knowledge Distillation]]: 层切片 + 蒸馏损失是压缩核心
- [[World-Action Model]]: 本文所属方法类别
- [[Action Chunking]]: 行动输出形式
- [[Decoupled Noise Scheduling]]: 非对称去噪的实现基础

### 硬件/数据相关
- [[Astribot S1]]: 真实部署平台，双臂机器人
- [[RoboTwin]]: 仿真评估基准

---

## 速查卡片

> [!summary] Efficient-WAM
> - **核心**: 高保真未来预测对控制非必要，粗粒度结构信息足以引导行动生成
> - **方法**: 层切片蒸馏(1B) + 低分辨率未来潜变量(60 tokens) + 非对称去噪(视频4步/行动10步)
> - **结果**: 86.7% Clean / 98ms 延迟（RTX 4090），32× 提速于 Motus
> - **代码**: [efficientwam.github.io](https://efficientwam.github.io/)

---

*笔记创建时间: 2026-06-10*
