---
title: "PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation"
method_name: "PiL-World"
authors: [Chong Ma, Taiyi Su, Jian Zhu, Jianjun Zhang, Zitai Huang, Yi Xu, Hanli Wang]
year: 2026
venue: arXiv
tags: [world-model, vla, closed-loop-evaluation, robot-manipulation, video-generation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.05773
created: 2026-06-05
---

# 论文笔记：PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确列出（来自 arXiv） |
| 日期 | June 2026 |
| 项目主页 | [pil-world.github.io](https://pil-world.github.io) |
| 对比基线 | [[OSCAR\|Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05773) / Code: N/A |

---

## 一句话总结

> PiL-World 是一个 chunk-wise [[世界模型|World Model]]，通过策略-在-环闭合循环推理，将 [[VLA]] 策略的实际-仿真成功率差距从 63.2% 大幅压缩至 12.0%。

---

## 核心贡献

1. **Chunk-Wise 闭合循环评估框架**: 设计 VLA 策略与世界模型交替推理的 policy-in-the-loop 评估流水线，解决现有世界模型只支持开环预测的缺陷
2. **Action-to-Control 投影机制**: 将绝对关节空间动作指令转换为头部视角抓手标记，作为视频生成的视觉控制条件
3. **Hallucination-Free Ratio (HFR) 指标**: 提出新评估指标，量化密集 rollout 中首次出现幻觉的归一化位置，弥补单纯成功率评估的不足

---

## 问题背景

### 要解决的问题

[[VLA]] 策略在真实机器人任务中以闭合循环运行：机器人观测场景 → 执行动作块 → 基于结果观测做出下一步决策。但真实机器人评估代价高昂，受限于硬件安全、场景重置和实验吞吐量，难以大规模进行。

### 现有方法的局限

现有[[世界模型|World Model]]主要支持**开环预测**，沿预先收集的固定动作序列进行预测，而非根据策略实时生成的动作块进行闭合循环评估。这导致三个关键问题：

1. **接口不匹配**：世界模型与 policy-in-the-loop 的需求之间存在接口鸿沟
2. **评估粒度不足**：仅靠任务级成功率无法验证 rollout 的过程一致性
3. **多视角一致性缺失**：想象出的 rollout 必须跨视角保持物理一致

### 本文的动机

如果能让世界模型直接消费 [[VLA]] 策略预测的动作块并生成下一帧观测，再将该观测反馈给策略，则可构建低成本的闭合循环虚拟评估环境，替代部分真实机器人评估。

---

## 方法详解

### 模型架构

PiL-World 采用**视频扩散模型 + 闭合循环滚动**架构：

- **输入**: 任务指令 $g$ + 当前多视角观测 $\mathbf{x}_t$ + 机器人本体感知状态 $\mathbf{s}_t$
- **Backbone**: [[Wan 2.1]]-14B 视频生成模型（LoRA 微调）
- **核心模块**: [[Action Chunking|Action-to-Control 投影]] 将动作块转化为视觉条件；[[Latent History Memory]] 维护跨轮次上下文
- **输出**: K 帧未来多视角观测 $\hat{\mathbf{x}}_{t+\Delta:t+K\Delta:\Delta}$
- **总参数**: 基于 Wan 2.1-14B

### 核心模块

#### 模块1: 闭合循环 Rollout 流水线

**设计动机**: 解决策略与世界模型之间的闭合循环接口问题

**具体实现**:

在推理时，策略与世界模型交替运行 $r = 1, \ldots, R$ 轮（$R=5$）：

1. [[VLA]] 策略从当前（想象的）观测预测动作块 $A_t$
2. PiL-World 将动作块转化为视觉控制条件 $C_t = \Gamma(A_{t\Delta,K})$
3. 模型生成 K 帧多视角未来帧
4. 末帧观测反馈给下一轮策略查询
5. 最近帧被重新编码为[[Latent History Memory|潜在历史记忆]] $\mathcal{H}_t$
6. 策略本体感知状态从步长对齐动作更新

每轮覆盖 $K \cdot \Delta = 45$ 个原始时间步，R=5 轮共覆盖 $5 \times 45 = 225$ 个时间步。

#### 模块2: Action-to-Control 投影

**设计动机**: 将抽象的关节空间动作命令转化为世界模型可理解的视觉条件

**具体实现**:

- 利用**机器人运动学和相机投影**，将绝对双臂动作指令转换为头部视角的抓手标记
- **标记位置** 编码投影后的抓手位置（来自关节角度 → 末端执行器位置 → 图像坐标）
- **标记大小** 编码对应的抓手状态（开/闭）
- 每个步长 $\Delta$ 生成一个控制帧，K 帧控制序列 $C_t = \{c_{t+\Delta}, c_{t+2\Delta}, \ldots, c_{t+K\Delta}\}$

#### 模块3: 潜在历史记忆（Latent History Memory）

**设计动机**: 维护跨 rollout 轮次的上下文，避免无记忆预测导致的时间不一致

**具体实现**:

使用 [[VAE]] 编码器 $E_\phi$ 将历史帧编码为潜在记忆：

- 当前帧潜变量: $Z_t^{v,0} = E_\phi(x_t^v)$
- 历史帧潜变量: $Z_t^{v,h} = E_\phi(\{x_t^v\}_{t \in \mathcal{I}_t^h})$（$H_h=5$ 帧）
- 未来目标潜变量: $Z_t^{v,f} = E_\phi(\{x_t^v\}_{t=t+\Delta:t+K\Delta:\Delta})$

历史记忆 $\mathcal{H}_t = \{Z_t^{v,h}\}_{v \in \mathcal{V}}$，训练时仅通过去噪目标监督 $Z_t^f$。

#### 模块4: 成功/失败混合训练数据

**设计动机**: 使世界模型能够准确模拟策略失败的情形，而非只生成成功轨迹

**具体实现**:

1. 在 RealSource World 数据集（14M+ 帧，11,428 集）上预训练
2. 在目标任务轨迹（含**成功演示**和**失败遥操作执行**）上用 [[LoRA]] 微调
3. 混合训练使模型接触目标到达和未到达两类轨迹，提升仿真保真度

---

## 关键公式

### 公式1: [[VLA|VLA 策略推理]]

$$
A_t = \{a_{t+1}, a_{t+2}, \ldots, a_{t+H_\pi}\} = \pi(\mathbf{x}_t, \mathbf{s}_t, g)
$$

**含义**: [[VLA]] 策略根据当前多视角观测、本体感知状态和任务指令，预测长度为 $H_\pi$ 的动作块

**符号说明**:
- $A_t$: 时间步 $t$ 的动作块
- $H_\pi = 50$: 动作块长度
- $\mathbf{x}_t = \{x_t^v\}_{v \in \mathcal{V}}$: 多视角观测集合
- $\mathbf{s}_t$: 机器人本体感知状态
- $g$: 任务指令

### 公式2: [[世界模型|World Model 预测]]

$$
\hat{\mathbf{x}}_{t+\Delta:t+K\Delta:\Delta} \sim W_\theta(\cdot | \mathbf{x}_t, \mathcal{H}_t, \Gamma(A_{t\Delta,K}), g)
$$

**含义**: 世界模型以当前观测、历史记忆、动作控制条件和任务指令为输入，生成 K 帧步长对齐的未来多视角观测

**符号说明**:
- $K = 15$: 预测帧数
- $\Delta = 3$: 步长（每帧对应 3 个原始时间步）
- $\Gamma(A_{t\Delta,K})$: Action-to-Control 投影函数
- $\mathcal{H}_t$: 潜在历史记忆
- $W_\theta$: PiL-World 世界模型

### 公式3: [[Action Chunking|Action-to-Control 投影]]

$$
C_t = \Gamma(A_{t\Delta,K}) = \{c_{t+\Delta}, c_{t+2\Delta}, \ldots, c_{t+K\Delta}\}
$$

**含义**: 将动作块中步长对齐的关节命令投影为视觉控制帧序列，每帧包含抓手位置和状态标记

**符号说明**:
- $C_t$: 视觉控制条件序列
- $c_{t+k\Delta}$: 第 $k$ 个控制帧（头部视角抓手标记图像）
- $\Gamma$: 运动学 + 相机投影复合函数

### 公式4: [[VAE|Latent History Memory 编码]]

$$
Z_t^{v,h} = E_\phi\left(\{x_t^v\}_{t \in \mathcal{I}_t^h}\right), \quad \mathcal{H}_t = \{Z_t^{v,h}\}_{v \in \mathcal{V}}
$$

**含义**: 用 [[VAE]] 编码器将最近 $H_h=5$ 帧历史观测编码为潜在历史记忆，跨 rollout 轮次传递上下文

**符号说明**:
- $E_\phi$: VAE 编码器
- $\mathcal{I}_t^h$: 历史帧索引集合
- $\mathcal{V}$: 相机视角集合

### 公式5: [[扩散模型|生成训练目标]]

$$
\tilde{Z}_{t,\lambda}^f = q_\lambda(Z_t^f, \varepsilon), \quad \mathcal{C}_t = (Z_t^0, Z_t^h, C_t, g)
$$

$$
\mathcal{L}_{gen} = \mathbb{E}_{t,\lambda,\varepsilon}\left[\|\Psi_\theta(\tilde{Z}_{t,\lambda}^f, \lambda, \mathcal{C}_t) - u_{t,\lambda}\|_2^2\right]
$$

**含义**: 在采样噪声水平 $\lambda$ 下，用去噪网络 $\Psi_\theta$ 预测去噪目标 $u_{t,\lambda}$，监督未来帧潜变量的生成

**符号说明**:
- $\lambda$: 噪声水平
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $\tilde{Z}_{t,\lambda}^f$: 加噪后的未来潜变量
- $\Psi_\theta$: 潜在预测网络（去噪器）
- $u_{t,\lambda}$: 去噪目标
- $\mathcal{C}_t$: 条件上下文（当前帧 + 历史帧 + 控制条件 + 任务指令）

### 公式6: [[Hallucination-Free Ratio|HFR 评估指标]]

$$
\text{HFR} = \frac{1}{M}\sum_m \frac{1}{N}\sum_i \frac{t_h^{(i,m)}}{T_i}
$$

**含义**: 衡量密集 rollout 中首次出现明显幻觉的归一化位置，值越高说明生成质量越好（幻觉出现越晚）

**符号说明**:
- $M$: 评估者数量
- $N$: rollout 数量
- $t_h^{(i,m)}$: 第 $m$ 个评估者在第 $i$ 个 rollout 中标注的首次幻觉帧位置
- $T_i$: 第 $i$ 个 rollout 的总帧数

### 公式7: [[实际-仿真成功率差距|ΔSR 评估指标]]

$$
|\Delta SR| = |SR_{imag} - SR_{real}|
$$

**含义**: 量化想象 rollout 和真实机器人执行之间的成功率差距，越低说明世界模型越忠实

**符号说明**:
- $SR_{imag}$: 在世界模型生成的想象环境中的成功率
- $SR_{real}$: 在真实机器人上的成功率

---

## 关键图表

### Figure 1: 研究动机

![Figure 1](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/Figure1.png)

**说明**: 展示 policy-in-the-loop 评估的动机。(a) 策略在环的 rollout 过程；(b) 开环预测 vs. 闭合循环预测的区别；(c) 一致 vs. 不一致想象 rollout 的对比。现有世界模型仅支持沿固定动作序列的开环预测，而 PiL-World 支持策略实时生成动作驱动的闭合循环评估。

### Figure 2: PiL-World 整体框架

![Figure 2](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/Figure2.jpg)

**说明**: PiL-World 的完整流水线概览。(a) 闭合循环 rollout 流程：策略与世界模型交替推理；(b) [[Action Chunking|Action-to-Control 投影]]：将关节命令转为视觉标记；(c) 成功/失败混合训练数据；(d) [[Latent History Memory|潜在历史记忆]]机制。

### Figure 3: 跨检查点成功率对齐

![Figure 3](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/rollout_success_trend.png)

**说明**: 在多个 VLA 检查点上，真实与想象成功率的 Pearson 相关系数达 **0.94**，说明 PiL-World 不仅能准确反映绝对成功率，还能可靠预测策略随训练步数的相对性能变化趋势。

### Figure 4: 逐帧 LPIPS 增益

![Figure 4](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/lpips_gain_pilworld_square_legend_v3.png)

**说明**: 在真值动作条件下的单步 LPIPS 逐帧增益对比。PiL-World 在整个预测时域内均优于 Ctrl-World，特别是在后期帧（随预测步长增加）差距更显著，体现了[[Latent History Memory|历史记忆]]的长程一致性优势。

### Figure 5a: Sort Cubes 任务 Rollout 示例

![Figure 5a](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/sortcubes_example.png)

**说明**: Sort Cubes 任务的 rollout 对比。红框标注 Ctrl-World 出现的不一致性（物体消失、跨视角冲突等幻觉），PiL-World 生成的序列保持物理一致。

### Figure 5b: Stack Bowls 任务 Rollout 示例

![Figure 5b](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/stackbowls_example.png)

**说明**: Stack Bowls 任务的 rollout 对比。同样展示 PiL-World 在高精度堆叠任务（$SR_{real}=96.7\%$）上更高的视觉保真度和过程一致性。

### Figure A.1: 任务定义与子任务分割

![Figure A.1](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/task_introduce.png)

**说明**: 三个目标任务（Sort Cubes、Stack Bowls、Stack Blocks）的示例和子任务分割方式。

### Figure A.2: Action-to-Control 投影流水线

![Figure A.2](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/action_projection.png)

**说明**: 详细展示从绝对关节空间命令到头部视角抓手标记的完整投影流水线（运动学正解 → 相机投影 → 标记渲染）。

### Figure A.3: RealSource World 单步预测示例

![Figure A.3](https://arxiv.org/html/2606.05773/2606.05773v1/Figures/realsource_qualitative_grid.png)

**说明**: 在大规模预训练数据集 RealSource World（14M+ 帧，11,428 集）上的定性单步预测效果。

### Table 1: 闭合循环 Rollout 评估结果（40k 步 VLA 检查点）

| 任务 | 方法 | $SR_{imag}$ | $|\Delta SR|$ ↓ | HFR ↑ |
|------|------|-------------|-----------------|-------|
| Sort Cubes ($SR_{real}=83.3\%$) | Ctrl-World | 11.5 | 71.8 | 39.5 |
| | **PiL-World** | **68.3** | **15.0** | **83.3** |
| Stack Bowls ($SR_{real}=96.7\%$) | Ctrl-World | 24.1 | 72.6 | 47.4 |
| | **PiL-World** | **92.5** | **4.2** | **83.9** |
| Stack Blocks ($SR_{real}=50.0\%$) | Ctrl-World | 4.9 | 45.1 | 37.7 |
| | **PiL-World** | **33.3** | **16.7** | **43.0** |

**关键发现**: PiL-World 将平均 $|\Delta SR|$ 从 63.2% 压缩至 12.0%，在所有任务上均大幅超越 Ctrl-World 基线。

### Table 2: 单步 LPIPS（真值动作条件）

| 任务 | 方法 | Overall ↓ | Head ↓ | Wrist Avg. ↓ |
|------|------|-----------|--------|-------------|
| Sort Cubes | Ctrl-World | 0.1454 | 0.1030 | 0.1666 |
| | **PiL-World** | **0.0965** | **0.0597** | **0.1148** |
| Stack Bowls | Ctrl-World | 0.1366 | 0.0959 | 0.1569 |
| | **PiL-World** | **0.1100** | **0.0597** | **0.1351** |
| Stack Blocks | Ctrl-World | 0.1277 | 0.0885 | 0.1474 |
| | **PiL-World** | **0.1208** | **0.0617** | **0.1503** |

**关键发现**: 相对 LPIPS 提升：Sort Cubes 33.7%，Stack Bowls 19.5%，Stack Blocks 5.4%。头部视角提升尤为显著。

### Table A.2: 潜在历史记忆消融实验（LPIPS）

| 配置 | Sort Cubes | Stack Bowls | Stack Blocks |
|------|------------|-------------|--------------|
| w/o Latent History Memory | 0.3146 | 0.2759 | 0.3403 |
| **Full PiL-World** | **0.0965** | **0.1100** | **0.1208** |

**关键发现**: 去除历史记忆后 LPIPS 急剧劣化（Sort Cubes 恶化 226%），证明[[Latent History Memory|潜在历史记忆]]是 PiL-World 视觉一致性的核心组件。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RealSource World | 14M+ 帧，11,428 集 | 真实机器人多视角视频 | 世界模型预训练 |
| Sort Cubes | 100 训练 / 20 测试集，60 子任务 | 双臂分类操作 | 微调 + 评估 |
| Stack Bowls | 100 训练 / 18 测试集，54 子任务 | 双臂精确堆叠 | 微调 + 评估 |
| Stack Blocks | 200 训练 / 20 测试集，40 子任务 | 双臂积木堆叠 | 微调 + 评估 |

### 实现细节

- **VLA 策略**: [[π₀]]（Black et al., 2024），仅在成功演示上微调，评估期间冻结
- **世界模型 Backbone**: [[Wan 2.1]]-14B，用 [[LoRA]] 微调
- **输入/输出分辨率**: 224×224（与 VLA 匹配）
- **动作块长度**: $H_\pi = 50$
- **动作维度**: 14-D（绝对关节命令 + 抓手状态）
- **历史帧数**: $H_h = 5$
- **预测帧数**: $K = 15$，步长 $\Delta = 3$
- **最大 Rollout 轮数**: $R = 5$
- **预训练**: 22 epochs，64 张 H20 GPU
- **微调**: 20 epochs，8 张 H20 GPU
- **滑动窗口提取**: 9 帧步长

### 可视化结果

PiL-World 在 Sort Cubes 和 Stack Bowls 任务上的想象 rollout 保持跨帧、跨视角的物理一致性。Ctrl-World 的失败案例（红框标注）包括物体突然消失、跨视角冲突、非物理接触等典型幻觉类型。

---

## 批判性思考

### 优点

1. **问题定义清晰**: 明确识别开环 vs. 闭合循环评估的本质差距，提出具体解决方案
2. **多维度评估**: 结合 $|\Delta SR|$、HFR、LPIPS 三类指标，全面量化评估质量
3. **成功/失败混合训练**: 将失败轨迹纳入训练，使世界模型能准确模拟策略失败场景，是本文关键设计选择

### 局限性

1. **任务范围有限**: 仅在三个双臂操作任务上验证，泛化性待检验
2. **接触丰富任务挑战**: 作者承认接触丰富操作仍较困难（如 Stack Blocks 上 LPIPS 提升仅 5.4%）
3. **人工标注依赖**: HFR 和想象成功率均依赖人工标注，规模化成本高
4. **严重遮挡下的投影局限**: 头部视角在严重遮挡情况下 Action-to-Control 投影效果下降

### 潜在改进方向

1. 探索自动化 HFR 计算，减少人工标注依赖
2. 扩展到移动操作、腿足机器人等更多任务类型
3. 将世界模型用于在线策略改进（不仅评估，也用于 MBRL 训练）

### 可复现性评估

- [ ] 代码开源（项目主页存在但未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、数据集规模均有详细说明）
- [x] 数据集可获取（RealSource World 需申请，目标任务数据集随论文）

---

## 关联笔记

### 基于

- [[π₀]]: 作为评估对象的 VLA 策略，PiL-World 在其基础上构建评估框架
- [[Wan 2.1]]: 世界模型的视频生成 Backbone

### 对比

- [[OSCAR|Ctrl-World]]: 主要对比基线，代表现有开环世界模型评估方法

### 方法相关

- [[世界模型|World Model]]: 核心框架类别
- [[VLA]]: 被评估的策略类型
- [[Action Chunking]]: VLA 输出动作块的核心机制
- [[VAE]]: 用于历史记忆编码的生成模型组件
- [[LoRA]]: 世界模型微调方法
- [[扩散模型]]: 世界模型视频生成的底层技术
- [[Latent History Memory]]: 跨轮次上下文维护机制

### 硬件/数据相关

- RealSource World: 14M+ 帧真实机器人视频预训练数据集

---

## 速查卡片

> [!summary] PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation
> - **核心**: 将 VLA 策略与世界模型闭合循环交替推理，实现低成本策略评估
> - **方法**: Chunk-wise 预测 + Action-to-Control 投影 + 潜在历史记忆 + 成功/失败混合训练
> - **结果**: 实际-仿真成功率差距从 63.2% → 12.0%，跨检查点 Pearson 相关 0.94
> - **代码**: [pil-world.github.io](https://pil-world.github.io)

---

*笔记创建时间: 2026-06-05*
