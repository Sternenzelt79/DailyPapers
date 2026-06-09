---
title: "MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models"
method_name: "MemoryVLApp"
authors: [Hao Shi, Weiye Li, Bin Xie, Yulin Wang, Renping Zhou, Tiancai Wang, Xiangyu Zhang, Ping Luo, Gao Huang]
year: 2026
venue: arXiv
tags: [vla, temporal-modeling, memory, world-model, diffusion-policy, robot-manipulation, imitation-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.09827
created: 2026-06-09
---

# 论文笔记：MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University (BNRist)、University of Hong Kong、Dexmal、StepFun |
| 日期 | June 2026 |
| 项目主页 | [shihao1895.github.io/MemoryVLA-PP-Web](https://shihao1895.github.io/MemoryVLA-PP-Web) |
| 对比基线 | [[CogACT]]、[[MemoryVLA]]、[[pi0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09827) / Code: N/A |

---

## 一句话总结

> MemoryVLA++ 通过感知-认知记忆库、潜在空间世界模型想象和扩散动作专家，将 [[VLA|VLA 模型]] 的时间建模从"当前感知"扩展到"过去记忆 + 现在 + 未来想象"的全时域范围，在真实机器人记忆/想象依赖任务上分别取得 +26%/+28% 的显著提升。

---

## 核心贡献

1. **全时域 VLA 框架**: 将 VLA 的时间感知从单帧现在扩展到"过去（记忆）+ 当前（感知）+ 未来（想象）"三维时域，首次在真实机器人平台上系统验证全时域建模的有效性。
2. **感知-认知记忆库 (PCMB)**: 分别存储高层语义认知 token 和低层视觉感知 token 的历史记忆，利用冗余感知合并策略保持记忆紧凑，通过带时间位置编码的交叉注意力检索相关历史。
3. **潜在空间世界模型想象**: 通过部分去噪（1 步即可）在潜在空间生成未来状态，配合记忆引导的想象融合模块将未来线索注入当前决策，无需完整去噪步骤即可捕获语义动态。
4. **时序感知扩散动作专家**: [[Diffusion Transformer|DiT]] 通过认知注意力和感知注意力双层结构，分别注入全时域语义和视觉细节，生成时间连贯的动作序列。

---

## 问题背景

### 要解决的问题

[[VLA|机器人 VLA 策略]]通常只处理**当前帧**，无法有效利用任务历史信息或预见未来状态，导致在需要记忆和长程规划的任务中失败——例如：先按按钮解锁再操作、传送带高速抓取时需预判物体位置。

### 现有方法的局限

- 单帧 VLA（如 [[RT-2]]、[[CogACT]]）：完全忽略时序上下文；
- 仅有记忆（如 MemoryVLA、HAMLET）：只看历史，不预见未来；
- 仅有预测（如 SuSIE、ForeAct）：不保留历史交互细节；
- 记忆和预测相互独立，缺乏统一时序建模。

### 本文的动机

受认知科学启发，人类的决策依赖**工作记忆**（当前感知）、**情景记忆**（过去经验）和**内部模型**（未来想象）三者的协同。MemoryVLA++ 将三者统一为机器人策略的时序感知框架，并在潜空间中以历史记忆指导未来想象，使世界模型更聚焦决策相关信息。

---

## 方法详解

### 模型架构

MemoryVLA++ 采用**四阶段编码-检索-想象-动作**架构：

- **输入**: 语言指令 $L$ + 当前图像观测 $I$ + 历史帧序列
- **Backbone**: 7B Prismatic [[VLM]]（DINOv2 + SigLIP + LLaMA-7B），预训练于 Open-X Embodiment
- **核心模块**:
  - [[Perceptual-Cognitive Memory Bank|感知-认知记忆库（PCMB）]] 存储和检索历史感知/认知 token
  - [[世界模型|潜空间世界模型]] 基于 Stable Video Diffusion (1.5B) 想象未来潜 token
  - [[Diffusion Policy|扩散动作专家（DiT）]] 条件化时序 token 生成动作
- **输出**: [[Action Chunking|动作块]] $\mathcal{A} = (a_1, \ldots, a_T)$，单臂 $a_t \in \mathbb{R}^7$（6-DoF 末端 + 夹爪）
- **总参数**: VLM 约 7B，世界模型 1.5B，动作专家约 300M

![Figure 2](https://arxiv.org/html/2606.09827/x2.png)

**说明**: 从左到右展示三种范式进化：(1) 典型 VLA 仅用当前帧；(2) MemoryVLA 加入历史记忆；(3) MemoryVLA++ 同时利用历史记忆和未来想象实现完整时序建模。

---

### 核心模块

#### 模块 1：视觉-语言-认知编码（工作记忆）

**设计动机**: 同时保留细粒度视觉特征和高层语义，供后续记忆检索和动作生成使用。

**具体实现**:

1. [[DINOv2]] 和 [[SigLIP]] 双分支并行提取视觉特征；
2. **SE-bottleneck 压缩** 生成感知 token $p \in \mathbb{R}^{N_p \times d_p}$（细粒度视觉细节）；
3. [[LLaMA]] 7B 处理语言-图像融合生成认知 token $c \in \mathbb{R}^{1 \times d_c}$（高层语义）；
4. 工作记忆 $M_{wk} = \{p, c\}$ 作为当前帧表征。

![Figure 3](https://arxiv.org/html/2606.09827/x3.png)

**说明**: 完整系统流程。左：VLM 编码生成工作记忆；中上：PCMB 检索历史记忆并融合；中下：世界模型想象未来潜 token；右：扩散动作专家条件化时序 token 生成动作序列。

---

#### 模块 2：感知-认知记忆库（PCMB）

**设计动机**: 利用[[Cross-Attention|跨注意力]]检索历史上下文，同时存储低级感知细节与高级语义，并通过冗余整合避免记忆重复。

**具体实现**:

- 记忆库 $M_{pcmb} = \{m^x | x \in \{p, c\}\}$，其中 $m^x = \{m^x_i \in \mathbb{R}^{N_x \times d_x}\}_{i=1}^{L}$，容量 $L=16$（可调）
- **记忆检索**：以当前工作记忆为 query，历史记忆为 key/value，用时间步位置编码区分历史
- **门控融合**：可学习门控权重自适应混合当前与历史表征
- **冗余感知整合**：计算相邻记忆项的余弦相似度，合并最相似的相邻项，防止 FIFO 淘汰关键信息

![Figure 4](https://arxiv.org/html/2606.09827/x4.png)

**说明**: PCMB 三步操作——(a) 跨注意力检索；(b) 自适应门控融合；(c) 冗余感知整合（token merge 替代 FIFO）。感知和认知双流并行处理后分别融合。

---

#### 模块 3：潜空间世界模型（Imagination）

**设计动机**: 在潜空间而非像素空间预测未来，计算高效；以历史记忆指导想象，抑制任务无关噪声。

**具体实现**:

1. 以当前帧 $I$ 和语言 $L$ 为条件，微调 Stable Video Diffusion 预测未来 $K$ 帧视频；
2. **部分去噪**（仅 1 步即可）提取多尺度 U-Net 特征 $\{U_s\}_{s=1}^{S}$；
3. **[[FPN]] 聚合**多尺度特征为 $z \in \mathbb{R}^{K \times N_z \times d_p}$，加时间步位置编码得 $\bar{z}$；
4. **空间注意力**：$\hat{q}_k = \operatorname{SpatAttn}(q_k, \bar{z}_k, \bar{z}_k)$，每帧独立聚合空间信息；
5. **时序注意力**：$z_{img} = \operatorname{FFN}(\operatorname{TempAttn}(\hat{q}_{1:K}))$，聚合多帧时序信息；
6. **记忆引导整合**：以融合后的感知 token $\tilde{p}$ 为 query，对想象 token $z_{img}$ 门控过滤。

![Figure 5](https://arxiv.org/html/2606.09827/x5.png)

**说明**: 世界模型部分去噪流程，展示空间-时序注意力聚合多帧潜特征，以及记忆引导整合抑制无关预测的机制。

---

#### 模块 4：时序感知扩散动作专家

**设计动机**: [[Diffusion Transformer|DiT]] 同时接收感知 token（空间细节）和认知 token（语义），生成时序一致的多步动作。

**具体实现**:

- 完整时序 token：$F_{temp} = \{\bar{p}, \tilde{c}\}$（融合了记忆和想象）
- **认知注意力层**（Cognition-Attention）：语义引导，$h_c = \operatorname{CogAttn}([\tilde{c} + TE(\tau); \mathcal{A}_\tau])$
- **感知注意力层**（Perception-Attention）：细粒度视觉注入，$\hat{A}_0 = \operatorname{FFN}(\operatorname{PerAttn}(h_c, \bar{p}, \bar{p}))$
- [[DDIM]] 采样，10 步推理，CFG scale = 1.5

![Figure 6](https://arxiv.org/html/2606.09827/x6.png)

**说明**: DiT 架构中认知注意力和感知注意力两层叠加，分别注入语义上下文和视觉细节，在 10 步 DDIM 去噪中生成动作序列。

---

## 关键公式

### 公式 1：[[Action Chunking|动作预测目标]]

$$
\mathcal{A} = (a_1, \ldots, a_T) = \pi(I, L)
$$

$$
a_t = [\Delta x, \Delta y, \Delta z, \Delta\theta_x, \Delta\theta_y, \Delta\theta_z, g]^\top
$$

**含义**: VLA 策略 $\pi$ 接收图像 $I$ 和语言指令 $L$，输出动作块；单臂动作为 7 维末端执行器增量加夹爪状态。

**符号说明**:
- $\mathcal{A}$: 动作块序列
- $\Delta x, \Delta y, \Delta z$: 末端执行器位置增量
- $\Delta\theta_x, \Delta\theta_y, \Delta\theta_z$: 旋转增量（欧拉角）
- $g$: 夹爪开合状态

---

### 公式 2：[[Perceptual-Cognitive Memory Bank|工作记忆定义]]

$$
M_{wk} = \{p \in \mathbb{R}^{N_p \times d_p},\ c \in \mathbb{R}^{1 \times d_c}\}
$$

**含义**: 工作记忆由当前帧的感知 token $p$（细粒度视觉，$N_p$ 个 token）和认知 token $c$（高层语义，单个向量）组成。

**符号说明**:
- $p$: 感知 token，来自 SE-bottleneck 压缩
- $c$: 认知 token，来自 LLaMA-7B 输出
- $N_p$: 感知 token 数量
- $d_p, d_c$: 对应特征维度

---

### 公式 3：[[Cross-Attention|记忆检索]]

$$
\hat{H}^x = \operatorname{softmax}\!\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x, \quad x \in \{p, c\}
$$

**含义**: 以当前工作记忆为 query，历史记忆库为 key/value，通过缩放点积注意力检索相关历史上下文，感知和认知双流分别检索。

**符号说明**:
- $q^x$: 当前帧 token 的 query 投影
- $K^x, V^x$: 历史记忆库的 key/value 投影
- $d_x$: token 维度（缩放因子）
- $\hat{H}^x$: 检索到的历史上下文

---

### 公式 4：[[Perceptual-Cognitive Memory Bank|门控记忆融合]]

$$
g^x = \sigma\!\left(\operatorname{MLP}\!\left(\operatorname{concat}[x, H^x]\right)\right)
$$

$$
\tilde{x} = g^x \odot H^x + (1 - g^x) \odot x
$$

**含义**: 可学习门控权重 $g^x$ 自适应决定当前 token 与历史检索结果的混合比例，输出融合后的时序增强表征 $\tilde{x}$。

**符号说明**:
- $\sigma$: Sigmoid 激活
- $g^x \in [0, 1]^{d_x}$: 元素级门控权重
- $\odot$: 元素级乘法
- $x$: 当前帧 token；$H^x$: 检索历史；$\tilde{x}$: 融合结果

---

### 公式 5：[[Perceptual-Cognitive Memory Bank|冗余感知整合]]

$$
i^*_x = \arg\max_{i=1,\ldots,L-1} \cos(m^x_i,\ m^x_{i+1})
$$

$$
m^x_{i^*_x} \leftarrow \tfrac{1}{2}\!\left(m^x_{i^*_x} + m^x_{i^*_x+1}\right)
$$

**含义**: 每次记忆库满时，找到余弦相似度最高的相邻记忆对并取均值合并，而非简单丢弃最旧的条目，保留更长时间跨度的历史信息。

**符号说明**:
- $\cos(\cdot, \cdot)$: 余弦相似度
- $m^x_i$: 记忆库第 $i$ 条目
- $L$: 记忆库容量（默认 16）

---

### 公式 6：[[世界模型|世界模型扩散训练损失]]

$$
x_\tau = \sqrt{\bar{\alpha}_\tau} x_0 + \sqrt{1 - \bar{\alpha}_\tau}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

$$
\mathcal{L}_{wm} = \mathbb{E}_{(I, L, x_0) \sim \mathcal{D}_{wm},\, \varepsilon,\, \tau}\!\left[\left\|\mathcal{W}_\phi(x_\tau, \tau, I, L) - x_0\right\|_2^2\right]
$$

**含义**: 标准扩散去噪目标，训练世界模型 $\mathcal{W}_\phi$ 在条件 $(I, L)$ 下预测干净潜变量 $x_0$；在推理时部分去噪（1 步）提取多尺度特征。

**符号说明**:
- $x_0$: 目标未来帧的潜变量
- $x_\tau$: $\tau$ 步加噪后的潜变量
- $\bar{\alpha}_\tau$: 噪声调度累积系数
- $\mathcal{W}_\phi$: 世界模型（基于 Stable Video Diffusion）
- $\mathcal{D}_{wm}$: 世界模型训练数据集

---

### 公式 7：[[FPN|FPN 特征聚合]]

$$
z = \operatorname{FPN}(\{U_s\}_{s=1}^{S}), \quad z \in \mathbb{R}^{K \times N_z \times d_p}
$$

$$
\bar{z} = z + e_{time}
$$

**含义**: 从 U-Net 中间特征 $\{U_s\}$ 提取多尺度特征并通过 FPN 聚合为统一维度的想象潜 token，加时间步位置编码区分未来帧序列。

**符号说明**:
- $S$: U-Net 特征尺度数
- $K$: 想象未来帧数（预测视频帧数）
- $N_z$: 每帧空间 token 数
- $e_{time}$: 时间步位置编码

---

### 公式 8：[[世界模型|空间-时序注意力]]

$$
\hat{q}_k = \operatorname{SpatAttn}(q_k, \bar{z}_k, \bar{z}_k), \quad k = 1, \ldots, K
$$

$$
z_{img} = \operatorname{FFN}(\operatorname{TempAttn}(\hat{q}_{1:K}))
$$

**含义**: 先对每帧独立做空间注意力（聚合当前帧内空间信息），再对所有帧做时序注意力（聚合跨帧时序信息），最终得到多帧聚合的想象表征 $z_{img}$。

**符号说明**:
- $q_k$: 第 $k$ 帧感知 token query
- $\bar{z}_k$: 第 $k$ 帧想象 token（含时间编码）
- $\operatorname{SpatAttn}$: 空间交叉注意力
- $\operatorname{TempAttn}$: 时序自注意力

---

### 公式 9：[[Perceptual-Cognitive Memory Bank|记忆引导想象整合]]

$$
h = \operatorname{FFN}(\operatorname{CrossAttn}(\tilde{p}, z_{img}, z_{img}))
$$

$$
g = \sigma(\operatorname{MLP}(\operatorname{concat}[h, \tilde{p}]))
$$

$$
\bar{p} = g \odot \tilde{p} + (1 - g) \odot h
$$

**含义**: 以记忆增强后的感知 token $\tilde{p}$ 为 query 过滤想象 token，门控融合保留决策相关的想象信息，抑制无关噪声预测。

**符号说明**:
- $\tilde{p}$: 记忆融合后的感知 token
- $z_{img}$: 世界模型想象 token
- $h$: 想象-感知交叉注意力结果
- $\bar{p}$: 最终时序增强感知 token

---

### 公式 10：[[Diffusion Policy|扩散动作专家推理]]

$$
h_c = \operatorname{CogAttn}([\tilde{c} + TE(\tau);\ \mathcal{A}_\tau])
$$

$$
\hat{A}_0 = \operatorname{FFN}(\operatorname{PerAttn}(h_c, \bar{p}, \bar{p}))
$$

**含义**: DiT 中先用认知注意力注入语义引导（$\tilde{c}$ 为记忆增强认知 token，$TE(\tau)$ 为扩散时间步编码），再用感知注意力注入细粒度视觉信息，最终预测干净动作 $\hat{A}_0$。

**符号说明**:
- $\tilde{c}$: 记忆增强认知 token
- $TE(\tau)$: 扩散时间步 $\tau$ 的位置编码
- $\mathcal{A}_\tau$: $\tau$ 步加噪动作序列
- $\bar{p}$: 时序增强感知 token
- $\hat{A}_0$: 预测的干净动作序列

---

## 关键图表

### Figure 1: 动机示例

![Figure 1](https://arxiv.org/html/2606.09827/x1.png)

**说明**: 两个典型任务说明需求——(左) 按钮需记忆历史状态判断解锁顺序；(右) 动态传送带需预测未来抓取时机。二者分别对应记忆依赖和想象依赖场景。认知科学三要素（工作记忆/情景记忆/内部模型）映射到 MemoryVLA++ 三模块。

---

### Figure 7: 实验设置全景

![Figure 7](https://arxiv.org/html/2606.09827/x7.png)

**说明**: 涵盖 5 个仿真环境（Libero、SimplerEnv、Mikasa-Robo、Calvin、Libero-Plus）和 3 个真实机器人平台（Franka、Dual-ARX5、WidowX），分通用操作、记忆依赖、想象依赖三类任务评测。

---

### Figure 8: 真实机器人平台

![Figure 8](https://arxiv.org/html/2606.09827/x8.png)

**说明**: 三种硬件平台展示：(1) Franka Panda 单臂（通用操作任务）；(2) Dual-ARX5 双臂（记忆依赖长程任务）；(3) WidowX（想象依赖传送带任务）。

---

### Figure 9: OOD 鲁棒性测试

![Figure 9](https://arxiv.org/html/2606.09827/x9.png)

**说明**: 在背景变化和光照变化两种 OOD 条件下测试，展示模型对分布外视觉输入的鲁棒性。

---

### Figure 10: 记忆机制可视化

![Figure 10](https://arxiv.org/html/2606.09827/x10.png)

**说明**: 真实任务和仿真任务中记忆检索的注意力权重热图，验证 PCMB 能聚焦到任务关键的历史帧（如按下按钮的时刻）。

---

### Figure 11: 真实机器人任务可视化

![Figure 11](https://arxiv.org/html/2606.09827/x11.png)

**说明**: 通用操作（Franka 抓取/堆叠）和长程任务（Dual-ARX5 换菜/清桌）的执行轨迹可视化。

---

### Figure 12: 世界模型想象质量

![Figure 12](https://arxiv.org/html/2606.09827/x12.png)

**说明**: 不同去噪步数下的想象帧与真实未来帧对比，验证仅 1 步去噪的潜特征已足够传递关键空间信息（无需完整重建像素级视频）。

---

### Table 1：Libero 通用操作基准（完整对比）

| Method | Spatial | Object | Goal | Long-10 | Long-90 | Avg. |
|--------|---------|--------|------|---------|---------|------|
| DP | 78.3 | 92.5 | 68.3 | 50.5 | — | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | — | 75.1 |
| MDT | 78.5 | 87.5 | 73.5 | 64.8 | — | 76.1 |
| UniACT | 77.0 | 87.0 | 77.0 | 70.0 | 73.0 | 76.8 |
| MaIL | 74.3 | 90.1 | 81.8 | 78.6 | — | 83.5 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | — | 81.8 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 83.1 | 85.0 |
| TriVLA | 91.2 | 93.8 | 89.8 | 73.2 | — | 87.0 |
| SmolVLA | 93.0 | 94.0 | 91.0 | 77.0 | — | 88.8 |
| 4D-VLA | 93.8 | 92.8 | 95.6 | 86.5 | — | 92.2 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | — | 92.6 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| UniVLA | 96.4 | 98.0 | 90.8 | 89.6 | — | 93.7 |
| GR00T-N1.5 | 94.4 | 97.6 | 93.0 | 90.6 | — | 93.9 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | — | 94.2 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 95.6 | 96.5 |
| **MemoryVLA++** | **99.8** | **100.0** | **98.2** | **96.0** | **97.8** | **98.4 (+5.2)** |

**说明**: MemoryVLA++ 在全部 5 个子集上均达到 SOTA，平均成功率 98.4%，较前代 MemoryVLA 提升 1.9 个百分点，Long-10/Long-90 尤为突出。

---

### Table 2：SimplerEnv 零样本评估（完整对比）

| Method | Spoon Towel | Carrot Plate | Stack Cube | Eggplant Basket | Avg. |
|--------|-------------|--------------|-----------|-----------------|------|
| RT-1-X | 0.0 | 4.2 | 0.0 | 0.0 | 1.1 |
| OpenVLA | 4.2 | 0.0 | 0.0 | 12.5 | 4.2 |
| Octo-Base | 15.8 | 12.5 | 0.0 | 41.7 | 17.5 |
| TraceVLA | 12.5 | 16.6 | 16.6 | 65.0 | 27.7 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | 100.0 | 42.7 |
| Magma | 37.5 | 29.2 | 20.8 | 91.7 | 44.8 |
| π₀-FAST | 62.5 | 29.2 | 20.8 | 83.3 | 49.0 |
| DreamVLA | 45.8 | 45.8 | 25.0 | 87.5 | 51.0 |
| VideoVLA | 75.0 | 20.8 | 45.8 | 70.8 | 53.1 |
| Mimic-video | 41.7 | 54.2 | 29.2 | 100.0 | 56.3 |
| CogACT | 58.3 | 45.8 | 29.2 | 95.8 | 57.3 |
| CronusVLA | 66.7 | 54.2 | 20.8 | 100.0 | 60.4 |
| GR00T-N1.5 | 75.3 | 54.3 | 57.0 | 61.3 | 61.9 |
| π₀ | 84.6 | 55.8 | 47.9 | 85.4 | 68.4 |
| MemoryVLA | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 |
| **MemoryVLA++** | **83.3** | 66.7 | 45.8 | **100.0** | **73.9 (+16.6)** |

**说明**: 平均成功率 73.9%，较 π₀ 提升 5.5 个百分点，Eggplant Basket 达满分。

---

### Table 3：Mikasa-Robo 时序任务（完整对比）

| Method | SGT | IM | RC3 | RC5 | RC9 | Avg. |
|--------|-----|----|-----|-----|-----|------|
| CronusVLA (1) | 32 | 5 | 31 | 13 | 9 | 18.0 |
| SpatialVLA (1) | 23 | 27 | 27 | 17 | 11 | 21.0 |
| OpenVLA-OFT (1) | 47 | 14 | 59 | 16 | 6 | 28.4 |
| π₀ (1) | 33 | 42 | 35 | 22 | 15 | 29.4 |
| Octo (10) | 46 | 39 | 45 | 17 | 11 | 31.6 |
| MemoryVLA (1) | 88 | 24 | 44 | 30 | 20 | 41.2 |
| **MemoryVLA++ (1)** | **97** | 40 | 50 | 19 | 16 | **44.4 (+15.0)** |

**说明**: SGT=ShellGameTouch，IM=InterceptMedium，RC3/5/9=RememberColor3/5/9。记忆依赖的 SGT 大幅领先（97%），仅用 1 帧输入即超越使用 10 帧历史的 Octo。

---

### Table 4：Calvin 长程任务 — ABC→D 零样本（完整对比）

| Method | 1-Task | 2-Tasks | 3-Tasks | 4-Tasks | 5-Tasks | Avg. Len. |
|--------|--------|---------|---------|---------|---------|-----------|
| DP | 40.2 | 12.3 | 2.6 | 0.8 | 0.0 | 0.56 |
| RT-1 | 53.3 | 22.2 | 9.4 | 3.8 | 1.3 | 0.90 |
| Roboflamingo | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| CogACT | 83.8 | 72.9 | 64.0 | 55.9 | 48.0 | 3.25 |
| OpenVLA-OFT | 89.1 | 79.4 | 67.4 | 59.8 | 51.5 | 3.47 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| π₀ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| MemoryVLA | 94.8 | 87.4 | 81.4 | 75.9 | 69.4 | 4.09 |
| **MemoryVLA++** | **95.6** | **90.2** | **85.7** | **81.7** | **76.1** | **4.29 (+1.04)** |

**说明**: 平均完成任务数 4.29（满分 5），领先 π₀ +0.37，Task 4-5 的性能衰减远低于基线，体现时序一致性的优势。

---

### Table 5：Libero-Plus 鲁棒性评估（完整对比）

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|--------|--------|-------|----------|-------|------------|-------|--------|------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| DP | 1.6 | 32.3 | 77.0 | 25.0 | 19.8 | 20.3 | 42.2 | 31.7 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 43.9 |
| Fast-WAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| MemoryVLA | 42.7 | 44.9 | 84.4 | 92.8 | 95.0 | 62.1 | 84.7 | 70.2 |
| **MemoryVLA++ (Zero-Shot)** | 36.4 | **68.9** | **88.7** | **93.8** | 90.6 | 63.5 | 83.8 | **73.1** |
| π₀ (Fine-Tuned) | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| OpenVLA-OFT (Fine-Tuned) | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.6 |
| MemoryVLA (Fine-Tuned) | 91.4 | 48.6 | 79.4 | 95.2 | 95.3 | 94.0 | 75.7 | 81.9 |
| **MemoryVLA++ (Fine-Tuned)** | **96.8** | 49.7 | 71.0 | **96.6** | **97.0** | **96.0** | 78.6 | **82.7** |

**说明**: 零样本 73.1%，微调后 82.7%。Robot 变体鲁棒性（零样本 68.9%）大幅领先其他方法；Camera 扰动零样本最弱（36.4%）但微调后最强（96.8%）。

---

### Table 6：真实机器人评估（成功率 %）

**(a) 通用操作**

| Method | Insert Circle | Egg Pan | Egg Oven | Stack Cups | Stack Blocks | Pick Fruits | Avg. |
|--------|--------------|---------|----------|-----------|-------------|------------|------|
| CogACT | 80 | 67 | 60 | 93 | 80 | 76 | 76 |
| **MemoryVLA++** | **87** | **80** | **80** | **93** | **87** | **84** | **85 (+9)** |

**(b) 长程记忆依赖任务**

| Method | Push Buttons | Change Food | Guess Where | Clean Table | Pick Place | Clean Rest. | Avg. |
|--------|-------------|-------------|------------|------------|-----------|-----------|------|
| CogACT | 15 | 47 | 40 | 67 | 90 | 84 | 57 |
| **MemoryVLA++** | **58** | **85** | **72** | **84** | **100** | **96** | **83 (+26)** |

**(c) 长程想象依赖任务**

| Method | Conv-Low | Conv-Mid | Conv-High | Conv-Scan | Bag Pack | Avg. |
|--------|----------|----------|-----------|-----------|----------|------|
| CogACT | 61 | 55 | 48 | 42 | 37 | 49 |
| MemoryVLA | 75 | 68 | 58 | 63 | 60 | 65 (+16) |
| **MemoryVLA++** | **84** | **80** | **72** | **75** | **73** | **77 (+28)** |

**说明**: 记忆依赖任务提升最显著（+26%），Push Buttons 任务从 15% 跃升至 58%；想象依赖任务（传送带速度递增）MemoryVLA++ 比 MemoryVLA 再提 +12%，验证想象模块的独立贡献。

---

### Table 7：记忆模块消融（完整数值）

**(a) 记忆长度**

| 记忆长度 | SimplerEnv | Libero Long-90 | Real-Temporal |
|----------|-----------|----------------|--------------|
| Small | 67.7 | 94.2 | 78 |
| **Default (16)** | **71.9** | **95.6** | **84** |
| Large | 67.7 | 95.6 | 81 |

**(b) 记忆融合方式**

| 融合方式 | SimplerEnv | Libero Long-90 | Real-Temporal |
|----------|-----------|----------------|--------------|
| Add | 67.7 | 93.8 | 78 |
| **Gate** | **71.9** | **95.6** | **84** |

**(c) 记忆合并策略**

| 合并策略 | SimplerEnv | Libero Long-90 | Real-Temporal |
|----------|-----------|----------------|--------------|
| FIFO | 66.7 | 94.9 | 76 |
| **Token Merge** | **71.9** | **95.6** | **84** |

**(d) 时间位置编码**

| 配置 | SimplerEnv |
|------|-----------|
| w/o Timestep PE | 69.8 |
| **w/ Timestep PE** | **71.9** |

**(e) 记忆类型**

| 记忆类型 | SimplerEnv |
|----------|-----------|
| Cognitive Memory only | 63.5 |
| Perceptual Memory only | 64.6 |
| **Both (Perceptual + Cognitive)** | **71.9** |

**关键发现**: 门控融合 > 加法，Token Merge > FIFO，感知+认知双路 > 任一单路；默认记忆长度 16 在大多数任务下最优。

---

### Table 8：想象模块消融（Mikasa-Robo 平均成功率）

**(a) 去噪步数**

| 去噪步数 | Avg. Succ. |
|----------|-----------|
| **1 step** | **44.4** |
| 3 steps | 44.6 |
| 5 steps | 43.6 |

**(b) 想象时域**

| 想象时域 K | Avg. Succ. |
|-----------|-----------|
| 4 | 43.4 |
| 8 | 43.8 |
| **16** | **44.4** |

**(c) 世界模型更新策略**

| 策略 | Avg. Succ. |
|------|-----------|
| w/o Freeze | 42.8 |
| **w/ Freeze** | **44.4** |

**(d) 融合策略**

| 融合策略 | Avg. Succ. |
|----------|-----------|
| Add | 41.2 |
| **Memory-Guided** | **44.4** |

**关键发现**: 仅 1 步去噪的潜特征已足够（略低于 3 步但差异极小）；K=16 最优；冻结世界模型权重 > 继续微调；记忆引导融合 > 简单加法。

---

### Table 9：世界模型量化评估

| 数据集 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | EPE ↓ |
|--------|--------|--------|---------|-------|-------|
| Libero | 20.26 | 0.820 | 0.182 | 101.93 | 0.5104 |
| Bridge | 17.44 | 0.712 | 0.276 | 132.31 | 1.4999 |
| Mikasa-Robo | 26.39 | 0.838 | 0.174 | 189.38 | 0.1540 |
| Calvin | 22.22 | 0.833 | 0.185 | 29.69 | 0.2049 |
| Real-Conv. Pick | 16.91 | 0.764 | 0.250 | 128.18 | 1.6276 |
| Real-Conv. Scan | 22.34 | 0.822 | 0.182 | 82.92 | 0.4163 |
| Real-Bag Pack | 16.94 | 0.768 | 0.266 | 70.56 | 1.7672 |
| **Average** | **20.36** | **0.794** | **0.216** | **105.00** | **0.8829** |

**说明**: Mikasa-Robo 质量最高（PSNR 26.39），Bridge 和真实任务质量偏低（~17），想象质量与场景复杂度相关。

---

### Table 10：推理效率对比

| 方法 | 延迟（RTX 4090） | 吞吐（4090） | 延迟（H20） | 吞吐（H20） | GPU 显存 |
|------|----------------|------------|------------|-----------|---------|
| Baseline | 0.187s | 85.6 Hz | 0.236s | 67.8 Hz | 15.8 GB |
| MemoryVLA | 0.194s | 82.5 Hz | 0.246s | 65.0 Hz | 16.6 GB |
| **MemoryVLA++** | 0.241s | 66.4 Hz | 0.301s | 53.2 Hz | 21.7 GB |

**说明**: 相比基线，MemoryVLA++ 引入约 29% 延迟增量和 37% 显存增量，实际控制频率仍远高于大多数机器人任务需求。

---

### Table 11：VLA 骨干网络分析

**(a) SimplerEnv**

| Backbone | 预训练 | Spoon Towel | Carrot Plate | Stack Cube | Eggplant Basket | Avg. |
|----------|--------|------------|-------------|-----------|----------------|------|
| LLaMA2 | CogACT | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 |
| Qwen2.5 | Dexbotic | 100.0 | 66.7 | 70.8 | 100.0 | **84.4** |

**(b) Libero**

| Backbone | 预训练 | Spatial | Object | Goal | Long-10 | Avg. |
|----------|--------|---------|--------|------|---------|------|
| LLaMA2 | CogACT | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| Qwen2.5 | Dexbotic | 97.2 | 99.2 | 98.4 | 93.2 | **97.0** |

**说明**: PCMB+想象框架对不同骨干均有效；Qwen2.5 骨干在 SimplerEnv 上显著更强（+12.5%），Libero 差异很小。

---

## 实验

### 数据集/Benchmark

| Benchmark | 规模/特点 | 用途 |
|-----------|---------|------|
| Libero（含 Long-10/Long-90）| 5 子套件，50 任务 | 仿真通用操作 |
| SimplerEnv | 4 任务，零样本评测 | 仿真零样本泛化 |
| Mikasa-Robo | 时序专项任务（SGT/IM/RC3/5/9）| 仿真记忆能力 |
| Calvin | 5 步长程任务序列 | 仿真长程规划 |
| Libero-Plus | 7 种扰动类型 | 仿真鲁棒性 |
| 真实机器人（Franka/ARX5/WidowX）| 17 真实任务 | 真实世界验证 |

### 实现细节

- **VLM Backbone**: 7B Prismatic（DINOv2 + SigLIP + LLaMA-7B），Open-X Embodiment 预训练
- **世界模型**: Stable Video Diffusion 1.5B，微调 40k 步（通用任务）/ 20k 步（长程任务）
- **动作专家**: DiT 约 300M 参数，DDIM 10 步，CFG scale 1.5
- **优化器**: AdamW，学习率 $2 \times 10^{-5}$
- **Batch Size**: 26-32 per GPU，全局 208-256
- **VLA 训练步数**: 50k-60k 步（依 benchmark）
- **硬件**: 8× NVIDIA A100 或 H20，PyTorch FSDP

### 可视化结果

记忆注意力权重可视化（Figure 10）显示 PCMB 在按钮序列任务中准确聚焦到目标按钮被按下的历史帧；世界模型可视化（Figure 12）显示仅 1 步去噪的想象帧已捕获物体运动趋势。

---

## 批判性思考

### 优点

1. **框架完整性**: 首次将工作记忆、情景记忆、未来想象三者统一到 VLA 框架，认知科学动机清晰；
2. **消融充分**: 分别验证记忆和想象模块的独立贡献，以及每个设计选择（门控/merge/双流）的有效性；
3. **广泛验证**: 5 个仿真 benchmark + 3 个真实机器人平台 + 3 类任务，覆盖全面；
4. **潜空间高效**: 1 步去噪提取特征而非生成完整视频，推理开销合理。

### 局限性

1. **显存增加 37%**: 21.7 GB GPU 显存对部署资源要求较高；
2. **Camera 扰动脆弱**: Libero-Plus 零样本下相机变化仅 36.4%，视觉泛化仍有挑战；
3. **代码未开源**: 可复现性受限；
4. **记忆长度超参敏感**: 不同任务最优 L 差异大（4~512），需任务特定调优。

### 潜在改进方向

1. 探索更高效的世界模型（如 flow matching）降低显存和延迟；
2. 自适应记忆长度调整，无需手动设定 $L$；
3. 结合在线学习机制，记忆在部署时持续更新。

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（论文提供）
- [x] 数据集可获取（公开 benchmark）

---

## 关联笔记

### 基于

- [[MemoryVLA]]: 直接前驱工作，本文在其基础上增加世界模型想象
- [[CogACT]]: 主要对比基线，7B VLM + 扩散动作专家架构

### 对比

- [[π₀]]: 物理智能 VLA 基线，SimplerEnv 上重点对比
- [[OpenVLA-OFT]]: Libero-Plus 微调设置下的对比基线

### 方法相关

- [[Diffusion Policy]]: 核心动作生成范式
- [[Action Chunking]]: 多步动作预测框架
- [[世界模型]]: 想象模块的理论基础
- [[扩散模型]]: 动作专家的生成方式
- [[Cross-Attention]]: 记忆检索的核心机制

### 硬件/数据相关

- [[CALVIN]]: 长程任务主要 benchmark
- [[Open-X-Embodiment]]: VLM 预训练数据集

---

## 速查卡片

> [!summary] MemoryVLA++
> - **核心**: 完整时序建模——记忆历史 + 想象未来，驱动扩散 VLA
> - **方法**: PCMB（感知-认知双流记忆）+ SVD 潜空间世界模型 + DiT 动作专家
> - **结果**: 真实机器人通用/记忆/想象任务分别 +9%/+26%/+28%（vs CogACT）
> - **代码**: 未开源（2025 年 6 月）

---

*笔记创建时间: 2026-06-09*
