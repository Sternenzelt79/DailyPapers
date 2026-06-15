---
title: "μ₀: A Scalable 3D Interaction-Trace World Model"
method_name: "mu0"
authors: [Seungjae Lee, Yoonkyo Jung, Jusuk Lee, Jonghun Shin, Amir Hossein Shahidzadeh, Yao-Chih Lee, H. Jin Kim, Jia-Bin Huang, Furong Huang]
year: 2026
venue: arXiv
tags: [world-model, 3d-trace, robot-manipulation, flow-matching, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.13769
created: 2026-06-15
---

# 论文笔记：μ₀: A Scalable 3D Interaction-Trace World Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Maryland, College Park; Seoul National University |
| 日期 | June 2026 |
| 项目主页 | [mu0-wm.github.io](https://mu0-wm.github.io/) |
| 对比基线 | [[TraceGen]], [[pi0]], [[Track2Act]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13769) / Code: N/A |

---

## 一句话总结

> μ₀ 是一个无需动作标签、从多样化视频中学习 3D 交互轨迹的世界模型，通过 TraceExtract 管道自动提取监督信号，实现跨实体的机器人策略迁移。

---

## 核心贡献

1. **TraceExtract 数据管道**: 自动从异构视频中提取事件感知的 3D 轨迹监督信号，缩放能力比先前数据集提升约 8 倍，无需实体特定标签。
2. **μ₀ 架构**: 置换等变的 Trace Expert 结合预训练 [[VLM]] 骨干，用 [[B-Spline]] 参数化轨迹并以 [[Flow Matching]] 训练，推理速度 0.29s，比 Track2Act 快 2.9×。
3. **Trace-Conditioned 动作适配**: 冻结预训练 μ₀，仅训练实体特定的动作专家头，在仿真和真实机器人上超越有动作监督的 [[VLA]] 基线。

---

## 问题背景

### 要解决的问题

当前机器人学习面临两难困境：像素级世界模型（如 [[Diffusion Policy]]）将容量浪费在外观重建上，而直接动作预测模型则依赖实体特定标签，无法跨机器人复用。

### 现有方法的局限

- **像素预测世界模型**：计算量大，容量消耗在外观细节而非运动语义
- **直接动作模型**（π₀、π₀.₅）：需要大量动作标注，难以利用无标签的多样化视频
- **2D 运动表征**（光流、2D 轨迹）：对相机运动敏感，无法捕捉真实 3D 空间几何
- **TraceGen**：使用固定网格采样、依赖深度传感器、没有可复用的预训练权重

### 本文的动机

3D 交互轨迹（interaction traces）是比像素更紧凑、比动作更通用的中间表征：它捕捉了"物体如何在空间中运动"而非"机器人关节如何运动"，因此可以从互联网视频中大规模提取，并迁移到不同实体。

---

## 方法详解

### 整体流程

μ₀ 分为两个阶段：

1. **TraceExtract**（离线数据构建）: 视频 → 语义关键点 → 3D 轨迹 → 事件感知字幕
2. **μ₀ 预训练 + 动作适配**（在线学习）: 图像+语言 → Trace Expert → 预测未来 3D 轨迹；冻结后 → 动作专家头 → 机器人控制

### 模块1: TraceExtract 数据管道

#### 2.1 语义关键点采样

**设计动机**: 固定网格采样在视觉同质区域（背景、空白区）浪费采样预算；改用 [[DINOv2]] 实体聚类在语义边界处密集采样。

**具体实现**:
- 用 [[DINOv2]] patch 特征对帧进行实体聚类
- 为每个实体分配固定关键点预算
- 在高可见帧上选择空间多样的点（避免遮挡密集区）

#### 2.2 3D 轨迹构建（混合全局-局部重建）

**设计动机**: 单一全局重建对相机运动敏感，纯局部重建积累漂移；混合策略用稀疏锚帧提供全局参考，用密集局部块重建细节。

**具体实现**:
- **稀疏全局参考**：在关键帧上建立稀疏 3D 坐标系 $\mathbf{E}_t^{\text{sparse}}$
- **密集局部重建**：对每个视频块 $c$ 独立重建局部坐标 $\mathbf{E}_t^{(c)}$
- **SE(3) 对齐**：将局部块对齐到全局参考（见公式1）
- **跨块渐进追踪**：在世界坐标中跨块追踪，保持相机运动鲁棒性

#### 2.3 事件感知字幕（Event-Centric Captioning）

**设计动机**: 均匀切片丢失运动结构；以加速度峰值为动作锚点、低谷为块边界，使每个块对应一个语义完整的动作单元。

**具体实现**:
- 用 Savitzky-Golay 滤波器平滑每帧轨迹加速度 $\tilde{a}_t$
- 在加速度峰值处检测动作锚点 $p_i$（公式2）
- 在相邻锚点之间的最低加速度处设置块边界 $b_i$（公式2）
- 对每个块进行层次化 [[VLM]] 字幕生成

### 模块2: μ₀ 架构

#### 3.1 多模态条件骨干

使用预训练 [[SmolVLM2]]-2.2B 作为语义编码器：
- RGB 图像和文本经标准 SigLIP → VLM 路径
- 度量深度图经独立可训练 patch stem 后汇入更深的 SigLIP 层
- 骨干权重冻结，仅微调深度 stem

#### 3.2 置换等变 Trace Expert

**设计动机**: 关键点是无序集合，需要对其顺序置换等变（permutation-equivariant）；用 [[B-Spline]] 控制点参数化轨迹，既平滑又压缩。

**具体实现**:
- 将关键点视为可交换的 queries
- 未来轨迹表示为三次 B-spline 控制点 $\mathbf{P} \in \mathbb{R}^{D \times 3}$，$D=10$
- Token 化: 片段嵌入 + Fourier 位置编码 + [[DINOv2]] 特征注入（见公式5）
- 20 层 Transformer，隐层宽度为 VLM 的 0.5×

#### 3.3 语义结构的 Flow Matching 训练

训练目标：在 B-spline 控制点空间上做 [[Conditional Flow Matching]]，附加遮挡有效性预测和刚性损失。

#### 3.4 Trace-Conditioned 动作专家

**设计动机**: 冻结 μ₀ 后，通过 [[Gated Cross-Attention]] 将中间 trace 去噪特征注入 VLM 特征，训练轻量动作专家，实现实体迁移。

---

## 关键公式

### 公式1: [[SE(3) 对齐|全局-局部坐标对齐]]

$$
\mathbf{A}^{(c)} = \arg\min_{\mathbf{A} \in \mathrm{SE}(3)} \sum_{t \in \mathcal{S} \cap c} \left\| \mathbf{A}\, \mathbf{E}_{t}^{(c)} - \mathbf{E}_{t}^{\text{sparse}} \right\|^{2}
$$

**含义**: 在稀疏全局参考帧 $\mathcal{S}$ 与局部块 $c$ 的重叠处，求最优刚体变换 $\mathbf{A}^{(c)}$，将局部坐标对齐到全局世界坐标。

**符号说明**:
- $\mathbf{A}^{(c)} \in SE(3)$: 块 $c$ 的刚体变换（旋转+平移）
- $\mathbf{E}_{t}^{(c)}$: 局部块 $c$ 在时刻 $t$ 的点云坐标
- $\mathbf{E}_{t}^{\text{sparse}}$: 稀疏全局参考帧坐标
- $\mathcal{S} \cap c$: 稀疏全局帧与块 $c$ 的重叠帧集合

### 公式2: [[事件分割|动作感知块边界]]

$$
b_{i} = \arg\min_{t \in [p_{i}, p_{i+1}]} \tilde{a}_{t}
$$

**含义**: 在相邻动作锚点 $p_i$ 和 $p_{i+1}$ 之间，选择轨迹加速度最低的时刻作为块边界，保证每块对应一个语义连贯的动作单元。

**符号说明**:
- $b_i$: 第 $i$ 个块边界时刻
- $p_i, p_{i+1}$: 相邻动作锚点（加速度峰值）
- $\tilde{a}_t$: Savitzky-Golay 平滑后的帧级轨迹加速度

### 公式3: [[归一化|锚点相对目标归一化]]

$$
\tilde{\mathbf{T}}^{1}_{n,k} = (\mathbf{T}^{1}_{n,k} - \mathbf{c}_{n}) / \mathbf{s}_{\Delta}
$$

**含义**: 将未来轨迹相对当前位置 $\mathbf{c}_n$ 归一化，并用尺度因子 $\mathbf{s}_{\Delta}$ 标准化，消除绝对坐标依赖。

**符号说明**:
- $\tilde{\mathbf{T}}^{1}_{n,k}$: 关键点 $n$ 在未来时刻 $k$ 的归一化目标位置
- $\mathbf{c}_{n}$: 关键点 $n$ 的当前（锚点）3D 位置
- $\mathbf{s}_{\Delta}$: 全局尺度归一化因子

### 公式4: [[B-Spline|B-Spline 控制点拟合]]

$$
\mathbf{P}^{\star}_{n} = \arg\min_{\mathbf{P} \in \mathbb{R}^{D \times 3}} \left\| \mathbf{M}_{n} \odot \left( \mathbf{B}\mathbf{P} - [\mathbf{0};\,\tilde{\mathbf{T}}^{1}_{n}] \right) \right\|_{F}^{2} + \lambda_{\text{bsp}}^{2} \left\| \boldsymbol{\Gamma}\mathbf{P} \right\|_{F}^{2}
$$

**含义**: 将关键点 $n$ 的归一化未来轨迹拟合为 $D=10$ 个三次 B-spline 控制点，带遮挡掩码 $\mathbf{M}_n$ 和 Tikhonov 正则化保证平滑性。

**符号说明**:
- $\mathbf{P}^{\star}_{n} \in \mathbb{R}^{D \times 3}$: 关键点 $n$ 的最优 B-spline 控制点
- $\mathbf{B}$: B-spline 基矩阵，将控制点映射到轨迹采样点
- $\mathbf{M}_{n}$: 遮挡掩码（遮挡帧权重为 0）
- $\lambda_{\text{bsp}}$: 平滑正则化系数
- $\boldsymbol{\Gamma}$: 差分矩阵（惩罚控制点间的二阶差分）

### 公式5: [[DINOv2]] 特征融合

$$
\mathbf{e}_{n,j} \leftarrow W_{2}\, \operatorname{SiLU}\!\left( W_{1}\, \operatorname{concat}(\mathbf{e}_{n,j},\, \mathbf{f}_{n}^{\text{dino}}) \right)
$$

**含义**: 将 DINOv2 局部语义特征 $\mathbf{f}_{n}^{\text{dino}}$ 拼接到 trace token 嵌入 $\mathbf{e}_{n,j}$ 后，用两层线性+SiLU 投影融合，增强 token 的语义感知。

**符号说明**:
- $\mathbf{e}_{n,j}$: 关键点 $n$ 第 $j$ 个时间片段的 token 嵌入
- $\mathbf{f}_{n}^{\text{dino}}$: 关键点 $n$ 处提取的 DINOv2 patch 特征
- $W_1, W_2$: 可学习投影矩阵

### 公式6: [[Conditional Flow Matching|线性概率路径]]

$$
\mathbf{P}^{\tau} = \tau \boldsymbol{\epsilon} + (1 - \tau) \mathbf{P}^{\star}
$$

**含义**: Flow Matching 中，在噪声 $\boldsymbol{\epsilon}$（$\tau=1$）和干净控制点 $\mathbf{P}^{\star}$（$\tau=0$）之间线性插值，定义训练时间步 $\tau$ 处的中间状态。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 流时间步（从干净到噪声）
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 高斯噪声
- $\mathbf{P}^{\star}$: 目标 B-spline 控制点（干净数据）

### 公式7: [[Flow Matching]] 损失

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{\tau, \boldsymbol{\epsilon}} \left[ \left\| v_{\theta}(\mathbf{P}^{\tau}, \tau, F_{\text{cond}}) - (\boldsymbol{\epsilon} - \mathbf{P}^{\star}) \right\|_{2}^{2} \right]
$$

**含义**: 训练速度场网络 $v_\theta$ 预测从中间状态指向噪声方向的向量场，条件为多模态特征 $F_{\text{cond}}$（图像+文本+深度）。

**符号说明**:
- $v_\theta$: Trace Expert 速度场网络（参数为 $\theta$）
- $F_{\text{cond}}$: 多模态条件特征（VLM 输出）
- $\boldsymbol{\epsilon} - \mathbf{P}^{\star}$: 从干净数据指向噪声的目标向量

### 公式8: [[有效性预测|遮挡终止损失]]

$$
\mathcal{L}_{\text{done}} = \frac{\sum_{t=1}^{H} \ell_{\text{BCE}}(\hat{d}_{n,t},\, y_{n,t})}{N}
$$

**含义**: 对每个关键点每个未来帧预测二元有效性标志（是否遮挡/超出视野），用 BCE 损失监督，平均到 $N$ 个关键点。

**符号说明**:
- $\hat{d}_{n,t}$: 关键点 $n$ 在时刻 $t$ 的预测有效性
- $y_{n,t}$: 真实有效性标签（1=可见，0=遮挡）
- $H$: 未来预测帧数（$H=32$）
- $N$: 关键点数量

### 公式9: [[刚性约束|语义刚性损失]]

$$
\mathcal{L}_{\text{rig}} = \mathbb{E}_{\tau, \boldsymbol{\epsilon}} \left[ \frac{1}{|R|} \sum_{(n, n') \in R} \operatorname{Var}_{d} \left( \left\| \hat{\mathbf{P}}_{n,d} - \hat{\mathbf{P}}_{n',d} \right\|_{2}^{2} \right) \right]
$$

**含义**: 对属于同一 DINO 实体簇的关键点对 $(n, n')$，惩罚其预测控制点间距离在 $d$ 维度上的方差，保持同簇关键点的相对几何一致性。

**符号说明**:
- $R$: 同簇关键点对集合（由 DINOv2 聚类定义）
- $\hat{\mathbf{P}}_{n,d}$: 关键点 $n$ 第 $d$ 个预测控制点的坐标
- $\operatorname{Var}_{d}(\cdot)$: 在控制点维度 $d$ 上的方差

### 公式10: [[总训练损失|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{done}} \mathcal{L}_{\text{done}} + \lambda_{\text{rig}} \mathcal{L}_{\text{rig}}
$$

**含义**: 流匹配损失 + 遮挡有效性损失 + 语义刚性损失的加权和，构成 μ₀ 的完整训练目标。

**符号说明**:
- $\lambda_{\text{done}}, \lambda_{\text{rig}}$: 各损失项的权重系数

### 公式11: [[Gated Cross-Attention|门控交叉注意力融合]]

$$
\mathbf{z}_{\text{guided}} = \mathbf{z} + \sigma(g) \cdot \mathrm{CA}\!\left( Q = \mathrm{LN}(\mathbf{z}),\; K = V = \tilde{\mathbf{h}}_{\text{trace}} \right)
$$

**含义**: 动作专家中，用可学习门控标量 $g$ 调制 trace 去噪中间特征 $\tilde{\mathbf{h}}_{\text{trace}}$ 对 VLM 特征 $\mathbf{z}$ 的注意力融合量，实现软性控制。

**符号说明**:
- $\mathbf{z}$: VLM 特征（来自冻结骨干）
- $g$: 可学习门控标量
- $\sigma(\cdot)$: Sigmoid 函数
- $\tilde{\mathbf{h}}_{\text{trace}}$: μ₀ 中间 trace 去噪特征（从冻结 Trace Expert 提取）
- $\mathrm{LN}$: Layer Normalization

### 公式12: [[动作 Flow Matching|动作生成损失]]

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau, \mathbf{a}, \boldsymbol{\epsilon}_a} \left\| v_{\phi}\!\left(\mathbf{a}^{\tau}, \tau, \mathbf{z}_{\text{guided}}, \mathbf{c}\right) - (\mathbf{a} - \boldsymbol{\epsilon}_a) \right\|_{2}^{2}
$$

**含义**: 动作专家以 trace 引导特征 $\mathbf{z}_{\text{guided}}$ 和任务条件 $\mathbf{c}$ 为条件，用流匹配损失训练动作去噪网络 $v_\phi$，生成机器人控制序列。

**符号说明**:
- $v_\phi$: 动作专家网络（参数为 $\phi$，μ₀ 冻结）
- $\mathbf{a}^{\tau}$: 时间步 $\tau$ 处的中间动作状态
- $\mathbf{c}$: 任务条件（语言指令等）
- $\boldsymbol{\epsilon}_a$: 动作空间高斯噪声

---

## 关键图表

### Figure 1: From Videos to Reusable Action Priors / 从视频到可复用动作先验

![Figure 1](https://arxiv.org/html/2606.13769v1/x1.png)

**说明**: μ₀ 的总体动机图。TraceExtract 从异构视频（烹饪、人手操作、机器人操作等）中提取事件感知的 3D 轨迹，预训练得到的 μ₀ 世界模型可作为可复用的动作先验，迁移到不同实体的机器人策略学习中。

### Figure 2: Overview of TraceExtract / TraceExtract 数据管道概览

![Figure 2](https://arxiv.org/html/2606.13769v1/x2.png)

**说明**: 三阶段 TraceExtract 管道。(1) **语义关键点采样**：DINOv2 聚类 → 实体分配预算 → 高可见帧采样；(2) **3D 轨迹构建**：混合全局-局部重建，SE(3) 对齐跨块轨迹；(3) **事件感知字幕**：加速度检测动作锚点 → 层次化 VLM 字幕。

### Figure 3: Overview of μ₀ and Its Action-Expert Interface / μ₀ 架构与动作专家接口

![Figure 3](https://arxiv.org/html/2606.13769v1/x3.png)

**说明**: μ₀ 完整架构。左：预训练阶段，SmolVLM2-2.2B 骨干 + Trace Expert，以图像+文本+深度为条件，用 Flow Matching 预测未来 B-spline 控制点。右：动作适配阶段，冻结 μ₀，通过 Gated Cross-Attention 将中间 trace 特征注入动作专家，训练实体特定的控制头。

### Figure 4: Qualitative Comparison of Predicted Traces / 预测轨迹定性对比

![Figure 4](https://arxiv.org/html/2606.13769v1/x4.png)

**说明**: 跨多个操作任务的轨迹预测对比。μ₀ 预测的 3D 轨迹（蓝色）与 TraceGen、Track2Act、Dream2Flow 相比更准确地捕捉物体运动路径，尤其在长时间跨度（T=32）时优势明显。

### Figure 5: Real-World Experimental Setup / 真实机器人实验设置

![Figure 5](https://arxiv.org/html/2606.13769v1/x5.png)

**说明**: UR3 机器人（双指夹爪）的三个真实任务可视化：(1) Pick into Sink（拾取放入水槽）；(2) Pour Almonds（倒杏仁）；(3) Unfold Towel（展开毛巾）。每个任务分别收集 90/80/50 个演示，20 次rollout 评估。

### Figure 6: Real-World Evaluation Results / 真实机器人评估结果

![Figure 6](https://arxiv.org/html/2606.13769v1/x6.png)

**说明**: 三个真实任务的成功率柱状图。μ₀ 平均 91.7%，显著超越 π₀（75%）、π₀.₅（85%）和纯 VLM（75%）。与 VLM baseline 相比，trace 特征带来 18.4 个百分点的提升。

### Figure 7: Additional Qualitative Comparisons / 附加定性对比（附录 D.2）

![Figure 7](https://arxiv.org/html/2606.13769v1/x7.png)

**说明**: 跨更多样化任务的扩展轨迹预测示例，进一步验证 μ₀ 在不同操作场景（抓取、放置、倒液体）中的泛化能力。

### Figure 8: RoboCasa365 Simulation Examples / RoboCasa365 仿真示例（附录 D.3）

![Figure 8](https://arxiv.org/html/2606.13769v1/x8.png)

**说明**: RoboCasa365 8 个厨房操作任务的评估场景可视化：CloseFridge、OpenFridge、CoffeeServeMug、PickPlaceFridgeShelfToDrawer、TurnOnMicrowave、SlideToasterOvenRack、PickPlaceCounterToCabinet、TurnOnToasterOven。

### Table 1: 2D 和 3D 轨迹预测评估（top5-DTW，多时间跨度）

| 方法 | top1-ADE (8/16/32) | top5-ADE (8/16/32) | top1-FDE (8/16/32) | top5-FDE (8/16/32) | top1-DTW (8/16/32) | top5-DTW (8/16/32) | 推理时间 |
|------|---|---|---|---|---|---|---|
| **2D 方法** |
| Gemini-3.1-pro | 0.190/0.274/0.305 | 0.161/0.232/0.253 | 0.311/0.425/0.424 | 0.254/0.321/0.311 | 0.183/0.258/0.284 | 0.152/0.208/0.224 | 78s |
| Gemini-3-flash | 0.187/0.271/0.299 | 0.158/0.231/0.254 | 0.312/0.414/0.405 | 0.252/0.329/0.316 | 0.183/0.260/0.281 | 0.150/0.211/0.227 | 62s |
| GPT-5.5 | 0.199/0.281/0.307 | 0.178/0.249/0.272 | 0.329/0.411/0.404 | 0.284/0.344/0.329 | 0.196/0.274/0.299 | 0.173/0.238/0.259 | 38s |
| Track2Act | 0.209/0.311/0.369 | 0.190/0.262/0.293 | 0.350/0.493/0.555 | 0.287/0.351/0.346 | 0.206/0.303/0.358 | 0.181/0.245/0.270 | 0.85s |
| Hamster | 0.202/0.276/0.297 | 0.178/0.239/0.256 | 0.326/0.400/0.411 | 0.274/0.320/0.330 | 0.197/0.261/0.277 | 0.170/0.220/0.233 | 14.4s |
| **μ₀（本文）** | **0.202/0.279/0.315** | **0.124/0.188/0.227** | **0.322/0.410/0.447** | **0.186/0.261/0.284** | **0.184/0.254/0.296** | **0.114/0.171/0.211** | **0.29s** |
| **3D 方法** |
| 3DFlowAction | 0.615/0.692/0.716 | 0.531/0.605/0.630 | 0.753/0.819/0.818 | 0.648/0.714/0.712 | 0.614/0.688/0.711 | 0.529/0.600/0.623 | 3.38s |
| Dream2Flow | 0.354/0.451/0.505 | 0.201/0.286/0.336 | 0.497/0.616/0.660 | 0.287/0.378/0.403 | 0.352/0.449/0.500 | 0.198/0.281/0.329 | 106.8s |
| TraceGen | 0.327/0.416/0.464 | 0.208/0.276/0.325 | 0.478/0.548/0.642 | 0.267/0.329/0.370 | 0.298/0.375/0.413 | 0.204/0.262/0.299 | 1.20s |
| **μ₀（本文）** | **0.209/0.288/0.325** | **0.132/0.199/0.239** | **0.331/0.425/0.464** | **0.200/0.278/0.305** | **0.191/0.263/0.308** | **0.127/0.187/0.223** | **0.29s** |

**关键发现**: μ₀ 在所有时间跨度的 top5 指标上均为最优（2D 和 3D），且推理速度 0.29s 比最快的 3D 基线 TraceGen（1.20s）快 4×，比 Track2Act（0.85s）快 2.9×。

### Table 2: RoboCasa365 仿真结果（成功率 %）

| 任务 | Diffusion Policy | π₀ | π₀.₅ | TraceGen+action | μ₀+action |
|------|---|---|---|---|---|
| CloseFridge | 34 | 44 | 34 | 38 | **54** |
| OpenFridge | 28 | 12 | 26 | 36 | **18** |
| CoffeeServeMug | 28 | 34 | **48** | 42 | 36 |
| PickPlaceFridgeShelfToDrawer | 28 | 30 | **66** | 30 | 40 |
| TurnOnMicrowave | 0 | 2 | 12 | 0 | **4** |
| SlideToasterOvenRack | 48 | 46 | **76** | 28 | 56 |
| PickPlaceCounterToCabinet | 6 | 18 | **54** | 0 | 12 |
| TurnOnToasterOven | 10 | 16 | **20** | 10 | **22** |
| **平均成功率 (%)** | 22.75 | 25.25 | **42** | 23 | **30.25** |

**关键发现**: μ₀ 以纯视频预训练超越有动作标注的 π₀（30.25% vs 25.25%），但与 π₀.₅（42%）仍有差距。

### Table 3: Fréchet 距离及参数量对比

| 方法 | top1-FD (8/16/32) | top5-FD (8/16/32) | 参数量 |
|------|---|---|---|
| **2D** |
| Gemini-3.1-pro | 0.324/0.467/0.505 | 0.269/0.385/0.416 | — |
| GPT-5.5 | 0.342/0.476/0.511 | 0.299/0.415/0.449 | — |
| Track2Act | 0.363/0.543/0.631 | 0.304/0.420/0.451 | 0.47B |
| Hamster | 0.339/0.462/0.505 | 0.291/0.390/0.429 | 13.5B |
| **μ₀（本文）** | **0.314/0.446/0.517** | **0.200/0.306/0.370** | **2.59B** |
| **3D** |
| 3DFlowAction | 0.765/0.843/0.866 | 0.664/0.747/0.772 | 2.04B |
| Dream2Flow | 0.547/0.710/0.787 | 0.325/0.464/0.530 | 11.3B |
| TraceGen | 0.450/0.560/0.642 | 0.291/0.395/0.457 | 0.67B |
| **μ₀（本文）** | **0.329/0.455/0.527** | **0.210/0.319/0.384** | **2.59B** |

**关键发现**: μ₀ 以 2.59B 参数在 Fréchet 距离上全面最优，Hamster（13.5B）参数量是其 5× 但性能更差。

### Table 4: RoboCasa365 训练超参数

| 超参数 | 值 |
|---|---|
| 动作维度 | 12 |
| 动作 horizon | 16 |
| 执行 horizon | 8 |
| Batch size | 32 |
| 优化器 | AdamW |
| 学习率 | 1×10⁻⁴ |
| Warmup 步数 | 1,000 |
| 训练步数 | 50,000 |

### Table 5: 真实机器人任务超参数

| 任务 | Action Dim | Action Horizon | Exec Horizon | Batch | LR | Warmup | Steps |
|------|---|---|---|---|---|---|---|
| Pick into Sink | 7 | 50 | 25 | 32 | 5×10⁻⁵ | 400 | 8,000 |
| Pour Almonds | 7 | 50 | 25 | 32 | 5×10⁻⁵ | 300 | 6,000 |
| Unfold Towel | 7 | 50 | 25 | 32 | 5×10⁻⁵ | 300 | 6,000 |

### Table 6: 消融实验（top5-DTW）

| 模型变体 | T=8 | T=16 | T=32 |
|---|---|---|---|
| **架构消融** |
| 完整 μ₀ | 0.127 | 0.187 | 0.223 |
| w/o B-spline（原始轨迹） | 0.156 | 0.222 | 0.258 |
| w/o DINOv2 特征 | 0.139 | 0.193 | 0.230 |
| w/o 刚性损失 | 0.138 | 0.193 | 0.227 |
| **输入鲁棒性** |
| w/ Depth + Trace History | 0.107 | 0.160 | 0.203 |
| w/o Depth | 0.112 | 0.168 | 0.207 |
| w/o Trace History | 0.126 | 0.183 | 0.224 |
| w/o Depth & Trace History | 0.127 | 0.187 | 0.223 |

**关键发现**: B-spline 参数化是最重要的组件（去除后 top5-DTW T=8 上升 22.8%）；深度输入对性能提升最显著（w/ Depth 比 w/o Depth 在 T=8 上提升 4.5%）。

### Table 7: 缩放分析（top5-DTW，2.59B 模型）

| 缩放维度 | T=8 | T=16 | T=32 |
|---|---|---|---|
| **模型规模缩放（100% 数据）** |
| 342M | 0.143 | 0.205 | 0.240 |
| 568M | 0.136 | 0.191 | 0.227 |
| 2.59B | 0.127 | 0.187 | 0.223 |
| **数据规模缩放（2.59B）** |
| 5% 数据 | 0.134 | 0.200 | 0.235 |
| 20% 数据 | 0.138 | 0.195 | 0.227 |
| 100% 数据 | 0.127 | 0.187 | 0.223 |

**关键发现**: 模型规模和数据规模均单调改善性能，表明 μ₀ 当前处于"容量受限"状态，更大规模训练仍有提升空间。

### Table 8: 动作头规模分析

| 模型变体 | 平均成功率 (%) |
|---|---|
| 200M 动作头（无 Trace） | 10.675 |
| 200M 动作头（μ₀ + action） | 25.625 |
| 400M 动作头（无 Trace） | 28.25 |
| 400M 动作头（μ₀ + action） | 30.25 |

**关键发现**: Trace 特征对小容量动作头（200M）提升更大（+14.95%），对大容量头（400M）提升较小（+2%），说明 Trace 特征更有效替代动作头容量。

---

## 实验

### 数据集

| 数据集/环境 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| TraceExtract 视频 | 多样化操作视频 | 无动作标签，自动提取 3D 轨迹 | 预训练 |
| RoboCasa365 | 8 任务，每任务 100 demos | 厨房操作仿真 | 仿真评估 |
| UR3 真实机器人 | 3 任务，90/80/50 demos | 双指夹爪，桌面操作 | 真实评估 |

### 实现细节

- **骨干网络**: SmolVLM2-2.2B（20 层文本解码器，SigLIP 视觉编码器）
- **Trace Expert**: 20 层 Transformer，隐层宽度 0.5× VLM
- **优化器**: AdamW，学习率 $10^{-4}$，训练 $2 \times 10^5$ 步，batch size 24
- **B-spline 控制点**: $D=10$，覆盖 $H=32$ 帧未来
- **关键点数量**: $N \sim \mathcal{U}(1, 256)$ 随机采样
- **Trace 历史**: $h=8$ 帧
- **总参数**: 2.59B（骨干 2.2B + Trace Expert ~0.39B）

---

## 批判性思考

### 优点

1. **范式创新**: 3D 交互轨迹作为中间表征，兼具语义紧凑性和跨实体通用性，比像素预测和直接动作预测都更合理。
2. **无需动作标签**: TraceExtract 完全从视频自动提取监督，大幅降低数据收集成本。
3. **推理效率突出**: 0.29s 比多数基线快 5-370×，具备实际部署潜力。
4. **缩放友好**: 模型规模和数据规模双维度缩放均有效，长期技术路线清晰。
5. **置换等变设计**: B-spline + 置换等变 Trace Expert 优雅地处理了关键点无序性。

### 局限性

1. **感知栈误差级联**: 依赖语义聚类、3D 重建、追踪、字幕生成等多个模块，任一模块误差均会传播。
2. **缺失触觉和力反馈**: 不建模接触力、触觉信息，对需要精细力控制的任务（如布料折叠的复杂情形）存在局限。
3. **实体覆盖有限**: 仅在桌面操作上验证，移动操作手和灵巧手的泛化性待验证。
4. **与 π₀.₅ 仍有差距**: 仿真平均成功率 30.25% vs 42%，差距较大。

### 潜在改进方向

1. 引入接触/力传感器信号到 Trace 表征中
2. 扩展到更多实体（灵巧手、移动机器人）
3. 结合主动感知改善遮挡处理
4. 端到端联合优化 TraceExtract 和 μ₀

### 可复现性评估

- [ ] 代码开源（项目主页存在但未见代码链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数表格完整）
- [x] 数据集可描述（RoboCasa365 公开）

---

## 关联笔记

### 基于

- [[SmolVLM2]]: 预训练 VLM 骨干（2.2B 参数）
- [[DINOv2]]: 语义关键点聚类和 token 特征融合
- [[Conditional Flow Matching]]: 轨迹生成训练目标
- [[B-Spline]]: 轨迹平滑参数化

### 对比

- [[TraceGen]]: 最相近的先前工作，固定网格 trace，无可复用预训练
- [[pi0]]: 有动作标注的 VLA 基线（仿真和真实）
- [[Track2Act]]: 2D 轨迹预测基线
- [[Dream2Flow]]: 基于视频扩散的 3D 流预测
- [[3DFlowAction]]: 直接预测 3D 动作流

### 方法相关

- [[Gated Cross-Attention]]: 动作专家适配机制
- [[SE(3) 对齐]]: 全局-局部 3D 重建对齐
- [[Savitzky-Golay 滤波]]: 加速度平滑用于事件检测

### 硬件/数据相关

- [[UR3]]: 真实机器人平台
- [[RoboCasa]]: 仿真评估环境

---

## 速查卡片

> [!summary] μ₀: A Scalable 3D Interaction-Trace World Model
> - **核心**: 从多样化视频中学习 3D 交互轨迹的世界模型，无需动作标签
> - **方法**: TraceExtract（DINOv2聚类+混合3D重建+事件字幕）→ μ₀（SmolVLM2骨干+B-spline Trace Expert + Flow Matching）→ Gated Cross-Attention 动作适配
> - **结果**: 仿真超越 π₀（30.25% vs 25.25%），真实机器人平均 91.7%（vs π₀ 75%）；推理 0.29s
> - **代码**: https://mu0-wm.github.io/

---

*笔记创建时间: 2026-06-15*
