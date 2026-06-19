---
title: "EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models"
method_name: "EquiVLA"
authors: [Thien-Loc Ha, Quang-Tan Nguyen, Trong-Bao Ho, Long Dinh, Minh Duc Nguyen, Gia-Binh Nguyen, Pham Tri Quang, Minh N. Vu, Duy M. H. Nguyen, An Thai Le, Ngo Anh Vien]
year: 2026
venue: arXiv
tags: [vla, equivariance, robot-manipulation, diffusion-transformer, geometric-learning, flow-matching, imitation-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.19784
created: 2026-06-19
---

# 论文笔记：EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未在摘要中列明 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19784) / Code: — |

---

## 一句话总结

> EquiVLA 将 SO(2) 旋转等变性引入 VLA 模型，通过 EquiPerceptor（等变视觉 token 融合）和 EquiActor（等变扩散 Transformer 动作头）两个即插即用模块，在不重新训练冻结 VLM 主干的前提下，将机器人操作成功率从 78.1% 提升至 92.6%。

---

## 核心贡献

1. **EquiPerceptor**: 将 [[Frame Averaging]] 扩展到 [[Vision Transformer|ViT]] 空间索引 patch token，生成近似 SO(2) 等变的视觉表示，并通过 Equivariant Adapter 融合语义与几何信息，无需修改冻结的 [[VLM]] 权重
2. **EquiActor**: 首个用于机器人策略学习的 SO(2) 等变 [[Diffusion Transformer]]，用可操控层（steerable layers）替换标准 DiT，从头训练，采用 [[Flow Matching]] 目标
3. **理论保证 + 端到端框架**: 建立了等变误差的形式近似界，证明 C₄ 子群可达精确等变，并提供完整的端到端 SO(2) 等变链路

---

## 问题背景

### 要解决的问题

当前 [[VLA]] 模型缺乏几何归纳偏置——它们将旋转后的场景视为完全不同的输入，需要大量额外训练数据才能泛化到不同方向的物体摆放。这在操作任务中尤其关键：末端执行器的旋转方向直接影响成功率。

### 现有方法的局限

- 先前的等变策略方法（EquivAct、EquiBot 等）通常从头训练小型专用模型，无法利用大规模视觉语言预训练
- 基于规范化的方法（canonicalization wrappers）在推理时引入额外误差且架构不够优雅
- 标准 [[Diffusion Transformer|DiT]] 动作头不具备任何旋转对称性

### 本文的动机

几何对称性与大规模视觉语言预训练并非对立——它们是互补的。通过在视觉适配器和动作头两个层面独立对称化，可以在不触及冻结 VLM 的前提下为现有大规模 VLA 架构（如 [[GR00T N1.5]]）注入等变性。

---

## 方法详解

### 模型架构

EquiVLA 以 **[[GR00T N1.5]]**（带 [[Flow Matching]] 扩散 Transformer 动作头）为基础，在其上增加两个模块：

- **输入**: 语言指令 $l$ + 静态相机观测 $o_t$ + 腕部图像 + 机器人本体状态 $s_t$
- **Backbone**: 冻结的 [[VLM]]（GR00T N1.5 的视觉语言主干）
- **核心模块**: [[EquiPerceptor]] 对静态视图做等变 token 融合，[[EquiActor]] 替换标准 DiT 动作头
- **输出**: 末端执行器动作 $\hat{a}_t$（绝对或相对控制）
- **对称群**: 离散循环群 $C_u$（默认 $C_8$，对应 45° 的 8 个旋转）

![Figure 1: EquiVLA 整体架构](https://arxiv.org/html/2606.19784/2606.19784v1/x1.png)

**说明**: （上）EquiPerceptor 通过 Token-level Frame Averaging 对冻结 VLM 的视觉流程进行对称化，产生等变 token $\mathbf{z}^{eq}$ 和不变 token $\mathbf{z}^{inv}$；$\mathbf{z}^{inv}$ 与腕部图像和语言指令一起输入冻结 VLM，得到上下文 token $\mathbf{z}^{ctx}$，再经 Equivariant Adapter 与 $\mathbf{z}^{eq}$ 融合。（下）EquiActor 用 SO(2) 等变 DiT 替换标准动作头，通过等变交叉注意力和自注意力精化状态与噪声动作 token，最终用 [[Flow Matching]] 解码动作。

---

### 核心模块

#### 模块1: EquiPerceptor（等变视觉 Token 融合）

**设计动机**: 利用 [[Frame Averaging]]（FA）将预训练 [[ViT]] 的 patch token 对称化，绕过对冻结权重的修改

**具体实现**:

1. **Token-level Frame Averaging**: 对每个旋转 $h \in C_u$，将旋转后的图像 $h \cdot x$ 经冻结 ViT 得到 patch token，再用逆旋转 $h^{-1}$ 还原空间排列，取平均得到等变 token 流 $\mathbf{z}^{eq}$ 和不变 token 流 $\mathbf{z}^{inv}$

2. **Equivariant Adapter**: 用可学习的不变门控 $\alpha$ 将语义信息（$\mathbf{z}^{inv}$ 路径的上下文 $s^{inv}$）与几何信息（$\mathbf{z}^{eq}$）融合，输出送入 EquiActor 的最终视觉表示 $\mathbf{z}^{out}_{eq}$

**理论性质**: 当空间置换 $\tau$ 是群同态时（$C_4$ 在方形 patch 网格上成立），Token-level FA 达到精确等变；$C_8$ 的 45° 旋转引入有界近似误差

---

#### 模块2: EquiActor（SO(2) 等变扩散 Transformer 动作头）

**设计动机**: 标准 DiT 动作头对旋转不敏感，需要用 [[Regular Representation]] 的可操控层从架构层面编码旋转对称性

**具体实现**:

- 用正则特征空间（regular feature space）中的 **可操控层**（steerable layers）替换所有线性层
- 实现等变 **交叉注意力**（cross-attention，条件化在视觉上下文 $\mathbf{z}^{out}$）和等变 **自注意力**（self-attention）
- 等变状态编码和动作解码
- **从头训练**（无法复用预训练 DiT 权重），使用 [[Flow Matching]] 目标

**动作表示**（见关键公式 Eq.4）:

- **绝对控制**: 末端执行器位置/方向分解为 SO(2) 向量；夹爪宽度为不变量
- **相对控制**: 分解为不可约 SO(2) 表示 $\rho_0^6 \oplus \rho_1^4 \oplus \rho_2$

---

## 关键公式

### 公式1: [[Frame Averaging|等变 Token 流]]（Eq. 1）

$$
\mathbf{z}^{eq}(x) = \frac{1}{|G|} \sum_{h \in G} \left[\tau(h^{-1}) \otimes \rho_{reg}(h^{-1})\right] \cdot f_\theta(h \cdot x)
$$

**含义**: 对群 $G$ 的所有元素取平均，每个方向下用冻结 ViT $f_\theta$ 提取特征后再逆变换回原空间，得到近似等变的视觉 token 流

**符号说明**:
- $G = C_u$: 离散旋转群（默认 $C_8$）
- $h$: 群元素（旋转）
- $\tau(h^{-1})$: 空间 patch 的逆置换
- $\rho_{reg}(h^{-1})$: 正则表示中的逆旋转
- $f_\theta$: 冻结的 ViT，参数 $\theta$ 固定

---

### 公式2: [[Frame Averaging|不变 Token 流]]（Eq. 2）

$$
\mathbf{z}^{inv}(x) = \frac{1}{|G|} \sum_{h \in G} f_\theta(h \cdot x)
$$

**含义**: 对各旋转方向下的 ViT 输出直接平均，得到旋转不变的语义 token 流，供冻结 VLM 处理

**符号说明**:
- 与公式1 相同，此处不含逆变换，因此输出对旋转不变而非等变

---

### 公式3: [[Equivariant Adapter|等变适配器门控融合]]（Eq. 3）

$$
\mathbf{z}^{out}_{eq} = \alpha^{reg} \odot W_s\left(s^{inv} \otimes \mathbf{1}_{|G|}\right) + (1 - \alpha^{reg}) \odot W_g(\tilde{\mathbf{z}}^{eq})
$$

其中：

$$
\alpha = \sigma\left(W_{gate}[s^{lang};\ s^{vis};\ \bar{\mathbf{z}}^{eq}]\right), \quad \alpha^{reg} = \alpha \otimes \mathbf{1}_{|G|}
$$

**含义**: 用可学习不变门控 $\alpha$ 在不变语义流（$s^{inv}$，来自冻结 VLM 的上下文）和等变几何流（$\tilde{\mathbf{z}}^{eq}$）之间自适应加权融合

**符号说明**:
- $s^{lang}, s^{vis}$: 语言和视觉不变摘要向量
- $\bar{\mathbf{z}}^{eq}$: 等变 token 的均值摘要
- $W_s, W_g, W_{gate}$: 可学习线性层
- $\odot$: 逐元素乘法
- $\otimes$: Kronecker/张量积（扩张到正则维度）

---

### 公式4: [[SO(2) Equivariance|动作等变表示]]（Eq. 4）

**绝对控制**:

$$
g \cdot a_t = \left(\rho_1^3 \oplus (\rho_1 \oplus \rho_0) \oplus \rho_0\right)(g)\, a_t
$$

**相对控制**:

$$
g \cdot a_t = P^{-1}\left(\rho_0^6 \oplus \rho_1^4 \oplus \rho_2\right)(g)\, P\, a_t
$$

**含义**: 将机器人动作向量分解为 SO(2) 的不可约表示之直和，使得当场景旋转 $g$ 时动作以对应的群表示变换，从而实现等变动作输出

**符号说明**:
- $\rho_0$: 平凡（不变）表示，用于夹爪宽度等标量
- $\rho_1$: 标准 2D 旋转表示，用于位置/方向向量
- $\rho_2$: 二阶表示
- $P$: 坐标系变换矩阵；上标表示重数（如 $\rho_0^6$ 表示 6 个不变量）

---

### 公式5: [[SO(2) Equivariance|端到端近似等变性]]（Eq. 5）

$$
\hat{a}_t(g \cdot o_t,\, \rho_s(g)\, s_t) \approx \rho_a(g) \cdot \hat{a}_t(o_t,\, s_t) \quad \forall g \in C_u
$$

**含义**: EquiVLA 整体近似满足 SO(2) 等变性——对旋转后的观测和状态的预测动作，近似等于对原始动作施加相应旋转

**符号说明**:
- $o_t$: 静态相机观测（图像）
- $s_t$: 机器人本体状态
- $\rho_s, \rho_a$: 状态和动作空间上的群表示
- $\approx$: 近似（误差由 Theorem 3 的界量化）

---

### 公式6: [[Flow Matching|流匹配训练目标]]（Eq. 7）

$$
\mathcal{L} = \mathbb{E}_{k,\varepsilon,a_t}\left[\left\|v_\theta\!\left(a_t^k,\, k,\, \mathbf{z}^{out},\, \mathbf{z}^s_t\right) - (a_t - \varepsilon)\right\|^2\right]
$$

其中：

$$
a_t^k = (1-k)\,\varepsilon + k\cdot a_t
$$

**含义**: 学习将噪声插值后的动作 $a_t^k$ 推向真实动作 $a_t$ 的速度场 $v_\theta$，即经典的条件流匹配（CFM）目标

**符号说明**:
- $k \in [0,1]$: 流匹配时间步（插值系数）
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $a_t^k$: 时间 $k$ 处的插值动作
- $\mathbf{z}^{out}$: EquiPerceptor 输出的视觉上下文
- $\mathbf{z}^s_t$: 编码后的机器人状态 token

---

### 公式7: [[Frame Averaging|等变近似误差界]]（Theorem 3）

$$
\|\varepsilon(g, x)\| \leq \Delta \cdot B(x)
$$

其中：

$$
\Delta = \max_{g,h \in G} \|D(g,h)\|_{op} \quad \text{（表示缺陷）}
$$

$$
B(x) = \max_{h \in G} \|f_\theta(h \cdot x)\| \quad \text{（特征范数界）}
$$

**含义**: EquiPerceptor 的等变误差被表示缺陷 $\Delta$ 和特征范数上界 $B(x)$ 的乘积控制；当 $u|4$（如 $C_4$）时 $\Delta = 0$，精确等变

**符号说明**:
- $D(g,h)$: 群表示在元素对 $(g,h)$ 上的偏差矩阵（representation defect）
- $\|\cdot\|_{op}$: 算子范数

---

## 关键图表

### Figure 1: EquiVLA 整体架构

![Figure 1](https://arxiv.org/html/2606.19784/2606.19784v1/x1.png)

**说明**: 上半部分展示 EquiPerceptor：冻结 ViT 对 8 个旋转方向的图像各自提取 patch token，Token-level FA 汇聚为等变流 $\mathbf{z}^{eq}$ 和不变流 $\mathbf{z}^{inv}$，后者喂入冻结 VLM，Equivariant Adapter 融合两者输出最终视觉表示。下半部分展示 EquiActor：用 SO(2) 等变 DiT 替换标准动作头，通过等变注意力层条件化在视觉上下文上，[[Flow Matching]] 解码输出动作。

---

### Figure 2: Mobile ALOHA 真实机器人任务

**任务图片（初始帧 → 目标帧）**:

| 任务 | 图片 |
|------|------|
| (a) Banana in Pot | ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/1_kitchen_place_banana.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/4_kitchen_place_banana.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/6_kitchen_place_banana.jpeg) |
| (b) Block Storing | ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/1_drawer_placing.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/2_drawer_placing.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/6_drawer_placing.jpeg) |
| (c) House Building | ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/1_stack_house.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/6_stack_house.jpeg) |
| (d) Letter Aligning | ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/1_letter_align.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/6_letter_align.jpeg) |
| (e) Shorts Folding | ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/1_short_folding.jpeg) → ![](https://arxiv.org/html/2606.19784/2606.19784v1/figures/experiment/6_short_folding.jpeg) |

**说明**: 5 个 Mobile ALOHA 真实机器人任务，涵盖物体放置、抽屉交互、方向对齐、堆叠和双手可变形物体操作，每个任务各有 150 条演示数据。

---

### Table 1: LIBERO Benchmark 结果

| 方法 | 控制模式 | LIBERO-10 | LIBERO-Goal | LIBERO-Obj | LIBERO-Spat | 平均↑ |
|------|----------|-----------|-------------|------------|-------------|-------|
| π₀ | Rel. | 73.0 | 93.0 | 86.0 | 90.0 | 86.0 |
| OpenVLA | Rel. | 55.0 | 79.2 | 88.4 | 84.7 | 76.8 |
| SmolVLA | Rel. | 61.0 | 61.4 | 66.0 | 74.0 | 65.6 |
| GR00T N1.5 | Rel. | 72.0 | 75.0 | 83.4 | 82.0 | 78.1 |
| GR00T + EquiActor | Rel. | 82.6 | 88.0 | 95.2 | 98.2 | 91.0 |
| **EquiVLA** | **Rel.** | **87.6** | **89.4** | **98.0** | **95.4** | **92.6** |
| GR00T N1.5 | Abs. | 52.0 | 55.2 | 74.6 | 68.6 | 62.6 |
| GR00T + EquiActor | Abs. | 63.0 | 70.0 | 79.4 | 82.0 | 73.6 |
| **EquiVLA** | **Abs.** | **73.6** | **70.4** | **83.0** | **77.6** | **76.1** |

**说明**: EquiVLA 在相对控制模式下将基线 GR00T N1.5 从 78.1% 提升至 92.6%（+14.5pp）。单独使用 EquiActor 已达 91.0%，说明两个模块均有贡献。

---

### Table 2: CALVIN ABCD→D Benchmark

| 方法 | T1 | T2 | T3 | T4 | T5 | 平均序列长度↑ |
|------|-----|-----|-----|-----|-----|-------------|
| HULC | 88.9 | 73.3 | 58.7 | 47.5 | 38.3 | 3.07 |
| MoDE | 97.1 | 92.5 | 87.9 | 83.5 | 77.9 | 4.39 |
| GR00T N1.5 | 89.0 | 79.2 | 68.7 | 59.4 | 48.5 | 3.45 |
| GR00T + EquiActor | 93.7 | 85.8 | 77.8 | 70.1 | 61.9 | 3.89 |
| **EquiVLA** | **95.0** | **88.5** | **81.1** | **73.8** | **64.3** | **4.03** |

**说明**: EquiVLA 将 CALVIN ABCD→D 上的平均序列长度从 3.45 提升至 4.03（+0.58），接近多帧方法 MoDE（4.39），但无需时序历史。

---

### Table 3: Mobile ALOHA 真实机器人结果

| 模型 | Banana in Pot | Block Storing | House Building | Letter Aligning | Shorts Folding | 平均 |
|------|--------------|---------------|----------------|-----------------|----------------|------|
| **EquiVLA** | 15/20 | 11/20 | 10/20 | 19/20 | 17/20 | **72%** |
| GR00T N1.5 | 12/20 | 9/20 | 3/20 | 13/20 | 17/20 | 54% |

**说明**: 真实机器人上成功率从 54% → 72%（+18pp）。方向敏感任务提升最大：Letter Aligning +30pp，House Building +35pp。Shorts Folding（无方向需求）两者持平，验证等变性在有旋转对称需求的任务上的针对性提升。

---

### Table 4: 样本效率分析（LIBERO）

| 方法 | 10%（≈5 demos） | 40%（≈16 demos） | 100%（40 demos） |
|------|----------------|-----------------|-----------------|
| GR00T N1.5 | 58.4 | 73.9 | 78.1 |
| GR00T + EquiActor | 58.8 | 84.1 | 91.0 |
| **EquiVLA** | **60.2** | **84.5** | **92.6** |

**关键发现**: EquiVLA 的优势随数据量增加而加大——10% 时仅 +1.8pp，100% 时达 +14.5pp，说明等变性与更多数据产生协同效应，而非作为数据短缺的补偿。

---

### Table 5: 对称群消融（EquiPerceptor 的群阶选择）

| 群 | LIBERO-10 | LIBERO-Goal | LIBERO-Obj | LIBERO-Spat | 平均↑ | 推理延迟↓（ms/step） |
|----|-----------|-------------|------------|-------------|-------|---------------------|
| C₄ | 82.5 | 94.0 | 94.5 | 95.5 | 91.6 | 161 |
| **C₈** | **87.5** | **89.5** | **98.0** | **95.5** | **92.6** | **194** |
| C₁₆ | 87.0 | 95.4 | 98.2 | 96.4 | 94.3 | 243 |

**关键发现**: C₈ 是精度与速度的最优折中：相比 C₁₆ 仅损失 1.7pp 精度，但节省 49 ms/step 延迟。C₄ 虽最快，但因空间 token 对齐误差较大，精度略低。

---

### Table 6: 表示缺陷分析（Representation Defect）

| ViT patch 分辨率 $n$ | Token 总数 $N$ | 最大空间偏移 $\delta_{max}$ | 错位 token 数 $m$ | 占比 $m/N$ | Frobenius 范数 $\|D\|_F$ | 非零对数 |
|---------------------|----------------|---------------------------|-------------------|------------|--------------------------|---------|
| 14 | 196 | 2.74 | 68 | 34.7% | 11.66 | 38/64 |
| 16 | 256 | 3.15 | 80 | 31.2% | 12.65 | 34/64 |
| 24 | 576 | 4.79 | 176 | 30.6% | 18.76 | 38/64 |
| 32 | 1024 | 6.44 | 328 | 32.0% | 25.61 | 38/64 |

**说明**: 45° 旋转（$C_8$ 的非 $C_4$ 元素）导致约 31-35% 的 patch 位置错位，验证了理论分析中近似误差的来源。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 套题组，每任务 40 demos | 仿真操作 benchmark，含 Spatial/Object/Goal/Long 等变体 | 主要仿真评估 |
| CALVIN ABCD→D | 跨环境语言条件长序列 | 测试多步骤连续操作泛化 | 仿真评估 |
| Mobile ALOHA | 5 任务，每任务 150 demos | 真实双臂移动机器人 | 真实世界评估 |

### 实现细节

- **Backbone**: 冻结 GR00T N1.5（视觉语言主干）+ 从头训练 EquiActor
- **对称群**: $C_8$（8 个旋转方向，对应 45° 间隔）
- **EquiActor**: 等变 DiT，正则特征空间的可操控层，Flow Matching 目标
- **推理延迟**: 194 ms/step（vs 基线 64 ms/step，慢约 3×，因需 8 次冻结 ViT 前向传播）
- **等变误差**: 相比基线降低 27.3×（0.284 vs 7.754）

### 可视化结果

EquiVLA 在方向敏感任务上表现出定性差异：Letter Aligning 任务中，基线 GR00T 在物体旋转后失败，EquiVLA 能正确对齐；House Building 中等变性使叠放操作更精确（+35pp 成功率）。Shorts Folding 无方向依赖，两者表现相当，进一步佐证等变性的针对性。

---

## 批判性思考

### 优点
1. **即插即用设计**: EquiPerceptor 不修改冻结 VLM，可无缝迁移到任何基于 ViT+DiT 的 VLA 架构
2. **有理论保证**: 提供了近似等变性的形式界（Theorem 3），而非仅凭经验观察，科学基础扎实
3. **真实世界验证**: 在 Mobile ALOHA 真实机器人上 +18pp 的成功率提升，说明方法具备实用价值而非仅限仿真

### 局限性
1. **仅限 SO(2) 平面旋转**: 不处理俯仰/滚转等非平面旋转（SO(3)），实际操作场景中存在局限
2. **推理延迟增加约 3×**: Token-level FA 需 $|G|=8$ 次冻结 ViT 前向传播（194 ms vs 64 ms），可能不满足实时控制需求
3. **EquiActor 无法复用预训练权重**: 等变 DiT 需从头训练，无法利用现有大规模预训练动作头，知识蒸馏到等变架构是未解问题

### 潜在改进方向
1. **随机帧平均**（Stochastic Frame Averaging）: 推理时随机采样子集旋转方向，在精度与速度间动态权衡
2. **SO(3) 等变扩展**: 引入三维等变表示（如 SE(3)-equivariant 网络），处理完整的三维旋转对称性
3. **知识蒸馏**: 将预训练 DiT 权重的知识蒸馏到等变架构中，减少从头训练的数据需求

### 可复现性评估
- [ ] 代码开源（论文未提及）
- [ ] 预训练模型（论文未提及）
- [x] 训练细节完整（Appendix B 提供详细配置）
- [x] 数据集可获取（LIBERO、CALVIN 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[GR00T N1.5]]: 作为基础 VLA 架构，EquiVLA 在其上增加等变模块
- [[Frame Averaging]]: EquiPerceptor 的核心对称化技术（Puny et al., 2021）
- [[Flow Matching]]: EquiActor 使用的动作生成目标

### 对比
- [[π₀]]: LIBERO 上的对比基线（86.0% vs 92.6%）
- [[OpenVLA]]: LIBERO 上的对比基线
- [[HULC]]: CALVIN 上的对比基线
- [[MoDE]]: CALVIN 上领先的多帧方法（4.39，EquiVLA 达 4.03）

### 方法相关
- [[SO(2) Equivariance]]: 核心数学框架
- [[Diffusion Transformer]]: EquiActor 的架构基础
- [[Regular Representation]]: 等变层实现所用的群表示理论
- [[Steerable CNN]]: 等变神经网络的通用框架

### 硬件/数据相关
- [[Mobile ALOHA]]: 真实机器人实验平台（双臂移动操作）
- [[LIBERO]]: 仿真操作 benchmark
- [[CALVIN]]: 多步骤语言条件操作 benchmark

---

## 速查卡片

> [!summary] EquiVLA
> - **核心**: 将 SO(2) 旋转等变性注入大规模 VLA，无需重训冻结 VLM
> - **方法**: EquiPerceptor（Token-level FA 等变视觉）+ EquiActor（等变 DiT 动作头）
> - **结果**: LIBERO 92.6% vs 78.1%，真实机器人 72% vs 54%，等变误差降低 27.3×
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-19*
