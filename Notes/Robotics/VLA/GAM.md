---
title: "Geometric Action Model for Robot Policy Learning"
method_name: "GAM"
authors: [Jisang Han, Seonghu Jeon, Jaewoo Jung, René Zurbrügg, Honggyu An, Tifanny Portela, Marco Hutter, Marc Pollefeys, Seungryong Kim, Sunghwan Hong]
year: 2026
venue: arXiv
tags: [geometric-foundation-model, robot-manipulation, world-action-model, 3d-geometry, vla, depth-estimation, causal-transformer]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17046
created: 2026-06-16
---

# 论文笔记：Geometric Action Model for Robot Policy Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | KAIST AI, ETH Zurich, ETH AI Center |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[π0.5]], [[Cosmos-Policy]], [[Fast-WAM]], [[Spatial Forcing]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17046) / Code: — |

---

## 一句话总结

> GAM 将预训练几何基础模型（GFM）在中间层分割，前半段作为编码器、后半段作为解码器，并在分割层插入因果未来预测器，统一实现 3D 几何预测和机器人动作生成，以 1.4B 参数实现 55× 速度提升并保持 SOTA 鲁棒性。

---

## 核心贡献

1. **几何基础模型重用架构**: 将 [[Depth Anything 3|DA3-Giant]] 在层 $L_s=12$ 分割，复用其 3D 几何先验，无需独立感知头即可同时预测未来深度图和机器人动作。
2. **因果未来预测器（Causal Future Predictor）**: 在分割层插入 12 层因果 Transformer，在 [[Block-Causal Self-Attention|块因果自注意力]] 约束下预测未来潜在几何 token，实现时序世界建模。
3. **高效推理与强鲁棒性**: 模型仅 1.4B 参数，推理延迟 6.9ms（145 Hz），在 LIBERO-Plus 摄像头扰动评测上 +9.7%p，比 Cosmos Policy 快 55×。

---

## 问题背景

### 要解决的问题

现有机器人操作策略在摄像头视角发生扰动（位置移动、旋转）时泛化性极差，即使在模拟环境下训练效果良好的模型，在真实部署中也会因摄像头位置微小变化而大幅失败。同时，大型世界动作模型（WAM）推理速度慢（>100ms），难以实时控制。

### 现有方法的局限

| 方法类别 | 代表工作 | 局限 |
|----------|----------|------|
| [[VLA|视觉-语言-动作模型]] | OpenVLA、π0.5 | 缺乏显式 3D 几何理解，摄像头扰动后 OOD 性能急剧下降 |
| [[World-Action Model|世界动作模型 (WAM)]] | Cosmos Policy、Fast-WAM、WorldVLA | 在 2D 像素空间建模，推理延迟高（382ms），几何鲁棒性不足 |
| 几何感知 VLA | Spatial Forcing、ROCKET | 被动特征蒸馏，几何监督未深度融入架构，OOD 鲁棒性有限 |

### 本文的动机

[[Depth Anything 3|几何基础模型（GFM）]] 在大规模数据上预训练，内化了强大的 3D 场景理解能力。直接将 GFM 的中间层特征用于时序预测，既能保留几何先验，又避免训练独立感知网络的开销。3D 几何预测与机器人动作共享同一骨干网络，可实现参数高效的统一架构。

---

## 方法详解

### 模型架构

GAM 采用 **分割-插入-重用** 架构，骨干为 [[Depth Anything 3|DA3-Giant]]（在 [[Track4World]] 上微调），总参数 1.4B：

- **输入**: 语言指令 $\ell$（[[T5]] 编码，冻结）+ 多视角 RGB 观测 $\{o_{t'}\}_{t'=t-H+1}^{t}$ + 本体感知 $\{s_{t'}\}$ + 动作历史 $\{a_{t'}\}$
- **Backbone**: [[Depth Anything 3|DA3-Giant]]，在第 $L_s=12$ 层处分割
- **核心模块**: [[Block-Causal Self-Attention|块因果预测器]] 插入分割层，预测未来几何潜在 token
- **输出**: 未来深度图 $\hat{d}_{t+1}$ + [[Action Chunking|动作块]] $\hat{a}_{t:t+k}$（7D 末端执行器，8 步 chunk）
- **总参数**: 1.4B（可训练 983M）

架构三阶段：

```
多视角 RGB 观测
      ↓
[观测编码器 E≤Ls]  ← DA3 前 12 层
      ↓ 几何潜在 token Z^(Ls)
[因果未来预测器 g_ϕ]  ← 12 层因果 Transformer（新增）
      ↓ 预测 token Z̃^(Ls) + 动作 token ã
[特征传播 & 解码 D>Ls]  ← DA3 后续层
      ↓
   深度头 h_depth + 动作头 h_act
```

### 核心模块

#### 模块1: 观测编码器（Observation Encoder）

**设计动机**: 利用 [[Depth Anything 3|GFM]] 的浅层提取蕴含 3D 先验的几何潜在特征，无需额外感知头。

**具体实现**:
- 将 $V$ 个视角的 RGB 图像各分为 $P$ 个 patch，每帧构造 token 序列：

$$
\mathbf{z}_{v}^{(0)}=\left[\mathbf{c}_{v}, \mathbf{x}_{v}^{1}, \ldots, \mathbf{x}_{v}^{P}\right] \in \mathbb{R}^{(1+P) \times d}
$$

- 通过 GFM 块（含帧内 frame attention 和跨视角/时序 global attention）逐层演化：

$$
\mathbf{Z}^{(m)}=f^{(m)}\left(\mathbf{Z}^{(m-1)}\right), \quad f^{(m)} \in\left\{f_{\text{frame}}^{(m)}, f_{\text{global}}^{(m)}\right\}
$$

- 在第 $L_s$ 层输出几何潜在状态序列 $\{\mathbf{Z}_{t-H+1}^{(L_s)}, \ldots, \mathbf{Z}_{t}^{(L_s)}\}$

#### 模块2: 因果未来预测器（Causal Future Predictor）

**设计动机**: 在不修改 GFM 权重的前提下，插入轻量因果 Transformer 实现 [[时序世界建模]]，预测下一帧的几何潜在状态。

**具体实现**:
- 将本体感知和动作历史通过轻量映射 $\psi_s, \psi_a$ 嵌入为辅助 token，与几何 token 拼接为输入块：

$$
\mathbf{U}_{t'} = [\mathbf{p}_{t'}; \mathbf{q}_{t'}; \mathbf{Z}_{t'}^{(L_s)}]
$$

- 采用 [[Block-Causal Self-Attention|块因果自注意力]]，确保预测 $t'+1$ 时刻特征时不泄露未来信息
- 12 层因果 Transformer（宽度 1024），输出预测未来潜在 token $\tilde{\mathbf{Z}}_{t'+1}^{(L_s)}$ 和动作 token $\tilde{\mathbf{a}}_{t'}$

#### 模块3: 特征传播与解码（Feature Propagation & Decoding）

**设计动机**: 复用 GFM 深层块（$D_{>L_s}$）同时解码未来几何和机器人动作，参数共享实现多任务高效推理。

**具体实现**:
- 单个动作 token 复制到所有 $V$ 个视角，与几何 token 拼接后通过 GFM 后续层传播：

$$
\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(M)}=\left(f^{(M)} \circ \cdots \circ f^{(L_{s}+1)}\right)\left(\left[\left[\tilde{\mathbf{Z}}_{1, t^{\prime}+1}^{(L_{s})} ; \tilde{\mathbf{a}}_{1, t^{\prime}}\right], \ldots,\left[\tilde{\mathbf{Z}}_{V, t^{\prime}+1}^{(L_{s})} ; \tilde{\mathbf{a}}_{V, t^{\prime}}\right]\right]\right)
$$

- 在 global attention 层中施加扩展因果掩码，防止信息泄露
- **动作头** $h_{act}$：从动作 token 回归 8 步动作 chunk（7D 末端执行器）
- **深度头** $h_{depth}$：从几何 token 解码未来深度图，提供辅助几何监督

---

## 关键公式

### 公式1: [[策略学习|策略函数定义]]

$$
\pi_{\theta}:\left(\{o_{t-H+1}, \ldots, o_{t}\},\{s_{t-H+1}, \ldots, s_{t}\},\{a_{t-H}, \ldots, a_{t-1}\}, \ell\right) \mapsto \hat{a}_{t}
$$

**含义**: GAM 的策略函数，将历史观测、本体感知、动作历史及语言指令映射为当前动作预测。

**符号说明**:
- $o_{t}$: $t$ 时刻多视角 RGB 观测
- $s_{t}$: $t$ 时刻本体感知状态
- $a_{t}$: $t$ 时刻动作（7D 末端执行器）
- $\ell$: 语言指令（T5 编码）
- $H$: 历史帧数（默认 $H=1$）

### 公式2: [[Block-Causal Self-Attention|GFM 编解码器分割]]

$$
E_{\leq L_{s}}=f^{(L_{s})} \circ \cdots \circ f^{(1)}, \quad D_{>L_{s}}=f^{(M)} \circ \cdots \circ f^{(L_{s}+1)}
$$

**含义**: 将 GFM 在第 $L_s$ 层处分割，前半段作为观测编码器，后半段作为特征解码器，两者之间插入因果预测器。

**符号说明**:
- $L_s$: 分割层（最优值为 12）
- $M$: GFM 总层数
- $f^{(m)}$: 第 $m$ 层 GFM 块（帧内或全局 attention）

### 公式3: [[时序世界建模|特征传播]]

$$
\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(M)}=\left(f^{(M)} \circ \cdots \circ f^{(L_{s}+1)}\right)\left(\left[\left[\tilde{\mathbf{Z}}_{1, t^{\prime}+1}^{(L_{s})} ; \tilde{\mathbf{a}}_{1, t^{\prime}}\right], \ldots,\left[\tilde{\mathbf{Z}}_{V, t^{\prime}+1}^{(L_{s})} ; \tilde{\mathbf{a}}_{V, t^{\prime}}\right]\right]\right)
$$

**含义**: 将预测的几何 token 与动作 token 拼接，通过 GFM 解码器传播，同时生成未来深度预测和可执行动作。

**符号说明**:
- $\tilde{\mathbf{Z}}_{v,t'+1}^{(L_s)}$: 视角 $v$ 在 $t'+1$ 时刻的预测几何 token
- $\tilde{\mathbf{a}}_{v,t'}$: 视角 $v$ 的动作 token（全视角共享同一动作）
- $V$: 摄像头数量（双摄像头设置）

### 公式4: [[World-Action Model|多任务训练损失]]

$$
\mathcal{L}_{\text{total}}=\lambda_{\text{act}} \mathcal{L}_{\text{act}}+\lambda_{\text{feat}} \mathcal{L}_{\text{feat}}+\lambda_{\text{depth}} \mathcal{L}_{\text{depth}}
$$

**含义**: 三任务联合训练，动作回归、潜在特征对齐、深度监督三者加权求和。

**符号说明**:
- $\lambda_{\text{act}}=3$: 动作损失权重
- $\lambda_{\text{feat}}=1$: 特征对齐损失权重
- $\lambda_{\text{depth}}=3$: 深度损失权重
- $\mathcal{L}_{\text{act}}$: 动作回归损失（MSE）
- $\mathcal{L}_{\text{feat}}$: 未来特征对齐损失（L1）
- $\mathcal{L}_{\text{depth}}$: 未来深度监督损失

### 公式5: [[时序世界建模|未来特征对齐损失]]

$$
\mathcal{L}_{\text{feat}}=\sum_{t^{\prime} \in \mathcal{H}}\left\|\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(L_{s})}-\mathbf{Z}_{t^{\prime}+1}^{(L_{s})}\right\|_{1}
$$

**含义**: 监督预测的未来几何潜在 token 与真实未来帧的 GFM 编码特征对齐，L1 范数保证鲁棒性。

**符号说明**:
- $\tilde{\mathbf{Z}}_{t'+1}^{(L_s)}$: 因果预测器输出的预测未来 token
- $\mathbf{Z}_{t'+1}^{(L_s)}$: 真实未来帧通过编码器得到的 token（教师信号）
- $\mathcal{H}$: 预测时间步集合

---

## 关键图表

### Figure 1: GAM 系统概览

![Figure 1](https://arxiv.org/html/2606.17046v1/x1.png)

**说明**: GAM 的整体框架。(a) 在共享 [[Depth Anything 3|几何骨干]] 内联合预测未来 3D 几何和动作 chunk；(b) 与现有基线对比，GAM 在保持高成功率的同时大幅降低延迟（6.9ms）并减小模型尺寸（1.4B）。

### Figure 2: 方法范式对比

![Figure 2](https://arxiv.org/html/2606.17046v1/x2.png)

**说明**: 三类方法对比示意图。(左) [[World-Action Model|视频 WAM]] 在 2D 像素空间建模，缺乏几何理解；(中) 几何感知 VLA 被动蒸馏几何特征；(右) GAM 统一感知、几何预测和动作解码于单一几何骨干。

### Figure 3: GAM 详细架构

![Figure 3](https://arxiv.org/html/2606.17046v1/x3.png)

**说明**: 展示编码器分割、[[Block-Causal Self-Attention|因果预测器]] 插入、以及通过 GFM 后续层进行特征传播与双头解码的完整流程。扩展因果掩码防止未来信息泄露。

### Figure 4: 真实世界实验结果

![Figure 4](https://arxiv.org/html/2606.17046v1/x4.png)

**说明**: 四项真实世界任务（抓放、叠放、锅具放置、立方体插入）在域内（ID）和域外（OOD，摄像头偏移 85cm + 旋转 45°）条件下的成功率。GAM 在 ID 和 OOD 下均优于 π0.5 和 Spatial Forcing。

### Figure 5: 摄像头扰动鲁棒性分析

![Figure 5](https://arxiv.org/html/2606.17046v1/x5.png)

**说明**: 随扰动难度递增，各方法成功率变化曲线。GAM 在高强度扰动下仍保持显著高于基线的成功率，体现了 3D 几何先验带来的视角不变性。

### Figure 6: 预训练数据集构成

![Figure 6](https://arxiv.org/html/2606.17046v1/x6.png)

**说明**: 预训练混合数据集比例。[[Open X-Embodiment]] (72%) + MimicGen (18%) + RoboCasa365 (10%)，共 784K 条单臂轨迹。

### Figure 7: 真实世界实验环境设置

![Figure 7](https://arxiv.org/html/2606.17046v1/x7.png)

**说明**: ID（标准摄像头位置）和 OOD（摄像头扰动）的实验设置对比，展示外部摄像头的平移/旋转幅度。

### Figure 8: 四项真实操作任务

![Figure 8](https://arxiv.org/html/2606.17046v1/x8.png)

**说明**: 四项任务的场景图示：(1) 抓放（284 演示）、(2) 叠放牛奶与立方体（202 演示）、(3) 将锅/平底锅放上烹饪台（184 演示）、(4) 将立方体插入带盖的锅中（169 演示）。

### Figure 9: LIBERO-Plus 细粒度鲁棒性

![Figure 9](https://arxiv.org/html/2606.17046v1/x9.png)

**说明**: 在 LIBERO-Plus 各扰动维度（摄像头、机器人、语言、光照、背景、噪声、布局）下的零样本鲁棒性详细对比。GAM 在摄像头扰动（83.1%）和噪声（95.3%）上显著领先。

### Figure 10: 未来深度预测可视化

![Figure 10a](https://arxiv.org/html/2606.17046v1/x10.png)

![Figure 10b](https://arxiv.org/html/2606.17046v1/x11.png)

![Figure 10c](https://arxiv.org/html/2606.17046v1/x12.png)

![Figure 10d](https://arxiv.org/html/2606.17046v1/x13.png)

**说明**: GAM 在 LIBERO 各任务套件上预测的未来深度图。预测结果与真实深度高度一致，证明 [[时序世界建模]] 在几何空间的有效性。

### Figure 11: 动作 Token 注意力可视化

![Figure 11](https://arxiv.org/html/2606.17046v1/x14.png)

**说明**: 动作 token 的注意力热图，显示模型在生成动作时重点关注任务相关的接触区域（物体表面、末端执行器附近），体现了几何先验的有效引导。

---

### Table 1: LIBERO 和 LIBERO-Plus 评测结果

| Method | Size | Orig. | Plus | Cam. | Robot | Lang. | Light | BG | Noise | Layout |
|--------|------|-------|------|------|-------|-------|-------|-----|-------|--------|
| [[π0.5]] | 3.3B | 96.9 | 84.6 (↓12.3) | 72.0 | 76.6 | 86.5 | 96.1 | 95.2 | 86.7 | 86.0 |
| OpenVLA-OFT | 7B | 97.1 | 69.6 (↓27.5) | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.3 |
| RIPT-VLA | 7B | 97.5 | 68.4 (↓29.1) | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 |
| π0 | 3.3B | 91.3 | 69.3 (↓22.0) | 61.0 | 40.8 | 63.7 | 89.3 | 84.1 | 80.1 | 75.9 |
| π0-FAST | 3.3B | 85.5 | 61.6 (↓23.9) | 65.1 | 21.6 | 61.0 | 73.2 | 73.3 | 74.4 | 68.8 |
| UniVLA | 8.5B | 95.2 | 42.9 (↓52.3) | 1.8 | 46.2 | 69.5 | 69.0 | 81.0 | 21.2 | 31.9 |
| NORA | 3B | 87.9 | 39.0 (↓48.9) | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 |
| OpenVLA | 7B | 76.5 | 15.6 (↓60.9) | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 |
| [[Cosmos-Policy]] | 2B | 98.5 | 82.4 (↓16.1) | 73.4 | 63.3 | 89.3 | 98.9 | 83.5 | 89.3 | 84.0 |
| [[Fast-WAM]] | 6B | 97.6 | 50.0 (↓47.5) | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 |
| WorldVLA | 7B | 79.1 | 25.0 (↓54.1) | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 11.0 | 38.0 |
| π0.5 + Spatial Forcing | 3.3B | 94.0 | 25.7 (↓58.3) | 0.1 | 0.3 | 26.8 | 66.0 | 45.9 | 0.1 | 59.8 |
| π0.5 + ROCKET | 3.3B | 95.3 | 47.5 (↓46.6) | 30.9 | 75.6 | 29.3 | 69.2 | 47.0 | 25.4 | 62.0 |
| **GAM (Ours)** | **1.4B** | **97.6** | **85.5 (↓12.1)** | **83.1** | **70.0** | **84.8** | **97.2** | **94.3** | **95.3** | **79.1** |

**说明**: GAM 以最小的 OOD 性能衰减（↓12.1）与 π0.5（↓12.3）并列最优，同时摄像头扰动（83.1%）和噪声扰动（95.3%）项目排名第一。UniVLA、Fast-WAM 等 WAM 系方法在摄像头扰动下几乎完全失效（1-16%）。

### Table 2: 组件消融实验（后训练阶段）

| Pretrain | $\mathcal{L}_{depth}$ | $\mathcal{L}_{feat}$ | $H$ | Orig. SR (%) | Plus SR (%) |
|----------|-----------------------|----------------------|-----|--------------|-------------|
| ✓ | ✓ | ✓ | 1 | **99.6** | **89.7** |
| ✓ | ✓ | ✓ | 2 | 97.2 | 84.4 |
| ✓ | ✓ | ✓ | 4 | 98.2 | 85.1 |
| ✓ | ✗ | ✓ | 1 | 98.4 | 89.0 |
| ✓ | ✗ | ✗ | 1 | 98.6 | 89.5 |
| ✓ | ✓ | ✗ | 1 | 99.6 | 89.7 |
| ✗ | ✓ | ✓ | 1 | 98.4 | 73.4 |
| ✗ | ✗ | ✓ | 1 | 95.2 | 66.5 |
| ✗ | ✓ | ✗ | 1 | 96.4 | 80.0 |
| ✗ | ✗ | ✗ | 1 | 93.6 | 50.0 |

**关键发现**: (1) 预训练是最关键组件，移除后 Plus 从 89.7% 降至 50.0%；(2) $H=1$ 最优，更长历史引入伪相关；(3) 有预训练时辅助损失（$\mathcal{L}_{depth}, \mathcal{L}_{feat}$）影响有限，无预训练时帮助显著。

### Table 3: 分割层消融（$L_s$）

| 分割层 $L_s$ | Orig. (%) | Plus (%) |
|--------------|-----------|----------|
| 0（无编码器） | 5.4 | 1.8 |
| **12（最优）** | **99.6** | **70.1** |
| 19 | 95.6 | 63.4 |
| 27 | 1.2 | 1.6 |
| 33 | 0.0 | 0.0 |
| 39 | 0.0 | 0.0 |

**关键发现**: 分割层选择对性能影响极大。$L_s=12$ 平衡了特征丰富度与解码器深度；过浅（$L_s=0$）缺乏几何先验；过深（$L_s \geq 27$）残留解码器层不足，性能崩溃。

### Table 4: 推理延迟对比

| Method | Size | 延迟 (ms) | 频率 (Hz) |
|--------|------|-----------|-----------|
| OpenVLA-OFT | 7B | 77.8 | ~13 |
| [[π0.5]] | 3.3B | 29.2 | ~34 |
| [[Cosmos-Policy]] | 2B | 382.4 | ~3 |
| **GAM (Ours)** | **1.4B** | **6.9** | **~145** |

**关键发现**: GAM 比 Cosmos Policy 快 **55×**，比 π0.5 快 **4.2×**，以最小参数量实现最快推理，适合实时机器人控制（>100 Hz）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Open X-Embodiment]] | ~563K 轨迹（72%）| 多机器人、多任务 | 预训练 |
| MimicGen | ~141K 轨迹（18%）| 自动生成演示 | 预训练 |
| RoboCasa365 | ~78K 轨迹（10%）| 厨房场景 | 预训练 |
| LIBERO-Orig | 50 trials/task | 5 任务套件 | 仿真评测 |
| LIBERO-Plus | 50 trials/task | 7 种扰动类型 | 仿真 OOD 评测 |
| RoboCasa-Kitchen | 24 任务 | 厨房操作 | 仿真评测 |
| 真实世界 | 4 任务，~200 demos | 双臂操作台 | 真实评测 |

### 实现细节

- **Backbone**: DA3-Giant（在 Track4World 上微调），在 $L_s=12$ 层分割
- **因果预测器**: 12 层 Transformer，宽度 1024
- **语言编码器**: 冻结 T5
- **动作空间**: 7D 末端执行器，8 步 chunk
- **输入分辨率**: 224×224 RGB，双摄像头
- **预训练硬件**: 64 NVIDIA GH200 GPU，batch size 1024，约 96 小时
- **可训练参数**: ~983M / 总 1.4B

### 可视化结果

Figure 10 展示的未来深度预测在 LIBERO 各套件中均准确预测了物体位移和机器人臂运动，表明模型在几何层面真正理解了操作动态。Figure 11 的注意力图显示动作 token 能自动定位到接触相关区域，无需显式监督。

---

## 批判性思考

### 优点

1. **几何先验高效复用**: 分割-插入架构几乎不修改 GFM，复用了大量预训练知识，参数效率极高（1.4B 优于众多 3-8B 模型）。
2. **推理速度极快**: 6.9ms（145 Hz）是目前同类方法中最快的，且无需专用硬件加速技巧，直接来自架构设计。
3. **OOD 摄像头鲁棒性**: LIBERO-Plus 摄像头扰动项（83.1%）遥遥领先，体现了 3D 几何先验的真实价值，而非单纯指标刷分。

### 局限性

1. **语言推理能力受限**: 使用冻结 T5 作为语言编码器，缺乏大型 LLM 的常识推理能力，复杂指令理解受限。
2. **依赖 GFM 预训练质量**: 方法效果高度依赖 DA3-Giant 的预训练质量和 Track4World 微调，不同 GFM 的迁移性未验证。
3. **双摄像头固定设置**: 目前仅验证了固定双摄像头配置，对更复杂多视角或单目设置的泛化性未探索。
4. **单臂操作**: 仅验证单臂场景，双臂协同操作的可扩展性未知。

### 潜在改进方向

1. **集成更强语言模型**: 将冻结 T5 替换为大型 LLM，提升指令理解和常识推理能力。
2. **多机器人平台迁移**: 探索方法在四足机器人、双臂机器人等平台上的适用性。
3. **在线适应**: 结合在线强化学习或人类纠正反馈，提升样本效率。

### 可复现性评估

- [ ] 代码开源（论文发布时未提供）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（附录提供了详细超参数）
- [x] 数据集可获取（Open X-Embodiment、MimicGen、RoboCasa 均公开）

---

## 关联笔记

### 基于

- [[Depth Anything 3]]: GFM 骨干，Track4World 微调版本（DA3-Giant）
- [[Track4World]]: 用于 GFM 微调的前馈密集 3D 追踪数据集
- [[Action Chunking]]: 动作预测范式（8 步 chunk，7D 末端执行器）

### 对比

- [[π0.5]]: 当前 SOTA VLA，LIBERO-Plus 84.6% vs GAM 85.5%
- [[Cosmos-Policy]]: 代表性 WAM，推理 382ms vs GAM 6.9ms
- [[Fast-WAM]]: 快速 WAM 基线，LIBERO-Plus 仅 50.0%
- [[Spatial Forcing]]: 几何感知 VLA，OOD 摄像头几乎失效（0.1%）

### 方法相关

- [[World-Action Model]]: GAM 融合了 WAM 的时序预测思想
- [[Block-Causal Self-Attention]]: 因果预测器的核心 attention 机制
- [[时序世界建模]]: GAM 在几何潜在空间进行时序世界建模
- [[Imitation Learning]]: 预训练和微调均基于模仿学习

### 硬件/数据相关

- [[Open X-Embodiment]]: 主要预训练数据集（72%）
- [[LIBERO]]: 主要仿真评测 benchmark
- [[RoboCasa-Kitchen]]: 厨房任务评测 benchmark

---

## 速查卡片

> [!summary] Geometric Action Model (GAM)
> - **核心**: 将几何基础模型（DA3-Giant）在中间层分割，插入因果预测器，统一 3D 几何预测与机器人动作生成
> - **方法**: GFM 分割（$L_s=12$）+ 12 层块因果 Transformer + 双头解码（动作 + 深度）
> - **结果**: LIBERO-Plus 85.5%（OOD 摄像头 83.1%，↓仅 12.1%），推理 6.9ms（145 Hz，比 Cosmos Policy 快 55×）
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-16*
