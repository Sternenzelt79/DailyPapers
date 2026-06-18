---
title: "Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation"
method_name: "Mem-World"
authors: [Zirui Zheng, Jiaqian Yu, Xiongfeng Peng, Jun Shi, Mingyi Li, Chao Zhang, Weiming Li, Dong Wang, Huchuan Lu, Xu Jia]
year: 2026
venue: arXiv
tags: [world-model, robot-manipulation, memory-augmented, video-prediction, surfel-map, action-conditioned]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.18960
created: 2026-06-18
---

# 论文笔记：Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Dalian University of Technology; Samsung R&D Institute China-Beijing |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[Ctrl-World]], [[Cosmos Predict 2.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.18960) / Code 未公开 |

---

## 一句话总结

> Mem-World 提出 4D surfel 索引的腕部视角记忆模块 W-VMem，让动作条件世界模型在长时域机器人操作中保持几何感知的场景一致性，相比基线将真实世界策略性能相关系数提升 14.5%。

---

## 核心贡献

1. **记忆增强的多视角世界模型**: 将 [[Surfel Map|4D surfel 记忆]] 与 [[Action-Conditioned Video Generation|动作条件视频预测]] 结合，实现持久的长时域多视角世界模型推演。
2. **W-VMem 架构**: 提出以腕部视角为中心的 surfel 索引记忆，每个 surfel 存储几何属性、时间戳和操作对象标志，支持几何感知的历史帧检索。
3. **策略评估可靠性提升**: 世界模型预测与真实策略成功率的 Pearson 相关系数从 0.85 提升到 0.97（14.5% 提升），并通过合成数据使长时任务成功率从 58% 提升到 72%。

---

## 问题背景

### 要解决的问题

长时域机器人操作任务（如多物体摆放）需要世界模型进行几十步的自回归推演。在此过程中，腕部相机的剧烈运动和末端执行器对场景的遮挡会导致传统世界模型"遗忘"历史场景信息，无法保持物体外观的一致性。

### 现有方法的局限

- [[Ctrl-World]] 等方法仅使用固定步长的历史帧（stride-based）作为条件，无法感知哪些历史帧对当前视角最具几何相关性。
- 短时记忆（相邻帧）只覆盖近期观测，在相机大幅度往返运动后丢失远期场景信息。
- [[VMem]] 等已有 surfel 方法使用静态 3D surfel，缺乏对动态操作场景的时间感知，无法区分被操作物体与静态背景。

### 本文的动机

腕部相机提供了最以操作为中心的观测，其位姿可通过前向运动学从关节动作精确计算。利用 surfel 结构将历史帧与 3D 表面元素关联，可以在相机运动后精确检索曾经可见的场景区域，从而维持一致的长时域视觉预测。

---

## 方法详解

### 模型架构

Mem-World 采用 **记忆增强的多视角动作条件世界模型** 架构：

- **输入**: 当前多视角观测 $\hat{o}_t$（第三视角 × 2 + 腕部视角）+ 历史帧记忆 $\mathcal{H}_t$ + 未来动作序列 $a_{t+1:t+H}$
- **Backbone**: [[Ctrl-World]]（预训练视频扩散模型，基于 [[Video Diffusion Model]] 微调）
- **核心模块**: [[Surfel Map|W-VMem]] 用于几何感知的历史帧检索与记忆维护
- **输出**: 未来多视角观测预测 $\hat{o}_{t+1:t+H}$

### 核心模块

#### 模块1: W-VMem（Wrist-View Memory，腕部视角记忆）

**设计动机**: 利用 [[Surfel Map|4D surfel 结构]] 将历史腕部观测锚定到随时间演化的 3D 表面元素上，实现几何感知的历史帧检索。

**Surfel 定义**：每个 surfel $s_k$ 存储五元组：

$$
s_k = (p_k,\; n_k,\; r_k,\; t_k,\; m_k)
$$

- $p_k$：3D 位置
- $n_k$：表面法向量
- $r_k$：surfel 半径
- $t_k$：创建/更新时间戳
- $m_k$：二值操作对象标志（1 = 被操作物体，0 = 静态背景）

**初始化**：从初始帧的三视角观测构建 surfel 地图，提供完整的初始场景覆盖。

**更新策略**：仅用腕部视角帧更新 surfel，原因是：
1. 腕部视角提供最以操作为中心的观测；
2. 若用广角第三视角更新，会混淆 surfel 被腕部相机实际观测的时间信息，破坏时间关联。

#### 模块2: 几何感知检索（Geometry-Aware Retrieval）

**设计动机**: 通过打分函数综合考虑几何可见性、任务相关性和时间近因性，为当前预测步骤检索最相关的历史帧。

**具体实现**:

- 使用 [[Surfel Map|surfel 打分函数]]（见公式 2）对所有 surfel 排序
- 通过 [[Non-Maximum Suppression|NMS（非极大值抑制）]] 防止重复采样同一区域
- 选出 Top-K surfel 对应的历史腕部帧作为世界模型的几何感知上下文

#### 模块3: 相机位姿锚定（Camera Pose Grounding）

**设计动机**: 将策略预测的未来关节动作转化为未来腕部相机位姿，无需额外的位姿估计模块。

**具体实现**:
- 利用 [[Forward Kinematics|前向运动学]] 将策略输出的关节动作转换为末端执行器位姿
- 利用腕部相机与末端执行器之间的固定刚性变换，推导未来腕部相机位姿
- 基于未来位姿序列计算平均观测方向 $\bar{v}_w$，用于 surfel 打分

---

## 关键公式

### 公式1: [[Action-Conditioned Video Generation|世界模型预测]]

$$
\hat{o}_{t+1:t+H} \sim W(\cdot \mid \hat{o}_t,\; \mathcal{H}_t,\; a_{t+1:t+H})
$$

**含义**: 世界模型 $W$ 以当前观测 $\hat{o}_t$、历史记忆 $\mathcal{H}_t$ 和未来动作序列 $a_{t+1:t+H}$ 为条件，自回归预测未来 $H$ 步的多视角观测。

**符号说明**:
- $\hat{o}_t$：当前多视角观测（含第三视角和腕部视角）
- $\mathcal{H}_t$：由 W-VMem 检索出的历史帧集合
- $a_{t+1:t+H}$：策略模型预测的未来动作块
- $H$：预测步长

### 公式2: [[Surfel Map|Surfel 打分函数]]

$$
\text{score}(s, t) = \frac{\langle n_s,\; \bar{v}_w \rangle}{1 + d_s} \cdot \ln(e + m_s) \cdot \left[\lambda_{\min} + (1 - \lambda_{\min}) \cdot 2^{-\frac{T - t}{H}}\right]
$$

**含义**: 对每个 surfel $s$ 在时刻 $t$ 的综合得分，融合了几何可见性、操作相关性和时间近因性三项因子，用于选出最值得检索的历史帧。

**符号说明**:
- $n_s$：surfel 的表面法向量
- $\bar{v}_w$：未来腕部相机的平均观测方向
- $d_s$：surfel 的深度（距离）
- $m_s$：操作对象二值标志（$m_s \in \{0, 1\}$），$\ln(e + m_s)$ 使被操作物体得分更高
- $\lambda_{\min}$：最小时间衰减因子，设为 0.1，防止过久历史帧被完全忽略
- $T$：当前最新时间步
- $t$：surfel 的最后更新时间步
- $H$：时间半衰期，设为 $0.3 \times T_{\max}$，控制时间衰减速度

**三项因子含义**:
- **因子1** $\frac{\langle n_s, \bar{v}_w \rangle}{1+d_s}$：几何可见性——法向量与观测方向对齐、且距离近的 surfel 得分更高
- **因子2** $\ln(e + m_s)$：任务相关性——被操作物体对应的 surfel 得分更高（$m_s=1$ 时约为 $m_s=0$ 的 1.31 倍）
- **因子3** $\lambda_{\min} + (1-\lambda_{\min}) \cdot 2^{-(T-t)/H}$：时间近因性——更近期更新的 surfel 得分更高，指数衰减

---

## 关键图表

### Figure 1: Mem-World 整体 Pipeline

![Figure 1 - Pipeline of Mem-World](https://arxiv.org/html/2606.18960v1/figs/pipeline.png)

**说明**: Mem-World 的完整推演流程。在推演过程中，[[Surfel Map|腕部视角 4D surfel 记忆]] W-VMem 维护历史观测的几何感知表示，并检索相关历史帧。给定检索到的历史帧和未来动作序列，[[Action-Conditioned Video Generation|多视角动作条件世界模型]] 预测后续观测，预测结果再被整合回记忆模块，实现带更新记忆的迭代推演。

### Figure 2: 长时域推演定性结果

![Figure 2 - Qualitative results on long-horizon rollouts](https://arxiv.org/html/2606.18960v1/figs/comparision.png)

**说明**: Mem-World 展示了持久且时间一致的世界建模：夹爪短暂遮挡后瓶盖仍可恢复，相机来回运动后黑色零食包仍能一致地再现。对比 [[Ctrl-World]]，后者在相机剧烈运动后出现物体外观扭曲或消失的问题。

### Figure 3: 记忆检索策略消融定性对比

![Figure 3 - Ablation on memory retrieval strategies](https://arxiv.org/html/2606.18960v1/figs/ablation.png)

**说明**: W-VMem 通过检索几何相关历史帧，帮助世界模型在未来观测中保持一致的物体外观。Short-term Mem（短时记忆）和 Stride Mem（固定步长记忆）提供的上下文信息不足，导致勺子在推演中出现变形或消失。

### Figure 4: 真实世界与世界模型的定量相关性

![Figure 4 - Quantitative correlations between real-world and world-model](https://arxiv.org/html/2606.18960v1/figs/correlation.png)

**说明**: Mem-World 与真实世界策略成功率的 Pearson 相关系数为 0.97，远高于 [[Ctrl-World]] 的 0.85，表明 Mem-World 能更可靠地用于策略筛选和评估。

### Figure 5: 策略提升结果

![Figure 5 - Policy improvement](https://arxiv.org/html/2606.18960v1/figs/policy_improvement.png)

**说明**: 使用 Mem-World 生成的合成数据对 [[π0.5]] 策略进行后训练后，长时域多物体放置任务的平均成功率从 58% 提升至 72%。

### Figure 6: 真实世界与世界模型推演对比

![Figure 6 - Comparisons between π0.5 rollouts](https://arxiv.org/html/2606.18960v1/figs/policy_eval.png)

**说明**: 对比真实世界、[[Ctrl-World]] 和 Mem-World 的 [[π0.5]] 策略推演，直观展示 Mem-World 在场景一致性上的优势。

### Table 1: 世界模型一致性定量评估

| 评估相机 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Obj. Con. ↑ |
|----------|------|---------|---------|---------|-------------|
| 第三视角 | Cosmos Predict 2.5 | 22.80 | 0.819 | 0.089 | 0.579 |
| 第三视角 | Ctrl-World | 23.17 | 0.828 | 0.076 | 0.573 |
| 第三视角 | **Mem-World (Ours)** | **25.30** | **0.878** | **0.054** | **0.619** |
| 腕部视角 | Ctrl-World | 17.34 | 0.623 | 0.281 | 0.476 |
| 腕部视角 | **Mem-World (Ours)** | **19.21** | **0.691** | **0.236** | **0.524** |

**说明**: Mem-World 在第三视角和腕部视角上全面超越 [[Ctrl-World]] 和 [[Cosmos Predict 2.5]]。其中第三视角 PSNR 提升约 2.1dB（+9%），腕部视角 PSNR 提升约 1.9dB（+11%）。

### Table 2: 记忆设计消融实验

| 评估相机 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Obj. Con. ↑ |
|----------|------|---------|---------|---------|-------------|
| 第三视角 | Short-term Mem | 21.25 | 0.802 | 0.082 | 0.526 |
| 第三视角 | Stride Mem | 22.58 | 0.814 | 0.079 | 0.544 |
| 第三视角 | **W-VMem (Ours)** | **24.78** | **0.869** | **0.062** | **0.597** |
| 腕部视角 | Short-term Mem | 15.04 | 0.526 | 0.353 | 0.401 |
| 腕部视角 | Stride Mem | 17.06 | 0.614 | 0.295 | 0.463 |
| 腕部视角 | **W-VMem (Ours)** | **18.97** | **0.680** | **0.248** | **0.502** |

**关键发现**: 几何感知检索策略（W-VMem）相比固定步长检索（Stride Mem）在腕部视角 PSNR 上提升约 1.9dB，相比短时记忆（Short-term Mem）提升约 4dB，验证了时间感知几何检索的必要性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | 95,599 条轨迹 | 多机器人、多场景；含多视角 RGB + 深度 | 训练世界模型（采样 11K 条） |
| 自采真实数据 | 34 条精选轨迹 | 含末端执行器遮挡和大幅腕部相机运动 | 一致性评估 |
| 合成数据 | Mem-World 生成 | 基于真实轨迹扩展的长时域合成场景 | 策略后训练 |

### 实现细节

- **Backbone**: [[Ctrl-World]]（预训练 [[Video Diffusion Model|视频扩散模型]]，基于机器人数据集微调）
- **机器人**: Franka Emika Panda，2-DOF 夹爪，1 个腕部相机 + 2 个外部相机
- **优化器**: 未明确说明（基于扩散模型标准训练流程）
- **Batch Size**: 32
- **训练时间**: ~2 天
- **硬件**: 8 × H100 GPU
- **关键超参数**: $\lambda_{\min} = 0.1$，$H = 0.3 \times T_{\max}$
- **腕部视角损失加权**: 对腕部视角预测的损失上调权重，强化腕部一致性学习

### 评估任务（5 个）

**短时任务（2 个）**:
- Pick cube to the plate（拾取方块到盘子）
- Wipe table（擦桌子）

**长时任务（3 个）**:
- Pick 1 object to 2 targets（拾取 1 个物体放到 2 个目标位置）
- Pick 2 objects to 1 target（拾取 2 个物体放到 1 个目标位置）
- Pick 2 objects to 2 targets（拾取 2 个物体分别放到 2 个目标位置）

### 评估指标

- **帧级别**: [[PSNR]]（峰值信噪比）、[[SSIM]]（结构相似度）、[[LPIPS]]（感知图像质量）
- **物体级别**: [[DINOv2]] 特征相似度（在 [[SAM]] 分割的物体区域上计算），即论文中的 Obj. Con.
- **策略评估**: 世界模型预测成功率与真实世界成功率之间的 [[Pearson Correlation|Pearson 相关系数]]

---

## 批判性思考

### 优点

1. **几何感知设计精巧**: 将 surfel 的法向量、深度、操作对象标志和时间戳统一纳入打分函数，体现了对机器人操作场景特性的深入理解。
2. **腕部视角更新的洞见**: 仅用腕部视角更新 surfel 而非用广角第三视角，保留了时间关联信息的纯粹性，是一个被实验充分验证的重要设计决策。
3. **与策略评估闭环**: 将世界模型与 Pearson 相关系数评估结合，直接量化了世界模型作为策略筛选工具的实用价值。

### 局限性

1. **初始可见性依赖**: 推演质量依赖于初始腕部视角的场景可见性；若初始帧遮挡严重，surfel 初始化不完整会影响后续所有检索。
2. **无显式物理约束**: 系统缺乏对物理接触的显式建模（如夹爪与物体是否真正接触），仅依赖视觉一致性无法保证物理合理性。
3. **Surfel 更新仅用腕部视角**: 虽然保持了时间关联纯粹性，但也意味着腕部未观测到的区域（如场景背景）在长时任务中可能得不到及时更新。

### 潜在改进方向

1. **多视角 surfel 更新**: 设计分离的 surfel 层（被操作物体层 vs 背景层），前者仅用腕部更新，后者允许第三视角更新，兼顾两者优势。
2. **显式接触状态建模**: 在 surfel 属性中加入接触状态标志，为世界模型预测提供更强的物理先验。

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节相对完整（超参数、硬件均有说明）
- [x] 数据集可获取（DROID 数据集公开）

---

## 关联笔记

### 基于

- [[Ctrl-World]]: 本文的 Backbone 和主要对比基线，Mem-World 在其多视角动作条件架构上添加记忆模块
- [[VMem]]: 提出 surfel 索引记忆的前驱工作，Mem-World 将其扩展为具有时间感知的 4D 版本
- [[π0.5]]: 用于策略评估和后训练的 VLA 策略模型

### 对比

- [[Ctrl-World]]: 固定步长历史帧条件，缺乏几何感知，腕部视角一致性差
- [[Cosmos Predict 2.5]]: 仅有第三视角评估，无腕部视角专项处理

### 方法相关

- [[Surfel Map]]: W-VMem 的核心数据结构，4D surfel 索引
- [[Action-Conditioned Video Generation]]: 本文世界模型的基础范式
- [[Forward Kinematics]]: 将关节动作转换为相机位姿的关键工具
- [[Non-Maximum Suppression]]: 检索过程中防止重复采样的技术
- [[Video Diffusion Model]]: Backbone 所基于的生成模型框架
- [[DINOv2]]: 用于物体级一致性评估的特征提取器
- [[Pearson Correlation]]: 量化世界模型与真实世界策略性能相关性的指标

### 硬件/数据相关

- [[DROID]]: 训练数据集，95,599 条多场景机器人轨迹
- [[Franka Emika Panda]]: 实验使用的机械臂平台

---

## 速查卡片

> [!summary] Mem-World (arXiv 2026)
> - **核心**: 4D surfel 腕部视角记忆 + 几何感知检索，解决长时域世界模型场景遗忘问题
> - **方法**: W-VMem 打分函数结合几何可见性、操作相关性、时间近因性三项因子检索历史帧
> - **结果**: 腕部视角 PSNR +1.87dB，策略评估 Pearson 相关系数 0.85→0.97，长时任务成功率 58%→72%
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-18*
