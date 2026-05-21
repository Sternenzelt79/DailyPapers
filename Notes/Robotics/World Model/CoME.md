---
title: "Composition of Memory Experts for Diffusion World Models"
method_name: "CoME"
authors: [Sebastian Stapf, Pablo Acuaviva Huertos, Aram Davtyan, Paolo Favaro]
year: 2026
venue: ICLR 2026
tags: [world-model, diffusion-model, memory, product-of-experts, navigation, test-time-finetuning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.18813
created: 2026-05-21
---

# 论文笔记：Composition of Memory Experts for Diffusion World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Bern, Computer Vision Group |
| 日期 | May 2026 |
| 项目主页 | [wiqzard.github.io/composition-of-memory-experts](https://wiqzard.github.io/composition-of-memory-experts/) |
| 对比基线 | [[状态空间模型]]、[[扩散世界模型]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.18813) / Code: N/A |

---

## 一句话总结

> 通过组合短期、长期和空间三类记忆专家（乘积专家框架 + 对比去基线），实现无二次复杂度的长时一致性扩散世界模型。

---

## 核心贡献

1. **记忆专家组合（CoME）**: 将世界模型的未来-过去一致性问题解耦，用三类专门化专家（短期/长期/空间长期）分别处理不同时间尺度的记忆，以[[乘积专家|Product of Experts]]框架统一组合
2. **对比乘积专家（PoCE）**: 引入对比基线权重，防止标准 PoE 中的模式崩塌问题，理论证明其在核密度估计下能重新加权混合成分而不缩小方差
3. **测试时 LoRA 微调长期记忆**: 利用[[LoRA]]将历史观测编码进模型权重，在 100-1000 帧上下文下无需额外 KV 缓存，配合空间先验（相机位姿+点图）抑制累积漂移

---

## 问题背景

### 要解决的问题

视频生成[[扩散世界模型|Diffusion World Model]]在长序列预测中存在严重的时间一致性问题：随着生成序列变长，模型对早期观测的记忆逐渐退化，导致场景重访时无法正确还原已见内容（物体、布局、颜色等）。

### 现有方法的局限

- **Transformer**：[[自注意力机制]]的二次复杂度限制了可用的上下文窗口，无法高效处理数百帧以上的历史
- **循环/状态空间模型（SSM）**：通过压缩历史信息来控制内存，但压缩过程不可避免地损失细节，导致长程一致性差
- **单一架构**：任何单一架构都面临"短期精度 vs 长期效率"的本质权衡

### 本文的动机

借鉴人类记忆系统的分工：海马体负责情节记忆，前额叶处理工作记忆。论文认为，**记忆应分布在多个专门化系统**而非单一架构，通过[[乘积专家]]框架在推理时灵活组合多个专家的得分函数。

---

## 方法详解

### 模型架构

CoME 采用**基于[[扩散Transformer|Diffusion Transformer]]的多专家组合**架构：

- **输入**: 当前帧观测 $x$，短期上下文 $c_{ST}$（10-100 帧），长期上下文 $c_{LT}$（100-1000 帧），空间地图 $S$
- **Backbone**: 共享基础扩散 Transformer（参数 $\theta$）
- **核心模块**: [[Product of Contrastive Experts|PoCE]] 用于组合四个专家的分数函数
- **输出**: 时间一致的未来帧生成 $x$
- **组合方式**: 推理时加权叠加各专家的噪声预测梯度

### 核心模块

#### 模块1：短期记忆专家（STM）

**设计动机**: 利用[[滑动窗口注意力]]捕捉局部动态，替代标准滑动窗口方法

**具体实现**:
- 在[[扩散Transformer]]中实现对最近 10-100 帧的直接[[自注意力机制|注意力]]
- 参数 $\phi$，条件输入 $c_{ST}$（短期上下文帧）
- 捕捉细粒度局部运动和场景变化

#### 模块2：长期记忆专家（LTM）

**设计动机**: 将情节历史编码进[[LoRA]]适配权重，实现 O(1) 复杂度的长程记忆访问

**具体实现**:
- 测试时对每段新历史运行 LoRA 微调，将历史信息写入低秩矩阵 $\Delta W$
- 参数 $\psi$，条件通过 LoRA 权重 $\psi(c_{LT})$ 隐式携带
- Rank 64（约占参数量 7%）达到近最优性能，避免全量微调的过拟合

#### 模块3：空间长期记忆专家（SLTM）

**设计动机**: 利用[[相机位姿估计|相机位姿]]和[[点图|点图（Point Map）]]提供几何约束，减少拓扑歧义场景中的漂移

**具体实现**:
- 编码器 $\text{Enc}_\lambda(M)$ 将空间地图（相机位姿 + 稠密点图）压缩为条件向量
- 参数 $\lambda$，条件输入 $S$（空间地图）
- 通过几何先验实现场景一致性的显式约束

---

## 关键公式

### 公式1：[[扩散模型]]训练目标

$$
\mathcal{L} = \mathbb{E}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]
$$

**含义**: 标准 DDPM 训练目标，学习在噪声图像 $x_t$ 上预测加入的噪声 $\varepsilon$

**符号说明**:
- $\varepsilon$: 真实加入的高斯噪声
- $\varepsilon_\theta$: 神经网络（扩散 Transformer）预测的噪声
- $x_t$: 第 $t$ 步加噪图像
- $t$: 扩散时间步

---

### 公式2：[[对比乘积专家|PoCE]] 单专家对比加权

$$
\tilde{p}_i(x) \propto p_i(x)^{\alpha_i} \cdot \bar{p}_i(x)^{1-\alpha_i}
$$

**含义**: 对每个专家引入无条件基线 $\bar{p}_i$，用对比系数 $\alpha_i > 1$ 放大条件分布、压制无条件部分，防止 PoE 合并时的模式坍缩

**符号说明**:
- $p_i(x)$: 第 $i$ 个专家的条件概率
- $\bar{p}_i(x)$: 第 $i$ 个专家的无条件基线概率
- $\alpha_i > 1$: 对比系数（contrast coefficient）

---

### 公式3：[[CoME|CoME 完整组合分布]]

$$
\begin{aligned}
p_{\text{CoME}}(x | c, c_{ST}, c_{LT}, S) \propto\; & \left[p_\theta(x|\emptyset)^{1-\alpha_0} \cdot p_\theta(x|c)^{\alpha_0}\right] \\
\times\; & \left[p_\phi(x|\emptyset)^{1-\alpha_1} \cdot p_\phi(x|c_{ST})^{\alpha_1}\right] \\
\times\; & \left[p_{\psi(\emptyset)}(x|c)^{1-\alpha_2} \cdot p_{\psi(c_{LT})}(x|c)^{\alpha_2}\right] \\
\times\; & \left[p_\lambda(x|\emptyset)^{1-\alpha_3} \cdot p_\lambda(x|S)^{\alpha_3}\right]
\end{aligned}
$$

**含义**: 将四个专家（基础、STM、LTM、SLTM）的对比加权分布相乘，得到综合记忆的生成分布

**符号说明**:
- $c$: 基础条件（最近帧）
- $c_{ST}$: 短期上下文条件
- $c_{LT}$: 长期上下文（隐式编码进 LoRA 权重）
- $S$: 空间地图条件
- $\alpha_0, \alpha_1, \alpha_2, \alpha_3$: 各专家的对比系数

---

### 公式4：[[复合分数函数|组合采样分数]]

$$
\nabla \log p_{\text{CoME}}(x_t) = \sum_k \left[\alpha_k \nabla \log p_k(x_t) + (1-\alpha_k) \nabla \log \bar{p}_k(x_t)\right]
$$

**含义**: 推理时将各专家的噪声预测梯度加权求和，实现无需重新训练的模块化组合采样

**符号说明**:
- $\nabla \log p_k(x_t)$: 第 $k$ 个条件专家的分数函数
- $\nabla \log \bar{p}_k(x_t)$: 第 $k$ 个无条件专家的分数函数
- $\alpha_k$: 第 $k$ 个专家的对比权重

---

### 公式5：[[LoRA]] 低秩适配

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)
$$

**含义**: 通过低秩矩阵分解 $BA$ 以最小参数量（rank $r$）将历史观测信息写入模型权重，实现测试时高效微调

**符号说明**:
- $W$: 预训练权重（冻结）
- $\Delta W = BA$: 低秩更新矩阵
- $r$: LoRA rank（实验取 64，约 7% 参数）

---

## 关键图表

### Figure 1: CoME 系统概览

![Figure 1 - Overview](https://arxiv.org/html/2605.18813v1/sections/image.png)

**说明**: CoME 的整体架构。输入当前观测 + 多时间尺度上下文，通过三类专家（STM/LTM/SLTM）分别处理，经[[乘积专家|Product of Experts]]框架组合后输出时间一致的未来帧。

---

### Figure 2: LPIPS 随上下文长度变化（Memory Maze）

![Figure 2 - Context Length](https://arxiv.org/html/2605.18813v1/sections/resub/context4.png)

**说明**: 在 Memory Maze 数据集上，LPIPS 随上下文帧数增加持续改善，直到 480 帧无饱和迹象，验证 CoME 对长程记忆的有效扩展性。

---

### Figure 3: 流式记忆评估（Memory Maze，11 个 chunk）

![Figure 3 - Streaming](https://arxiv.org/html/2605.18813v1/sections/resub/long_gt2.png)

**说明**: CoME 在连续 11 个 20 帧 chunk 上增量式记忆，时间一致性随 chunk 推进持续改善，展示 LTM 的在线记忆积累能力。

---

### Figure 4: RealEstate10K 定性结果

![Figure 4 - RealEstate10K](https://arxiv.org/html/2605.18813v1/x1.png)

**说明**: CoME 在真实室内场景中正确回忆初始帧的场景细节（颜色、物体布局），而无长期记忆的基线模型出现明显的视觉漂移。

---

### Figure 5: PoCE 玩具示例（高斯混合可视化）

![Figure 5 - Toy Example](https://arxiv.org/html/2605.18813v1/sections/resub/toy_example.png)

**说明**: 在高斯混合分布上可视化 PoCE 的作用：对比变换消除了各专家记忆中不存在的虚假共识模式，同时保留主导核的结构特性，避免标准 PoE 的方差收缩。

---

### Table 1: Memory Maze 消融实验（LPIPS↓ / SSIM↑ / PSNR↑）

| 方法 | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|--------|-------|-------|
| Base | 0.209 | 0.771 | 19.16 |
| + STM | 0.156 | 0.820 | 21.29 |
| + LTM | 0.171 | 0.805 | 19.98 |
| + SLTM | 0.150 | 0.833 | 20.65 |
| + STM+LTM | 0.114 | 0.862 | 22.32 |
| **CoME（全部）** | **0.097** | **0.892** | **23.07** |
| Sliding Window | 0.183 | 0.753 | 19.02 |
| SSM | 0.158 | 0.828 | 20.62 |
| Full-attention | 0.113 | 0.859 | 22.78 |

**关键发现**: CoME（全部专家）以远低于全注意力的计算代价超越所有基线，LPIPS 提升 53.6%（vs Base）。

---

### Table 2: 导航规划精度（RECON benchmark）

| 方法 | ATE↓ | RPE↓ |
|------|------|------|
| GNM | 1.87 | 0.73 |
| NOMAD | 1.93 | 0.52 |
| NWM | 1.13 | 0.35 |
| CoME (STM) | 1.05 | 0.32 |
| CoME (LTM) | 1.10 | 0.32 |
| NWM+STM | 0.98 | 0.30 |
| NWM+LTM | 1.07 | 0.33 |
| **CoME（全部）** | **0.96** | **0.28** |

**关键发现**: CoME（STM）在轨迹误差（ATE）上超越纯 NWM 约 15%，证明世界模型的时间一致性改善直接带来规划性能提升。

---

### Table 3: RealEstate10K 结果

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|-------|
| Base | 0.405 | 19.8 | 0.794 |
| HG-v | 0.414 | 19.2 | 0.764 |
| HG-t | 0.400 | 19.7 | 0.788 |
| **CoME** | **0.359** | **21.3** | **0.830** |

**关键发现**: 在真实室内场景数据集上，CoME 全面优于层级生成基线（HG），LPIPS 改善 11.4%。

---

### Table 4: 对比消融——有无 PoCE（LPIPS）

| 配置 | 无对比 | 有对比 |
|------|--------|--------|
| Base | 0.203 | 0.200 |
| STM | 0.175 | 0.156 |
| LTM | 0.188 | 0.171 |
| SLTM | 0.178 | 0.150 |
| STM+LTM | 0.170 | 0.114 |
| **All** | 0.192 | **0.097** |

**关键发现**: 不加对比基线时，All 专家组合（0.192）甚至差于单独 STM（0.175），说明朴素 PoE 发生模式坍缩；加入对比后性能大幅提升，验证 PoCE 的必要性。

---

### Table 5: LTM LoRA Rank 分析（LPIPS，不同上下文帧数）

| 上下文帧数 | Rank 8 | Rank 32 | Rank 64 | Rank 256 | Full FT |
|-----------|--------|---------|---------|----------|---------|
| 50  | 0.193 | 0.175 | 0.161 | 0.162 | 0.221 |
| 150 | 0.161 | 0.169 | 0.158 | 0.142 | 0.188 |
| 450 | 0.146 | 0.128 | 0.124 | 0.118 | 0.125 |

**关键发现**: Rank 64（约 7% 参数）在各上下文长度下均达到近最优，Full FT 反而因过拟合而性能下降。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Memory Maze | 30k 轨迹，每条 1000 帧 | 合成 3D 导航，含位置信号 | 主要评估 |
| RealEstate10K | 真实室内视频，256×256 | 相机位姿已知 | 泛化评估 |
| RECON | 5k+ 户外导航轨迹 | 绝对相机轨迹 | 导航规划 |
| DMLab-40K | 40k 序列，64×64 | 程序化迷宫视频 | 长程记忆 |
| Minecraft-200K | 200k 游戏视频，256×256 | 开放世界游戏场景 | 大规模评估 |
| Memory Cards | 自制 4×4 卡片网格 | 显示/隐藏物体，测试召回率 | 记忆召回专项 |

### 实现细节

- **Backbone**: 扩散 Transformer，基础模型共享权重
- **LTM**: 测试时 LoRA 微调，Rank 64，约 4× 采样开销
- **全注意力训练对比**: 需约 60× 计算量
- **评估指标**: LPIPS、SSIM、PSNR（感知质量）；ATE、RPE（导航精度）；物体召回率（记忆测试）

### 可视化结果

CoME 在 RealEstate10K 上正确回忆初始场景的颜色与布局；在 Memory Cards 上物体召回率达 **79.2%**，而标准扩散基线仅 **13%**，差距显著。

---

## 批判性思考

### 优点

1. **无二次复杂度**：通过 LoRA 测试时微调实现长程记忆，避免 Transformer 全注意力的 O(n²) 瓶颈
2. **模块化可组合**：各专家独立训练，推理时通过分数加权组合，无需联合训练即可扩展
3. **理论支撑**：Proposition 1 在 KDE 假设下证明 PoCE 避免方差缩小，toyexample 可视化直观验证

### 局限性

1. **在线记忆开销**：LoRA 测试时微调引入约 4× 采样延迟（Rank 64），实时应用受限
2. **LTM 无显式时序依赖**：当前 LTM 在 LoRA 权重中隐式编码历史，缺乏对时序顺序的建模，长序列中时序信息可能混叠
3. **空间条件限制**：SLTM 依赖已知相机位姿和点图，无法在位姿未知的场景直接应用

### 潜在改进方向

1. 在 LTM 中引入显式时序依赖建模（如时间戳编码或序列结构化 LoRA）
2. 探索更丰富的空间条件策略，放松对精确相机位姿的依赖
3. 将 LoRA 替换为更高效的参数更新方案（如前缀调优、适配器）以降低推理开销

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文附录中有超参数描述）
- [x] 数据集可获取（Memory Maze、RealEstate10K、RECON 均为公开数据集）

---

## 关联笔记

### 基于
- [[扩散Transformer|Diffusion Transformer]]：基础生成骨干网络
- [[乘积专家|Product of Experts]]：多专家组合的理论框架
- [[LoRA]]：长期记忆的测试时写入机制

### 对比
- [[GigaWorld]]：Transformer 全注意力世界模型，二次复杂度瓶颈
- [[PLDM]]：基于潜在扩散的预测型世界模型
- [[状态空间模型]]：压缩历史但损失细节的替代方案

### 方法相关
- [[Product of Contrastive Experts|PoCE]]：本文提出的核心创新
- [[扩散世界模型|Diffusion World Model]]：本文所属的研究范式
- [[相机位姿估计]]：SLTM 的空间条件来源

### 硬件/数据相关
- [[Memory Maze]]：主要评估数据集，合成 3D 导航环境
- [[RealEstate10K]]：真实室内场景泛化评估

---

## 速查卡片

> [!summary] CoME: Composition of Memory Experts for Diffusion World Models
> - **核心**: 三类记忆专家（STM/LTM/SLTM）通过对比乘积专家框架组合，解决扩散世界模型的长程一致性问题
> - **方法**: PoCE（对比加权 PoE）+ LoRA 测试时微调长期记忆 + 相机位姿空间先验
> - **结果**: Memory Maze LPIPS 0.097（vs Full-attention 0.113），物体召回率 79.2%（vs 基线 13%）
> - **代码**: 未开源（ICLR 2026）

---

*笔记创建时间: 2026-05-21*
