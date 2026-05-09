---
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
method_name: "LeWM"
authors: [Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, jepa, embodied-ai, planning, self-supervised-learning, representation-learning]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2603.19312v2
created: 2026-05-09
---

# 论文笔记：LeWorldModel — Stable End-to-End JEPA from Pixels

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Brown University · Meta FAIR · Mila |
| 日期 | March 2026 |
| 项目主页 | https://le-wm.github.io |
| 对比基线 | [[DINO-WM]] · [[PLDM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.19312) / [HTML](https://arxiv.org/html/2603.19312v2) / [Code](https://github.com/lucas-maes/le-wm) |

---

## 一句话总结

> 用仅 **2 项 loss** 端到端从像素训练 [[JEPA|联合嵌入预测架构]] 世界模型，无 EMA/无预训练编码器，~15M 参数即可在单卡上击败 DINO-WM、PLDM 等基线，规划速度提升 **48×**。

---

## 核心贡献

1. **极简训练目标**：将 [[PLDM]] 的 7 项 loss、6 个待调超参缩减到 **1 个 prediction loss + 1 个 [[SIGReg]] 正则项**，仅剩 1 个有效超参 $\lambda$。
2. **稳定无塌缩**：不需要 [[EMA|exponential moving average]] target encoder、不需要 stop-gradient、不需要预训练 backbone，仍然避免 [[Representation Collapse|表示塌缩]]。
3. **端到端从像素**：编码器 + 预测器 联合训练，约 15M 参数，单 GPU 数小时收敛。
4. **快且强**：相比 [[DINO-WM]] 规划快 48×；在 [[Push-T]] 上比 PLDM 提升 18% 成功率；2D/3D 控制任务全面 competitive。
5. **物理可解释 latent**：t-SNE 显示 latent 编码物体位置等物理量；可用作 [[Violation-of-Expectation]] 检测物理违规事件。

---

## 问题背景

### 要解决的问题

如何**端到端、稳定地**从原始像素训练一个用于规划的 [[World Model|世界模型]]？要求 latent 既可预测、又包含足够物理结构以支撑下游 [[Model Predictive Control|MPC]] 规划。

### 现有方法的局限

- **像素重建型 world model**（如 Dreamer 系）必须重建像素，浪费容量在背景/纹理上。
- **基于预训练编码器**（如 [[DINO-WM]]）依赖外部 backbone，规划器在冻结特征上学习，分布不匹配且推理慢。
- **JEPA 风格端到端**（如 [[PLDM]]）需要堆叠 [[VICReg]]、协方差正则、EMA target、stop-gradient 等 6+ 个 loss/技巧，超参搜索代价巨大。
- 现有方法在规划开销上动辄秒级/帧，严重制约实时机器人。

### 本文的动机

作者主张：**避免塌缩的本质条件**是 latent 的边缘分布接近各向同性高斯。利用 [[Cramér-Wold 定理]]，匹配所有一维投影的 marginal 等价于匹配高维联合分布，因而只需用统计检验把若干随机投影"按到"标准正态即可——不需要任何 EMA 或非对称结构。

---

## 方法详解

### 模型架构

LeWM 是一个标准 [[JEPA]]：编码器 $\text{enc}_\theta$ 把观测 $o_t$ 映射到 latent $z_t = \text{enc}_\theta(o_t)$；预测器 $\text{pred}_\phi$ 在 latent 空间里基于动作前推 $\hat z_{t+1} = \text{pred}_\phi(z_t, a_t)$。

- **输入**：RGB 像素观测 $o_t \in \mathbb{R}^{H\times W\times 3}$ + 动作 $a_t$
- **编码器**：[[Vision Transformer|ViT-tiny]]，patch=14、12 层、3 heads、hidden=192，约 5M 参数；取 [CLS] token 接 1 层 MLP + [[Batch Normalization|BN]]
- **预测器**：6 层 Transformer，16 heads，dropout 0.1，约 10M 参数；动作通过 [[Adaptive Layer Normalization|AdaLN]] 注入；后接 projector
- **正则**：[[SIGReg]]（Sketched Isotropic Gaussian Regularizer）
- **总参数**：~15M（单卡训练）
- **输出**：未来 latent $\hat z_{t+1:t+H}$，供 [[Cross-Entropy Method|CEM]] 规划器消费

### 核心模块

#### 模块 1: SIGReg — 各向同性高斯正则

**设计动机**: 直接迫使 latent 的边缘分布服从 $\mathcal{N}(0, I)$，从根上抑制 [[Representation Collapse]]，替代 [[VICReg]] 的 variance / invariance / covariance 三项组合。

**具体实现**:
1. 采样 $M=1024$ 个随机方向 $u^{(m)} \sim \mathcal{U}(S^{d-1})$；
2. 对 batch 内每个 latent $z_i$ 计算一维投影 $h_i^{(m)} = \langle u^{(m)}, z_i\rangle$；
3. 对每条投影做 [[Epps-Pulley Test|Epps-Pulley]] 单变量正态性检验 $T(\cdot)$；
4. 把所有方向上的检验值平均，作为正则项。

由 [[Cramér-Wold 定理]]，所有一维投影都正态 $\Leftrightarrow$ 联合分布正态，因此一个标量损失就能"打住"整个 latent 几何。

#### 模块 2: 端到端预测目标

**设计动机**: 不引入 stop-gradient 也不引入 EMA target，让编码器和预测器一起被训。SIGReg 已经避免了 trivial 解（常数 latent），所以无需非对称化。

**具体实现**: latent 上做 MSE 即可。预测器 6 层 Transformer + AdaLN 把动作 $a_t$ 注入每层 LN 的 scale/shift。

#### 模块 3: 规划器（[[Cross-Entropy Method|CEM]] + [[Model Predictive Control|MPC]]）

- 每步采样 300 条动作序列、迭代 30 次精简高斯分布；
- 评估 cost $C(\hat z_H) = \|\hat z_H - z_g\|_2^2$，$z_g$ 是目标观测的 latent；
- 找到 $a^*_{1:H}$，执行前 $K$ 步后重规划。

---

## 关键公式

### 公式 1: [[JEPA|预测损失]]

$$
\mathcal{L}_{\text{pred}} = \big\| \hat z_{t+1} - z_{t+1} \big\|_2^2,\quad \hat z_{t+1} = \text{pred}_\phi(z_t, a_t),\; z_{t+1} = \text{enc}_\theta(o_{t+1})
$$

**含义**: 在 latent 空间里做下一步预测的 MSE，是 JEPA 的"主菜"。注意 target $z_{t+1}$ 由 **同一** encoder 产生（无 EMA target）。

**符号说明**:
- $o_t, a_t$: 时刻 $t$ 的像素观测与动作
- $z_t = \text{enc}_\theta(o_t)$: 编码器输出 latent
- $\hat z_{t+1}$: 预测器产生的下一步 latent

### 公式 2: [[SIGReg]] 正则项

$$
\mathrm{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^{M} T\big(h^{(m)}\big),\quad h_i^{(m)} = \langle u^{(m)}, z_i \rangle
$$

**含义**: 对 $M$ 个随机方向上的一维投影各做一次 Epps-Pulley 正态性检验，再取均值。最小化它等于把 latent 的所有 1D marginal 拉向标准正态。

**符号说明**:
- $Z = \{z_i\}_{i=1}^B$: 一个 batch 的 latent
- $u^{(m)} \sim \mathcal{U}(S^{d-1})$: 单位球面上随机方向（$M=1024$）
- $T(\cdot)$: Epps-Pulley 正态性检验统计量
- $B$: batch size

### 公式 3: 总训练目标

$$
\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathrm{SIGReg}(Z)
$$

**含义**: 全篇唯一的 loss，只有 **1 个** 待调超参 $\lambda$（默认 $0.1$）。论文证明 $\lambda$ 在 $[10^{-2}, 1]$ 内 robust，可对数尺度搜一次即可。

**符号说明**:
- $\lambda$: 正则强度，唯一有效超参
- 不再需要 variance / covariance / invariance 三项的相对权重

### 公式 4: 规划 cost（终端目标匹配）

$$
C(\hat z_H) = \big\| \hat z_H - z_g \big\|_2^2,\quad z_g = \text{enc}_\theta(o_g)
$$

**含义**: latent 空间里对目标观测的距离作为代价；不需要解码到像素。

**符号说明**:
- $H$: 规划 horizon
- $o_g$: 目标观测，$z_g$: 其 latent
- $\hat z_H$: 由预测器 $H$ 步 rollout 得到

### 公式 5: 有限时域最优控制

$$
a^*_{1:H} \;=\; \arg\min_{a_{1:H}}\; C\!\left(\hat z_H\right) \quad \text{s.t.}\; \hat z_{t+1} = \text{pred}_\phi(\hat z_t, a_t),\; \hat z_0 = \text{enc}_\theta(o_0)
$$

**含义**: 形式化的开环最优控制问题，由 [[Cross-Entropy Method|CEM]] 数值求解，外层套 [[Model Predictive Control|MPC]] 重规划。

**符号说明**:
- $a^*_{1:H}$: 最优动作序列
- 约束：rollout 由学到的 latent 动力学确定

---

## 关键图表

### Figure 1: Training Pipeline

![Figure 1](https://arxiv.org/html/2603.19312v2/x1.png)

**说明**: LeWM 训练流水线。$o_t$ 与 $o_{t+1}$ 共用一个 encoder（无 EMA），predictor 用 [[AdaLN]] 接 $a_t$，输出 $\hat z_{t+1}$ 与 target $z_{t+1}$ 比 MSE，同时 batch 上的 $Z$ 进 [[SIGReg]] 拉向各向同性高斯。

### Figure 2: 三类世界模型对比

![Figure 2](https://arxiv.org/html/2603.19312v2/x2.png)

**说明**: End-to-end（本文）vs. foundation-based（DINO-WM）vs. task-specific（PLDM）的特性矩阵。LeWM 同时拿到"端到端可训 / 单超参 / 快规划 / 小参数"四个勾。

### Figure 3: 规划时间 & 性能对比（PushT / Reacher / Cube）

![Figure 3a](https://arxiv.org/html/2603.19312v2/x3.png)
![Figure 3b](https://arxiv.org/html/2603.19312v2/x4.png)
![Figure 3c](https://arxiv.org/html/2603.19312v2/x5.png)

**说明**: 横轴规划时间，纵轴成功率。LeWM 处于 Pareto 前沿，比 [[DINO-WM]] 快约 48×、比 [[PLDM]] 在 PushT 高 18%。

### Figure 4: Latent Planning 流程图

![Figure 4](https://arxiv.org/html/2603.19312v2/x6.png)

**说明**: 从 $o_0$ 编码出 $\hat z_0$，CEM 在 latent 空间里前推 $H$ 步得到 $\hat z_H$，与 $z_g$ 求距离作为 cost，迭代优化动作分布。

### Figure 5: LeWM 训练伪代码

```python
# 伪代码摘要（论文 Fig. 5）
for (o_t, a_t, o_{t+1}) in loader:
    z_t      = enc(o_t)
    z_tp1    = enc(o_{t+1})
    z_hat    = pred(z_t, a_t)
    L_pred   = mse(z_hat, z_tp1)
    L_sig    = sigreg(stack([z_t, z_tp1]), M=1024)
    loss     = L_pred + lam * L_sig
    loss.backward(); opt.step()
```

**说明**: 整套训练循环就这几行，没有 EMA、没有 stop-gradient、没有多分支。

### Figure 6: 四个环境上的规划性能

![Figure 6a](https://arxiv.org/html/2603.19312v2/x8.png)
![Figure 6b](https://arxiv.org/html/2603.19312v2/x9.png)
![Figure 6c](https://arxiv.org/html/2603.19312v2/x10.png)
![Figure 6d](https://arxiv.org/html/2603.19312v2/x11.png)

**说明**: Two-Room、Reacher、PushT、OGBench-Cube 四个环境上 LeWM 全面 competitive 或领先，包括需要 3D 推理的 Cube。

### Figure 7: Predictor Rollouts 可视化

![Figure 7a](https://arxiv.org/html/2603.19312v2/x12.png)
![Figure 7b](https://arxiv.org/html/2603.19312v2/x13.png)

**说明**: 在 PushT 与 OGBench-Cube 上长 horizon rollout 仍可被解码出可识别的物体配置，说明 latent 动力学没有发散。

### Figure 8: Decoder 训练进程

![Figure 8](https://arxiv.org/html/2603.19312v2/x14.png)

**说明**: 用一个轻量解码器探针看 latent 表达力如何随训练演变；训练越久细节越清晰。

### Figure 9: PushT latent 的 t-SNE

![Figure 9](https://arxiv.org/html/2603.19312v2/x15.png)

**说明**: latent 沿物理量（agent 位置 / block 位置）形成连续流形，证实 SIGReg 没有把结构压扁。

### Figure 10: Violation-of-Expectation

![Figure 10a](https://arxiv.org/html/2603.19312v2/x16.png)
![Figure 10b](https://arxiv.org/html/2603.19312v2/x17.png)
![Figure 10c](https://arxiv.org/html/2603.19312v2/x18.png)

**说明**: 三个环境上注入"违反物理"的事件（物体瞬移等），LeWM 的预测误差显著抬升，可作为 [[Violation-of-Expectation|VoE]] 异常检测器。

### Table 1: PushT 上的 latent 物理探针

| 物理量 | 模型 | Linear MSE ↓ | Linear $r$ ↑ | MLP MSE ↓ | MLP $r$ ↑ |
|--------|------|--------------|--------------|-----------|-----------|
| Agent Loc. | DINO-WM | 1.888 ± 0.500 | **0.977** | **0.003 ± 0.022** | **0.999** |
| Agent Loc. | PLDM    | 0.090 ± 0.311 | 0.955 | 0.014 ± 0.119 | 0.993 |
| Agent Loc. | **LeWM** | **0.052 ± 0.149** | 0.974 | 0.004 ± 0.056 | **0.998** |
| Block Loc. | DINO-WM | **0.006 ± 0.007** | **0.997** | 0.002 ± 0.003 | **0.999** |
| Block Loc. | PLDM    | 0.122 ± 0.341 | 0.938 | 0.011 ± 0.066 | 0.994 |
| Block Loc. | **LeWM** | 0.029 ± 0.073 | 0.986 | **0.001 ± 0.006** | **0.999** |

**说明**: 用线性 / MLP 探针从 latent 回归物理坐标。LeWM 在 MLP 探针上几乎全胜，线性探针上也与 DINO-WM 接近，**它的 latent 已经把物理状态学进去了**。

### Table 2: 消融（论文附录综合）

| 配置 | 影响 | 结论 |
|------|------|------|
| $M$ (SIGReg 投影数) 64 → 4096 | 性能基本平坦 | 1024 足够 |
| Embedding dim 64 → 512 | 超过阈值后饱和 | 192 已可 |
| Encoder ViT-tiny → ResNet-18 | 略降但 competitive | 架构无关 |
| 去掉 SIGReg | latent 塌缩、loss 趋零 | 必需 |
| 去掉 $\mathcal{L}_{\text{pred}}$ | 不学动力学 | 必需 |
| EMA target on/off | 几乎无差 | 不需要 |

**关键发现**: SIGReg 和 prediction loss 是仅有的两个真必需项，其余设计选择对结果不敏感。

---

## 实验

### 数据集 / 环境

| 环境 | 维度 | 特点 | 用途 |
|------|------|------|------|
| Two-Room | 2D | 离散障碍、navigation | 规划 |
| Reacher | 2D | 连续控制、reaching | 规划 |
| Push-T | 2D | 物体推动、接触动力学 | 规划 + latent 探针 |
| OGBench-Cube | 3D | 操作魔方 | 3D 规划 |

### 实现细节

- **Backbone**: ViT-tiny（patch 14, depth 12, hidden 192, heads 3）
- **Predictor**: 6 层 Transformer，16 heads，AdaLN，dropout 0.1
- **总参数**: ~15M
- **正则**: SIGReg, $M=1024$ 随机方向，Epps-Pulley 检验
- **Loss 权重**: $\lambda = 0.1$
- **规划器**: CEM, 300 samples × 30 iters, MPC 重规划
- **硬件**: 单卡 GPU，训练数小时
- **优化器/lr/batch**: Adam 系，详见论文附录 D

### 可视化结果

- 长 horizon rollout 通过解码探针仍清晰；
- t-SNE latent 沿物理量分流形；
- VoE：异常事件下 prediction loss 显著上升，可作为 anomaly score。

---

## 批判性思考

### 优点
1. **极简而稳**：把 JEPA 的工程"黑魔法"（EMA、stop-grad、多 loss 调权）一举抽走，仍然不塌缩。
2. **理论清晰**：基于 Cramér-Wold 把"分布匹配"降到一维投影，给"为什么不需要协方差正则"一个原理性解释。
3. **规划够快**：48× 提速 + 单卡 15M 参数，对真机机器人友好。
4. **latent 可解释**：物理探针 + VoE 直接验证 latent 不是垃圾。

### 局限性
1. **任务尺度有限**：仍以 2D + 简单 3D 为主，没有上 RoboCasa / RT-2 级别的真实机器人长任务。
2. **CEM 规划上限**：终端目标匹配在长 horizon 多目标任务上可能饱和；和 [[Diffusion Policy]] / 大规模 [[VLA]] 对比缺位。
3. **SIGReg 计算**：$M=1024$ 投影 + Epps-Pulley 在大 batch 下不便宜，论文未报每步开销分解。
4. **目标观测假设**：规划假设有 $o_g$，对 language-conditioned 任务需要再嫁接一层。

### 潜在改进方向
1. 把目标从图像泛化为语言/草图，对接 [[VLA]] 范式。
2. 用 [[Diffusion Model|扩散]] 规划器替换 CEM，做长 horizon 多模态动作分布。
3. 在真实机械臂（如 [[Push-T]] 实机或厨房任务）上做 transfer，验证 sim-to-real。
4. 把 SIGReg 推广到时序 latent 流（一阶差分也各向同性）以稳化更长 rollout。

### 可复现性评估
- [x] 代码开源（论文承诺）
- [ ] 预训练模型
- [x] 训练细节完整（附录 D）
- [x] 数据集可获取（公开 benchmark）

---

## 关联笔记

### 基于
- [[JEPA]]：本文是 JEPA 框架在像素世界模型上的最简实例化
- [[VICReg]]：被 SIGReg 取代的多项式正则
- [[I-JEPA]] / [[V-JEPA]]：LeCun 阵营的 JEPA 系列前作

### 对比
- [[DINO-WM]]：用预训练 DINO 做特征 + planner，慢且依赖 backbone
- [[PLDM]]：端到端但 7 项 loss、6 个超参，工程复杂
- [[Dreamer]] / [[DreamerV3]]：像素重建型 world model，浪费容量
- [[Diffusion Policy]]：直接学 policy，不显式建动力学

### 方法相关
- [[SIGReg]]：核心正则项
- [[Cramér-Wold 定理]]：理论依据
- [[Epps-Pulley Test]]：正态性检验
- [[Cross-Entropy Method]]：规划求解器
- [[Model Predictive Control]]：闭环执行
- [[AdaLN]]：动作注入方式
- [[Vision Transformer]]：编码器骨干
- [[Representation Collapse]]：要避免的问题

### 硬件/数据相关
- [[Push-T]]：2D 接触动力学 benchmark
- [[OGBench]]：3D 操作 benchmark

---

## 速查卡片

> [!summary] LeWorldModel
> - **核心**: 端到端 JEPA 世界模型，仅 MSE + SIGReg 两项 loss
> - **方法**: ViT-tiny encoder + 6 层 Transformer predictor + AdaLN，SIGReg 用 1024 随机投影 + Epps-Pulley 把 latent 拉向各向同性高斯
> - **结果**: 15M 参数、单卡数小时；PushT +18% over PLDM；规划比 DINO-WM 快 48×
> - **代码**: 论文承诺开源（见 arXiv 2603.19312）

---

*笔记创建时间: 2026-05-09*
