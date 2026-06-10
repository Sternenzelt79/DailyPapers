---
title: "TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation"
method_name: "TacForeSight"
authors: [Yujie Zang, Yuhang Zheng, Xian Nie, Yupeng Zheng, Shuai Tian, Songen Gu, Chen Gao, Zining Wang, Shuicheng Yan, Wenchao Ding]
year: 2026
venue: arXiv
tags: [tactile-sensing, world-model, contact-rich-manipulation, force-torque, robot-policy]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.11184
created: 2026-06-10
---

# 论文笔记：TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（多机构合作） |
| 日期 | June 2026 |
| 项目主页 | 待发布 |
| 对比基线 | [[RDP]], [[FoAR]], [[KineDex]], [[Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.11184) / Code（即将开源） |

---

## 一句话总结

> 通过显式建模腕部力矩信号对触觉动态的"预先感知"关系，TacForeSight 让机器人在接触丰富操作任务中实现前瞻性触觉推理，显著提升扰动鲁棒性。

---

## 核心贡献

1. **TacForceWM（力条件触觉世界模型）**: 用腕部力/力矩信号作为条件，在潜空间中预测未来触觉动态，利用力信号比触觉响应更早出现的时序先行关系
2. **预测触觉条件策略**: 将预测出的未来触觉潜码与当前触觉潜码通过[[Cross-Attention|交叉注意力]]交互，并使用自适应门控融合视觉与触觉特征
3. **轻量高效**: TacForceWM 仅 11.8M 参数，支持实时推理；在 5 类接触丰富任务上平均成功率 79%，扰动恢复平均达 86.7%

---

## 问题背景

### 要解决的问题

接触丰富操作（contact-rich manipulation）要求机器人持续感知和调节不断变化的物理交互。现有方法要么依赖视觉（感知局限），要么仅做多模态被动融合，无法在接触事件发生之前做出预判。

### 现有方法的局限

- **纯视觉策略**（如 [[Diffusion Policy]]）：无法捕捉精细接触信息
- **触觉直接融合方法**（如 RDP、FoAR）：将力/触觉作为额外观测输入，属于被动融合，无法预见未来接触变化
- **无世界模型**：缺乏对接触动力学的显式建模，遇到外部扰动时响应滞后

### 本文的动机

关键洞察：**腕部力矩信号比触觉传感器响应更早**（高频全局线索 vs. 精细局部形变），力变化是未来触觉状态的"先行指标"（leading indicators）。因此可以用力信号条件化触觉潜动力学预测，将被动融合升级为主动前瞻。

---

## 方法详解

### 模型架构

![Figure 2: TacForeSight Framework Overview](https://arxiv.org/html/2606.11184v1/x1.png)

**说明**: TacForeSight 整体框架，包含两个耦合组件：TacForceWM（上方）和预测触觉条件策略（下方）。

TacForeSight 采用**两阶段解耦**架构：
- **输入**: RGB 图像 + 双指[[触觉传感器]]观测 + 腕部六轴力/力矩 + 本体感知状态
- **Stage 1**: [[触觉世界模型|TacForceWM]] 预测短时域触觉潜码
- **Stage 2**: [[机器人策略|预测触觉条件策略]] 融合预测触觉、当前触觉、视觉、本体感知，输出动作序列
- **动作头**: [[Flow Matching|流匹配]] 动作预测头
- **总参数**: TacForceWM 11.8M + Policy 68.9M

---

### 核心模块

#### 模块一：TacForceWM（力条件触觉世界模型）

**设计动机**: 利用[[腕力传感器|腕部力矩]]的时序先行性，对未来触觉潜码进行前瞻预测。

**组成部分**:

1. **触觉分词器（Tactile Tokenizer）**
   - 混合 [[CNN-Transformer]] 架构
   - 将双指触觉图像场转换为紧凑的帧级 token
   - 每帧输出空间特征图 $\mathbf{F}_t^s \in \mathbb{R}^{H' \times W' \times D_h}$，加入位置编码和手指 ID 嵌入

2. **力编码器（Force Encoder）**
   - 时序[[Transformer]]编码器
   - 将高采样率腕部力/力矩序列压缩为触觉对齐的条件向量

3. **潜在动力学预测器（Latent Dynamics Predictor）**
   - 力条件化 [[Transformer]] 模型
   - 给定当前触觉潜码块和力条件，预测未来触觉潜码块

**训练目标**: 最小化预测潜码与真实潜码之间的 MSE，同时加入动力学一致性项和 VICReg 风格的自监督约束（防表示坍塌）

---

#### 模块二：预测触觉条件策略

**设计动机**: 让策略不仅看到"当前触觉状态"，还能感知"即将到来的触觉变化"，从而提前做出响应。

**感知模态提取**:
- **视觉**: [[DINOv2]] 提取 RGB 图像特征
- **本体感知**: [[MLP]] 处理关节状态
- **触觉**: 使用冻结的 TacForceWM 分词器提取当前 + 预测未来触觉潜码

**当前-未来触觉交互模块**:
- 使用[[Cross-Attention|交叉注意力]]让当前触觉 query 到未来触觉信息
- 加入时序位置编码区分当前帧与预测帧

**自适应视觉-触觉融合**:
- [[Gating Mechanism|自适应门控]]动态权衡视觉特征与触觉特征
- 门控权重由触觉特征条件化的 MLP 生成，经 sigmoid 归一化

**动作预测**:
- [[Flow Matching]] 动作头，从噪声分布流向真实动作分布
- 支持 ODE 求解器实时推理

---

## 关键公式

### 公式 1-2：[[触觉分词器|触觉特征提取]]

$$
\mathbf{F}_t^s = \Phi_{sp}(\mathbf{X}_t^s) \in \mathbb{R}^{H' \times W' \times D_h}
$$

$$
\tilde{\mathbf{F}}_t^s = \mathbf{F}_t^s + \mathbf{E}_{pos} + \mathbf{E}_{id}^s
$$

**含义**: 将第 $s$ 个手指在 $t$ 时刻的触觉图像 $\mathbf{X}_t^s$ 经空间编码器 $\Phi_{sp}$ 映射为特征图，加入位置嵌入 $\mathbf{E}_{pos}$ 和手指 ID 嵌入 $\mathbf{E}_{id}^s$ 区分左右指。

**符号说明**:
- $\mathbf{X}_t^s$: 第 $s$ 个手指在 $t$ 时刻的触觉图像
- $H', W'$: 特征图空间分辨率
- $D_h$: 隐层维度
- $\mathbf{E}_{pos}$: 空间位置编码
- $\mathbf{E}_{id}^s$: 手指身份编码（区分左/右）

---

### 公式 3：[[腕力传感器|力编码]]

$$
\mathbf{c}_{t-h:t} = G_\phi(\mathbf{w}_{t-n_h:t})
$$

**含义**: 力编码器 $G_\phi$ 将高频腕部力矩序列 $\mathbf{w}$ 压缩为时域条件向量 $\mathbf{c}$，用于条件化触觉潜动力学预测。

**符号说明**:
- $\mathbf{w}_{t-n_h:t}$: 从 $t-n_h$ 到 $t$ 时刻的腕部六轴力/力矩序列（高频采样）
- $h$: 触觉历史窗口长度
- $n_h$: 力信号采样窗口（通常比触觉窗口更长，因为采样率更高）
- $G_\phi$: 时序 Transformer 力编码器，参数 $\phi$

---

### 公式 4：[[触觉世界模型|潜在动力学预测]]

$$
\hat{\mathbf{z}}_{t-h+\Delta:t+\Delta}^{tac} = T_\psi(\mathbf{z}_{t-h:t}^{tac},\ \mathbf{c}_{t-h:t}^{tac})
$$

**含义**: 动力学预测器 $T_\psi$ 以当前触觉潜码块 $\mathbf{z}^{tac}$ 和力条件 $\mathbf{c}^{tac}$ 为输入，预测未来 $\Delta$ 步触觉潜码。

**符号说明**:
- $\mathbf{z}_{t-h:t}^{tac}$: 历史 $h$ 帧触觉潜码序列
- $\mathbf{c}_{t-h:t}^{tac}$: 力编码后的条件向量
- $\Delta$: 预测步长（前瞻视野）
- $T_\psi$: 潜在动力学预测 Transformer

---

### 公式 5-6：[[触觉世界模型|世界模型损失]]

$$
\mathcal{L}_{pred} = \text{MSE}(\hat{\mathbf{Z}}_t^{tac},\ \mathbf{Z}_t^{tac}) + \lambda_{dyn} \cdot \text{MSE}(\nabla \hat{\mathbf{Z}}_t^{tac},\ \nabla \mathbf{Z}_t^{tac})
$$

$$
\mathcal{L}_{WM} = \mathcal{L}_{pred} + \lambda_{sig} \cdot \mathcal{L}_{sig}
$$

**含义**: 世界模型总损失由预测损失和自监督稳定性损失组成；预测损失包含绝对值 MSE 和动力学一致性（一阶差分 MSE），后者强制预测序列的变化趋势与真实趋势一致。

**符号说明**:
- $\hat{\mathbf{Z}}_t^{tac}$: 预测触觉潜码序列
- $\mathbf{Z}_t^{tac}$: 真实触觉潜码序列
- $\nabla$: 时序一阶差分算子（动力学一致性）
- $\lambda_{dyn} = 1.00$: 动力学一致性权重
- $\mathcal{L}_{sig}$: VICReg 风格自监督损失（防表示坍塌）
- $\lambda_{sig} = 0.09$: 自监督损失权重

---

### 公式 7-8：[[Cross-Attention|当前-未来触觉交叉注意力]]

$$
\bar{\mathbf{Z}}_{t,cur}^{tac} = \mathbf{z}_{t-h:t}^{tac} + \mathbf{E}_{temp}
$$

$$
\mathbf{H}_t^{tac} = \bar{\mathbf{Z}}_{cur}^{tac} + \text{CA}(Q=\bar{\mathbf{Z}}_{t,cur}^{tac},\ K,V=\bar{\mathbf{Z}}_{t,fut}^{tac})
$$

**含义**: 当前触觉潜码（加时序编码后）作为 Query，未来预测触觉潜码作为 Key/Value，通过交叉注意力将前瞻信息注入当前表示，得到融合了未来感知的触觉特征 $\mathbf{H}_t^{tac}$。

**符号说明**:
- $\mathbf{E}_{temp}$: 时序位置编码（区分当前帧与预测帧）
- $\text{CA}(\cdot)$: 交叉注意力操作
- $\bar{\mathbf{Z}}_{t,fut}^{tac}$: 预测的未来触觉潜码（加时序编码后）

---

### 公式 9-10：[[Gating Mechanism|自适应视觉-触觉融合]]

$$
\boldsymbol{\alpha} = \sigma(\text{MLP}(\mathbf{h}_t^{tac}))
$$

$$
\mathbf{h}_t^{vt} = (1 - \boldsymbol{\alpha}) \odot \mathbf{h}_t^{img} + \boldsymbol{\alpha} \odot \mathbf{h}_t^{tac}
$$

**含义**: 基于触觉特征动态生成融合权重 $\boldsymbol{\alpha}$，接触强度大时触觉主导，无接触时视觉主导，实现条件自适应的多模态融合。

**符号说明**:
- $\sigma(\cdot)$: Sigmoid 激活函数
- $\boldsymbol{\alpha} \in (0,1)^d$: 元素级门控权重
- $\mathbf{h}_t^{img}$: DINOv2 视觉特征
- $\mathbf{h}_t^{tac}$: 融合了未来感知的触觉特征
- $\odot$: 逐元素乘法

---

### 公式 11-13：[[Flow Matching|流匹配动作预测]]

$$
\mathbf{A}_t^{(\tau)} = (1-\tau)\mathbf{A}_t^{(0)} + \tau \mathbf{A}_t^{(1)}, \quad \mathbf{u}_t = \mathbf{A}_t^{(1)} - \mathbf{A}_t^{(0)}
$$

$$
\mathcal{L}_{FM} = \mathbb{E}\left[\left\|v_\theta(\mathbf{A}_t^{(\tau)}, \tau, \mathbf{y}_t) - \mathbf{u}_t\right\|_2^2\right]
$$

$$
\frac{d\mathbf{A}_t^{(\tau)}}{d\tau} = v_\theta(\mathbf{A}_t^{(\tau)}, \tau, \mathbf{y}_t)
$$

**含义**: 采用条件流匹配训练动作头；训练时学习从噪声 $\mathbf{A}^{(0)}$ 到真实动作 $\mathbf{A}^{(1)}$ 的速度场 $v_\theta$，推理时通过 ODE 求解器从噪声积分到动作。

**符号说明**:
- $\tau \in [0,1]$: 流匹配时间参数
- $\mathbf{A}_t^{(0)}$: 噪声动作（从标准分布采样）
- $\mathbf{A}_t^{(1)}$: 真实示教动作
- $\mathbf{u}_t$: 目标速度场（直线插值的切线向量）
- $v_\theta$: 学习的速度场网络
- $\mathbf{y}_t$: 多模态条件特征（视觉+触觉+本体感知）

---

## 关键图表

### Figure 1：任务示意（Teaser）

**说明**: 灯泡插入与锁定任务示意，展示力信号（蓝色曲线）相对于触觉响应（橙色曲线）的时序先行关系，以及 TacForeSight 对扰动的预判能力。

---

### Figure 2：TacForeSight 整体框架

![Figure 2: Framework Overview](https://arxiv.org/html/2606.11184v1/x1.png)

**说明**: 上半部分为 TacForceWM，展示触觉分词器、力编码器和潜在动力学预测器的连接方式；下半部分为策略网络，包含多模态编码、交叉注意力触觉交互、自适应融合和流匹配动作头。

---

### Figure 3：五类接触丰富操作任务

![Figure 3: Contact-Rich Manipulation Tasks](https://arxiv.org/html/2606.11184v1/x2.png)

**说明**: 五类任务展示：花瓶擦拭（Wiping）、卡片刷卡（Swiping）、管道调整（Adjustment）、灯泡插入与锁定（Locking）、导线插入（Insertion），以及带扰动变体（-P 后缀）。

---

### Figure 4：触觉潜码表示分析

![Figure 4: Tactile Latent Representation Analysis](https://arxiv.org/html/2606.11184v1/x3.png)

**说明**: (a) 在灯泡插入和花瓶擦拭任务中，触觉潜码的时序演化可视化，清晰呈现不同接触阶段的潜码变化轨迹；(b) 不同交互原语下触觉潜码嵌入的 t-SNE 聚类，各原语形成清晰簇，验证表示的区分性。

---

### Figure 5：扰动恢复中的触觉门控可视化

![Figure 5: Tactile Gating Visualization](https://arxiv.org/html/2606.11184v1/x4.png)

**说明**: 花瓶擦拭扰动任务中，触觉门控权重 $\alpha$ 的动态变化；扰动发生时 $\alpha$ 升高（触觉主导），恢复后下降（视觉主导），验证自适应融合的合理性。

---

### Table I：主实验结果（成功率 %）

| 方法 | Wiping | Swiping | Adjustment | Locking | Insertion | Wiping-P | Swiping-P | Adjustment-P |
|------|--------|---------|------------|---------|-----------|----------|-----------|--------------|
| KineDex | 30% | 35% | 25% | 45% | 30% | — | — | — |
| FoAR | 50% | 50% | 35% | 25% | 20% | — | — | — |
| DP+Tactile | 80% | 40% | 35% | 30% | 15% | — | — | — |
| RDP | 85% | 50% | 25% | 55% | 0% | — | — | — |
| **TacForeSight** | **100%** | **85%** | **70%** | **80%** | **60%** | **90%** | **85%** | **85%** |

**关键发现**: TacForeSight 在所有任务上均显著超越基线，尤其在 Wire Insertion（0% vs. 60%）上优势最大；扰动测试平均达 86.7%，验证预测性触觉感知对扰动鲁棒性的重要作用。

---

### Table II：世界模型条件化消融（预测质量）

| 条件信号 | MSE ↓ | Cosine ↑ | KL_sym ↓ |
|----------|-------|----------|----------|
| 腕部力矩（本文） | **0.017** | **0.992** | **0.009** |
| 机器人状态 | 0.022 | 0.975 | 0.011 |
| 无条件 | 0.027 | 0.954 | 0.014 |

**关键发现**: 腕部力矩条件在三项指标上均最优，验证力信号对触觉动力学预测的独特价值（不可替换为机器人关节状态）。

---

### Table III：策略组件消融（成功率 %）

| 配置 | Wiping | Swiping-P | Adjustment | 说明 |
|------|--------|-----------|------------|------|
| **完整模型** | **100%** | **85%** | **70%** | — |
| w/o 交叉注意力 | 100% | 0% | — | 失去扰动恢复能力 |
| w/o 力条件 | 70% | — | — | 基础性能下降 |
| w/o 预测触觉 | 65% | — | — | 仅用当前触觉 |

**关键发现**: 交叉注意力模块对扰动恢复至关重要（Swiping-P: 85% → 0%）；力条件化和预测触觉各自都对基础性能有贡献。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 五任务演示数据 | 2,700 episodes | 双指触觉 + 六轴力矩 + RGB | TacForceWM 训练 |
| 任务策略数据 | 与上同 | 多模态同步采集 | Policy 训练 |

### 实现细节

- **触觉传感器**: 双 Xense 光学触觉传感器（30 Hz）
- **力传感器**: 六轴腕力矩传感器（120 Hz）
- **视觉**: Intel RealSense D435 RGB-D（30 Hz）
- **机械臂**: 7-DoF UFactory xArm7 + Robotiq 夹爪
- **视觉骨干**: [[DINOv2]]（冻结）
- **优化器**: Adam
- **TacForceWM 训练**: 150k steps，2,700 episodes
- **Policy 训练**: 8 × NVIDIA A100 GPU
- **超参数**: $\lambda_{dyn}=1.00$，$\lambda_{sig}=0.09$

### 可视化结果

- 触觉潜码 t-SNE 图（Figure 4b）清楚显示不同接触阶段的聚类分离，说明潜空间捕获了有意义的接触语义
- 触觉门控可视化（Figure 5）定性验证自适应融合模块在扰动事件前后的行为符合预期

---

## 批判性思考

### 优点
1. **时序先行洞察深刻**: 力-触觉时序关系的发现是真正的新颖贡献，而非工程堆叠
2. **轻量设计**: TacForceWM 仅 11.8M 参数，实时推理友好，工程实用价值高
3. **全面消融**: 对各模块均有细粒度消融，有说服力

### 局限性
1. **硬件专用性强**: 依赖双指 Xense 触觉传感器 + 六轴力矩传感器，硬件迁移成本高
2. **任务多样性受限**: 全部为桌面接触操作，尚未验证灵巧手或全身操作场景
3. **力-触觉先行假设**: 文中未量化力信号领先触觉响应的具体时间差，可能因任务而异

### 潜在改进方向
1. 将力-触觉预测扩展到更长预测视野，结合规划做更远期的接触策略
2. 探索跨传感器迁移，如从 Xense 迁移到 GelSight/GelSlim 的零样本适应

### 可复现性评估
- [ ] 代码开源（承诺将发布）
- [ ] 预训练模型（承诺将发布）
- [x] 训练细节完整（超参数、硬件均有记录）
- [ ] 数据集可获取（承诺将发布）

---

## 关联笔记

### 基于
- [[Diffusion Policy]]: 扩散策略框架，TacForeSight Policy 的基础参考
- [[DINOv2]]: 视觉特征提取骨干（冻结使用）
- [[Flow Matching]]: 动作预测头采用流匹配代替扩散

### 对比
- [[RDP]]: Reactive Diffusion Policy，触觉+视觉被动融合基线
- [[FoAR]]: 触觉-视觉融合策略基线
- [[KineDex]]: 基于关节动力学的灵巧操作基线

### 方法相关
- [[触觉世界模型]]: 本文核心方法
- [[Cross-Attention]]: 当前-未来触觉交互的实现机制
- [[Gating Mechanism]]: 自适应视觉-触觉融合的核心组件
- [[Flow Matching]]: 动作头设计

### 硬件/数据相关
- [[触觉传感器]]: Xense 光学触觉传感器
- [[腕力传感器]]: 六轴力/力矩传感器
- [[Force-Torque Sensing]]: 力矩传感相关概念

---

## 速查卡片

> [!summary] TacForeSight
> - **核心**: 用腕部力矩信号预测未来触觉潜码，将被动触觉融合升级为主动前瞻式接触推理
> - **方法**: TacForceWM（力条件触觉世界模型）+ 预测触觉条件策略（交叉注意力 + 自适应门控 + 流匹配）
> - **结果**: 5 任务平均 79% 成功率，扰动恢复平均 86.7%，显著超越所有基线
> - **代码**: 即将开源（arXiv 2606.11184）

---

*笔记创建时间: 2026-06-10*
