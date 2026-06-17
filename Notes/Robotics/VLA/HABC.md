---
title: "Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes"
method_name: "HABC"
authors: [Tongyan Fang, Siyuan Huang, Naiyu Fang, Ganlong Zhao, Zhongjin Luo, Jianbo Liu, Xiaogang Wang, Ying Dong, Hongsheng Li]
year: 2026
venue: arXiv
tags: [reinforcement-learning, vla, policy-learning, advantage-weighting, online-finetuning, robot-manipulation, bimanual]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17043v1
created: 2026-06-16
---

# 论文笔记：Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ACE Robotics, Tsinghua University, The Chinese University of Hong Kong |
| 日期 | June 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[π0.5]], [[DAgger]], [[Recap]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17043) |

---

## 一句话总结

> 提出 HABC，将稀疏二值任务结果分解为**存活性**与**效率**双目标，通过双头 Critic 和状态自适应门控实现 VLA 在线 RL 微调，将三项双臂任务成功率提升 2-3 倍。

---

## 核心贡献

1. **信号分解（Signal Decomposition）**: 将单一二值结果 $y \in \{0,1\}$ 分解为两个独立目标——**存活性**（成功概率 $p_v$）和**效率**（距成功步数 $\hat{V}_e$），避免混淆不同反馈类型。
2. **层次优势加权（Hierarchical Advantage Weighting）**: 双头 Critic 分别估计两个信号的优势值 $A_v, A_e$，通过状态自适应门控 $g_t$ 动态合并，软加权取代硬阈值过滤。
3. **干预感知信用分配（Intervention-Aware Credit Assignment）**: 严格区分自主执行片段与人工干预窗口的标签归属，防止 label leakage，允许在高失败率任务中安全利用人类干预数据。

---

## 问题背景

### 要解决的问题

在线 [[强化学习|RL]] 微调 [[VLA（Vision-Language-Action Model）|VLA]] 时，每个 rollout episode 仅产生一个二值结果（成功/失败），而 actor 更新需要**逐 transition 的监督信号**。如何从稀疏的 episode 结果中合理分配每一步的权重？

### 现有方法的局限

- **单一奖励信号**：现有方法将稀疏结果压缩为单一奖励，**混淆了存活性（能否成功）和效率（多快成功）**两类不同的反馈，导致接近饱和时无法继续优化。
- **硬阈值过滤**：如 Recap 等方法采用固定阈值丢弃"差"的 transition，但会损失有价值的负向学习信号。
- **干预数据利用不当**：人工干预帮助机器人完成任务时，若不加区分地将结果标签分配给干预窗口，会引入 label leakage，扭曲策略学习。

### 本文的动机

对于长周期接触丰富任务，"成功"本身就涉及多个阶段挑战——先要确保**能够成功**（存活性），再优化**如何更快成功**（效率）。将两者解耦并分层融合，在不同任务阶段动态倾斜权重，可以使训练信号在全程保持信息量。

---

## 方法详解

### 模型架构

HABC（Hierarchical Advantage-Weighted Behavior Cloning）在 [[π0.5]] 基础上添加**双头 Critic** 模块：

- **输入**: 语言指令 $\ell$ + 图像观测 $I_t$ + 本体感知 $q_t$，状态表示 $s_t = (I_t, q_t, \ell)$
- **Backbone**: [[π0.5]] 的视觉-语言主干 $\phi(\cdot)$（冻结共享特征提取器）
- **核心模块**:
  - [[双头Critic|存活性头]] $f_v$：预测当前状态下成功的概率
  - [[双头Critic|效率头]] $f_e$：预测距成功所需步数（仅在成功轨迹上训练）
  - [[状态自适应门控]] $g_t$：根据 $p_v$ 动态混合两类优势信号
- **输出**: 经过 $g_t$ 加权的 [[Action Chunking|动作块]] 更新权重 $w_i$
- **Actor**: [[Flow Matching|流匹配]] 策略，使用加权 BC 损失微调

### 核心模块

#### 模块1：双头 Critic（Dual-Head Critic）

**设计动机**: 分别建模 [[存活性（Viability）]] 与 [[效率（Efficiency）]]，针对不同数据子集训练，避免混淆。

**具体实现**:
- **存活性头**：$z_v(s) = f_v(\phi(s))$，所有有标注的策略执行窗口（含成功与失败）上用 [[二元交叉熵]] 训练，目标 $y \in \{0,1\}$
- **效率头**：$\hat{V}_e(s) = f_e(\phi(s))$，仅在**成功轨迹**上用 [[Huber 损失]] 训练，预测剩余步数（负数越小越好）

#### 模块2：状态自适应门控（State-Adaptive Gate）

**设计动机**: 任务早期存活性低，应优先提升成功概率；接近成功时存活性高，应优化效率。

**具体实现**:
- 当 $p_v(s_t)$ 低时，门控倾向于 $A_v$（鼓励向成功状态迈进）
- 当 $p_v(s_t)$ 高时，门控倾向于 $A_e$（减少低效的重复动作）
- 通过 $\tanh$ 压缩保证权重有界，配合均值归一化解耦有效学习率

#### 模块3：干预感知信用分配（Intervention-Aware Credit Assignment）

**设计动机**: 人工干预（IR）可帮助机器人从失败状态恢复，但干预本身的结果标签不应归属于被干预的动作。

**具体实现**:
- **全自主 episode**：整条轨迹用结果 $y$ 标注
- **含干预 episode**：仅干预后的**策略执行后缀**获得结果标签
- **干预窗口本身**：作为纯模仿学习数据（$\tilde{w}_i = 1$）和 $V_e$ 进度目标，永不用于结果归因

---

## 关键公式

### 公式1：[[加权行为克隆|Actor 损失]]

$$
\mathcal{L}_{\pi} = \frac{1}{B}\sum_{i=1}^{B} w_i \lVert v_{\theta}(s_i, a_i, \sigma) - u_i \rVert_2^2
$$

**含义**: 加权行为克隆损失，通过 $w_i$ 上调正向 transition、下调负向 transition 的梯度贡献。

**符号说明**:
- $v_{\theta}$: [[Flow Matching|流匹配]] 速度网络（策略主干）
- $u_i$: 地面真实流匹配目标（ground-truth flow target）
- $\sigma$: 流匹配时间步噪声水平
- $w_i$: 来自双头 Critic 的组合权重（见公式8）
- $B$: batch size（256）

### 公式2：[[双头Critic|存活性头与效率头]]

$$
z_v(s) = f_v(\phi(s)), \quad p_v(s) = \text{sigmoid}(z_v(s))
$$

$$
\hat{V}_e(s) = f_e(\phi(s))
$$

**含义**: 存活性头输出 logit $z_v$ 后经 sigmoid 得到成功概率 $p_v$；效率头直接输出剩余步数估计。

**符号说明**:
- $\phi(\cdot)$: 共享的视觉-语言特征提取器（来自 [[π0.5]]，冻结）
- $f_v, f_e$: 两个独立的线性头
- $p_v(s)$: 当前状态下最终成功的概率估计
- $\hat{V}_e(s)$: 当前状态距成功所需步数估计（负值，越大表示越近）

### 公式3：[[效率头|效率目标]]

$$
y_e(s_t) = \begin{cases} 0, & d_t = 1 \\ -1 + \text{sg}[\hat{V}_e(s_{t+1})], & d_t = 0 \end{cases}
$$

**含义**: 效率头的 TD（时序差分）训练目标：成功时为 0，否则施加每步 -1 惩罚并自举下一步估计。

**符号说明**:
- $d_t = 1$: 当前步骤为终止成功状态
- $\text{sg}[\cdot]$: 停止梯度算子（stop-gradient），避免 bootstrap 目标的梯度穿透
- $-1$: 每步存在的代价，促使策略加速完成任务

### 公式4：[[联合Critic损失|Critic 联合损失]]

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_{\mathcal{D}^{\text{lab}}_{\text{auto}}}[\text{BCE}(z_v, y)] + \mathbb{E}_{\mathcal{D}_{\text{succ}}}[\text{Huber}(\hat{V}_e, y_e)]
$$

**含义**: 存活性头在所有有标注数据上用 BCE 训练，效率头仅在成功轨迹上用 Huber 损失训练，两者损失求和。

**符号说明**:
- $\mathcal{D}^{\text{lab}}_{\text{auto}}$: 所有有结果标注的策略执行数据（含成功与失败）
- $\mathcal{D}_{\text{succ}}$: 仅包含成功轨迹的数据集
- $\text{BCE}$: [[二元交叉熵]] 损失
- $\text{Huber}$: [[Huber 损失]]，对异常值鲁棒（$\delta=1.0$）

### 公式5：[[存活性优势|存活性优势值]]

$$
A_v = z_v(s_{t+1}) - z_v(s_t)
$$

**含义**: 以 logit 空间（而非概率空间）计算优势，在 $p_v \approx 1$ 时仍保持分辨率，避免 sigmoid 饱和区梯度消失。

**符号说明**:
- $z_v(s_t)$: 当前状态的存活性 logit
- $z_v(s_{t+1})$: 下一状态的存活性 logit
- $A_v > 0$: 当前动作使成功概率增加（正向贡献）

### 公式6：[[效率优势|效率优势值]]

$$
A_e = -1 + \hat{V}_e(s_{t+1}) - \hat{V}_e(s_t)
$$

**含义**: 效率优势包含每步 -1 的时间代价，当步骤缩短距成功的距离超过 1 步时为正，鼓励高效执行。

**符号说明**:
- $-1$: 每步固定代价
- $\hat{V}_e(s_{t+1}) - \hat{V}_e(s_t)$: 效率状态值的变化量

### 公式7：[[状态自适应门控|状态自适应门控]]

$$
g_t = 1 + \tanh\left((1 - p_v(s_t)) \cdot A_v + p_v(s_t) \cdot A_e\right)
$$

**含义**: 核心融合公式。当 $p_v$ 低时 $(1-p_v)$ 大，权重倾向于存活性优势；当 $p_v$ 高时，权重倾向于效率优势。$\tanh$ 将输出压缩到 $(0, 2)$，加 1 保证恒正。

**符号说明**:
- $p_v(s_t)$: 当前时刻的成功概率估计（自适应混合系数）
- $A_v$: 存活性优势（存活性低时起主导作用）
- $A_e$: 效率优势（存活性高时起主导作用）
- $g_t \in (0, 2)$: 每个 transition 的最终加权系数

### 公式8：[[权重归一化|训练权重归一化]]

$$
c = \frac{1}{|\mathcal{D}_{\text{valid}}|}\sum_{i \in \mathcal{D}_{\text{valid}}} \tilde{w}_i, \quad w_i = \tilde{w}_i / \max(c, \varepsilon)
$$

**含义**: 对所有有效 transition（$\tilde{w}_i > 0$）的预归一化权重求均值，再做归一化，使权重均值恒为 1，将加权从有效学习率中解耦。

**符号说明**:
- $\tilde{w}_i$: 预归一化权重（成功自主 rollout 为 $g_t$，失败 rollout 为 0，SFT 和干预样本为 1）
- $\mathcal{D}_{\text{valid}}$: 非零权重的 transition 集合
- $\varepsilon$: 防止除零的小常数
- $w_i$: 最终用于损失计算的归一化权重

---

## 关键图表

### Figure 1：HABC 系统概览

![Figure 1](https://arxiv.org/html/2606.17043v1/x1.png)

**说明**: HABC 的整体框架。左侧展示 episode 级别的结果标注与干预感知数据分区；右侧展示双头 [[双头Critic|Critic]] 如何分别生成 $A_v, A_e$，经 [[状态自适应门控]] $g_t$ 合并后对 [[加权行为克隆]] Actor 提供逐 transition 权重。

### Figure 2：真实机器人双臂任务

![Figure 2](https://arxiv.org/html/2606.17043v1/x2.png)

**说明**: 三项接触丰富的双臂操作任务：(1) **Pencil Pouch** 将马克笔插入软袋并拉链；(2) **Paper Bag** 展开折叠纸袋、使其站立、放入瓶子；(3) **Snack Bag** 将三件物品放入束口袋并收紧绳子。均使用 ARX X5 双臂机器人和三台 Intel RealSense 摄像头。

### Figure 3：主实验结果（成功率 + 轨迹长度）

![Figure 3](https://arxiv.org/html/2606.17043v1/x3.png)

**说明**: 三项任务在等算力预算（Step 1）和继续训练+干预（Step 2）下的成功率曲线及轨迹长度对比。HABC 在全部任务上均优于 Imit-DAgger 和 Imit-Recap；加入 IR（Intervention Replay）后成功率进一步提升，轨迹长度相比 HABC-V 缩短 32-162 帧，证明效率头的有效性。

**主实验结果表（Step 1：等算力初始微调）**

| 方法 | Pencil Pouch | Paper Bag | Snack Bag |
|------|:---:|:---:|:---:|
| SFT（基线）| 36% | 44% | 12% |
| Imit-DAgger | 34% | 38% | 6% |
| Imit-Recap | 46% | 66% | 16% |
| HABC-V（仅存活性头）| 54% | 72% | 18% |
| **HABC（完整方法）**| **60%** | **78%** | **22%** |

**主实验结果表（Step 2：继续训练 + 干预）**

| 方法 | Pencil Pouch | Paper Bag | Snack Bag |
|------|:---:|:---:|:---:|
| HABC（续训）| 68% | 80% | 26% |
| HABC+IR | 78% | 84% | 30% |
| **最优 HABC+IR** | **92%** | **88%** | **38%** |

### Figure 4：存活性头泛化分析

![Figure 4a](https://arxiv.org/html/2606.17043v1/x4.png)
![Figure 4b](https://arxiv.org/html/2606.17043v1/x5.png)

**说明**: 可视化存活性预测 $p_v$ 随训练进行的变化。训练过程中 $p_v > 0.6$ 的高存活性区域持续扩大，说明 [[存活性（Viability）|存活性头]] 能够泛化到训练数据之外的状态。单条 rollout 中，$p_v$ 在 OOD 抓握失败时急剧下降，在人工干预阶段稳步回升，定性证明了存活性估计的合理性。

### Figure 5：逐 transition 权重分析

![Figure 5](https://arxiv.org/html/2606.17043v1/x6.png)

**说明**: 沿一条成功 rollout 轨迹可视化 $g_t$ 权重及其分解。可观察到三类行为模式：(1) **成功抓握**时 $p_v$ 急升，$A_v$ 主导上调权重；(2) **低效重复抓握**时 $p_v$ 稳定但 $\hat{V}_e$ 恶化，$A_e$ 主导下调权重；(3) **恢复动作**时两者均改善，权重双重上调。

### Figure 6：恢复行为对比（HABC vs SFT）

![Figure 6](https://arxiv.org/html/2606.17043v1/x7.png)

**说明**: 定性对比 SFT 基线与 HABC 策略在遇到抓握失败时的响应。SFT 策略重复执行失败动作或陷入不可恢复循环；HABC 策略能检测失败状态并执行纠正动作，展示了无需人工干预的自发恢复能力。

### Figure 7：干预感知信用分配示意图

![Figure 7](https://arxiv.org/html/2606.17043v1/x8.png)

**说明**: 展示干预感知信用分配的数据分区规则。蓝色为策略自主执行段（获得结果标签 $y$），红色为人工干预窗口（作为纯模仿数据），绿色为干预后策略执行后缀（同样获得结果标签）。这一分区防止干预行为"污染"策略的结果归因。

**消融实验（HABC-V vs HABC，Step 1 成功率）**

| 配置 | Pencil Pouch | Paper Bag | Snack Bag | 说明 |
|------|:---:|:---:|:---:|------|
| HABC-V（无效率头）| 54% | 72% | 18% | 仅存活性优势 |
| **HABC（完整）** | **60%** | **78%** | **22%** | 双头完整方法 |
| 轨迹长度差（HABC-V - HABC）| +55帧 | +162帧 | +32帧 | HABC 更高效 |

**关键发现**: 效率头带来的提升不仅体现在成功率，更显著地体现在轨迹效率（减少低效重复动作）。

---

## 实验

### 数据集

| 数据 | 规模 | 特点 | 用途 |
|------|------|------|------|
| SFT 示范数据 | 200 episodes/任务 | 人工示范，双臂操作 | 初始监督训练 |
| 在线 rollout 数据 | 100 episodes/轮 | 自主执行，含成功/失败 | 在线 RL 微调 |
| 干预数据（IR）| ~50% 失败 rollout | 人工干预帮助恢复 | 干预回放数据 |

### 实现细节

- **基础 VLA**: [[π0.5]]（基于流匹配的视觉-语言-动作模型）
- **机器人**: ARX X5 双臂系统
- **摄像头**: 顶部 Intel RealSense D455 × 1 + 腕部 D405 × 2
- **动作空间**: 末端执行器坐标系，[[Action Chunking]] size = 50（训练）/ 25（推理）
- **硬件**: 8 × A800 GPU
- **每轮训练**: 100 次 rollout 后进行 6000 步梯度更新
- **总轮数**: 3 轮初始微调 + 继续 HABC+IR 训练
- **超参数**: $\gamma=0.99$，$\delta=1.0$（Huber），Warmup 500 步，Batch size 256，过时 rollout 截断 10000 步

### 关键超参数

| 参数 | 值 |
|------|-----|
| Episode 失败惩罚 $C$ | 100 |
| Warmup 步数 $N_{wu}$ | 500 |
| Huber $\delta$ | 1.0 |
| 折扣因子 $\gamma$ | 0.99 |
| Batch size $B$ | 256 |
| 过时 rollout 截断 | 10000 步 |

### 权重统计（Appendix）

| 统计量 | Pencil Pouch | Paper Bag | Snack Bag |
|--------|:---:|:---:|:---:|
| 成功自主轨迹平均 $g_t$ | 0.76 | 0.68 | 0.86 |
| 干预轨迹平均 $g_t$ | 1.0 | 1.1 | 0.95 |
| Imit-Recap 通过率 | 19-27% | 19-27% | 19-27% |

---

## 批判性思考

### 优点

1. **信号分解合理**: 将单一二值结果解耦为存活性与效率两个层次，从信息论角度减少了监督信号的信息损失。
2. **软加权优于硬过滤**: 相比 Imit-Recap 的硬阈值（仅 19-27% transition 通过），HABC 保留了所有成功 transition 的软性贡献，梯度信号更丰富。
3. **干预数据利用规范**: 干预感知信用分配是对 [[DAgger]] 系列方法的重要改进，解决了长期被忽视的 label leakage 问题。
4. **实机验证**: 在三项接触丰富的双臂真实任务上验证，比仅在仿真中测试更有说服力。

### 局限性

1. **干预边界检测依赖**: 需要可靠地检测人工干预的起止点，在某些系统中可能难以自动化获取。
2. **稀疏成功时效率头退化**: 当任务成功率极低时，成功轨迹数据稀少，效率头 $V_e$ 的估计质量下降（如 Snack Bag 仅 38% 最优成功率）。
3. **单任务微调**: 目前每个任务独立训练，多任务和跨体态泛化能力未经验证。
4. **基础模型依赖**: 以 [[π0.5]] 为基础，方法在其他 VLA 架构上的适用性需进一步验证。

### 潜在改进方向

1. **多步优势估计**: 当前使用单步 TD，可引入 $n$-step 或 GAE 提升价值估计质量。
2. **自适应门控机制**: $g_t$ 中的线性混合可替换为学习到的混合函数，更灵活地适应复杂任务。
3. **更密集的结果信号**: 探索利用中间接触检测等信号作为里程碑奖励，缓解 Snack Bag 类低成功率任务的冷启动问题。

### 可复现性评估

- [ ] 代码开源（未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数表完整）
- [ ] 数据集可获取（真实机器人数据，难以公开）

---

## 关联笔记

### 基于

- [[π0.5]]: 基础 VLA 模型，提供共享视觉-语言特征提取器 $\phi$
- [[Flow Matching]]: Actor 使用的生成建模框架，HABC 通过加权 BC 损失微调其速度网络

### 对比

- [[DAgger]]: 经典干预模仿学习基线，HABC-IR 是其改进版
- [[Recap]]: 基于 TD 残差的硬阈值过滤方法，HABC 用软加权替代

### 方法相关

- [[强化学习]]: 核心优化框架，HABC 通过离线优势加权近似在线 RL
- [[优势加权回归（AWR）]]: HABC 所属的优势加权 BC 方法家族
- [[Action Chunking]]: VLA 动作预测的分块机制，chunk size=50/25
- [[双头Critic]]: HABC 的核心创新，分别建模存活性与效率
- [[状态自适应门控]]: 动态混合两类优势信号的融合机制
- [[存活性（Viability）]]: 成功概率预测目标
- [[效率（Efficiency）]]: 步数-到-成功预测目标

### 硬件/数据相关

- [[ARX X5]]: 使用的双臂机器人平台
- [[Intel RealSense]]: 视觉感知传感器（D455 + D405）

---

## 速查卡片

> [!summary] HABC: Hierarchical Advantage-Weighted Behavior Cloning
> - **核心**: 将稀疏二值任务结果分解为存活性+效率双目标，实现 VLA 在线 RL 微调
> - **方法**: 双头 Critic（$p_v$ + $\hat{V}_e$）+ 状态自适应门控 $g_t$ + 干预感知信用分配
> - **结果**: 三项双臂真实任务成功率从 36%/44%/12% 提升至 92%/88%/38%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-16*
