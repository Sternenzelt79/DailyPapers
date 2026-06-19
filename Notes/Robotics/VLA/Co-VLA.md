---
title: "Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm Vision-Language-Action Systems"
method_name: "Co-VLA"
authors: [Yandong Wang, Jiaqian Yu, Xiongfeng Peng, Lu Xu, Yamin Mao, Weiming Li, Jaewook Yoo, Dongwook Lee, Daehyun Ji, Mingbo Zhao, Chao Zhang]
year: 2026
venue: arXiv
tags: [dual-arm-manipulation, vla, bimanual-coordination, structured-action, action-decomposition]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.20285
created: 2026-06-19
---

# 论文笔记：Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm VLA Systems

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多家机构联合（未标注具体单位） |
| 日期 | June 2026 |
| 项目主页 | - |
| 对比基线 | [[π0]] / [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.20285) / Code: - |

---

## 一句话总结

> Co-VLA 通过显式结构化先验将双臂协调分解为共享与残差潜变量，配合部署时自适应控制器，使双臂 VLA 系统在紧耦合任务中成功率提升 27%，OOD 场景性能翻倍。

---

## 核心贡献

1. **Structured Action Expert（SAE）**: 将双臂动作分解为共享协调潜变量和臂特定残差潜变量，以任务自适应的辅助损失塑造协调结构
2. **Latent-Aware Controller（LAC）**: 基于潜变量语义在部署时自适应调整动作平滑度，无需专用力/阻抗控制硬件
3. **Co-Motion 示范范式**: 在 RoboTwin 2.0 框架内生成并发双臂轨迹，将示范生成效率提升 10-25%

---

## 问题背景

### 要解决的问题

标准 [[VLA]] 系统将双臂动作视为单体向量（14 维关节命令）直接回归，不对双臂协调结构作任何显式建模。这限制了模型在需要紧密协调的任务（如双手同步移动、一臂稳定另一臂精调）上的泛化能力。

### 现有方法的局限

- **[[π0]] / [[π0.5]]** 等 VLA 基础模型采用流匹配动作头，生成 $a_t = [a_{t,L}; a_{t,R}] \in \mathbb{R}^{14}$，左右臂耦合隐含于模型权重中，缺乏可解释性
- 经典双臂控制（虚拟链接模型、混合位置/力协调）依赖专用力传感，难以与关节级 VLA 管线兼容
- 多任务学习中的共享-私有分解思路（QMIX、MAPPO）从未被引入 VLA 动作头

### 本文的动机

> "双臂操作中的协作不是动作本身，而是动作之上的结构。"

作者认为，将协调结构显式注入表示层，可使模型在未见过的物体位置、光照、背景下保持协调能力，实现从隐式到显式的结构建模。

---

## 方法详解

### 模型架构

Co-VLA 以 [[π0]] 为预训练骨干，在动作头处替换为 **Structured Action Expert（SAE）**，并在部署时叠加 **Latent-Aware Controller（LAC）**：

- **输入**: 语言指令 $l$ + 多视角观测 $o_t$（含腕部相机）+ 关节状态 $s_t$
- **Backbone**: [[π0]]（7B 参数流匹配 VLA）
- **核心模块**: [[SAE|Structured Action Expert]] 进行共享-残差分解，[[LAC|Latent-Aware Controller]] 进行动作细化
- **输出**: [[Action Chunking|动作块]] $\{q_t\}_{t=1}^{H}$，其中 $q_t = [a_{t,L}; a_{t,R}] \in \mathbb{R}^{14}$

### 核心模块

#### 模块1: Structured Action Expert（SAE）

**设计动机**: 利用[[共享-残差分解|Shared-Residual Decomposition]]将任务级协调意图与执行细节解耦，让共享潜变量捕获双臂通用运动趋势，残差潜变量捕获臂特定调整。

**具体实现**:

给定骨干特征 $h_t$，SAE 计算三个潜变量：

- $z^s_t = g_s(h_t) \in \mathbb{R}^L$：**共享潜变量**，编码双臂公共协调信息
- $z^L_t = g_L(h_t) \in \mathbb{R}^L$：**左臂残差潜变量**，编码左臂特有调整
- $z^R_t = g_R(h_t) \in \mathbb{R}^L$：**右臂残差潜变量**，编码右臂特有调整

通过独立解码器生成动作分量：
- 共享分量：$a^s_{t,L} = \phi^s_L(z^s_t)$，$a^s_{t,R} = \phi^s_R(z^s_t)$
- 残差分量：$a^r_{t,L} = \phi^r_L(z^L_t)$，$a^r_{t,R} = \phi^r_R(z^R_t)$
- 最终关节命令：$a_{t,L} = a^s_{t,L} + a^r_{t,L}$，$a_{t,R} = a^s_{t,R} + a^r_{t,R}$

#### 模块2: Latent-Aware Controller（LAC）

**设计动机**: 利用[[自适应平滑|Adaptive Smoothing]]在不依赖力传感硬件的前提下，在宏观运动与精细调整之间动态切换刚度，既抑制噪声又保留协调关键动作。

**具体实现**:

对预测动作块 $\{q_t\}_{t=1}^{H}$ 进行逐步细化：
1. 计算共享能量 $E^s_t$ 和残差能量 $E^r_t$，得微运动比 $\rho_t$
2. 计算残差对立信号 $\omega_t$（余弦相似度取负），判断双臂是否协同对抗（如夹持）
3. 根据 $\rho_t$ 和 $\omega_t$ 进入三种执行模式，调整滤波系数 $\hat{\alpha}_t$
4. 时间平滑 $\alpha_t = (1-\beta)\alpha_{t-1} + \beta\hat{\alpha}_t$
5. 关节细化：$\tilde{q}_t = (1-\alpha_t)\tilde{q}_{t-1} + \alpha_t q_t$

#### 模块3: Co-Motion 示范范式

**设计动机**: 经典顺序示范法（先完成一臂再执行另一臂）低估时序协调，Co-Motion 在 RoboTwin 2.0 代码生成管线中引入并行轨迹规划。

**三个支撑机制**:
1. **共享参考系**：设定公共空间锚点（如交接中点、目标位姿）
2. **前瞻预计算**：预估近期交互目标，生成更平滑的并发轨迹
3. **安全中间目标**：显式留出安全余量，失败时回退顺序执行

---

## 关键公式

### 公式1: [[共享-残差分解|双臂动作分解]]

$$
a_{t,L} = a^s_{t,L} + a^r_{t,L}, \quad a_{t,R} = a^s_{t,R} + a^r_{t,R}
$$

**含义**: 将左右臂的最终关节命令分解为来自共享潜变量的协调分量与来自各臂残差潜变量的特化分量之和。

**符号说明**:
- $a^s_{t,L}, a^s_{t,R} \in \mathbb{R}^7$：共享潜变量解码的协调动作分量
- $a^r_{t,L}, a^r_{t,R} \in \mathbb{R}^7$：残差潜变量解码的臂特定调整分量

---

### 公式2: [[稀疏正则化|稀疏残差正则损失]]

$$
\mathcal{L}_{\text{sparse}} = \mathbb{E}_t \left[ \|a^r_{t,L}\|_1 + \|a^r_{t,R}\|_1 \right]
$$

**含义**: 通过 L1 范数鼓励残差分量趋近零，使共享潜变量主导对称任务（如双手同步抬升）。

**符号说明**:
- $a^r_{t,L}, a^r_{t,R}$：左、右臂残差动作分量
- $\|\cdot\|_1$：L1 范数，促进稀疏性

---

### 公式3: [[共享一致性损失|共享均值速度一致性损失]]

$$
\mathcal{L}_{\text{shared}} = \mathbb{E}_t \left[ \|a^s_{t,L} - \bar{u}_t\|^2_2 + \|a^s_{t,R} - \bar{u}_t\|^2_2 \right]
$$

$$
\bar{u}_t = \frac{u_{t,L} + u_{t,R}}{2}
$$

**含义**: 将共享分量对齐到双臂平均运动趋势 $\bar{u}_t$，使共享潜变量捕获两臂共有的宏观运动模式，适用于非对称任务。

**符号说明**:
- $u_{t,L}, u_{t,R}$：左、右臂的真实速度/动作参考
- $\bar{u}_t$：双臂平均参考
- $\|\cdot\|^2_2$：L2 范数平方

---

### 公式4: [[时序同步损失|时序同步损失]]

$$
\mathcal{L}_{\text{sync}} = 1 - \text{corr}_{\text{pred}}, \quad \text{corr}_{\text{pred}} = \mathbb{E}_t[\tilde{m}_{t,L} \cdot \tilde{m}_{t,R}]
$$

**含义**: 通过最大化预测动作的跨臂相关性，鼓励双臂同步加速/减速，适用于时序耦合型任务（如双手同步收拢）。

**符号说明**:
- $\tilde{m}_{t,L}, \tilde{m}_{t,R}$：归一化的左、右臂运动信号
- $\text{corr}_{\text{pred}}$：预测动作的跨臂相关系数

---

### 公式5: [[辅助损失|训练总目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \cdot \mathcal{L}_{\text{aux}}, \quad \lambda = 0.001
$$

**含义**: 在 VLA 动作回归损失基础上加入任务自适应辅助损失，通过超参 $\lambda$ 控制结构化先验的强度。

**符号说明**:
- $\mathcal{L}_{\text{action}}$：主动作预测损失（流匹配损失）
- $\mathcal{L}_{\text{aux}} \in \{\mathcal{L}_{\text{sparse}}, \mathcal{L}_{\text{shared}}, \mathcal{L}_{\text{sync}}\}$：根据任务主导协调模式选择
- $\lambda = 0.001$：固定权重系数

---

### 公式6: [[微运动比|潜变量能量分析]]

$$
E^s_t = \frac{1}{2}(\|a^s_{t,L}\|_2 + \|a^s_{t,R}\|_2), \quad E^r_t = \frac{1}{2}(\|a^r_{t,L}\|_2 + \|a^r_{t,R}\|_2)
$$

$$
\rho_t = \frac{E^r_t}{E^s_t + \varepsilon}
$$

**含义**: 微运动比 $\rho_t$ 量化残差能量相对共享能量的比例，高 $\rho_t$ 表示当前时刻需要精细调整，低 $\rho_t$ 表示宏观协调运动主导。

**符号说明**:
- $E^s_t$：共享能量，反映宏观协调运动强度
- $E^r_t$：残差能量，反映臂特定精细调整强度
- $\rho_t$：微运动比（越大越需精细控制）
- $\varepsilon$：数值稳定性小量

---

### 公式7: [[对立信号|残差对立与协作信号]]

$$
\cos_t = \frac{\langle a^r_{t,L}, a^r_{t,R} \rangle}{\|a^r_{t,L}\|_2 \cdot \|a^r_{t,R}\|_2 + \varepsilon}, \quad \omega_t = -\cos_t
$$

**含义**: 对立分数 $\omega_t$ 高（接近1）时，左右臂残差方向相反，指示协同对抗行为（如双手夹持、稳定），LAC 此时保留精细动作。

**符号说明**:
- $\cos_t$：左右臂残差动作的余弦相似度
- $\omega_t = -\cos_t$：对立分数，越高越表示协调夹持/对抗行为

---

### 公式8: [[自适应平滑|LAC 关节级动作细化]]

$$
\tilde{q}_t = (1 - \alpha_t) \cdot \tilde{q}_{t-1} + \alpha_t \cdot q_t
$$

$$
\alpha_t = (1-\beta) \cdot \alpha_{t-1} + \beta \cdot \hat{\alpha}_t, \quad \beta = 0.2
$$

**含义**: 以时变滤波系数 $\alpha_t$ 对原始预测动作做低通滤波，$\alpha_t$ 大时跟随预测（精细模式），$\alpha_t$ 小时平滑过渡（噪声抑制模式）。

**符号说明**:
- $q_t \in \mathbb{R}^{14}$：原始预测关节命令
- $\tilde{q}_t$：细化后的关节命令
- $\alpha_t \in [0.05, 0.95]$：自适应滤波系数
- $\hat{\alpha}_t$：三种执行模式对应的目标滤波系数
- $\beta = 0.2$：平滑率

---

## 关键图表

### Figure 1: Structured Action Expert（SAE）架构

![Figure 1 - SAE Architecture](https://arxiv.org/html/2606.20285v1/fig/co-vla-SAE-new.png)

**说明**: Co-VLA 的核心架构图。预训练骨干特征 $h_t$ 通过三个并行编码器生成共享潜变量 $z^s_t$ 和左、右臂残差潜变量 $z^L_t, z^R_t$，经独立解码器输出共享分量和残差分量后相加得到最终关节命令。[[π0]] 骨干保持冻结（热身阶段），SAE 层单独训练。

---

### Figure 2: 顺序运动 vs 协作运动范式

![Figure 2 - Co-Motion Paradigm](https://arxiv.org/html/2606.20285v1/fig/co-motion.png)

**说明**: 左侧为传统顺序示范（先执行一臂完成再执行另一臂），右侧为 Co-Motion 并行示范。Co-Motion 通过共享参考系和前瞻预计算实现双臂并发轨迹，减少 10-25% 的示范生成时间，但因时序耦合更紧密而增大学习难度。

---

### Figure 3: 评估任务

![Figure 3a - RoboTwin Tasks](https://arxiv.org/html/2606.20285v1/fig/robotwin_task.png)

![Figure 3b - Real-World Tasks](https://arxiv.org/html/2606.20285v1/fig/real_task.png)

**说明**: 上方为 RoboTwin 2.0 仿真中的 8 个双臂任务（Aloha-AgileX 机器人），涵盖 Handover Block、Lift Pot 等需要并发运动的场景；下方为真实 AgileX Cobot Magic 双臂机器人上的 3 个任务（Handover、Pick Bottles、Lift Pot）。

---

### Figure 4: OOD 真实世界评估条件

![Figure 4 - OOD Conditions](https://arxiv.org/html/2606.20285v1/fig/real_task-OOD.png)

**说明**: OOD 评估引入三类干扰：杂乱背景（cluttered backgrounds）、干扰物（distractor objects）、低光照（low-lighting），以 Lift Pot 任务为例展示。Co-VLA 完整系统在 Handover OOD 场景从 13% 提升至 27%，Lift Pot OOD 从 33% 提升至 37%。

---

### Figure 5: 潜变量-行为对齐分析（Pick Bottles 任务）

![Figure 5a - Shared Energy vs Synchronization](https://arxiv.org/html/2606.20285v1/fig/norm_s_vs_bimanual_coordination_pick.png)

![Figure 5b - Residual Energy vs Synchronization](https://arxiv.org/html/2606.20285v1/fig/norm_r_vs_bimanual_coordination_pick.png)

**说明**: 左图显示共享能量 $E^s_t$ 与双臂同步性的正相关关系；右图显示残差能量 $E^r_t$ 与双臂同步性的负相关关系。验证了共享潜变量作为协调驱动器、残差潜变量编码臂特定偏差的设计假设。

---

### Figure 6: LAC 对关节轨迹的影响

![Figure 6 - LAC Effect](https://arxiv.org/html/2606.20285v1/fig/acc-zoom-in.png)

**说明**: 对比原始预测、LAC 细化和朴素 EMA 滤波的关节轨迹。朴素 EMA 产生最平滑轨迹但任务成功率下降（60% vs LAC 的 67%），因为 EMA 过度平滑了精细调整所需的微运动。LAC 选择性保留协调关键调整，同时抑制高频噪声。

---

### Table 1: RoboTwin 2.0 仿真任务成功率（100 次 rollout）

| 任务 | π₀.₅ Easy | π₀.₅ Hard | π₀ Easy | π₀ Hard | Co-VLA Easy | Co-VLA Hard |
|------|-----------|-----------|---------|---------|-------------|-------------|
| Handover Block | 44% | 10% | 64% | 7% | **91%** | 12% |
| Lift Pot | 100% | 60% | 100% | 63% | **100%** | **65%** |
| Pick Diverse Bottles | 95% | 18% | 91% | 13% | **95%** | 16% |
| Pick Dual Bottles | 100% | 27% | 100% | 16% | **100%** | 18% |
| Place Bread Basket | 92% | 28% | 70% | 46% | 69% | **48%** |
| Place Bread Skillet | 38% | 7% | 63% | 8% | **70%** | 8% |
| Put Object Cabinet | 70% | 19% | 79% | 8% | **81%** | 5% |
| Scan Object | 45% | 6% | 41% | 3% | **50%** | 2% |
| **平均** | **73%** | **21.9%** | **76%** | **21%** | **82%** | **22%** |

**关键发现**: Co-VLA 在 Easy 设置下将平均成功率从 76%（π₀）提升至 82%。Handover Block（最需协调）提升最显著（64%→91%），说明共享-残差分解对角色耦合任务效果最强。

---

### Table 2: Co-Motion 示范生成效率（生成 1000 条示范的平均时间，秒）

| 任务 | 顺序范式 | Co-Motion |
|------|---------|-----------|
| Handover Block | 9.47 | **8.49** |
| Scan Object | 5.65 | **4.52** |
| Place Bread Skillet | 5.51 | **4.59** |
| Put Object Cabinet | 8.95 | **6.45** |

**关键发现**: Co-Motion 节省 10-25% 示范生成时间，但成功 rollout 的平均完成时间（Table 3）也相应改善，Co-Motion + Co-VLA 组合平均 18.18 秒 vs 顺序 + π₀ 的 21.58 秒。

---

### Table 3: 成功 Rollout 平均完成时间（秒）

| 任务 | 顺序 π₀ | 顺序 Co-VLA | Co-Motion π₀ | Co-Motion Co-VLA |
|------|---------|------------|-------------|-----------------|
| Handover Block | 28.39 | 28.56 | 32.87 | **25.38** |
| Place Bread Skillet | 15.02 | 15.28 | 12.79 | **12.35** |
| Put Object Cabinet | 26.96 | 28.69 | 20.87 | **21.08** |
| Scan Object | 15.94 | 19.00 | 12.57 | **13.92** |
| **平均** | 21.58 | 22.88 | 19.78 | **18.18** |

---

### Table 4: Co-Motion 训练可学习性（成功率 %，100 次 rollout）

| 任务 | π₀ Easy | π₀ Hard | Co-VLA Easy | Co-VLA Hard |
|------|---------|---------|-------------|-------------|
| Handover Block | 15% | 8% | **38%** | 8% |
| Place Bread Skillet | 60% | 3% | **62%** | **8%** |
| Put Object Cabinet | 83% | **45%** | 70% | **47%** |
| Scan Object | 43% | 8% | **53%** | 3% |
| **平均** | 50% | 16% | **56%** | **17%** |

**关键发现**: 相比顺序示范（82% Easy），Co-Motion 训练后成功率下降（Co-VLA 56% Easy），揭示了效率-可学习性权衡：并发轨迹带来更紧密的时序耦合，增加了策略学习的难度。

---

### Table 5: 真实机器人实验（30 次 rollout）

| 任务 | 方法 | ID | OOD |
|------|------|----|-----|
| Handover | π₀ | 63% | 13% |
| | Co-VLA（无 LAC） | 57% | 17% |
| | **Co-VLA（含 LAC）** | **73%** | **27%** |
| Pick Bottles | π₀ | 57% | 37% |
| | Co-VLA（无 LAC） | 57% | 50% |
| | **Co-VLA（含 LAC）** | **67%** | 47% |
| Lift Pot | π₀ | 43% | 33% |
| | Co-VLA（无 LAC） | 63% | 27% |
| | **Co-VLA（含 LAC）** | **67%** | **37%** |

**关键发现**: 完整系统在 6 个评估设置中 5 个最优。LAC 的必要性：Co-VLA 无 LAC 的 Handover ID 成功率（57%）低于 π₀（63%），说明 SAE 的结构化分解在无精细化控制时反而引入不稳定，LAC 是协调结构实际生效的关键。

---

### Table 6: 辅助损失消融（Easy 设置，100 次 rollout）

| 辅助损失 | Lift Pot（对称） | Cabinet（非对称） | Skillet（时序耦合） |
|---------|----------------|----------------|-----------------|
| 无 | 99% | 49% | 51% |
| + $\mathcal{L}_{\text{sparse}}$ | 100% | 51% | 55% |
| + $\mathcal{L}_{\text{shared}}$ | 100% | **72%** | 55% |
| + $\mathcal{L}_{\text{sync}}$ | 100% | 70% | **62%** |

**关键发现**: 辅助损失的选择应与任务主导协调模式匹配：非对称任务（Cabinet）受益于共享一致性损失（+23pp），时序耦合任务（Skillet）受益于同步损失（+11pp）。任务无关的基线（Lift Pot）三种损失效果相近。

---

## 实验

### 数据集

| 数据集/平台 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| RoboTwin 2.0 | 8 双臂任务，各 1000 条示范 | LLM 生成脚本，含并发/顺序示范 | 仿真训练与评估 |
| AgileX Cobot Magic 真实数据 | 每任务 50 条人类遥操作示范 | 随机化物体位置/姿态 | 真实世界训练与评估 |

### 实现细节

- **Backbone**: [[π0]]（7B 参数，[[流匹配|Flow Matching]] 动作头）
- **训练策略**:
  - 热身阶段：1,000 步，冻结骨干，仅训练 SAE；峰值学习率 $5\times10^{-5}$
  - 全量微调：30,000 步，解冻所有参数；学习率从 $2.5\times10^{-5}$ 衰减至 $2.5\times10^{-6}$
- **Batch Size**: 32，4 GPU，[[FSDP|Fully Sharded Data Parallel]]
- **LAC 超参**: $\alpha_{\text{base}}=0.4$，$\Delta_{\text{macro}}=0.4$，$\Delta_{\text{prec}}=0.3$，$\Delta_{\text{noise}}=0.1$，$\tau_\rho=0.01$，$\tau_\omega=0.3$，$\beta=0.2$

---

## 批判性思考

### 优点
1. **架构无关性强**: SAE 仅替换动作头，兼容任意连续动作 VLA 骨干，无需修改主干网络
2. **无额外硬件依赖**: LAC 在关节命令级别工作，无需力传感器或阻抗控制器
3. **结构先验可解释**: 共享/残差潜变量与双臂协调语义对齐，有 Figure 5 的定量验证

### 局限性
1. **辅助损失需人工选择**: 三种辅助损失（$\mathcal{L}_{\text{sparse}}$、$\mathcal{L}_{\text{shared}}$、$\mathcal{L}_{\text{sync}}$）需根据任务类型手动配置，消融实验也仅测试单损失变体，自动路由尚未实现
2. **Co-Motion 效率-可学习性权衡未解**: 并发示范将平均成功率从 82% 降至 56%，高效示范与策略可学习性之间的矛盾是开放问题
3. **跨骨干泛化验证不足**: 仅在 π₀ 上验证，是否适用于 [[Diffusion Policy]]、[[Octo]] 等其他范式有待验证

### 潜在改进方向
1. **自适应损失路由**: 自动识别任务的主导协调模式，动态加权 $\mathcal{L}_{\text{sparse}}$、$\mathcal{L}_{\text{shared}}$、$\mathcal{L}_{\text{sync}}$
2. **Co-Motion 课程学习**: 从顺序示范逐步过渡到并发示范，缓解可学习性下降
3. **多骨干验证**: 在 [[Diffusion Policy]]、[[ACT]] 等动作头范式上验证共享-残差分解的通用性

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（超参数、阶段划分均有报告）
- [x] 数据集可获取（RoboTwin 2.0 公开，真实数据为自收集）

---

## 关联笔记

### 基于
- [[π0]]: 预训练 VLA 骨干，Co-VLA 在其动作头上添加 SAE
- [[π0.5]]: 对比基线，Co-VLA Easy 成功率（82%）优于 π₀.₅（73%）

### 对比
- [[π0.5]]: 同为双臂 VLA，Co-VLA 在协调任务上显著领先
- [[RoboTwin]]: Co-Motion 在 RoboTwin 2.0 框架内实现

### 方法相关
- [[共享-残差分解]]: SAE 的核心设计思路
- [[FSDP]]: 分布式训练策略
- [[流匹配]]: π₀ 骨干的动作生成范式
- [[Action Chunking]]: 动作块预测

### 硬件/数据相关
- [[AgileX Cobot Magic]]: 真实实验平台（双臂机器人）
- [[RoboTwin]]: 仿真评估平台

---

## 速查卡片

> [!summary] Co-VLA (2026)
> - **核心**: 将双臂协调显式分解为共享+残差潜变量，辅以自适应部署控制器
> - **方法**: SAE（共享-残差动作头）+ LAC（自适应平滑控制器）+ Co-Motion（并发示范）
> - **结果**: 紧耦合任务成功率 +27%，OOD 性能翻倍（13%→27%），任务完成时间降低 25%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-19*
