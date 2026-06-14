---
title: "FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation"
method_name: "FAWAM"
authors: [Haotian He, Zeyu Yan, Qipeng Liu, Ning Guo, Wenzhao Lian]
year: 2026
venue: arXiv
tags: [world-action-model, force-aware, contact-rich-manipulation, residual-correction, closed-loop]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.08555
created: 2026-06-09
---

# 论文笔记：FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, Shanghai Jiao Tong University |
| 日期 | June 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[Pi0.5]], [[ForceVLA]], [[GE-Act]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.08555) / Code 未提供 |

---

## 一句话总结

> FAWAM 在感知、预测、闭环执行三个层级融入力/力矩信号，实现接触丰富操作任务的在线残差修正，成功率达 85%，超越纯视觉基线 36.25%。

---

## 核心贡献

1. **三层级力信号集成**: 在感知（编码历史力矩）、预测（联合预测动作与力矩轨迹）、执行（在线残差修正）三个层次全面融入 [[Force-Torque Sensing|力/力矩信号]]
2. **力预见动作模型（Force-Envisioned Action Model）**: 基于 [[World Action Model|世界动作模型]] 的两阶段训练框架，联合预测未来动作块与末端力矩轨迹，显式建模接触演化
3. **力引导残差修正器（Force-Guided Residual Corrector）**: 以预测力矩为参考基准，实时检测执行偏差并在线修正动作，实现真正的闭环控制

---

## 问题背景

### 要解决的问题

接触丰富的机器人操作任务（如擦白板、削黄瓜、旋转箱子、擦花瓶）需要精确的力控制，而现有 [[Vision-Language-Action Model|VLA]] 方法主要依赖视觉信息，无法有效利用 [[Force-Torque Sensing|力/力矩信号]] 来处理接触动态变化。

### 现有方法的局限

- **纯视觉方法**（[[Pi0.5]]、[[GE-Act]]）：完全忽略力传感器信息，在接触丰富任务中泛化能力差
- **现有力感知方法**（[[ForceVLA]]、TA-VLA）：仅在感知层次融入力信号（输入编码），缺乏对接触演化的建模与在线反馈修正能力
- 缺乏大规模同步力/视频数据集，限制了力信号融入视频生成模型

### 本文的动机

力/力矩信号是接触动态的直接反映。若能在预测时显式建模未来力矩变化，并将预测与实测之间的偏差作为在线修正信号，就能在不依赖高精度力控制器的情况下实现鲁棒的接触丰富操作。

---

## 方法详解

### 模型架构

FAWAM 采用**双组件分层控制**架构：

- **力预见动作模型（Force-Envisioned Action Model，1 Hz）**: 编码历史力矩历史，基于 [[World Action Model|世界动作模型]] 骨干联合生成动作块 $a_{t+1:t+H_p}^{base}$ 与预测力矩轨迹 $w_{t+1:t+H_p}^{pred}$
- **力引导残差修正器（Force-Guided Residual Corrector，10 Hz）**: 计算预测与实测力矩偏差 $e^w$，生成残差修正量 $\Delta a^{res}$ 并通过门控机制筛选是否激活修正

![Figure 2: FAWAM 整体架构](https://arxiv.org/html/2606.08555v1/x2.png)

两阶段双频率设计使粗粒度规划与细粒度力反应分离，兼顾计算效率与实时性。

### 核心模块

#### 模块1：接触特征编码（Contact Feature Encoding）

**设计动机**: 利用 [[Force-Torque Sensing|历史力矩序列]] 提取接触上下文特征，注入到 [[Adaptive Layer Normalization|AdaLN]] 调制参数中。

**具体实现**:
- 输入历史 $H_f$ 步的腕部力矩测量值 $w_{t-H_f+1:t} \in \mathbb{R}^{H_f \times 6}$
- 通过力编码器 $\text{Enc}_f$ 提取接触特征 $c_t$
- 通过 [[MLP]] 将 $c_t$ 转化为 [[Adaptive Layer Normalization|AdaLN-Zero]] 参数的偏移量，调制 Transformer 块中的归一化行为

#### 模块2：两阶段训练（Two-Stage Training）

**设计动机**: 分离视频生成预训练与动作-力矩联合微调，规避大规模同步力/视频数据稀缺的问题。

**具体实现**:
- **阶段一（视频生成预训练）**: 冻结部分参数，训练 [[Flow Matching|流匹配]] 视频预测头
- **阶段二（动作-力矩联合微调）**: 在阶段一基础上，联合优化动作预测损失 $\mathcal{L}_{action}$ 与力矩预测损失 $\mathcal{L}_{force}$，权重 $\lambda_F = 1$

#### 模块3：力引导残差修正器（Force-Guided Residual Corrector）

**设计动机**: 执行时利用预测-实测力矩偏差作为误差信号，在线修正基础动作，无需重新推理完整策略。

**具体实现**:
- 计算窗口内力矩跟踪误差 $e^w$
- 将误差、接触特征 $c_t$、基础动作 $a^{base}$、预测力矩 $w^{pred}$、关节状态 $q_t$ 拼接输入修正网络 $\text{Res}_\phi$
- 输出残差修正量 $\Delta a^{res}$ 与门控信号 $g_t^{res}$（阈值 $\tau = 0.8$）
- 门控二值化后乘以残差，叠加到基础动作上

---

## 关键公式

### 公式1: [[Force-Torque Sensing|接触特征编码]]

$$
c_t = \text{Enc}_f(w_{t-H_f+1:t})
$$

**含义**: 将历史 $H_f$ 步腕部力矩序列编码为紧凑的接触上下文特征向量。

**符号说明**:
- $c_t$: 时刻 $t$ 的接触特征向量
- $w_{t-H_f+1:t} \in \mathbb{R}^{H_f \times 6}$: 历史力矩测量序列（6 轴：3 力 + 3 力矩）
- $H_f$: 历史窗口长度

---

### 公式2: [[Adaptive Layer Normalization|力条件调制]]

$$
\bar{u}_t^{(i)} = u_t^{(i)} + \text{MLP}_f^{(i)}(c_t)
$$

**含义**: 将力接触特征通过 MLP 变换为偏移量，叠加到第 $i$ 个 Transformer 块的 AdaLN-Zero 参数上，实现力感知的层归一化调制。

**符号说明**:
- $u_t^{(i)}$: 第 $i$ 个块的原始 AdaLN-Zero 调制参数
- $\bar{u}_t^{(i)}$: 力条件调制后的参数
- $\text{MLP}_f^{(i)}$: 第 $i$ 个块对应的力特征映射网络

---

### 公式3: [[Flow Matching|阶段一视频生成损失]]

$$
\mathcal{L}_{\text{stage1}} = \mathbb{E}\left[\left\| v_\psi^O\!\left(\hat{O}_t^\alpha, \alpha, L_t, I_{t-K:t}\right) - \left(O_t - \varepsilon^O\right) \right\|_2^2\right]
$$

**含义**: 以流匹配目标训练视频预测头，预测未来观测帧 $O_t$ 的去噪速度场。

**符号说明**:
- $v_\psi^O$: 观测帧速度预测网络，参数 $\psi$
- $\hat{O}_t^\alpha$: 时间步 $\alpha$ 的含噪观测帧
- $\alpha$: 流匹配时间步（$\alpha \sim \mathcal{U}(0,1)$）
- $L_t$: 语言指令
- $I_{t-K:t}$: 过去 $K$ 帧图像上下文
- $\varepsilon^O$: 添加到观测帧的噪声

---

### 公式4: [[Flow Matching|阶段二动作-力矩联合损失]]

$$
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{action}} + \lambda_F \mathcal{L}_{\text{force}}
$$

$$
\mathcal{L}_{\text{action}} = \mathbb{E}\left[\left\| \hat{v}_t^A - (A_t - \varepsilon^A) \right\|_2^2\right]
$$

$$
\mathcal{L}_{\text{force}} = \mathbb{E}\left[\left\| \hat{v}_t^W - (W_t - \varepsilon^W) \right\|_2^2\right]
$$

**含义**: 联合训练动作块预测与力矩轨迹预测，$\lambda_F = 1$ 使两者权重相等。

**符号说明**:
- $\hat{v}_t^A$: 动作的预测速度场
- $A_t$: 真实动作块
- $\hat{v}_t^W$: 力矩的预测速度场
- $W_t$: 真实力矩轨迹
- $\varepsilon^A, \varepsilon^W$: 对应的添加噪声
- $\lambda_F = 1$: 力矩损失权重

---

### 公式5: [[Residual Policy Learning|力矩跟踪误差]]

$$
e_{t-H_e+1:t}^w = w_{t-H_e+1:t}^{\text{meas}} - w_{t-H_e+1:t}^{\text{pred}}
$$

**含义**: 在窗口 $H_e$ 内逐步计算实测力矩与预测力矩之差，作为执行偏差信号输入修正器。

**符号说明**:
- $w^{\text{meas}}$: 传感器实测力矩
- $w^{\text{pred}}$: 模型预测力矩
- $H_e$: 误差窗口长度

---

### 公式6: [[Residual Policy Learning|残差修正输出与门控]]

$$
\Delta a_{t+1:t+H_r}^{\text{res}},\ g_t^{\text{res}} = \text{Res}_\phi\!\left(e_{t-H_e+1:t}^w,\ c_t,\ a_{t+1:t+H_p}^{\text{base}},\ w_{t+1:t+H_p}^{\text{pred}},\ q_t\right)
$$

$$
a_{t+1:t+H_r}^{\text{cor}} = a_{t+1:t+H_r}^{\text{base}} + \kappa_t\, \Delta a_{t+1:t+H_r}^{\text{res}}
$$

**含义**: 修正器输出残差动作量与激活门控，门控二值化后（阈值 $\tau = 0.8$）决定是否将残差叠加到基础动作上。

**符号说明**:
- $\text{Res}_\phi$: 残差修正网络，参数 $\phi$
- $\Delta a^{\text{res}}$: 预测残差动作量
- $g_t^{\text{res}}$: 门控置信度分数
- $\kappa_t \in \{0, 1\}$: 二值化门控（$\kappa_t = 1$ 若 $g_t^{\text{res}} \geq \tau$）
- $a^{\text{cor}}$: 修正后动作
- $q_t$: 当前关节状态
- $H_r$: 残差修正水平线长度

---

### 公式7: [[Residual Policy Learning|残差修正器训练损失]]

$$
\mathcal{L}_{\text{res}} = \mathbb{E}\left[\left\| \Delta a^{\text{res}} - \Delta a^{\text{gt}} \right\|_2^2 + \lambda_{\text{gate}}\, \text{BCE}(g_t^{\text{res}},\ y_t)\right]
$$

**含义**: 监督残差量与 GT 偏差之差的 L2 损失，加上门控激活的二值交叉熵损失。

**符号说明**:
- $\Delta a^{\text{gt}}$: 人工干预提供的真实修正量（来自 DAgger 风格的 20 次人工干预演示）
- $y_t \in \{0, 1\}$: 是否需要修正的真实标签
- $\lambda_{\text{gate}} = 1$: 门控损失权重
- $\text{BCE}$: 二值交叉熵损失

---

## 关键图表

### Figure 1: 方法对比

![Figure 1: 力条件预测 vs. FAWAM 三层级集成](https://arxiv.org/html/2606.08555v1/x1.png)

**说明**: 左侧为仅使用力条件输入的方法（只在感知层集成力信号），右侧为 FAWAM 在感知、预测、执行三层级全面集成力信号的方法。擦花瓶任务示例显示 FAWAM 的在线修正能力。

---

### Figure 2: FAWAM 整体架构

![Figure 2: FAWAM 系统架构概览](https://arxiv.org/html/2606.08555v1/x2.png)

**说明**: 上方为 1 Hz 的力预见动作模型，输入历史图像、语言指令和力矩历史，输出未来动作块与力矩预测轨迹。下方为 10 Hz 的力引导残差修正器，实时计算预测-实测偏差并在线修正动作。

---

### Figure 3: 实验任务设置

![Figure 3: 四个接触丰富操作任务](https://arxiv.org/html/2606.08555v1/x3.png)

**说明**: 四个任务（擦白板、削黄瓜、旋转箱子、擦花瓶）在多样化接触条件下进行评估，每任务设计了不同的接触变化场景。

---

### Figure 4: 执行时间对比

![Figure 4: 各任务变体的完成时间比较](https://arxiv.org/html/2606.08555v1/x4.png)

**说明**: 完整 FAWAM 在所有任务上完成速度最快，表明在线力引导修正不但提升了成功率，还减少了不必要的调整动作，提升了执行效率。

---

### Figure 5: 扰动下的修正效果

![Figure 5: 有/无残差修正器的任务展开对比](https://arxiv.org/html/2606.08555v1/x5.png)

**说明**: 在人工施加扰动后，无修正器的策略失败退出，而带残差修正器的 FAWAM 能在线感知力矩偏差并成功恢复，展示闭环修正的鲁棒性。

---

### Figure 6: 实验环境

![Figure 6: 物理实验环境](https://arxiv.org/html/2606.08555v1/fig/experimental_setup.png)

**说明**: Franka Research 3 机器人臂，腕部安装 ATI Axia80-M8 6 轴力矩传感器，3 台 Intel RealSense D435i 相机，使用 FACTR 遥操作系统采集演示数据。

---

### Figure 15: 残差激活阈值分析

![Figure 15: 门控阈值 τ 对成功率的影响](https://arxiv.org/html/2606.08555v1/fig/tau_success_rate.png)

**说明**: 分析门控激活阈值 $\tau$ 对四个任务成功率的影响，最优阈值为 $\tau = 0.8$，过低阈值导致过度修正，过高阈值减少有效修正。

---

### Table 1: 主实验结果

| 方法 | 擦白板 | 削黄瓜 | 旋转箱子 | 擦花瓶 | 平均成功率 |
|------|--------|--------|----------|--------|-----------|
| [[Pi0.5]] | 9/20 | 8/20 | 10/20 | 11/20 | 47.50% |
| GE-Act | 9/20 | 10/20 | 9/20 | 11/20 | 48.75% |
| [[ForceVLA]] | 11/20 | 14/20 | 13/20 | 13/20 | 63.75% |
| TA-VLA | 10/20 | 17/20 | 14/20 | 10/20 | 63.75% |
| FAWAM w/o Res | 14/20 | 15/20 | 16/20 | 14/20 | 73.75% |
| **FAWAM** | **15/20** | **19/20** | **17/20** | **17/20** | **85.00%** |

**关键发现**: FAWAM 达到 85% 平均成功率，比纯视觉基线高 36.25%，比现有力感知方法高 21.25%；削黄瓜任务接近完美（19/20）。

---

### Table 2: 消融实验

| 力观测（Obs）| 力预测（Pre）| 残差修正（Res）| 擦白板 | 削黄瓜 | 旋转箱子 | 擦花瓶 | 平均成功率 |
|:---:|:---:|:---:|--------|--------|----------|--------|-----------|
| - | - | - | 9/20 | 10/20 | 9/20 | 11/20 | 48.75% |
| ✓ | - | - | 12/20 | 12/20 | 9/20 | 12/20 | 56.25% |
| ✓ | ✓ | - | 14/20 | 15/20 | 16/20 | 14/20 | 73.75% |
| ✓ | - | ✓ | 12/20 | 15/20 | 12/20 | 15/20 | 67.50% |
| ✓ | ✓ | ✓ | 15/20 | 19/20 | 17/20 | 17/20 | **85.00%** |

**关键发现**: 力观测单独贡献 +7.5%；联合力预测额外贡献 +17.5%（最大单项贡献）；残差修正额外贡献 +18.75%；三者协同效果最优，证明各层级缺一不可。

---

### Table 3: 扰动鲁棒性

| 策略 | 擦白板 | 削黄瓜 | 切黄瓜 | 擦花瓶 |
|------|--------|--------|--------|--------|
| 无残差修正器 | 0/5 | 2/5 | 0/5 | 1/5 |
| **带残差修正器** | **3/5** | **5/5** | **4/5** | **4/5** |

**关键发现**: 在人工施加扰动时，残差修正器将平均成功率从 15% 提升至 80%，证明其在非预期接触条件下的强鲁棒性。

---

## 实验

### 数据集

| 数据来源 | 规模 | 特点 | 用途 |
|----------|------|------|------|
| 人工演示（FACTR 遥操作） | 90 次 | 4 任务 × 多样接触条件 | 训练基础动作-力矩模型 |
| 人工干预演示（DAgger 风格） | 20 次 | 含失败恢复序列 | 训练残差修正器 |
| 评估 | 20 次 / 任务 | 60 秒时限 | 测试（每条件共 80 次） |

### 实现细节

- **骨干网络**: 基于 [[World Action Model]] 的视频-动作联合预测框架（π₀ 架构）
- **优化器**: AdamW（$\beta_1=0.9$，$\beta_2=0.95$，$\varepsilon=10^{-8}$）
- **精度**: bfloat16 混合精度
- **硬件**: 8× NVIDIA A100 GPU，Batch Size 64
- **阶段一**: 30,000 步，学习率 $3\times10^{-5}$
- **阶段二**: 30,000 步，学习率 $1.5\times10^{-4}$，$\lambda_F = 1$
- **残差修正器**: 500 epochs，Batch Size 128（人工干预/非干预均衡采样），学习率 $3\times10^{-4}$，$\lambda_{\text{gate}} = 1$
- **控制频率**: 动作模型 1 Hz，残差修正器 10 Hz

### 硬件平台

- 机器人：Franka Research 3（7-DoF）
- 力矩传感器：ATI Axia80-M8 6 轴（120 Hz）
- 相机：3× Intel RealSense D435i（30 Hz）
- 遥操作：[[FACTR]] 主从系统（带力反馈）

---

## 批判性思考

### 优点

1. **三层级一体化设计**: 同时解决感知、预测和执行三个层面的力信息利用问题，相较于只在输入层加入力信号的方法有质的提升
2. **轻量化闭环修正**: 残差修正器以 10 Hz 高频运行，仅需 20 次人工干预演示即可训练，数据效率高
3. **门控机制**: 通过置信度门控避免不必要的修正干预，防止过度修正带来的振荡

### 局限性

1. **力信号未融入视频生成模型**: 受限于大规模同步力/视频数据集的缺乏，力信号没有影响视觉预测，理论上会限制接触预见能力的上限
2. **干预数据覆盖有限**: 仅 20 次人工干预演示可能无法覆盖真实部署中的多样失败模式
3. **末端执行器力矩是间接信号**: 预测腕部力矩而非触觉传感器信号，对精细指尖接触建模不够精确

### 潜在改进方向

1. 收集大规模同步力/视频数据，将力信号融入视频生成预训练阶段
2. 引入指尖触觉传感器（如 GelSight）替代腕部力矩，获取更直接的接触信息
3. 探索自动化的干预数据收集（减少人工标注依赖）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（附录中有详细超参数）
- [ ] 数据集可获取（需自行采集）

---

## 关联笔记

### 基于

- [[World Action Model]]: FAWAM 在 WAM 框架上扩展力感知能力
- [[Pi0.5]]: 基础视觉-动作模型，同时作为对比基线
- [[Flow Matching]]: 动作与力矩预测的生成框架

### 对比

- [[ForceVLA]]: 同为力感知 VLA，但仅在输入层集成力信号，无预测与闭环修正
- [[Pi0.5]]: 纯视觉 VLA 基线，成功率低 36.25%
- [[GE-Act]]: 视觉世界模型方法，无力信号

### 方法相关

- [[Force-Torque Sensing]]: 核心传感器模态
- [[Residual Policy Learning]]: 残差修正器的学习范式
- [[Adaptive Layer Normalization]]: 力条件调制机制
- [[Action Chunking]]: 动作块预测方式
- [[DAgger]]: 残差修正器的训练数据收集策略

### 硬件/数据相关

- [[Franka Panda]]: 实验平台（Franka Research 3）
- [[遥操作]]: FACTR 系统用于数据采集
- [[FACTR]]: 带力反馈的遥操作系统

---

## 速查卡片

> [!summary] FAWAM: Force-Aware World Action Models
> - **核心**: 在感知/预测/执行三层级融入力/力矩信号，实现接触丰富操作的在线闭环修正
> - **方法**: 力预见动作模型（联合预测动作+力矩）+ 力引导残差修正器（实时偏差修正）
> - **结果**: 85% 平均成功率，超纯视觉基线 36.25%，超现有力感知方法 21.25%
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-09*
