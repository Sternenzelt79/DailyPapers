---
title: "ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining"
method_name: "ACE-Ego-0"
authors: [Hao Li, Ganlong Zhao, Yufei Liu, Haotian Hou, Guoquan Ye, Tongyan Fang, Chunxiao Liu, Siyuan Huang, Jianbo Liu, Xiaogang Wang, Hongsheng Li]
year: 2026
venue: arXiv
tags: [vla, cross-embodiment, egocentric-video, flow-matching, bimanual-manipulation, pretraining, pseudo-action]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17200
created: 2026-06-17
---

# 论文笔记：ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CUHK MMLab, AgiBot, 上海人工智能实验室 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[Pi0.5]], [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17200) / Code: — |

---

## 一句话总结

> ACE-Ego-0 通过统一的相机空间动作表示、形态学条件化和可靠性感知训练目标，将 1.48K 小时的第一人称人类视频与 4.5K 小时机器人数据联合用于 VLA 预训练，在多个仿真与真实机器人基准上达到最优性能。

---

## 核心贡献

1. **统一动作表示（Camera-Space Canonical Action）**: 将人类手部运动与机器人末端执行器动作统一到头部相机坐标系，消除跨具身坐标系差异
2. **跨具身形态学条件化（Cross-Embodiment Morphology Conditioning）**: 使用 [[URDF]] 图编码器（机器人）与可学习代理嵌入（人类）对不同形态结构建模，并注入 Action Expert
3. **可靠性感知训练目标（Reliability-Aware Training）**: 层次化权重区分高保真机器人数据与嘈杂伪动作，防止伪标签噪声污染策略

---

## 问题背景

### 要解决的问题

[[VLA]] 模型的训练依赖大规模多样化具身数据，但机器人轨迹采集成本高昂。第一人称（[[Egocentric Video|自视角]]）人类视频提供了丰富的现实世界操作先验，但与机器人数据联合训练存在三类异质性：
- **空间异质性**：坐标系不对齐
- **结构异质性**：人类手部与机器人末端执行器的运动学结构差异
- **时域异质性**：控制频率差异导致固定动作块覆盖不同物理时长

### 现有方法的局限

- 共享动作格式（RT-1、Octo）不保证坐标系对齐
- 固定长度 [[Action Chunking]] 在不同控制频率下跨越不同物理时长
- 运动学结构通常被隐式吸收而非显式建模
- 从视觉流水线生成的[[伪动作]]（Pseudo-Action）往往被当作高精度传感器数据处理，将跟踪抖动等噪声传播到策略中

### 本文的动机

通过系统性地同时解决三个维度的异质性，并设计可靠性感知损失区分数据质量，使 1.48K 小时人类视频能够真正有效地增强 VLA 预训练。

---

## 方法详解

### 模型架构

![Figure 2: ACE-Ego-0 Architecture](https://arxiv.org/html/2606.17200v1/x2.png)

ACE-Ego-0 采用**视觉-语言主干 + Action Expert DiT**架构：
- **输入**: 多视角图像（256×256）+ 语言指令
- **Backbone**: [[VLM|Qwen3-VL-4B-Instruct]]（~4B 参数）
- **Action Expert**: [[Flow Matching]] [[DiT]]（36 层, hidden=1024, heads=16, ~600M 参数）
- **形态学令牌**: [[URDF]] 图编码器（机器人）/ 可学习代理嵌入（人类）→ 注入 Action Expert
- **输出**: 22D 双臂动作块 $a_{t:t+H_d}$
- **总参数**: ~4.6B（主干 4B + Action Expert 600M）

### 核心模块

#### 模块1: 相机空间统一动作表示（Canonical Action Space）

**设计动机**: 利用[[跨具身学习|跨具身]]表示，在同一坐标系下对齐机器人末端执行器和人类手腕运动

**机器人动作转换**:

$$
p_{cam} = R_{cam \leftarrow s} p_s + t_{cam \leftarrow s}, \quad R_{cam,ee} = R_{cam \leftarrow s} R_{s,ee}
$$

将末端执行器位姿从源坐标系变换到头部相机坐标系。

**人类等效末端执行器**:
- 手腕关节作为原点（重建最稳定的关节）
- 手掌平面 + 手腕-手指向量构造手中心坐标系 $R \in SO(3)$
- 使用[[6D旋转表示]]（6D rotation representation）避免四元数不连续性
- 拇指到手掌距离归一化为机器人夹爪行程代理

**人类手中心坐标系构造**:

$$
x = \frac{p_{palm} - p_{wrist}}{\|p_{palm} - p_{wrist}\|_2}, \quad z = \hat{n}(p_{wrist}, p_{thumb}, p_{middle}), \quad y = z \times x
$$

**统一 22D 动作格式**（双臂）:

$$
a = [a_{left}; a_{right}] \in \mathbb{R}^{22}, \quad a_{arm} = [p_x, p_y, p_z \;|\; r_1,...,r_6 \;|\; g \;|\; \alpha]
$$

其中 3D 位置 + 6D 旋转 + 1D 夹爪 + 1D 活动标志，每臂 11D。

#### 模块2: 跨具身形态学条件化（Morphology Conditioning）

**设计动机**: 显式建模不同运动学结构，使同一 Action Expert 服务于多种具身形态

**两条并行路径**:

$$
h_{morph} = \begin{cases} P_{morph}(E_{urdf}(\mathcal{G}_r)), & \text{机器人数据} \\ P_{surr}(e_d), & \text{人类数据} \end{cases}
$$

- **机器人路径**: [[URDF]] 图编码器，29D 节点特征，消息传递聚合体态与链式特征
- **人类路径**: 每数据集可学习代理嵌入 $e_d$，捕捉相机位置、视觉领域、标注质量等数据集特定先验

**图编码器消息传递**（$L$ 层残差）:

$$
H^{(0)} = \phi_{in}(X_r), \quad H^{(\ell+1)} = H^{(\ell)} + \phi_\ell([H^{(\ell)}; \bar{A}_r H^{(\ell)}])
$$

$$
\bar{A}_r = D_r^{-1}(A_r + I) \quad \text{（归一化邻接矩阵）}
$$

**池化**:

$$
z_{body}^r = \rho_{body}\!\left(\text{mean}(H_j^{(L)}, j \in \mathcal{J}_r)\right), \quad z_{chain}^r = \rho_{chain}([\text{mean}(C_L); \text{mean}(C_R)])
$$

形态学令牌从视觉-语言主干中隔离，仅在 Action Expert 解码阶段注入。

#### 模块3: 时间对齐动作分块（Time-Aligned Action Chunking）

**问题**: 不同数据集控制频率不同（如 10Hz vs 50Hz），固定步数的动作块覆盖不同物理时长

**解决方案**: 以目标物理时长 $T^* = 2s$ 归一化分块大小：

$$
H_d = \text{round}(f_d \cdot T^*)
$$

**批次采样策略**: 按任务类别 $c_{task}$、归一化幕阶段 $\varphi = \text{clip}((t + H_d/2)/L_e, 0, 1)$ 和时域桶 $b_H$ 组成复合键 $k = (c_{task}, b_\varphi, b_H)$ 进行采样，保证跨数据集的时域分布均衡。

---

## 关键公式

### 公式1: [[Flow Matching|机器人主损失（流匹配）]]

$$
\mathcal{L}_{action} = \mathbb{E}_{s,\epsilon} \sum_{t,j} M_{t,j} \|\hat{v}_\theta(a_s, s)_{t,j} - (a - \epsilon)_{t,j}\|^2
$$

**含义**: 对高保真机器人数据用标准流匹配损失，预测流速场使噪声 $\epsilon$ 向干净动作 $a$ 流动

**符号说明**:
- $a_s = sa + (1-s)\epsilon$: 流插值，$s \sim \mathcal{U}(0,1)$
- $\hat{v}_\theta$: Action Expert 预测的流速场
- $M_{t,j}$: 有效步骤掩码（排除 padding）
- $a$: 干净动作；$\epsilon \sim \mathcal{N}(0, I)$

### 公式2: [[伪动作|人类辅助损失（可靠性加权 Huber 回归）]]

$$
\mathcal{L}_{haux} = \mathbb{E}_{s,\epsilon} \frac{1}{Z} \sum_{t,j} M_{t,j} W_{t,j} \cdot \text{Huber}_\beta\!\left(\hat{v}_\theta(a_s, s)_{t,j} - (\tilde{a} - \epsilon)_{t,j}\right)
$$

**含义**: 对嘈杂伪动作用可靠性权重 $W_{t,j}$ 降权处理，Huber 损失对异常值鲁棒

**符号说明**:
- $\tilde{a}$: 时域平滑后的人类目标动作
- $Z = \sum_{t,j} M_{t,j} W_{t,j}$: 归一化因子
- $W_{t,j} = \rho_j \cdot w_{t,j}$: 层次化可靠性权重

### 公式3: [[Reliability-Aware Training|层次化可靠性权重]]

$$
W_{t,j} = \rho_j \cdot w_{data}(d, h(j)) \cdot w_{step}(t, h(j))
$$

**含义**: 三层分解：通道级静态先验 × 数据集级权重 × 步骤级平滑权重

**符号说明**:
- $\rho_j \in [0,1]$: 通道级静态先验（位置通道=1.0，旋转/夹爪=0.001）
- $w_{data}$: 数据集/手部级权重
- $w_{step}$: 步骤级平滑权重，基于跳变和抖动检测

### 公式4: [[Reliability-Aware Training|步骤级平滑权重]]

$$
q_{t,h} = \max\!\left(\frac{\Delta p_t^h}{\tau_{jump}}, \frac{\Delta^2 p_t^h}{\tau_{jerk}}\right)
$$

$$
w_{step}(t,h) = \begin{cases} 1, & q_{t,h} \leq 1 \\ \max\{w_{min}, \exp[-\alpha(q_{t,h}-1)]\}, & q_{t,h} > 1 \end{cases}
$$

**含义**: 检测轨迹中的跳变和抖动（加速度异常），超出阈值的步骤指数衰减权重

**符号说明**:
- $\Delta p_t^h = \|p_t^h - p_{t-1}^h\|_2$: 逐帧位移
- $\Delta^2 p_t^h = \|p_{t+1}^h - 2p_t^h + p_{t-1}^h\|_2$: 离散加速度（抖动）
- $\tau_{jump}, \tau_{jerk}$: 每数据集/手部的第95百分位阈值

### 公式5: [[Flow Matching|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{action} + \lambda_{haux} \mathcal{L}_{haux}, \quad \lambda_{haux} = 0.1
$$

**含义**: 机器人流匹配损失为主，人类辅助损失为正则，权重 0.1 防止伪动作噪声主导训练

### 公式6: [[相机空间动作|轨迹平滑损失]]

$$
\mathcal{L}_{smooth} = \mathcal{L}_{reproj} + \lambda_{tv} \sum_t \|t^{global}_{t+1} - 2t^{global}_t + t^{global}_{t-1}\|_2^2
$$

**含义**: 3D 手部轨迹全局优化中的平滑项，抑制重建轨迹的抖动

**符号说明**:
- $\mathcal{L}_{reproj}$: 重投影误差损失
- $t^{global}_t$: 第 $t$ 帧的全局位置
- $\lambda_{tv}$: 全变分平滑系数

### 公式7: [[相机空间动作|相机空间推理转换]]

$$
\hat{p}_s = R_{cam \leftarrow s}^T (\hat{p}_{cam} - t_{cam \leftarrow s}), \quad \hat{R}_{s,ee} = R_{cam \leftarrow s}^T \hat{R}_{cam,ee}
$$

**含义**: 推理时将预测的相机空间动作转回机器人本体坐标系，只需标准外参矩阵

---

## 关键图表

### Figure 1: 系统概览与数据构成

![Figure 1: ACE-Ego-0 Overview](https://arxiv.org/html/2606.17200v1/x1.png)

**说明**: ACE-Ego-0 在 6K+ 小时混合具身数据上预训练统一 VLA 策略，包含大规模第一人称人类视频、多具身机器人演示和仿真数据。展示空间/结构/时域对齐流水线。

### Figure 2: 模型架构

![Figure 2: ACE-Ego-0 Architecture](https://arxiv.org/html/2606.17200v1/x2.png)

**说明**: 视觉-语言主干处理多视角图像和语言指令，Action Expert 接收共享表示和形态学令牌。机器人数据使用流匹配主损失，人类数据使用可靠性加权 Huber 辅助损失。

### Figure 3: 数据处理流水线概览

![Figure 3: Data Processing Pipeline Overview](https://arxiv.org/html/2606.17200v1/x3.png)

**说明**: 将大规模第一人称人类视频转化为训练就绪的具身操作数据的五阶段流水线概览：数据集策划 → 视频选择 → 3D 手部重建 → 动作参数化 → 质量控制。

### Figure 4: 自视角视频转伪动作流水线

![Figure 4: Egocentric Video to Pseudo-Action Pipeline](https://arxiv.org/html/2606.17200v1/x4.png)

**说明**: 详细展示五阶段处理流程，包括视频选择过滤器、SAM3 手部跟踪、HaMeR MANO 姿态估计、全局轨迹优化（含重投影损失 + 平滑正则）和 VIPE 相机位姿估计。

### Figure 5: 真实机器人评估与消融实验

![Figure 5: Real-Robot Results and Ablation](https://arxiv.org/html/2606.17200v1/x5.png)
![Figure 5b: Component Ablation](https://arxiv.org/html/2606.17200v1/x6.png)

**说明**: (a) 六项真实 ARX 双臂任务对比，ACE-Ego-0 平均 78.3% 对比 π₀.₅ 71.7%；(b) 模块消融显示可靠性感知损失贡献最大（−3.6%）。

### Figure 7: 相机空间动作可视化

![Figure 7: Camera-Space Action Visualization](https://arxiv.org/html/2606.17200v1/x8.png)

**说明**: 展示真实机器人演示、仿真数据和人类第一人称视频在对齐坐标系中的统一动作表示。

### Figure 8: 真实机器人定性展示

![Figure 8: Qualitative Rollout Sequences](https://arxiv.org/html/2606.17200v1/x9.png)

**说明**: ACE-Ego-0 在真实 ARX 双臂平台上的操作序列，展示接触丰富的双臂协调（冲咖啡）和长时序任务（装鞋子）。

### Table 1: RoboCasa GR1 TableTop 仿真结果

| 方法 | 成功率 (%) |
|------|------------|
| **ACE-Ego-0** | **72.8** |
| DIAL | 70.2 |
| JoyAI-RA | 63.2 |
| ABot-M0 | 58.3 |
| FLARE | 55.0 |
| GR00T-N1.6 | 47.6 |
| Qwen3PI | 43.9 |

**说明**: ACE-Ego-0 在 24 项人形桌面操作任务中全面领先，包括关节物体操作和拾取放置两大类别。

### Table 2: RoboTwin 2.0 仿真结果（50 任务）

| 方法 | Easy (%) | Hard (%) |
|------|----------|----------|
| **ACE-Ego-0** | **91.12** | **90.62** |
| Hy-VLA | 90.9 | 90.1 |
| JoyAI-RA | 90.48 | 89.28 |
| LingBot-VLA | 88.56 | 86.68 |
| Motus | 88.66 | 87.02 |
| ABot-M0 | 86.06 | 85.08 |
| [[Pi0.5\|π₀.₅]] | 82.74 | 76.76 |

**说明**: 双臂仿真基准上以 Easy 91.12% / Hard 90.62% 达到最优，Hard 难度提升显著。

### Table 3: ARX 双臂真实机器人评估

| 任务 | ACE-Ego-0 (%) | π₀.₅ (%) | GR00T-N1.7 (%) |
|------|---------------|----------|-----------------|
| Pick Tea | 96.7 | 93.3 | 20.0 |
| Scoop Coffee | 86.7 | 70.0 | 36.7 |
| Category Sorting | 90.0 | 80.0 | 83.3 |
| Sweep Cubes | 76.7 | 66.7 | 6.7 |
| Stack Bowls | 80.0 | 73.3 | 73.3 |
| Pack Shoes | 53.3 | 56.7 | 20.0 |
| **平均** | **78.3** | **71.7** | **35.6** |

**说明**: 真实机器人上 ACE-Ego-0 在接触丰富任务（Scoop Coffee: +16.7%）和精细任务（Sweep Cubes: +10.0%）上优势明显，Pack Shoes 略逊于 π₀.₅。

### Table 4: 模块消融实验（RoboCasa GR1 TableTop）

| 配置 | 成功率 (%) | 变化 |
|------|------------|------|
| **Full ACE-Ego-0** | **72.8** | — |
| − 形态学令牌 | 70.9 | −1.9 |
| − 时间对齐分块 | 71.7 | −1.1 |
| − 可靠性感知损失 | 69.2 | −3.6 |

**关键发现**: 可靠性感知人类辅助损失贡献最大（−3.6%），验证伪动作噪声对策略训练的负面影响；形态学条件化次之（−1.9%）。

### Table 5: 数据源消融实验

| 配置 | 成功率 (%) | 变化 |
|------|------------|------|
| Robot + Human | 72.8 | — |
| Robot Only | 68.3 | −4.5 |
| Qwen（无具身数据） | 65.4 | −7.4 |

**关键发现**: 人类视频贡献最大单项增益（+4.5%），证明有效融合人类视频对 VLA 预训练的实质价值。

### Table 6: 人类数据辅助微调（Sweep Cubes，34 条机器人演示）

| 配置 | 成功率 (%) |
|------|------------|
| Robot-only 微调 | 10 |
| Robot + 419 人类片段 | **40** |

**关键发现**: 419 条人类视频片段覆盖 4.8× 更广工作空间（0.296 m² vs 0.062 m²），成功率提升 4 倍，验证人类数据在数据稀缺场景的价值。

---

## 自视角视频五阶段处理流水线

### Stage 1: 数据集策划
- 条件：第一人称视角、多样真实场景、高质量字幕
- 片段时长：4~30 秒

### Stage 2: 视频选择
- **自我交互过滤**：移除高人脸检测置信度（>0.5）片段
- **字幕过滤**：要求含操作动词 + 可操作物体名词

### Stage 3: 3D 手部重建
- SAM3 追踪（关键点置信度 $\tau_{kp}=0.4$，最短轨迹 $\ell_{min}=15$ 帧）
- [[MANO]] 姿态/形状估计（HaMeR）
- 全局轨迹优化（30 次根节点拟合 + 200 次平滑拟合迭代）
- VIPE 相机位姿估计实现世界坐标系一致性

### Stage 4: 动作参数化
- 磁盘存储：16D 双臂（3D 位置 + 3D 欧拉角 + 1D 夹爪 + 1D 活动标志，每臂）
- 训练时：转换为 22D（连续 6D 旋转表示）
- 夹爪归一化：[0.04, 0.10] m 行程范围

### Stage 5: 质量控制
- **完整性**：无 NaN/Inf，帧连续，四元数范数 $\tau_{quat}=10^{-3}$
- **静态过滤**：运动能量低于数据集特定阈值
- **峰值过滤**：帧间变化超过 $\kappa_{spike} \cdot \sigma = 3\sigma$ 且占比 >5% 的片段剔除
- **双臂过滤**：双手间距异常或弱相关的片段剔除

---

## 实验

### 数据集

| 数据集 | 规模 | 类型 | 说明 |
|--------|------|------|------|
| AgiBot Alpha/Beta | 1,937.8 h | 机器人 | 多具身真实演示 |
| Galaxea R1Lite | 488.1 h | 机器人 | 双臂机器人 |
| AgiBot DigitalWorld | 225.3 h | 仿真 | — |
| RoboCasa Tabletop | 83.6 h | 仿真 | GR1 人形桌面 |
| Galbot 自采 | 1,800+ h | 机器人 | — |
| [[Ego4D]] | 216.6 h | 人类视频 | 第一人称 |
| EgoExo4D | 10.3 h | 人类视频 | 第一人称+外视角 |
| EPIC-KITCHENS-100 | 32.3 h | 人类视频 | 厨房操作 |
| HOI4D | 7.2 h | 人类视频 | 手-物交互 |
| EgoDex | 776.8 h | 人类视频 | 精细操作 |
| Xperience-10M | 435.7 h | 人类视频 | 大规模 |
| **总计** | **6,013.7+ h** | — | 机器人 4,534.8h + 人类 1,478.9h |

### 实现细节

- **VLM Backbone**: Qwen3-VL-4B-Instruct
- **Action Expert**: Flow-matching DiT，36 层，hidden=1024，16 头
- **图像分辨率**: 256×256
- **推理步数**: 4
- **预训练**: 128× A800 (80GB)，200K steps，AdamW + cosine schedule
- **微调**: 16× A800
- **学习率**: VLM 2e-5，Action Expert 1e-4
- **Batch Size**: 8 per device
- **梯度裁剪**: 1.0
- **目标物理时长**: $T^* = 2s$

### 基准

- **RoboCasa GR1 TableTop**: 24 任务，GR1 人形桌面操作
- **RoboTwin 2.0**: 50 任务，双臂 Easy/Hard 两个难度
- **ARX 双臂真实平台**: 6 项真实操作任务，每任务 30 次评估

---

## 批判性思考

### 优点

1. **系统性解决三维异质性**: 首次同时处理空间/结构/时域三类跨具身差异，而非单独处理
2. **可靠性感知设计务实有效**: 层次化权重（静态通道先验 + 动态步骤平滑）消融显示贡献最大（−3.6%），设计简单但有效
3. **数据规模优势**: 6K+ 小时混合数据为同类工作中规模最大之一，且详细公开了数据构成

### 局限性

1. **评估场景局限**: 仅评估桌面操作，移动/全身/可变形物体场景未验证
2. **旋转与手指动作保真度低**: 伪动作中旋转和手指运动的重建质量有限，通道级权重 $\rho_j=0.001$ 实际上近乎放弃旋转监督
3. **无力/触觉感知**: 预训练数据缺乏力矩传感，接触丰富任务依赖视觉推断
4. **长时序漂移**: 为 VLA 架构的共性问题，本文未专门解决

### 潜在改进方向

1. **更精细的伪动作旋转重建**: 引入更强的手部姿态先验提升旋转监督质量
2. **扩展到移动操作和双足机器人**: 相机空间表示理论上可推广但需验证
3. **力觉感知集成**: 结合触觉数据进一步提升接触丰富任务性能

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文附录详尽）
- [ ] 数据集可获取（部分公开数据集 + 私有数据）

---

## 关联笔记

### 基于

- [[Pi0.5]]: 流匹配 VLA 基线，ACE-Ego-0 的主要对比对象
- [[GR00T N1.5]]: NVIDIA 的跨具身 VLA，真实机器人对比基线
- [[Ego4D]]: 主要人类视频来源数据集之一
- [[MANO]]: 手部 3D 重建使用的参数化模型

### 对比

- [[Pi0.5]]: 真实机器人平均 +6.6%，双臂协调任务优势明显
- [[GR00T N1.5]]: 真实机器人平均 +42.7%，GR00T 在精细双臂任务表现欠佳

### 方法相关

- [[Flow Matching]]: Action Expert 使用的核心生成方法
- [[Action Chunking]]: 时间对齐动作分块的改进基础
- [[跨具身学习]]: 形态学条件化所属技术范畴
- [[DiT]]: Action Expert 的架构选择
- [[URDF]]: 机器人运动学图编码的输入格式
- [[6D旋转表示]]: 解决旋转空间连续性的关键技术
- [[伪动作]]: 人类视频转动作标签的核心概念
- [[Egocentric Video]]: 第一人称视频数据的基础概念

### 数据相关

- [[Ego4D]]: 216.6 小时第一人称数据
- [[MANO]]: 3D 手部重建参数化模型

---

## 速查卡片

> [!summary] ACE-Ego-0
> - **核心**: 统一第一人称人类视频与机器人数据的 VLA 预训练框架
> - **方法**: 相机空间统一表示 + URDF 形态学条件化 + 可靠性加权损失 + 5 阶段伪动作流水线
> - **结果**: RoboCasa 72.8% SOTA / RoboTwin Easy 91.12% / 真实 ARX 双臂 78.3%（+6.6% vs π₀.₅）
> - **数据**: 6K+ 小时（4.5K 机器人 + 1.5K 人类视频伪动作）
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-17*
