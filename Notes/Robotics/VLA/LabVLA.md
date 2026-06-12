---
title: "LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories"
method_name: "LabVLA"
authors: [Baochang Ren, Xinjie Liu, Xi Chen, Yanshuo Liu, Chenxi Li, Daqi Gao, Zeqin Su, Jintao Xing, Zirui Xue, Rui Li, Xiangyu Zhao, Shuofei Qiao, Minting Pan, Wangmeng Zuo, Lei Bai, Dongzhan Zhou, Ningyu Zhang, Huajun Chen]
year: 2026
venue: arXiv
tags: [vla, laboratory-automation, robot-manipulation, flow-matching, simulation-to-real]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2606.13578
created: 2026-06-12
---

# 论文笔记：LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University, Shanghai AI Lab, Harbin Institute of Technology |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[π0]], [[X-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13578) / Code — |

---

## 一句话总结

> LabVLA 是首个专为科学实验室协议执行设计的 VLA 系统，通过 RoboGenesis 仿真引擎生成实验室专属训练数据，结合 FAST 动作预训练与流匹配微调，在 LabUtopia 基准上取得最优成绩。

---

## 核心贡献

1. **RoboGenesis**: 支持 16 种机器人平台的可编程实验室仿真引擎，自动生成资产、场景和任务，构建 2,947 个标注资产的 LabAssetLibrary 以及 10,000 个实验室场景
2. **LabVLA 训练方案**: 两阶段训练——基于 [[FAST]] 动作 token 的 VLM 预训练 + 带 Knowledge Insulation 的 [[Flow Matching]] 后训练，骨干网络为 Qwen3-VL-4B + 18 层 [[DiT]] 动作专家
3. **LabUtopia 基准**: 覆盖 6 种实验室操作系列（取物、按压、开门、倒液、加热、转移）的同/异分布评测体系，建立了从 Apprentice 到 Scientist 的四级能力金字塔

---

## 问题背景

### 要解决的问题

现有 [[VLA|视觉-语言-动作模型]] 主要在家庭场景任务上训练，无法直接迁移到科学实验室的精细操作场景——例如移液管吸液、液体转移、加热操作等，这些任务具有更严格的精度要求、物理状态变化以及多机器人平台约束。

### 现有方法的局限

- 通用 VLA 模型（如 [[π0]]、[[OpenVLA]]）缺乏实验室领域的专属训练数据
- 现有仿真引擎（IsaacLab、RoboSuite 等）不支持实验室协议的自动化任务生成
- 实验室操作失败模式（如移液吸空、倒液溢出）与日常操作截然不同，现有方案无法捕获

### 本文的动机

数据瓶颈和机器人平台多样性是实验室自动化的核心障碍。通过 RoboGenesis 生成大量领域专属仿真数据，并设计针对实验室场景的两阶段训练方案（先预训练动作 token 理解，再通过 [[Flow Matching]] 精细化运动生成），可以有效跨越 sim-to-real 鸿沟。

---

## 方法详解

### 模型架构

![[LabVLA_fig1_overview.jpeg]]

**说明**: LabVLA 框架全貌。中心：LabVLA 策略将预训练 VLM（Qwen3-VL-4B）与 DiT 动作专家耦合；左侧：RoboGenesis 数据合成流水线；右侧：LabUtopia 六类任务系列。

LabVLA 采用 **VLM + DiT 动作专家** 双模块架构：

- **输入**: 语言指令 $l$（实验室协议步骤）+ 多视角观测 $o_t$ + 关节状态 $s_t$
- **VLM Backbone**: [[Qwen3-VL]]-4B-Instruct（隐层维度 2560）
- **动作专家**: 18 层 [[DiT]]（宽度 1024，8 个注意力头），通过 [[Cross-Attention]] 与 VLM 隐状态交互
- **输出**: 连续动作块 $a_{t:t+k}$（经 K 步预测）
- **总参数**: ~4B（VLM）+ DiT 额外参数

### 核心模块

#### 模块 1: RoboGenesis 仿真引擎

**设计动机**: 利用 [[领域随机化|Domain Randomization]] 和 Agent 辅助工作流生成克服实验室数据稀缺问题。

**具体实现**:
- **资产流水线**: 文本提示 → 产品摄影图像 → [[TRELLIS]] 2.0 三维重建 → 带物理属性的 USD 格式，生成 LabAssetLibrary（2,947 个标注资产）
- **场景构建**: 种子驱动布局算法，生成 10,000 个实验室场景，校验工作台间距、机器人通道宽度、无重叠放置
- **工作流生成**: 原子技能库（pick/place/pour/stir/shake/press/open/close/navigate），Agent 辅助工作流编写 + 离线验证
- **随机化六轴**: 场景、视觉杂乱度、摄像机、物体、光照、空间位置
- **15 类标注**: 机器人状态、相机内外参、步骤时序、物体状态、碰撞事件等，含技能级成功过滤器

#### 模块 2: FAST 动作 Token 预训练

**设计动机**: 通过 [[FAST|Frequency-space Action Sequence Tokenization]] 将连续动作序列压缩为离散 token，使 [[VLM]] 能在语言建模框架内理解动作语义。

**具体实现**:
- 将动作序列通过 DCT 变换编码为离散 FAST token 序列 $z_{1:L_z}$
- VLM 以自回归方式预测 token，掩码 $m_i$ 过滤填充位置
- 实现动作知识的语言级别表示，为后续 [[Flow Matching]] 微调提供良好初始化

#### 模块 3: Flow Matching 后训练 + Knowledge Insulation

**设计动机**: [[Flow Matching]] 生成精确连续轨迹；[[Knowledge Insulation]] 防止 Flow Matching 梯度破坏 VLM 预训练的语言理解能力。

**具体实现**:
- Flow Matching 在 VLM 隐状态条件下学习从噪声 $\epsilon$ 到目标动作 $\widetilde{A}_t^r$ 的流场
- Stop-gradient 机制切断 Flow Matching 梯度对 VLM 前缀 $H_{\phi,p}$ 的影响
- 推理时使用 $N=10$ 步 Euler 积分生成确定性轨迹
- 支持多种机器人平台：单臂、双臂、移动操作臂，通过 $d_{max}$ 填充和有效掩码 $M^{act}$ 统一处理

---

## 关键公式

### 公式 1: [[FAST|FAST 动作 Token 预训练损失]]

$$
\mathcal{L}_{FAST} = -\frac{1}{\sum_{i=1}^{L_z} m_i} \sum_{i=1}^{L_z} m_i \log p_\phi(z_i \mid v_t, c_t, y_t, z_{<i})
$$

**含义**: 以掩码加权的自回归交叉熵，使 VLM 学习在视觉观测 $v_t$、相机信息 $c_t$、语言指令 $y_t$ 条件下预测 FAST 动作 token 序列。

**符号说明**:
- $z_{1:L_z}$: FAST 动作 token 序列，长度 $L_z$
- $m_i \in \{0, 1\}$: token 有效掩码（过滤填充位）
- $p_\phi$: VLM 参数化的 token 分布
- $v_t, c_t, y_t$: 视觉观测、相机参数、语言指令

### 公式 2: [[Flow Matching|Flow Matching 线性插值]]

$$
X_\tau = \tau \widetilde{A}_t^r + (1 - \tau) \epsilon, \quad U_\tau = \widetilde{A}_t^r - \epsilon
$$

**含义**: 在时间步 $\tau \in [0,1]$ 上构造从噪声 $\epsilon$ 到目标动作 $\widetilde{A}_t^r$ 的线性插值路径，$U_\tau$ 为该路径的切向量（目标流场方向）。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 流匹配时间步，均匀采样
- $\widetilde{A}_t^r$: 归一化后的目标动作序列（delta 动作表示）
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $X_\tau$: 时间步 $\tau$ 处的插值状态
- $U_\tau$: 目标向量场方向

### 公式 3: [[Flow Matching|Flow Matching 掩码 MSE 损失]]

$$
\mathcal{L}_{FM} = \begin{cases} S_M^{-1} \displaystyle\sum_{k,d} M^{act}_{k,d} \left( V_{\theta,k,d} - U_{\tau,k,d} \right)^2 & S_M > 0 \\ 0 & S_M = 0 \end{cases}
$$

**含义**: 用有效动作掩码 $M^{act}$ 加权的均方误差，训练 DiT 网络 $V_\theta$ 预测流场 $U_\tau$，忽略填充维度的贡献。

**符号说明**:
- $M^{act}_{k,d} \in \{0, 1\}^{K \times d_{max}}$: 动作有效性掩码（$K$ 步、$d_{max}$ 维）
- $S_M = \sum_{k,d} M^{act}_{k,d}$: 有效动作元素总数
- $V_{\theta,k,d}$: DiT 网络预测的流场值
- $U_{\tau,k,d}$: 目标流场值

### 公式 4: [[Knowledge Insulation|Knowledge Insulation Stop-Gradient]]

$$
\widetilde{H}_{\phi,p}^{KI} = \text{sg}(H_{\phi,p}^{KI})
$$

**含义**: 对 VLM 前缀隐状态施加 stop-gradient 操作，阻断 Flow Matching 损失的梯度反传，保护预训练语言知识不被覆写。

**符号说明**:
- $H_{\phi,p}^{KI}$: VLM 在前缀 token 位置的隐状态
- $\text{sg}(\cdot)$: stop-gradient 算子（前向传递值不变，反向梯度置零）

### 公式 5: [[Knowledge Insulation|Knowledge Insulation 联合训练目标]]

$$
\mathcal{L}_{KI} = \alpha \mathcal{L}_{FM} + \mathcal{L}_{FAST} + \sum_j \lambda_j \mathcal{L}^{(j)}_{CE}, \quad \alpha = 10
$$

**含义**: Flow Matching 损失、FAST 预训练损失与多任务交叉熵损失的加权联合优化目标，$\alpha=10$ 放大 Flow Matching 的优化信号。

**符号说明**:
- $\alpha = 10$: Flow Matching 损失权重
- $\mathcal{L}_{FM}$: Flow Matching 掩码 MSE 损失
- $\mathcal{L}_{FAST}$: FAST token 预训练损失
- $\lambda_j$: 第 $j$ 个交叉熵任务的权重
- $\mathcal{L}^{(j)}_{CE}$: 第 $j$ 个辅助语言建模任务的交叉熵损失

---

## 关键图表

### Figure 1: 框架概览

![[LabVLA_fig1_overview.jpeg]]

**说明**: LabVLA 系统全景图。左侧为 RoboGenesis 三阶段数据合成（资产生成→场景构建→工作流生成），中心为 LabVLA 策略架构（VLM + DiT 动作专家），右侧为 LabUtopia 六类任务系列（取物/按压/开门/倒液/加热/转移）。

### Figure 2: RoboGenesis 数据合成流水线

![[LabVLA_fig2_robogenesis_18.jpeg]]
![[LabVLA_fig2_robogenesis_19.jpeg]]

**说明**: 三阶段流水线详细展示。(1) 环境构建：文本 → 参考图像 → [[TRELLIS]] 三维重建 → 带物理参数的 USD 资产；(2) 工作流生成：原子技能组合 → Agent 编排 → 六轴随机化；(3) 结构化导出：15 类标注提供商，含成功过滤器。

### Figure 3: LabVLA 训练方案

![[LabVLA_fig3_training.png]]

**说明**: 两阶段训练对比。左：FAST token 预训练阶段，VLM 以自回归方式学习动作 token 预测；右：Flow Matching 后训练阶段，[[DiT]] 动作专家在 Knowledge Insulation 约束下学习精确轨迹生成，stop-gradient 切断梯度回流至 VLM 前缀。

### Figure 4: 六类任务滚动快照

![[LabVLA_fig4_task1.jpeg]] ![[LabVLA_fig4_task2.jpeg]] ![[LabVLA_fig4_task3.jpeg]]
![[LabVLA_fig4_task4.jpeg]] ![[LabVLA_fig4_task5.jpeg]] ![[LabVLA_fig4_task6.jpeg]]

**说明**: LabUtopia 六类任务的第三视角执行过程截图（取物/按压/开门/倒液/加热/转移烧杯），展示 LabVLA 在不同操作语义下的执行成功率差异。

### Figure 5: 真实机器人实验设置

![[LabVLA_fig5_realworld.jpeg]]

**说明**: Franka 平台真实实验室工作空间，包含烧杯、锥形瓶、电子元件等道具，验证 LabVLA 的 sim-to-real 迁移能力，四类复合任务在不同目标位置和杂乱程度下测试。

### Figure 6: 四级实验室能力金字塔

![[LabVLA_fig6_pyramid.png]]

**说明**: 实验室 VLA 能力分级体系。从低到高：Apprentice（基础操作）→ Technician（协议执行，LabVLA 当前水平）→ Specialist（测量感知）→ Scientist（实验设计与自适应）。

### Figure 7: RoboGenesis 多样化场景展示（附录）

![[LabVLA_fig7_scenes.jpeg]]

**说明**: RoboGenesis 通过域随机化生成的九个代表性实验室场景，展示场景多样性（不同工作台布局、光照条件、视觉杂乱程度）。

---

### Table 1: 仿真引擎特性对比

| 引擎 | 机器人平台数 | 资产自动生成 | 场景自动生成 | 任务自动生成 | 成功 QA | 结构化标注 | 长程组合 | 实验室协议 |
|------|------------|------------|------------|------------|---------|-----------|---------|-----------|
| IsaacLab | 多 | ✗ | ✗ | ✗ | ✗ | ✗ | 部分 | ✗ |
| RoboSuite | 少 | ✗ | ✗ | 部分 | ✗ | ✗ | ✗ | ✗ |
| **RoboGenesis** | **16** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**: RoboGenesis 是唯一支持全部特性的仿真引擎，特别是实验室协议自动化和结构化标注生成。

### Table 2: LabUtopia 成功率对比（ID/OOD，%）

| 方法 | 参数量 | 取物 | 按压 | 开门 | 倒液 | 加热 | 转移烧杯 | **平均** |
|------|--------|------|------|------|------|------|---------|---------|
| LabVLA | 4B | 49.2 / 48.3 | 100 / 98.3 | 65.0 / 65.8 | 43.3 / 34.2 | 83.3 / 87.5 | 85.8 / 85.8 | **71.1 / 70.0** |
| π₀ | 3B | 21.7 / 19.2 | 92.5 / 89.1 | 51.6 / 53.3 | 37.5 / 38.3 | 90.0 / 90.8 | 86.7 / 88.3 | 63.3 / 63.2 |

（每格格式为 ID 成功率 / OOD 成功率，各任务 120 个 episode）

**关键发现**: LabVLA 在 ID/OOD 平均成功率上均优于所有基线，ID→OOD 仅下降 1.1pp，体现强领域随机化的泛化能力；倒液任务（Pour Liquid）OOD 仅 34.2%，是最难突破的接触敏感任务。

### Table 3: LabEmbodied-Data 可迁移性验证

| 基线方法 | 微调数据 | ID 提升 | OOD 提升 |
|----------|---------|---------|---------|
| X-VLA | LabEmbodied-Data | +15.0pp | +19.3pp |

**关键发现**: 在 X-VLA 上微调 LabEmbodied-Data 后，ID 和 OOD 成功率分别提升 15.0 和 19.3 个百分点，证明 RoboGenesis 生成的实验室数据对通用 VLA 模型具有良好的迁移价值。

### Table 4: 真实机器人（Franka）评估结果

| 任务 | LabVLA（clean） | LabVLA（clutter） | DreamZero（clean） | DreamZero（clutter） |
|------|----------------|------------------|-------------------|---------------------|
| 复合任务平均 | 86.5% | — | 约 86% | — |

（四类复合任务，测试目标位置变化与杂乱程度变化）

**关键发现**: LabVLA 在真实 Franka 平台上平均达到 86.5%，与 DreamZero 基线相当，验证 sim-to-real 迁移的可行性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LabEmbodied-Data（RoboGenesis） | 10,000 场景 × 多 workflow | 16 机器人平台，15 类标注，六轴随机化 | LabVLA 预训练 + 后训练 |
| LabAssetLibrary | 2,947 个资产 | 文本→三维重建，含物理参数 | 场景构建原材料 |
| LabUtopia | 6 类任务 × 120 episodes × 2 split | ID/OOD 双评测，覆盖实验室核心操作 | 测试 |

### 实现细节

- **Backbone**: Qwen3-VL-4B-Instruct（视觉-语言主干）
- **动作专家**: 18 层 DiT，宽度 1024，8 个注意力头，通过 Cross-Attention 与 VLM 隐状态交互
- **训练阶段 1（预训练）**: FAST token 交叉熵，学习动作语义表示
- **训练阶段 2（后训练）**: Flow Matching + Knowledge Insulation（$\alpha=10$）
- **推理**: $N=10$ 步 Euler 积分
- **动作归一化**: delta 动作目标，关节用均值-标准差归一化，夹爪用 q01/q99 对齐
- **平台支持**: 单臂/双臂/移动操作臂，$d_{max}$ 填充 + 有效掩码统一处理

### 可视化结果

六类任务执行快照（Figure 4）显示：按压按钮（100% ID）和转移烧杯（85.8%）表现最优，证明 LabVLA 对离散动作和轨迹规划任务的掌握；倒液操作因需要精确倾斜角控制而成功率最低，反映接触丰富操作仍是核心挑战。

---

## 批判性思考

### 优点

1. **系统完整性**: RoboGenesis + LabEmbodied-Data + LabVLA + LabUtopia 形成端到端的完整研究基础设施，可复用性强
2. **领域针对性**: 实验室操作的失败模式分类（移液/倒液/加热）和成功过滤器设计体现深度领域理解
3. **泛化能力**: ID/OOD 仅 1.1pp 差距，六轴随机化策略效果显著

### 局限性

1. **真实验证不足**: 真实机器人实验仅在单一 Franka 平台的桌面任务上验证，尚未在真实实验室环境（含危险化学品、精密仪器）中测试
2. **智能层级受限**: 当前处于"Technician"级别，只能执行预定义协议，不具备测量感知或基于实验结果自适应调整的能力
3. **倒液任务瓶颈**: Pour Liquid OOD 仅 34.2%，接触丰富操作的精度问题尚未解决
4. **安全与污染风险**: 论文承认真实实验室的危险品管理、交叉污染等问题未被当前基准覆盖

### 潜在改进方向

1. 引入触觉感知或力控反馈解决倒液精度问题
2. 扩展到 Specialist 级别：集成液体体积测量、温度反馈等物理量感知
3. 开发人机协作机制，支持实验异常时的人工干预说明

### 可复现性评估

- [ ] 代码开源（论文提交时未明确）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（附录 A 提供超参数）
- [ ] 数据集可获取（LabEmbodied-Data 公开状态未知）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: VLM 骨干网络
- [[FAST]]: 动作 token 化方法，预训练基础
- [[Flow Matching]]: 后训练阶段的运动生成范式
- [[DiT]]: 动作专家架构

### 对比

- [[π0]]: 最强基线（3B），平均低 LabVLA 约 8pp
- [[X-VLA]]: 通用 VLA 基线，验证 LabEmbodied-Data 迁移价值
- [[OpenVLA]]: 通用 VLA 基线之一

### 方法相关

- [[Knowledge Insulation]]: 防止 Flow Matching 梯度破坏 VLM 知识
- [[Domain Randomization|领域随机化]]: RoboGenesis 六轴随机化策略
- [[TRELLIS]]: 资产三维重建模型

### 硬件 / 数据相关

- [[Franka]]: 真实机器人实验平台
- [[LabUtopia]]: 本文提出的实验室 VLA 基准

---

## 速查卡片

> [!summary] LabVLA: Grounding VLA Models in Scientific Laboratories
> - **核心**: 首个实验室 VLA 系统，含仿真引擎 RoboGenesis + 两阶段训练 + LabUtopia 基准
> - **方法**: Qwen3-VL-4B + 18 层 DiT，FAST 预训练 + Flow Matching + Knowledge Insulation
> - **结果**: LabUtopia 平均 71.1%（ID）/ 70.0%（OOD），优于 π₀（63.3%/63.2%）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-12*
