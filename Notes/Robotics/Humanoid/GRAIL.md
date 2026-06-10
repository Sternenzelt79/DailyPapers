---
title: "GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors"
method_name: "GRAIL"
authors: [Tianyi Xie, Haotian Zhang, Jinhyung Park, Zi Wang, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo, Umar Iqbal, Ye Yuan]
year: 2026
venue: arXiv
tags: [humanoid-robot, loco-manipulation, data-generation, video-foundation-model, motion-tracking, reinforcement-learning, sim-to-real]
zotero_collection: Robotics/Humanoid
image_source: local
arxiv_html: https://arxiv.org/html/2606.05160
created: 2026-06-04
---

# 论文笔记：GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA DAIR Lab |
| 日期 | June 2026 |
| 项目主页 | [research.nvidia.com/labs/dair/grail](https://research.nvidia.com/labs/dair/grail/) |
| 对比基线 | [[SONIC]], [[ResMimic]], [[HDMI]], [[DAViD]], [[HOIDiff]], [[CHOIS]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05160) / Code: N/A |

---

## 一句话总结

> GRAIL 是一个全数字化流水线，无需物理采集，通过 3D 资产 + 视频基础模型生成 2 万余条仿人机器人全身操作示范，并在真实 Unitree G1 上实现 84% 取物成功率。

---

## 核心贡献

1. **全数字化数据生成流水线**: 无需遥操作或动捕，通过 3D 资产 + [[视频基础模型|Video Foundation Model (VFM)]] 合成具身操作视频，再重建为物理可执行 4D HOI 轨迹
2. **任务通用跟踪策略**: 设计 Object-Aware Adaptor 与 Scene-Aware Tracker，冻结预训练 [[SONIC]] 控制器主体，仅适配隐空间与高度图编码器，实现 pick-up + terrain 跨任务泛化
3. **大规模真实部署验证**: 在 Unitree G1 上部署，达到 84%（已见物体）/ 80%（未见物体）取物成功率与 90% 爬楼梯成功率

---

## 问题背景

### 要解决的问题

仿人机器人的全身 [[Loco-Manipulation|移动操作]] 需要同时协调下肢运动与上肢操作，而高质量训练数据极度稀缺——物理遥操作成本高、动作捕捉难以扩展到多样场景。

### 现有方法的局限

- [[HOIDiff]]、[[CHOIS]] 等 HOI 生成方法不以真实机器人形态为约束，追踪成功率极低（10–16%）
- [[DAViD]] 虽使用视频先验，但未利用完整 3D 场景信息，追踪成功率仅 24%
- ResMimic、HDMI 等依赖预录制人类示范，无法规模化

### 本文的动机

若在生成前就将机器人形态、物体几何、相机参数完全确定下来（"privileged 3D setup"），视频生成后的重建问题将大幅简化，可获得度量尺度精确的 4D HOI 轨迹，再通过策略学习转化为可执行机器人控制。

---

## 方法详解

### 模型架构

GRAIL 采用**三阶段流水线**：

- **Stage 1（机器人视角视频生成）**: 组装含物体、角色（预适配目标机器人）和相机参数的 3D 配置 → 渲染初始帧 → VFM（[[Kling]] 2.5 Turbo Pro）合成交互参考视频；[[VLM]] 自动生成交互提示词
- **Stage 2（交互感知 4D HOI 重建）**: 人体运动初估（[[GENMO]] + [[WiLoR]]）+ 物体姿态（[[FoundationPose]]）→ 五项损失联合优化，输出度量尺度精确的轨迹
- **Stage 3（任务通用跟踪策略）**: Object-Aware Adaptor（调制 [[SONIC]] 冻结控制器的隐变量 + 输出手部动作） + Scene-Aware Tracker（含高度图编码器微调）；基于 4D HOI 轨迹池训练，非逐序列拟合

### 核心模块

#### 模块1: Object-Aware Adaptor（物体感知适配器）

**设计动机**: 冻结预训练 [[SONIC]] 全身控制器，通过残差隐变量 $\Delta z_t$ 注入物体感知信号，避免破坏运动先验。

**输入观测**:
- [[本体感知|Proprioception]]：关节位置（43 维）、速度、基座角速度（3 维）、上一动作（66 维）
- 参考运动锚点：位置（3 维）、旋转 6D、当前指令（58 维）、多步未来指令（580 维）
- 物体状态：位置（3 维）、旋转 6D、位置增量（30 维）、旋转增量（60 维）、目标位置（3 维）、[[BPS]] 形状编码（10 维）、桌面信息（9 维）
- 手-物接触：变换（9 维）、指尖力（12 维）

**具体实现**:
- 输出 64 维隐变量残差 $\Delta z_t$ + 2 维手部原语 $a^{hand}_t$
- 残差乘以 $\lambda=0.1$ 后通过 [[FSQ]] 量化加入 SONIC 隐变量
- 辅助 $\ell_2$ 惩罚约束 $\Delta z_t$ 不偏离过远

#### 模块2: Scene-Aware Tracker（场景感知追踪器）

**设计动机**: 利用 [[高度图|Height Map]] 感知地形几何，使策略在爬楼梯、斜坡等非平地场景中保持稳定运动。

**具体实现**:
- 高度图：$11 \times 11$ 网格，覆盖 1.5 m 范围，分辨率 0.15 m（121 维）
- [[CNN]] 编码器：3 层 [64, 128, 256] 通道，$3 \times 3$ 核，stride 2
- 辅助并行运动学解码器 $G_{rec}$ 提供 MSE 监督损失
- 微调（而非冻结）SONIC 控制器以适配地形

#### 模块3: 4D HOI 重建优化

**设计动机**: 利用预知的 3D 几何信息消歧视频重建中的尺度与接触歧义。

**具体实现**:
- 人体初估：[[GENMO]]（全身）+ [[WiLoR]]（手部）
- 物体初估：[[FoundationPose]]（6-DoF 姿态追踪）
- 联合优化五项损失（见公式部分）
- 过滤：用 [[SAM2]] mask 与 FoundationPose 预测对比，丢弃 mask 追踪误差 $e_M > 0.2$ 的序列

---

## 关键公式

### 公式1: [[联合优化损失|Joint Optimization Loss]]

$$
\mathcal{L} = \lambda_{kp} \mathcal{L}_{kp} + \lambda_{proj} \mathcal{L}_{proj} + \lambda_{depth} \mathcal{L}_{depth} + \lambda_{cont} \mathcal{L}_{cont} + \lambda_{reg} \mathcal{L}_{reg}
$$

**含义**: 4D HOI 重建的总优化目标，从关键点、投影、深度、接触、正则五个维度约束人体与物体轨迹质量。

**符号说明**:
- $\lambda_{kp}, \lambda_{proj}, \lambda_{depth}, \lambda_{cont}, \lambda_{reg}$: 各项损失权重
- $\mathcal{L}_{kp}$: 关键点对齐损失
- $\mathcal{L}_{proj}$: 物体投影损失
- $\mathcal{L}_{depth}$: 深度对齐损失
- $\mathcal{L}_{cont}$: 接触损失
- $\mathcal{L}_{reg}$: 正则化损失

### 公式2: [[关键点对齐损失|Keypoint Alignment Loss]]

$$
\mathcal{L}_{kp} = \frac{1}{T} \sum_{t=1}^{T} \| K^H(\Theta^H_t) - p_t \|
$$

**含义**: 约束重建的人体关键点位置与视频估计的 2D 关键点一致，纠正运动初估误差。

**符号说明**:
- $K^H(\Theta^H_t)$: 由人体参数 $\Theta^H_t$ 经正向运动学计算出的关键点投影
- $p_t$: 时刻 $t$ 从视频中估计的 2D 关键点位置
- $T$: 序列帧数

### 公式3: [[物体投影损失|Object Projection Loss]]

$$
\mathcal{L}_{proj} = \sum_{t} \| P(V^O_t) - P(\hat{Y}^O_t) \|
$$

**含义**: 约束重建物体的 2D 投影与视频中观测到的物体轮廓对齐。

**符号说明**:
- $P(\cdot)$: 相机投影算子
- $V^O_t$: 时刻 $t$ 物体的 3D 顶点
- $\hat{Y}^O_t$: 视频中物体的 2D 观测（mask 或轮廓）

### 公式4: [[深度对齐损失|Depth Alignment Loss]]

$$
\mathcal{L}_{depth} = \frac{1}{T} \sum_{t=1}^{T} \left[ \text{CD}(V^{H,vis}_t,\, P^H_t) + \text{CD}(V^{O,vis}_t,\, P^O_t) \right]
$$

**含义**: 利用单目深度估计的点云约束人体与物体的度量尺度，解决单目重建的尺度歧义。

**符号说明**:
- $\text{CD}(\cdot, \cdot)$: [[Chamfer Distance|倒角距离]]
- $V^{H,vis}_t, V^{O,vis}_t$: 可见人体/物体顶点
- $P^H_t, P^O_t$: 从深度图反投影的人体/物体点云

### 公式5: [[接触损失|Contact Loss]]

$$
\mathcal{L}_{cont} = \frac{1}{|\mathcal{T}_c|} \sum_{t \in \mathcal{T}_c} \text{CD}_z(V^{H,cont}_t,\, V^{O,cont}_t)
$$

**含义**: 在发生接触的帧内，约束手部接触顶点与物体表面在 z 方向的距离为零，确保手物接触的物理合理性。

**符号说明**:
- $\mathcal{T}_c$: 接触帧集合
- $\text{CD}_z(\cdot, \cdot)$: 仅沿 z 轴方向的倒角距离
- $V^{H,cont}_t$: 手部接触顶点
- $V^{O,cont}_t$: 物体表面顶点

### 公式6: [[隐变量适配|Latent Adaptation]]

$$
(\Delta z_t,\; a^{hand}_t) = \pi_\phi(s_t, o_t), \qquad a^{body}_t = G(z_t + \lambda \Delta z_t)
$$

**含义**: 适配器策略 $\pi_\phi$ 输出隐残差与手部动作；SONIC 生成器 $G$ 以调制后的隐变量输出全身动作，实现在保留运动先验的同时注入物体感知。

**符号说明**:
- $\pi_\phi$: Object-Aware Adaptor 策略（可训练）
- $s_t, o_t$: 时刻 $t$ 的状态与物体观测
- $\Delta z_t$: 64 维隐变量残差
- $a^{hand}_t$: 2 维手部原语（张/合）
- $G(\cdot)$: 冻结的 SONIC 生成器
- $z_t$: 原始 SONIC 隐变量
- $\lambda = 0.1$: 残差缩放系数

### 公式7: [[运动追踪奖励|Motion Tracking Reward]]

$$
R^{motion}_t = \sum_{i} w_i \exp\!\left( -\frac{\| \tilde{x}_{i,t} - x_{i,t} \|^2}{\sigma^2_i} \right)
$$

**含义**: 通过指数衰减惩罚形式给予运动追踪奖励，鼓励策略使关节/锚点位置贴近参考轨迹。

**符号说明**:
- $w_i$: 第 $i$ 项追踪奖励权重（0.5–5.0）
- $\tilde{x}_{i,t}$: 第 $i$ 个追踪目标的参考值
- $x_{i,t}$: 当前仿真状态值
- $\sigma^2_i$: 第 $i$ 项的容差超参数

### 公式8: [[物体追踪奖励|Object Tracking Reward]]

$$
R^{obj}_t = w_p \exp(-\alpha_p \| \tilde{p}^O_t - p^O_t \|) + w_r \exp(-\alpha_r \| \tilde{r}^O_t \ominus r^O_t \|)
$$

**含义**: 分别对物体位置误差与旋转误差给予指数奖励，引导策略追踪参考物体轨迹。

**符号说明**:
- $w_p, w_r$: 位置/旋转奖励权重（均为 20.0）
- $\alpha_p, \alpha_r$: 衰减系数
- $\tilde{p}^O_t, \tilde{r}^O_t$: 参考物体位置与旋转
- $\ominus$: 旋转差算子

### 公式9: [[抓取奖励|Grasp Reward]]

$$
R^{grasp}_t = w_{ct} \cdot r_{contact} + w_{dir} \cdot r_{direction} + w_{prox} \cdot r_{proximity}
$$

**含义**: 综合接触时间、拇指-食指对握姿势、指尖接近度三项，引导策略形成稳定抓握。

**符号说明**:
- $r_{contact}$: 手指与物体接触数量项（权重 40.0）
- $r_{direction}$: 拇指-食指对握方向项（权重 10.0）
- $r_{proximity}$: 指尖与物体距离项（权重 0.1）

### 公式10: [[Mask 追踪误差|Mask Tracking Error]]

$$
e_M = \frac{\sum_t \sum \left((1 - M_t) \cdot \hat{M}_t\right)}{\sum_t \sum M_t}
$$

**含义**: 量化 SAM2 预测 mask 与 FoundationPose 重投影 mask 之间的不一致性，用于过滤低质量重建序列（阈值 $e_M > 0.2$）。

**符号说明**:
- $M_t$: SAM2 在帧 $t$ 的 mask 预测
- $\hat{M}_t$: FoundationPose 重投影 mask

---

## 关键图表

### Figure 1: 整体流水线与真实部署

![[GRAIL_fig1_teaser.jpeg]]

**说明**: 从全数字化数据生成到真实世界部署的完整流程。左侧展示三阶段流水线（视频生成 → HOI 重建 → 策略学习），右侧展示 Unitree G1 在真实场景中执行取物和爬楼梯任务。

### Figure 2: 资产驱动 4D HOI 生成

![[GRAIL_fig2_hoi_generation.jpeg]]

**说明**: Stage 1 + Stage 2 的详细示意。左侧为 3D 场景组装（物体资产 + 角色 + 相机），中间为 VFM 合成的交互视频，右侧为联合优化后的 4D HOI 重建结果，展示精确的手物接触。

### Figure 3: 任务通用追踪策略

![[GRAIL_fig3_motion_tracking.jpeg]]

**说明**: Object-Aware Adaptor 与 Scene-Aware Tracker 的架构。左侧展示适配器如何将 $\Delta z_t$ 注入冻结的 [[SONIC]] 控制器，右侧展示高度图编码器如何为地形感知提供输入。

### Figure 4: 生成的多样化运动数据

![[GRAIL_fig4_diverse_motion.jpeg]]

**说明**: GRAIL 生成的四类任务数据示例：桌面/地面取物、全身操作、多样椅型坐下、地形穿越（台阶、斜坡）。展示数据集的多样性与物理可行性。

### Figure 5: Sim-to-Real 部署

![[GRAIL_fig5_deployment.jpeg]]

**说明**: 真实 Unitree G1 机器人执行取物任务的过程图。传感器配置为 Luxonis OAK-D 相机，推理频率 10 Hz。展示已见/未见物体的成功案例。

### Figure 6: 定性对比

![[GRAIL_fig6_qualitative_1.png]]

![[GRAIL_fig6_qualitative_2.png]]

**说明**: GRAIL 与 HOIDiff、CHOIS、DAViD 的生成质量对比，展示 GRAIL 在手物接触自然性和物理真实性上的显著优势。

### Table 1: HOI 生成质量对比

| 方法 | Contact ↓ | Pen. ↓ | Inter. Score ↑ | Human Smo. ↓ | Obj Smo. ↓ | SR ↑ | Body Dev. ↓ | Obj Dev. ↓ |
|------|-----------|--------|----------------|--------------|------------|------|-------------|------------|
| HOIDiff | 0.012 | 2.07% | 1.79 | 0.0043 | 0.0118 | 15.8% | 0.2120 | 0.3352 |
| CHOIS | 0.034 | 3.74% | 2.47 | 0.0055 | 0.0062 | 10.5% | 0.2564 | 0.3642 |
| DAViD | 0.246 | 1.46% | 2.74 | 0.0024 | 0.0605 | 24.0% | 0.4723 | 0.5826 |
| **GRAIL (Ours)** | **0.008** | **0.90%** | **3.58** | 0.0033 | **0.0022** | **88.9%** | **0.0913** | **0.0851** |

**关键发现**: GRAIL 在追踪成功率（88.9% vs 最高 24.0%）和接触质量（Contact 0.008 vs 最低 0.012）上全面超越基线。

### Table 2: 任务通用 Loco-Manipulation 追踪

| 方法 | SR ↑ | ObjPos (m) ↓ | MPJPE-L ↓ |
|------|------|--------------|-----------|
| HDMI | 48.5% | 0.283 | 122.3 |
| ResMimic | 49.2% | 0.393 | 80.9 |
| Ours w/o SONIC | 45.0% | 0.395 | 243.5 |
| Ours w/o $\pi_\phi$ | 39.7% | 0.303 | 37.1 |
| Ours w/o Rel. Obs. | 57.9% | 0.257 | 43.0 |
| **GRAIL (Full)** | **81.4%** | **0.135** | **41.8** |

**关键发现**: 完整的 GRAIL 超越 ResMimic（81.4% vs 49.2%）；去掉 $\pi_\phi$（适配器策略）成功率骤降至 39.7%，证明隐变量适配的关键作用。

### Table 3: 真实世界取物实验

| 物体（已见） | 成功率 | 物体（未见） | 成功率 |
|------------|--------|------------|--------|
| Cube | 100% | Spray Can | 100% |
| Apple | 60% | Lint Roller | 50% |
| Tea Box | 100% | Peach | 90% |
| Carrot | 70% | Flashlight | 80% |
| Wet Wipes | 90% | Medicine Bottle | 80% |
| **平均（已见）** | **84%** | **平均（未见）** | **80%** |

**关键发现**: 已见物体 84%、未见物体 80% 的成功率，证明策略具备良好泛化能力。

### Table 4: 生成流水线运行时间

| 阶段 | 耗时 |
|------|------|
| 视频生成（Kling API） | ~1 min |
| 人体运动估计 | ~2 min |
| 物体姿态追踪 | ~1 min |
| 优化预处理 | ~2 min |
| 联合优化 | ~8 min |
| **合计（每条 5s 序列）** | **~14 min** |

### Table 7: 用户研究（30 名参与者）

| 方法 | 功能可信度 ↑ | 物理可信度 ↑ |
|------|------------|------------|
| HOIDiff | 2.0% | 1.9% |
| CHOIS | 12.2% | 16.8% |
| DAViD | 11.2% | 10.4% |
| **GRAIL** | **74.7%** | **70.9%** |

**关键发现**: 人类评估者在两项主观指标上均有约 75% 的比例认为 GRAIL 生成效果最优。

### Table 8: 重建损失消融实验

| 配置 | Contact ↓ | Pen. ↓ | MPJPE ↓ | SR ↑ | ObjPos ↓ |
|------|-----------|--------|---------|------|----------|
| w/o $\mathcal{L}_{proj}$ | 0.016 | 1.93% | 16.70 | 41.6% | 0.374 |
| w/o $\mathcal{L}_{depth}$ | 0.017 | 1.97% | 4.35 | 42.6% | 0.372 |
| w/o $\mathcal{L}_{cont}$ | 0.024 | 1.52% | 4.81 | 53.3% | 0.332 |
| **Full Model** | **0.015** | **1.81%** | **4.89** | **81.4%** | **0.135** |

**关键发现**: 三项损失缺一不可，去掉任意一项追踪成功率均下降至 54% 以下；$\mathcal{L}_{proj}$ 和 $\mathcal{L}_{depth}$ 的缺失对 SR 影响最大（分别降至 41.6%、42.6%）。

---

## 实验

### 数据集规模

| 类别 | 规模 | 特点 |
|------|------|------|
| Pick-up（桌面/地面） | 主体 | 1000 种物体资产（Robocasa, ComAsset, OMOMO, Hunyuan3D） |
| Whole-body Manipulation | 部分 | 全身参与的复杂操作 |
| Sitting | 部分 | 多样椅型（高/低椅、沙发） |
| Terrain Traversal | 部分 | 1000 个程序化地形（台阶、斜坡、楼梯） |
| **总计** | **20,000+ 序列** | — |

### 实现细节

- **物理仿真器**: [[Isaac Lab]]
- **策略训练**: [[PPO]]，30,000 次迭代
- **并行环境**: 每 GPU 1,024 个并行环境
- **训练硬件**: 64 块 NVIDIA L40 GPU
- **训练时长**: ~30 小时/次完整训练
- **部署机器人**: Unitree G1 + Luxonis OAK-D 相机
- **推理频率**: 10 Hz
- **视频生成模型**: Kling 2.5 Turbo Pro（静态相机模式）
- **人体重定向**: [[GMR]] 框架
- **过滤阈值**: Mask 追踪误差 $e_M > 0.2$ 的序列丢弃

### 奖励权重

| 奖励类型 | 主要项目 | 权重范围 |
|--------|---------|---------|
| 运动追踪 | 锚点位置/旋转、关节速度 | 0.5–5.0 |
| 物体追踪 | 位置误差、旋转误差 | 20.0 |
| 抓取 | 接触计数、方向、接近度 | 0.1–40.0 |
| 正则化 | 隐变量 $\ell_2$、动作变化率 | 0.1–(-10.0) |

---

## 批判性思考

### 优点

1. **privileged 3D 设置消歧**: 预设物体几何、相机参数、机器人形态，使视频重建问题从欠约束变为良定，是本文最核心的洞见
2. **任务通用泛化**: 基于轨迹族（trajectory pool）训练而非逐序列拟合，单条序列适配时间仅 0.5–0.9 min，具备良好扩展性
3. **真实部署验证充分**: 在多样物体和真实地形上验证，成功率数据可信
4. **人类评估与定量指标一致**: 用户研究（74.7%）与 SR（88.9%）双重验证方法优越性

### 局限性

1. **依赖 3D 资产与仿真就绪场景**: 需要预先获取或建模目标物体的 3D 模型，限制了对新颖物体的泛化
2. **重建在严重遮挡/快速运动下退化**: 极端场景的失败率仍不可忽略（过滤率约 20%）
3. **VFM 一致性依赖**: 生成视频质量直接影响重建精度，对 Kling 等商业 API 的依赖可能带来可复现性问题
4. **推理速度限制**: 10 Hz 的控制频率对高动态操作场景可能不足

### 潜在改进方向

1. 引入无需 CAD 模型的物体 3D 重建方法（如 NeRF/3DGS），降低对预建资产库的依赖
2. 探索更快的 VFM 推理路径，将每条序列生成时间从 14 分钟压缩至分钟级以下
3. 将方法扩展至双臂操作和灵巧手场景

### 可复现性评估

- [ ] 代码开源（截至论文发布未见 GitHub 链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、架构、硬件均有描述）
- [ ] 数据集可获取（20,000 序列未公开，但物体资产来源已注明）

---

## 关联笔记

### 基于

- [[SONIC]]: 预训练全身运动控制器，GRAIL 在其基础上添加适配器
- [[GMR]]: 人体运动重定向框架，用于将人类示范对齐至 G1 形态
- [[GENMO]]: 人体全身运动估计，Stage 2 初估模块
- [[WiLoR]]: 手部运动估计，Stage 2 初估模块

### 对比

- [[ResMimic]]: 残差模仿学习基线（SR 49.2% vs GRAIL 81.4%）
- [[HDMI]]: 人类示范运动追踪基线（SR 48.5%）
- [[DAViD]]: 视频驱动的 HOI 生成对比方法（SR 24.0%）
- [[HOIDiff]]: 扩散模型 HOI 生成（SR 15.8%）
- [[CHOIS]]: 接触感知 HOI 生成（SR 10.5%）

### 方法相关

- [[视频基础模型]]: Kling 2.5 Turbo Pro，提供交互视频先验
- [[FoundationPose]]: 物体 6-DoF 姿态追踪
- [[BPS]]: Basis Point Set，物体形状编码方法
- [[FSQ]]: Finite Scalar Quantization，隐变量离散化
- [[PPO]]: 近端策略优化，策略训练算法
- [[Chamfer Distance]]: 点云距离度量，深度对齐损失核心

### 硬件/数据相关

- [[Unitree G1]]: 部署的仿人机器人
- [[Isaac Lab]]: NVIDIA 物理仿真平台
- [[SAM2]]: 视频目标分割，用于质量过滤

---

## 速查卡片

> [!summary] GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
> - **核心**: 全数字化流水线，无需物理采集，生成 2 万+ 条仿人机器人操作示范
> - **方法**: 3D 资产 + VFM 视频生成 → 4D HOI 重建 → Object-Aware Adaptor + Scene-Aware Tracker
> - **结果**: 真实 Unitree G1 取物 84%（已见）/ 80%（未见），爬楼梯 90%
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-04*
