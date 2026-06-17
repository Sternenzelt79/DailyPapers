---
title: "LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies"
method_name: "LaWAM"
authors: [Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu]
year: 2026
venue: arXiv
tags: [world-action-model, latent-action, vla, robot-manipulation, diffusion-policy]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.15768
created: 2026-06-16
---

# 论文笔记：LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确列出（多作者合作） |
| 日期 | June 2026 |
| 项目主页 | 无 |
| 对比基线 | [[Fast-WAM]], [[GR00T-N1.6]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.15768) / Code: 未开源 |

---

## 一句话总结

> LaWAM 用紧凑的潜在视觉子目标替代昂贵的像素级视频预测，在保持 SOTA 成功率（LIBERO 98.6%）的同时将推理延迟降低至 187ms（比像素级 WAM 快 24 倍）。

---

## 核心贡献

1. **Latent World Model（LaWM）预训练**: 在 4500 小时机器人+人类自我中心视频上预训练 [[逆动力学编码器]]，学习具身无关的潜在动作表示
2. **高效潜在子目标接口**: 用紧凑的潜在特征（而非像素级视频）桥接世界模型预测与动作生成，消除昂贵的扩散去噪视频合成
3. **混合频率物理时间对齐**: 引入物理时间戳编码 $\phi(t_{b,i})$，使跨不同控制频率（5/10/20 Hz）的联合训练成为可能

---

## 问题背景

### 要解决的问题

现有 [[World-Action Model|WAM]] 方法（如 [[Fast-WAM]]、Cosmos-Policy）将未来像素级视频作为中间表示来指导动作生成，但像素级视频生成需要迭代扩散去噪，导致推理延迟极高（Fast-WAM 486ms、LingBot-VA 4482ms）。

### 现有方法的局限

- **像素级 WAM**：推理慢（数百 ms 到数秒），计算代价大，对策略性能的提升和延迟之间难以平衡
- **潜在动作方法**（LAPA、UniVLA）：仅在单一具身或单一数据集上学习，泛化能力受限；动作表示与物理世界动力学脱钩
- **直接 VLA**（π₀.₅、GR00T-N1.6）：缺乏显式的预测性动力学模块，对需要长程规划的任务支持有限

### 本文的动机

预测"将要发生什么"对机器人决策有价值，但不需要在像素空间重建；仅在特征空间预测未来潜在状态就足以为动作专家提供足够的动态先验，同时大幅降低计算代价。

---

## 方法详解

### 模型架构

LaWAM 采用**两阶段**训练架构：

- **Stage 1** — 预训练 [[Latent World Model|LaWM]]（独立于策略）
- **Stage 2** — 将 LaWM 集成进 [[Vision-Language-Action Model|VLA]] 策略，联合训练动作专家

**整体推理流程**：
- **输入**: 语言指令 $l$ + 当前观测 $o$（RGB，256×256）
- **Backbone**: Qwen-GR00T（Qwen3-VL 前 16 层）提取视觉-语言特征 $u$
- **策略先验**: 预测潜在动作 $\hat{z}$
- **LaWM 解码器**: 由 $\hat{z}$ 生成潜在子目标 $\hat{u}_T$
- **动作专家**: [[Alternate-DiT]]（4块，16层）条件化于 $(u, \hat{u}_T)$ 生成动作块
- **总参数**: 2.3B

### 核心模块

#### 模块1: LaWM（Latent World Model）

**设计动机**: 在冻结的 [[DINOv3 ViT]] 特征空间中学习视觉状态转移，避免像素重建的巨大代价。

**具体实现**:
- 编码器：24 层 [[Transformer]] 架构，[[逆动力学编码器|V-JEPA2 风格时空设计]]，将当前帧 $u$ 和目标帧 $u_T$ 的视觉 patch 展平拼接，推断潜在动作后验 $q_\phi(z|u, u_T)$
- 解码器：24 层 Transformer，通过 [[自适应层归一化|Adaptive Layer Normalization]] 注入潜在动作 $z$（比加法注入更稳定），预测未来特征 $\tilde{u}_T$
- 视觉编码器：[[DINOv3]] ViT-B/16（冻结），输出 $u \in \mathbb{R}^{D}$
- 辅助状态预测器：MLP，训练时预测 $s_T$（训练后丢弃）

**预训练数据**: 3000 小时机器人视频 + 1500 小时自我中心人类视频（Open X-Embodiment, DROID, RoboCOIN, Robomind, LEGO, Ego4D, EgoDex, AgiBot World Colosseo）

#### 模块2: Alternate-DiT 动作专家

**设计动机**: 同时利用 [[Vision-Language Model|VLM]] Backbone 的语义任务信息和 LaWM 预测的动力学上下文，两路交替融合，实现任务意图与场景演化的协同指导。

**具体实现**:
- 一路流：VLM Backbone 的语义上下文（语言 + 视觉）
- 另一路流：LaWM 预测的潜在动力学上下文 $(u, \hat{u}_T)$
- 两路流交替作用于 4 个 [[DiT]] 块（共 16 个 Transformer 层），隐藏维度 1024
- 使用 [[条件流匹配|Conditional Flow Matching]] 生成动作块 $a_{1:T}$

#### 模块3: 知识绝缘（Knowledge Insulation）

**设计动机**: 防止动作专家的梯度污染（覆写）预训练好的 LaWM 动力学参数，避免[[灾难性遗忘]]。

**具体实现**: 在 Stage 2 联合训练时，阻断动作专家梯度对 LaWM 解码器权重的更新（参考文献 [10] 的 KI 机制）。

#### 模块4: 混合频率物理时间对齐

**设计动机**: 不同机器人平台控制频率不同（5/10/20 Hz），直接联合训练会产生频率混淆。

**具体实现**: 为每个动作 token 分配物理时间戳 $t_{b,i} = i/h_b$，使用 [[正弦位置编码|正弦时间编码]] 而非离散频率 ID，实现跨频率统一表达。

---

## 关键公式

### 公式1: [[World-Action Model|WAM 因式分解]]

$$
p(a_{1:T}, o_T | o, l) = p(o_T | o, l) \cdot p(a_{1:T} | o, o_T)
$$

**含义**: 标准 WAM 将轨迹生成分解为未来观测预测和条件动作生成两步

**符号说明**:
- $a_{1:T}$: 动作序列
- $o_T$: 地平线（horizon）时刻的像素级观测
- $o$: 当前观测
- $l$: 语言任务指令

### 公式2: [[Latent Action|潜在动作模型（LAM）]]

$$
z \sim q_\phi(z | u, u_T), \quad \tilde{u}_T = \text{LaWM}_\omega(u, z)
$$

**含义**: 逆动力学编码器从视觉特征转移中推断潜在动作，解码器用潜在动作预测未来特征

**符号说明**:
- $z$: 潜在动作向量（服从近似后验 $q_\phi$）
- $u, u_T$: 当前与目标时刻的冻结视觉编码器特征
- $\tilde{u}_T$: LaWM 解码器预测的未来特征
- $\phi, \omega$: 分别为编码器和解码器参数

### 公式3: [[LaWAM]] 因式分解

$$
p(a_{1:T}, \hat{u}_T, \hat{z} | o, l) = p_\theta(\hat{z} | o, l) \cdot p_\omega(\hat{u}_T | u, \hat{z}) \cdot p_\eta(a_{1:T} | o, l, u, \hat{u}_T)
$$

**含义**: LaWAM 将策略分解为三个可学习组件——策略先验、LaWM 解码器、动作专家

**符号说明**:
- $p_\theta$: 策略先验（预测潜在动作 $\hat{z}$）
- $p_\omega$: LaWM 解码器（固定权重，生成子目标 $\hat{u}_T$）
- $p_\eta$: Alternate-DiT 动作专家
- $\hat{z}, \hat{u}_T$: 策略推断的潜在动作和子目标（区别于训练时后验采样的 $z, u_T$）

### 公式4: [[Latent Action Model|LAM 训练目标]]

$$
\mathcal{L}_{\text{LAM}} = \mathcal{L}_{\text{wm}} + \mathcal{L}_{\text{aux}} + \beta \cdot D_{\text{KL}}(q_\phi(z | u, u_T) \| \mathcal{N}(0, I))
$$

其中：

$$
\mathcal{L}_{\text{wm}} = \|\tilde{u}_T - u_T\|_2^2, \quad \mathcal{L}_{\text{aux}} = \|g(s, z) - s_T\|_2^2
$$

**含义**: Stage 1 的 LAM 预训练损失，包含世界模型重建损失、辅助状态预测损失和 KL 正则化

**符号说明**:
- $\mathcal{L}_{\text{wm}}$: 特征空间重建损失（预测特征与真实未来特征的 L2 距离）
- $\mathcal{L}_{\text{aux}}$: 辅助 MLP 状态预测损失（训练后丢弃 MLP）
- $\beta = 10^{-5}$: KL 权重，控制潜在动作的正则化强度
- $g(\cdot)$: 辅助 MLP 状态预测器

### 公式5: [[LaWAM]] 训练目标

$$
\mathcal{L}_{\text{LaWAM}} = \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} + \lambda_{\text{wm}} \mathcal{L}_{\text{wm}} + \mathcal{L}_{\text{act}}
$$

其中：

$$
\mathcal{L}_{\text{distill}} = \mathbb{E}[\|\hat{z} - z\|_2^2]
$$

**含义**: Stage 2 联合训练损失，蒸馏项让策略先验学习预测正确的潜在动作，$\mathcal{L}_{\text{wm}}$ 保持 LaWM 的预测准确性，$\mathcal{L}_{\text{act}}$ 用条件流匹配训练动作专家

**符号说明**:
- $\mathcal{L}_{\text{distill}}$: 潜在动作蒸馏损失（策略先验与后验编码器输出的 L2 距离）
- $\lambda_{\text{distill}}, \lambda_{\text{wm}}$: 损失权重超参数
- $\mathcal{L}_{\text{act}}$: [[条件流匹配]] 动作生成损失

### 公式6: [[混合频率对齐|物理时间地平线对齐]]

$$
H_b = \text{round}(\tau \cdot h_b)
$$

**含义**: 将固定物理时间间隔 $\tau$ 转换为各控制频率分支的离散动作地平线

**符号说明**:
- $\tau$: 固定物理时间间隔（机器人 1.2s，人类视频 0.4s）
- $h_b$: 分支 $b$ 的控制频率（Hz）
- $H_b$: 离散动作地平线（步数）

### 公式7: [[正弦位置编码|物理时间编码]]

$$
\phi(t_{b,i}) = \text{Concat}\left[\sin(t_{b,i} \omega_k),\ \cos(t_{b,i} \omega_k)\right]_{k=0}^{K-1}
$$

其中 $t_{b,i} = i / h_b$，$\omega_k = \exp\!\left(-\frac{\log P_{\max} \cdot k}{\max(K-1, 1)}\right)$，$K = \lceil d/2 \rceil$

**含义**: 为每个动作 token 分配物理时间戳的正弦编码，使不同控制频率的数据在同一时间坐标系下对齐

**符号说明**:
- $t_{b,i}$: 第 $b$ 分支第 $i$ 步的物理时间（秒）
- $\omega_k$: 第 $k$ 维的角频率
- $P_{\max}$: 最大周期参数
- $d$: 编码维度

---

## 关键图表

### Figure 1: 延迟-成功率权衡

![Figure 1](https://arxiv.org/html/2606.15768v1/x1.png)

**说明**: LIBERO benchmark 上各方法的推理延迟（A100 GPU，10步去噪）与成功率的权衡图。标记面积表示模型大小，粉色扇区表示世界建模参数占比。LaWAM（2.3B）在 187ms 延迟下达到 98.6% 成功率，相比像素级 WAM 方法（Fast-WAM 486ms、LingBot-VA 4482ms、Motus 3231ms）大幅降低延迟，同时维持最高成功率。

### Figure 2: LaWAM 整体架构

![Figure 2](https://arxiv.org/html/2606.15768v1/x2.png)

**说明**: 两阶段训练概览。Stage 1 从视觉转移中学习潜在动作条件化世界模型：逆动力学编码器推断潜在动作，解码器（保留为 LaWM）预测未来观测特征。Stage 2 将 LaWM 集成进 VLA 策略：潜在动作蒸馏训练策略先验，LaWM 预测的潜在视觉子目标传入 Alternate-DiT 动作专家用于子目标条件化动作生成。

### Figure 3: 子目标引导动作块执行

![Figure 3](https://arxiv.org/html/2606.15768v1/x3.png)

**说明**: 上行展示一个 LIBERO 执行块内的观测序列及预测的潜在子目标；下行叠加子目标导出的机器人手臂热力图，说明执行动作如何向预测的子目标区域靠近，验证了 LaWM 子目标的空间引导效果。

### Figure 4: 真实世界操作演示

![Figure 4](https://arxiv.org/html/2606.15768v1/x4.png)

**说明**: 在两个机器人平台上的 pick-and-place、开抽屉和叠毛巾三类任务的代表性真实世界运行片段。

### Figure 5: 跨具身开环 LaWM 推演

![Figure 5](https://arxiv.org/html/2606.15768v1/x5.png)

**说明**: 跨具身泛化验证。(a) 从源视频推断潜在动作轨迹；(b)-(e) 给定不同环境/具身的单帧初始观测，LaWM 应用相同的提取潜在动作生成各具身特定的潜在推演。(d)(e) 使用未见过的场景截图，验证了潜在动作的具身无关性与具身特定落地能力的同时存在。

### Figure 6: 组件消融（LIBERO）

![Figure 6](https://arxiv.org/html/2606.15768v1/x6.png)

**说明**: LIBERO 上各组件贡献消融。结果显示：移除 LaWM 导致最大性能下降；潜在动作蒸馏对性能有实质性影响；Knowledge Insulation 防止灾难性遗忘至关重要；预训练同样不可或缺。

### Figure 7: 混合频率训练（LIBERO）

![Figure 7](https://arxiv.org/html/2606.15768v1/x7.png)

**说明**: 在从同一 LIBERO 数据下采样的 5/10/20 Hz 版本上联合训练的结果，证明物理时间编码有效解决了控制频率混淆问题，联合训练性能优于单频率训练。

### Figure 8: LIBERO 定性示例

![Figure 8](https://arxiv.org/html/2606.15768v1/x8.png)

**说明**: LaWAM 在 LIBERO 上的动作执行与潜在世界推演定性展示。预测的潜在子目标提供紧凑的未来动力学，无需迭代像素级视频预测即可引导动作生成。

### Figure 9: 真实机器人平台配置

![Figure 9a - Franka](https://arxiv.org/html/2606.15768v1/x9.png)

![Figure 9b - Quanta X1](https://arxiv.org/html/2606.15768v1/picture/arx_robot.jpg)

**说明**: (a) Franka Emika Panda 单臂配置；(b) Quanta X1 双臂配置，用于测试双臂协调和长时序灵巧操作任务。

### Figure 10: LaWM 推演误差曲线（500 条 LIBERO 轨迹）

![Figure 10](https://arxiv.org/html/2606.15768v1/picture/error_curves.png)

**说明**: 500 条 LIBERO 轨迹的平均 LaWM 推演结果。蓝色曲线（推演 vs 真实未来状态余弦相似度）持续保持高值，而绿色曲线（推演 vs 初始状态）随时间下降，证明 LaWM 确实追踪真实潜在动力学，而非停留在初始观测附近。

### Figure 11: 真实 Pick-and-Place 各块后观测

![Figure 11](https://arxiv.org/html/2606.15768v1/picture/pp_duck_all.jpg)

**说明**: Franka 平台上抓取玩具鸭 pick-and-place 任务的各动作块后观测，叠加未来相似度热力图和 LaWM 子目标 PCA 可视化，展示动作专家如何跟随 LaWM 子目标完成抓取、运输和放置。

### Figure 12: 真实开抽屉各块后观测

![Figure 12](https://arxiv.org/html/2606.15768v1/picture/open_drawer_all.jpg)

**说明**: Franka 平台上开抽屉任务的各动作块后观测，机器人手臂热力图与预测子目标高度重叠，验证动作专家对 LaWM 子目标的跟踪准确性。

### Figure 13: 真实叠毛巾各块后观测

![Figure 13](https://arxiv.org/html/2606.15768v1/picture/fold_towel_all.jpg)

**说明**: Quanta X1 双臂平台上叠毛巾任务的各块后观测。任务需先抖平毛巾，再完成两次长边折叠和一次短边折叠，测试长时序双臂可变形物体操作，是最具挑战性的任务之一。

### Figure 14: 单臂跨具身推演

![Figure 14](https://arxiv.org/html/2606.15768v1/x10.png)

**说明**: 单臂场景下的跨具身 LaWM 开环推演结果，展示相同潜在动作在不同单臂具身环境中的迁移能力。

### Figure 15: 双臂跨具身推演

![Figure 15](https://arxiv.org/html/2606.15768v1/x11.png)

**说明**: 双臂场景下的跨具身推演。(e)-(j) 为未见过的场景；(f)-(j) 使用 pi.website 截图，进一步验证 LaWM 对未见双臂具身的泛化能力。

---

### Table 1: LIBERO Benchmark 结果

| 方法 | 模型大小 | 延迟(ms) | Long | Goal | Object | Spatial | 平均 |
|------|---------|---------|------|------|--------|---------|------|
| OpenVLA-OFT | 7B | — | 94.5 | 97.9 | 98.4 | 97.6 | 97.1 |
| π₀ | 3.5B | 220 | 88.4 | 94.4 | 96.8 | 98.0 | 94.4 |
| π₀.₅ | 3.5B | 220 | 92.4 | 98.0 | 98.2 | 98.8 | 96.9 |
| GR00T-N1.6 | 3.3B | 259 | 94.4 | 97.5 | 98.5 | 97.7 | 97.0 |
| LAPA | 7B | — | 55.4 | 58.8 | 74.6 | 73.8 | 65.7 |
| UniVLA | 7B | — | 92.0 | 95.6 | 96.8 | 96.5 | 95.2 |
| Mantis | 5.8B | — | 94.2 | 94.4 | 99.2 | 98.8 | 96.7 |
| VLA-JEPA | 3B | — | 95.8 | 97.2 | 99.6 | 96.2 | 97.2 |
| F1 | 4B | 399 | 91.3 | 95.4 | 97.8 | 98.2 | 95.7 |
| Motus | 8B | 3231 | 97.6 | 96.6 | 99.8 | 96.8 | 97.7 |
| Cosmos-Policy | 2.1B | 1413 | 97.6 | 98.2 | 100.0 | 98.1 | 98.5 |
| LingBot-VA | 5.5B | 4482 | 98.5 | 97.2 | 99.6 | 98.5 | 98.5 |
| Fast-WAM | 6B | 486 | 95.2 | 97.0 | 100.0 | 98.2 | 97.6 |
| **LaWAM** | **2.3B** | **187** | **97.0** | **98.4** | **99.6** | **99.4** | **98.6** |

**关键发现**: LaWAM 以最小的模型（2.3B）、最低的延迟（187ms）达到最高成功率（98.6%），相比延迟最接近的 π₀/π₀.₅（220ms）提升 1.7pp，相比像素级 WAM 方法（Fast-WAM 486ms）延迟降低 2.6×。

### Table 2: RoboTwin Benchmark 结果（汇总）

| 方法 | Clean（%） | Rand.（%） |
|------|-----------|-----------|
| Fast-WAM | 91.98 | 90.52 |
| GigaWorld-Policy | 86.36 | 85.04 |
| LingBot-VA | 91.50 | 90.92 |
| π₀.₅ | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| **LaWAM** | **92.64** | **89.80** |

**关键发现**: LaWAM 在 50 个操作任务的 Clean 场景下以 92.64% 排名第一，随机化场景下以 89.80% 排名第二（略低于 LingBot-VA 的 90.92%，但 LingBot-VA 延迟高达 4482ms）。

RoboTwin 代表性任务数据（部分）：

| 任务 | Fast-WAM C | Fast-WAM R | GigaWorld C | GigaWorld R | LingBot-VA C | LingBot-VA R | π₀.₅ C | π₀.₅ R | Motus C | Motus R | LaWAM C | LaWAM R |
|------|---|---|---|---|---|---|---|---|---|---|---|---|
| Move Can Pot | 95 | 92 | 76 | 78 | 93 | 93 | 51 | 55 | 34 | 74 | **98** | 93 |
| Move Stapler Pad | 84 | 63 | 92 | 82 | 59 | 71 | 56 | 42 | 83 | 85 | **94** | **87** |
| Open Laptop | 100 | 100 | 96 | 98 | 96 | 90 | 90 | 96 | 95 | 91 | **100** | **100** |
| Pick Dual Bottles | 100 | 94 | 86 | 86 | **100** | **100** | 93 | 63 | 96 | 90 | **100** | 95 |
| Place A2B Left | 94 | 89 | 94 | 88 | 96 | 94 | 87 | 82 | 88 | 79 | **98** | 91 |
| Place Container Plate | 98 | **100** | 98 | 96 | 97 | 98 | 99 | 95 | 98 | 99 | **100** | **100** |
| Place Dual Shoes | 88 | 88 | **96** | 84 | 94 | 81 | 75 | 75 | 93 | 87 | **98** | **94** |
| Place Object Basket | 82 | 90 | **90** | **92** | 89 | 85 | 80 | 76 | 81 | 87 | 92 | 90 |
| Place Object Scale | 83 | 88 | 88 | 80 | **98** | 87 | 86 | 80 | 88 | 85 | 95 | **88** |
| Place Shoe | 97 | 98 | 98 | 96 | 99 | 97 | 92 | 93 | **99** | 97 | **100** | **100** |
| Put Bottles Dustbin | 93 | 82 | 72 | 70 | 82 | 87 | 84 | 79 | 81 | 79 | **94** | **92** |
| Put Object Cabinet | 94 | 82 | 74 | 74 | 87 | 80 | 80 | 79 | 88 | 71 | **90** | 82 |
| Scan Object | **96** | 86 | 60 | 64 | 90 | 89 | 72 | 65 | 67 | 66 | **96** | **90** |
| Stack Bowls Two | 90 | 96 | 96 | 92 | 98 | **99** | 95 | 96 | 98 | 98 | **100** | **99** |

### Table 3: 真实世界成功率（%）

| 方法 | Pick-and-Place | Open Drawer | Fold Towel | 平均 |
|------|---|---|---|---|
| π₀.₅ | 86.7 | 80.0 | 83.3 | 83.3 |
| GR00T-N1.6 | 83.3 | 76.7 | 46.7 | 68.9 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | 0.0 | 53.3 |
| **LaWAM** | **93.3** | **86.7** | **90.0** | **90.0** |

**关键发现**: LaWAM 在三项真实任务上全面领先，尤其在叠毛巾（可变形物体操作）上以 90.0% 对比 Fast-WAM 70.0% 和 LingBot-VA 0.0% 的差距最为显著。

---

## 实验

### 数据集

| 数据集 | 来源 | 类型 | 用途 |
|--------|------|------|------|
| Open X-Embodiment | Google 等 | 多具身机器人视频 | LaWM 预训练 |
| DROID | Stanford 等 | 机器人操作视频 | LaWM 预训练 |
| RoboCOIN | — | 机器人视频 | LaWM 预训练 |
| Robomind | — | 机器人视频 | LaWM 预训练 |
| LEGO | — | 机器人视频 | LaWM 预训练 |
| Ego4D | Meta | 自我中心人类视频 | LaWM 预训练 |
| EgoDex | — | 自我中心手部操作 | LaWM 预训练 |
| AgiBot World Colosseo | AgiBot | 机器人数据 | LaWM 预训练 |
| LIBERO | — | 仿真操作基准 | 策略训练/评估 |
| RoboTwin | — | 50 任务仿真基准 | 策略训练/评估 |

### 实现细节

- **视觉编码器**: [[DINOv3]] ViT-B/16（冻结）
- **政策 Backbone**: Qwen-GR00T（Qwen3-VL 前 16 层）
- **LaWM 预训练优化器**: AdamW，学习率 $3\times10^{-4}$，100k 步，16×H100
- **策略联合训练**: 200k 步，64×H100，全局 batch size 1024
- **KL 权重**: $\beta = 10^{-5}$
- **物理时间间隔**: 机器人 $\tau = 1.2$s，人类视频 $\tau = 0.4$s
- **观测分辨率**: RGB 256×256
- **推理延迟**: 187ms（A100 GPU，10 步 [[Flow Matching]] 去噪）

### 可视化结果

Figure 3、11、12、13 展示了 LaWM 预测的潜在子目标热力图与实际机器人手臂执行轨迹高度重合，定性验证了动作专家确实在跟随 LaWM 的动力学预测。Figure 10 的误差曲线从量化角度证明 LaWM 追踪真实潜在动力学而非退化为初始观测的平凡预测。

---

## 批判性思考

### 优点

1. **效率-性能帕累托最优**: 2.3B 参数配合 187ms 延迟达到 LIBERO SOTA，打破了像素级 WAM 的延迟-性能权衡
2. **跨具身泛化**: 在人类自我中心视频上预训练的潜在动作表示能有效迁移至未见机器人具身（Figure 5、14、15）
3. **混合频率联合训练**: 物理时间编码优雅解决了不同机器人平台频率不一致的问题，无需为每个频率单独训练
4. **原则性模块化**: Knowledge Insulation 在实践中确保了 LaWM 预训练知识的保护，消融实验证明其必要性

### 局限性

1. **相机运动敏感**: 对自我中心抖动或视角剧变鲁棒性差，限制了移动底座机器人的应用场景
2. **可变形物体数据不足**: 训练数据中细粒度织物变形数据稀缺，影响精细布料操作（如复杂叠衣动作）
3. **代码未开源**: 论文未提供代码和预训练权重，可复现性受限
4. **RoboTwin 随机场景略弱**: 随机化场景 89.80%（第二）低于 LingBot-VA 的 90.92%（第一），说明对未知扰动的鲁棒性仍有提升空间

### 潜在改进方向

1. **时间/视角增强**: 在预训练中引入更多相机运动数据，提升移动具身鲁棒性
2. **接触感知潜在动作**: 整合触觉传感器信号到潜在动作编码，改善可变形物体操作
3. **在线潜在动作更新**: 探索策略执行过程中根据感知反馈在线修正 $\hat{z}$，提升动态环境适应性

### 可复现性评估

- [ ] 代码开源（❌ 未提供）
- [ ] 预训练模型（❌ 未提供）
- [x] 训练细节完整（✅ Appendix 详细描述超参数）
- [x] 数据集可获取（✅ 大部分为公开数据集）

---

## 关联笔记

### 基于

- [[World-Action Model]]: LaWAM 属于 WAM 系列方法的潜在空间分支
- [[Latent Action]]: Stage 1 的核心技术
- [[DINOv3]]: 视觉特征提取 Backbone（冻结）
- [[条件流匹配]]: 动作专家的训练目标

### 对比

- [[Fast-WAM]]: 同为 WAM 方法，但像素级预测导致 486ms 延迟，LaWAM 以 187ms 延迟超越
- [[π0.5]]: 主要直接 VLA 基线，无显式动力学建模
- [[GR00T-N1.6]]: NVIDIA 的 VLA，LaWAM 基于其 Qwen-GR00T Backbone 构建

### 方法相关

- [[Alternate-DiT]]: LaWAM 动作专家的核心架构
- [[逆动力学编码器]]: LaWM Stage 1 的关键组件
- [[Knowledge Insulation]]: 防止灾难性遗忘的训练技巧
- [[V-JEPA2]]: LaWM 编码器的设计参考

### 硬件/数据相关

- [[LIBERO]]: 主要仿真评估基准（4 套件，共 40 任务）
- [[RoboTwin]]: 50 任务大规模仿真基准
- [[Franka Emika Panda]]: 单臂真实机器人平台
- [[Quanta X1]]: 双臂真实机器人平台

---

## 速查卡片

> [!summary] LaWAM (2606.15768)
> - **核心**: 用潜在视觉子目标替代像素级视频预测，高效实现动力学感知机器人策略
> - **方法**: 两阶段——先预训练 LaWM（逆动力学+解码器），再将 LaWM 潜在子目标条件化 Alternate-DiT 动作专家
> - **结果**: LIBERO 98.6%（SOTA），RoboTwin Clean 92.64%（第一），真实任务 90.0%，延迟 187ms（快 24×）
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-16*
