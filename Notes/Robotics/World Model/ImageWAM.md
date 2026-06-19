---
title: "ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?"
method_name: "ImageWAM"
authors: [Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, Xin Jin]
year: 2026
venue: arXiv
tags: [world-action-model, image-editing, robot-manipulation, flow-matching, action-prediction]
zotero_collection: 3-Robotics/World-Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.19531
created: 2026-06-19
---

# 论文笔记：ImageWAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多家单位（作者来自不同机构） |
| 日期 | June 2026 |
| 项目主页 | [zhangwenyao1.github.io/ImageWAM](https://zhangwenyao1.github.io/ImageWAM/) |
| 对比基线 | [[FastWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19531) / Code: N/A |

---

## 一句话总结

> 用图像编辑模型替代视频生成模型作为 WAM 骨干，只需预测单帧目标画面，将推理 FLOPs 降至原来的 1/6、延迟降至 1/4，同时在多个基准上超越视频生成 WAM。

---

## 核心贡献

1. **重新定义 WAM 代理任务**: 将机器人操作建模为"指令引导的视觉变换"而非密集视频预测，用单帧编辑目标帧代替多帧视频生成
2. **ImageWAM 框架**: 将预训练图像编辑模型（OmniGen2、Ovis-U1、FLUX.2）作为 WAM 骨干，通过 [[KV Cache]] 复用将编辑表示传递给动作专家
3. **高效推理**: 推理时无需生成任何视频帧，直接从编辑缓存预测动作，实现 FLOPs 1/6、延迟 1/4 的显著提升

---

## 问题背景

### 要解决的问题

[[World-Action Model|WAM（World Action Model）]] 是当前机器人控制的主流范式：通过预测未来视频帧来辅助动作预测。但这一范式存在根本性的效率问题：机器人只需知道"接下来做什么动作"，却要生成完整的未来视频序列。

### 现有方法的局限

基于视频生成的 [[World-Action Model|WAM]] 存在三个核心问题：

1. **计算开销大**: 密集多帧未来 token 预测代价昂贵
2. **容量浪费**: 全视频预测需要对动作无关的视觉细节（背景、光照等）建模，浪费模型容量
3. **误差累积**: 长时序视觉生成中的幻觉会误导动作预测（见 Figure 5）

### 本文的动机

图像编辑模型天然适合机器人操作：

- **指令-变化对齐**: 图像编辑预训练将语言指令与视觉变化紧密耦合，天然匹配"根据指令改变场景"的操作语义
- **更简单的代理任务**: 只需建模从当前帧到目标帧的变换，比预测完整视频序列容易得多
- **紧凑推理**: 推理时只需单次前向传播获取 KV 缓存，无需解码视频帧

---

## 方法详解

### 任务公式化对比

**传统 WAM 流水线**:

$$
(o_t, l) \rightarrow \hat{o}_{t+1:t+H+1} \rightarrow a_{t:t+H}
$$

**ImageWAM 流水线**:

$$
(o_t, l) \rightarrow \hat{o}_{\text{edit}} \equiv \hat{o}_{t+H+1} \rightarrow a_{t:t+H}
$$

其中 $\hat{o}_{\text{edit}}$ 是以当前帧为源、指令为条件的单帧目标编辑结果，而非完整视频。

### 模型架构

ImageWAM 采用**解耦双分支**架构：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$ + 机器人状态
- **图像编辑骨干**: 预训练图像编辑模型（OmniGen2 / Ovis-U1 / FLUX.2），生成目标帧 $\hat{o}_{t+H+1}$
- **核心桥梁**: [[KV Cache Conditioning]] — 从编辑骨干的去噪过程中提取 KV 缓存
- **动作专家**: 通过 [[Joint Self-Attention]] 整合编辑 KV 缓存，预测 [[Action Chunking|动作块]] $a_{t:t+H}$
- **输出**: 未来 H 步动作序列

### 核心模块

#### 模块1: 编辑缓存提取（KV Cache Extraction）

**设计动机**: 不直接传递生成的图像像素，而是复用图像编辑模型内部的注意力表示。这些 KV 缓存已经编码了"任务相关的视觉变化"信息（通过注意力可视化验证，Figure 4）。

**具体实现**:

训练时，从采样的去噪时间步 $\tau$ 收集每层 [[Transformer]] 的 KV 缓存：

$$
C_{\text{edit}}^{\tau} = \{(K_\ell^\tau, V_\ell^\tau)\}_{\ell=1}^{L} = f_{\text{edit}}^{\tau}(o_t, l)
$$

- $L$：[[Diffusion Transformer]] 的层数
- $K_\ell^\tau, V_\ell^\tau$：第 $\ell$ 层在时间步 $\tau$ 的键值缓存
- $f_{\text{edit}}^{\tau}$：图像编辑骨干的前向函数

**推理时**: 使用固定时间步 $\tau^*$（无需随机采样），只做一次前向传播即可获取缓存，无需真正解码图像。

#### 模块2: 图像编辑分支（Diffusion Image Branch）

**设计动机**: 利用 [[Flow Matching|流匹配]] 目标函数，训练编辑骨干生成任务相关的目标帧，同时保持预训练图像编辑能力。

**具体实现**:

构造带噪声的目标帧潜变量（[[Conditional Flow Matching|条件流匹配]] 插值）：

$$
z_r = (1-r) z^*_{t+H+1} + r\varepsilon_z
$$

其中：
- $z^*_{t+H+1} = E_{\text{vae}}(o_{t+H+1})$：目标帧经 [[VAE（变分自编码器）|VAE]] 编码后的潜变量
- $\varepsilon_z \sim \mathcal{N}(0, I)$：高斯噪声
- $r \in (0,1)$：图像流时间步

图像编辑损失（[[Flow Matching|流匹配]] 目标）：

$$
\mathcal{L}_{\text{img}} = \mathbb{E}_{z^*, \varepsilon_z, r}\left[\left\| u_\phi(z_r, r \mid o_t, l) - (\varepsilon_z - z^*_{t+H+1}) \right\|^2_2\right]
$$

其中 $u_\phi$ 是图像扩散分支的速度预测器。

#### 模块3: 动作专家（Action Expert）

**设计动机**: 用 [[Flow Matching|流匹配]] 直接预测连续动作序列，通过 [[Joint Self-Attention]] 整合来自图像编辑骨干的任务感知表示。

**具体实现**:

构造带噪声的动作（[[Conditional Flow Matching|条件流匹配]] 插值）：

$$
a_s = (1-s) a^*_{t:t+H} + s\varepsilon_a
$$

其中：
- $a^*_{t:t+H}$：专家示范的动作块（[[Action Chunking|动作分块]]）
- $\varepsilon_a \sim \mathcal{N}(0, I)$：高斯噪声
- $s \in (0,1)$：动作流时间步

动作流匹配损失：

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{a^*, \varepsilon_a, s, \tau}\left[\left\| v_\theta(a_s, s \mid o_t, l, C_{\text{edit}}^{\tau}) - (\varepsilon_a - a^*_{t:t+H}) \right\|^2_2\right]
$$

其中 $v_\theta$ 是动作专家的速度预测器。

### 联合训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{img}}
$$

两个损失联合训练，确保图像编辑骨干保持任务相关的视觉预测能力，同时动作专家学会利用编辑缓存。

### 高效推理流程

推理时无需生成视频，只需：

1. 单次前向传播获取编辑缓存：$C_{\text{edit}}^{\tau^*} = f_{\text{edit}}^{\tau^*}(o_t, l)$
2. 动作专家基于缓存预测动作：$\hat{a}_{t:t+H} \sim p_\theta(a_{t:t+H} \mid o_t, l, C_{\text{edit}}^{\tau^*})$

---

## 关键公式

### 公式1: [[KV Cache Conditioning|编辑缓存提取]]

$$
C_{\text{edit}}^{\tau} = \{(K_\ell^\tau, V_\ell^\tau)\}_{\ell=1}^{L} = f_{\text{edit}}^{\tau}(o_t, l)
$$

**含义**: 从图像编辑骨干的第 $\tau$ 步去噪过程中收集所有层的 KV 缓存，作为动作专家的条件信号

**符号说明**:
- $C_{\text{edit}}^{\tau}$：去噪时间步 $\tau$ 下的完整 KV 缓存集合
- $K_\ell^\tau, V_\ell^\tau$：第 $\ell$ 层在时间步 $\tau$ 的键（Key）和值（Value）矩阵
- $L$：Transformer 总层数
- $f_{\text{edit}}^{\tau}$：图像编辑骨干的前向函数

### 公式2: [[Conditional Flow Matching|图像流匹配插值]]

$$
z_r = (1-r) z^*_{t+H+1} + r\varepsilon_z
$$

**含义**: 在目标帧潜变量与纯噪声之间线性插值，构造带噪声的训练样本

**符号说明**:
- $z_r$：流时间步 $r$ 处的插值潜变量
- $z^*_{t+H+1} = E_{\text{vae}}(o_{t+H+1})$：目标未来帧的 VAE 潜变量
- $\varepsilon_z \sim \mathcal{N}(0, I)$：标准高斯噪声
- $r \in (0,1)$：图像流时间步

### 公式3: [[Flow Matching|图像编辑损失]]

$$
\mathcal{L}_{\text{img}} = \mathbb{E}_{z^*, \varepsilon_z, r}\left[\left\| u_\phi(z_r, r \mid o_t, l) - (\varepsilon_z - z^*_{t+H+1}) \right\|^2_2\right]
$$

**含义**: 训练图像扩散分支预测从带噪样本到目标帧的速度场

**符号说明**:
- $u_\phi$：图像扩散分支的速度预测器（参数为 $\phi$）
- $(\varepsilon_z - z^*_{t+H+1})$：目标速度（从噪声指向干净样本的方向）

### 公式4: [[Conditional Flow Matching|动作流匹配插值]]

$$
a_s = (1-s) a^*_{t:t+H} + s\varepsilon_a
$$

**含义**: 在专家动作与噪声之间线性插值，构造动作流匹配的训练样本

**符号说明**:
- $a_s$：动作流时间步 $s$ 处的插值动作
- $a^*_{t:t+H}$：专家示范的未来 H 步动作块
- $\varepsilon_a \sim \mathcal{N}(0, I)$：标准高斯噪声
- $s \in (0,1)$：动作流时间步

### 公式5: [[Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{a^*, \varepsilon_a, s, \tau}\left[\left\| v_\theta(a_s, s \mid o_t, l, C_{\text{edit}}^{\tau}) - (\varepsilon_a - a^*_{t:t+H}) \right\|^2_2\right]
$$

**含义**: 训练动作专家在编辑缓存条件下预测动作速度场，实现精确动作预测

**符号说明**:
- $v_\theta$：动作专家的速度预测器（参数为 $\theta$）
- $C_{\text{edit}}^{\tau}$：采样时间步 $\tau$ 的编辑 KV 缓存
- $(\varepsilon_a - a^*_{t:t+H})$：目标速度（从噪声指向干净动作的方向）

### 公式6: 联合训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{img}}
$$

**含义**: 同时优化图像编辑质量和动作预测精度，两个任务相互促进

---

## 关键图表

### Figure 1: 方法对比概览

![Figure 1](https://arxiv.org/html/2606.19531/2606.19531v1/x1.png)

**说明**: 对比视频生成 WAM 与 ImageWAM 的核心差异。左侧：传统视频 WAM 需要预测密集的多帧未来 token，计算开销大且可能包含大量动作无关信息。右侧：ImageWAM 用图像编辑骨干预测单帧目标变换，只建模"当前到目标"的视觉差异，更聚焦于动作相关的变化区域。

### Figure 2: ImageWAM 流水线

![Figure 2](https://arxiv.org/html/2606.19531/2606.19531v1/x2.png)

**说明**: ImageWAM 的完整训练流水线。给定语言指令 $l$ 和当前观测 $o_t$，图像编辑骨干通过 [[Flow Matching|流匹配]] 合成目标帧 $\hat{o}_{t+H+1}$。动作专家通过 [[Joint Self-Attention]] 整合来自去噪过程的中间 KV 特征，在当前机器人状态和动作噪声的条件下预测未来动作序列 $a_{t:t+H}$。注意推理时无需真正解码目标图像。

### Figure 3: 实验设置

![Figure 3](https://arxiv.org/html/2606.19531/2606.19531v1/x3.png)

**说明**: 四个评测环境的可视化。从左到右：RoboTwin 2.0（仿真，多任务操作）、LIBERO（仿真，四个子任务套件）、LIBERO-Plus（仿真，多维度鲁棒性扰动）、真实机器人（4 个桌面操作任务，每任务约 100 条示教轨迹）。

### Figure 4: 注意力可视化

![Figure 4](https://arxiv.org/html/2606.19531/2606.19531v1/x4.png)

**说明**: 图像编辑缓存的注意力热图分析，验证 ImageWAM 的核心假设。编辑缓存的注意力集中在"任务相关变化区域"（如被操作物体），而非整个场景，这证明图像编辑表示天然聚焦于动作相关信息，比密集视频预测更精准。

### Figure 5: 视频 WAM 的幻觉问题

![Figure 5](https://arxiv.org/html/2606.19531/2606.19531v1/x5.png)

**说明**: 展示视频生成 WAM 的失效案例。视频 WAM 基线在任务相关物体周围生成了扭曲/失真的未来观测，这种不可靠的动作条件信号导致任务执行失败。这一现象验证了长时序视频生成中的误差累积会误导动作预测。

### Table 1: RoboTwin 2.0 结果

| 方法 | 预训练 | Clean | Rand. | Avg. |
|------|--------|-------|-------|------|
| π₀ | ✓ | 65.92 | 58.40 | 62.16 |
| π₀.₅ | ✓ | 82.74 | 76.76 | 79.75 |
| ABot-M0 | ✗ | 81.20 | 80.40 | 80.80 |
| Motus | ✓ | 88.66 | 87.02 | 87.80 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| FastWAM | ✗ | 91.88 | 91.78 | 91.83 |
| **ImageWAM** | **✗** | **93.20** | **93.56** | **93.38** |

**说明**: ImageWAM 在不使用额外预训练的情况下，在 RoboTwin 2.0 上超越所有基线，包括使用预训练的 LingBot-VA（92.20%），整体平均 93.38%。

### Table 2: LIBERO 结果

| 方法 | 预训练 | Spatial | Object | Goal | Long | Avg. |
|------|--------|---------|--------|------|------|------|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| GR00T N1 | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| **ImageWAM** | **✗** | **97.2** | **99.2** | **98.8** | **98.4** | **98.4** |

**说明**: ImageWAM 在 LIBERO 四个子套件上均表现优异，平均 98.4%，超越同样无预训练的 Fast-WAM（97.6%），接近使用预训练的最佳方法 LingBot-VA（98.5%）。

### Table 3: LIBERO-Plus 鲁棒性结果

| 方法 | 预训练 | Camera | Robot | Language | Light | BG | Noise | Layout | Avg |
|------|--------|--------|-------|----------|-------|-----|-------|--------|-----|
| UniVLA | ✓ | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | ✓ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | ✓ | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| FastWAM | ✗ | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| ImageWAM (OmniGen2) | ✗ | 80.0 | 49.2 | 70.9 | 82.6 | 69.4 | 77.1 | 71.8 | 71.8 |
| ImageWAM (Ovis-U1) | ✗ | 63.3 | 58.4 | 75.4 | 86.3 | 66.7 | 75.2 | 74.6 | 71.2 |
| **ImageWAM (FLUX.2 4B)** | **✗** | **80.8** | **50.3** | **91.4** | **98.1** | **85.5** | **93.8** | **80.5** | **83.1** |

**说明**: LIBERO-Plus 测试 7 种扰动下的鲁棒性。ImageWAM (FLUX.2 4B) 以 83.1% 的平均成功率大幅超越所有基线（次优为 OpenVLA-OFT 的 69.6%），在光照、噪声、语言扰动上表现尤为突出，验证了图像编辑预训练带来的强泛化能力。

### Table 4: 真实机器人评测

| 方法 | T1 | T2 | T3 | T4 | Avg |
|------|----|----|----|----|-----|
| π₀ | 57 | 58 | 54 | 54 | 55.8 |
| π₀.₅ | 83 | 77 | 74 | 55 | 72.3 |
| FastWAM | 88 | 75 | 77 | 76 | 79.0 |
| **ImageWAM** | **94** | **84** | **78** | **82** | **84.5** |

**说明**: 在 4 个真实机器人任务上，ImageWAM 平均成功率 84.5%，超越 FastWAM（79.0%）和 π₀.₅（72.3%），验证了方法在真实环境中的有效性。

### Table 5: 效率对比

| 方法 | 延迟 (ms) | TFLOPs | 中间表示 |
|------|-----------|--------|----------|
| FastWAM-IDM | 1081 | 63.65 | 视频 |
| FastWAM (1 Step) | 302 | 13.21 | Cache |
| **ImageWAM** | **263** | **9.72** | Cache |

**说明**: ImageWAM 的 FLOPs 仅为视频生成 WAM（FastWAM-IDM）的 1/6.5（9.72 vs 63.65），推理延迟降至 1/4（263ms vs 1081ms），同时性能更优。即使对比单步 FastWAM，也实现了约 15% 的延迟降低和 26% 的 FLOPs 减少。

### Table 6: 解耦设计 vs 统一理解-生成模型

| 方法 | 预训练 | LIBERO | RoboTwin Clean | RoboTwin C2H |
|------|--------|--------|----------------|--------------|
| UniVLA | ✓ | 95.5 | — | — |
| BagelVLA (w/ K.F.) | ✓ | — | 75.3 | 20.9 |
| BagelVLA (w/o K.F.) | ✓ | — | 56.7 | 15.9 |
| **ImageWAM** | **✗** | **98.4** | **84.4** | **18.3** |

**说明**: 与使用统一理解-生成模型（同时理解和生成图像）的方法对比。ImageWAM 的解耦设计（专用图像编辑骨干 + 独立动作专家）在 LIBERO 上大幅超越 UniVLA（98.4% vs 95.5%），验证了解耦设计的优越性。

### Table 7: 编辑骨干规模消融

| 方法 | 预训练 | Camera | Robot | Language | Light | BG | Noise | Layout | Avg |
|------|--------|--------|-------|----------|-------|-----|-------|--------|-----|
| ImageWAM (FLUX.2 4B) | ✗ | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 |
| **ImageWAM (FLUX.2 9B)** | **✗** | **79.8** | **58.7** | **95.2** | **96.1** | **91.2** | **93.3** | **83.1** | **85.2** |

**说明**: 更大的编辑骨干（9B vs 4B）在 LIBERO-Plus 上进一步提升鲁棒性（85.2% vs 83.1%），尤其在 Robot 和 Background 扰动上提升明显，表明骨干规模是提升泛化的有效途径。

### Table 8: 推理延迟详细对比

| 配置 | 延迟 (ms) | 加速比 |
|------|-----------|--------|
| FastWAM (1× Vid. Denoise) | 302 | 1.00× |
| ImageWAM (1× Vid. Denoise) | 263 | 1.15× |
| FastWAM (Prefix Only) | 194 | 1.56× |
| + Compiled | 80 | 3.78× |
| ImageWAM (Prefix Only) | 198 | 1.53× |
| + Action Loop Compile | 85 | 3.55× |
| + Image Prefill Compile | 77 | 3.92× |
| + Action Static Graph | 69 | 4.38× |

**说明**: 通过逐步编译优化，ImageWAM 可将延迟从 263ms 降至 69ms（4.38× 加速），全面超越 FastWAM 在相同优化层级下的速度。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 多任务仿真 | Clean / Randomized 两种难度 | 训练 + 测试 |
| LIBERO | 4 个子套件 | Spatial / Object / Goal / Long | 训练 + 测试 |
| LIBERO-Plus | 7 种扰动 | Camera / Robot / Language / Light / BG / Noise / Layout | 鲁棒性测试 |
| 真实机器人 | 4 任务，~100 条/任务 | 桌面操作，真实环境 | 真实机器人评测 |

### 实现细节

- **编辑骨干**: OmniGen2、Ovis-U1、FLUX.2 4B、FLUX.2 9B（多种选择）
- **优化器**: AdamW，betas=(0.9, 0.95)，weight decay=1×10⁻²
- **学习率**: 1×10⁻⁴，warmup cosine scheduler
- **梯度裁剪**: 1.0
- **精度**: bf16
- **分布式策略**: DeepSpeed ZeRO-1（9B 用 ZeRO-2）
- **硬件**: 8× NVIDIA H20 GPU
- **LIBERO**: 输入 224×448，训练 10 epochs，batch 10-16/GPU，~18 小时
- **RoboTwin**: 输入 288×256，训练 5 epochs，batch 48/GPU（梯度累积），~5 天
- **真实机器人**: 训练 10 epochs，~18 小时
- **动作块长度**: 16 帧，预测视野 H=16

### 可视化结果

注意力可视化（Figure 4）直接证明了核心假设：图像编辑缓存中的注意力权重高度集中在任务相关的物体变化区域，而非背景或无关场景元素，这与视频 WAM 需要建模完整场景动态形成鲜明对比。

---

## 批判性思考

### 优点

1. **问题定义精准**: 准确识别了视频 WAM 的根本冗余——机器人策略不需要视频，只需要知道"目标状态"和"如何到达"
2. **实践有效性强**: 在仿真和真实机器人上均有充分验证，结果令人信服
3. **效率提升显著**: 6× FLOPs 减少和 4× 延迟降低对实际部署意义重大
4. **骨干选择灵活**: 支持多种图像编辑模型（OmniGen2、FLUX.2、Ovis-U1），验证了框架的通用性

### 局限性

1. **单帧目标假设**: 只预测终止帧，对需要精细中间状态控制的任务（如复杂装配）可能不足
2. **图像编辑质量依赖**: 性能上限受限于图像编辑骨干的质量，当编辑骨干产生不准确预测时可能影响动作
3. **固定时间步选择**: 推理时固定 $\tau^*$ 的选择策略未深入讨论，可能影响不同任务的最优性
4. **长时序任务**: 对于需要多步规划的长时序任务，单帧目标是否足够仍有待验证

### 潜在改进方向

1. **自适应时间步选择**: 根据任务难度动态选择 $\tau^*$，提升不同任务的适应性
2. **多帧目标融合**: 对于复杂任务，预测少量关键帧而非单帧，在效率和精度间取得更好平衡
3. **在线微调**: 结合真实机器人的在线交互数据持续改进

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码状态未确认）
- [ ] 预训练模型（未确认）
- [x] 训练细节完整（论文附录提供完整超参数）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均公开可用）

---

## 关联笔记

### 基于

- [[World-Action Model]]: WAM 框架基础
- [[Flow Matching]]: 图像和动作预测的核心训练目标
- [[Conditional Flow Matching]]: 具体使用的流匹配变体

### 对比

- [[FastWAM]]: 主要视频生成 WAM 基线，ImageWAM 在效率和性能上均超越它
- [[Fast-WAM]]: 单步视频 WAM 变体，直接效率对比
- [[Embodied World Model]]: 通用 WAM 概念框架

### 方法相关

- [[KV Cache Conditioning]]: 编辑缓存传递给动作专家的核心机制
- [[Joint Self-Attention]]: 动作专家整合编辑缓存的注意力方式
- [[Action Chunking]]: 动作预测的基本形式（预测 H 步动作块）
- [[Diffusion Transformer]]: 图像编辑骨干的架构基础
- [[VAE（变分自编码器）]]: 用于编码目标帧到潜空间

### 硬件/数据相关

- [[LIBERO]]: 主要评测数据集
- [[RoboTwin 2.0]]: 仿真评测平台

---

## 速查卡片

> [!summary] ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?
> - **核心**: 用图像编辑替代视频生成作为 WAM 骨干，用 KV 缓存桥接编辑表示与动作预测
> - **方法**: 图像编辑骨干 + 动作专家通过联合注意力整合编辑 KV 缓存；联合流匹配训练
> - **结果**: RoboTwin 93.38%、LIBERO 98.4%、真实机器人 84.5%；FLOPs 降至 1/6、延迟降至 1/4
> - **代码**: [zhangwenyao1.github.io/ImageWAM](https://zhangwenyao1.github.io/ImageWAM/)

---

*笔记创建时间: 2026-06-19*
