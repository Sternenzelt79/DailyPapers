---
title: "Latent Spatial Memory for Video World Models"
method_name: "Mirage"
authors: [Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, Bohan Zhuang]
year: 2026
venue: arXiv
tags: [video-world-model, spatial-memory, novel-view-synthesis, video-generation, camera-control, 3d-consistency]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.09828
created: 2026-06-13
---

# 论文笔记：Latent Spatial Memory for Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft Research |
| 日期 | June 2026 |
| 项目主页 | [aka.ms/latent-spatial-memory](https://aka.ms/latent-spatial-memory) |
| 对比基线 | [[Voyager]], [[Spatia]], [[WonderJourney]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09828) |

---

## 一句话总结

> Mirage 将 3D 空间记忆直接存储在 [[VAE|扩散模型潜空间]] 而非 RGB 点云，通过单次潜分辨率投影实现记忆读取，在 WorldScore 达到 SOTA 的同时实现 10.57× 加速和 55× 内存降低。

---

## 核心贡献

1. **潜空间空间记忆（Latent Spatial Memory）**: 将场景信息缓存为世界坐标 + 潜特征对 $(\mathbf{p}_i, \mathbf{f}_i)$，避免 RGB 点云方法昂贵的光栅化-编码往返。
2. **深度引导反投影（Depth-Guided Back-Projection）**: 用 [[DepthAnything|DepthAnything3]] 估计度量深度，将潜 token 提升到 3D 世界坐标，再通过针孔相机模型反投影。
3. **自回归缓存更新 + 动态过滤**: 每帧生成后重新编码并回填缓存，用 [[SAM-3]] 和 Qwen3-VL 过滤运动物体，防止动态内容污染几何一致性。

---

## 问题背景

### 要解决的问题

视频世界模型在长程摄像机轨迹下容易出现几何漂移——当相机大范围运动后返回已观测区域时，场景外观缺乏一致性。现有方法需要维护一个显式的 3D 记忆来实现空间一致性。

### 现有方法的局限

基于 RGB 点云的方法（Voyager、Spatia、VMem）在每个调节步骤都需要：
1. 将点云**光栅化**为像素级别的 RGB 图像（计算昂贵）
2. 用 VAE 将 RGB 重新**编码**为潜表示（引入信息损失）

这一"光栅化-编码往返"（rasterize-and-encode round trip）使每帧推理时间高达 2.64 秒，且 3D 缓存内存占用巨大。

### 本文的动机

VAE 的空间压缩因子 $s=16$，意味着潜分辨率是像素分辨率的 $1/16$，潜缓存占用仅是 RGB 缓存的 $\frac{1}{s^2} = \frac{1}{256}$。若直接在潜空间存储和读取记忆，既能消除往返开销，又能大幅压缩内存，同时保留语义特征（而非有损 RGB 像素）。

---

## 方法详解

### 模型架构

Mirage 采用 **自回归生成 + 3D 潜缓存** 架构：

- **输入**: 单张参考图像 $I_0$ + 用户指定相机轨迹（内参 $K$、外参 $\mathbf{E}^t$ 序列）
- **Backbone**: Wan2.2-TI2V-5B（5B 参数 [[Video Diffusion Model|视频扩散模型]]）
- **核心模块**: [[潜空间空间记忆|Latent Spatial Memory]] $\mathcal{M}$ + [[ControlNet]] 旁路注入
- **输出**: 时空一致的视频帧序列
- **总参数**: ~5B（backbone） + rank-64 [[LoRA]] 适配器

整体流程分三个循环步骤：**初始化 → 读取 → 更新**。

### 核心模块

#### 模块1: 潜缓存初始化（Initialization）

**设计动机**: 将参考图像的视觉信息"锚定"在世界坐标系中，供后续任意视角查询。

**具体实现**:
- 将参考帧 $I_0$ 编码为 [[VAE]] 潜表示 $\mathbf{z} \in \mathbb{R}^{C \times h \times w}$，其中 $h=H/s, w=W/s$
- 调用 [[DepthAnything|DepthAnything3]] 估计度量深度图 $D(u,v)$
- 对每个潜 token 位置 $(u,v)$，通过[[针孔相机模型|深度引导反投影]]（Eq. 8）计算世界坐标 $\mathbf{p}_{uv} \in \mathbb{R}^3$
- 构建初始缓存 $\mathcal{M} = \{(\mathbf{p}_{uv}, \mathbf{z}[:,v,u])\}$

#### 模块2: 潜分辨率记忆读取（Readout）

**设计动机**: 通过单次潜分辨率投影直接提取目标视角的记忆特征，避免 RGB 光栅化。

**具体实现**:
- 将缓存中每个 3D 点 $\mathbf{p}_i$ 投影到目标相机坐标系
- 计算目标视角的**潜分辨率内参** $K^\ell$（Eq. 7），在潜网格上分配投影点
- 用 [[Z-Buffering|z-buffering]] 处理遮挡，选取每格最近的点（Eq. 5）
- 得到目标视角的记忆读取张量 $\hat{\mathbf{z}}^t$，通过 [[ControlNet]] 旁路注入 backbone

#### 模块3: 自回归缓存更新（Update）

**设计动机**: 持续扩展缓存以覆盖新观测区域，同时防止动态物体污染几何记忆。

**具体实现**:
- 解码生成帧后，用 [[DepthAnything|DepthAnything3]] 重新估计深度
- 用 [[SAM-3]] 分割 + Qwen3-VL 检测运动物体和天空，创建动态过滤掩码 $\Lambda^t$
- 对未遮挡的静态区域执行缓存回填（Eq. 6），扩充记忆集合

#### 模块4: ControlNet 旁路注入

- 读取张量 $\hat{\mathbf{z}}^t$ 通过镜像 backbone 结构的旁路注入，操作在 48 通道潜分辨率下进行
- 使用**分段感知旋转位置编码**（Segment-Aware [[Rotary Position Encoding|RoPE]]），区分噪声目标帧、干净前置帧和干净参考帧

---

## 关键公式

### 公式1: [[潜空间空间记忆|记忆集合定义]]（Eq. 3）

$$
\mathcal{M}=\{(\mathbf{p}_i,\mathbf{f}_i)\}, \quad \mathbf{p}_i\in\mathbb{R}^3, \quad \mathbf{f}_i\in\mathbb{R}^C
$$

**含义**: 记忆缓存由若干三维世界坐标 $\mathbf{p}_i$ 与对应潜特征向量 $\mathbf{f}_i$ 的配对构成。

**符号说明**:
- $\mathbf{p}_i \in \mathbb{R}^3$: 第 $i$ 个记忆点的世界坐标（由深度反投影获得）
- $\mathbf{f}_i \in \mathbb{R}^C$: 对应的 $C$ 维潜特征向量（由 VAE 编码获得）
- $\mathcal{M}$: 整体记忆缓存集合，随生成进程自回归扩充

### 公式2: [[针孔相机模型|深度引导反投影初始化]]（Eq. 4）

$$
\mathbf{p}_{uv}=\pi^{-1}(u,v,D(u,v);K,\mathbf{E}), \quad \mathbf{F}_{uv}=\mathbf{z}[:,v,u]
$$

**含义**: 将每个潜 token 位置 $(u,v)$ 与其估计深度 $D(u,v)$ 联合反投影到世界坐标系，同时记录对应的潜特征列向量。

**符号说明**:
- $(u,v)$: 潜分辨率网格上的像素位置
- $D(u,v)$: 该位置的估计度量深度（由 DepthAnything3 提供）
- $K$: 相机内参矩阵
- $\mathbf{E}$: 相机外参（世界到相机变换矩阵）
- $\pi^{-1}$: 针孔反投影函数（见 Eq. 8）
- $\mathbf{z}[:,v,u]$: VAE 潜表示在 $(u,v)$ 处的 $C$ 维特征向量

### 公式3: [[Z-Buffering|潜分辨率记忆读取]]（Eq. 5）

$$
i^t(u,v)=\arg\min_{i\in\Omega^t(u,v)}[\mathbf{E}^t\mathbf{p}_i]_z, \quad \hat{\mathbf{z}}^t(u,v)=\mathbf{F}_{i^t(u,v)}
$$

**含义**: 对目标视角 $t$ 的每个潜格 $(u,v)$，从投影到该格的所有候选点 $\Omega^t(u,v)$ 中选取相机坐标系下深度最小（最近）的点，取其特征作为读取结果——即潜分辨率 z-buffering。

**符号说明**:
- $\Omega^t(u,v)$: 在目标视角 $t$ 下投影落在潜格 $(u,v)$ 的所有缓存点索引集合
- $[\mathbf{E}^t\mathbf{p}_i]_z$: 点 $\mathbf{p}_i$ 变换到目标相机坐标系后的深度分量
- $\hat{\mathbf{z}}^t(u,v)$: 读取出的目标视角潜特征，用于 ControlNet 注入

### 公式4: [[自回归缓存更新|缓存自回归扩充]]（Eq. 6）

$$
\mathcal{M}\leftarrow\mathcal{M}\cup\{(\mathbf{p}_{uv},\mathbf{F}_{uv})\}_{(u,v)\in\Lambda^t}
$$

**含义**: 将当前帧生成后重新估计的静态区域潜特征回填到全局缓存，逐帧扩大空间记忆覆盖范围。

**符号说明**:
- $\Lambda^t$: 第 $t$ 帧的静态区域掩码（排除运动物体和天空）
- $\{(\mathbf{p}_{uv},\mathbf{F}_{uv})\}_{(u,v)\in\Lambda^t}$: 新增的静态世界点-特征对

### 公式5: [[针孔相机模型|潜分辨率内参缩放]]（Eq. 7）

$$
K^\ell=\text{diag}(w/W,h/H,1)K
$$

**含义**: 将原始像素内参 $K$ 缩放到潜分辨率下，保证投影操作在潜空间网格上精确对齐。

**符号说明**:
- $K$: 原始像素分辨率相机内参（$W \times H$）
- $K^\ell$: 潜分辨率相机内参（$w \times h$，$w=W/s, h=H/s$）
- $s=16$: VAE 空间压缩因子

### 公式6: [[针孔相机模型|针孔反投影函数]]（Eq. 8）

$$
\pi^{-1}(u,v,d;K^\ell,\mathbf{E})=\mathbf{E}^{-1}\begin{bmatrix}d(K^\ell)^{-1}[u+\tfrac{1}{2},v+\tfrac{1}{2},1]^\top\\1\end{bmatrix}\bigg|_{1:3}
$$

**含义**: 将像素坐标 $(u,v)$ 和深度 $d$ 通过相机内外参联合反投影为齐次世界坐标，取前三维。

**符号说明**:
- $(u+\frac{1}{2}, v+\frac{1}{2})$: 像素中心亚像素偏移
- $d$: 该位置的度量深度
- $\mathbf{E}^{-1}$: 相机到世界变换矩阵（外参逆）
- $|_{1:3}$: 取齐次坐标的前三维（去除齐次分量）

---

## 关键图表

### Figure 1: Mirage 生成的几何一致视频示例

![[Mirage_fig1.png]]

**说明**: 给定单张输入图像和用户指定相机轨迹（左），Mirage 通过在潜空间缓存 3D 信息（而非 RGB 点云）来保持空间一致性。即使经历大范围相机绕行，仍能忠实还原已观测区域，同时比 RGB 缓存基线快 10.57×，内存低 55×。

### Figure 2: 潜空间记忆 vs. RGB 点云记忆对比

![[Mirage_fig2.png]]

**说明**: 上方：已有方法将记忆存为彩色点云，每次调节需要光栅化→重编码往返。下方：Mirage 将潜特征存于世界坐标，仅需单次潜分辨率投影读取，消除逐步像素空间往返，缓存占用减少 $s^2=256$ 倍（VAE 压缩平方）。

### Figure 3: Mirage 系统流水线概览

![[Mirage_fig3.png]]

**说明**: Mirage 从 $I_0$ 初始化 3D 潜缓存（编码+深度引导反投影），每个目标视角通过潜分辨率投影读取缓存，生成输出后重估深度、重编码、回填缓存，逐块自回归延伸。

### Figure 4: 开放域视频生成对比

![[Mirage_fig4.png]]

**说明**: 在 RealEstate10K 训练分布之外的室外自然场景测试。Mirage 在大角度相机运动下保持时间平滑与 3D 一致，而 RGB 点云基线在新布局下出现纹理拉伸，视频扩散基线出现几何漂移。

### Figure 5: 推理效率随生成进程的扩展

![[Mirage_fig5.png]]

**说明**: 在单张 NVIDIA H100 上测量 5 个自回归块的每帧缓存读取时间（左）和峰值缓存占用（右）。首块均摊初始化后，Mirage 每帧稳定在 0.25 s，缓存每块增长不足 0.5 MiB；而 RGB 方法随生成进程缓存和时间线性膨胀。

### Figure 6: RealEstate10K 视频生成对比

![[Mirage_fig6.png]]

**说明**: 室内外场景的 RealEstate10K 轨迹对比（Voyager、Spatia、VMem、Mirage）。Mirage 在相机运动下保持更锐利的结构和稳定的外观，基线方法出现几何漂移、纹理变形或累积伪影。

### Figure 7: 闭环重访对比

![[Mirage_fig7.png]]

**说明**: 相机轨迹逐渐返回起点的闭环测试。最后一帧与输入帧的比较表明，Mirage 在重访场景时保持强一致性，这正是潜空间记忆的核心优势——避免往返开销带来的误差积累。

### Figure 8: 复杂室内轨迹额外对比

![[Mirage_fig8.png]]

**说明**: 在高难度室内轨迹上，Mirage 在完整轨迹中保持连贯的场景布局，而基线出现视角依赖变形、模糊和不一致的场景重建。

### Table 1: WorldScore 基准测试结果

| Method | Avg Score | Static Score | Dynamic Score | 3D Consistency |
|--------|-----------|--------------|---------------|----------------|
| WonderJourney | 54.19 | 63.75 | 44.63 | 80.60 |
| Voyager | 66.08 | 77.62 | 54.53 | 81.56 |
| Spatia | 69.73 | 72.63 | 66.82 | 86.40 |
| **Mirage (Ours)** | **70.36** | **73.60** | **67.11** | **92.21** |

**关键发现**: Mirage 在 3D Consistency 指标上大幅领先（92.21 vs. 86.40），验证了潜空间记忆对几何一致性的显著提升，整体平均分也达到 SOTA。

### Table 2: RealEstate10K 定量结果

| Method | PSNR | SSIM | LPIPS | PSNR_C | SSIM_C | LPIPS_C |
|--------|------|------|-------|--------|--------|---------|
| Spatia | 18.58 | 0.646 | 0.254 | 19.38 | 0.579 | 0.213 |
| **Mirage (Ours)** | **18.38** | **0.779** | **0.250** | **20.05** | **0.825** | **0.228** |

**关键发现**: Mirage 在 SSIM 和闭环指标（带 _C 后缀）上大幅领先，SSIM 从 0.646 提升至 0.779（+20.6%），SSIM_C 从 0.579 提升至 0.825（+42.5%），体现了其在闭环重访场景下的极强一致性。

### Table 3: 消融实验

| 变体 | Avg Score | 3D Consistency | Photometric Consistency |
|------|-----------|----------------|------------------------|
| RGB 点云缓存 | 67.71 | 90.75 | 91.10 |
| 无动态过滤 | 61.20 | 80.88 | 76.10 |
| **完整 Mirage** | **70.36** | **92.21** | **93.95** |

**关键发现**: 动态过滤对性能影响最大（去掉后 Avg Score 从 70.36 降至 61.20，降幅 12.9%），说明运动物体污染缓存是主要失败模式；潜空间 vs. RGB 点云也带来 +2.65 分的显著提升。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RealEstate10K | ~80k 视频片段 | 室内外房产场景，连续相机运动 | 训练 + 测试 |
| WorldScore Benchmark | 标准化评测集 | 多维度评测（静态/动态/3D一致性） | 评测 |

### 实现细节

- **Backbone**: Wan2.2-TI2V-5B（5B 参数视频扩散模型，[[Flow Matching|流匹配]]训练目标）
- **VAE**: stride $s=16$，48 潜通道，空间压缩 256×
- **深度估计**: DepthAnything3（度量深度）
- **动态过滤**: [[SAM-3]] 分割 + Qwen3-VL 实体检测
- **调节注入**: [[ControlNet]] 旁路（与 backbone 镜像结构，48 通道）
- **LoRA**: rank-64，作用于自注意力投影
- **两阶段训练**: Stage 1 冻结 backbone + VAE，仅训 ControlNet（lr=$10^{-5}$）；Stage 2 解冻 LoRA 联合优化（lr=$10^{-4}$）
- **硬件**: 32 × A100 GPU
- **位置编码**: 分段感知 [[Rotary Position Encoding|RoPE]]，区分噪声帧/干净前置帧/干净参考帧

### 推理效率

| 指标 | Mirage | RGB 点云基线 |
|------|--------|-------------|
| 每帧缓存读取时间 | 0.25 s | 2.64 s |
| 缓存内存每块增量 | <0.5 MiB | ~100+ MiB |
| 端到端加速比 | 10.57× | 1× |
| 缓存内存降低 | 55× | 1× |

---

## 批判性思考

### 优点

1. **理论上优雅**: 潜空间缓存的内存节省严格来源于 VAE 压缩平方（$s^2=256$），不是工程优化而是设计的必然结果，物理意义清晰。
2. **端到端效率提升突出**: 10.57× 加速和 55× 内存降低使实时应用成为可能，实际意义重大。
3. **可组合性强**: 基于 [[ControlNet]] 的注入方式与 backbone 解耦，原则上可以迁移到其他扩散模型。

### 局限性

1. **动态场景受限**: 排除运动实体意味着无法持久化动态物体（人、车等）的记忆，在高动态场景下性能受限。
2. **深度估计依赖**: 整个框架依赖 DepthAnything3 的度量深度质量，反射、透明物体、纹理缺乏区域的深度估计误差会直接传播到空间记忆。
3. **闭环积累误差**: 尽管效果优于基线，但自回归回填的误差仍会在极长轨迹中积累（每次回填都引入重新编码的量化误差）。
4. **训练数据偏置**: 主要在 RealEstate10K（室内/房产场景）上训练，开放域泛化依赖 backbone 的先验，垂直领域（如工业场景）需额外微调。

### 潜在改进方向

1. **动态物体记忆**: 用独立的动态缓存追踪运动实体，实现全场景持久化记忆。
2. **不确定性感知更新**: 在缓存回填时引入不确定性权重，减少重复观测区域的更新噪声。
3. **跨模态应用**: 将潜空间记忆框架迁移到自动驾驶、机器人导航等需要大范围场景一致性的任务。

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码状态未确认）
- [ ] 预训练模型（未明确提供）
- [x] 训练细节完整（学习率、GPU 数量、两阶段训练均有说明）
- [x] 数据集可获取（RealEstate10K 公开可用）

---

## 关联笔记

### 基于

- [[Video Diffusion Model]]: Backbone Wan2.2-TI2V-5B 是视频扩散模型
- [[VAE]]: 潜空间记忆的存储空间由 VAE 编码器提供
- [[ControlNet]]: 记忆读取张量通过 ControlNet 旁路注入扩散 backbone
- [[DepthAnything]]: 提供深度引导反投影所需的度量深度估计

### 对比

- [[Voyager]]: RGB 点云记忆基线，需要光栅化-编码往返
- [[Spatia]]: RGB 点云记忆基线，RealEstate10K 上主要对比方法
- [[WonderJourney]]: 较早的视频世界模型基线

### 方法相关

- [[针孔相机模型]]: 深度引导反投影的核心数学工具
- [[Z-Buffering]]: 潜分辨率记忆读取中的可见性判断方法
- [[SAM-3]]: 动态物体过滤中的分割工具
- [[Rotary Position Encoding]]: 分段感知 RoPE 用于帧类型区分
- [[LoRA]]: rank-64 低秩适配器用于 Stage 2 微调
- [[Flow Matching]]: 训练目标（流匹配）

### 硬件/数据相关

- [[RealEstate10K]]: 主要训练和评测数据集

---

## 速查卡片

> [!summary] Mirage: Latent Spatial Memory for Video World Models
> - **核心**: 将 3D 空间记忆存储在扩散潜空间而非 RGB 点云，通过单次潜分辨率投影读取
> - **方法**: 深度引导反投影建缓存 → ControlNet 注入 → 自回归更新 + SAM 动态过滤
> - **结果**: WorldScore SOTA (70.36)，10.57× 加速，55× 内存降低
> - **代码**: [aka.ms/latent-spatial-memory](https://aka.ms/latent-spatial-memory)

---

*笔记创建时间: 2026-06-13*
