---
type: concept
aliases: [Surfel, 面元地图, 4D Surfel, surfel-indexed memory, surfel 索引记忆]
---

# Surfel Map

## 定义

Surfel（Surface Element，表面元素）是一种用于表示 3D 场景表面的基本单元，每个 surfel 存储局部几何属性（位置、法向量、半径）；Surfel Map 是由大量 surfel 组成的场景表示，兼顾几何精度和计算效率。

## 数学形式

标准 surfel 定义：

$$
s_k = (p_k,\; n_k,\; r_k)
$$

[[Mem-World]] 扩展的 4D surfel（含时间和语义属性）：

$$
s_k = (p_k,\; n_k,\; r_k,\; t_k,\; m_k)
$$

- $p_k \in \mathbb{R}^3$：3D 位置
- $n_k \in \mathbb{R}^3$：表面法向量（单位向量）
- $r_k \in \mathbb{R}^+$：surfel 半径（影响渲染覆盖范围）
- $t_k$：创建/最后更新时间戳（4D 扩展）
- $m_k \in \{0,1\}$：操作对象标志（语义扩展）

## 核心要点

1. **与点云对比**：surfel 额外存储法向量和半径，可用于可见性判断和表面渲染，比纯点云更适合视觉任务
2. **与 NeRF/3DGS 对比**：surfel 是显式离散表示，更新效率高，适合在线场景维护；NeRF 是隐式连续表示，重建质量更高但更新成本大
3. **4D surfel（时序扩展）**：加入时间戳使 surfel 能跟踪"哪个时刻见过这个表面"，支持时间感知检索
4. **几何可见性打分**：法向量与观测方向的点积 $\langle n_s, \bar{v}_w \rangle$ 量化 surfel 对当前相机的可见性

## W-VMem 打分函数

[[Mem-World]] 中的 4D surfel 打分函数：

$$
\text{score}(s, t) = \frac{\langle n_s, \bar{v}_w \rangle}{1+d_s} \cdot \ln(e + m_s) \cdot \left[\lambda_{\min} + (1-\lambda_{\min}) \cdot 2^{-\frac{T-t}{H}}\right]
$$

融合几何可见性、操作相关性、时间近因性三项因子。

## 代表工作

- [[VMem]]: 提出 surfel 索引记忆用于静态场景视频生成
- [[Mem-World]]: 将 surfel 扩展为 4D（含时间戳和操作对象标志），用于机器人操作的持久世界建模

## 相关概念

- [[Action-Conditioned Video Generation]]
- [[Forward Kinematics]]
- [[Non-Maximum Suppression]]
- [[Video Diffusion Model]]
