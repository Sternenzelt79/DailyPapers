---
type: concept
aliases: [MCTD, Morphological Tree Diffusion, 形态树扩散]
---

# MCTD（Morphological Tree Diffusion）

## 定义
X-DiffVLA 中提出的结构化扩散动作生成方法：把机器人运动学树结构编码进扩散过程，用 Monte Carlo Tree Search 对动作空间做层级采样，处理多峰分布动作。

## 数学形式
$$p(\mathbf{a} | \mathbf{o}, \mathbf{l}, \mathbf{m}) = \prod_{k=1}^{K} p(a_k | a_{<k}, \mathbf{o}, \mathbf{l}, \mathcal{T}_\mathbf{m})$$

其中 $\mathcal{T}_\mathbf{m}$ 是由形态参数 $\mathbf{m}$ 定义的运动学树，$K$ 是关节数。

## 核心要点
1. **运动学树结构化**：把机器人关节层级关系编码进扩散 action head，而非平铺所有 DoF
2. **Monte Carlo Tree 采样**：对动作空间做 MCTS 风格的结构化探索，处理高维多峰分布
3. **配合 EBF**：与 [[EBF]] 配合使用，[[EBF]] 编码形态，MCTD 利用形态做动作生成

## 代表工作
- [[X-DiffVLA]]：Li et al. 2026，提出 MCTD 的原始工作

## 相关概念
- [[EBF]]
- [[X-DiffVLA]]
- [[Diffusion Model]]
- [[Action Chunking]]
