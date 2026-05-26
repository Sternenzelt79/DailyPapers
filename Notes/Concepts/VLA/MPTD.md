---
type: concept
aliases: [MPTD, Morphological Tree Diffusion, 形态树扩散]
---

# MPTD（Morphological Tree Diffusion）

## 定义
X-DiffVLA 中提出的跨具身扩散动作引导方法：将 Monte Carlo Tree Search（MCTS）范式扩展到扩散去噪过程，通过树搜索挖掘异构机器人末端执行器之间的行为相关性，促进知识迁移。

## 数学形式

节点价值函数（同维度 vs. 异维度体现）：

$$
v = \begin{cases}
-\|\mathbf{x}-\mathbf{y}\|^2 + M_0, & \text{if } \text{dim}(\mathbf{x})=\text{dim}(\mathbf{y}) \\
-\|\mathbf{x}_{1:d_{\min}}-\mathbf{y}_{1:d_{\min}}\|^2 + M_1, & \text{if } \text{dim}(\mathbf{x})\neq\text{dim}(\mathbf{y})
\end{cases}
$$

其中 $M_1 > M_0$，激励异质体现间的迁移；$d_{\min}$ 为两体现公共最小自由度维度。

## 核心要点
1. **四阶段 MCTS**: 选择 → 扩展（生成元动作节点）→ 模拟（快速跳跃去噪）→ 回传
2. **跨维度对齐**: 不同 DoF 的机器人（夹爪 vs. 灵巧手）通过截取公共维度实现价值比较
3. **与 EBF 协同**: EBF 解决体现判别，MPTD 解决知识迁移，二者结合效果显著

## 代表工作
- [[X-DiffVLA]]: Li et al. 2026，首次提出 MPTD

## 相关概念
- [[MCTD]]
- [[EBF]]
- [[X-DiffVLA]]
- [[Monte Carlo Tree Search]]
