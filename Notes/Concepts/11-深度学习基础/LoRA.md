---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配, LoRA微调, 低秩适配器]
---

# LoRA

## 定义

LoRA（Low-Rank Adaptation）是一种参数高效微调方法，通过在预训练模型的权重矩阵旁并联一个低秩分解矩阵 $\Delta W = BA$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$），以极少可训练参数实现大模型的任务适配。

## 数学形式

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

其中 $r$ 为秩（rank），通常取 4、8、16、64 等远小于 $d, k$ 的值；$A$ 用随机高斯初始化，$B$ 初始化为零，保证训练起始时 $\Delta W = 0$。

## 核心要点

1. **冻结预训练权重**：原始权重 $W_0$ 保持不变，只训练 $A, B$，可训练参数量为 $r(d+k) \ll dk$
2. **推理无额外延迟**：$W_0 + BA$ 可预先合并，推理时无额外计算开销
3. **Rank 选择**：rank 越大表达能力越强，rank-64 适合需要较大适配幅度的任务（如视频生成微调）
4. **作用位置**：通常施加于 Transformer 自注意力的 Q、K、V、O 投影矩阵

## 代表工作

- [[Mirage]]: Stage 2 训练解冻 rank-64 LoRA 适配器，作用于自注意力投影，与 ControlNet 旁路联合优化

## 相关概念

- [[ControlNet]]
- [[Adaptive Weighting]]
- [[Feature Alignment]]
