---
type: concept
aliases: [FA, Frame Averaging, 帧平均, 群等变对称化]
---

# Frame Averaging

## 定义

Frame Averaging（FA）是一种将任意神经网络对称化为群等变函数的通用技术：对输入在群 $G$ 的所有变换下分别计算网络输出，再取等变/不变聚合，从而使函数满足群等变或不变性，无需修改网络权重。

## 数学形式

**等变输出**（对旋转等变）：

$$
z^{eq}(x) = \frac{1}{|G|} \sum_{h \in G} \rho^{-1}(h) \cdot f_\theta(h \cdot x)
$$

**不变输出**：

$$
z^{inv}(x) = \frac{1}{|G|} \sum_{h \in G} f_\theta(h \cdot x)
$$

其中 $G$ 是有限群，$\rho$ 是输出空间上的群表示，$f_\theta$ 是任意神经网络。

## 核心要点

1. **无侵入性**: 冻结预训练网络权重，仅在输入/输出层做对称化，不改变网络参数
2. **精确等变**: 对有限群 $G$，FA 输出严格满足等变性（当空间变换与群表示相容时）
3. **计算代价**: 需要 $|G|$ 次前向传播，推理时间线性增加
4. **Token-level 扩展**: 在 ViT patch token 层级应用 FA，需处理空间 patch 的置换对应关系

## 代表工作

- [[EquiVLA]]: 将 Token-level FA 应用于冻结 ViT，生成 SO(2) 等变视觉 token（EquiPerceptor）
- [[EquiLLM]]: 将 FA 原理扩展到语言模型的等变化

## 相关概念

- [[SO(2) Equivariance]]
- [[Regular Representation]]
- [[Steerable CNN]]
