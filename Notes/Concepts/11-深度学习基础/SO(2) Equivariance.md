---
type: concept
aliases: [SO2等变性, 旋转等变, SO(2) equivariance, 群等变性]
---

# SO(2) Equivariance

## 定义

SO(2) 等变性（Rotational Equivariance）是指函数 $f$ 对二维平面旋转群 $\text{SO}(2)$ 的等变性：当输入旋转时，输出以对应的群表示同步变换，而非忽略旋转或对所有旋转输出相同结果。

## 数学形式

$$
f(g \cdot x) = \rho(g) \cdot f(x) \quad \forall g \in \text{SO}(2)
$$

其中 $g$ 是旋转变换，$\rho$ 是输出空间上的群表示（可以与输入的群作用不同）。

对于近似等变（如 EquiVLA 中的情形）：

$$
\|f(g \cdot x) - \rho(g) \cdot f(x)\| \leq \Delta \cdot B(x)
$$

其中 $\Delta$ 是表示缺陷（representation defect），$B(x)$ 是特征范数上界。

## 核心要点

1. **离散化实现**: 实践中常用离散循环群 $C_u$（如 $C_4, C_8, C_{16}$）近似 $\text{SO}(2)$
2. **不可约表示**: $\text{SO}(2)$ 的不可约表示为 $\rho_k$（$k$ 阶旋转表示），用于分解动作空间
3. **精确 vs 近似**: $C_4$ 在方形 patch 网格上可达精确等变；$C_8$ 引入有界近似误差（约 31-35% patch 错位）
4. **任务相关性**: 对方向敏感的操作任务（如物体对齐、叠放）等变性带来显著提升

## 代表工作

- [[EquiVLA]]: 将 SO(2) 等变性引入大规模 VLA 模型，EquiPerceptor + EquiActor 双模块实现

## 相关概念

- [[Frame Averaging]]
- [[Regular Representation]]
- [[Steerable CNN]]
