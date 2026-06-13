---
type: concept
aliases: [Options, 选项框架, Semi-MDP, 分层强化学习框架, Options-style Control]
---

# Options Framework

## 定义

一种分层强化学习框架，将策略分解为高层"选项（options）"和底层原子动作，每个选项包含初始集合、内部策略和终止条件，用于解决需要时间抽象的长时程任务。

## 数学形式

一个 Option $\omega = (I_\omega, \pi_\omega, \beta_\omega)$ 定义为：

$$
\omega = (I_\omega,\, \pi_\omega,\, \beta_\omega)
$$

其中：
- $I_\omega \subseteq S$：选项的初始状态集合
- $\pi_\omega: S \times A \to [0,1]$：选项的内部策略
- $\beta_\omega: S \to [0,1]$：状态相关的终止概率

Hi-VLA 系统在此框架下表示为：

$$
\pi_{\text{HiVLA}}(a \mid [o_i]_{i \leq t}, I) = \int_l \pi_{\text{VLA}}(a \mid o_t, l)\, \pi_{\text{VLM}}(l \mid \mathbf{o}, I)\, \mathrm{d}l
$$

## 核心要点

1. **时间抽象**：高层策略以"选项"为单位做决策，无需每步介入
2. **层次解耦**：高层规划与低层执行分离，便于模块化设计和复用
3. **终止条件关键**：选项的终止时机直接影响分层系统性能
4. **VLA 中的应用**：用语言子目标作为"选项"，VLM 做高层规划，VLA 做低层执行

## 代表工作

- [[HiVLA-Study]]: 用 Options 框架统一 Hi-VLA 设计空间，系统研究六大设计维度

## 相关概念

- [[VLA]]
- [[VLM]]
- [[Chain-of-Thought]]
