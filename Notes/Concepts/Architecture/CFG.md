---
type: concept
aliases: [Classifier-Free Guidance, 无分类器引导, CFG Scale]
---

# CFG

## 定义
Classifier-Free Guidance：扩散模型中提升条件生成质量的采样技术，通过对条件和无条件预测的线性外推来增强文本/图像对齐，以牺牲多样性换取更高的条件一致性。

## 数学形式
$$
\hat{\epsilon}_\theta(\mathbf{x}_t, c) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + w \cdot \left(\epsilon_\theta(\mathbf{x}_t, c) - \epsilon_\theta(\mathbf{x}_t, \emptyset)\right)
$$

其中 $w$ 为 guidance scale（通常 1-20），$c$ 为条件，$\emptyset$ 为 null 条件。

## 核心要点
1. 训练时随机以概率 $p_{\text{uncond}}$（约 10-20%）丢弃条件，使模型同时学习条件和无条件预测
2. 推理时 guidance scale $w > 1$ 放大条件方向，$w = 1$ 等价于普通条件生成
3. $w$ 越大文本对齐越好但多样性降低，可能出现过饱和
4. [[CDM]] 等工作在蒸馏后仍使用 CFG 增强采样质量

## 代表工作
- [[CDM]]: 在连续时间扩散蒸馏中使用 CFG 增强采样

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[DMD]]
