---
type: concept
aliases: [SNR偏移调度器, SNR-Shifted Sampler, SNR Shifted Scheduler]
---

# SNR-Shifted Scheduler

## 定义
一种扩散/流匹配训练中的噪声调度策略，通过偏移量 $s$ 将训练质量集中到特定噪声水平区间，高维度数据（如视频）通常需要更大的偏移使训练集中于高噪声区间。

## 数学形式

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0,1]
$$

其中 $s > 1$ 将分布偏移至高噪声（$\sigma \to 1$），$s = 1$ 退化为均匀采样。

## 核心要点
1. 视频等高维信号需要更大偏移量 $s$，训练质量集中于高噪声区间（大 σ）
2. 动作等低维信号偏移量较小，训练质量集中于低噪声区间（小 σ）
3. 该不对称性导致视频与动作流在联合 WAM 模型中面临不同的一致性蒸馏挑战
4. 正是这一不对称性使得 Flash-WAM 需要模态感知的参数化设计

## 代表工作
- [[Flash-WAM]]: 分析 SNR-Shifted Scheduler 导致视频/动作训练制度不对称，并据此设计模态感知蒸馏
- [[LingBot]]: LingBot-VA 在视频和动作流上均使用 SNR 偏移调度器

## 相关概念
- [[Flow Matching]]: SNR-Shifted Scheduler 常用于 Flow Matching 框架
- [[一致性蒸馏]]: 蒸馏方法需要适配 SNR 偏移带来的训练制度差异
- [[v-prediction]]: 低噪声区间适合的参数化形式
