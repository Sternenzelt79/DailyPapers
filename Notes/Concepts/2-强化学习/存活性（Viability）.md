---
type: concept
aliases: [Viability, 存活性, 任务可行性]
---

# 存活性（Viability）

## 定义

存活性（Viability）是 HABC 框架中定义的 RL 目标之一，表示从当前状态出发、最终完成任务的概率估计 $p_v(s) = P(\text{success} \mid s)$。

## 数学形式

存活性头输出 logit，经 sigmoid 得到概率：

$$
z_v(s) = f_v(\phi(s)), \quad p_v(s) = \text{sigmoid}(z_v(s))
$$

存活性优势（logit 空间计算以保持分辨率）：

$$
A_v = z_v(s_{t+1}) - z_v(s_t)
$$

## 核心要点

1. **语义**: 衡量当前状态离任务成功有多"可行"，$p_v \to 1$ 表示几乎必然成功
2. **训练数据**: 所有有结果标注的策略执行轨迹（含成功和失败），用 BCE 损失
3. **logit 空间优势**: 在 $p_v \approx 1$ 时，logit 空间仍有分辨率，避免 sigmoid 饱和导致优势信号消失
4. **与效率的关系**: 存活性解决"能不能成功"，效率解决"多快能成功"

## 代表工作

- [[HABC]]: 存活性是双头 Critic 的第一个目标，驱动策略向高成功概率状态转移

## 相关概念

- [[效率（Efficiency）]]
- [[双头Critic]]
- [[存活性优势]]
- [[状态自适应门控]]
