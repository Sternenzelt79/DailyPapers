---
type: concept
aliases: [RPRO, Flow-PRO, Proximalized Preference Optimization for Flow Matching]
---

# FlowPRO

## 定义

FlowPRO（Flow-Matching Proximalized Reward-free Preference Optimization）是一种专为[[条件流匹配|Flow Matching]]策略设计的无奖励函数RL后训练算法，通过干预-回滚数据构造正负偏好对，利用隐式奖励对比损失进行策略改进。

## 数学形式

隐式奖励：

$$
r_\theta(s, a) = \frac{\beta}{2}\left(\ell_{ref}(s, a) - \ell_\theta(s, a)\right)
$$

RPRO损失：

$$
\mathcal{L}_{PRO}(\theta) = -\mathbb{E}\left[\log\sigma\left(r_\theta(s, a^w) - r_\theta(s, a^l)\right) + \sum_{a \in \{a^w, a^l\}} \frac{1}{2}\left[\log\sigma(r_\theta(s,a)) + \log\sigma(-r_\theta(s,a))\right]\right]
$$

## 核心要点

1. **无奖励、无Critic**: 不训练额外的奖励模型或价值函数，直接利用策略的负对数似然差作为隐式奖励
2. **干预-回滚数据采集**: 操作员在策略Rollout中实时干预，系统回滚到前序状态，记录执行片段为负样本、纠正操作为正样本
3. **平滑插值**: 通过三次Bézier曲线在正负轨迹之间构造稠密每状态偏好对 $(s, a^w, a^l)$
4. **近端正则化**: 损失中的锚定项防止奖励劫持（reward hacking），保持与SFT参考策略的接近
5. **批次混合**: Round 1 用80%新对+20% SFT数据，后续轮次分配给历史数据一定比例

## 代表工作

- [[HyVLA05|HyVLA-0.5]]: FlowPRO 的提出论文，在真实机器人上实现99%成功率，比DAgger快40%

## 相关概念

- [[条件流匹配|Flow Matching]]
- [[DPO|Direct Preference Optimization]]
- [[DAgger]]
- [[Action Chunking]]
