---
type: concept
aliases: [Adversarial Skill Embeddings, ASE]
---

# ASE（Adversarial Skill Embeddings）

## 定义
AMP 的扩展，在对抗动作先验基础上引入技能嵌入（skill embedding）空间，支持学习可组合的多技能潜空间，实现对运动风格的潜变量控制。

## 数学形式

$$
\pi(a | s, z), \quad z \sim p(z)
$$

其中 $z$ 为技能潜变量，策略以技能条件执行。

## 核心要点
1. 扩展 AMP，增加技能潜变量 $z$，实现可控多技能生成
2. 使用信息最大化目标保证技能多样性
3. 与 AMP 同样面临数据多样性增加时的 mode collapse 问题

## 代表工作
- [[SONIC]]: 以 ASE 为对比基线，展示规模化动作追踪在大规模多样数据下的优势

## 相关概念
- [[AMP]]
- [[PPO]]
- [[运动跟踪]]
