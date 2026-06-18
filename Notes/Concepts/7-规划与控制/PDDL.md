---
type: concept
aliases: [PDDL, Planning Domain Definition Language, 规划域定义语言]
---

# PDDL

## 定义
Planning Domain Definition Language，符号 AI 中用于描述规划问题的标准语言，将世界状态表示为 predicates 的集合，动作表示为前提条件（preconditions）和效果（effects）的对，规划器在此基础上搜索达到目标状态的动作序列。

## 数学形式
$$\text{Action}(a) = \langle \text{pre}(a), \text{eff}^+(a), \text{eff}^-(a) \rangle$$

其中 $\text{pre}(a)$ 为前提条件 predicates，$\text{eff}^+(a)$ 为添加的 predicates，$\text{eff}^-(a)$ 为删除的 predicates。

## 核心要点
1. 将世界建模为离散符号状态，便于形式化推理和验证
2. 规划器（如 FAST-Downward、SPIN）可在 PDDL 上做精确搜索
3. 优点：可解释、可验证、保证正确性
4. 缺点：需要手工设计 predicates，在 open-world 场景下难以完备建模
5. 近年结合 LLM 自动生成 PDDL predicates，降低工程成本

## 代表工作
- [[ReSYNC]]: 用 VLM 从失败中自动发现 causal predicates，整合进 PDDL 规划
- [[IVNTR]]: 传统符号规划 baseline

## 相关概念
- [[7-规划与控制]]
- [[ReAct]]
