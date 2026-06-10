---
type: concept
aliases: [CHOIS, 接触感知HOI合成]
---

# CHOIS（Contact-aware HOI Synthesis）

## 定义

接触感知的人-物交互合成方法，通过显式建模手物接触约束来提升 HOI 生成的物理可行性。

## 核心要点

1. 显式建模接触时序与接触位置，减少穿透问题
2. 未利用 3D 场景约束，在机器人追踪任务中泛化性有限
3. 在 GRAIL 对比中：SR 10.5%，Penetration 3.74%（高于 GRAIL 0.90%）

## 代表工作

- [[GRAIL]]: 作为接触感知 HOI 生成基线，GRAIL 在所有指标上超越

## 相关概念

- [[HOI]]
- [[HOIDiff]]
- [[DAViD]]
- [[Chamfer Distance]]
