---
type: concept
aliases: [CLEVR-Skills, ClevrSkills benchmark]
---

# ClevrSkills

## 定义
基于 CLEVR 场景的组合推理与技能泛化基准，测试模型将视觉推理与具身操作技能结合的能力。

## 核心要点
1. 在合成 CLEVR 场景中测试语言条件化推理 + 操作
2. 要求模型组合多个基础技能完成复杂语言指令
3. 用于评估 VLA 在链式推理（CoT）和直接动作预测之间的权衡（见 [[HyT]]）
4. 合成环境意味着结论泛化到真实场景需要额外验证

## 代表工作
- 主要作为 VLA 推理能力的评估基准使用

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[RoboTwin2]]
