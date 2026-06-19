---
type: concept
aliases: [Mark-based Open-vocabulary Key Action]
---

# MOKA

## 定义
通过在图像上标注关键动作位置（affordance marks）并用 VLM 推理动作的开放词汇操作方法，将 VLM 的语义理解转化为空间动作指令。

## 核心要点
1. 将操作分解为关键位置标注 + VLM 推理两阶段
2. 使用视觉 mark（如圆圈、箭头）在图像上标注 affordance 区域
3. 无需专门训练，利用 VLM 的 zero-shot 视觉推理能力

## 代表工作
- [[VAP]]: 对比基线

## 相关概念
- [[PIVOT]]
- [[VAP]]
