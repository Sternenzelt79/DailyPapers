---
type: concept
aliases: [Xtrainer, Dual-arm Xtrainer]
---

# Xtrainer

## 定义
Xtrainer 是一款双臂机器人平台，用于真实世界机器人操作研究，配备多路视觉传感器支持双臂协同操作任务。

## 核心要点
1. **双臂设计**: 支持双臂协同操作，可完成需要双手配合的复杂任务
2. **视觉传感配置**:
   - RealSense D455（Eye-on-Base，头顶第三视角）：提供全局场景视角
   - RealSense D405（Eye-on-Hand，手腕近景）：提供精确手部操作反馈
3. **研究应用**: 在 [[MaskWAM]] 论文中用于语言清晰和语言歧义两类真机操作任务评测

## 代表工作
- [[MaskWAM]]: 在 Xtrainer 平台上验证了 Mask 提示在语言歧义场景下的泛化能力（Novel Instances 74.6%，Distractors 90.4%）

## 相关概念
- [[World Action Model]]
- [[遥操作]]
