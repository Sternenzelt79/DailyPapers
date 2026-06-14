---
type: concept
aliases: [Segment Anything Model 3, SAM3, SAM-3]
---

# SAM-3

## 定义
SAM-3（Segment Anything Model 3）是 Meta 发布的第三代通用实例分割模型，支持从用户点击、边界框或文本提示出发，在视频序列中自动跟踪目标对象并生成逐帧 [[Instance Segmentation Mask]]。

## 核心要点
1. **零样本分割**: 仅需首帧单次用户点击即可初始化，无需对特定对象预训练
2. **视频跟踪**: 内置跨帧跟踪能力，自动传播首帧 Mask 到后续帧，适合机器人操作数据标注
3. **标注效率**: MaskWAM 报告使用 SAM-3 标注时，91% 的 episode 无需人工修正，约每 50 个 episode 仅需 3 分钟审核

## 代表工作
- [[MaskWAM]]: 利用 SAM-3 为训练数据自动生成 RGB-Mask-Action 三元组，并在真机部署时从首帧用户点击获取目标 Mask

## 相关概念
- [[Instance Segmentation Mask]]
- [[Future Mask Prediction]]
- [[World Action Model]]
