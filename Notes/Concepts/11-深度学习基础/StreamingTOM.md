---
type: concept
aliases: [StreamingTOM]
---

# StreamingTOM

## 定义
专为流式视频理解设计的模型，侧重 Theory of Mind（ToM）推理，即从连续视频帧中推断物体/场景的持续状态，是 [[OVO-S-Bench]] 的参与评测模型之一。

## 核心要点
1. 处理 streaming（连续增量）视频输入，而非完整视频
2. 关注视频流中不在当前视野内的信息推断（ToM 能力）
3. 在 OVO-S-Bench 四级空间智能评测中被用作基线

## 相关概念
- [[OVO-S-Bench]]
- [[RoboBrain2]]
- [[VeBrain]]
