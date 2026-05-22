---
type: concept
aliases: [Grounding DINO, Open-Set Detection]
---

# GroundingDINO

## 定义
开放词汇目标检测模型，结合 DINO 检测器和语言 grounding，支持用自然语言描述指定检测目标，无需固定类别集合。

## 核心要点
1. 将 CLIP 语言特征与 DINO 视觉特征融合
2. 支持任意文本描述作为查询
3. 在 GesVLA 中用于把手势指向区域关联到目标物体

## 相关概念
- [[CLIP]]
- [[GesVLA]]
