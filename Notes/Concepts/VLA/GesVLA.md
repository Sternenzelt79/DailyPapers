---
type: concept
aliases: [Gesture-Aware VLA, Gesture Instruction]
---

# GesVLA

## 定义
在 VLA 中引入手势作为并行指令模态，用 MediaPipe 检测手势指向，用 GroundingDINO 关联目标物体，解决语言指令在多目标复杂场景下的空间歧义问题。

## 核心要点
1. 手势 + 语言双模态指令输入
2. MediaPipe → 手势位置；GroundingDINO → 目标区域定位
3. Backbone 使用 PaliGemma，在真实机械臂上验证

## 代表工作
- [[GesVLA]]: Guo Wenxuan et al., 2026 (arXiv 2605.22812)

## 相关概念
- [[VLA]]
- [[GroundingDINO]]
