---
type: concept
aliases: [Action with Visual Primitives, Visual Primitives]
---

# AVP

## 定义
将 VLA 的感知与控制解耦：先用 VLM+SAM 提取视觉原语（semantic segmentation masks + keypoint waypoints），再用轻量 action head 根据原语生成动作。

## 核心要点
1. 解耦 instruction comprehension / scene understanding / motor control
2. 视觉原语 = SAM 分割 mask + 关键点 waypoints
3. 更适合 OOD 场景，在跨域操作任务上验证

## 代表工作
- [[AVP]]: Weilong Guo et al., 2026 (arXiv 2605.22183)

## 相关概念
- [[VLA]]
- [[OpenVLA]]
