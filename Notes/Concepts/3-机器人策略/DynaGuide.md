---
type: concept
aliases: [DynaGuide]
---

# DynaGuide

## 定义
DynaGuide 是一种在 VLA 推理时通过修改目标奖励信号动态引导生成轨迹的方法，无需重新训练策略即可实现 test-time steering。

## 核心要点
1. 在自回归 VLA 的 token 生成过程中插入 guidance 信号
2. 通过调整奖励/目标函数引导策略偏向指定行为
3. 不需要 fine-tuning，属于 inference-time intervention 方法
4. 可与 teleoperation 信号结合，实现人机协作控制

## 代表工作
- DynaGuide: Dynamic Guidance for VLA Policies (2025)

## 相关概念
- [[Token Steering]]
- [[SAPS]]
- [[OpenVLA]]
