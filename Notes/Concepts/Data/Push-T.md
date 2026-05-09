---
type: concept
aliases: [PushT]
---

# Push-T

## 定义
2D 接触动力学操作 benchmark：用一个圆形 agent 把 T 形物体推到目标位姿。

## 核心要点
1. 富接触、需要精细规划。
2. 已成为 [[Diffusion Policy]]、world model 等方法的标准测试床。
3. 状态包括 agent 位置和 block 位置/角度。

## 代表工作
- [[Diffusion Policy]]：在 PushT 上首次证明扩散策略
- [[LeWM]]：用 latent 探针在 PushT 上验证物理结构

## 相关概念
- [[World Model]]
- [[Diffusion Policy]]
