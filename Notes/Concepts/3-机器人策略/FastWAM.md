---
type: concept
aliases: [Fast World-Action Model, FastWAM]
---

# FastWAM

## 定义
FastWAM 是一种高效的 [[World Action Model]]，通过优化视频生成与动作预测的解耦和推理速度，在保持高任务成功率的同时大幅降低推理延迟。

## 核心要点
1. 与 [[World Action Model]] 家族方法相同，联合建模未来 RGB 帧与机器人动作
2. 使用冻结 [[Video VAE]] 编码视觉观测，[[Flow Matching]] 框架训练
3. 在 LIBERO（97.6%）和 RoboTwin 2.0（87.7%）上取得竞争性成绩

## 代表工作
- [[MaskWAM]]: 在 FastWAM 基础上加入 Mask 预测与提示，进一步提升空间定位和语言歧义处理能力

## 相关概念
- [[World Action Model]]
- [[Video VAE]]
- [[Flow Matching]]
- [[LIBERO]]
- [[RoboTwin]]
