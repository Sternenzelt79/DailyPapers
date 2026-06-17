---
type: concept
aliases: [MimicGen数据增强, 自动演示生成]
---

# MimicGen

## 定义
MimicGen 是一个从少量人类演示自动生成大规模机器人演示数据集的系统，通过轨迹分割和重组在仿真中批量生成新场景的演示。

## 核心要点
1. 将人类演示分割为以物体为中心的子任务片段，在新场景中重组
2. 无需额外人类标注即可生成数千条演示，显著降低数据采集成本
3. 在 [[MuJoCo]] / [[ManiSkill3]] 等仿真器中运行，支持 sim-to-real 迁移
4. 适用于接触丰富的操作任务（contact-rich manipulation）

## 代表工作
- MimicGen (NVIDIA, 2023): 原始系统论文

## 相关概念
- [[Imitation Learning]]
- [[DAgger]]
- [[MuJoCo]]
