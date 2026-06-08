---
type: concept
aliases: [RLBench, Robot Learning Benchmark]
---

# RLBench

## 定义
由 Imperial College London 提出的机器人学习 benchmark 和学习环境，基于 CoppeliaSim（V-REP）仿真器，提供大量多样化操作任务，支持模仿学习和强化学习评估。

## 核心要点
1. 包含 100+ 个操作任务（抓取、放置、开关、拼图等），任务难度跨度大
2. 提供 RGB、深度、点云、关节状态等多模态观测，有 GT 深度直接可用
3. 支持将生成的视频轨迹在仿真器中重放以评估成功率
4. 常用评估任务：Lift、Numbered Block Put、Rubbish InBin、Reach Target、Lamp On、Pick Up Cup、Slide Block to Target、Solve Puzzle

## 代表工作
- [[GEM-4D]]: 在 7 个 RLBench 任务上评估，成功率 63-82%，大幅超越 TesserACT 基线

## 相关概念
- [[ManiSkill3]]
- [[视频世界模型]]
- [[3-机器人策略]]
