---
type: concept
aliases: [视频预测, Future Video Prediction, 未来视频预测]
---

# Video Prediction

## 定义
基于当前观测和条件信息（语言、状态）预测未来帧序列的任务，在机器人学习中作为世界模型的核心能力。

## 核心要点
1. 可分为高保真预测（追求外观细节）和结构性预测（只需捕获几何/运动线索）
2. 机器人控制场景中，结构性粗粒度预测即足够引导动作生成，无需高保真重建
3. 常与流匹配或扩散模型结合实现

## 代表工作
- [[Efficient-WAM]]: 验证了低保真未来预测对控制的有效性，视频分支仅用 2-5 去噪步
- [[GigaWorld-Policy]]: 高保真视频预测用于机器人策略

## 相关概念
- [[Video Diffusion Model]]
- [[World-Action Model]]
- [[Flow Matching]]
