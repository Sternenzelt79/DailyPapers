---
type: concept
aliases: [Memory Maze Dataset, Memory Maze Benchmark]
---

# Memory Maze

## 定义
用于评估视频世界模型长程记忆能力的合成 3D 导航 benchmark，包含 30k 离线轨迹、每条轨迹 1000 帧，场景为程序生成的 3D 迷宫，含位置信号监督。

## 核心要点
1. 30k 轨迹 × 1000 帧/轨迹，覆盖超长上下文场景
2. 包含位置信号（position signals），可评估空间记忆
3. 评估指标：LPIPS、SSIM、PSNR（帧重建质量）
4. 常用于测试世界模型在数百帧上下文下的历史帧召回能力

## 代表工作
- [[CoME]]: 在 Memory Maze 上验证三类记忆专家组合的有效性，LPIPS 从 0.209 降至 0.097
