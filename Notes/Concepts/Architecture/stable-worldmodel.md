---
type: concept
aliases: [stable-worldmodel, WM Platform]
---

# stable-worldmodel

## 定义
LeCun 组（Mila/NYU）开发的世界模型研究标准化平台，统一训练/评测接口，解决 WM 研究的可复现性问题；引入 MakeFramed 和 FrameRestore 两个新评测任务。

## 核心要点
1. 基于 PLDM/LeWM 架构的统一训练接口
2. 高效视频数据加载，解决 WM 训练 I/O 瓶颈
3. MakeFramed（填帧预测）和 FrameRestore（帧恢复）用于 latent 质量评估
4. 集成 CEM/MPPI 规划器实现完整 WM-based control

## 代表工作
- [[stable-worldmodel]]: Lucas Maes et al., 2026 (arXiv 2605.21800)

## 相关概念
- [[LeWM]]
- [[PLDM]]
- [[Model Predictive Control]]
