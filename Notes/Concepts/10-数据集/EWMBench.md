---
type: concept
aliases: [EWMBench, Embodied World Model Benchmark, 具身世界模型基准]
---

# EWMBench（具身世界模型基准）

## 定义

EWMBench（Embodied World Model Benchmark）是专门评估具身世界模型（EWM）的综合基准框架，从视觉场景一致性、动作正确性和语义对齐三个维度系统评估模型生成视频的质量。基于 AgiBot-World 真实机器人操作数据构建评估集。

## 核心要点

1. **三维度评估体系**:
   - **视觉场景一致性**: 使用微调的 DINOv2 特征提取器进行补丁级余弦相似性计算，衡量背景、物体和机器人结构的稳定性
   - **动作正确性**: 对称 Hausdorff 距离（HSD）+ 归一化动态时间规整（nDTW）+ Wasserstein 距离度量动态一致性
   - **语义对齐**: BLEU 分数（指令匹配）+ CLIP 跨模态相似性（细粒度步骤比较）+ 逻辑错误惩罚机制

2. **数据来源**: 基于 AgiBot-World 真实机器人操作数据集，包含多种机器人操作任务场景
3. **持续更新**: 测试数据集于 2026.3.6 发布最新版本

## 代表工作

- [[QwenRobotWorld]]: 在 EWMBench 上排名第一

## 相关概念

- [[具身世界模型|Embodied World Model]]: EWMBench 所评估的模型类别
- [[DreamGen]]: 类似的机器人世界模型评估框架
- [[WorldModelBench]]: 视频世界模型的另一评估基准
