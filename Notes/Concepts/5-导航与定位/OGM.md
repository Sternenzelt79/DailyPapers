---
type: concept
aliases: [Occupancy Grid Map, 占用栅格地图]
---

# OGM (Occupancy Grid Map)

## 定义
将环境离散化为固定分辨率的栅格，每个格子存储占用概率（0=free，1=occupied），是移动机器人最常用的地图表示之一。

## 数学形式
$$P(\text{occ}_{i,j} | z_{1:t}) = \frac{P(z_t | \text{occ}_{i,j}) P(\text{occ}_{i,j} | z_{1:t-1})}{P(z_t | z_{1:t-1})}$$

## 核心要点
1. 来自激光雷达/深度相机的测量，用贝叶斯更新每格概率
2. 优点：表示简单，查询 O(1)；缺点：分辨率固定，存储随范围平方增长
3. 与 World Model 结合（如 MAD）：OGM 提供显式空间记忆，WM 提供预测能力

## 代表工作
- [[MAD]]: OGM-conditioned DreamerV3 用于无人机敏捷飞行
- SkyDreamer: 基于 OGM 的无人机世界模型

## 相关概念
- [[MuJoCo]] — 仿真环境中常用 OGM 做碰撞检测
- [[DreamerV3]] — 与 OGM 结合的世界模型
- [[RSSM]] — OGM 的潜在空间对应物
