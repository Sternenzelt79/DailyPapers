---
type: concept
aliases: [PartNet Dataset, Part-level 3D Dataset]
---

# PartNet

## 定义
一个大规模的 part-level 3D 形状标注数据集，提供层次化的语义 part 分割标注，是 3D 内容理解和生成研究的重要基础数据集。

## 核心要点
1. 包含 26,671 个 3D 模型的 573,585 个 part 实例标注
2. 支持层次化 part 分解（粗→细粒度）
3. 覆盖 24 类常见物体（家具、电器等）
4. [[PhysForge]] 在 PartNet 基础上构建 PhysDB 数据集，增加物理/运动学属性标注

## 代表工作
- [[PhysForge]]: 基于 PartNet 扩展构建物理感知的 PhysDB 数据集

## 相关概念
- [[PhysForge]]
- [[RoboTwin]]
