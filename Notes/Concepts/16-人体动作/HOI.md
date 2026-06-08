---
type: concept
aliases: [Human-Object Interaction, 人物交互, 人体-物体交互]
---

# HOI (Human-Object Interaction)

## 定义
人体与物体的交互建模，包括接触区域、抓握姿态、运动序列的联合生成与重建。

## 核心要点
1. 需要同时建模人体姿态（通常用 [[SMPL]]）和物体位姿（6-DoF）
2. 关键挑战：物理合理性（防穿透）、接触一致性、多样性
3. 常见方法：从视频/3D资产合成 HOI 数据，再通过运动重定向迁移到机器人

## 代表工作
- [[GRAIL]]: 从3D资产+视频先验生成人形机器人 loco-manipulation 演示
- CHOIS: 语言条件下的 HOI 生成
- GENMO: 通用人体动作生成

## 相关概念
- [[SMPL]] — 人体参数化模型
- [[FoundationPose]] — 物体位姿估计
- [[运动重定向]] — HOI→机器人的迁移工具
- [[SONIC]] — 基于 HOI 数据训练的人形机器人策略
