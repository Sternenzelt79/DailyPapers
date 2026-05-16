---
type: concept
aliases: [OXE, Open X-Embodiment, Open-X Embodiment Dataset]
---

# OXE Dataset（Open X-Embodiment）

## 定义

Open X-Embodiment（OXE）是一个跨机器人平台的大规模机器人操作数据集聚合，统一了来自多个研究机构的超过 100 万条轨迹，跨越 22 种机器人形态和 527+ 种技能。

## 核心要点

1. **规模**：1M+ 轨迹，22 个机器人平台，527 种技能，160k+ 任务，60 个环境
2. **收集方式**：遥操作、脚本、人工演示混合
3. **观测模态**：RGB、深度、文本、点云
4. **意义**：首个大规模跨机器人数据聚合，为 VLA 和 WAM 的跨形态泛化训练奠定基础
5. **WAM 价值**：迫使世界模型将普遍物理定律与特定机器人运动学解耦，实现鲁棒的形态无关泛化

## 代表工作

- [[WAMSurvey]]: WAM 训练数据生态中最重要的机器人遥操作数据集聚合

## 相关概念

- [[World Action Model]]
- [[VLA]]
- [[LIBERO]]
