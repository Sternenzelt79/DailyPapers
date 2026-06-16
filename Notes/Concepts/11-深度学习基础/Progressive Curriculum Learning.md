---
type: concept
aliases: [渐进课程学习, Progressive Curriculum, Curriculum Learning, 课程学习]
---

# 渐进课程学习（Progressive Curriculum Learning）

## 定义

渐进课程学习是一种训练策略，模仿人类学习的由易到难过程：先在简单或通用的数据/任务上预训练，再逐步过渡到复杂或专业的数据/任务上微调。其核心思想是用简单任务建立良好的初始化，为后续专业化训练提供更好的起点。

## 数学形式

$$
\theta^* = \underbrace{\arg\min_\theta \mathcal{L}_{\text{expert}}(\theta; \mathcal{D}_{\text{expert}})}_{\text{阶段2：专家微调}} \quad \text{初始化自} \quad \underbrace{\arg\min_\theta \mathcal{L}_{\text{general}}(\theta; \mathcal{D}_{\text{general}})}_{\text{阶段1：通用预训练}}
$$

## 核心要点

1. **阶段划分**: 通常分为通用预训练阶段（General Stage）和专家微调阶段（Expert Stage），后者依赖前者建立的表示基础
2. **知识保留**: 阶段二在专业数据上微调时，需注意防止灾难性遗忘（Catastrophic Forgetting）
3. **共享接口**: 优秀的课程设计会保持各阶段任务的接口一致（如统一用语言作为动作条件），便于知识迁移
4. **泛化-精度权衡**: 通用阶段提升泛化性，专家阶段提升任务精度，两阶段共同优化

## 代表工作

- [[QwenRobotWorld]]: General+Expert 渐进课程——先学通用视觉先验，再在 EWK 具身数据上注入专业化知识

## 相关概念

- [[迁移学习|Transfer Learning]]: 课程学习的思想与迁移学习高度相关
- [[微调|Fine-tuning]]: 专家阶段通常以微调形式实现
- [[灾难性遗忘|Catastrophic Forgetting]]: 多阶段训练需要解决的核心挑战
