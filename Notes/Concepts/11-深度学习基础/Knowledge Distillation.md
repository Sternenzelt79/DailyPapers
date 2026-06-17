---
type: concept
aliases: [知识蒸馏, 模型蒸馏, KD]
---

# Knowledge Distillation

## 定义

将大型"教师"模型的知识（输出分布、中间特征或结构先验）迁移给小型"学生"模型，使学生在保持轻量推理的同时获得接近教师的性能。

## 数学形式

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \hat{y}) + (1-\alpha) \mathcal{L}_{soft}(p_T, p_S)
$$

其中 $p_T, p_S$ 分别为教师和学生的 softmax 输出（温度 $T$ 软化），$\alpha$ 控制两路损失比例。

## 核心要点

1. **软标签蒸馏**: 用教师的概率分布（而非硬标签）监督学生，提供更丰富的类间关系信号
2. **特征蒸馏**: 在中间层对齐教师和学生的特征表示，传递结构性知识
3. **训练-推理不对称**: 蒸馏完成后学生独立推理，教师仅在训练时参与
4. **跨模态蒸馏**: 可将几何、语义等异质先验蒸馏到视觉-动作模型中

## 代表工作

- [[WAM4D]]: 将 [[Depth Anything 3]] 的几何先验通过空间寄存器 Token 蒸馏到 WAM，推理时移除几何分支
- [[Flash-WAM]]: 将 WAM 的视频生成能力蒸馏为一致性模型，实现 23× 推理加速

## 相关概念

- [[Register Tokens]]
- [[Flow Matching]]
- [[Mixture-of-Transformers]]
