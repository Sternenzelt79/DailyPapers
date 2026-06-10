---
type: concept
aliases: [Binary Cross-Entropy Loss, BCE, 二元交叉熵损失, 二分类交叉熵]
---

# BCE 损失（Binary Cross-Entropy Loss）

## 定义

BCE 损失是用于二分类任务的标准损失函数，度量预测概率与真实二值标签之间的交叉熵，数值越小表示预测越接近真实标签。

## 数学形式

$$
\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]
$$

多样本版本：

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)\right]
$$

## 核心要点

1. **适用场景**: 输出为 0/1 二值标签的监督分类任务（如路由器激活判断）
2. **与 sigmoid 配合**: 通常在 BCE 损失前对 logit 施加 sigmoid 归一化为概率
3. **对不平衡数据敏感**: 正负样本极不平衡时可用 Focal Loss 或加权 BCE 改进

## 代表工作

- [[AdaWAM]]: 用 BCE 损失训练动态路由器，监督 `<TR>` 和 `<VR>` 两个二值路由 token

## 相关概念

- [[Flow Matching]]: AdaWAM 中与 BCE 联合使用的生成模型损失
- [[CoT]]: BCE 损失监督的文本推理路由决策
