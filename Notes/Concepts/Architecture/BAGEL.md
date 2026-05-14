---
type: concept
aliases: [BAGEL, Bootstrapped Augmented Generation]
---

# BAGEL

## 定义
BAGEL：统一多模态理解与生成的大型视觉-语言模型，通过共享表示空间同时处理图像理解和生成任务，是 unified multimodal model 领域的代表工作之一。

## 数学形式
$$p(y | x) = p_\theta(\text{understand}(x)) \oplus p_\theta(\text{generate}(x))$$

## 核心要点
1. 统一 encoder 和 decoder，不再将理解/生成分为两个独立模型
2. 训练时混合理解任务（VQA、captioning）和生成任务（image synthesis）
3. 与 [[OmniGen2]]、[[SenseNova-U1]] 属于同一 "unified multimodal" 范式

## 代表工作
- BAGEL 论文（ByteDance 出品，具体 arXiv 待查）

## 相关概念
- [[OmniGen2]]
- [[LDM]]
- [[VLA]]
