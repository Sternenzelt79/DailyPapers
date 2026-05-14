---
type: concept
aliases: [OmniGen2]
---

# OmniGen2

## 定义
OmniGen2：统一多模态理解与生成的视觉-语言模型，与 [[BAGEL]]、[[SenseNova-U1]] 属于同一"unified multimodal"范式，试图消除理解模型和生成模型之间的架构分割。

## 数学形式
$$p_\theta(y | x_{multi}) \text{ where } x_{multi} \in \{\text{text, image, video}\}$$

## 核心要点
1. 统一架构同时处理 image understanding、captioning、image generation、editing
2. 共享表示空间使得理解信息能直接指导生成，减少 cascade pipeline 的误差累积
3. 与 [[BAGEL]] 竞争同一赛道

## 代表工作
- OmniGen2 论文（具体 arXiv 待查）

## 相关概念
- [[BAGEL]]
- [[LDM]]
- [[DiT]]
