---
type: concept
aliases: [LeRobot dataset format, HuggingFace LeRobot]
---

# LeRobot

## 定义

HuggingFace 推出的机器人学习开源库与数据集格式，提供标准化的机器人演示数据收集、训练和评估工具链，旨在降低机器人学习的入门门槛。

## 核心要点

1. 统一的数据集格式，支持多种机器人平台（SO-100、Aloha、Koch 等）
2. 内置扩散策略（Diffusion Policy）、ACT 等模仿学习算法实现
3. 数据集托管在 HuggingFace Hub，方便共享和复用
4. 与 Gymnasium 接口兼容，便于下游规划任务

## 代表工作

- [[StableWorldModel]]: swm 支持 LeRobot 格式数据集直接转换为 Lance 格式，无缝接入训练管道

## 相关概念

- [[HDF5]]
- [[Lance]]
- [[World Model]]
