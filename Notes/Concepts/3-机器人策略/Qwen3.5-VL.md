---
type: concept
aliases: [Qwen3.5-VL, Qwen VL, Qwen3.5 Visual Language Model]
---

# Qwen3.5-VL

## 定义
Qwen3.5-VL 是阿里云发布的视觉-语言模型系列，涵盖 4B 到 72B+ 参数规模，在多模态理解、视频理解、细粒度视觉问答等任务上展现出优异性能，是 FineVLA 的骨干模型。

## 核心要点
1. 支持图像、视频等多种视觉输入
2. 4B 轻量版适合端侧部署和策略网络骨干
3. 397B MoE 大版（Qwen3.5-397B-A17B）用于 FineVLA 标注模型 RoboFine-VLM

## 代表工作
- [[FineVLA]]: 使用 Qwen3.5-4B 作为两种策略架构（StarVLA-OFT 和 StarVLA-GR00T）的共享骨干，使用 Qwen3.5-397B-A17B 微调为 RoboFine-VLM

## 相关概念
- [[GR00T N1.5]]
- [[OpenVLA-OFT]]
- [[DiT]]
