---
type: concept
aliases: [LAPA, Latent Action Pre-training from videos]
---

# LAPA

## 定义
LAPA（Latent Action Pre-training from videos）通过 VQ-VAE 学习视频中的离散 latent action 表示，然后用这些 latent action 预训练机器人策略，无需真实动作标注。

## 核心要点
1. VQ-VAE 把帧间变化编码为离散 latent token
2. 用大量无标注互联网视频预训练 latent action model
3. 下游任务只需少量带动作标注的机器人数据微调

## 代表工作
- LAPA (2024), Latent Action Pretraining from Videos

## 相关概念
- [[LAPO]]
- [[AdaWorld]]
- [[CLAW]]
