---
type: concept
aliases: [SONIC, Supersizing Motion Tracking]
---

# SONIC

## 定义
NVIDIA 开发的规模化人形全身控制系统，用 3M 动作数据 + SMPL+FSQ tokenization + PPO 训练，展示人形控制的清晰 scaling law。

## 核心要点
1. 3M 动作数据规模，SMPL 参数化 + FSQ 离散化
2. PPO 训练，不依赖 imitation learning reference matching
3. 明确的 scaling law：data / model / compute 三个维度均有效
4. 对比 OpenHomie、PHUMA、BeyondMimic

## 代表工作
- [[SONIC]]: Zhengyi Luo, Ye Yuan, et al., NVIDIA 2025/2026 (arXiv 2511.07820)

## 相关概念
- [[SMPL]]
- [[GR00T-N1.6]]
