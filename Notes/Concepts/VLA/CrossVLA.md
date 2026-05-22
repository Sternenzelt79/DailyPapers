---
type: concept
aliases: [Cross-Paradigm DPO, SnapFlow]
---

# CrossVLA

## 定义
跨范式 VLA post-training 框架，将 DPO 偏好对齐从 autoregressive VLA 推广到 flow-matching VLA；提出 SnapFlow 适配 ODE-based 生成过程的偏好优化。

## 核心要点
1. 现有 DPO 主要研究 AR VLA（OpenVLA），CrossVLA 同时支持 flow-matching（pi0 系）
2. SnapFlow 是适配 flow-matching 的 DPO 变体
3. 在 LIBERO benchmark 测试

## 代表工作
- [[CrossVLA]]: Liu Zhi, 2026 (arXiv 2605.21854)

## 相关概念
- [[OpenVLA]]
- [[Diffusion Policy]]
- [[Flow Matching]]
