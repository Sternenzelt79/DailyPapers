---
type: concept
aliases: [Preemptive Runtime Verification, NoiseGate, ARGUS]
---

# Pre-VLA

## 定义
VLA 运行时抢先验证框架，在动作执行前检测动作质量（NoiseGate）和 OOD 异常（ARGUS），拦截低质量动作，避免触发代价高的 world model rollout。

## 核心要点
1. NoiseGate 检测 action distribution 不确定性
2. ARGUS 做 OOD detection
3. 预防性拦截，减少物理失败和 WM 渲染浪费
4. 在 LIBERO-10 上验证

## 代表工作
- [[Pre-VLA]]: Zhen Sun, 2026 (arXiv 2605.22446)

## 相关概念
- [[OpenVLA]]
- [[World Model]]
