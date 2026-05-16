---
type: concept
aliases: [Continuous-Time Distribution Matching, Continuous-Time DMD]
---

# CDM

## 定义
将 [[DMD]] (Distribution Matching Distillation) 从离散时步对齐扩展到连续时间的扩散蒸馏方法，无需 GAN 或奖励模型辅助即可获得高保真少步生成。

## 核心要点
1. 原版 DMD 只在少数预定义离散时步上做分布匹配，导致模式搜索偏差和视觉模糊
2. CDM 引入动态连续调度（随机长度），在采样轨迹上任意时刻执行分布匹配
3. 连续时间对齐目标：在 student 速度场推断的外插点（off-trajectory）上做匹配，提升泛化
4. 在 SD3-Medium 和 Longcat-Image 上验证，无辅助模块

## 数学形式
$$\mathcal{L}_\text{CDM} = \mathbb{E}_{t \sim \mathcal{U}[0,T]}\left[D_{KL}(p_\theta(x_0|x_t) \| p_\text{real})\right]$$

## 代表工作
- [[CDM]] (2605.06376): Continuous-Time Distribution Matching for Few-Step Diffusion Distillation

## 相关概念
- [[DMD]] — 前驱离散时步版本
- [[Diffusion Model]] — 基础框架
- [[CFG]] — 推理时配合使用
