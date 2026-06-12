---
type: concept
aliases: [领域随机化, domain randomization, DR]
---

# Domain Randomization

## 定义

在仿真训练中，通过随机化视觉外观、物理参数、光照、相机位姿等环境属性，使模型对真实世界的分布变化具有鲁棒性，从而减小 sim-to-real 迁移差距。

## 数学形式

$$
\pi^* = \arg\max_\pi \mathbb{E}_{\xi \sim p(\xi)} \left[ \sum_t r(s_t, a_t) \right]
$$

其中 $\xi$ 为从随机化分布 $p(\xi)$ 采样的环境参数（纹理、光照、物理属性等）。

## 核心要点

1. **视觉随机化**: 纹理颜色、光照强度/方向、相机内外参随机扰动
2. **物理随机化**: 质量、摩擦系数、关节阻尼等物理属性随机化
3. **空间随机化**: 物体位置、姿态的随机布置
4. **六轴随机化（RoboGenesis）**: 场景、视觉杂乱度、摄像机、物体、光照、空间位置

## 代表工作

- [[LabVLA]]: 采用六轴随机化策略，ID→OOD 成功率仅下降 1.1pp

## 相关概念

- [[Flow Matching]]
- [[VLA]]
