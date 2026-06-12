---
type: concept
aliases: [FDM, 前向动态模型, Forward Model]
---

# Forward Dynamics Model

## 定义

给定当前状态和动作（或潜在动作），预测下一个状态的模型；在潜在空间操作时，常将动作分解为运输算子（Transport Operator）和残差两部分。

## 数学形式

$$
K_t, \delta_t = f_\psi(z_t, \ell_t), \quad \hat{z}_{t+1} = K_t z_t + \delta_t
$$

其中 $K_t$ 为运输算子（软空间路由），$\delta_t$ 为残差，$\ell_t$ 为潜在动作。

## 核心要点

1. 与逆向动态模型（IDM）配对使用，形成 encoder-decoder 式的潜在动作学习闭环
2. 运输算子 $K_t$ 类比语义空间中的光流，描述 token 的空间迁移
3. 残差 $\delta_t$ 捕捉无法被纯运输解释的语义变化（如物体消失/出现）
4. 训练时用前向预测损失和反向一致性损失约束

## 代表工作

- [[RepWAM]]: FDM 在语义对齐的视觉空间内学习，使潜在动作更易迁移到机器人动作解码

## 相关概念

- [[Inverse Dynamics Model]]
- [[Latent Action]]
- [[World-Action Model]]
