---
type: concept
aliases: [光流, 光学流, Optical Flow]
---

# Optical Flow

## 定义

图像序列中每个像素在相邻帧之间的运动向量场，表示场景中物体或摄像机运动引起的视觉位移。

## 数学形式

对于像素 $(x, y)$，光流向量 $\Phi_t(x,y) = (u, v)$ 满足亮度恒常性假设：

$$
I(x, y, t) = I(x+u, y+v, t+1)
$$

在机器人学中常按视图归一化：
$$
\tilde{\Phi}^{(v)}_t(x,y) = \left[\frac{\Phi^{(v)}_{t,x}(x,y)}{W_v},\; \frac{\Phi^{(v)}_{t,y}(x,y)}{H_v}\right]
$$

## 核心要点

1. **运动信息**: 光流编码了场景中的运动，是视频理解和机器人操作的重要线索
2. **无监督监督信号**: 光流可作为学习运动表示（[[Latent Action]]）的无监督监督，不需要动作标注
3. **训练时使用**: 在策略学习中，光流通常只在训练时提取，推理时不需要（因果策略）
4. **估计方法**: FlowFormer、RAFT、PWC-Net 等深度学习方法可高效估计稠密光流

## 代表工作

- [[HiMem-WAM]]: 用光流重建作为低级潜动作分词器的自监督信号（Stage I）

## 相关概念

- [[Latent Action]]
- [[Variational Autoencoder]]
- [[World-Action Model]]
