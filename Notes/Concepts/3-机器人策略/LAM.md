---
type: concept
aliases: [LAM, Latent Action Model, 潜在动作模型]
---

# LAM（Latent Action Model）

## 定义
潜在动作模型，通过自监督学习从无标注视频中提取潜在动作表示，将人类视频与机器人控制空间连接起来，用于跨体现学习。

## 数学形式
$$z_a = \text{Enc}(s_t, s_{t+1}), \quad \hat{a} = \text{Dec}(z_a)$$

其中 $z_a$ 为潜在动作向量，通过前后帧对比推断隐式动作。

## 核心要点
1. 核心动机：人类视频数量远超机器人演示数据，LAM 使无动作标注的视频可用于策略训练
2. 编码器从连续帧对中推断"动作"，解码器将潜在动作映射到机器人关节空间
3. 与 VQ（Vector Quantization）结合可离散化潜在动作空间
4. HARP-VLA 等工作在 LAM 基础上进一步对齐人机视觉表征

## 代表工作
- [[HARP-VLA]]: 在 LAM 基础上加入双向视觉对齐，同时解决视觉和动作跨体现问题
- [[UniVLA]]: 使用 LAM 框架统一多种体现的策略训练

## 相关概念
- [[跨体现学习]]
- [[VLA]]
- [[Latent-Action]]
