---
type: concept
aliases: [潜变量引导, 场景演化潜变量注入]
---

# Latent Steering

## 定义
将 [[World-Action Model|WAM]] 的场景演化潜变量通过交叉注意力机制注入 VLM 隐层状态，为感知阶段提供动态场景先验，而非使用解码后的像素级未来帧。

## 数学形式

$$
D^w_t = f_{\text{dyn}}(Z^w_t) + \rho_{\text{fut}}
$$

$$
\bar{H}_t = H_t + \text{CrossAttn}(H_t, D^w_t)
$$

其中 $Z^w_t$ 为 WAM 场景演化潜变量，$f_{\text{dyn}}$ 为动态编码器，$\rho_{\text{fut}}$ 为未来时刻嵌入，$H_t$ 为原始 VLM 隐层状态。

## 核心要点
1. 使用潜空间而非像素空间：过滤纹理、光照等行动无关细节，保留控制相关的动态结构
2. 残差交叉注意力：保留 VLM 原始语义的同时引入场景演化先验
3. 对 WAM 去噪步数不敏感：1步/3步/5步去噪结果相近（84.5~84.7%）

## 代表工作
- [[WorldPilot]]: 提出 Latent Steering，在 LIBERO-Plus 零样本 OOD 基准单独带来 +3.2 点提升

## 相关概念
- [[Action Steering]]
- [[World-Action Model]]
- [[Cross-Attention]]
- [[VAE]]
