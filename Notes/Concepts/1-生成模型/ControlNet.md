---
type: concept
aliases: [ControlNet, 条件控制扩散网络]
---

# ControlNet

## 定义
Zhang 等人提出的扩散模型空间控制框架，通过在 U-Net/DiT 旁接一个可训练的复制分支，实现对生成过程的精确空间条件控制（边缘、深度、骨架等）。

## 数学形式
$$y = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c); \Theta_{z2})$$

其中 $\mathcal{Z}$ 为 zero convolution（初始化为零，训练初期不破坏预训练模型），$c$ 为控制条件。

## 核心要点
1. 旁路分支：复制 U-Net encoder，可训练；原始模型参数锁定
2. Zero Convolution：零初始化卷积保证训练初期输出不受影响
3. 支持多种空间条件：Canny边缘、深度图、HED、骨架、法线图等
4. 推理时多条件可叠加

## 代表工作
- Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models" (ICCV 2023)

## 相关概念
- [[Diffusion Model]]
- [[DiT]]
- [[条件注入]]
