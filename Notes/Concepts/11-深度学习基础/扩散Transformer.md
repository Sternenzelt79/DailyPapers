---
type: concept
aliases: [Diffusion Transformer, DiT, Transformer-based Diffusion]
---

# 扩散Transformer

## 定义

以 Transformer 架构（而非传统 U-Net）为骨干的扩散模型，通过自注意力机制处理图像 patch 序列，代表工作为 DiT（Diffusion Transformer）。

## 数学形式

将图像 $x$ 分割为 patch 序列 $\{p_1, ..., p_N\}$，Transformer 处理带时间步嵌入的 patch 序列：

$$\varepsilon_\theta(x_t, t) = \text{Transformer}(\text{Patchify}(x_t), \text{Embed}(t))$$

## 核心要点

1. 替换 U-Net 的卷积归纳偏置，改用全局自注意力，更适合处理长序列
2. 时间步 $t$ 通过 AdaLN（自适应层归一化）或 cross-attention 注入
3. 可通过滑动窗口注意力扩展到视频/长序列生成
4. 比 U-Net 更易于扩展参数量（scaling law 更友好）

## 代表工作

- [[CoME]]: 以 DiT 为基础骨干，叠加多记忆专家
- [[GigaWorld]]: 大规模扩散 Transformer 世界模型

## 相关概念

- [[扩散模型]]
- [[自注意力机制]]
- [[滑动窗口注意力]]
- [[Transformer]]
