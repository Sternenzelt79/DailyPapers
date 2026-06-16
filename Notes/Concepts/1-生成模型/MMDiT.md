---
type: concept
aliases: [MMDiT, Multi-Modal Diffusion Transformer, 多模态扩散变换器, Double-Stream MMDiT]
---

# MMDiT（多模态扩散变换器）

## 定义

MMDiT（Multi-Modal Diffusion Transformer）是一种多流扩散变换器架构，为每个模态（如文本流和图像/视频流）维护独立的参数集，再通过[[Joint Self-Attention|联合注意力]]机制在每个Transformer层实现跨模态交互。与单流共享参数的方案相比，MMDiT能更好地保留各模态的独立特征，同时实现深度的模态融合。

## 数学形式

$$
\begin{aligned}
[Q_v, K_v, V_v] &= \text{Norm}(X_v) W_v^{QKV} \\
[Q_t, K_t, V_t] &= \text{Norm}(X_t) W_t^{QKV} \\
\text{Attn} &= \text{softmax}\!\left(\frac{[Q_v; Q_t][K_v; K_t]^\top}{\sqrt{d}}\right)[V_v; V_t]
\end{aligned}
$$

两个流的Token拼接后做统一的自注意力，输出再拆回各自流。

## 核心要点

1. **双流独立参数**: 视觉流和语言流各自拥有独立的 QKV 投影矩阵，避免参数共享带来的模态干扰
2. **联合注意力**: 拼接两个流的 K/V 后做一次自注意力，实现每层的双向模态交互
3. **灵活扩展**: 可扩展为多流架构（如同时处理视觉、语言、动作三个流）
4. **冻结与微调**: 实践中常冻结预训练的语言流（如 MLLM 编码器），只微调视觉流参数

## 代表工作

- [[SD3]] / [[FLUX]]: 文本-图像生成中的标准 MMDiT 实现
- [[QwenRobotWorld]]: 60层 Double-Stream MMDiT，耦合冻结 Qwen2.5-VL 与 Video-VAE，用于具身视频世界建模

## 相关概念

- [[Diffusion Transformer]]: MMDiT 的单流版本（参数共享）
- [[Joint Self-Attention]]: MMDiT 中的核心注意力机制
- [[Video VAE]]: 常与 MMDiT 结合处理视频生成任务
- [[扩散模型]]: MMDiT 所属的生成模型范畴
