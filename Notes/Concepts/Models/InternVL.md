---
type: concept
aliases: [InternVL, InternVL2, InternVL3]
---

# InternVL

## 定义
InternVL：上海 AI Lab 开源的视觉-语言大模型系列，以强视觉编码器（InternViT）和大语言模型结合为特征，在多模态理解 benchmark 上持续保持 SOTA。

## 数学形式
$$p(y | I, q) = p_{LLM}(y | f_{vision}(I), q)$$

## 核心要点
1. InternViT：专门训练的强视觉编码器（6B 参数），能力远超 CLIP ViT-L
2. Dynamic Resolution：自适应分辨率，保留高分辨率图像细节
3. 系列包括 InternVL 1.0/1.5/2.0/3.0，持续迭代
4. 作为 backbone 被多个机器人/具身 AI 工作采用

## 代表工作
- InternVL / InternVL-Chat 系列（Chen et al., 上海 AI Lab）

## 相关概念
- [[VLA]]
- [[SigLIP2]]
- [[Qwen3-VL]]
