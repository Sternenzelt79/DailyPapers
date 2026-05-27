---
type: concept
aliases: [Multi-Layer Perceptron, 多层感知机, 全连接网络]
---

# MLP

## 定义
多层感知机（Multi-Layer Perceptron，MLP）是最基础的前馈神经网络，由输入层、若干隐藏层和输出层组成，每层使用全连接和非线性激活函数。

## 数学形式

$$
h^{(l+1)} = \sigma(W^{(l)} h^{(l)} + b^{(l)})
$$

其中 $\sigma$ 为激活函数（如 ReLU、GELU），$W^{(l)}$ 和 $b^{(l)}$ 为第 $l$ 层参数。

## 核心要点
1. 万能近似定理：足够宽的单隐层 MLP 可近似任意连续函数
2. 在 VLA 策略中常作为轻量级动作解码头
3. 相比注意力机制参数效率更高，适合资源受限场景

## 代表工作
- [[FineVLA]]: StarVLA-OFT 使用轻量级 MLP 回归头解码动作块
- [[OpenVLA-OFT]]: 同样采用 MLP 动作头设计

## 相关概念
- [[Autoregressive Policy]]
- [[Flow Matching]]
