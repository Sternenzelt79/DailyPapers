---
type: concept
aliases: [曝光偏差, exposure bias, train-test mismatch]
---

# Exposure Bias

## 定义

在序列生成任务中，训练时模型以真实（ground-truth）序列为参考（teacher forcing），而推理时模型以自身预测结果为参考，导致训练与测试分布不匹配的系统性问题。

## 数学形式

**训练时**（teacher forcing）：

$$
p_\theta(y_t \mid y_1^*, y_2^*, \ldots, y_{t-1}^*)
$$

**推理时**（autoregressive）：

$$
p_\theta(y_t \mid \hat{y}_1, \hat{y}_2, \ldots, \hat{y}_{t-1})
$$

其中 $y_t^*$ 为真实 token，$\hat{y}_t$ 为模型预测 token。随着时间步增长，预测误差积累，分布偏差越来越大。

## 核心要点

1. 最早在语言模型（RNN seq2seq）中发现，后来在视频生成、机器人策略等领域同样存在
2. 解决方案包括：Scheduled Sampling（逐渐用预测替换真实）、DAGGER、Self-Forcing
3. 在视频 world model 长时生成中尤为严重，因为每一帧的误差都会影响后续帧
4. [[iMaC]] 通过 Training-time Rollout（one-step clean estimate 作为参考帧）缓解此问题

## 代表工作

- [[iMaC]]: 提出 Training-time Rollout 策略减少 exposure bias
- Self-Forcing (Huang et al. 2026): 专门针对视频生成中 train-test mismatch 的方法

## 相关概念

- [[Flow Matching]]
- [[Autoregressive Policy]]
- [[世界模型]]
