---
type: concept
aliases: [动作引导, 轨迹先验注入]
---

# Action Steering

## 定义
将 [[World-Action Model|WAM]] 预期的动作轨迹编码为单个 prefix token，插入基于流匹配的 VLA 动作生成器，为运动规划提供轨迹级方向先验。

## 数学形式

$$
s^w_t = f_{\text{act}}(\text{Align}_K(\tilde{A}^w_t))
$$

生成器输入序列：$[u_t;\ s^w_t;\ Q_t;\ X_{\tau,t}]$

其中 $\tilde{A}^w_t$ 为 WAM 预期轨迹，$\text{Align}_K$ 为重采样对齐操作，$f_{\text{act}}$ 为动作编码器，$s^w_t$ 为单个先验 token。

## 核心要点
1. 单 token 设计：保留动作生成器的决策自由度，避免过度绑定到有噪 WAM 信号
2. 轨迹对齐：将 WAM 轨迹重采样到 VLA 的 action horizon $K$
3. 优于逐步 token：单 token 84.7% vs 逐步 token 83.6%（LIBERO-Plus）

## 代表工作
- [[WorldPilot]]: 提出 Action Steering，在 LIBERO-Plus 单独带来 +2.6 点提升，与 [[Latent Steering]] 组合取得 +4.2 点

## 相关概念
- [[Latent Steering]]
- [[World-Action Model]]
- [[Flow Matching]]
- [[Action Chunking]]
