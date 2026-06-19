---
type: concept
aliases: [Fully Sharded Data Parallel, 全分片数据并行]
---

# FSDP（Fully Sharded Data Parallel）

## 定义

PyTorch 提供的分布式训练策略，将模型参数、梯度和优化器状态跨所有 GPU 完全分片存储，相比 DDP 大幅减少每卡显存占用，支持更大规模模型的分布式训练。

## 数学形式

对模型参数 $\theta$ 按 GPU 数量 $N$ 均匀分片：

$$
\theta = [\theta_1, \theta_2, \ldots, \theta_N], \quad \theta_i \text{ 存储在 GPU}_i
$$

前向/反向时通过 All-Gather 临时重构完整参数，计算完成后立即释放。

## 核心要点

1. **完全分片**：参数 + 梯度 + 优化器状态全部分片，每卡显存约为 DDP 的 $1/N$
2. **通信开销**：前向和反向各需一次 All-Gather + 一次 Reduce-Scatter
3. **适用场景**：数十亿参数的大模型，单卡无法容纳完整模型时
4. **与 DDP 对比**：DDP 每卡保存完整副本，FSDP 以通信换显存

## 代表工作

- [[Co-VLA]]: 使用 FSDP 在 4 GPU 上训练 7B 参数的 π₀ + SAE

## 相关概念

- [[分布式训练]]
- [[梯度检查点]]
