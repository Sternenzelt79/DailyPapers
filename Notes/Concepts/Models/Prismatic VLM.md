---
type: concept
aliases: [Prismatic, PrismaticVLM, prismatic-vlm]
---

# Prismatic VLM

## 定义

一种开源视觉-语言模型（VLM）框架，核心特点是使用**双视觉编码器并行架构**——同时运行 DINOv2（低层视觉细节）和 SigLIP（高层语义理解），通过 SE-bottleneck 压缩融合后作为语言模型的视觉输入。

## 数学形式

$$
p = \text{SE-Bottleneck}(\text{concat}[\text{DINOv2}(I), \text{SigLIP}(I)])
$$

$$
c = \text{LLM}_{\text{EOS}}(p, L)
$$

其中 $p$ 为感知 token，$c$ 为认知 token（EOS 位置输出）。

## 核心要点

1. **双编码器设计**：DINOv2 捕获低层视觉细节（利于精细操作），SigLIP 捕获高层语义（利于语言-视觉对齐）
2. **SE-bottleneck 压缩**：将双编码器特征高效融合并降维，减少传入语言模型的 token 数量
3. **可扩展骨干**：支持多种 LLM（LLaMA-7B、Qwen2.5 等），实验验证架构对骨干无关
4. **开源基础**：训练于 Open-X Embodiment 等大规模具身数据集，广泛用于机器人 VLA 研究

## 代表工作

- [[MemoryVLApp]]: 使用 7B Prismatic（DINOv2 + SigLIP + LLaMA-7B）作为 VLC 编码模块，生成感知 token $p$ 和认知 token $c$
- [[MemoryVLA]]: 前代工作同样基于 Prismatic VLM

## 相关概念

- [[DINOv2]]: Prismatic 的低层视觉编码器
- [[SigLIP]]: Prismatic 的高层语义编码器
- [[LLaMA]]: Prismatic 默认语言模型骨干
- [[VLM]]: 更广泛的视觉语言模型类别
- [[VLA]]: Prismatic 常作为 VLA 系统的感知骨干
