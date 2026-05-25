---
type: concept
aliases: [FLUX Kontext, FLUX.1-Kontext, FLUX.1-Kontext-dev, Kontext]
---

# FLUX-Kontext

## 定义
FLUX-Kontext：Black Forest Labs 推出的基于 [[FLUX]] 架构的 in-context 图像编辑模型，通过 Flow Matching 实现在潜空间中对输入图像进行指令引导的高保真编辑，能在保留源场景结构的同时按文本指令修改目标区域。

## 数学形式
$$\hat{x} = E_\theta(x_{src}, l)$$

其中 $x_{src}$ 为源图像，$l$ 为编辑指令，$\hat{x}$ 为编辑后图像。

## 核心要点
1. **In-context 编辑**: 以源图像为上下文条件，保留背景、物体身份和场景布局，仅修改与指令相关的区域
2. **架构基础**: 继承 [[FLUX]] 的 MM-DiT 双流 Transformer + [[Flow Matching]] 训练范式
3. **训练数据**: 在配对的图像变换数据（人工标注编辑、合成指令-编辑对、视频帧对）上训练
4. **机器人应用**: 在 SWEET 中作为稀疏视觉规划器，通过 [[LoRA]] rank=32 微调用于机器人操作关键帧预测

## 代表工作
- [[SWEET]]: 将 FLUX-Kontext 微调用于机器人操作稀疏关键帧预测，比视频生成快 40x 以上

## 相关概念
- [[FLUX]]: 基础生成模型
- [[Flow Matching]]: 训练范式
- [[LoRA]]: 常用的轻量微调方式
- [[InstructPix2Pix]]: 早期图像编辑方法，FLUX-Kontext 的前驱
