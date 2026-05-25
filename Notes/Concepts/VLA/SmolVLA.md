---
type: concept
aliases: [SmolVLA]
---

# SmolVLA

## 定义
HuggingFace 推出的轻量级 Vision-Language-Action 模型，以较小的参数量（约 2.5B）在机器人操作任务上取得有竞争力的性能，设计目标是降低 VLA 的计算门槛和部署成本。

## 核心要点
1. 参数量相比 OpenVLA 等大型 VLA 更小，便于边缘部署
2. 基于 HuggingFace 生态，使用 LeRobot 框架训练和评估
3. 用 action chunking 方式生成动作序列
4. 常被用作 few-shot adaptation 研究的基础 VLA 模型

## 代表工作
- [[VGAS]]: 以 SmolVLA 为基础 VLA，在 action chunk 上叠加 value-guided 选择实现 few-shot 适应

## 相关概念
- [[Action Chunking]]
- [[OpenVLA]]
- [[Diffusion Policy]]
