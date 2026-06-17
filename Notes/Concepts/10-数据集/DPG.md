---
type: concept
aliases: [Dense Prompt Graph, DPG-Bench]
---

# DPG

## 定义
Dense Prompt Graph Benchmark，用于评估文本到图像模型对复杂、多属性文本描述的组合生成能力。

## 核心要点
1. 使用图结构表示 prompt 中各实体之间的属性和关系
2. 覆盖属性绑定、对象计数、空间关系等多维度
3. 比 COCO 等传统 caption benchmark 更难，更贴近真实使用场景

## 代表工作
- FLUX, i1 等 T2I 模型均在此评测

## 相关概念
- [[GenEval]]
- [[FLUX]]
