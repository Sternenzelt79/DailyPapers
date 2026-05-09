---
concept: OpenFAST
aliases: [Open FAST Tokenizer]
category: VLA
tags: [action-tokenizer, frequency-domain, bpe]
created: 2026-05-09
---

# OpenFAST

## 定义

OpenFAST 是 [[MolmoAct2]] 提出的开源动作分词器，把 1 秒连续动作轨迹压缩为 2048 词表的离散 token，便于在 [[LLM Backbone|LLM]] 中预训练。

## 三阶段流水线

1. **频域变换**: 对 1 秒轨迹做 DCT 类变换。
2. **量化**: 1–99 percentile 统计 normalize → 系数量化。
3. **[[Byte-Pair Encoding|BPE]] 压缩**: 输出 2048 词表 token；夹爪指令单独通道。

## 关键设计

- 动作维度统一 padding 到 32（兼容多本体）
- 训练混合：BimanualYAM 30% / SO-100/101 30% / DROID 30% / Fractal+BC-Z+Bridge 10%
- 与 [[Flow Matching]] 互补：预训练用 token、推理用 flow

## 与 [[FAST]] 区别

| 维度 | FAST | OpenFAST |
|------|------|----------|
| 数据规模 | 私有 | 开源大规模 |
| 词表 | 较大 | 2048 |
| 多本体 | 不统一 | 32 维 padding 统一 |

## 代表工作

- [[MolmoAct2]] (2026): 提出并部署。

## 相关概念

- [[FAST]]
- [[Action Tokenizer]]
- [[Frequency-Domain Transform]]
