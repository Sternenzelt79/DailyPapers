---
concept: OpenFAST
category: Training
tags: [tokenizer, action, vla, open-source]
created: 2026-05-09
---

# OpenFAST

## 定义

开放权重、开放数据的机器人动作 tokenizer。把 1 秒、32 维连续动作序列通过"频域变换 → 量化 → [[BPE|字节对编码]]"压缩为 2048 词表的离散 token。是 [[FAST]] tokenizer 的开源化版本。

## 核心要点

1. **频域压缩**：先做 [[DCT|离散余弦变换]]，把时间冗余转成低频系数；
2. **1-99 百分位归一化**：抗离群值；
3. **量化 + BPE**：与文本 tokenizer 同构，方便 VLM 共享词表；
4. **5 种本体训练**：YAM / SO-100/101 / Franka(DROID) / Google Robot(Fractal+BC-Z) / WidowX(Bridge)，各 1M 序列；
5. **2 种动作表示**：绝对关节 + delta 末端，覆盖主流接口。

## 训练数据混合

| 数据集 | 比例 | 机器人 | 动作 |
|--------|------|--------|------|
| MolmoAct2-BimanualYAM | 30% | YAM | 绝对关节 |
| MolmoAct2-SO100/101 | 30% | SO-100/101 | 绝对关节 |
| MolmoAct2-DROID | 30% | Franka | 绝对关节 |
| Fractal | 3.33% | Google | delta 末端 |
| BC-Z | 3.33% | Google | delta 末端 |
| Bridge | 3.33% | WidowX | delta 末端 |

## 代表工作

- [[MolmoAct2]]: 首发并开源该 tokenizer。

## 相关概念

- [[FAST]]
- [[Action Chunking]]
- [[BPE]]
- [[DCT]]
