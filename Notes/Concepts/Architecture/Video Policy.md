---
type: concept
aliases: [Video Policy, 视频策略]
---

# Video Policy

## 定义

Video Policy 是指结构上继承了视频生成骨干（如视频扩散 Transformer）并将其预训练时空表示直接映射到动作空间的机器人策略，形式上为 $p(a|o)$。

## 核心要点

1. **结构遗产（Structural Heritage）**：与 [[World Action Model|WAM]] 的区别之一——Video Policy 在概念上与视频生成骨干绑定（如视频扩散 Transformer），WAM 是骨干无关的
2. **无预测承诺（No Predictive Commitment）**：Video Policy 可仅继承预训练时空表示将观测直接映射到动作，无需显式未来状态预测目标；WAM 则要求下一状态 $o'$ 的合成是模型推理的显式组成部分
3. **与 WAM 的兼容性**：若 Video Policy 增加了世界建模监督目标（预测未来状态），则可被纳入 WAM 框架

## 与 WAM 的区分

| 维度 | Video Policy | WAM |
|------|-------------|-----|
| 骨干依赖 | 依赖视频生成骨干 | 骨干无关 |
| 预测承诺 | 可无（直接 $p(a\|o)$）| 必须有显式 $o'$ 预测 |
| 数据利用 | 主要有动作标注数据 | 可利用无标注视频 |

## 代表工作

- [[WAMSurvey]]: 与 WAM 的概念区分

## 相关概念

- [[World Action Model]]
- [[VLA]]
- [[DiT]]
- [[Diffusion Model]]
