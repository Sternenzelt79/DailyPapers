---
type: concept
aliases: [VBE, Phase-Based Decoder, Behavioral Representation]
---

# BehaviorVLA

## 定义
通过构建通用行为表征（Visual Behavior Encoder, VBE）来解决 VLA 在 distribution shift 下性能下滑的问题；配合 Phase-Based Decoder (PBD) 按任务阶段条件化动作生成。

## 核心要点
1. VBE 提取与具体任务阶段无关的行为 embedding
2. PBD 根据当前阶段条件化动作解码
3. 测试在 CALVIN / LIBERO / RoboTwin + 真实机器人

## 代表工作
- [[BehaviorVLA]]: Bing Hu et al., 2026 (arXiv 2605.22671)

## 相关概念
- [[OpenVLA]]
- [[Diffusion Policy]]
