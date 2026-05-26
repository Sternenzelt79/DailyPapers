---
type: concept
aliases: [RealEstate10K Dataset, RE10K]
---

# RealEstate10K

## 定义
大规模真实世界房产视频数据集，包含带有 3D 相机位姿标注的室内场景视频（256×256），常用于评估新视角合成和视频生成模型的场景一致性。

## 核心要点
1. 室内真实场景，相机位姿已知，适合评估几何一致性
2. 分辨率 256×256，视频长度覆盖数十至数百帧
3. 被广泛用于 novel view synthesis 和 video world model 的泛化测试
4. 支持场景"回溯"（reverse trajectory）评估历史帧召回能力

## 代表工作
- [[CoME]]: 在 RealEstate10K 上验证 LTM 的情节记忆效果，LPIPS 从 0.405 降至 0.359
