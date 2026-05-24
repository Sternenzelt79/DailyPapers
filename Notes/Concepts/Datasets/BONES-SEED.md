---
type: concept
aliases: [BONES-SEED, BONES SEED dataset]
---

# BONES-SEED

## 定义
由 NVIDIA 随 SONIC 论文公开发布的大规模人体动作捕捉数据集，包含 14.2 万标注序列，覆盖 33 类运动风格。

## 核心要点
1. **规模**: 142,220 标注序列，来自 522 名演员，700 小时（训练集 611 小时，100M+ 帧@50 Hz）
2. **多样性**: 33 大类，8,447 独特子类，涵盖步行/手势/舞蹈/格斗/道具操作/工具使用/受伤步态/角色扮演等
3. **公开获取**: 发布于 HuggingFace（bones-studio/seed）
4. **评测分割**: 训练集 317,189 clips / Test-Content 6,998 clips（无内容重叠）/ Test-Repetition 6,306 clips

## 代表工作
- [[SONIC]]: 使用 BONES-SEED 训练通用人形控制策略，验证 scaling law

## 相关概念
- [[运动捕捉]]
- [[运动跟踪]]
- [[SMPL]]
