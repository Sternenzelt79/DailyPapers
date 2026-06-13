---
type: concept
aliases: [LabUtopia benchmark, 实验室操作基准]
---

# LabUtopia

## 定义

LabVLA 论文提出的实验室机器人操作基准，覆盖六类科学实验室核心操作，支持同分布（ID）和异分布（OOD）双评测，每类任务 120 个 episode。

## 核心要点

1. **六类任务系列**: 取物（Pick Up）、按压按钮（Press Button）、开门（Open Door）、倒液（Pour Liquid）、加热烧杯（Heat Beaker）、转移烧杯（Transport Beaker）
2. **双评测设定**: ID（同分布）和 OOD（异分布），评估模型泛化能力
3. **任务难度层级**: 按压按钮（最易，100%）> 加热/转移 > 取物/开门 > 倒液（最难，34.2% OOD）
4. **评测标准**: LabVLA 平均 71.1% ID / 70.0% OOD，优于 π₀（63.3%/63.2%）

## 代表工作

- [[LabVLA]]: 提出方，报告六类任务完整成功率

## 相关概念

- [[VLA]]
- [[Domain Randomization]]
