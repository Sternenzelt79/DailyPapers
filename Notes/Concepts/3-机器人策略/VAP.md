---
type: concept
aliases: [Visual Attentive Prompting, 视觉注意力提示]
---

# VAP

## 定义
Visual Attentive Prompting，一种免训练的感知适配器，通过用少量参考图作为非参数视觉记忆，在场景中定位个性化目标实例并注入 VLA，实现实例级个性化操作。

## 核心要点
1. Training-free：不修改 VLA 任何参数
2. 用 open-vocabulary 检测（[[DINO]]）+ embedding 匹配做实例定位
3. 将定位结果转为视觉 prompt（高亮目标）+ 指令重写注入 frozen VLA
4. 解决 VLA 对"我的杯子"此类个性化指令无法区分实例的问题

## 代表工作
- [[VAP]] (2512.20014): Bring My Cup! 论文

## 相关概念
- [[VLA]]
- [[PIVOT]]
- [[TraceVLA]]
