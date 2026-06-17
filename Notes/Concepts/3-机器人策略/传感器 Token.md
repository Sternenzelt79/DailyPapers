---
type: concept
aliases: [Sensor Token, 传感器选择Token, Adaptive Sensor Selection Token]
---

# 传感器 Token（Sensor Token）

## 定义

传感器 Token 是在 VLA 模型词汇表中新增的可学习特殊 token，用于表示当前任务所需的感知模态（如热成像、声学、mmWave 雷达等），由 VLM Backbone 根据任务指令自动预测，实现按需激活传感器。

## 数学形式

VLM 以交叉熵损失监督传感器 token 的分类预测：

$$
\mathcal{L}_{\text{sensor}} = \text{CrossEntropy}(l_s^{\text{pred}},\, l_s^{\text{gt}})
$$

其中 $l_s^{\text{pred}}$ 为 VLM 输出的传感器 token 概率分布，$l_s^{\text{gt}}$ 为真实标注的传感模态。

## 核心要点

1. **按需激活**: 不同任务对应不同传感器 token，模型在推理时只激活所需传感器，避免冗余传感器引入噪声。
2. **显存效率**: 与同时处理所有传感器相比，自适应选择将推理显存从 13.23GB 降至 6.61GB。
3. **可扩展性**: 新增传感器类型只需在词汇表中添加新 token，无需修改 backbone 架构。
4. **MuseVLA 中的实现**: 四类 token —— `<None>`（纯 RGB）、`<Thermal>`（热成像）、`<Acoustic>`（声学）、`<mmWave>`（毫米波雷达）。

## 代表工作

- [[MuseVLA]]: 首次提出将传感器选择参数化为 VLM 词汇表中的可学习 token，与目标描述生成联合训练，实现端到端自适应多传感器 VLA。

## 相关概念

- [[VLA（Vision-Language-Action Model）]]
- [[Vision-Language Model]]
- [[mmWave 雷达]]
- [[热成像相机]]
- [[麦克风阵列]]
