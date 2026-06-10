---
type: concept
aliases: [Residual Policy Learning, 残差策略学习, 残差修正策略]
---

# Residual Policy Learning

## 定义

残差策略学习（Residual Policy Learning）是一种将基础策略（base policy）与残差修正网络（residual network）结合的学习范式：基础策略负责全局规划，残差网络学习修正基础策略在特定状态下的偏差，最终输出为两者之和。

## 数学形式

$$
a^{\text{final}} = a^{\text{base}} + \kappa \cdot \Delta a^{\text{res}}
$$

其中 $\kappa \in \{0, 1\}$ 为门控（或连续权重），控制残差是否激活。

## 核心要点

1. **解耦规划与修正**: 基础策略处理全局意图，残差策略专注于细粒度修正，各司其职
2. **数据效率**: 残差网络只需少量错误修正样本即可训练（相较于从头学习策略）
3. **门控机制**: 避免在不需要修正时引入噪声，防止"过度修正"（overcorrection）
4. **高频在线运行**: 残差网络通常比基础策略频率更高，实现实时反应

## 代表工作

- [[FAWAM]]: 力引导残差修正器以 10 Hz 运行，通过力矩跟踪误差触发在线残差修正，仅需 20 次人工干预演示训练

## 相关概念

- [[DAgger]]
- [[Force/Torque Sensing]]
- [[Action Chunking]]
