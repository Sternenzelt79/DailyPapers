---
type: concept
aliases: [可靠性感知训练, Reliability-Aware Loss, 可靠性加权损失]
---

# Reliability-Aware Training（可靠性感知训练）

## 定义

在混合质量数据训练中，根据数据来源的可靠性（精度、噪声水平）对各样本/通道/时间步分配不同的损失权重，防止低质量数据（如视觉重建的伪动作）污染策略学习。

## 核心要点

1. **问题来源**：机器人传感器数据精度高（位置误差 < 1mm），而视觉重建的[[伪动作]]存在跟踪抖动和估计偏差
2. **层次化设计**：通常从通道级（位置 vs 旋转）、数据集级（机器人 vs 人类）、步骤级（当前帧是否存在跳变）三个层次分解可靠性
3. **稳健损失函数**：常与 Huber Loss 配合使用，进一步抑制异常值影响

## 数学形式

层次化可靠性权重分解：

$$
W_{t,j} = \rho_j \cdot w_{data}(d, h(j)) \cdot w_{step}(t, h(j))
$$

步骤级平滑权重（跳变+抖动检测）：

$$
q_{t,h} = \max\!\left(\frac{\Delta p_t^h}{\tau_{jump}}, \frac{\Delta^2 p_t^h}{\tau_{jerk}}\right)
$$

$$
w_{step}(t,h) = \begin{cases} 1, & q_{t,h} \leq 1 \\ \max\{w_{min}, \exp[-\alpha(q_{t,h}-1)]\}, & q_{t,h} > 1 \end{cases}
$$

人类辅助损失：

$$
\mathcal{L}_{haux} = \frac{1}{Z} \sum_{t,j} M_{t,j} W_{t,j} \cdot \text{Huber}_\beta\!\left(\hat{v}_{t,j} - (\tilde{a} - \epsilon)_{t,j}\right)
$$

## 应用场景

1. **人类视频 + 机器人数据联合训练**：人类伪动作位置通道权重=1.0，旋转/夹爪权重=0.001
2. **多质量数据混合训练**：互联网视频 vs 专业采集数据
3. **长尾/噪声标注数据集**：异常帧自动降权

## 代表工作

- [[ACE-Ego-0]]: 消融实验显示去掉可靠性感知损失后成功率下降 −3.6%（贡献最大的单个模块）

## 相关概念

- [[伪动作]]
- [[Flow Matching]]
- [[Egocentric Video]]
