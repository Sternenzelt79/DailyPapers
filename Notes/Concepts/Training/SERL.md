---
type: concept
aliases: [SERL, Sample Efficient Real-world RL]
---

# SERL

## 定义
Sample Efficient Real-world Reinforcement Learning：面向真实机器人的高样本效率 RL 框架，结合 off-policy 学习（RLPD）与 human demonstrations，在数百次交互内学会真实操作任务。

## 数学形式
$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t r_t\right], \quad \text{with replay buffer } \mathcal{D} = \mathcal{D}_\text{demo} \cup \mathcal{D}_\text{online}$$

同时从 human demo buffer 和 online experience buffer 中采样，demo:online = 50%:50%。

## 核心要点
1. **Off-policy 基础**：基于 RLPD（高 UTD replay ratio），大幅提升样本效率
2. **Human Demo 混合**：初始化阶段直接用 human demo 填充 replay buffer，加速起步
3. **真机部署**：设计为在真实机器人上运行，有完整的传感器延迟和异步控制处理
4. **VLA 扩展**：[[EXPO-FT]] 在 SERL 基础上加入 VLA 预训练先验，进一步提升效率

## 代表工作
- Luo et al.《SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning》
- [[EXPO-FT]]：在 VLA 上扩展 SERL 框架

## 相关概念
- [[Reinforcement Learning]]
- [[Actor-Critic]]
- [[EXPO-FT]]
