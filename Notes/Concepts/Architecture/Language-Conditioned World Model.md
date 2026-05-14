---
type: concept
aliases: [Language-Conditioned WM, 语言条件世界模型, LCWM]
---

# Language-Conditioned World Model（语言条件世界模型）

## 定义
语言条件世界模型使用语言作为高层次、更抽象的条件形式来预测未来状态，而非使用低级控制信号。语言提供关于期望场景、事件或演化过程的语义引导。

## 数学形式

$$
P(o' \mid o, l)
$$

其中 $l$ 为语言条件（文本描述、指令等），$o$ 为当前观测，$o'$ 为预测的未来状态。

## 核心要点
1. **演进路径**: GAN 视频生成 → 扩散式视频基础模型（U-Net → DiT）→ 大规模视频基础模型（Sora、Wan、Kling 等）
2. **两类应用**: 数据合成（生成多样化训练轨迹）+ 任务规划（基于语言指令预测任务执行视频）
3. **与 WAM 的关系**: 语言条件 WM 缺乏动作耦合，WAM 将两者统一为 $p(o', a \mid o, l)$

## 代表工作
- [[WAM-Survey]]: Sec 3.2.2 对该范式的详细综述
- Wan、Sora 2、Veo 3：代表性大规模语言条件视频基础模型

## 相关概念
- [[World Model]]: 上层概念
- [[Action-Conditioned World Model]]: 以低级动作为条件的互补范式
- [[WAM]]: 将语言条件预测与动作生成统一的更高级范式
- [[DiT]]: 现代语言条件视频模型的主流骨干架构
