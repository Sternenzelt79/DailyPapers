---
type: concept
aliases: [Action-Conditioned Video WM, 动作条件视频世界模型, ACVWM]
---

# Action-Conditioned Video World Model（动作条件视频世界模型）

## 定义
在像素/视频空间直接预测未来帧的动作条件世界模型：给定初始帧（或帧序列）和动作序列，生成对应的未来视频，用于机器人策略评估、数据增强或规划。

## 数学形式

$$
\hat{V}_{1:T} = f_\theta(I_0, a_{1:T}, c)
$$

其中 $I_0$ 为初始帧，$a_{1:T}$ 为动作序列，$c$ 为可选的文本/骨骼等条件，$\hat{V}$ 为预测视频。

## 核心要点
1. **条件类型多样**: 文本、隐变量动作、几何渲染（骨骼/网格/点云）、夹爪图像等
2. **与 Action-Conditioned WM 的区别**: 在像素空间而非潜空间显式生成视频帧，可直接用于视觉策略评估
3. **策略评估代理**: 通过在生成视频上打分来估计策略真实性能，替代昂贵的实体测试

## 代表工作
- [[OSCAR]]: 2D 骨骼条件，跨形态统一，Spearman ρ=0.750 策略评估相关性
- [[Genie Envisioner]]: 夹爪渲染条件，2B 参数
- [[Kinema4D]]: 点图条件，14B 参数
- [[IRASim]]: 隐变量动作条件，早期代表工作

## 相关概念
- [[Action-Conditioned World Model]]: 上层概念（含潜空间方法）
- [[Diffusion Transformer]]: 常用生成架构
- [[Rectified Flow]]: 常用训练目标
