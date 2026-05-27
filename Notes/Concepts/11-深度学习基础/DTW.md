---
type: concept
aliases: [DTW, Dynamic Time Warping, 动态时间规整]
---

# DTW (Dynamic Time Warping)

## 定义
DTW 是一种计算两段时间序列之间相似度的算法，通过允许时间轴上的弹性对齐（非线性规整），解决序列长度不同或速度不一致的匹配问题。

## 数学形式
$$\text{DTW}(A, B) = \min_{\text{warp path}} \sum_{(i,j) \in \text{path}} d(a_i, b_j)$$

其中 warping path 满足单调性和连续性约束。

## 核心要点
1. 与欧氏距离不同，DTW 允许"时间弯曲"——序列的不同部分可以在时间上延展或压缩
2. 通过动态规划求解，复杂度 $O(nm)$（$n, m$ 为两序列长度）
3. 在机器人学中用于对齐不同速度执行的同类动作轨迹

## 在 FineVLA 中的应用
FineVLA 用 DTW 将专家演示轨迹与细粒度动作描述进行时序对齐，实现"在第 t 帧，机器人正在做 X"这类时序绑定标注。

## 代表工作
- [[FineVLA]]: 利用 DTW 对齐机器人轨迹与细粒度语言指令

## 相关概念
- [[Action Chunking]]
- [[Autoregressive Policy]]
