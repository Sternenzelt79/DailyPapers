---
type: concept
aliases: [FineVLA Tool, 细粒度标注流水线]
---

# FineVLA-Tool

## 定义
FineVLA-Tool 是 FineVLA 框架中的数据构建工具，通过 4 阶段自动化流水线（格式转换 → 规范化 → DTW 聚类 → 多维标注）将异构机器人演示数据转化为 10 维细粒度指令对齐数据。

## 核心要点
1. 4 阶段流水线处理 972K → 47K 轨迹（19.6:1 压缩比）
2. 基于 DTW 的轨迹聚类保留操作策略多样性
3. Qwen3.5-Plus 自动标注 + 人工验证保障质量
4. 指令密度从 9.3 词/轨迹提升至 96.8 词/轨迹（10.4×）

## 数学形式
DTW 距离驱动的轨迹聚类：

$$
d_{DTW}(\tau_i, \tau_j) = \min_{\text{path}} \sum_{(k,l) \in \text{path}} \|a_k^i - a_l^j\|
$$

## 代表工作
- [[FineVLA]]: FineVLA-Tool 的宿主框架

## 相关概念
- [[DTW]]
- [[LeRobot]]
- [[Qwen3.5-VL]]
