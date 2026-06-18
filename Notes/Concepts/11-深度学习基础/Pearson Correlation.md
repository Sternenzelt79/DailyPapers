---
type: concept
aliases: [Pearson 相关系数, 皮尔逊相关系数, r, PCC]
---

# Pearson Correlation

## 定义

衡量两个连续变量之间线性相关程度的统计量，取值范围 $[-1, 1]$；在机器人学中常用于评估仿真/世界模型预测性能与真实世界性能之间的一致性。

## 数学形式

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

其中 $x_i, y_i$ 为配对观测值，$\bar{x}, \bar{y}$ 为各自均值。

## 核心要点

1. **$r=1$**：完全正相关；**$r=-1$**：完全负相关；**$r=0$**：线性无关
2. **世界模型评估**：若世界模型中策略的成功率排名与真实世界一致，则 Pearson 系数接近 1，说明世界模型可作为可靠的策略筛选工具
3. **[[Mem-World]] 结果**：Mem-World 达到 0.97 vs [[Ctrl-World]] 的 0.85，提升 14.5%，说明记忆增强后世界模型对真实策略性能的预测更可靠

## 代表工作

- [[Mem-World]]: 用 Pearson 相关系数量化世界模型预测成功率与真实成功率的一致性

## 相关概念

- [[Action-Conditioned Video Generation]]
- [[PSNR]]
- [[SSIM]]
