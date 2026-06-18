---
type: concept
aliases: [NMS, 非极大值抑制]
---

# Non-Maximum Suppression

## 定义

一种后处理算法，在有大量候选检测/采样结果中，通过抑制非局部极大值来去除冗余、保留代表性结果；广泛用于目标检测和点云采样。

## 数学形式

对于候选集合 $\mathcal{S}$ 中按得分降序排列的元素，NMS 迭代地：
1. 选出当前最高分元素 $s^*$，加入保留集
2. 移除与 $s^*$ 重叠度（IoU 或距离）超过阈值 $\tau$ 的所有候选

$$
\mathcal{S} \leftarrow \mathcal{S} \setminus \{s \in \mathcal{S} : \text{overlap}(s, s^*) > \tau\}
$$

## 核心要点

1. **目标检测中**：以 IoU 为重叠度量，去除同一目标的冗余边界框
2. **点云/surfel 采样中**：以 3D 欧氏距离为度量，确保采样点在空间上均匀分布
3. **Mem-World 中的应用**：对打分后的 surfel 执行 NMS，防止反复访问同一空间区域的 surfel 被重复选入历史帧检索集，保证检索到的历史帧在场景空间中的多样性

## 代表工作

- [[Mem-World]]: 在 surfel 几何感知检索中使用 NMS 去除冗余采样

## 相关概念

- [[Surfel Map]]
- [[Action-Conditioned Video Generation]]
