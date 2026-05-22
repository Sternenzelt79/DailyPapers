---
type: concept
aliases: [Hierarchical Data Format 5, h5py]
---

# HDF5

## 定义

HDF5（Hierarchical Data Format version 5）是一种用于存储和管理大规模科学数据的开放格式，支持层级组织、压缩和随机访问。

## 核心要点

1. 支持任意维度的数组数据，常用于机器学习数据集（图像、轨迹）存储
2. 原生支持 GZIP/LZF 压缩，节省磁盘空间
3. 随机访问性能中等：本地吞吐量约 1,416 样本/秒（vs Lance 的 4,815 样本/秒）
4. Python 中通过 `h5py` 库操作，`datasets` 库也有原生支持

## 代表工作

- [[StableWorldModel]]: swm 提供 HDF5 → Lance 格式转换工具，解决 HDF5 在大规模训练中的数据瓶颈

## 相关概念

- [[Lance]]
- [[LeRobot]]
