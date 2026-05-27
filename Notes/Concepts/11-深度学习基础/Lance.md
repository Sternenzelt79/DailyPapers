---
type: concept
aliases: [Lance format, Lance存储, Lance columnar format]
---

# Lance

## 定义
Lance 是一种为机器学习工作负载优化的列式存储格式，原生支持随机访问、向量索引和高效流式读取，尤其擅长视频/图像序列数据的高吞吐加载。

## 核心要点
1. 列式存储使批量特征读取比行式格式（HDF5）快约 3.4 倍
2. 支持本地和 S3 远程存储，S3 流式吞吐（3,184 样本/秒）远优于 HDF5-S3（757 样本/秒）
3. 被 stable-worldmodel 平台选为默认数据后端
4. 提供 MP4、HDF5、LeRobot 一键转换工具

## 性能对比（Push-T，样本/秒）

| 格式 | 本地 | S3 |
|------|------|----|
| HDF5 | 1,416 | 9 (无缓存) / 757 (缓存) |
| **Lance** | **4,815** | **3,184** |
| Video | 1,331 | — |

## 代表工作
- [[stable-worldmodel]]: 核心数据层，解决世界模型训练 I/O 瓶颈（2026）

## 相关概念
- [[stable-worldmodel]]
- [[世界模型]]
