---
type: concept
aliases: [GelSight, Gelsight]
---

# GelSight

## 定义
MIT 研发的光学触觉传感器，通过内置摄像头拍摄凝胶与物体接触时的变形图像，提供高分辨率接触几何信息（法向力分布、接触形状），是机器人精细操作中最广泛使用的触觉传感器之一。

## 数学形式
$$\mathbf{D} = f(I_{\text{tactile}})$$

从触觉图像 $I_{\text{tactile}}$ 估计接触深度图 $\mathbf{D}$，进而计算 3D 接触几何和法向力分布。

## 核心要点
1. 基于光学反射原理：LED 照明 + 透明凝胶 + 内置摄像头
2. 提供稠密 3D 接触几何（相比点式/阵列式传感器信息量大得多）
3. 不需要逐个触点标定，空间分辨率由摄像头分辨率决定
4. 在精细操作（fragile objects、精密组装）中提供关键力感知

## 代表工作
- [[TactileReflex]]: 使用 GelSight 提供触觉反馈，实现脆性容器夹持力自适应控制

## 相关概念
- [[GelSlim]]
- [[Bimanual Manipulation]]
