---
type: concept
aliases: [毫米波雷达, mmWave Radar, Millimeter Wave Radar, 77GHz Radar]
---

# mmWave 雷达（毫米波雷达）

## 定义

毫米波雷达是工作在 30–300 GHz 频段（波长 1–10mm）的无线电传感器，通过发射毫米波并接收反射信号，可穿透遮挡物（箱体、布料等）感知物体位置与速度，在暗光、烟雾等 RGB 相机失效的场景下仍可工作。

## 数学形式

数字波束成形计算空间功率分布（用于生成热图）：

$$
\begin{aligned}
P(\theta, \phi) &= 20 \log_{10} \left\| \sum_k a_k e^{j\varphi_k} e^{-j\Delta_k(\theta,\phi)} \right\| \\
\Delta_k(\theta,\phi) &= \frac{2\pi}{\lambda}\bigl(d_k^x \cos\phi \sin\theta + d_k^y \sin\phi\bigr)
\end{aligned}
$$

其中 $P(\theta,\phi)$ 为方向 $(\theta,\phi)$ 上的功率谱（dB），$a_k, \varphi_k$ 为第 $k$ 根天线的幅度和相位，$\lambda$ 为雷达波长。

## 核心要点

1. **穿透能力**: 毫米波可穿透纸板、布料等非金属遮挡物，适用于隐藏物体检测。
2. **全天候感知**: 不受光照、烟雾、灰尘影响，与 RGB 相机互补。
3. **空间分辨率有限**: 与 RGB 相机相比角分辨率较低，通常需要与 RGB 融合使用。
4. **数字波束成形**: 通过相位阵列天线合成空间方向性，生成二维角度功率热图。
5. **机器人应用**: 隐藏物体检索、穿墙感知、人体姿态估计等。

## 代表工作

- [[MuseVLA]]: 将 mmWave 雷达热图通过单应矩阵对齐并叠加到 RGB 图像，用于辅助隐藏物体检索任务，在遮挡场景下成功率达 87.5%。

## 相关概念

- [[Homography|单应矩阵]]
- [[传感器 Token]]
- [[热成像相机]]
- [[麦克风阵列]]
