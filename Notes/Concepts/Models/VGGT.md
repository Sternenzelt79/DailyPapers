---
type: concept
aliases: [VGGT, Visual Geometry Grounded Transformers]
---

# VGGT

## 定义
Visual Geometry Grounded Transformers：一类以 3D 几何重建为核心预训练任务的视觉基础模型，输入多视角图像，输出点云、深度图、相机位姿等几何属性。

## 数学形式
$$\mathbf{G} = f_\theta(\{I_1, \ldots, I_N\})$$

其中 $\mathbf{G}$ 包含每帧的深度图 $D_i$、相机内外参 $\{R_i, t_i\}$ 以及稠密点云 $P$。

## 核心要点
1. **几何优先预训练**：在大规模多视角数据上预训练，获得强几何理解能力
2. **VLA 注入**：[[GeoVLA]] 等工作研究如何把 VGGT 的几何特征注入 VLA，通过 cross-attention fusion 而非 early fusion 效果更好
3. **Camera 数量敏感**：MIT/Amazon 研究发现 camera 数量对 VLA 几何性能影响比 GFM 本身更大

## 代表工作
- Wang et al.《VGGT: Visual Geometry Grounded Transformers》
- [[GeoVLA]]（MIT + Amazon）：研究 VGGT 对 VLA 的影响

## 相关概念
- [[3D Gaussian Splatting]]
- [[SpatialVLA]]
- [[Vision Transformer]]
