---
type: concept
aliases: [实例分割掩码, Instance Segmentation Mask, Instance Mask]
---

# Instance Segmentation Mask

## 定义
为图像中每个对象实例生成的像素级二值掩码 $M \in \{0,1\}^{H \times W}$，区分同类不同实例（与语义分割的逐类别标注不同）。

## 核心要点
1. **像素级精度**: 每个像素属于某一具体实例（非类别），提供比边界框更精确的空间定位
2. **身份识别**: 在机器人操作场景中，Mask 明确指定"哪个对象"，解决语言歧义（"pick the red cup" 存在多个红杯时）
3. **与跟踪结合**: SAM-3 等模型支持从首帧单次点击出发，自动跟踪生成视频序列中的逐帧 Mask

## 数学形式
$$M \in \{0, 1\}^{H \times W}, \quad M_{ij} = 1 \text{ iff pixel } (i,j) \text{ belongs to target instance}$$

## 代表工作
- [[MaskWAM]]: 将实例 Mask 作为 WAM 的输入提示与预测目标，实现语言歧义场景下的精确对象定位
- [[SAM-3]]: 用于从用户点击提示生成并跟踪目标对象 Mask

## 相关概念
- [[SAM-3]]
- [[Future Mask Prediction]]
- [[World Action Model]]
