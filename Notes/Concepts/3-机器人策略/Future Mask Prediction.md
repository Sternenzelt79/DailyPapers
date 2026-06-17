---
type: concept
aliases: [未来 Mask 预测, Future Mask Prediction]
---

# Future Mask Prediction

## 定义
在视频预测或世界模型框架中，将未来帧的目标对象实例分割 Mask 作为额外预测目标，提供对象中心的语义监督信号，迫使模型关注任务相关区域而非背景噪声。

## 核心要点
1. **语义监督**: 强制模型在预测未来状态时关注任务相关对象（如被操作物体），抑制背景视觉干扰
2. **联合预测**: 通常与 RGB 帧预测并行，共享噪声调度或解码器，形成 RGB-Mask 联合生成
3. **对象中心表示**: 促进学到对象级的结构化特征，而非逐像素的外观特征

## 数学形式
在联合 Flow Matching 框架中，Future Mask Prediction 对应损失项：

$$\mathcal{L}_{\text{mask}} = \mathbb{E}_{\tau_v}\left[\|v_\theta(z_m^{\tau_v}, \tau_v) - (z_m - \epsilon_m)\|^2\right]$$

## 代表工作
- [[MaskWAM]]: 首次将 Future Mask Prediction 与首帧 Mask 提示统一在 WAM 框架中，证明未来 Mask 预测是视觉提示有效 grounding 的必要条件

## 相关概念
- [[Instance Segmentation Mask]]
- [[World Action Model]]
- [[Video VAE]]
- [[Flow Matching]]
