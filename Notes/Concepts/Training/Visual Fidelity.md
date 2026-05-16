---
type: concept
aliases: [视觉保真度, Visual Quality, 视觉质量评估]
---

# Visual Fidelity（视觉保真度）

## 定义
视觉保真度是评估世界模型或视频生成模型输出质量的基础维度，衡量生成视频在像素精度、感知相似性、语义一致性和分布真实性等方面与真实观测的匹配程度。

## 数学形式

**PSNR**（像素级重建精度）:
$$\text{PSNR}(x, y) = 10 \log \frac{\text{MAX}^2}{\text{MSE}(x, y)}$$

**SSIM**（结构相似性）:
$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**FVD**（视频分布距离）:
$$\text{FVD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

## 核心要点
1. **三层评估**: 低级（PSNR/SSIM）+ 感知语义（LPIPS/DreamSim/DINO）+ 分布级（FVD）
2. **局限性**: 视觉保真度高不代表物理合理，也不代表动作可执行性——这是 WAM 评测的核心挑战
3. **与物理常识/动作可信度的关系**: WAM 的三维评测框架中，视觉保真度是最基础但也是最不充分的维度

## 代表工作
- [[WAM-Survey]]: Sec 6.1.1 详细综述视觉保真度评估指标

## 相关概念
- [[WAM]]: WAM 的世界建模评测需要超越视觉保真度
- [[World Model]]: 世界模型质量的评估维度之一
