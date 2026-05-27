---
type: concept
aliases: [PGD, Projected Gradient Descent Attack, 投影梯度下降攻击]
---

# PGD Attack

## 定义
Projected Gradient Descent（投影梯度下降）对抗攻击：在 $\ell_\infty$ 约束的扰动球内，对输入做多步梯度上升来最大化模型损失，是最强的 white-box 对抗攻击之一。

## 数学形式
$$\mathbf{x}^{t+1} = \Pi_{\mathcal{B}(\mathbf{x}_0, \epsilon)}\left(\mathbf{x}^t + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(\theta, \mathbf{x}^t, y))\right)$$

其中 $\Pi$ 是投影算子，$\epsilon$ 是扰动半径（如 $16/255$），$\alpha$ 是步长。

## 核心要点
1. **多步迭代**：比 FGSM（单步）更强，$k$ 步 PGD 通常被称为 $k$-PGD
2. **VLA 脆弱性**：$16/255$ PGD 把 [[OpenVLA]] 在 LIBERO 的成功率从 95% 打到 <5%
3. **能力-鲁棒性 tradeoff**：[[Cap-Rob-Bound]] 从信息论角度证明 VLA 无法同时免费获得高能力和高鲁棒性

## 代表工作
- Madry et al.《Towards Deep Learning Models Resistant to Adversarial Attacks》
- [[Cap-Rob-Bound]]：用 PGD 验证 VLA 的能力-鲁棒性 tradeoff

## 相关概念
- [[OpenVLA]]
- [[Cap-Rob-Bound]]
