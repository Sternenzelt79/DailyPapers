---
type: concept
aliases: [ARX X5, ARX双臂机器人]
---

# ARX X5

## 定义

ARX X5 是一款用于科研和开发的双臂协作机器人平台，支持双臂同步操作，常用于接触丰富的灵巧操作任务研究。

## 核心要点

1. **双臂结构**: 左右两条机械臂，支持双臂协调操作（bimanual manipulation）
2. **末端执行器**: 可配置不同末端工具，支持夹持、接触丰富操作
3. **传感器集成**: 通常配合 Intel RealSense 系列摄像头使用（顶部 D455 + 腕部 D405）
4. **控制接口**: 支持末端执行器坐标系控制，动作空间兼容 VLA 框架

## 代表工作

- [[HABC]]: 在 ARX X5 上验证了双臂接触丰富任务的在线 RL 微调（Pencil Pouch、Paper Bag、Snack Bag）

## 相关概念

- [[Intel RealSense]]
- [[Bimanual Manipulation]]
- [[π0.5]]
