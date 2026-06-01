---
type: concept
aliases: [WorldPlay, HY-WorldPlay]
---

# WorldPlay

## 定义
Tencent 发布的基于 HunyuanVideo 的交互式视频世界模型，支持用户通过相机运动控制生成连续视频帧，可用于具身 AI 训练环境和游戏仿真。

## 核心要点
1. 以 HunyuanVideo 为基础模型，加入相机控制信号实现交互式生成
2. AR（Autoregressive）推理模式，逐帧生成响应用户动作
3. 常作为 WM 推理加速（如 Light Interaction）的测试床
4. 与 Matrix-Game 并列为主流交互式 video WM benchmark

## 代表工作
- [[Light-Interaction]]: 在 HY-WorldPlay 上验证无训练推理加速方案
- [[minWM]]: 提供与 WorldPlay 同类系统的开源替代实现；同时利用 WorldPlay 生成 GT 相机轨迹训练数据

## 相关概念
- [[World Model]]
- [[DMD]]
- [[扩散世界模型]]
