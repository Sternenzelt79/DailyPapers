---
type: concept
aliases: [ManiSkill, ManiSkill2, ManiSkill3]
---

# ManiSkill

## 定义
Haosu Tang 等人（UC San Diego）开发的开源机器人操作仿真 benchmark 和平台，基于 SAPIEN 物理引擎，提供大量标准操作任务和统一评测接口。

## 数学形式
$$s_{t+1} = f(s_t, a_t), \quad r_t = R(s_t, a_t, s_{t+1})$$

（SAPIEN 物理引擎的状态转移与奖励定义）

## 核心要点
1. 基于 SAPIEN 物理引擎，支持刚体、软体和关节体仿真
2. 提供标准化的 gym 接口，方便 RL 和 IL 算法接入
3. ManiSkill2 引入多样化操作任务（抓取、移动操作、多步骤任务）
4. ManiSkill3 支持 GPU 并行仿真，大幅提升采样效率
5. 常用于 sim-to-real 迁移和操作策略评测

## 代表工作
- [[ReSYNC]]: 用 ManiSkill 测试从失败中学习技能
- [[WAM-RL]]: world model + RL 在 ManiSkill 上评测

## 相关概念
- [[MuJoCo]]
- [[IsaacLab]]
- [[RoboSuite]]
