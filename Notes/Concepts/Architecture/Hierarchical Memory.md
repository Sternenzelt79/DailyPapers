---
type: concept
category: Architecture
tags: [memory, long-horizon]
created: 2026-05-09
---

# Hierarchical Memory

按时间尺度分层组织的记忆机制（短期 / 中期 / 长期），常通过可学习的压缩 / 摘要 token 把远端历史压缩，近端历史保留细节，以扩展上下文长度。

## 代表工作

- 在 [[RLDX-1]] 的潜在改进方向中，被建议用于替换 [[Memory Module]] 的固定 FIFO 队列以支持分钟级长程任务。
