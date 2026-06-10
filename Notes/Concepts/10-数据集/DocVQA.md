---
type: concept
aliases: [DocVQA, Document Visual Question Answering]
---

# DocVQA

## 定义
专注于文档图像理解的视觉问答 benchmark，要求模型从扫描文档、表格、表单等中提取并理解信息。

## 核心要点
1. 图像来自真实文档（发票、报告、表单），文字密集
2. 强调 OCR + 语言理解的结合能力
3. 是分辨率敏感任务的典型代表，高分辨率编码器在此受益明显
4. [[Eagle]] 用此 benchmark 验证 mixture-of-encoders 在文档理解上的提升

## 相关概念
- [[ChartQA]]
- [[POPE]]
- [[InternVL]]
