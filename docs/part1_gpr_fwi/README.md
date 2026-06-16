# GPR-FWI Part 1 Research Workspace

本文档目录用于逐步整理“第一部分：GPR 数据 FWI 的数学理论与方法综述”。

## Goal

从第一性原理出发，建立一条自洽链条：

```text
Maxwell 方程
-> GPR 时域正演
-> 数据观测算子
-> FWI 目标函数
-> 伴随状态法
-> permittivity / conductivity 梯度
-> 单参数与双参数反演
-> crosstalk、regularization、illumination compensation
-> 后续实验设计
```

## Working Files

- `progress.md`：heartbeat 分步执行记录。
- `literature_matrix.md`：本地 reference 和扩展文献矩阵。
- `core_paper_notes.md`：核心文献精读笔记，后续逐篇补充。
- `formula_summary.md`：GPR-FWI 公式摘要，后续从理论推导中抽出。
- `Part1_GPR_FWI_theory_and_methods.md`：第一部分汇报初稿，后续生成。
- `Part1_to_experiment_recommendations.md`：从理论到实验的建议，后续生成。

## Delivery Files

- `Part1_GPR_FWI_theory_and_methods.md`：第一部分主报告草稿。
- `GPR_FWI_formula_summary.md`：按原任务命名的公式摘要。
- `GPR_FWI_core_paper_notes.md`：按原任务命名的核心文献笔记。
- `Part1_to_experiment_recommendations.md`：理论到实验的路线建议。
- `literature_matrix.md`：文献矩阵和阅读状态。
- `source_status.md`：当前结论的来源状态和可信度标注。
- `citation_anchors.md`：核心文献的 PDF 关键词页码和后续人工校对锚点。
- `citation_anchor_qa.md`：引用锚点的一致性检查和后续边界说明。
- `final_qa.md`：当前第一部分文档交付物的一致性检查记录。
- `review_checklist.md`：下一轮校对和补充清单。

兼容说明：

- `formula_summary.md` 是工作文件，内容已复制到 `GPR_FWI_formula_summary.md`。
- `core_paper_notes.md` 是工作文件，内容已复制到 `GPR_FWI_core_paper_notes.md`。

## Local Source Material

本地文献目录：

```text
E:\sci_research\GPR\GPR_references
```

轻量索引：

```text
docs/GPR_REFERENCE_INDEX.md
```

重要项目说明：

- `README.md`
- `docs/PROJECT_INVENTORY.md`
- `docs/DATA_AND_GIT_POLICY.md`
- `docs/GPT_PRO_REFERENCE_WORKFLOW.md`

## Safety Rules

- 不批量删除文件或目录。
- 不提交 `GPR_references/` PDF。
- 不运行昂贵的大规模训练或反演。
- 中文 Markdown 默认按 UTF-8 读写。
- 文献结论必须标明来源；无法确认时写 `待确认`。

## Current Strategy

当前策略是先形成可用草稿，再逐步做引用和公式校对：

1. 固定文献矩阵字段。
2. 选出核心文献。
3. 按主题补充精读笔记。
4. 再写数学推导和方法综述。
5. 最后把理论结论转化为可执行实验。

这样后续写汇报时，公式、引用和实验设置都有出处。
