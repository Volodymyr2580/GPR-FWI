# GPT Pro Reference Workflow

本文说明如何把本地 `GPR_references/` 文献交给 GPT Pro 代理使用，以及 GitHub 仓库能做什么、不能做什么。

## 结论

GPT Pro 代理通常可以查看 GitHub 仓库中的文件，但有几个前提：

- 如果仓库是公开的，代理一般可以通过网页或 GitHub 链接查看。
- 如果仓库是私有的，需要你在 GPT Pro 里显式授权 GitHub 连接器或把内容复制/上传给它。
- 云端代理看不到你电脑上的本地路径，例如 `E:\sci_research\GPR\GPR_references`。
- 被 `.gitignore` 忽略、没有提交到 GitHub 的文件，代理也看不到。

当前仓库故意没有提交 `GPR_references/` 里的 PDF，因为论文 PDF 常有版权限制，而且 GitHub/Git 不适合作为大文件文献库。

## 推荐方案

### 方案 A：直接上传给 GPT Pro

这是最简单、最可靠的办法。

做法：

1. 在本地选择需要 GPT Pro 阅读的 PDF。
2. 如果数量不多，直接在 GPT Pro 对话中上传。
3. 如果数量较多，压缩成一个 zip 后上传。
4. 把 `docs/GPT_PRO_AGENT_PROMPT.md` 里的提示词复制给 GPT Pro。

优点：

- GPT Pro 能直接读取 PDF 内容。
- 不需要把可能有版权限制的论文传到公开仓库。
- 不受 GitHub 文件大小和仓库体积影响。

缺点：

- 每次新对话可能需要重新上传。
- 大量 PDF 可能超过上传限制。

### 方案 B：GitHub 仓库 + 手动上传 PDF

这是推荐的长期工作流。

GitHub 仓库负责保存：

- 项目结构说明
- 文献索引
- 代理提示词
- 综述草稿
- 实验设计文档
- 可公开的代码和配置

PDF 文献负责通过 GPT Pro 上传附件提供。

这样代理可以先通过 GitHub 理解项目，再通过附件读取具体文献内容。

### 方案 C：只使用 GitHub

只有当这些文献满足以下条件时，才建议把 PDF 放到 GitHub：

- 你确认有权上传和共享这些 PDF。
- 仓库是私有仓库，且访问范围受控。
- 单个文件不超过 GitHub 限制。
- 仓库长期不会因为大文件变得臃肿。

如果只是为了让 GPT Pro 读取文献，通常不建议这样做。

## 给初学者的解释

GitHub 仓库像是项目的“代码和文档档案馆”。它很适合存代码、配置、Markdown 笔记和小型文本文件。

但 PDF、`.npy`、图片、实验输出这类文件通常比较大，而且有些 PDF 还有版权问题。把它们放进 Git 仓库后，哪怕以后删除，历史记录里仍然可能保留它们，仓库会越来越大。因此当前项目的 `.gitignore` 明确忽略了：

- `GPR_references/`
- `*.pdf`
- `*.npy`
- `outputs/`
- `runs/`
- `logs/`

这不会删除本地文件，只是告诉 Git：“这些文件不要提交到 GitHub”。

## 当前建议

本项目建议采用：

1. 把本仓库推送到 GitHub，让 GPT Pro 读取项目结构、索引和提示词。
2. 把核心 PDF 手动上传给 GPT Pro。
3. 让 GPT Pro 输出结构化 Markdown 文件，再把这些输出放回仓库。

这样既安全，也更适合持续研究。
