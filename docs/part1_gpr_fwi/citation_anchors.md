# Citation Anchors For Part 1

本文记录当前已核实或半核实文献的轻量引用锚点。页码为本地 PDF 文本抽取的页序号（从 1 开始），不一定等同于期刊印刷页码。正式汇报或论文写作时，仍应回到 PDF 人工核对 section、equation number 和 printed page。

## Anchor Status

- `keyword-located`: 已用本地 PDF 文本抽取定位关键词页，但尚未人工核对版面。
- `formula-note-linked`: 已经和 `core_paper_notes.md` 或 `formula_summary.md` 中的公式笔记对应。
- `needs-manual-page-check`: 需要人工打开 PDF 确认印刷页码、公式编号和上下文。

## Core Formula Anchors

| Paper | Anchor status | Useful local PDF pages | What to check there |
| --- | --- | --- | --- |
| Meles et al. 2012 | keyword-located; formula-note-linked | sensitivity function: 2, 3, 5; Jacobian: 2, 3, 5, 9-12, 14, 15; pseudo-Hessian: 5, 9, 10, 13, 15, 16; cumulative sensitivity/model resolution: 10-15 | FDTD adjoint sensitivity definitions, \(J_{\mu,\eta}\), \(J^T\Delta E\), pseudo-Hessian \(J^TJ\), cumulative sensitivity and resolution interpretation. |
| Lavoue et al. 2014 | keyword-located; formula-note-linked | L-BFGS-B: 1-3, 11, 19; scaling: 7-16; Tikhonov: 9, 15, 16, 18, 19, 21; trade-off: 2, 5-9, 11 | Frequency-domain TE equation, adjoint gradient, L-BFGS-B setup, permittivity/conductivity scaling, crosstalk/trade-off, conductivity Tikhonov regularization. |
| Liu et al. 2022 | keyword-located; formula-note-linked | Hilbert: 4; envelope: 1, 3, 4, 7, 10, 11, 15, 19-22; source-independent: 1, 3, 11, 12, 21, 22; convolution: 1-4, 6-11, 14, 16; gradient: 1-7, 20 | Source-independent envelope objective, convolutional residual construction, Hilbert-transform envelope definition, backward residual source, dual-parameter gradient/update logic. |
| Meng et al. 2019 | keyword-located; formula-note-linked | Laplace: 1-3, 5-13; logarithmic: 1, 3, 5, 6, 13, 15; virtual source: 3, 4; gradient: 1-7, 12; stepped: 5, 6, 13 | Laplace-domain Maxwell system, logarithmic objective, residual scaling by \(\tilde{E}\), virtual sources for \(\epsilon\)/\(\sigma\), gradient structure, stepped update schedule. |

## Skimmed Method Anchors

| Paper | Anchor status | Current use | Next check |
| --- | --- | --- | --- |
| Busch et al. 2012 | needs-manual-page-check | On-ground quantitative permittivity/conductivity inversion; source wavelet amplitude/conductivity coupling. | Locate objective, search strategy, layered/CMP model, and waveguide experiment settings. |
| Sun et al. 2024 | needs-manual-page-check | Implicit multiparameter GPR-FWI; neural implicit representation and frequency-principle interpretation. | Locate forward solver, network mapping, parameter constraints, and crosstalk evidence. |
| Ernst et al. 2007 | needs-manual-page-check | Early crosshole 2D FDTD GPR-FWI and ray-tomography comparison. | Locate FDTD equations, synthetic model setup, inversion parameters, and limitations. |
| Hunziker et al. 2025 | needs-manual-page-check | OT-to-LS objective switching and master-point gradient handling. | Locate objective definitions, switch strategy, master-point gradient details, and experiment geometry. |

## Reporting Use

- In the current Part 1 report, cite these anchors as internal review aids, not as final publication-ready page references.
- For slides or internal meetings, these anchors are enough to quickly reopen the relevant PDF pages.
- For formal writing, convert each row into exact section/equation/page references after visual PDF inspection.
