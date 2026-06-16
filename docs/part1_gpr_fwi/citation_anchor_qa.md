# Citation Anchor QA

本文记录第 21 轮 heartbeat 对 `citation_anchors.md` 的一致性检查。目标是说明哪些引用锚点已经可用于内部复核，哪些仍不能当作正式论文引用。

## Current Status

已完成第一轮 `manual-text-checked` 的 formula-checked 核心文献：

| Paper | Current anchor level | Main formula/method anchors |
| --- | --- | --- |
| Meles et al. 2012 | manual-text-checked; formula-note-linked | FDTD adjoint sensitivity, cost function, Jacobian, pseudo-Hessian, model resolution, Gauss-Newton update, Appendix A Hessian relations |
| Lavoue et al. 2014 | manual-text-checked; formula-note-linked | Frequency-domain TE equation, objective, adjoint gradient, L-BFGS-B, parameter scaling, Hessian/scaling structure, conductivity Tikhonov regularization |
| Liu et al. 2022 | manual-text-checked; formula-note-linked | Source-independent objective, envelope objective, Hilbert-transform envelope, gradient derivation, backward residual sources, simultaneous \(\epsilon/\sigma\) updates |
| Meng et al. 2019 | manual-text-checked; formula-note-linked | Laplace-domain Maxwell system, logarithmic objective, residual scaling, virtual sources, gradient structure, stepped update, damping-constant discussion |

仍为 `needs-manual-page-check` 的 skimmed method references：

- Busch et al. 2012
- Sun et al. 2024
- Ernst et al. 2007
- Hunziker et al. 2025

## QA Findings

- `citation_anchors.md` now separates internal review anchors from formal citation claims.
- All four `PDF-formula-checked` papers in `source_status.md` have first-round section/equation clues.
- The same anchor summaries have been synchronized into both `core_paper_notes.md` and `GPR_FWI_core_paper_notes.md`.
- The anchors are based on local PDF page indices and text extraction, not final printed page verification.
- No PDF files from `GPR_references/` are included in the documentation commits.

## Remaining Citation Work

- Open each PDF visually and verify printed page numbers, section headers, equation numbers, and surrounding context.
- Upgrade the four skimmed method references if they become central to the final report:
  - Busch et al. 2012 for on-ground quantitative inversion and source-wavelet coupling.
  - Sun et al. 2024 for implicit multiparameter FWI.
  - Ernst et al. 2007 for early time-domain crosshole FDTD FWI.
  - Hunziker et al. 2025 for OT-to-LS objective switching.
- Keep any unverified DOI/page/equation detail marked as pending rather than promoting it into the main report.

## Recommended Next Boundary

The Part 1 theory/report package is now ready for internal review. Further heartbeat work should branch explicitly into one of two modes:

1. Citation polishing mode: visual PDF verification and publication-style references.
2. Implementation mode: convert `minimal_eps_l2_gradient_check` into a runnable config/runner/test after user confirmation.
