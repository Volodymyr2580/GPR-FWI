# Source Status

本文记录 `docs/part1_gpr_fwi/` 当前草稿中不同结论的来源状态。目的是防止把“PDF 已核实结论”“本地实验记录”和“理论推理/后续建议”混在一起。

## Status Labels

- `PDF-skimmed`: 已从本地 PDF 的标题、摘要、公式页或相关段落提取信息，但尚未完整精读全文。
- `PDF-formula-checked`: 已从本地 PDF 中定位到公式或方法段落，并转写成自己的说明。
- `local-doc-checked`: 已从本地 README、progress、migration map 等轻量项目文档中读取。
- `inference`: 基于 PDE/FWI 理论和已读材料形成的推理，需要在正式报告中谨慎表达。
- `pending`: 仍需补充文献或实验验证。

## Literature-Supported Claims

| Claim area | Status | Source | Notes |
| --- | --- | --- | --- |
| Time-domain FDTD adjoint sensitivity and resolution | PDF-formula-checked | Meles et al. 2012 | 已提取 \(\epsilon\)/\(\sigma\) sensitivity、Jacobian、pseudo-Hessian、gradient relation。 |
| On-ground quantitative permittivity/conductivity/source-wavelet inversion | PDF-skimmed | Busch et al. 2012 | 已核实 frequency-domain layered Maxwell solver、gradient-free optimization、source wavelet coupling；还需提取实验细节。 |
| Frequency-domain biparameter quasi-Newton GPR-FWI | PDF-formula-checked | Lavoue et al. 2014 | 已提取 TE 方程、objective、adjoint gradient、L-BFGS-B、parameter scaling、Tikhonov regularization。 |
| Source-independent envelope objective | PDF-formula-checked | Liu et al. 2022 | 已提取 envelope objective、Hilbert transform、backward residual source、\(\epsilon/\sigma\) update structure。 |
| Implicit multiparameter GPR-FWI | PDF-skimmed | Sun et al. 2024 | 已核实 IFWI、implicit neural representation、frequency principle；还需确认 solver details 和实验设置。 |
| Early time-domain crosshole 2D FDTD GPR-FWI | PDF-skimmed | Ernst et al. 2007 | 已核实 2D FDTD Maxwell、crosshole synthetic benchmarks、ray tomography 对比和主要限制。 |
| Laplace-domain initial-model building | PDF-formula-checked | Meng et al. 2019 | 已提取 Laplace Maxwell system、logarithmic objective、\(\epsilon/\sigma\) gradient、stepped update。 |
| OT-to-LS objective switching | PDF-skimmed | Hunziker et al. 2025 | 已核实 OT early phase、LS late phase、master-point gradient 和 near-antenna gradient handling。 |

## Local-Experiment-Supported Claims

| Claim area | Status | Source | Notes |
| --- | --- | --- | --- |
| Minimal baseline FWI workflow desire | local-doc-checked | `Gpr_fwi/ReadMe.md` | 本地笔记明确希望回归基本 FWI：生成 `d_obs`、迭代更新、简化空气层和输入输出。 |
| Tikhonov implementation | local-doc-checked | `Gpr_fwi/mode2_Tikhonov/README_Tikhonov.md` | 已读取二阶 Tikhonov 目标函数和 \(\nabla^2(\nabla^2m)\) 梯度说明。 |
| UNet dual-parameter reparameterization | local-doc-checked | `Gpr_fwi/mode2_unet/README_GPR_FWI.md` | 已读取双 UNet、\(\epsilon/\sigma\) 范围映射、400 MHz FDTD 设置。 |
| MPI shot-level parallelization | local-doc-checked | `Gpr_fwi/mode1_unet_fix_parallel/README_MPI.md` | 已读取 source/gradient shot parallelization 和 gather/bcast 机制。 |
| CUDA/PyTorch fast dual-parameter framework | local-doc-checked | `Fast-GPR-FWI/README.md` | 仓库未跟踪，但本地 README 已读；未提交该目录。 |
| Clean migration target and experiment naming | local-doc-checked | `marmousi_paper/gpr-inversion/docs/MIGRATION_MAP.md` | 已读取 eps-only、sig-only、twopara、eps-then-sig 等迁移方向。 |
| IFWI/dropout diagnostics and illumination observations | local-doc-checked | `隐式FWI/progress.md` | 已读取高层实验记录；该文件很长，当前只用于方向性连接，不作为正式文献证据。 |

## Inference Or Recommendation Claims

| Claim area | Status | How to validate |
| --- | --- | --- |
| Start with single-parameter \(\epsilon_r\) before dual-parameter inversion | inference | 用 minimal gradient-check experiment 和 eps-only baseline 验证。 |
| Wrong \(\epsilon_r\) may map into \(\sigma\) artifacts | PDF-supported + inference | Lavoue et al. 2014 支持该机制；本地需做 wrong-fixed-parameter 实验。 |
| Illumination compensation is different from regularization | inference | Meles sensitivity/resolution 支持 illumination 逻辑；本地可用 gradient energy / pseudo-Hessian normalization 对比。 |
| Neural/implicit parameterization may reduce artifacts but not remove crosstalk mathematically | inference | 需要 traditional vs UNet vs IFWI 对照实验。 |
| True inversion dropout may destabilize physical forward loop | local-doc-checked + inference | 来自 `隐式FWI/progress.md` 的实验审计；需在正式报告中标为本地观察。 |

## Pending Literature Work

- 完整精读 Busch et al. 2012 的 objective、search strategy、CMP/waveguide model setup。
- 完整精读 Sun et al. 2024 的 IFWI forward solver、network mapping、参数设置和 crosstalk 证据。
- 继续补充 Kuroda 2007 early crosshole GPR-FWI 文献。
- 继续补充 frequency-dependent / attenuation / modified TV 工作。
- 对 OT/LS 2025 文献补充 objective 细节和 open-source implementation 信息。

## Reporting Rule

正式汇报中建议采用以下措辞：

- 对 `PDF-formula-checked` 内容，可以写“某文献提出/使用/定义”。
- 对 `PDF-skimmed` 内容，可以写“某文献摘要和方法概览显示”，并保留进一步核查空间。
- 对 `local-doc-checked` 内容，只能写“本地实验记录/项目文档显示”，不能当作外部文献结论。
- 对 `inference` 内容，写“这提示我们/因此建议/需要通过实验验证”。

## Citation Anchor Rule

`citation_anchors.md` 记录的是本地 PDF 文本抽取得到的页序号和关键词命中范围。它适合用来快速回到 PDF 复核公式和方法段落，但还不是最终论文级 page/equation citation。正式引用前仍需人工检查 PDF 版面、印刷页码和公式编号。

当前已对 Meles et al. (2012) 做了针对性 PDF 抽取文本检查，确认了 Section IV-A、cost function (12)、Jacobian definition (13)、pseudo-Hessian/Taylor expansion (15)、Gauss-Newton update (20)、Appendix A (A-1)-(A-6) 等线索。正式引用时仍建议打开 PDF 视觉核对版面和印刷页码。
