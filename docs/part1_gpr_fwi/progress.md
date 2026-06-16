# Part 1 Progress

## 2026-06-16 Heartbeat 1

Status: completed initial workspace setup.

Completed:

- Created `docs/part1_gpr_fwi/`.
- Added workspace overview in `README.md`.
- Added this progress log.
- Added first-pass `literature_matrix.md` based on `docs/GPR_REFERENCE_INDEX.md`.
- Classified references into `core`, `important`, `support`, and `background`.
- Marked all PDF-derived content as `not_read` until the PDF text is actually inspected.

Important decisions:

- Do not commit or move local PDFs.
- Use the local PDF filenames only as a starting index.
- Treat filename-inferred metadata as provisional until verified from PDF text or reliable web sources.
- Start with core GPR-FWI theory papers before broad survey writing.

Next heartbeat:

1. Read available text/metadata for the highest-priority core papers.
2. Begin `core_paper_notes.md` with verified fields.
3. Start a `formula_summary.md` skeleton for Maxwell-to-gradient derivation.

Open questions:

- Need to confirm which PDF text extraction tools are available locally.
- Need to verify exact citation metadata for several filename-only entries.
- Need to decide whether `DX203108.pdf` is directly relevant or a broad background reference.

## 2026-06-16 Heartbeat 2

Status: completed first PDF extraction pass and created theory skeleton.

Completed:

- Confirmed bundled Python has `pdfplumber` and `pypdf`.
- Skimmed title/abstract/keyword contexts from five core PDFs:
  - Meles et al. 2012, FDTD adjoint sensitivity and resolution analysis.
  - Busch et al. 2012, quantitative on-ground permittivity/conductivity/source-wavelet inversion.
  - Lavoue et al. 2014, frequency-domain quasi-Newton biparameter inversion.
  - Sun et al. 2024, implicit multiparameter GPR-FWI.
  - Liu et al. 2022, source-independent envelope objective for cross-hole GPR.
- Updated `literature_matrix.md` statuses for those papers from `not_read` to `skimmed`.
- Added `core_paper_notes.md` with first verified notes and pending extraction fields.
- Added `formula_summary.md` with Maxwell equations, second-order electric-field equation, basic objective, Lagrangian, formal epsilon/sigma gradients, velocity conversion, crosstalk block-Hessian skeleton, and regularization skeleton.

Technical note:

- One PowerShell/Python command initially failed because a quoted Python executable path needs the PowerShell call operator `&`.
- One PDF extraction print initially hit a GBK console encoding issue; setting `PYTHONIOENCODING=utf-8` fixed it. Future PDF extraction should use UTF-8 output explicitly.

Next heartbeat:

1. Extract exact formulas from Meles et al. 2012 and Liu et al. 2022 if text extraction is clean.
2. Begin the main draft `Part1_GPR_FWI_theory_and_methods.md` with Sections 1-4 skeleton.
3. Add an `open_questions.md` or pending list if citation metadata remains uncertain.

Open questions:

- Need exact Meles adjoint/sensitivity equations, not only abstract-level notes.
- Need exact Liu source-independent envelope objective and adjoint-source formula.
- Need to verify whether Sun et al. IFWI examples use time-domain FDTD and exactly which parameters are inverted.

## 2026-06-16 Heartbeat 3

Status: extracted formula-level notes and started the main Part 1 draft.

Completed:

- Located formula-heavy pages in Meles et al. 2012 and Liu et al. 2022 using UTF-8 PDF extraction.
- Added Meles formula notes to `core_paper_notes.md`:
  - \(\epsilon\) sensitivity uses a time derivative of the forward electric field correlated with an adjoint receiver wavefield.
  - \(\sigma\) sensitivity uses the forward electric field itself correlated with an adjoint receiver wavefield.
  - \(J_{\mu,\eta}=\partial d_\mu/\partial m_\eta\).
  - \(H_A=J^TJ\).
  - \(\nabla_m S=J^T\Delta E\).
  - damped Gauss-Newton update form.
- Added Liu formula notes to `core_paper_notes.md`:
  - source-independent envelope objective based on convolutional wavefields.
  - Hilbert-transform envelope definitions.
  - backward residual field source structure.
  - simultaneous permittivity/conductivity conjugate-gradient update.
- Expanded `formula_summary.md` with:
  - first-order/FDTD sensitivity viewpoint from Meles et al. 2012.
  - source-independent envelope objective and gradient skeleton from Liu et al. 2022.
- Created `Part1_GPR_FWI_theory_and_methods.md` with draft Sections 1-4:
  - PDE-constrained optimization framing.
  - Maxwell-to-GPR forward model.
  - observation operator and waveform objective.
  - adjoint-state gradient derivation overview.

Next heartbeat:

1. Expand Sections 5-6 on single-parameter and dual-parameter inversion.
2. Use Lavoue et al. 2014 to write the first serious crosstalk/scaling/regularization discussion.
3. Verify Sun et al. 2024 method details and decide how cautiously to present neural implicit parameterization.

Open questions:

- Need to reconcile derivative order between the second-order electric-field derivation and Meles' first-order/vector field sensitivity notation.
- Need to extract exact Lavoue parameter scaling definitions and Tikhonov regularization setup.
- Need to avoid overstating source-independent/envelope objective advantages because Liu et al. also note convolution/cross-correlation increases nonlinearity.

## 2026-06-16 Heartbeat 4

Status: expanded biparameter inversion and crosstalk material using Lavoue et al. 2014.

Completed:

- Extracted formula-level notes from Lavoue et al. 2014:
  - frequency-domain TE scalar equation.
  - effective permittivity \(\epsilon_e=\epsilon+i\sigma/\omega\).
  - discrete system \(A(\omega,\epsilon,\sigma)u=s\).
  - data misfit \(C(m)=1/2\sum_{\omega,s}\Delta d^\dagger\Delta d\).
  - adjoint-state gradient \(G_i=\Re\sum u^T(\partial A/\partial m_i)^T v^*\).
  - finite-difference derivatives \(\partial A/\partial\epsilon=-\omega^2\delta\), \(\partial A/\partial\sigma=-i\omega\delta\).
  - L-BFGS-B update and Wolfe line search.
  - parameter scaling \((\epsilon_r,\sigma_r/\beta)\).
  - conductivity Tikhonov term \(C_M=1/2\sigma_r^TD\sigma_r\).
- Updated `core_paper_notes.md` with Lavoue details.
- Expanded `formula_summary.md` with frequency-domain biparameter formulas, Hessian block scaling, and conductivity regularization.
- Replaced the main draft placeholders for Sections 5-6 with:
  - single-parameter inversion rationale.
  - dual-parameter inversion and crosstalk explanation.
  - practical implications of parameter scaling and regularization.

Next heartbeat:

1. Expand Section 7 on time-domain GPR-FWI workflow.
2. Expand Section 8 on objective functions, regularization, and illumination compensation.
3. Add an experiment-design file or draft Section 9 with minimal verification and crosstalk diagnostics.

Open questions:

- Need to bring Meles time-domain illumination/resolution material into Section 8.
- Need to verify more time-domain examples beyond Meles/Liu because Lavoue is frequency-domain.
- Need to inspect local project experiments so the theory report connects to completed experiments.

## 2026-06-16 Heartbeat 5

Status: expanded time-domain workflow, objective functions, regularization, and illumination compensation.

Completed:

- Replaced the main draft placeholders for Sections 7-8:
  - Section 7 now describes the time-domain GPR-FWI loop from forward FDTD to residual injection, adjoint propagation, gradient accumulation, gradient processing, and model update.
  - Section 8 now compares L2 waveform, normalized/correlation, source-independent, envelope, frequency-domain, and Laplace-domain objectives.
  - Section 8 also distinguishes Tikhonov, TV, bound constraints, and neural/implicit parameterization.
  - Section 8 adds an illumination compensation discussion based on Meles et al. 2012 sensitivity/resolution logic.
- Expanded `formula_summary.md` with:
  - multi-source gradient accumulation.
  - pseudo-Hessian / illumination normalization.

Next heartbeat:

1. Fill Section 9 with typical experiment settings and a bridge to local completed experiments.
2. Create `Part1_to_experiment_recommendations.md` with minimal verification, crosstalk diagnostics, illumination compensation, objective comparison, and regularization experiments.
3. Inspect lightweight README/docs in local experiment folders, without running heavy experiments.

Open questions:

- Need to inspect project experiment folders and map them to theory categories.
- Need to decide which local experiment line is the cleanest baseline for finite-difference gradient check.
- Need to keep separating literature-verified statements from our own experimental hypotheses.

## 2026-06-16 Heartbeat 6

Status: connected theory report to local experiment lines and created experiment recommendations.

Completed:

- Read lightweight local experiment documentation:
  - `Gpr_fwi/ReadMe.md`
  - `Gpr_fwi/mode2_Tikhonov/README_Tikhonov.md`
  - `Gpr_fwi/mode2_unet/README_GPR_FWI.md`
  - `Fast-GPR-FWI/README.md`
  - `Gpr_fwi/mode1_unet_fix_parallel/README_MPI.md`
  - `marmousi_paper/gpr-inversion/docs/MIGRATION_MAP.md`
  - selected high-level status from `隐式FWI/progress.md`
- Expanded Section 9 in `Part1_GPR_FWI_theory_and_methods.md`:
  - crosshole, on-ground/surface-to-surface, simple anomaly, and complex benchmark experiment types.
  - mapping from local experiment folders to theory questions.
  - recommended staged experiment ladder.
- Added `Part1_to_experiment_recommendations.md` with:
  - minimal verification experiment.
  - crosstalk diagnostics.
  - illumination compensation experiments.
  - objective function comparison.
  - regularization sweep.
  - network/implicit representation comparison.
  - recommended execution order.

Technical note:

- Tried to read two guessed Marmousi docs paths that did not exist. The actual docs include `PROJECT_STATUS_AND_PROGRESS.md` and `MARMOUSI_EXPERIMENTS_GUIDE.md`; this was a harmless filename mismatch.

Next heartbeat:

1. Add or copy final-name files expected by the original plan:
   - `GPR_FWI_formula_summary.md`
   - `GPR_FWI_core_paper_notes.md`
2. Perform a light consistency pass over the main report.
3. Decide whether to commit the `docs/part1_gpr_fwi/` research artifacts.

Open questions:

- Need exact final filenames versus current working filenames.
- Need a final consistency pass on notation: \(\epsilon\), \(\epsilon_r\), and \(\sigma_r\).
- Need to decide whether to broaden literature beyond the first five core PDFs before calling Part 1 draft complete.

## 2026-06-16 Heartbeat 7

Status: aligned filenames with the original deliverable plan and performed a light consistency pass.

Completed:

- Created final-name copies:
  - `GPR_FWI_formula_summary.md` from `formula_summary.md`.
  - `GPR_FWI_core_paper_notes.md` from `core_paper_notes.md`.
- Updated `README.md` with a delivery-file list and compatibility notes.
- Scanned the main report for remaining placeholders. No empty sections remain.
- Reworded one table entry that contained the word `placeholder` but was not an actual empty section.
- Checked that the formula summary includes definitions/usage for \(\epsilon\), \(\epsilon_r\), \(\sigma\), and \(\sigma_r\).

Current deliverables:

- `Part1_GPR_FWI_theory_and_methods.md`
- `GPR_FWI_formula_summary.md`
- `GPR_FWI_core_paper_notes.md`
- `Part1_to_experiment_recommendations.md`
- `literature_matrix.md`
- `progress.md`

Next heartbeat:

1. Do a light citation/source-status pass: distinguish PDF-verified notes, local experiment notes, and inference.
2. Optionally add a short `review_checklist.md`.
3. If the artifact set looks stable, stage and commit only `docs/part1_gpr_fwi/`.

Open questions:

- The literature matrix is useful but not yet a full expanded web literature database.
- Only five core PDFs have been skimmed at formula/detail level.
- A future pass should broaden coverage before treating this as a final polished literature review.

## 2026-06-16 Heartbeat 8

Status: added source-status and review-control documents.

Completed:

- Added `source_status.md` to distinguish:
  - `PDF-formula-checked`
  - `PDF-skimmed`
  - `local-doc-checked`
  - `inference`
  - `pending`
- Added `review_checklist.md` for main report, formula summary, literature matrix, experiment recommendations, and Git/data safety.
- Updated `README.md` to list the new quality-control files.

Next heartbeat:

1. Either broaden literature coverage with one or two important pending PDFs.
2. Or stage/commit the source-status pass if stable.
3. Then decide whether the heartbeat should keep running or pause after the next literature-expansion step.

Open questions:

- Whether to prioritize more PDF reading or start turning the draft into a polished Chinese report.
- Whether the next concrete experiment should be a gradient check config in the clean `marmousi_paper/gpr-inversion/` target or the older `Gpr_fwi/` baseline.

## 2026-06-16 Heartbeat 9

Status: broadened literature coverage beyond the first five core PDFs.

Completed:

- Skimmed and extracted notes from:
  - Ernst et al. 2007, early crosshole GPR-FWI based on 2D FDTD Maxwell solutions.
  - Meng et al. 2019, Laplace-domain waveform inversion for cross-hole radar initial model building.
  - Hunziker et al. 2025, OT-to-LS objective switching for crosshole GPR-FWI.
- Updated `literature_matrix.md` statuses for these papers.
- Added new sections to both `core_paper_notes.md` and `GPR_FWI_core_paper_notes.md`.
- Added Laplace-domain logarithmic objective and OT-to-LS switching to both formula summary files.
- Updated the main report's objective-function section and reference list.
- Updated `source_status.md` and `review_checklist.md`.

Next heartbeat:

1. Commit this literature-expansion pass if the diff is clean.
2. Then prioritize either polished Chinese report rewriting or one more method-family expansion, such as frequency-dependent/attenuation or modified TV.

Open questions:

- Need exact DOI for Ernst et al. 2007 and Hunziker et al. 2025 if not already in local metadata.
- Kuroda 2007 remains unread and could complement Ernst 2007.

## 2026-06-16 Heartbeat 10

Status: started polishing the main report into a more presentation-ready Chinese draft.

Completed:

- Replaced the old English `working draft` status line in `Part1_GPR_FWI_theory_and_methods.md`.
- Added:
  - `0. 执行摘要`
  - `0.1 证据边界`
  - `0.2 术语说明`
- Clarified the current evidence levels:
  - PDF-formula-checked
  - PDF-skimmed
  - local-doc-checked
- Updated `review_checklist.md` to mark the Chinese presentation-tone pass as started/completed for the opening section.

Next heartbeat:

1. Continue Chinese polishing section by section, especially Sections 1-4.
2. Add page/equation anchors for the most important formulas if time permits.
3. Commit this polish pass if the diff is clean.

Open questions:

- Whether to keep English technical terms inline for readability or move more of them into a glossary.
- Whether to split the final report into a concise presentation version and a longer technical appendix.

## 2026-06-16 Heartbeat 11

Status: polished the main mathematical-theory opening of the report.

Completed:

- Rewrote Sections 1-4 of `Part1_GPR_FWI_theory_and_methods.md` into a more coherent Chinese report style.
- Clarified why GPR-FWI is a nonlinear PDE-constrained optimization problem instead of ordinary curve fitting.
- Strengthened the Maxwell-to-GPR forward-model explanation, including the physical roles of \(\epsilon_r\) and \(\sigma\).
- Refined the L2 objective discussion to connect cycle skipping, source wavelet uncertainty, amplitude ambiguity, and crosstalk.
- Added the \(\epsilon_r\) chain-rule expression and an explicit reminder that adjoint-gradient signs must be checked by finite-difference gradient tests in code.

Next heartbeat:

1. Continue polishing Sections 5-8, especially single-parameter inversion, biparameter crosstalk, time-domain workflow, and objective/regularization/illumination methods.
2. If time permits, add a short “implementation checklist” after the gradient section to connect formulas with the local FDTD code.
3. Keep the scope limited to Markdown research artifacts; do not touch `GPR_references/` PDFs or the untracked `Fast-GPR-FWI/` directory.

## 2026-06-16 Heartbeat 12

Status: polished the inversion-method sections of the main report.

Completed:

- Rewrote Sections 5-8 of `Part1_GPR_FWI_theory_and_methods.md` to align their tone with the polished mathematical opening.
- Clarified why single-parameter \(\epsilon_r\) inversion should be the first diagnostic experiment before \(\sigma\) or dual-parameter inversion.
- Strengthened the biparameter crosstalk discussion around Jacobian columns, Hessian off-diagonal blocks, parameter scaling, and the risk of trusting data misfit alone.
- Added implementation-level notes to the time-domain workflow, especially wavefield storage, staggered-grid alignment, PML masking, source accumulation, parameter scaling, and finite-difference gradient checks.
- Added a short method taxonomy for objective functions and a sharper boundary between regularization and illumination compensation.

Next heartbeat:

1. Polish Section 9 and the local-experiment mapping so the theory-to-experiment bridge reads like a clear roadmap.
2. Add a concrete first gradient-check config sketch to `Part1_to_experiment_recommendations.md`.
3. Consider one final consistency pass to keep `formula_summary.md` and `GPR_FWI_formula_summary.md` aligned with the main report wording.

## 2026-06-16 Heartbeat 13

Status: strengthened the theory-to-experiment bridge.

Completed:

- Polished Section 9 of `Part1_GPR_FWI_theory_and_methods.md` to distinguish old experiment lines as evidence sources from `marmousi_paper/gpr-inversion/` as the cleaner future landing path.
- Added a two-layer experiment roadmap:
  - minimal verification layer for forward/adjoint/gradient closure;
  - method-comparison layer for \(\sigma\), biparameter scaling, regularization, illumination, objectives, and network/implicit parameterizations.
- Added a concrete `minimal_eps_l2_gradient_check` YAML-style config sketch to `Part1_to_experiment_recommendations.md`.
- Added explicit gradient-check diagnostics: residual sign, adjoint source time order, PML mask, \(\epsilon\)-to-\(\epsilon_r\) chain rule, and staggered-grid alignment.

Next heartbeat:

1. Sync `formula_summary.md` and `GPR_FWI_formula_summary.md` with the new diagnostic wording from the main report and experiment recommendations.
2. Add short “implementation cautions” near the formula summary sections for single-parameter and biparameter experiments.
3. Then do a small consistency pass across `README.md`, `progress.md`, and `review_checklist.md`.

## 2026-06-16 Heartbeat 14

Status: synchronized formula summaries with implementation diagnostics.

Completed:

- Updated both `formula_summary.md` and `GPR_FWI_formula_summary.md` from an initial skeleton label to a formula companion description.
- Added a finite-difference directional derivative check near the adjoint-gradient formulas.
- Added implementation cautions for residual sign, adjoint-source time order, PML masking, \(\epsilon\)-to-\(\epsilon_r\) chain rule, and staggered-grid interpolation.
- Added diagnostic guidance for single-parameter and biparameter experiments before trusting simultaneous \((\epsilon_r,\sigma)\) inversion.
- Added recommended raw-gradient / compensated-gradient / illumination-map diagnostics for pseudo-Hessian or illumination compensation tests.

Next heartbeat:

1. Do a compact consistency pass over `README.md`, `source_status.md`, and the final report references.
2. If still useful, add page/equation anchors for the most important formula-checked papers.
3. Keep runnable config creation as a separate implementation task after the theory/report artifacts stabilize.

## 2026-06-16 Heartbeat 15

Status: added lightweight citation anchors for formula-checked papers.

Completed:

- Used local PDF text extraction to locate keyword pages for Meles et al. 2012, Lavoue et al. 2014, Liu et al. 2022, and Meng et al. 2019.
- Added `citation_anchors.md` with local PDF page indices for sensitivity/Jacobian/pseudo-Hessian, frequency-domain biparameter scaling/Tikhonov, source-independent envelope objective, and Laplace-domain logarithmic objective.
- Updated `README.md` to list the citation-anchor file.
- Updated `source_status.md` to clarify that these anchors are internal review aids, not final publication-ready page/equation citations.

Next heartbeat:

1. Do one final consistency read of the main report, formula summary, core notes, and experiment recommendations.
2. Decide whether the theory/report heartbeat should pause after final QA or move into runnable config implementation.
3. Leave any unverified page/equation numbers as pending rather than inventing exact references.

## 2026-06-16 Heartbeat 16

Status: completed a final consistency QA pass for the Part 1 documentation artifacts.

Completed:

- Checked that all expected delivery files exist in `docs/part1_gpr_fwi/`.
- Verified that `formula_summary.md` and `GPR_FWI_formula_summary.md` have no detected text differences.
- Verified that `core_paper_notes.md` and `GPR_FWI_core_paper_notes.md` have no detected text differences.
- Checked that the main report has a complete section structure and no `To be expanded` placeholder.
- Confirmed that remaining `metadata_pending` entries are intentionally confined to the literature matrix.
- Added `final_qa.md` to record current readiness, remaining work, and the boundary between theory/report writing and runnable config implementation.

Next heartbeat:

1. If the heartbeat continues, either perform manual citation-anchor upgrades or start the runnable `minimal_eps_l2_gradient_check` config implementation after user confirmation.
2. Keep the theory/report artifacts stable unless a specific correction is found.
3. Continue avoiding `GPR_references/` PDF commits and any large training/inversion runs.
