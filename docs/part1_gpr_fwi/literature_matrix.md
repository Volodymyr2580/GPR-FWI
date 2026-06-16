# GPR-FWI Literature Matrix

This matrix starts from `docs/GPR_REFERENCE_INDEX.md`. It is intentionally conservative: entries inferred only from filenames are marked `not_read` and `metadata_pending`.

Status values:

- `not_read`: PDF/text not yet inspected.
- `metadata_pending`: title/authors/year need verification.
- `skimmed`: partial text inspected.
- `read`: core fields verified from PDF or reliable source.

Priority values:

- `core`: likely central for Part 1 theory/methods.
- `important`: likely useful for method comparison or experiments.
- `support`: useful but not central.
- `background`: textbook, optimization, FDTD, or broad GPR background.

## Core Queue

| Priority | Status | Local path | Inferred topic | Domain | Parameters | Key fields to extract next |
| --- | --- | --- | --- | --- | --- | --- |
| core | skimmed | `GPR_references\gpr-fwi\2012 Meles GPR Full-Waveform Sensitivity and Resolution Analysis Using an FDTD Adjoint Method.pdf` | FDTD adjoint sensitivity/resolution analysis for GPR-FWI | time-domain | permittivity/conductivity sensitivity | governing equation; exact sensitivity formulas; resolution tests; illumination |
| core | skimmed | `GPR_references\gpr-fwi\2012 Quantitative conductivity and permittivity estimation using FWI of on-ground GPR data.pdf` | Quantitative epsilon/sigma estimation from on-ground GPR | frequency-domain | permittivity + conductivity + source wavelet | objective; gradient-free search; wavelet coupling; layered/CMP experiment geometry |
| core | skimmed | `GPR_references\gpr-fwi\2014 Two-dimensional permittivity and conductivity imaging Multioffset freq-domain quasi-Newton.pdf` | 2D biparameter frequency-domain quasi-Newton GPR-FWI | frequency-domain | permittivity + conductivity | L-BFGS-B; parameter scaling; Tikhonov regularization; broad-band frequency strategy |
| core | not_read; metadata_pending | `GPR_references\gpr-fwi\2018 Inverts permittivity and conductivity with structural constraint in GPR FWI based on truncated Newton method.pdf` | Structural constraint and truncated Newton dual-parameter inversion | pending | permittivity + conductivity | structural constraint; optimizer; crosstalk mitigation |
| core | not_read; metadata_pending | `GPR_references\gpr-fwi\2019 Multiscale_Full-Waveform_Dual-Parameter_Inversion_Based_on_Total_Variation_Regularization_to_On-Ground_GPR_Data.pdf` | Multiscale dual-parameter GPR-FWI with TV regularization | pending | permittivity + conductivity | multiscale strategy; TV formula; model setup |
| core | not_read; metadata_pending | `GPR_references\gpr-fwi\2021 Multiparameter 3D on-ground GPR with a modified Total Variation Regularization Scheme.pdf` | 3D multiparameter on-ground GPR-FWI with modified TV | pending | multiparameter | 3D setup; modified TV; computational strategy |
| core | skimmed | `GPR_references\gpr-fwi\2022 Source-independent envelop objective function.pdf` | Source-independent envelope objective for cross-hole GPR-FWI | time-domain | permittivity/conductivity pending | source-independent gradient; envelope objective; convolutional multiscale strategy |
| core | skimmed | `GPR_references\2021以来较新工作\Implicit multiparameter FWI.pdf` | Implicit multiparameter FWI for multioffset GPR data | time-domain likely; verify solver details | permittivity + conductivity | implicit neural representation; frequency principle; reduced initial-model dependence |
| core | not_read; metadata_pending | `GPR_references\Real-time dual-parameter fwi of GPR data based on robust deep learning.pdf` | Robust deep learning for real-time dual-parameter GPR-FWI | learning/surrogate | permittivity + conductivity likely | network role; training data; objective; limitations |
| core | not_read; metadata_pending | `GPR_references\gpr-fwi\GPR-FWI-Py.pdf` | GPR-FWI Python implementation/reference | pending | pending | equations; implementation assumptions; reproducibility value |

## Important Method References

| Priority | Status | Local path | Inferred topic | Domain | Parameters | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2007 Kuroda FWI algorithm crosshole theoretical approach.pdf` | Early crosshole GPR-FWI theoretical approach | pending | pending | likely historical foundation |
| important | skimmed | `GPR_references\gpr-fwi\2007a cross-hole 2D FDTD.pdf` | Early crosshole GPR-FWI based on 2D FDTD Maxwell solver | time-domain | permittivity + conductivity | benchmark synthetic models; ray vs FWI; resolution limits |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2010 fwi of cross-hole gpr data to characterize a gravel aquifer.pdf` | Crosshole GPR-FWI field/aquifer characterization | pending | likely permittivity/conductivity | field geometry and experiment setting |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2010 New Vector WI algorithem for simultaneous updating.pdf` | Vector waveform inversion and simultaneous updating | pending | multiparameter likely | may discuss vector EM fields |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2011 Taming non-linearity problem for high contrast media.pdf` | Nonlinearity/cycle skipping in high-contrast GPR-FWI | pending | pending | relevant to objective/multiscale |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2012 Evaluation of the reconstruction limits of a freq-independent crosshole fwi scheme in the presence of dispersion.pdf` | Dispersion effects and reconstruction limits | pending | pending | important for GPR material assumptions |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2013 Improvements in crosshole GPR FWI and application on data.pdf` | Crosshole GPR-FWI improvements and field application | pending | pending | field validation |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2015 3D FWI of crosshole actually 2.5D.pdf` | 3D/2.5D crosshole GPR-FWI | pending | pending | geometry and dimensionality |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2017 FWI Crosshole GPR Implications for porosity estimation in chalk.pdf` | Crosshole GPR-FWI and porosity estimation | pending | permittivity/porosity | petrophysical interpretation |
| important | formula_checked | `GPR_references\gpr-fwi\2019 Laplace Domain FWI to cross-hole Radar.pdf` | Laplace-domain crosshole radar waveform inversion for initial model building | Laplace-domain + time-domain FWI initialization | permittivity + conductivity | logarithmic objective; Laplace gradient; damping constant; stepped update |
| important | not_read; metadata_pending | `GPR_references\gpr-fwi\2021 A_Frequency-Domain_Quasi-Newton-Based_Biparameter_Synchronous_Imaging_Scheme_for_Ground_Penetrating_Radar.pdf` | Frequency-domain biparameter synchronous imaging | frequency-domain | biparameter | crosstalk and Hessian/preconditioning |
| important | skimmed | `GPR_references\2021以来较新工作\Crosshole ground-penetrating radar FWI by combining OT and Least-square distances.pdf` | Optimal transport + least squares for crosshole GPR-FWI | crosshole objective strategy | permittivity in example | OT-to-LS switching; master-point gradient; cycle skipping |
| important | not_read; metadata_pending | `GPR_references\2021以来较新工作\FWIof gpr in freq-dependent media involving permittivity attenuation.pdf` | Frequency-dependent media, permittivity attenuation | pending | permittivity + attenuation | material dispersion/attenuation |
| important | not_read; metadata_pending | `GPR_references\Vector Waveform Inversion algorithm.pdf` | Vector waveform inversion algorithm | pending | multiparameter likely | theoretical method support |

## Experiment and Application References

| Priority | Status | Local path | Inferred topic | Notes |
| --- | --- | --- | --- | --- |
| important | not_read; metadata_pending | `GPR_references\2012_Klotzscheetal.FWIwaveguidecrosshole.pdf` | Crosshole/waveguide GPR-FWI | likely important for field setup |
| important | not_read; metadata_pending | `GPR_references\2015 Imaging and characterization of facies heterogeneity in an alluvial aquifer.pdf` | Alluvial aquifer facies characterization | experiment/model setting |
| important | not_read; metadata_pending | `GPR_references\3-D characterization Klotzsche.pdf` | 3D characterization | field setting and 3D interpretation |
| support | not_read; metadata_pending | `GPR_references\Estimation of Subsurface Cylindrical Object properties from GPR FWI.pdf` | Cylindrical target/property inversion | useful simple synthetic setup |
| support | not_read; metadata_pending | `GPR_references\Water Resources Research - 2014 - Klotzsche - Detection of spatially limited high‐porosity layers using crosshole GPR.pdf` | High-porosity layers with crosshole GPR | hydrogeophysical application |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2012 # Monte Carlo full-waveform inversion of crosshole GPR data using multiple-point geostatistical simulation.pdf` | Monte Carlo / geostatistical crosshole GPR-FWI | uncertainty/prior discussion |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2013 3D characterizatin of an Aquifer Klotzsche.pdf` | Aquifer 3D characterization | field example |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2013 Improved Characterization of Fine-Texture soils using on-ground GPRFWI.pdf` | Fine-texture soils on-ground GPR-FWI | surface/on-ground example |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2016 Better Imaging for Landmine Detection An exploration of 3D fwi for gpr 写了一整本关于landmine的书.pdf` | Landmine detection 3D GPR-FWI | application; possibly large thesis/book |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2018 Cross hole Xiuyan Jade Mine.pdf` | Crosshole GPR in mine setting | application |
| support | not_read; metadata_pending | `GPR_references\gpr-fwi\2021 fractal-constrained Crosshole Borehole-to-surface.pdf` | Fractal-constrained borehole-to-surface inversion | regularization/prior |

## Background and Supporting Theory

| Priority | Status | Local path | Inferred topic | How it helps |
| --- | --- | --- | --- | --- |
| background | not_read; metadata_pending | `GPR_references\Numerical Optimization.pdf` | Numerical optimization textbook/reference | L-BFGS, Newton, line search, regularization framing |
| background | not_read; metadata_pending | `GPR_references\Numerical_solution_of_initial_boundary_value_problems_involving_maxwells_equations_in_isotropic_media.pdf` | Maxwell initial-boundary value problems | PDE well-posedness and discretization background |
| background | not_read; metadata_pending | `GPR_references\Finite-difference_time-domain_simulation_of_ground_penetrating_radar_on_dispersive_inhomogeneous_and_conductive_soil.pdf` | FDTD GPR simulation in dispersive/conductive soil | forward modeling assumptions |
| background | not_read; metadata_pending | `GPR_references\gpr-fwi\2012 Joint Inverse for EM and elastic full-waveform data.pdf` | Joint EM and elastic FWI | contrast with seismic and multiphysics coupling |
| background | not_read; metadata_pending | `GPR_references\gpr-fwi\2014 F Lavoue Freq-domain modeling on-ground data.pdf` | Frequency-domain on-ground GPR modeling | frequency-domain comparison |
| background | not_read; metadata_pending | `GPR_references\赵sir推荐\A_Realistic_FDTD_Numerical_Modeling_Framework_of_Ground_Penetrating_Radar_for_Landmine_Detection.pdf` | Realistic FDTD GPR modeling framework | source/antenna/PML setup |
| background | not_read; metadata_pending | `GPR_references\赵sir推荐\GPR_Full-Waveform_Inversion_With_Deep-Learning_Forward_Modeling_A_Case_Study_From_Non-Destructive_Testing.pdf` | Deep-learning forward modeling for GPR-FWI | surrogate modeling comparison |
| background | not_read; metadata_pending | `GPR_references\赵sir推荐\ground penetrating radar fundamentals.pdf` | GPR fundamentals | conceptual background |
| background | not_read; metadata_pending | `GPR_references\赵sir推荐\Ground-Penetrating-Radar-THEORY-AND-APPLICATIONS book.pdf` | GPR theory and applications | broad physical context |
| background | not_read; metadata_pending | `GPR_references\赵sir推荐\3-D characterization Klotzsche.pdf` | Duplicate/related 3D characterization reference | compare with top-level duplicate |
| background | not_read; metadata_pending | `GPR_references\DX203108.pdf` | Unknown from filename | inspect metadata before classification |

## Extraction Template

Use this template for each core paper after reading:

| Field | Notes |
| --- | --- |
| Citation |  |
| DOI/URL |  |
| Geometry | crosshole / on-ground / borehole-to-surface / synthetic / field |
| Domain | time / frequency / Laplace / mixed |
| Governing equation | Maxwell / vector wave / scalar approximation / other |
| Forward solver | FDTD / FEM / frequency-domain solver / surrogate |
| Inversion parameters | \(\epsilon_r\), \(\sigma\), \(v\), attenuation, source wavelet |
| Objective | L2 / normalized / envelope / source-independent / OT / robust |
| Gradient method | adjoint-state / sensitivity matrix / autodiff / surrogate |
| Optimizer | steepest descent / CG / L-BFGS / Gauss-Newton / truncated Newton / Adam |
| Regularization | Tikhonov / TV / structural / fractal / bounds / neural prior |
| Illumination handling | pseudo-Hessian / gradient normalization / none / unclear |
| Crosstalk handling | sequential / alternating / Hessian / scaling / constraints / unclear |
| Experiment setup | model size, grid, sources, receivers, frequency, noise |
| Main contribution |  |
| Limitations |  |
| Use for Part 1 | theory / method / experiment design / background |
