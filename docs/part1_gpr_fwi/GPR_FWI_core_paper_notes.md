# GPR-FWI Core Paper Notes

This file records verified or partially verified notes from local PDFs. `Skimmed` means only title page, abstract, and selected keyword contexts have been inspected.

## Meles et al. 2012

Status: skimmed from local PDF.

Citation:

Giovanni Angelo Meles, Stewart A. Greenhalgh, Alan G. Green, Hansruedi Maurer, and Jan van der Kruk, "GPR Full-Waveform Sensitivity and Resolution Analysis Using an FDTD Adjoint Method," IEEE Transactions on Geoscience and Remote Sensing, 50(5), 2012. DOI: `10.1109/TGRS.2011.2170078`.

Problem:

The paper addresses how to compute GPR full-waveform sensitivity functions and model resolution information efficiently. This matters because convergence in data space alone does not tell us which parts of an inverted permittivity/conductivity model are actually resolved.

Governing equation / solver:

- Uses time-domain GPR full-waveform modeling.
- The local PDF abstract states that the implementation uses finite-difference time-domain modeling.
- The key method is an adjoint approach for computing selected Jacobian/sensitivity values.

Objective / inversion context:

- The paper is mainly sensitivity and resolution analysis, not just a production inversion recipe.
- It is directly useful for understanding the Jacobian, approximate Hessian, illumination, and why gradients have spatially uneven reliability.

Parameters:

- Need to verify exact parameterization from the methods section.
- Filename and abstract context indicate GPR permittivity and conductivity distributions.

Gradient / sensitivity method:

- Time-domain adjoint method.
- Important for Part 1 because it links forward FDTD wavefields, receiver residuals/adjoint sources, and sensitivity kernels.

Use for Part 1:

- Core reference for the mathematical bridge between Maxwell/FDTD and adjoint sensitivity.
- Should be used in the gradient derivation and illumination/resolution discussion.

Still to extract:

- Exact discrete implementation details for the FDTD fields.
- Numerical model and acquisition settings.

Verified formula-level notes:

- The paper defines sensitivity functions as how a perturbation in \(\epsilon\) or \(\sigma\) at a model cell changes the measured electric field for a transmitter-receiver pair.
- The adjoint method reduces the computational burden from approximately \(M \times S\) forward simulations to \(S + R\) simulations, where \(M\) is the number of model parameters, \(S\) is the number of sources, and \(R\) is the number of receivers.
- The extracted kernel structure is:

\[
J_\epsilon \sim
\left\langle
\delta(\mathbf{x}-\mathbf{x}')
\partial_t \mathbf{E}^{s},
G^*\delta_i(\mathbf{x}-\mathbf{x}_r,t-t_\tau)
\right\rangle ,
\]

\[
J_\sigma \sim
\left\langle
\delta(\mathbf{x}-\mathbf{x}')
\mathbf{E}^{s},
G^*\delta_i(\mathbf{x}-\mathbf{x}_r,t-t_\tau)
\right\rangle .
\]

Interpretation:

- The \(\epsilon\) sensitivity correlates the adjoint receiver wavefield with the time derivative of the forward electric field.
- The \(\sigma\) sensitivity correlates the adjoint receiver wavefield with the forward electric field itself.
- This is closely related to the formal second-order PDE derivation in `formula_summary.md`, but the exact time derivative order depends on the first-order/vector wavefield formalism used by the paper.

Cost function and linear algebra:

- The paper uses a waveform least-squares cost over selected transmitters, receivers, and observation times.
- The Jacobian is defined as \(J_{\mu,\eta}=\partial d_\mu/\partial m_\eta\).
- The pseudo-Hessian is \(H_A = J^T J\).
- The gradient relation is \(\nabla_m S = J^T\Delta E\).
- A Gauss-Newton update can be written as:

\[
\mathbf{m}_{k+1}
=
\mathbf{m}_k
-
(J^T J + \lambda I)^{-1}J^T\Delta E.
\]

Illumination and resolution relevance:

- Near-source and near-receiver arrivals can dominate sensitivity.
- The paper explicitly discusses cumulative sensitivity and formal model resolution as tools for judging where the inversion is actually constrained.
- For our report, this is the cleanest local reference for explaining illumination imbalance from first principles.

Citation anchors:

- Local PDF page 9: Section IV-A introduces the cost function (12), Jacobian definition (13), and pseudo-Hessian/Taylor expansion around (15).
- Local PDF page 10: Section IV-B discusses cumulative sensitivity and formal model resolution; Section IV-C gives the Gauss-Newton update (20).
- Local PDF page 15: Appendix A gives cost-function, gradient, full Hessian, and pseudo-Hessian relations around (A-1)-(A-6).
- These are local PDF page indices and equation-number anchors; formal reporting should still verify printed page numbers visually.

## Busch et al. 2012

Status: skimmed from local PDF.

Citation:

Sebastian Busch, Jan van der Kruk, Jutta Bikowski, and Harry Vereecken, "Quantitative conductivity and permittivity estimation using full-waveform inversion of on-ground GPR data," Geophysics, 77(6), H79-H91, 2012. DOI: `10.1190/GEO2012-0045.1`.

Problem:

The paper targets quantitative estimation of permittivity and conductivity from on-ground common-midpoint GPR data. A key issue is that conventional ray-based or far-field assumptions do not provide reliable conductivity for near-surface on-ground GPR.

Governing equation / solver:

- Uses a 3D frequency-domain solution of Maxwell's equations for a horizontally layered subsurface.
- Models the time-domain electric field as source wavelet convolved with the Green's function, becoming multiplication in the frequency domain.

Objective / optimizer:

- Full-waveform inversion minimizes mismatch between observed and modeled data.
- The method uses a gradient-free optimization strategy rather than Jacobian/gradient-based minimization.

Parameters:

- Permittivity.
- Conductivity.
- Phase and amplitude of the source wavelet.

Important technical point:

Conductivity and source wavelet amplitude are coupled. The abstract notes that inaccurate conductivity starting models can lead to erroneous effective wavelet amplitudes and inversion results. This is important for our crosstalk/source-uncertainty section.

Use for Part 1:

- Core reference for on-ground quantitative dual-parameter inversion.
- Useful contrast against time-domain adjoint FWI because it solves a frequency-domain layered problem and avoids gradients.

Still to extract:

- Exact objective function.
- Search strategy details.
- Synthetic and measured CMP/waveguide model settings.

## Lavoue et al. 2014

Status: skimmed from local PDF.

Citation:

F. Lavoue, R. Brossier, L. Metivier, S. Garambois, and J. Virieux, "Two-dimensional permittivity and conductivity imaging by full waveform inversion of multioffset GPR data: a frequency-domain quasi-Newton approach," Geophysical Journal International, 197, 248-268, 2014. DOI: `10.1093/gji/ggt528`.

Problem:

The paper develops 2D frequency-domain FWI for simultaneous reconstruction of dielectric permittivity and electrical conductivity from multioffset GPR data.

Governing equation / solver:

- Frequency-domain GPR modeling.
- Need to extract exact PDE form from methods section.

Objective / optimizer:

- Quasi-Newton optimization.
- Uses L-BFGS-B to approximate inverse Hessian effects and handle parameter bounds.

Parameters:

- Dielectric permittivity.
- Electrical conductivity.

Regularization / scaling:

- Parameter scaling is central. The paper reports that dual-parameter inversion remains highly sensitive to scaling even with approximate Hessian information.
- A proper scaling should respect the natural data sensitivity; in their case, permittivity has stronger impact on data.
- Tikhonov regularization is used to prevent high-wavenumber conductivity artifacts that compensate for erroneous low-wavenumber structure.

Crosstalk relevance:

Very high. The paper directly shows why multiparameter inversion is not simply "update both gradients": scaling, bandwidth, Hessian approximation, and regularization control whether conductivity becomes an artifact sink.

Use for Part 1:

- Core reference for frequency-domain biparameter FWI.
- Important for crosstalk, Hessian/preconditioning, parameter scaling, and regularization discussion.

Still to extract:

- Experiment geometry and frequency sampling strategy.

Verified formula-level notes:

Frequency-domain TE formulation:

\[
\nabla^2 E_y(\omega,x,z)
+
\epsilon_e(\omega,x,z)\mu\omega^2 E_y(\omega,x,z)
=
\delta(x-x_s)\delta(z-z_s),
\]

with:

\[
\epsilon_e(\omega,x,z)=\epsilon(x,z)+i\sigma(x,z)/\omega.
\]

After finite-difference discretization:

\[
A(\omega,\epsilon,\sigma)u(\omega)=s(\omega).
\]

Objective:

\[
C(\mathbf{m})
=
\frac{1}{2}
\sum_{i=1}^{N_\omega}
\sum_{j=1}^{N_s}
\Delta d(\omega_i,s_j)^\dagger
\Delta d(\omega_i,s_j),
\]

where \(\Delta d=d_{obs}-d_{cal}\), and \(d_{cal}=Ru\).

Gradient:

The paper computes the gradient with the adjoint-state method:

\[
G_i(\mathbf{m})
=
\Re
\sum_{\omega,s}
u^T
\left(\frac{\partial A}{\partial m_i}\right)^T
v^*,
\]

where the adjoint wavefield satisfies \(A^\dagger v=R^\dagger\Delta d\). In their finite-difference scheme:

\[
\frac{\partial A_{ij}}{\partial \epsilon_i}
=
-\omega^2\delta_{ij},
\]

\[
\frac{\partial A_{ij}}{\partial \sigma_i}
=
-i\omega\delta_{ij}.
\]

Optimization:

- Model vector contains both permittivity and conductivity at each grid point.
- Update:

\[
\mathbf{m}_{k+1}
=
\mathbf{m}_k-\alpha_k B_k^{-1}G_k.
\]

- \(B_k\) is an L-BFGS-B approximation of Hessian information.
- Step length uses an inexact line search with Wolfe conditions.
- L-BFGS-B is useful because it approximates curvature while avoiding the cost of storing or inverting the full Hessian.

Parameter scaling:

The paper introduces relative permittivity and relative conductivity, then optimizes:

\[
(\epsilon_r,\sigma_r/\beta),
\]

where \(\beta\) is a dimensionless scaling factor controlling the relative update weight of conductivity versus permittivity.

Key interpretation:

- Small \(\beta\): conductivity update is penalized, giving smoother conductivity.
- Large \(\beta\): conductivity update is emphasized, but artifacts/instabilities may appear.
- Scaling is a reparameterization of the model space, not the same thing as gradient preconditioning.

Tikhonov regularization:

The regularized objective is:

\[
C(\mathbf{m})=C_D(\mathbf{m})+\lambda C_M(\mathbf{m}),
\]

with a conductivity model term:

\[
C_M(\mathbf{m})
=
\frac{1}{2}\sigma_r^T D\sigma_r,
\]

where \(D\) corresponds to a Laplacian operator. In their workflow, regularization is mainly applied to conductivity because conductivity is less constrained and more prone to high-wavenumber artifacts.

Crosstalk relevance:

- The paper explicitly argues that cascaded/alternating strategies may fail because conductivity contrasts can imprint both amplitude and phase, contaminating the first permittivity step.
- Conversely, small errors in permittivity systematically map into conductivity artifacts.
- This is a direct expression of multiparameter trade-off/crosstalk.

Workflow:

For realistic surface-to-surface acquisition with partial illumination, different \(\beta\) values can yield similar data misfit but very different conductivity models. Their proposed workflow is:

1. Run inversions for multiple parameter scalings \(\beta\) and regularization weights \(\lambda\).
2. Plot final data misfit versus \(\beta\) for each \(\lambda\).
3. Prefer small \(\lambda\) values for which the misfit curve has a clear minimum versus \(\beta\).

Important distinction:

- Parameter scaling guides the inversion path according to relative data sensitivity.
- Regularization constrains the conductivity model and suppresses high-wavenumber structures that can compensate for incorrect low-wavenumber conductivity.

Citation anchors:

- Local PDF page 3: Section 2.1 introduces the frequency-domain forward problem; Section 2.2 introduces the inverse problem, objective (6), quasi-Newton update (7), adjoint gradient (8), and L-BFGS-B setup.
- Local PDF pages 5-7: Section 3.1 discusses parameter sensitivity and trade-off; eq. (11) is used in the trade-off discussion.
- Local PDF page 7: Section 3.2 introduces parameter scaling \((\epsilon_r,\sigma_r/\beta)\) and the model/gradient vector around (12).
- Local PDF page 8: Hessian/scaling structure is discussed around (15)-(16), including the role of \(\beta\) in Hessian blocks.
- Local PDF page 9: Conductivity Tikhonov regularization and the regularization-gradient contribution are discussed around (17)-(20).
- Local PDF pages 15-16: Scaling/regularization sweep is used to identify reasonable \(\beta\) and \(\lambda\) ranges.
- Local PDF pages 18-19: Discussion and conclusion emphasize that robust biparameter reconstruction requires both parameter scaling and regularization.
- These are local PDF page indices and equation-number anchors; formal reporting should still verify printed page numbers visually.

## Sun et al. 2024

Status: skimmed from local PDF.

Citation:

Jian Sun, Ying Liu, Yuzhao Lin, Lei Xing, and Huaishan Liu, "Implicit multiparameter full waveform inversion of multioffset ground penetrating radar data," Geophysical Journal International, 240, 904-919, 2024/2025 issue context. DOI: `10.1093/gji/ggae420`.

Problem:

Classical GPR-FWI is ill-posed and sensitive to initial models. The paper proposes implicit full waveform inversion (IFWI) for GPR by representing multiple subsurface parameters with a neural network.

Governing equation / solver:

- Need to verify exact forward solver and whether the main examples are time-domain FDTD.

Objective / optimizer:

- Physical data misfit is optimized through neural-network parameters.
- The network is treated as a continuous implicit representation of subsurface parameters.

Parameters:

- Multiple GPR parameters, likely permittivity and conductivity. Need exact confirmation from method section.

Regularization / prior:

- The implicit neural representation acts as a parameterization prior.
- The paper emphasizes the frequency principle/spectral bias of neural networks: low-frequency structure is learned before fine detail.

Crosstalk relevance:

- Important but must be evaluated carefully. Neural parameterization may reduce some artifacts by imposing a smooth-to-detailed learning path, but it does not remove the underlying multiparameter sensitivity coupling unless the data/objective/parameterization makes the Jacobian blocks better conditioned.

Use for Part 1:

- Core modern reference for implicit/neural reparameterization.
- Useful for comparing classical regularization with neural priors.

Still to extract:

- Exact parameter mapping and network architecture.
- Objective formula.
- Claimed improvement over FWI/multiscale FWI.
- Experiment setup and crosstalk evidence.

## Liu et al. 2022

Status: skimmed from local PDF.

Citation:

Xintong Liu, Sixin Liu, Chaopeng Luo, Hejun Jiang, Hong Li, Xu Meng, and Zhihui Feng, "Source-Independent Waveform Inversion Method for Ground Penetrating Radar Based on Envelope Objective Function," Remote Sensing, 14, 4878, 2022. DOI: `10.3390/rs14194878`.

Problem:

FWI needs an accurate source wavelet. Deconvolution-based wavelet estimation can be inaccurate and operator-intensive. The paper proposes a source-independent waveform inversion scheme for cross-hole GPR and combines it with an envelope objective to reduce nonlinearity.

Governing equation / solver:

- Cross-hole GPR.
- Need to extract exact forward equation and discretization.

Objective:

- Source-independent waveform inversion.
- Envelope objective function.
- Multiscale strategy through time-domain convolutions/frequency-band decomposition.

Parameters:

- Need to verify exact inverted parameters.

Gradient:

- The residual field used to construct the gradient inherits envelope-wavefield characteristics.
- This is relevant to cycle skipping and source-wavelet uncertainty.

Use for Part 1:

- Core reference for source uncertainty, envelope objective, and time-domain multiscale strategy.

Still to extract:

- Exact source-independent objective formula.
- Envelope residual and adjoint source formula.
- Cross-hole model settings and frequency ranges.

Verified formula-level notes:

Let \(E_{i,j}\) denote simulated data for source \(i\) and receiver \(j\), and let \(E^{obs}_{i,j}\) denote observed data. The paper builds source-independent residuals by cross-convolving modeled and observed traces against reference traces. The envelope objective is:

\[
S
=
\frac{1}{2}
\sum_i^{n_s}
\sum_j^{n_r}
\left\|
E_c - E_c^{obs}
\right\|^2 ,
\]

where:

\[
E_c
=
\sqrt{
(E_{i,j} * E^{obs}_{i,k})^2
+
H(E_{i,j} * E^{obs}_{i,k})^2
},
\]

\[
E_c^{obs}
=
\sqrt{
(E^{obs}_{i,j} * E_{i,k})^2
+
H(E^{obs}_{i,j} * E_{i,k})^2
}.
\]

Here:

- \(*\) is convolution.
- \(H\) is the Hilbert transform.
- \(k\) is a reference trace index.

The gradient derivation rewrites the variation of the envelope objective into cross-correlation residual sources. The final reported structure is:

\[
\begin{bmatrix}
\nabla S_\epsilon(\mathbf{x}')
\\
\nabla S_\sigma(\mathbf{x}')
\end{bmatrix}
=
\sum_s
\begin{bmatrix}
\int_0^T dt'
(\partial_t E(\mathbf{x}',t'))\cdot
\sum_d\sum_\tau T_{s,d,\tau}(\mathbf{x}',t')
\\
\int_0^T dt'
E(\mathbf{x}',t')\cdot
\sum_d\sum_\tau T_{s,d,\tau}(\mathbf{x}',t')
\end{bmatrix},
\]

where \(T\) is a backward-propagated residual field constructed from two merged residual sources, \(r_{12}\) and \(r_{34}\).

Update method:

- The paper updates permittivity and conductivity simultaneously with conjugate-gradient directions:

\[
\epsilon_{k+1}(\mathbf{x})
=
\epsilon_k(\mathbf{x})
-
\zeta_{\epsilon,k} C_{\epsilon,k}(\mathbf{x}),
\]

\[
\sigma_{k+1}(\mathbf{x})
=
\sigma_k(\mathbf{x})
-
\zeta_{\sigma,k} C_{\sigma,k}(\mathbf{x}).
\]

Use for Part 1:

- This paper is useful for explaining how objective-function design changes the adjoint source. The physical forward/adjoint propagation is still wave-equation based, but the residual injected backward is no longer a simple waveform residual.

## Ernst et al. 2007

Status: skimmed from local PDF.

Citation:

Jacques R. Ernst, Hansruedi Maurer, Alan G. Green, and Klaus Holliger, "Full-Waveform Inversion of Crosshole Radar Data Based on 2-D Finite-Difference Time-Domain Solutions of Maxwell's Equations," IEEE Transactions on Geoscience and Remote Sensing, 45(9), 2007.

Problem:

The paper is an early core reference for crosshole GPR full-waveform inversion using 2D FDTD solutions of Maxwell's equations. It argues that ray tomography uses only a small part of the radar trace and is limited to larger-scale features.

Governing equation / solver:

- Time-domain Maxwell equations.
- 2D finite-difference time-domain forward modeling.
- Crosshole transmitter/receiver geometry.

Parameters:

- Dielectric permittivity.
- Electrical conductivity.

Experiment setup:

- Synthetic crosshole data.
- Increasingly complex models: isolated subwavelength objects, adjacent subwavelength objects, heterogeneous layered media, water-filled tunnels, and closely spaced pipes.
- Nominal borehole radar frequency range discussed as 20-250 MHz.

Key contribution:

- Demonstrates that crosshole GPR-FWI can reconstruct subwavelength dielectric/conductive objects more accurately than ray tomography under favorable conditions.
- Shows robustness to uncorrelated noise in synthetic examples.

Limitations:

- Small resistive bodies and closely spaced dielectric objects can be difficult to resolve.
- Electrical property contrasts may be underestimated.
- Some configurations approach the resolution limits of the inversion.

Use for Part 1:

- Important historical time-domain baseline.
- Supports the report's claim that FWI exploits more waveform information than ray tomography but still has resolution and nonuniqueness limits.

## Meng et al. 2019

Status: formula-checked from local PDF.

Citation:

Xu Meng, Sixin Liu, Yi Xu, and Lei Fu, "Application of Laplace Domain Waveform Inversion to Cross-Hole Radar Data," Remote Sensing, 11, 1839, 2019. DOI: `10.3390/rs11161839`.

Problem:

Time-domain FWI is highly nonlinear and needs an adequate initial model. Conventional ray-based initial models have shortcomings. This paper uses Laplace-domain waveform inversion to produce smoother initial models for subsequent time-domain FWI.

Forward problem:

Starting from time-domain Maxwell equations, the paper writes a Laplace-domain system:

\[
\tilde{M}\tilde{E}=\tilde{J},
\]

where \(\tilde{M}\) is the Maxwell operator in the Laplace domain. The Green operator is:

\[
\tilde{G}=\tilde{M}^{-1}.
\]

Objective:

The paper uses a logarithmic objective because Laplace-domain electric fields can be small:

\[
O(\epsilon,\sigma)
=
\frac{1}{2}
\sum_{n_s}
\sum_{n_r}
\left[
\ln \tilde{E}(\epsilon,\sigma)
-
\ln \tilde{E}^{obs}
\right]^2.
\]

Gradient:

The residual-like term is:

\[
r
=
\frac{\ln\tilde{E}-\ln\tilde{E}^{obs}}{\tilde{E}}.
\]

The virtual sources are:

\[
\tilde{v}_{\epsilon}=s\tilde{E},
\quad
\tilde{v}_{\sigma}=\tilde{E}.
\]

The gradient structure is:

\[
\begin{bmatrix}
\nabla O_\epsilon\\
\nabla O_\sigma
\end{bmatrix}
=
\sum_{n_s}
\begin{bmatrix}
(s\tilde{E})\tilde{G}r\\
(\tilde{E})\tilde{G}r
\end{bmatrix}.
\]

Optimization:

- Permittivity and conductivity are updated with conjugate-gradient directions.
- The paper uses a stepped update strategy because of large gradient differences between permittivity and conductivity.
- Parameters are updated in the logarithmic domain to keep positive values and improve convergence.

Use for Part 1:

- Provides a concrete Laplace-domain contrast to time-domain FWI.
- Supports the idea that Laplace-domain inversion is primarily useful for smooth initial model building rather than replacing detailed time-domain FWI.

## Hunziker et al. 2025

Status: skimmed from local PDF.

Citation:

Jurg Hunziker, Giovanni Meles, and Niklas Linde, "Crosshole ground-penetrating radar full-waveform inversion by combining optimal-transport and least-squares distances," Journal of Applied Geophysics, 237, 105655, 2025.

Problem:

Least-squares FWI can converge to a local minimum if the starting model is not close enough. The paper proposes a crosshole GPR-FWI strategy that uses optimal-transport distance early and switches to least-squares distance once the model is close enough.

Objective strategy:

- Use optimal transport in early iterations because it has a broader basin of attraction for common GPR-FWI problems.
- Switch to least squares when many traces are shifted by less than half a period relative to observed traces.

Gradient strategy:

- Computes gradients explicitly at random master points and interpolates the remaining model gradient.
- Avoids placing master points close to antennas, reducing extreme near-antenna gradient values.
- The sparse master-point gradient also smooths the model without an explicit model-regularization term.

Use for Part 1:

- Important modern objective-function reference.
- Useful for the cycle-skipping section and for distinguishing objective design from regularization.
