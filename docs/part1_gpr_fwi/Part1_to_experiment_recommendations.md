# Part 1 To Experiment Recommendations

本文把第一部分理论整理转化为后续实验计划。核心原则是：先验证物理和梯度，再诊断双参数 crosstalk，最后比较正则化、照明补偿和网络重参数化。

## 1. 最小可验证实验

Purpose:

验证 GPR-FWI 的最小闭环是否正确。

Model:

- 2D 小网格，例如 `80 x 120` 或 `100 x 200`。
- 背景 \(\epsilon_r=4\)，\(\sigma=0.003\) S/m。
- 放一个简单块状或圆形 \(\epsilon_r\) 异常。
- 先不放 \(\sigma\) 异常，或固定 \(\sigma\) 为背景。

Setup:

- Time-domain FDTD。
- Ricker source。
- 先用少量 source/receiver。
- L2 waveform objective。
- 只反演 \(\epsilon_r\)。

Checks:

- `d_obs` 和 `d_syn` shape 正确。
- loss 初始值非零。
- adjoint gradient 有合理空间分布。
- 做 finite-difference gradient check：

\[
\frac{\Phi(m+h\delta m)-\Phi(m-h\delta m)}{2h}
\approx
\langle \nabla\Phi(m), \delta m\rangle.
\]

Expected:

- loss 应能下降。
- \(\epsilon_r\) 异常位置应有可解释更新。
- 如果 gradient check 不通过，先不要进入复杂实验。

Recommended local landing path:

- 首选：`marmousi_paper/gpr-inversion/`。理由是该目录已经有 `configs/`、`src/`、`tests/` 和安全输出目录设计，适合把实验做成可恢复、可对比的配置化任务。
- 参考：`Gpr_fwi/ReadMe.md`。理由是它已经清楚写出了最小 FWI 流程需求：输入网格、源函数、真实模型、初始模型、计算 `d_obs`、再从初始模型迭代更新。
- 暂不建议直接从复杂 UNet/IFWI 线路开始，因为那会把梯度、优化器、网络先验、数据几何和正则化全部混在一起。

First config sketch: `minimal_eps_l2_gradient_check`

这份配置的目的不是追求最终图像好看，而是验证三件事：正演数据维度正确、伴随梯度方向正确、一次或少量迭代能让 L2 loss 沿下降方向变化。

```yaml
experiment:
  name: minimal_eps_l2_gradient_check
  purpose: verify_forward_adjoint_gradient_before_large_inversion
  output_policy: create_unique_run_directory

domain:
  dimension: 2
  nx: 80
  nz: 120
  dx: 0.02
  dz: 0.02
  nt: 800
  dt: 4.0e-11
  npml: 10
  air_layer: false

source:
  type: ricker
  center_frequency_hz: 4.0e8
  source_count: 3
  source_depth_index: 12
  source_x_indices: [20, 40, 60]

receivers:
  mode: common_receiver_array
  receiver_depth_index: 12
  receiver_x_indices: [10, 20, 30, 40, 50, 60, 70]
  component: Ez

true_model:
  epsilon_r_background: 4.0
  sigma_background_s_per_m: 0.003
  epsilon_r_anomaly:
    shape: circle
    center_index: [40, 60]
    radius_cells: 10
    value: 6.0
  sigma_anomaly: none

initial_model:
  epsilon_r: homogeneous_background
  epsilon_r_value: 4.0
  sigma: fixed_background
  sigma_value_s_per_m: 0.003

inversion:
  parameters: [epsilon_r]
  objective: l2_waveform
  optimizer: steepest_descent_or_lbfgs
  max_iterations: 3
  pml_gradient_mask: true
  gradient_smoothing: none
  illumination_compensation: none

gradient_check:
  enabled: true
  parameter: epsilon_r
  perturbation_type: random_roi
  roi_exclude_pml: true
  step_sizes: [1.0e-2, 3.0e-3, 1.0e-3]
  comparison:
    finite_difference: central
    tolerance_relative: 1.0e-2

diagnostics:
  save_loss_curve: true
  save_observed_data: true
  save_synthetic_data: true
  save_initial_model: true
  save_gradient_snapshot: true
  save_update_snapshot: true
```

最关键的检查公式是：

\[
\frac{\Phi(m+h\delta m)-\Phi(m-h\delta m)}{2h}
\approx
\langle \nabla\Phi(m), \delta m\rangle.
\]

如果多个 \(h\) 下两边数量级和符号都不一致，应先检查 residual 符号、伴随源注入时间顺序、PML mask、\(\epsilon\) 与 \(\epsilon_r\) 的链式法则，以及 staggered grid 上梯度和参数是否位于同一网格位置。

## 2. Crosstalk 诊断实验

Purpose:

分离 \(\epsilon_r\) 和 \(\sigma\) 的敏感性，观察双参数 trade-off。

Experiment A: fixed \(\sigma\), invert \(\epsilon_r\)

- \(\sigma\) 固定为真值或可信背景。
- 只更新 \(\epsilon_r\)。
- 观察相位和走时是否改善。

Experiment B: fixed \(\epsilon_r\), invert \(\sigma\)

- \(\epsilon_r\) 固定为真值。
- 只更新 \(\sigma\)。
- 观察振幅衰减是否改善。

Experiment C: simultaneous \(\epsilon_r,\sigma\)

- 两个参数同时更新。
- 比较不同 learning rate / scaling。

Experiment D: wrong fixed parameter contamination

- 用错误 \(\sigma\) 固定，反演 \(\epsilon_r\)。
- 用错误 \(\epsilon_r\) 固定，反演 \(\sigma\)。

Expected:

- 错误 \(\epsilon_r\) 很可能导致 \(\sigma\) artifact。
- 错误 \(\sigma\) 会影响振幅和 source-wavelet 等效估计，进而污染 \(\epsilon_r\)。
- simultaneous inversion 需要 scaling，否则 \(\sigma\) 可能吸收错误。

## 3. Illumination Compensation 实验

Purpose:

判断梯度不均匀来自物理照明还是优化器问题。

Variants:

- Raw gradient。
- Gradient smoothing。
- PML mask。
- Pseudo-Hessian / illumination normalization。
- Source/receiver coverage change。
- Shot contribution heatmap。

Metrics:

- loss 曲线。
- gradient energy map。
- update energy map。
- anomaly contribution energy。
- left/right or shallow/deep ROI recovery contrast。

Expected:

- 强照明区域 raw gradient 更大。
- illumination normalization 可能让更新更均衡，但不能凭空恢复无照明区域。
- 如果某异常体数据贡献明显更弱，长训练不一定能解决，需要 acquisition/weighting 或先验。

## 4. Objective Function 对比实验

Purpose:

比较不同目标函数对 cycle skipping、source uncertainty 和振幅误差的影响。

Variants:

- L2 waveform。
- Normalized waveform。
- Correlation-based objective。
- Source-independent objective。
- Envelope objective。
- Source-independent + envelope objective。

Control:

- 同一真值模型。
- 同一初始模型。
- 同一 source/receiver geometry。
- 同一 optimizer。

Expected:

- L2 最简单，但对初始模型和 source wavelet 敏感。
- Normalized/correlation objective 对振幅误差更鲁棒。
- Envelope objective 对大尺度结构更友好，但精细结构可能需要后续 waveform objective。
- Source-independent objective 适合 source wavelet 不可信的场景，但 residual 构造更复杂。

## 5. Regularization 实验

Purpose:

区分“数据拟合下降”和“模型恢复可信”。

Variants:

- No regularization。
- Tikhonov on \(\epsilon_r\)。
- Tikhonov on \(\sigma\)。
- TV on \(\epsilon_r\) 或 \(\sigma\)。
- Bound constraints。
- Parameter scaling \((\epsilon_r,\sigma_r/\beta)\)。

Recommended sweep:

- \(\beta \in \{0.1, 0.2, 0.5, 1.0\}\)。
- \(\lambda \in \{0, 1e-6, 1e-5, 1e-4\}\)，具体数量级按代码中参数尺度调整。

Expected:

- \(\sigma\) 更需要正则化。
- 太强 Tikhonov 会过度平滑。
- TV 更利于块状异常，但可能产生 staircasing。
- 不同 \(\beta\) 可能给出相似 data misfit 但不同 \(\sigma\) 模型，因此必须看模型指标和可视化。

## 6. Network / Implicit Representation 对比实验

Purpose:

判断 UNet 或 implicit representation 是真正改善反演，还是只是改变模型先验。

Variants:

- Traditional grid parameterization。
- UNet parameterization。
- SIREN / FR-INR implicit parameterization。
- Dropout pretraining。
- True inversion dropout with small rate or fixed mask。

Control:

- 必须和 traditional baseline 使用同一观测数据、初始模型、source/receiver geometry。
- 先单参数，再双参数。

Expected:

- 网络重参数化可能让模型更平滑，降低高频噪声。
- 也可能限制可表达结构，导致过平滑。
- Dropout 在物理反演主循环中可能引入随机模型抖动，应谨慎使用 fixed mask 或小 dropout。

## 7. 推荐执行顺序

1. `minimal_eps_l2_gradient_check`
2. `eps_only_vs_sig_only_sensitivity`
3. `twopara_scaling_beta_sweep`
4. `twopara_tikhonov_sigma_sweep`
5. `illumination_raw_vs_normalized`
6. `objective_l2_vs_normalized_vs_envelope`
7. `traditional_vs_unet_vs_ifwi`

每个实验都应保存：

- config。
- command。
- loss curve。
- final synthetic data。
- initial/final model。
- gradient snapshot。
- key metrics。
- 简短结论。

这和 `隐式FWI/progress.md` 里的记录方式一致，后续可以沿用。
