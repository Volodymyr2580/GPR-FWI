# Experiment Migration Matrix

This table records the intended migration order. Old folders remain untouched
and are treated as source material until each line is migrated into
`gpr-inversion`.

| Area | Experiment line | New config/runner status |
| --- | --- | --- |
| OverThrust Unet | mode2, eps then sig, Adam, optional illumination | Supported via `configs/overthrust_mode2_unet_eps_then_sig.yaml` |
| traditional | Marmousi mode1 LBFGS eps-only | Supported via `configs/marmousi_mode1_traditional_eps_lbfgs.yaml` |
| traditional | Marmousi mode1 RMSprop eps-only | Supported via `configs/marmousi_mode1_traditional_eps_rmsprop.yaml` |
| traditional | Marmousi mode2 LBFGS eps-only | Supported via `configs/marmousi_mode2_traditional_eps_lbfgs.yaml` |
| traditional | Marmousi mode2 RMSprop eps-only | Supported via `configs/marmousi_mode2_traditional_eps_rmsprop.yaml` |
| unet | Marmousi mode1 fixed shallow layer, no illumination | Supported via `configs/marmousi_mode1_unet_eps_adam.yaml` |
| unet | Marmousi mode1 fixed shallow layer, illumination | Supported via `configs/marmousi_mode1_unet_eps_adam_illumination.yaml` |
| unet | Marmousi mode2 fixed shallow layer, no illumination | Supported via `configs/marmousi_mode2_unet_eps_adam.yaml` |
| unet | Marmousi mode2 fixed shallow layer, illumination | Supported via `configs/marmousi_mode2_unet_eps_adam_illumination.yaml` |
| OverThrust Traditional | mode1/mode2 single and two-parameter variants | Planned |
| OverThrust Unet | mode1/mode2 eps-first, sigma-later, pretrain variants | Planned |

Use this command to see what is currently connected to the unified runner:

```powershell
$env:PYTHONPATH="src"
conda run -n gprfwi python -m gpr_inversion.run --list-supported
conda run -n gprfwi python -m gpr_inversion.run --list-planned
conda run -n gprfwi python -m gpr_inversion.run --check-configs
```
