# Example command for the migrated OverThrust Mode2 UNet eps-then-sig experiment.
# Run this from the gpr-inversion folder in the conda gprfwi environment.

$env:PYTHONPATH = "src"
mpirun -np 20 conda run -n gprfwi python -m gpr_inversion.run `
  --config configs\overthrust_mode2_unet_eps_then_sig.yaml
