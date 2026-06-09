# JAX FDTD Lab Progress

更新时间：2026-05-20

## 当前状态

Planning initialized. A tiny CPU baseline runner, JAX forward prototype, JAX adjoint-gradient prototype, WSL GPU JAX environment, and epsilon/sigma gradient-check scaffold now exist.

本目录目前用于记录和验证 JAX/CUDA 时域 FDTD 改写计划。已经建立 tiny case、CPU baseline 调用、JAX forward prototype、JAX 版原始伴随梯度 prototype、命令行对比入口和 unittest。Windows `gprfwi` 环境可做 CPU JAX correctness tests；WSL2 `gpr-jax311` 环境已确认 JAX CUDA 可用，`jax.default_backend()` 返回 `gpu`，设备为 `cuda:0`。

## 已完成决策

| 决策 | 当前选择 | 理由 |
| --- | --- | --- |
| 第一阶段 backend | JAX | 开发速度快，支持 GPU JIT 和自动微分，适合先验证时域 FDTD 数学路径。 |
| CUDA 顺序 | JAX 跑通后再尝试 | 手写 CUDA 性能上限更高，但开发难度和调试成本更高。 |
| 目标环境 | Windows `gprfwi` 做 CPU correctness；WSL2 `gpr-jax311` 做 GPU validation | Windows 原生 JAX GPU 不支持；WSL2 能看到 RTX 4090 并已跑通 JAX CUDA。 |
| 试验位置 | `jax_fdtd_lab/` | 和现有稳定源码隔离，跑通后再迁入 `src/gpr_inversion`。 |
| `jaxwell` 使用 | 不直接依赖 | `jaxwell` 是频域 FDFD，本项目需要时域 GPR FDTD。 |
| 正确性基线 | 当前 CPU/Numba FDTD | GPU 版必须先和现有结果对齐。 |

## 里程碑进度

| 阶段 | 状态 | 验证结果 | 阻塞 | 下一步 |
| --- | --- | --- | --- | --- |
| Phase 0: CPU baseline freeze | Completed for tiny case | tiny case CPU forward 可运行，8/24/80-step 输出均 finite；80-step relative L2 约 `2.50e-7` | 尚未保存长期 baseline 数据文件 | 后续需要时再保存 `.npz` 固定基线。 |
| Phase 1: JAX forward prototype | Completed for small/full-ish validation cases | JAX CPU backend 已在 `gprfwi` 跑通；WSL2 JAX GPU backend 已跑通；`100x200, 1000 steps, 40 shots, 100 receivers` forward relative L2 约 `1.85e-6` | 尚未接入正式 runner | 设计 backend 接口并保持 CPU 默认路径。 |
| Phase 2: JAX gradient validation | In progress | JAX 版原始伴随梯度和 illumination 修正后梯度均已对齐 CPU 路径；`100x200, 1000 steps, 40 shots, 100 receivers` batched sigma gradient relative L2 约 `1.31e-5` | 当前需要 shot batching 控制显存；尚未做长 epoch 训练循环 | 扩展到真实 residual、优化 batching 和 memory benchmark。 |
| Phase 3: integration into `gpr_inversion` | In progress | `twopara_epsfirst.py` 已可调用 JAX forward、illumination、batched adjoint gradient，并保留 CPU 默认路径 | 服务器/WSL 工作树需要同步最新补丁后再跑完整 smoke | 在 GPU 环境运行 `--fdtd-backend jax --illumination` 真实 OverThrust smoke。 |
| Phase 4: CUDA exploration | Not started | 无 | 依赖 JAX backend 稳定和性能数据 | 根据热点决定是否写 CUDA/Triton/extension。 |

## 当前待办

- 2026-05-22 与老板讨论后的实验路线记录：
  - 后续实验统一采用 JAX 框架作为主要 FDTD backend。
  - 第一步：继续检查 sigma 梯度求解，以及加入 U-Net 网络参数化后反传链路是否稳定。
  - 第二步：给 sigma 的 U-Net 恢复浅层两个 skip connection，保证浅层细节和梯度通路。
  - 第三步：继续测试 `eps first then sig` 路线，固定上方 50 层，只反演中深部；stage 1 固定 sigma，仅反演 epsilon，使用 L1 data loss；stage 2 epsilon 和 sigma 双参数同时反演，使用 L2 data loss。
  - 第四步：后续用 RL 或 Optuna 搜索最优超参数，评估指标使用反演模型相对于真实模型的 MSE 和 SSIM。
- 固化 forward comparison 容差：tiny 8/24-step `relative_l2_error < 1e-5`；80-step 参考目标 `relative_l2_error < 1e-5`。
- 扩展 JAX gradient check：更多 epsilon/sigma 方向、不同 steps。
- 扩展原始 baseline 对比：更多 residual、更多 source/receiver 布局、更大网格。
- 针对正式 `100x200, 1000 steps, 40 shots, 100 receivers`，继续优化 JAX adjoint batching，避免一次性保存全量 `[shots, steps, nx, nz]` 波场。
- 增加 runtime benchmark，分别记录 CPU/Numba、JAX CPU、JAX GPU 的 warmup 与 steady-state 时间。
- 准备 larger tiny case，例如 `40x80` 或 `50x100`，验证 GPU 加速是否开始显现。
- 正式训练入口新增 `--fdtd-backend {numpy,jax}`；`jax` backend 已支持 illumination 累计和 illumination 修正后的伴随梯度。默认按 5 炮分批，可用环境变量 `JAX_FDTD_SHOT_BATCH_SIZE` 调整。

## 验证记录

| 日期 | 验证项 | 命令或方法 | 结果 |
| --- | --- | --- | --- |
| 2026-05-20 | 文档计划初始化 | 创建 `plan.md`、`AGENTS.md`、`progress.md` | Passed |
| 2026-05-20 | `gprfwi` 环境检查 | `conda run -n gprfwi python -c "... importlib.util.find_spec(...)"` | Python 位于 `D:\anaconda\envs\gprfwi\python.exe`；`torch=True`，`mpi4py=True`，`jax=False` |
| 2026-05-20 | lab unittest | `conda run -n gprfwi python -m unittest jax_fdtd_lab.test_fdtd_lab -v` | Passed: 3 ok, 1 skipped because JAX is not installed |
| 2026-05-20 | CPU/JAX smoke CLI | `conda run -n gprfwi python -m jax_fdtd_lab.compare_forward --steps 8 --backend both` | CPU baseline finite, shape `(2, 4, 8)`；JAX skipped with clear missing-dependency reason |
| 2026-05-20 | 安装 CPU JAX | `conda run -n gprfwi python -m pip install --upgrade jax` | Installed `jax==0.10.0`, `jaxlib==0.10.0` |
| 2026-05-20 | JAX unittest | `conda run -n gprfwi python -m unittest jax_fdtd_lab.test_fdtd_lab -v` | Passed: 4 ok |
| 2026-05-20 | 8-step CPU vs JAX | `conda run -n gprfwi python -m jax_fdtd_lab.compare_forward --steps 8 --backend both` | max abs error `2.54e-6`; relative L2 `1.16e-7`; JAX backend `cpu` |
| 2026-05-20 | 24-step CPU vs JAX | `conda run -n gprfwi python -m jax_fdtd_lab.compare_forward --steps 24 --backend both` | max abs error `7.71e-5`; relative L2 `2.04e-7`; JAX backend `cpu` |
| 2026-05-20 | 80-step CPU vs JAX | `conda run -n gprfwi python -m jax_fdtd_lab.compare_forward --steps 80 --backend both` | max abs error `9.46e-4`; relative L2 `2.50e-7`; JAX backend `cpu` |
| 2026-05-20 | GPU/WSL capability check | `nvidia-smi`; `wsl -d Ubuntu-22.04 -e sh -lc "nvidia-smi"` | Windows and WSL2 both see RTX 4090 Laptop GPU; WSL2 is preferred for JAX CUDA validation |
| 2026-05-20 | WSL default Python JAX attempt | `wsl -d Ubuntu-22.04 -e python3 -m pip install --user --upgrade "jax[cuda13]"` | Installed `jax==0.6.2` CPU only; pip warned that this version does not provide `cuda13`; `jax.devices()` returned CPU only |
| 2026-05-20 | WSL Python 3.11 venv attempt | Created `/mnt/c/tmp/gpr-jax311`; attempted `pip install --upgrade "jax[cuda13]"` | Install exceeded 15-minute timeout; lingering pip process was stopped; venv currently contains only pip/setuptools |
| 2026-05-20 | JAX epsilon gradient check | `conda run -n gprfwi python -c "from jax_fdtd_lab.gradient_check import ..."` | loss `1.15e-7`; analytic dir deriv `-4.586e-6`; finite diff `-4.592e-6`; relative error `1.26e-3` |
| 2026-05-20 | WSL runtime check | `wsl -d Ubuntu-22.04 -e sh -lc "pwd; python3 --version; python3.11 --version; nvidia-smi ..."` | WSL runs from project path `/mnt/e/...`; Python 3.10.12 and 3.11.15 available; RTX 4090 Laptop GPU visible |
| 2026-05-20 | WSL Linux-home venv CUDA JAX attempt | Created `/home/volodymyr/venvs/gpr-jax311`; attempted `pip install --upgrade "jax[cuda13]"` | Install exceeded 20-minute timeout; process was stopped; venv currently contains only pip/setuptools |
| 2026-05-20 | WSL JAX CUDA verification | User ran JAX device probe in `(gpr-jax311)` | `jax==0.10.0`, `jaxlib==0.10.0`, backend `gpu`, devices `[CudaDevice(id=0)]`, matrix multiply returned `2048.0` |
| 2026-05-20 | WSL JAX lab unittest | `python -m unittest jax_fdtd_lab.test_fdtd_lab -v` in WSL `(gpr-jax311)` | Passed: 5 ok. CPU baseline ran serial after `mpi4py` uninstall. |
| 2026-05-20 | WSL 24-step CPU vs JAX GPU | `python -m jax_fdtd_lab.compare_forward --steps 24 --backend both` | JAX backend `gpu`, device `cuda:0`; max abs error `9.23e-5`; relative L2 `2.50e-7`; output finite |
| 2026-05-20 | Windows JAX lab unittest after sigma gradient check | `conda run -n gprfwi python -c "import sys, unittest; sys.path[:0] = ['src', '.']; unittest.main(module='jax_fdtd_lab.test_fdtd_lab', verbosity=2)"` | Passed: 6 ok |
| 2026-05-20 | WSL JAX lab unittest after sigma gradient check | `wsl -d Ubuntu-22.04 -e env PYTHONPATH=... /home/volodymyr/venvs/gpr-jax311/bin/python -m unittest jax_fdtd_lab.test_fdtd_lab -v` | Passed: 6 ok；JAX backend `gpu` |
| 2026-05-20 | WSL 80-step CPU vs JAX GPU | `python -m jax_fdtd_lab.compare_forward --steps 80 --backend both` in WSL `(gpr-jax311)` | JAX backend `gpu`, device `cuda:0`; max abs error `9.46e-4`; relative L2 `2.43e-7`; output finite |
| 2026-05-20 | WSL epsilon/sigma gradient check | `run_epsilon_directional_gradient_check` and `run_sigma_directional_gradient_check` in WSL `(gpr-jax311)` | epsilon relative error `1.49e-3`; sigma relative error `5.03e-3`; both finite |
| 2026-05-20 | Windows JAX lab unittest after original wavefield/gradient baseline tests | `conda run -n gprfwi python -c "import sys, unittest; sys.path[:0] = ['src', '.']; unittest.main(module='jax_fdtd_lab.test_fdtd_lab', verbosity=2)"` | Passed: 8 ok；新增 JAX wavefield vs original CPU wavefield、JAX adjoint gradient vs original CPU `compute_gradient` |
| 2026-05-20 | WSL JAX lab unittest after original wavefield/gradient baseline tests | `wsl -d Ubuntu-22.04 -e env PYTHONPATH=... /home/volodymyr/venvs/gpr-jax311/bin/python -m unittest jax_fdtd_lab.test_fdtd_lab -v` | Passed: 8 ok；JAX backend `gpu` |
| 2026-05-20 | WSL 80-step original baseline wavefield/gradient comparison | custom one-line Python probe in WSL `(gpr-jax311)` | wavefield max abs `8.33e-4`, wavefield relative L2 `5.23e-7`; eps grad relative L2 `1.18e-6`; sigma grad relative L2 `1.18e-6` |
| 2026-05-20 | Windows JAX lab unittest after 500-step wavelet fix | `conda run -n gprfwi python -c "import sys, unittest; sys.path[:0] = ['src', '.']; unittest.main(module='jax_fdtd_lab.test_fdtd_lab', verbosity=2)"` | Passed: 9 ok；修复 `np.arange(0, steps*dt, dt)` 在 `steps=500` 生成 501 个采样点的问题 |
| 2026-05-20 | WSL JAX lab unittest after benchmark work | `python -m unittest jax_fdtd_lab.test_fdtd_lab -v` in WSL `(gpr-jax311)` | Passed: 9 ok；JAX backend `gpu` |
| 2026-05-20 | WSL benchmark `40x80, steps=120, 4 shots, 16 receivers` | `python -m jax_fdtd_lab.benchmark --case smooth --xl 40 --zl 80 --steps 120 --sources 4 --receivers 16 --repeats 3` | forward relative L2 `4.35e-7`; JAX forward `13.1x` faster than CPU steady forward；JAX adjoint total `3.18x` faster than CPU gradient-only；peak pool about `130 MB` |
| 2026-05-20 | WSL benchmark `60x120, steps=160, 6 shots, 24 receivers` | `python -m jax_fdtd_lab.benchmark --case smooth --xl 60 --zl 120 --steps 160 --sources 6 --receivers 24 --repeats 3` | forward relative L2 `7.40e-7`; JAX forward `36.7x`; JAX adjoint total `10.0x`; peak pool about `514 MB` |
| 2026-05-20 | WSL benchmark `100x200, steps=500, 10 shots, 50 receivers` | `python -m jax_fdtd_lab.benchmark --case smooth --xl 100 --zl 200 --steps 500 --sources 10 --receivers 50 --repeats 2` | forward relative L2 `1.66e-6`; JAX forward `46.5x`; JAX adjoint total `79.0x`; peak pool about `2.56 GB` |
| 2026-05-20 | WSL full-ish benchmark with shot batching | `python -m jax_fdtd_lab.benchmark --case smooth --xl 100 --zl 200 --steps 1000 --sources 40 --receivers 100 --repeats 1 --shot-batch-size 5 --skip-wavefield` | forward relative L2 `1.85e-6`; eps gradient relative L2 `3.85e-6`; sigma gradient relative L2 `1.31e-5`; JAX forward `239x`; batched JAX adjoint total `56.4x`; peak pool about `1.13 GB` |
| 2026-05-20 | Full-ish all-shot wavefield materialization attempt | `python -m jax_fdtd_lab.benchmark --case smooth --xl 100 --zl 200 --steps 1000 --sources 40 --receivers 100 --shot-batch-size 5` | Failed before batched gradient: all-shot JAX wavefield comparison tried to materialize `[40, 1000, 120, 220]` and OOM at a `3.93 GiB` temporary allocation；benchmark now supports `--skip-wavefield` for large cases |
| 2026-05-20 | Windows JAX lab unittest after L1/network checks | `conda run -n gprfwi python -c "import sys, unittest; sys.path[:0] = ['src', '.']; unittest.main(module='jax_fdtd_lab.test_fdtd_lab', verbosity=2)"` | Passed: 12 ok；新增 L1 residual backward、sigma L1 finite-difference、network-parameterized sigma gradient checks |
| 2026-05-20 | Windows JAX local sanity checks | `conda run -n gprfwi python -m jax_fdtd_lab.run_sanity_checks --steps 8` | JAX backend `cpu`; L1 zero-subgradient check exact；sigma L2 relative error `4.71e-3`; sigma L1 relative error `3.44e-3`; network sigma L2 relative error `4.53e-2`; network sigma L1 relative error `1.02e-2` |
| 2026-05-20 | OverThrust sigma U-Net shallow skip/output-size smoke check | one-line `UNet(... output_size=(50,200))` import/forward/backward probe | Eps/Sig outputs both shape `(50, 200)` for fixed top 50；sigma U-Net with shallow skip connections receives parameter gradients |
| 2026-05-20 | Stage-specific loss argument check | `parse_args(['--fixed-top-rows','50','--stage1-data-loss','l1','--stage2-data-loss','l2'])` and Optuna launcher parse probe | `twopara_epsfirst` now accepts fixed top 50 with stage1 L1/stage2 L2；Optuna launcher defaults to stage1 L1/stage2 L2 |
| 2026-05-21 | Server JAX GPU sanity checks | `python -B -m jax_fdtd_lab.run_sanity_checks --steps 8` in `gpr-jax-cu121` | JAX `0.6.2`, backend `gpu`, devices `cuda:0..3`; epsilon L2 relative error `1.49e-3`; sigma L2 `8.00e-3`; sigma L1 `6.40e-3`; network sigma L1 `7.02e-3`; all finite |
| 2026-05-21 | Server JAX GPU tiny benchmark | `python -B -m jax_fdtd_lab.benchmark --case tiny --steps 80 --repeats 1` | forward relative L2 `2.86e-7`; wavefield relative L2 `5.23e-7`; eps gradient relative L2 `1.04e-6`; sigma gradient relative L2 `1.12e-6`; JAX forward `2.23x`; JAX adjoint gradient `3.33x`; GPU backend sees 4 CUDA devices |
| 2026-05-21 | Server JAX GPU validation bundle | `bash scripts/run_jax_gpu_validation.sh` | Saved validation outputs under `outputs/jax_fdtd_lab/server_validation`; sanity backend `gpu`; tiny forward relative L2 `2.86e-7`, eps grad `1.04e-6`, sigma grad `1.12e-6`; smooth `40x80, 120 steps, 4 shots, 16 receivers` forward relative L2 `4.37e-7`, wavefield `9.45e-7`, eps grad `1.25e-6`, sigma grad `1.16e-6`, JAX forward `10.0x`, JAX adjoint `8.87x`; GPU memory stayed below `466 MB` |
| 2026-05-21 | Initial JAX backend integration smoke | `ForwardModelFunction.apply(..., fdtd_backend='jax')` on tiny 80-step case | PyTorch custom autograd can call JAX forward and hand-written JAX adjoint backward; illumination compensation is supported by JAX `sum(Ey^2)` accumulation; data shape `(2,4,80)`; eps/sigma gradients finite and non-zero; existing `jax_fdtd_lab` unittest still `12 ok` |
| 2026-05-22 | JAX illumination integration and tests | `conda run -n gprfwi powershell -NoProfile -Command "$env:PYTHONPATH='src'; python -m unittest jax_fdtd_lab.test_fdtd_lab -q"` | Passed: 14 ok；新增 JAX illumination vs CPU illumination、分批 illumination 一致性、illumination 修正后 eps/sigma 梯度对齐测试 |
| 2026-05-22 | `twopara_epsfirst.py` JAX illumination entry smoke | direct call to `_jax_forward_with_illumination_numpy` and `_jax_adjoint_gradient_numpy` on 24-step tiny case | Passed；入口返回 data `(2,4,24)`、illumination `(20,40)`、eps/sigma gradient `(20,40)`；本地训练入口无残留 “JAX FDTD backend does not support --illumination” guard |
| 2026-05-22 | JAX device and memory controls | `parse_args(['--fdtd-backend','jax','--device','cuda:0','--jax-device','cuda:1','--jax-shot-batch-size','1'])`; `python -m unittest jax_fdtd_lab.test_fdtd_lab -q` | Passed: 14 ok；训练入口新增 `--jax-device` 和 `--jax-shot-batch-size`，并在 JAX backend 下默认设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| 2026-05-22 | Multi-GPU MPI controls | `parse_args(['--fdtd-backend','jax','--device','auto','--jax-device','auto','--auto-jax-shot-batch-size','--max-jax-shot-batch-size','5'])`; `bash -n scripts/run_overthrust_jax_top50_multigpu_sweep.sh`; `python -m unittest jax_fdtd_lab.test_fdtd_lab -q` | Passed: 14 ok；新增 MPI rank 自动选 GPU、按空闲显存估计初始 JAX shot batch size、OOM 时自动减半重试，以及多 GPU sweep 脚本 |
| 2026-05-22 | Training-path JIT cache optimization | tiny 24-step probe for cached forward illumination and cached adjoint; `python -m unittest jax_fdtd_lab.test_fdtd_lab -q` | Passed: 14 ok；训练入口使用 cached `jax.jit` kernels，tiny probe forward first/second `3.30s/0.002s`，adjoint first/second `2.56s/0.004s`；说明后续 epoch 可复用编译结果 |
| 2026-05-25 | Optuna model-quality objective | `compute_model_quality_metrics` identity check; `bash -n scripts/run_overthrust_model_quality_optuna.sh`; `python -m unittest jax_fdtd_lab.test_fdtd_lab -q` | Passed: 14 ok；训练 metrics 新增 eps/sigma full 与 active 区域 MSE、NMSE、SSIM；新增 `overthrust_optuna_model_quality_search.py`，默认用 active 区域 `NMSE + (1-SSIM)` 作为 Optuna objective |
| 2026-05-25 | Stage1 checkpoint reuse for Optuna | `bash -n scripts/run_overthrust_jax_stage1_eps_checkpoint.sh`; `build_command(... --stage1-checkpoint-path ckpt.pt --num-epochs-stage2 2500)`; `python -m unittest jax_fdtd_lab.test_fdtd_lab -q` | Passed: 14 ok；新增 stage1-only checkpoint 脚本；Optuna model-quality launcher 支持加载 `STAGE1_CHECKPOINT_PATH` 后直接 resume 到 stage2，避免每个 trial 重跑 eps-first 5000 epoch |
| 2026-05-25 | Align model-quality Optuna search space to previous `optuna_eps_then_sig` | Read `outputs/overthrust/optuna_eps_then_sig/trials_summary.jsonl`; checked new `build_command` output and bash syntax | 新 Optuna 保持旧搜索参数：固定 `learning_rate_eps=1e-4`，搜索 `lr_eps_stage2 1e-6..1e-4`、`learning_rate_sig 1e-6..1e-4`、`alpha_tv_eps 1e-2..5e-2`、`alpha_tv_sig 3e-2..4e-2`、`alpha_l1_data 0..0.5`；保留当前新实验必要差异：JAX backend、checkpoint resume、stage2 2500、默认 fixed-top 20 |

## 技术说明

JAX first 并不表示放弃 CUDA。它表示先用更高层、更容易调试的 GPU 数值框架验证物理和梯度。等 JAX 版能稳定复现 CPU 结果后，再用 JAX/CPU 双基线去约束 CUDA 原型，这样排错成本会低很多。

L1 data loss 的局部反传可以理解为“残差的符号”传回去：残差为正传正号，残差为负传负号。当前 Windows `gprfwi` 环境里的 JAX 默认 `grad(abs(x))` 在 `x == 0` 处选择 `+1`，因此 lab 中新增了显式 zero-subgradient 的 `l1_data_loss`，让精确零残差处的梯度为 0，便于和常见次梯度约定对齐。
