# Marmousi 实验线迁移与运行说明

本文档说明 `gpr-inversion` 中 Marmousi 实验线的整理结果、目录结构、配置文件、统一 runner 的使用方式，以及后续修改参数时应该看哪里。

## 当前状态

Marmousi 的 8 条核心 epsilon-only 实验线已经接入统一 runner：

| 配置文件 | 原始来源 | 新状态 |
| --- | --- | --- |
| `configs/marmousi_mode1_traditional_eps_lbfgs.yaml` | `traditional/mode1_LBFGS/main.py` | supported |
| `configs/marmousi_mode1_traditional_eps_rmsprop.yaml` | `traditional/mode1_RMSprop/main.py` | supported |
| `configs/marmousi_mode2_traditional_eps_lbfgs.yaml` | `traditional/mode2_LBFGS/marmousi.py` | supported |
| `configs/marmousi_mode2_traditional_eps_rmsprop.yaml` | `traditional/mode2_RMSprop/marmousi.py` | supported |
| `configs/marmousi_mode1_unet_eps_adam.yaml` | `unet/mode1_unet_fix/gpr_train_mpi.py` | supported |
| `configs/marmousi_mode1_unet_eps_adam_illumination.yaml` | `unet/mode1_unet_fix_illu/gpr_train_mpi.py` | supported |
| `configs/marmousi_mode2_unet_eps_adam.yaml` | `unet/mode2_unet_fix/gpr_train_mpi.py` | supported |
| `configs/marmousi_mode2_unet_eps_adam_illumination.yaml` | `unet/mode2_unet_fix_illu/gpr_train_mpi.py` | supported |

这里的 `supported` 表示：配置文件能被统一入口 `python -m gpr_inversion.run --config ...` 识别，并能分发到对应的新代码模块。

## 给初学者的概念说明

`runner` 可以理解成“总开关”。以前每个实验都要进不同文件夹运行不同脚本，现在统一从 `gpr_inversion.run` 进入，再由配置文件决定跑哪条实验线。

`config` 是实验参数表。它是 YAML 文本文件，用来记录模型、采集模式、优化器、学习率、输出目录等。以后优先改 `configs/*.yaml`，不要一上来就在大型 Python 脚本里搜索硬编码参数。

`package` 是 Python 可导入的代码包。`src/gpr_inversion` 下面的目录都属于这个项目包，所以可以用 `python -m gpr_inversion.run` 这种方式运行。

`MPI` 是多进程并行运行方式。这个项目的 FDTD 正演和梯度计算很重，MPI 用来把多个炮点分给多个进程计算。

`UNet` 是一种神经网络结构。这里不是直接优化每个 epsilon 网格点，而是让 UNet 输出一个 epsilon 更新量，再通过 FDTD 损失反传训练网络。

`Tikhonov` 是正则化项。通俗说，它给模型加一个“不要过度抖动”的约束，避免反演结果太不平滑。

`illumination` 是照明补偿相关路径。它会根据波场能量形成补偿项，使不同区域的梯度更新更均衡。

## 新目录结构

核心目录如下：

```text
gpr-inversion/
  configs/
    marmousi_*.yaml
  src/gpr_inversion/
    run.py
    config.py
    common/
    io/
    models/
    visualization/
    experiments/
      marmousi/
        traditional_eps_only/
        unet_eps_only/
```

`traditional_eps_only/` 放传统数值反演线。

```text
traditional_eps_only/
  runner.py
  mode1/
  mode2_lbfgs/
  mode2_rmsprop/
```

`unet_eps_only/` 放 UNet 重参数化反演线。

```text
unet_eps_only/
  mode1/
  mode1_illumination/
  mode2/
  mode2_illumination/
```

这些目录保留了旧代码里确实不同的数值核。也就是说，目前迁移优先保证行为接近旧实验；后续如果要进一步抽象公共逻辑，可以在这之上继续整理。

## 运行前准备

从 `gpr-inversion` 目录运行：

```powershell
cd E:\sci_research\GPR\marmousi_paper\gpr-inversion
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
```

`PYTHONPATH=src` 的作用是告诉 Python：项目源码在 `src` 目录下。否则 Python 可能找不到 `gpr_inversion` 这个包。

`PYTHONDONTWRITEBYTECODE=1` 的作用是尽量避免生成新的 `__pycache__` 文件。它不影响实验计算结果。

推荐使用已有 conda 环境：

```powershell
conda run -n gprfwi python -m gpr_inversion.run --check-configs
```

`conda run -n gprfwi` 表示在名为 `gprfwi` 的环境里运行命令。这个环境通常包含 `torch`、`numba`、`mpi4py` 等数值计算依赖。

## 查看实验状态

查看所有已接入 runner 的实验：

```powershell
$env:PYTHONPATH="src"
conda run -n gprfwi python -m gpr_inversion.run --list-supported
```

查看是否还有只登记但未接线的配置：

```powershell
$env:PYTHONPATH="src"
conda run -n gprfwi python -m gpr_inversion.run --list-planned
```

当前 Marmousi 8 条核心线已全部 supported，`--list-planned` 应显示没有 planned config。

检查所有配置文件是否合法：

```powershell
$env:PYTHONPATH="src"
conda run -n gprfwi python -m gpr_inversion.run --check-configs
```

## Dry-run 检查

`--dry-run` 不会启动完整 FDTD 反演，只会打印统一配置被翻译成了哪些参数。适合在正式跑实验前检查路径、学习率、epoch、输出目录是否正确。

示例：

```powershell
$env:PYTHONPATH="src"
conda run -n gprfwi python -m gpr_inversion.run --config configs\marmousi_mode2_unet_eps_adam.yaml --dry-run
```

如果输出里看到类似下面的信息，说明 runner 已经能分发：

```text
Translated legacy module arguments:
--device cuda:0 --data-path ... --output-dir ... --learning-rate-eps ...
```

## 正式运行实验

传统 mode1 LBFGS：

```powershell
$env:PYTHONPATH="src"
mpirun -np 20 conda run -n gprfwi python -m gpr_inversion.run --config configs\marmousi_mode1_traditional_eps_lbfgs.yaml
```

传统 mode2 RMSprop：

```powershell
$env:PYTHONPATH="src"
mpirun -np 20 conda run -n gprfwi python -m gpr_inversion.run --config configs\marmousi_mode2_traditional_eps_rmsprop.yaml
```

UNet mode1 无照明：

```powershell
$env:PYTHONPATH="src"
mpirun -np 20 conda run -n gprfwi python -m gpr_inversion.run --config configs\marmousi_mode1_unet_eps_adam.yaml
```

UNet mode2 照明：

```powershell
$env:PYTHONPATH="src"
mpirun -np 20 conda run -n gprfwi python -m gpr_inversion.run --config configs\marmousi_mode2_unet_eps_adam_illumination.yaml
```

完整反演非常耗时，也可能占用 GPU。建议正式运行前先用 `--dry-run` 检查配置。

## 配置字段怎么改

常用字段：

```yaml
runtime:
  device: cuda:0
  output_dir: outputs/marmousi/mode2_unet_eps_adam

training:
  learning_rate_eps: 1e-4
  num_epochs_stage1: 2001
  offset: 0.0

regularization:
  tikhonov:
    eps: 0.01

fixed_top_rows: 10
snapshot_interval: 1000
```

`runtime.device` 控制使用哪个设备。`cuda:0` 表示第一张 GPU；如果没有 GPU，代码会回退到 CPU。

`runtime.output_dir` 控制结果保存位置。

`training.learning_rate_eps` 是 epsilon 相关学习率。

`training.num_epochs_stage1` 是训练轮数。调试时可以临时改成很小，比如 `1` 或 `2`。

`regularization.tikhonov.eps` 是 epsilon 的 Tikhonov 正则权重。

`fixed_top_rows` 是固定浅层行数。固定浅层表示这些顶部行不参与更新，保持真实或初始浅层结构。

`snapshot_interval` 控制保存 `.npy` 快照的间隔。

## 输出安全规则

旧脚本里有 `shutil.rmtree` 这类清空结果目录的逻辑。迁移后不再使用这种删除行为。

新逻辑是：

```text
outputs/marmousi/mode2_unet_eps_adam
outputs/marmousi/mode2_unet_eps_adam_run001
outputs/marmousi/mode2_unet_eps_adam_run002
```

如果目标目录已经存在，程序会自动创建带后缀的新目录，不会删除旧实验结果。

## 迁移时保留的差异

traditional：

- mode1 LBFGS 和 RMSprop 共享 mode1 数值核。
- mode2 LBFGS 和 mode2 RMSprop 的旧数值核不同，因此分别保留。

UNet：

- mode1 无照明和 mode1 照明保留为两个入口。
- mode2 无照明和 mode2 照明保留为两个入口。
- mode2 线保留旧脚本中的 `alpha_eps=0.1` 默认行为。
- mode2 illumination 保留旧脚本中的 `fixed_top_rows=20` 配置。

## 已验证内容

本次迁移完成后已做轻量验证：

```powershell
$env:PYTHONPATH="src"
python -m gpr_inversion.run --check-configs
python -m gpr_inversion.run --list-supported
python -m gpr_inversion.run --list-planned
python -m unittest tests\test_config_and_helpers.py
```

并在 `gprfwi` conda 环境中验证了 4 条 Marmousi UNet 迁移模块可以导入并解析参数。

没有运行完整反演，因为完整 FDTD/MPI/UNet 训练耗时长、资源占用高。正式实验前建议先对目标配置执行 `--dry-run`。
