# JAX FDTD Lab 阶段报告

更新时间：2026-05-20

## 1. 结论

JAX 版时域 FDTD 原型已经完成第一轮闭环验证：

- WSL2 中 JAX CUDA 环境已可用。
- JAX 能识别 RTX 4090 Laptop GPU，`jax.default_backend()` 返回 `gpu`。
- tiny 2D GPR FDTD case 的 JAX GPU 正演输出与现有 CPU/Numba baseline 对齐。
- tiny epsilon 和 sigma 方向导数的 JAX 自动微分检查已通过。
- JAX 版原始伴随梯度已经能复现 `gpr-inversion` 的 `compute_gradient` 输出。

当前结果说明：继续沿 JAX 路线推进是可行的。下一步重点应从“能不能跑 GPU”转向“更大网格下是否稳定、是否更快、梯度是否可用于反演”。

## 2. 环境状态

### Windows `gprfwi`

- Python: `D:\anaconda\envs\gprfwi\python.exe`
- JAX: `0.10.0`
- JAX backend: `cpu`
- 用途：CPU JAX correctness tests。

### WSL2 `gpr-jax311`

- WSL distro: `Ubuntu-22.04`
- venv: `/home/volodymyr/venvs/gpr-jax311`
- Python: `3.11.15`
- JAX: `0.10.0`
- jaxlib: `0.10.0`
- JAX backend: `gpu`
- JAX device: `cuda:0`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU, 16GB

验证命令：

```bash
python - <<'PY'
import jax
import jaxlib

print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())

x = jax.numpy.ones((2048, 2048))
y = x @ x
print(y.block_until_ready()[0, 0])
PY
```

关键输出：

```text
jax 0.10.0
jaxlib 0.10.0
backend gpu
devices [CudaDevice(id=0)]
2048.0
```

JAX 在 WSL 中会打印：

```text
Could not get kernel mode driver version
```

当前这只是 WSL/JAX 读取驱动版本格式时的 warning。因为后续已经显示 `backend gpu` 和 `CudaDevice(id=0)`，所以 GPU backend 实际可用。

## 3. 已完成代码

新增 lab 原型文件：

- `cases.py`：构造 tiny FDTD case。
- `cpu_baseline.py`：调用现有 CPU/Numba FDTD 作为正确性基线。
- `jax_forward.py`：JAX 版 2D TEz FDTD 正演原型。
- `jax_adjoint.py`：JAX 版原始伴随梯度原型，对齐 `compute_gradient`。
- `compare_forward.py`：CPU vs JAX 正演对比命令行入口。
- `gradient_check.py`：JAX epsilon 方向导数有限差分检查。
- `test_fdtd_lab.py`：lab unittest。

当前 JAX 正演已经使用：

- `jax.numpy` 表达数组计算。
- `jax.lax.scan` 表达时间循环。
- `jax.vmap` 表达多炮并行。
- JAX 内部计算 `ca/cb` 材料系数，使 `epsilon` 和 `sigma` 可进入自动微分路径。

## 4. 验证结果

### Unit tests

WSL `(gpr-jax311)` 中运行：

```bash
export PYTHONPATH="src:."
python -m unittest jax_fdtd_lab.test_fdtd_lab -v
```

结果：

```text
Ran 5 tests in 24.291s
OK
```

### 24-step CPU vs JAX GPU

WSL `(gpr-jax311)` 中运行：

```bash
python -m jax_fdtd_lab.compare_forward --steps 24 --backend both
```

关键结果：

```text
JAX backend: gpu
JAX device: cuda:0
CPU shape: (2, 4, 24)
JAX shape: (2, 4, 24)
max_abs_error: 9.231650705032735e-05
relative_l2_error: 2.50268312121555e-07
finite: true
```

这个误差水平对 float32 JAX vs float64 CPU baseline 来说是合理的。相对 L2 误差约 `2.50e-7`，说明当前 tiny 正演路径高度一致。

### Gradient check

Windows `gprfwi` 和 WSL2 `gpr-jax311` 中已完成 epsilon/sigma 方向导数检查：

```text
Windows epsilon relative error: 5.94e-4
Windows sigma relative error: 6.85e-3
WSL GPU epsilon relative error: 1.49e-3
WSL GPU sigma relative error: 5.03e-3
```

这说明 JAX 自动微分路径在 tiny case 上初步可信。后续仍需扩展到更多方向和更长 steps。

### Original baseline gradient comparison

新增 JAX 版原始伴随梯度路径，直接复刻 `gpr-inversion` 的 `reverse_time_loop + forward_wavefield` 梯度公式，而不是直接使用 `jax.grad` 的离散自动微分结果。

WSL `(gpr-jax311)` 中 80-step tiny case 对比结果：

```text
wavefield relative L2: 5.23e-7
epsilon gradient relative L2: 1.18e-6
sigma gradient relative L2: 1.18e-6
```

这说明在“以原 `gpr-inversion` 波场和梯度为正确基准”的标准下，当前 JAX prototype 已经能对齐 tiny case。

## 6. Expanded Validation And Benchmark

新增 `benchmark.py`，用于同时记录 correctness、runtime 和 GPU memory。JAX benchmark 默认使用 `jax.jit`，并区分首次编译时间和后续稳态运行时间。

关键修复：

- JAX 子波时间轴改为 `np.arange(steps) * dt`，避免浮点 `np.arange(0, steps*dt, dt)` 在 `steps=500` 时生成 501 个采样点。
- 大规模 benchmark 增加 `--skip-wavefield`，避免一次性取回全量 JAX 波场导致 OOM。
- JAX adjoint benchmark 增加 `--shot-batch-size`，按炮分批累加梯度，控制显存。

WSL2 RTX 4090 Laptop GPU 结果：

| Case | Forward relative L2 | JAX forward speedup | JAX adjoint speedup | Peak JAX pool |
| --- | ---: | ---: | ---: | ---: |
| `40x80, steps=120, 4 shots, 16 receivers` | `4.35e-7` | `13.1x` | `3.18x` | `~130 MB` |
| `60x120, steps=160, 6 shots, 24 receivers` | `7.40e-7` | `36.7x` | `10.0x` | `~514 MB` |
| `100x200, steps=500, 10 shots, 50 receivers` | `1.66e-6` | `46.5x` | `79.0x` | `~2.56 GB` |
| `100x200, steps=1000, 40 shots, 100 receivers, batch=5` | `1.85e-6` | `239x` | `56.4x` | `~1.13 GB` |

The full-ish `100x200, 1000 steps, 40 shots, 100 receivers` run used `--skip-wavefield` because all-shot wavefield materialization would require a large `[40, 1000, 120, 220]` array and triggered an OOM during a `3.93 GiB` temporary allocation. This confirms the integration path should use shot batching or checkpointing instead of saving all shot wavefields at once.

## 5. 环境处理记录

关键环境问题和处理：

- Windows 原生 JAX GPU 不支持，因此 Windows `gprfwi` 只作为 CPU JAX correctness 环境。
- WSL 默认 Python 3.10 曾安装到旧版 `jax==0.6.2`，该版本没有 `cuda13` extra，只能 CPU。
- WSL Python 3.11 venv 最终由用户手动完成 JAX CUDA 安装，并通过 GPU device probe。
- `mpi4py` 在 WSL venv 中会寻找系统 `libmpi.so`。当前 lab 不需要 MPI，因此卸载 `mpi4py` 后原代码会自动退回 serial CPU baseline。

对初学者来说：

- `mpi4py` 是 Python 调 MPI 多进程的包。
- `libmpi.so` 是 Linux 系统里的 MPI 动态库。
- 当前测试只需要单进程 CPU baseline，所以不装 MPI 是可以的。

## 6. 下一步建议

优先做三件事：

1. 增加 benchmark 脚本。
   - 记录 CPU/Numba、JAX CPU、JAX GPU 的运行时间。
   - 分清 JAX 第一次编译时间和后续运行时间。

2. 扩展 correctness cases。
   - 增加 `40x80` 或 `50x100` 网格。
   - 增加 80/200 steps 对比。
   - 检查 receiver waveform、max abs error、relative L2 error、NaN/Inf。

3. 扩展 gradient validation。
   - 多个 epsilon perturbation directions。
   - sigma directional gradient check。
   - 更长 steps 下的显存和稳定性记录。

在这些完成前，不建议把 JAX backend 接入正式 `gpr_inversion.run`。当前仍应保持 `jax_fdtd_lab/` 隔离。
