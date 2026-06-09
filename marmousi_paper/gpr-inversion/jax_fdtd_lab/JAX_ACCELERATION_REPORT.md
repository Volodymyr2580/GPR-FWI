# JAX 加速版 GPR FDTD 框架阶段报告

更新时间：2026-05-20

## 1. 背景与目标

原 `gpr-inversion` 中最耗时的部分是二维 GPR 时域 FDTD 正演和伴随梯度计算。当前工作在 `jax_fdtd_lab/` 中建立了一个隔离实验框架，用 JAX 重写底层数值路径，并以原 CPU/Numba 实现生成的正演数据、波场和伴随梯度作为正确性基准。

目标不是立刻替换主程序，而是先验证：

- JAX 正演是否能复现原 CPU/Numba 正演。
- JAX 保存的 `Ey` 波场是否能对齐原程序波场。
- JAX 版伴随梯度是否能复现原 `compute_gradient` 输出。
- 在 WSL2 + NVIDIA GPU 下是否有实际速度和显存优势。

## 2. 当前实现

新增和扩展的核心文件：

- `cases.py`：构造 tiny 和 smooth validation cases。
- `jax_forward.py`：JAX 版 2D TEz FDTD 正演，使用 `jax.lax.scan` 表达时间循环，使用 `jax.vmap` 表达多炮并行。
- `jax_adjoint.py`：JAX 版原始伴随梯度，复刻 `gpr-inversion` 的 `reverse_time_loop + forward_wavefield` 梯度公式。
- `cpu_baseline.py`：调用原 CPU/Numba 正演、波场保存和 `compute_gradient` 作为基准。
- `benchmark.py`：统一输出 correctness、runtime 和 GPU memory 统计。
- `test_fdtd_lab.py`：包含正演、波场、梯度和 step-count 回归测试。

需要特别说明：直接使用 `jax.grad` 得到的梯度与原 `compute_gradient` 并不是同一个梯度定义。前者是对离散 JAX 正演代码做自动微分；后者是原项目中的手写伴随梯度。因此当前用于对齐原程序的主路线是 `jax_adjoint.py`，而不是直接用 `jax.grad` 替代原伴随公式。

## 3. 正确性验证

当前 JAX prototype 已通过以下验证：

- Windows `gprfwi`：`9 tests OK`。
- WSL2 `gpr-jax311`：`9 tests OK`，JAX backend 为 `gpu`，device 为 `cuda:0`。
- tiny case 下，JAX `Ey` 波场与原 CPU/Numba 波场逐点对齐。
- tiny case 下，JAX 伴随梯度与原 `compute_gradient` 的 epsilon / sigma 梯度对齐。
- expanded cases 下，接收器正演数据、epsilon 梯度和 sigma 梯度均保持有限值，无 NaN/Inf。

一次重要 bug 修复：

原 JAX 子波构造使用过 `np.arange(0, steps * dt, dt)`。在 `steps=500` 时，由于浮点步长累计误差，JAX 子波长度会变成 501，导致 JAX 输出时间维度和 CPU baseline 不一致。现已改为：

```python
time_axis = np.arange(case.steps, dtype=np.float64) * case.dt
```

这样可以保证时间采样点数严格等于 `steps`。

## 4. 性能结果

测试环境：

- WSL2 Ubuntu-22.04
- Python 3.11.15 venv：`/home/volodymyr/venvs/gpr-jax311`
- JAX / jaxlib：`0.10.0`
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB
- benchmark 中设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免 JAX 一开始预占大块显存。

当前加速对比的 CPU baseline 是单进程 CPU/Numba，不是 MPI 并行 CPU。

| Case | Forward relative L2 | JAX forward speedup | JAX adjoint speedup | Peak JAX pool |
| --- | ---: | ---: | ---: | ---: |
| `40x80, steps=120, 4 shots, 16 receivers` | `4.35e-7` | `13.1x` | `3.18x` | `~130 MB` |
| `60x120, steps=160, 6 shots, 24 receivers` | `7.40e-7` | `36.7x` | `10.0x` | `~514 MB` |
| `100x200, steps=500, 10 shots, 50 receivers` | `1.66e-6` | `46.5x` | `79.0x` | `~2.56 GB` |
| `100x200, steps=1000, 40 shots, 100 receivers, batch=5` | `1.85e-6` | `239x` | `56.4x` | `~1.13 GB` |

full-ish case 的梯度误差：

- epsilon gradient relative L2：`3.85e-6`
- sigma gradient relative L2：`1.31e-5`

这些误差来自 float32 JAX GPU 与原 CPU/Numba baseline 的对比，当前量级可接受。

## 5. MPI 对比的解释

当前表格中的加速倍数是：

```text
JAX GPU vs 单进程 CPU/Numba
```

不是：

```text
JAX GPU vs 20 进程 MPI CPU
```

因此，小 case 上的 `13.1x` 不应被理解为 JAX 相对现有 MPI 方案的最终收益。原 CPU 代码的 MPI 主要按 shot 并行，如果使用 20 个 MPI 进程，并且 shot 数足够多，小 case 上完全可能追上或超过 JAX GPU。

不过在 full-ish case 上，单进程 CPU forward 为 `39.8 s`，JAX forward 为 `0.166 s`。即使用理想的 20 进程线性加速粗略估算：

```text
CPU MPI ideal forward ~= 39.8 / 20 = 1.99 s
JAX forward ~= 0.166 s
```

JAX 仍可能有约 `12x` 的正演优势。梯度部分，单进程 CPU gradient 为 `55.5 s`，JAX batched adjoint 为 `0.984 s`；理想 20 进程折算后，JAX 仍可能有约 `2.8x` 的优势。

这只是估算。严格结论需要补测：

```text
CPU serial Numba
CPU MPI 20 processes
JAX GPU batched
```

## 6. 显存与当前限制

伴随梯度需要用到正演波场。若一次性保存所有炮的完整波场，显存会随以下量增长：

```text
n_shots * n_steps * (nx + 2*npml) * (nz + 2*npml)
```

在 `100x200, steps=1000, 40 shots` 下，一次性 materialize 全量 JAX 波场 `[40, 1000, 120, 220]` 会触发 OOM，其中一次转置临时分配约 `3.93 GiB`。因此当前 benchmark 增加了：

- `--skip-wavefield`：大规模时不回传全量波场用于展示对比。
- `--shot-batch-size`：按炮分批计算并累加梯度，例如 batch size 为 5。

在 `100x200, steps=1000, 40 shots, 100 receivers, batch=5` 下，JAX adjoint 可完整跑通，peak JAX pool 约 `1.13 GB`。

## 7. 结论与下一步

当前 JAX 框架已经在隔离 lab 中完成了从 tiny 到 full-ish case 的正确性和性能验证。以原 `gpr-inversion` 的正演波场和 `compute_gradient` 为基准，JAX 版正演和 JAX 版原始伴随梯度均能对齐，且在 GPU 上表现出明显加速。

建议下一步：

1. 保持现有 CPU/Numba 为默认 backend。
2. 新增显式 backend 选择，例如 `cpu_numba` 与 `jax_gpu`。
3. 正式接入时默认使用 shot batching，避免一次性保存全量波场。
4. 补充 MPI CPU benchmark，给出更公平的三方对比。
5. 在真实 OverThrust residual 和训练 loop 中做 smoke test，再决定是否替换主训练路径。
