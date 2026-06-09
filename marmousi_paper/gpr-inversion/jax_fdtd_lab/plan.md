# JAX FDTD Lab Plan

更新时间：2026-05-20

## 1. 目标

本目录用于规划和验证一个新的 GPU-accelerated GPR 时域电磁反演框架。第一阶段只做 JAX 版 2D time-domain FDTD 原型，确认正演结果、梯度和性能都可信之后，再考虑手写 CUDA backend。

这里的核心目标不是改写论文实验脚本的外壳，而是把最耗时的底层数值计算搬到 GPU 上：

```text
epsilon/sigma 模型 -> FDTD 正演 -> d_syn 合成数据 -> loss -> gradient -> 反演更新
```

对初学者来说：

- FDTD 是 finite-difference time-domain，意思是按时间步一帧一帧推进电磁波场，天然适合 GPR 雷达道数据。
- JAX 是一个 Python 数值计算框架，可以把数组计算编译到 GPU，并支持自动微分。
- CUDA 是 NVIDIA GPU 的底层编程接口，性能上限更高，但开发和调试难度也更高。

## 2. 核心决策

本项目不直接使用 `jaxwell` 作为底层求解器。原因是 `jaxwell` 面向 frequency-domain finite-difference，也就是频域 FDFD；当前 `gpr-inversion` 的物理和数据形式是 time-domain FDTD，需要输出随时间变化的接收器波形。

JAX 仍然适合本项目，但需要自己实现时域 FDTD 更新、CPML 边界、震源注入、接收器采样和梯度路径。

当前 CPU/Numba 实现必须保留为 correctness baseline。所有 GPU 结果都要先和现有 CPU 输出对齐，不能只看运行速度。

## 3. 阶段路线

### Phase 0: CPU baseline freeze

目标：固定一个小规模、可重复的 CPU 基准算例。

任务：

- 选择一个小网格，例如 `xl=20, zl=40, steps=80`。
- 使用现有 NumPy/Numba FDTD 生成参考 `d_syn`、关键波场统计和梯度结果。
- 保存参数说明、运行命令、输出形状和容差标准。

验收标准：

- 能稳定复现 CPU 输出。
- 记录 source/receiver、dx/dz/dt、npml、freq、epsilon/sigma 初值。
- 明确后续 JAX 对比容差，例如 waveform relative error 和 max absolute error。

### Phase 1: JAX forward prototype

目标：实现 JAX 版 2D TEz FDTD 正演。

任务：

- 用 `jax.numpy` 表达 `Ey/Hx/Hz` 数组。
- 用 `jax.lax.scan` 表达时间循环，避免 Python for-loop 成为性能瓶颈。
- 实现 CPML 参数和记忆变量更新。
- 实现 Ricker wavelet source 注入。
- 支持 single-shot 和 multi-shot receiver sampling。
- 初期先支持 float32，后续按需要评估 float64。

验收标准：

- 小网格 JAX `d_syn` 形状和 CPU 一致。
- single-shot 波形和 CPU baseline 在既定容差内。
- multi-shot 输出维度为 `[n_shots, n_receivers, n_steps]`。
- 可在 `gprfwi` 环境中检测 JAX device，并打印 CPU/GPU 后端信息。

### Phase 2: JAX gradient and inversion validation

目标：验证 JAX 自动微分或伴随梯度可用于反演。

任务：

- 定义最小 loss：`MSE(d_syn, d_obs)`。
- 在 tiny model 上做 finite-difference gradient check。
- 评估 JAX 自动微分的显存占用。
- 如果显存过高，设计 checkpoint 或 custom adjoint 路线。

验收标准：

- tiny model 上 JAX gradient 和有限差分方向导数符号、量级一致。
- 梯度无 NaN/Inf。
- 梯度 mask、TV 正则和固定浅层策略有明确迁移方案。

### Phase 3: integrate into `gpr_inversion`

目标：把通过验证的 JAX backend 接入现有项目，但不破坏 CPU 路径。

任务：

- 增加 backend 选择，例如 `cpu_numba` 和 `jax_gpu`。
- 保留现有 `forward.py` / `gradient.py` 作为 CPU baseline。
- 让 config 或 CLI 可以选择 JAX backend。
- 只在验证通过后接入 OverThrust mode2 eps-then-sig 路线。

验收标准：

- 默认行为仍然使用现有 CPU/Numba 或当前稳定路径。
- 显式选择 JAX backend 时才进入新实现。
- dry-run、配置检查和小规模 smoke test 通过。

### Phase 4: CUDA exploration

目标：在 JAX 原型数学路径稳定后，评估是否值得手写 CUDA。

任务：

- 用 JAX backend 的结果作为 CUDA 正确性参考。
- 确定需要 CUDA 优化的热点：时间步更新、CPML、shot batching 或 gradient。
- 评估 PyTorch extension、Numba CUDA、Triton 或原生 CUDA/C++ 的实现成本。

验收标准：

- 有明确性能瓶颈数据支持 CUDA 投入。
- 有 JAX/CPU 双基线可用于 CUDA 回归测试。
- CUDA 原型不替换 JAX backend，除非通过同样的 correctness tests。

## 4. 测试计划

文档阶段检查：

- `jax_fdtd_lab/plan.md`、`jax_fdtd_lab/AGENTS.md`、`jax_fdtd_lab/progress.md` 均存在。
- 三个文件可用 PowerShell `Get-Content -Encoding UTF8` 正常读取。
- 文档明确 JAX time-domain FDTD first。
- 文档明确 CUDA only after JAX prototype passes tests。
- 文档明确 current CPU/Numba implementation remains correctness baseline。
- 文档明确 no direct dependency on `jaxwell`。

未来实现阶段检查：

- Small-grid CPU vs JAX forward waveform comparison。
- CPML boundary sanity check。
- Single-shot and multi-shot receiver data shape checks。
- JAX gradient check against finite differences on a tiny model。
- Memory and runtime benchmark against current CPU/Numba path。

## 5. 环境约定

目标运行环境为现有 conda 环境：

```powershell
conda run -n gprfwi python ...
```

后续安装 JAX 时，应明确记录：

- Python 版本。
- JAX / jaxlib 版本。
- CUDA 版本。
- GPU 型号。
- 是否启用 float64。
- 运行命令和环境变量。

不要把依赖安装到不相关的默认 Python 中。当前项目 README 已经使用 `gprfwi`，因此 GPU 原型也优先沿用该环境。

截至 2026-05-20 的本机检查结果：

- Windows `gprfwi` 环境可以安装并运行 CPU 版 JAX。
- Windows 原生 PyTorch CUDA 可用，但 JAX 官方平台表不支持 Windows 原生 NVIDIA GPU。
- WSL2 Ubuntu-22.04 可以看到 NVIDIA GPU，后续更适合作为 JAX CUDA 验证环境。
- 因此本地 Windows `gprfwi` 先用于 CPU JAX correctness tests，GPU JAX benchmark 后续迁移到 WSL2 或 Linux server。

## 6. 边界和安全规则

- 本目录先作为 isolated lab，不直接改动稳定实验代码。
- 不删除旧实验结果，不批量删除文件或目录。
- 不使用 `Remove-Item -Recurse`、`rm -rf` 等递归删除命令。
- 任何数值重写都必须先有小规模正确性对比，再考虑大规模实验。
- 速度提升不是唯一目标，物理结果可信更重要。
