# AGENTS.md for jax_fdtd_lab

本目录是 `gpr-inversion` 的 JAX/CUDA 时域 FDTD 实验沙盒。所有 agent 在本目录工作时，必须遵守仓库根目录给出的安全规则，并额外遵守以下 lab-specific 规则。

## 1. 文件安全

- 禁止批量删除文件或目录。
- 禁止使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 如确实需要删除文件，只能一次删除一个明确路径的文件，并说明原因。
- 如果需要批量清理实验输出，停止操作，请用户手动删除或明确确认下一步。
- 不要移动、删除或覆盖现有 `src/`、`configs/`、`docs/`、`outputs/` 中的内容。

## 2. 文档和编码

- 所有中文 Markdown、TXT、说明文档和报告默认使用 UTF-8。
- PowerShell 读取中文文档时使用：

```powershell
Get-Content -Encoding UTF8 "文件路径"
```

- 新建或修改中文文档时必须确保保存为 UTF-8。
- 面向用户解释时，要照顾前后端开发初学者背景，说明技术概念、命令用途、是否影响全局配置。

## 3. Lab 范围

- `jax_fdtd_lab/` 是试验区，先用于计划、原型、验证脚本和记录。
- 当前稳定的 CPU/Numba FDTD 实现是 correctness baseline，不能为了 JAX 或 CUDA 原型而删除或绕开。
- JAX 原型通过小网格测试前，不要接入正式 `gpr_inversion.run`。
- CUDA 只在 JAX 原型跑通并完成正确性测试后再探索。

## 4. 数值实现原则

- 第一优先级是数值正确，第二优先级才是速度。
- 每个 GPU 实验都要记录：
  - conda 环境名。
  - Python / JAX / jaxlib / CUDA 版本。
  - GPU 型号。
  - dtype，例如 float32 或 float64。
  - CPU baseline 命令和 GPU 命令。
  - 对比容差，例如 max absolute error 和 relative error。
- 任何 JAX 输出必须先和当前 CPU/Numba 输出比较。
- 不允许只凭 loss 下降判断正演正确，必须检查波形、形状、NaN/Inf 和边界行为。

## 5. JAX 方向

- 使用 JAX 编写 time-domain FDTD，不直接依赖 `jaxwell`。
- 推荐使用 `jax.numpy` 表达数组计算。
- 推荐使用 `jax.lax.scan` 表达时间循环。
- 推荐使用 `jax.vmap` 或批处理方式处理多炮，但必须先保证 single-shot 正确。
- 初期以 float32 为默认精度，遇到稳定性问题再评估 float64。

## 6. CUDA 方向

- 手写 CUDA 是后续优化路线，不是第一阶段目标。
- CUDA 原型必须以 CPU baseline 和 JAX baseline 作为双重正确性参考。
- 未通过小网格对比前，不得替代 JAX backend 或现有 CPU backend。

## 7. Git 和网络代理

- 访问 GitHub 的 Git 命令不要依赖全局代理，优先临时清空代理：

```powershell
git -c http.proxy= -c https.proxy= fetch
```

- 不要主动修改用户全局 Git 代理配置。
- 不要执行 `git config --global --unset http.proxy` 或 `git config --global --unset https.proxy`，除非用户明确要求。

## 8. 推荐工作流

1. 阅读 `plan.md` 和 `progress.md`。
2. 确认当前任务属于文档、JAX 原型、测试验证，还是 CUDA 探索。
3. 若是实现任务，先找 CPU baseline，再写最小 JAX 对照。
4. 每完成一个可验证步骤，更新 `progress.md`。
5. 大规模实验前先跑 tiny/small case。

