# IFWI / dropout-IFWI 复现计划

更新时间：2026-06-06

## 1. 目标

本目录复现论文 `Implicit multiparameter full waveform inversion of multioffset ground penetrating radar data` 中的 IFWI 和 dropout-IFWI。

当前阶段用 CVPR 2024 FR-INR 仓库作为坐标网络架构标准，但复现目标仍然是 GPR IFWI/dropout-IFWI 论文：

1. 用 FR-INR 风格坐标网络替代论文原始普通 MLP，表示双参数地下模型。
2. 使用 PyTorch 训练网络，不再走 JAX 主线。
3. 先通过已有 CPU/MPI GPR 正演和梯度代码验证小规模链路。
4. 后续再把论文级网格、炮检几何、频率和训练轮数逐步对齐。

传统 FWI、多尺度 FWI、UNet 反演暂不复现。

## 2. 技术路线

当前计算链路：

```text
坐标网格 (x,z)
  -> FR-INR / SIREN 坐标网络 Nθ
  -> epsilon_r, sigma
  -> CPU/MPI GPR FDTD 正演
  -> d_syn
  -> loss(d_syn, d_obs)
  -> 显式梯度桥接回 PyTorch
  -> Adam 更新网络权重 θ
```

通俗解释：

- `epsilon_r` 是相对介电常数，主要影响雷达波速度。
- `sigma` 是电导率，主要影响雷达波衰减。
- 坐标网络输入的是网格点位置 `(x,z)`，输出整张地下参数图。
- FR-INR 不改变 GPR 反演目标，它只是把坐标网络层换成更可靠的傅里叶重参数化结构，让网络更容易表达边界和细节。
- 论文中的 IFWI 参数输出方式是 `m = m_tilde * std + mean`；当前代码已支持这种标准化映射，也保留早期 smoke 用的 bounds/sigmoid 映射。
- PyTorch autograd bridge 的作用是把外部 GPR 求解器的梯度接回网络训练。

## 3. 当前代码入口

核心代码：

- `src/ifwi_gpr/networks/fr_inr.py`：按 FR-INR 标准改写的网络层与双参数 IFWI 包装。
- `src/ifwi_gpr/acquisition.py`：显式炮检列表和 perimeter 围绕边界炮检几何生成。
- `src/ifwi_gpr/networks/siren.py`：原 SIREN baseline。
- `src/ifwi_gpr/train.py`：根据 config 构建网络和训练链路。
- `src/ifwi_gpr/autograd.py`：把 CPU/MPI 求解器包装成 PyTorch 可反传算子。
- `src/ifwi_gpr/solver_bridge/cpu_mpi.py`：调用参考 GPR 正演和梯度。

参考仓库：

- `FR-INR/`：CVL-UESTC/FR-INR 原始仓库副本，只作为参考，不直接作为复现主代码。

推荐 smoke config：

```powershell
python -m ifwi_gpr.run --config configs/paper_cross_dropout_ifwi_frinr_smoke.json --dry-run
```

## 4. 阶段任务

### Phase 1: FR-INR 网络标准接入

- [x] 克隆 FR-INR 参考仓库。
- [x] 将核心 Fourier reparameterized layer 改写到本项目包内。
- [x] 支持 `network.architecture = "fr_inr"`。
- [x] 新增 FR-INR dry-run 测试。
- [x] 支持论文式 `mean/std` 标准化参数输出。

验收：

- 网络输出 `epsilon_r` 和 `sigma` shape 正确。
- 输出无 NaN/Inf。
- bounds/sigmoid 配置下，参数范围被限制在 config 的物理上下界内。
- standardized 配置下，网络输出按 `mean/std` 还原到物理量。

### Phase 2: 非 JAX 链路收口

- [x] 删除 JAX 专用源码入口。
- [x] 删除 JAX 专用测试。
- [x] 删除 JAX 专用 config。
- [ ] 逐步清理历史 JAX 运行产物和缓存；禁止批量删除，需要单文件确认。

验收：

- `src/`、`tests/`、`configs/` 中不再有 JAX 主线源码或配置。
- 单元测试通过。

### Phase 3: 小规模 GPR 训练验证

- [x] 运行 `configs/frinr_cross_dropout_smoke.json` dry-run。
- [x] 运行 micro/tiny bridge 训练，确认 loss 有限。
- [x] 保存 `metrics.json`、`epoch_*_epsilon.npy`、`epoch_*_sigma.npy`。
- [x] 运行 101x101 cross-shape dropout 短训练。
- [x] 增加均匀背景预训练并完成 pretrain smoke。

验收：

- forward 返回 `[shots, receivers, time]` 数据。
- 反传能更新 FR-INR 网络参数。
- loss 为有限值。
- 训练结束自动保存参数数组和 figure 文件。

注意：参考 `Add_CPML` 内部固定 `npml=10`，因此当前 CPU/MPI bridge config 中 `solver.npml` 也需要保持 10。

### Phase 4: 论文级设置对齐

- [x] 按论文整理 cross-shape 模型设置。
- [ ] 按论文整理 overthrust 模型设置。
- [x] 对齐 cross-shape 网格间距、时间步长、频率和 32/64 perimeter 炮检模板。
- [ ] 对齐 overthrust 炮检几何、网格间距、时间步长、频率。
- [x] 设计 cross-shape dropout-IFWI 的 FR-INR smoke config。
- [x] 设计 cross-shape dropout-IFWI 的 FR-INR full template config。
- [x] 完成 cross-shape IFWI/dropout-IFWI 中等敏感性训练 probe。
- [ ] 设计论文级 IFWI 与 dropout-IFWI 的正式训练 config。
- [ ] 输出参数图、误差图、loss 曲线。

验收：

- 图像和指标可以与论文定性对比。
- `progress.md` 记录每次实验命令、config、产物目录和结论。
