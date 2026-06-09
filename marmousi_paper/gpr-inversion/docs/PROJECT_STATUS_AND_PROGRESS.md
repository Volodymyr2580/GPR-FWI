# GPR 反演项目说明与实验进展

更新时间：2026-05-17

本文档记录当前 `gpr-inversion` 项目的整体目标、代码组织、已经完成的实验线、OverThrust mode2 双参数反演进展，以及下一步 mode1 超参数搜索的准备方向。

## 1. 项目目标

本项目围绕 GPR/FWI 反演实验展开，核心目标是从雷达观测数据中反推出地下介质参数。

当前重点参数包括：

- `epsilon` / `eps`：相对介电常数，主要控制电磁波传播速度。
- `sigma` / `sig`：电导率，主要影响衰减和导电性。

当前主要研究路线是：

1. 先用 UNet 重参数化方式训练 epsilon 网络。
2. 从固定的 stage1 epsilon checkpoint 出发。
3. 在 stage2 同时更新 epsilon 网络和 sigma 网络。
4. 用数据拟合误差、模型残差、可视化结果共同评估超参数。

## 2. 代码组织

新的整理版代码位于：

```text
gpr-inversion/
```

旧实验代码仍然保留在原目录中，不主动删除、不批量移动。`gpr-inversion` 是新的迁移和实验管理目标。

主要目录如下：

```text
gpr-inversion/
  configs/                     # 实验配置文件
  docs/                        # 项目文档
  scripts/                     # 搜参和服务器运行脚本
  src/gpr_inversion/           # Python 包源码
    run.py                     # 统一 config runner
    config.py                  # 配置解析与校验
    acquisition/               # mode1/mode2 采集几何辅助
    models/                    # 模型读取和初始化辅助
    regularization/            # TV 等正则化辅助
    experiments/               # 已迁移实验线
```

对初学者来说，可以把 `configs` 理解成“实验参数表”，把 `src/gpr_inversion/experiments` 理解成“真正执行实验的代码”。统一入口 `python -m gpr_inversion.run --config ...` 会读取配置，然后调用对应实验脚本。

## 3. 已迁移实验线

目前 runner 已连接的实验包括：

- Marmousi traditional epsilon-only：
  - mode1 + LBFGS
  - mode1 + RMSprop
  - mode2 + LBFGS
  - mode2 + RMSprop
- Marmousi UNet epsilon-only：
  - mode1 no illumination
  - mode1 illumination
  - mode2 no illumination
  - mode2 illumination
- OverThrust UNet eps-then-sig：
  - mode2 no illumination
  - mode2 illumination

当前重点工作集中在：

```text
src/gpr_inversion/experiments/overthrust/mode2_eps_then_sig/
```

对应原始来源脚本是：

```text
OverThrust_Tests/Unet/mode2/twopara_epsfirst.py
```

## 4. OverThrust Mode2 双参数反演现状

当前 OverThrust mode2 实验采用：

```text
stage1: 训练 epsilon UNet
stage2: 从 stage1 checkpoint 恢复，同时更新 epsilon UNet 和 sigma UNet
```

服务器上使用过的基础命令形式是：

```bash
mpirun -np 20 python twopara_epsfirst.py \
  --stage1-checkpoint-path stage1_eps_checkpoint.pt \
  --resume \
  --illumination \
  --device cuda:1 \
  --alpha-tv-sig 7e-3 \
  --num-epochs-stage2 10001
```

在整理版 `gpr-inversion` 中，当前 stage2 关键参数包括：

- `--lr-eps-stage2`：stage2 中 epsilon 网络学习率。
- `--learning-rate-sig`：stage2 中 sigma 网络学习率。
- `--alpha-tv-eps`：epsilon TV 正则梯度权重。
- `--alpha-tv-sig`：sigma TV 正则梯度权重。
- `--alpha-l1-data`：混合 L1 数据项系数。

当前数据 loss 形式为：

```text
data_loss = MSE(d_syn, d_obs) + alpha_l1_data * L1(d_syn, d_obs)
```

其中：

- `MSE` 是 L2 类型的数据拟合项，对大残差更敏感。
- `L1` 是绝对值残差项，对异常值更稳健。
- 搜参目标仍主要使用 `data_misfit_mse`，这样不同 L1 系数之间仍有一个统一可比的 MSE 指标。

## 5. 当前输出内容

每次训练会在结果目录中保存：

```text
metrics.json
metrics_history.csv
loss_curves_epoch_*.png
epoch_*_epsilon.png
epoch_*_sigma.png
epoch_*_epsilon_residual.png
epoch_*_sigma_residual.png
epoch_*_epsilon.npy
epoch_*_sigma.npy
```

其中：

- `metrics.json`：最后一个 epoch 的指标和完整历史。
- `metrics_history.csv`：每个 epoch 的指标表，方便画图或导入表格。
- `epoch_*_epsilon.png` / `epoch_*_sigma.png`：反演模型图。
- `epoch_*_epsilon_residual.png` / `epoch_*_sigma_residual.png`：反演模型减真实模型的残差图。
- `.npy` 文件：模型数组，便于后续重新画图或论文制图。

当前记录的重要指标包括：

- `data_misfit_mse`：合成数据和观测数据之间的 MSE，是 Optuna 主目标。
- `data_l1`：合成数据和观测数据之间的 L1 残差。
- `eps_model_loss`：epsilon 模型相对真实模型的归一化 MSE。
- `sig_model_loss`：sigma 模型相对真实模型的归一化 MSE。
- `eps_residual_mae`：epsilon 模型残差的平均绝对值。
- `sig_residual_mae`：sigma 模型残差的平均绝对值。
- `model_residual_score`：`eps_model_loss + sig_model_loss`，作为模型残差综合参考。

## 6. Optuna 搜参方案

已新增服务器搜参脚本：

```text
scripts/run_overthrust_optuna_search.sh
scripts/overthrust_optuna_search.py
```

一行运行形式：

```bash
STAGE1_CHECKPOINT_PATH=/path/to/stage1_eps_checkpoint.pt DEVICE=cuda:1 N_TRIALS=50 NUM_EPOCHS_STAGE2=1000 bash scripts/run_overthrust_optuna_search.sh
```

默认设置：

```text
MPI_PROCESSES = 30
N_TRIALS = 50
NUM_EPOCHS_STAGE2 = 1000
alpha_l1_data search range = 0.0 ~ 0.5
```

默认搜索空间：

```text
lr_eps_stage2:      1e-6 ~ 1e-4，log scale
learning_rate_sig:  1e-6 ~ 1e-4，log scale
alpha_tv_eps:       1e-2 ~ 5e-2
alpha_tv_sig:       3e-2 ~ 4e-2
alpha_l1_data:      0.0 ~ 0.5
```

Optuna 使用 TPE sampler。可以简单理解为：先试一些组合，然后根据已经完成 trial 的好坏，把后续采样集中到更可能有效的区域。

## 7. 已完成的 OverThrust Mode2 搜参结果

一次搜索中已完成：

```text
completed trials: 71
failed trials: 2
best trial: 62
best data_misfit_mse: 1.61136771e-04
```

当前最佳参数为：

```text
lr_eps_stage2:      3.701816311652135e-06
learning_rate_sig:  1.0226577919012332e-06
alpha_tv_eps:       0.027858282847238437
alpha_tv_sig:       0.038829523720510684
alpha_l1_data:      0.00821093666463582
```

Top trials 呈现出的趋势：

```text
lr_eps_stage2:      约 2.5e-6 ~ 6e-6
learning_rate_sig:  约 1.0e-6 ~ 1.8e-6
alpha_tv_eps:       约 0.025 ~ 0.035
alpha_tv_sig:       约 0.038 ~ 0.040
alpha_l1_data:      约 0.0 ~ 0.02，偏小更优
```

下一轮局部精搜建议：

```bash
STAGE1_CHECKPOINT_PATH=/path/to/stage1_eps_checkpoint.pt DEVICE=cuda:1 MPI_PROCESSES=30 N_TRIALS=80 NUM_EPOCHS_STAGE2=1000 LR_EPS_STAGE2_MIN=2e-6 LR_EPS_STAGE2_MAX=8e-6 LR_SIG_MIN=8e-7 LR_SIG_MAX=2.5e-6 ALPHA_TV_EPS_MIN=0.022 ALPHA_TV_EPS_MAX=0.038 ALPHA_TV_SIG_MIN=0.037 ALPHA_TV_SIG_MAX=0.041 ALPHA_L1_DATA_MAX=0.05 bash scripts/run_overthrust_optuna_search.sh
```

需要注意：最低 `data_misfit_mse` 不一定代表模型结构最好。因此应同时查看 top trials 的 residual 图和 `model_residual_score`。

## 8. 已知问题和注意事项

1. `return code 143` / `signal 15` 通常表示进程被外部终止，例如服务器任务时限、调度器终止或资源管理策略，不一定是超参数数值发散。
2. Optuna study 保存在 `optuna_study.db` 中，脚本重新运行时会继续追加 trial，不会丢失已完成结果。
3. `N_TRIALS` 表示本次脚本调用还要再运行多少个 trial，不是总 trial 数。
4. 搜参时每个 trial 都会创建独立目录，便于对比模型图、残差图和指标文件。
5. 不应只看 data misfit，也应查看 epsilon/sigma residual 图，避免出现“数据拟合好但模型不合理”的情况。

## 9. 下一步：Mode1 超参数调整

接下来准备检查 mode1 对应实验线，并设计类似的自动化搜参脚本。

mode1 和 mode2 的主要区别是采集几何：

- mode1：source 和 receiver 通常一一对应，单发单收或同位置收发。
- mode2：一个 source 对应多个 receivers，多接收器记录更丰富。

对 mode1 的下一步检查重点：

1. 确认目标是 Marmousi mode1 还是 OverThrust mode1。
2. 确认是 epsilon-only，还是也要做 epsilon/sigma 双参数。
3. 检查 mode1 当前脚本是否已有：
   - 机器可读 metrics 输出；
   - 模型图；
   - residual 图；
   - Optuna 可调用的 CLI 参数；
   - 安全输出目录；
   - 可恢复 checkpoint。
4. 根据 mode1 的训练成本设置 trial 数和 epoch 数。
5. 写对应的 `run_*_optuna_search.sh` 一键脚本。

## 10. 推荐工作流

后续每条实验线建议按以下顺序推进：

1. 先检查旧脚本和迁移脚本是否数值路径一致。
2. 加入统一 CLI 参数。
3. 加入 `metrics.json` / `metrics_history.csv`。
4. 加入模型图和 residual 图。
5. 写 Optuna 调度器。
6. 先小规模 dry run 或短 epoch 试运行。
7. 再做正式 50-100 个 trial 搜参。
8. 对 top trials 同时比较 data misfit 和 model residual。

