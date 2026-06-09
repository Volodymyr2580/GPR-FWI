#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单炮窗口梯度验证脚本。

构造一个 100x200 的简单模型，炮点放在窗口中心顶部，
使用现有的 ForwardModelFunction 计算 d_syn 与 d_obs，
对 MSE 损失求梯度并输出梯度统计信息/图片，便于验证
正传波场与伴随波场的互相关是否正确。
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from gpr_train_mpi import (
    ForwardModelFunction,
    generate_ricker_wavelet,
    DT,
    DX,
    DZ,
    NPML,
    FREQ,
    STEPS,
)
from utils.forward import forward_model
from utils.gradient import compute_gradient


def build_models():
    """构造真实/初始模型与电导率。"""
    depth, width = 100, 200
    background = 6.0
    true_model = np.full((depth, width), background, dtype=np.float32)

    # 在中心添加一个 20x20 的低速异常区
    anomaly_size = 20
    top = 40
    left = width // 2 - anomaly_size // 2
    true_model[top : top + anomaly_size, left : left + anomaly_size] = 3.0

    # 初始模型为均匀背景
    init_model = np.full_like(true_model, background, dtype=np.float32)

    # 简化：电导率取常数
    sigma_model = np.ones_like(true_model, dtype=np.float32) * 0.005

    return true_model, init_model, sigma_model


def run_single_shot_test(output_dir: Optional[Path] = None, device: Optional[torch.device] = None):
    """执行单炮梯度测试。"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "single_shot_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_true_np, epsilon_init_np, sigma_np = build_models()

    epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
    epsilon_init = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device, requires_grad=True)
    sigma_tensor = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    sigma_np = sigma_np.astype(np.float32)

    wavelet_np = generate_ricker_wavelet(FREQ, DT, STEPS)

    # 炮点与检波点：顶部（深度索引 0）、横向中心
    source_pos = (0, epsilon_true_np.shape[1] // 2)
    source_list = [source_pos]
    receiver_list = [source_pos]

    # 生成观测数据（真实模型）
    with torch.no_grad():
        d_obs_tensor = ForwardModelFunction.apply(
            epsilon_true,
            sigma_tensor,
            source_list,
            receiver_list,
            DT,
            DX,
            DZ,
            NPML,
            FREQ,
            STEPS,
            wavelet_np,
            False,
        ).detach()

    # 初始模型的正演
    d_syn = ForwardModelFunction.apply(
        epsilon_init,
        sigma_tensor,
        source_list,
        receiver_list,
        DT,
        DX,
        DZ,
        NPML,
        FREQ,
        STEPS,
        wavelet_np,
        False,
    )

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(d_syn, d_obs_tensor)
    loss.backward()

    grad_np = epsilon_init.grad.detach().cpu().numpy()
    d_obs_np = d_obs_tensor.cpu().numpy()
    d_syn_np = d_syn.detach().cpu().numpy()

    # 额外计算参考梯度：直接调用 forward_model + compute_gradient
    epsilon_init_np32 = epsilon_init_np.astype(np.float32)
    d_obs_np_ref = forward_model(
        epsilon_true_np,
        sigma_np,
        source_list,
        receiver_list,
        DT,
        DX,
        DZ,
        NPML,
        FREQ,
        STEPS,
        save_wavefield=False,
        wavelet=wavelet_np,
        force_serial=True,
    )

    d_syn_np_ref, wavefield_ref = forward_model(
        epsilon_init_np32,
        sigma_np,
        source_list,
        receiver_list,
        DT,
        DX,
        DZ,
        NPML,
        FREQ,
        STEPS,
        save_wavefield=True,
        wavelet=wavelet_np,
        force_serial=True,
    )

    residual_np = (d_syn_np_ref - d_obs_np_ref).astype(np.float32)
    grad_manual_np, _ = compute_gradient(
        epsilon_init_np32,
        sigma_np,
        residual_np,
        source_list,
        receiver_list,
        DT,
        DX,
        DZ,
        NPML,
        FREQ,
        STEPS,
        sigma_required_gradient=False,
        wavefield_data=wavefield_ref,
        force_serial=True,
    )
    grad_diff_np = grad_np - grad_manual_np.astype(np.float32)

    # 保存数据与可视化
    np.save(output_dir / "epsilon_true.npy", epsilon_true_np)
    np.save(output_dir / "epsilon_init.npy", epsilon_init_np)
    np.save(output_dir / "epsilon_true.npy", epsilon_true_np)
    np.save(output_dir / "epsilon_init.npy", epsilon_init_np)
    np.save(output_dir / "gradient_autograd.npy", grad_np)
    np.save(output_dir / "gradient_manual.npy", grad_manual_np)
    np.save(output_dir / "gradient_diff.npy", grad_diff_np)
    np.save(output_dir / "d_syn.npy", d_syn_np)
    np.save(output_dir / "d_obs.npy", d_obs_np)

    plt.figure(figsize=(6, 4))
    plt.imshow(epsilon_true_np, aspect="auto", cmap="jet")
    plt.title("True Epsilon")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_true.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(epsilon_init_np, aspect="auto", cmap="jet")
    plt.title("Initial Epsilon")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_init.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(grad_np, aspect="auto", cmap="seismic")
    plt.title("Gradient (autograd)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_autograd.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(grad_manual_np, aspect="auto", cmap="seismic")
    plt.title("Gradient (manual compute_gradient)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_manual.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(grad_diff_np, aspect="auto", cmap="seismic")
    plt.title("Gradient Difference (autograd - manual)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_diff.png")
    plt.close()

    time_axis = np.arange(STEPS)
    plt.figure(figsize=(8, 3))
    plt.plot(time_axis, d_obs_np.squeeze(), label="d_obs", linewidth=1.2)
    plt.plot(time_axis, d_syn_np.squeeze(), label="d_syn", linestyle="--", linewidth=1.0)
    plt.title("Time Series (single shot)")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries.png")
    plt.close()

    print("Single-shot gradient test completed.")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient stats (autograd) - min: {grad_np.min():.6e}, max: {grad_np.max():.6e}, mean: {grad_np.mean():.6e}")
    print(
        f"Gradient stats (manual) - min: {grad_manual_np.min():.6e}, "
        f"max: {grad_manual_np.max():.6e}, mean: {grad_manual_np.mean():.6e}"
    )
    diff_norm = np.linalg.norm(grad_diff_np) / (np.linalg.norm(grad_manual_np) + 1e-12)
    print(f"Gradient difference relative norm: {diff_norm:.6e}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_single_shot_test()

