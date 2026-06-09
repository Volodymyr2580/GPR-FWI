#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI版本双参数（epsilon、sigma）FWI训练程序（纯NumPy实现）
改造点：
- 移除所有 PyTorch/UNet，统一使用 NumPy
- 初始模型为真实模型的高斯平滑
- 使用与 mode1_Tikhonov 相似的 RMSprop 更新（双参数）
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time
from scipy.ndimage import gaussian_filter
from forward import forward_model
from gradient import compute_gradient, compute_tikhonov_gradient

# ==== MPI 初始化 ====
try:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	MPI_AVAILABLE = True
	if rank == 0:
		print(f"MPI初始化成功，进程数: {size}")
except Exception:
	# 兼容无 MPI 环境（串行）
	class _DummyComm:
		def Barrier(self):
			pass
		def bcast(self, x, root=0):
			return x
		def gather(self, x, root=0):
			return [x]
		def Get_rank(self):
			return 0
		def Get_size(self):
			return 1
	comm = _DummyComm()
	rank, size = 0, 1
	MPI_AVAILABLE = False
	print("MPI不可用，运行串行模式")


def ensure_clean_dir(target_dir):
	if os.path.exists(target_dir):
		if os.path.isdir(target_dir):
			shutil.rmtree(target_dir)
		else:
			os.remove(target_dir)
	os.makedirs(target_dir, exist_ok=True)


def should_save_epoch(epoch):
	save_epochs = [0, 1, 2, 5, 10, 20, 50]
	return epoch in save_epochs or (epoch >= 50 and epoch % 50 == 0)


def check_nan_inf(arr, name):
	if rank == 0:
		if np.any(np.isnan(arr)):
			print(f"[警告] {name} 出现nan!")
		if np.any(np.isinf(arr)):
			print(f"[警告] {name} 出现inf!")
		print(f"[{name}] min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")


def main():
	# 创建结果文件夹（仅在rank 0执行）
	if rank == 0:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		base_output_dir = os.path.join(script_dir, 'result_gpr_mpi_numpy')
		try:
			ensure_clean_dir(base_output_dir)
		except PermissionError:
			home_dir = os.path.expanduser('~')
			base_output_dir = os.path.join(home_dir, 'result_gpr_mpi_numpy')
			ensure_clean_dir(base_output_dir)
		print(f"结果将保存到: {base_output_dir}")
	else:
		base_output_dir = None

	base_output_dir = comm.bcast(base_output_dir if rank == 0 else None, root=0)
	comm.Barrier()

	# ========== 参数设置 ==========
	xl, zl = 100, 200
	dx, dz = 0.02, 0.02
	dt = 4e-11
	npml = 10
	steps = 1000
	freq = 4e8

	# RMSprop 与正则参数
	epochs = 2001
	beta_eps = 0.9
	beta_sig = 0.9
	alpha_eps = 0.03
	alpha_sig = 1e-6
	alpha_tikhonov_eps = 0.015
	alpha_tikhonov_sig = 0.01

	# 真实模型（与先前版本保持一致分布）
	epsilon_true = np.ones((xl, zl)) * 3
	epsilon_true[40:60, 60:80] = 1.5
	epsilon_true[85:, :] = 6

	# 使用与正确实现一致的电导率数量级，避免过大阻尼导致波场近零
	sigma_true = np.ones((xl, zl)) * 1e-4
	center_x, center_z = 50, 130
	radius = 10
	for x in range(sigma_true.shape[0]):
		for z in range(sigma_true.shape[1]):
			if (x - center_x) ** 2 + (z - center_z) ** 2 <= radius ** 2:
				sigma_true[x, z] = 5e-4
	sigma_true[85:, :] = 5e-5

	# 初始模型 = 真实模型的高斯平滑
	epsilon0 = np.ones((xl, zl)) * 3
	epsilon0[85:, :] = 6
	sigma0 = np.ones((xl, zl)) * 1e-4
	sigma0[85:, :] = 5e-5


	# 源、检波配置（mode2: 多接收器）
	source_list = [(0, i) for i in range(10, zl-10, 10)]
	receiver_list = [(0, i) for i in range(10, zl-10, 2)]

	# 保存真实/初始模型
	if rank == 0:
		path = base_output_dir if base_output_dir.endswith(os.sep) else base_output_dir + os.sep
		plt.figure(); plt.imshow(epsilon_true, aspect='auto', cmap='jet', vmin=0, vmax=10); plt.colorbar(); plt.title('epsilon_true'); plt.savefig(f'{path}epsilon_true.png'); plt.close()
		plt.figure(); plt.imshow(sigma_true, aspect='auto', cmap='jet', vmin=1e-6, vmax=1e-3); plt.colorbar(); plt.title('sigma_true'); plt.savefig(f'{path}sigma_true.png'); plt.close()
		plt.figure(); plt.imshow(epsilon0, aspect='auto', cmap='viridis', vmin=0, vmax=10); plt.colorbar(); plt.title('epsilon_init'); plt.savefig(f'{path}epsilon_init.png'); plt.close()
		plt.figure(); plt.imshow(sigma0, aspect='auto', cmap='viridis', vmin=1e-6, vmax=1e-3); plt.colorbar(); plt.title('sigma_init'); plt.savefig(f'{path}sigma_init.png'); plt.close()

	# ========== 生成观测数据（MPI并行） ==========
	if rank == 0:
		print("生成观测数据...")

	local_d_obs = forward_model(
		epsilon_true,
		sigma_true,
		source_list,
		receiver_list,
		dt, dx, dz, npml, freq, steps, save_wavefield=False
	)

	d_obs_list = comm.gather(local_d_obs, root=0)
	if rank == 0:
		d_obs = np.concatenate(d_obs_list, axis=0) if len(d_obs_list) > 0 else np.zeros((0, len(receiver_list), steps), dtype=np.float32)
		plt.figure(); plt.imshow(d_obs.reshape(len(source_list)*len(receiver_list), steps).T, aspect='auto', cmap='seismic'); plt.colorbar(); plt.title('Observed data'); plt.savefig(f'{path}d_obs.png'); plt.close()
	else:
		d_obs = None
	d_obs = comm.bcast(d_obs, root=0)

	# ========== 反演初始化 ==========
	model_eps = epsilon0.copy()
	model_sig = sigma0.copy()
	V = np.zeros_like(model_eps.flatten())
	U = np.zeros_like(model_sig.flatten())
	loss_list = []

	for epoch in range(epochs):
		epoch_start = time.time()
		if rank == 0:
			print(f"Epoch {epoch+1}/{epochs}")

		# 正演（传入完整source_list，由正演内部按MPI切分并返回本地数据与波场）
		local_d_syn, local_wavefield = forward_model(
			model_eps, model_sig, source_list, receiver_list,
			dt, dx, dz, npml, freq, steps, save_wavefield=True
		)

		# 收集并重构全局 d_syn（只在root重构）
		all_local_d_syn = comm.gather(local_d_syn, root=0)
		if rank == 0:
			d_syn = np.zeros_like(d_obs)
			data_idx = 0
			for proc_d in all_local_d_syn:
				if proc_d is not None and proc_d.shape[0] > 0:
					n_local = proc_d.shape[0]
					d_syn[data_idx:data_idx+n_local] = proc_d
					data_idx += n_local
			assert d_syn.shape == d_obs.shape, "d_syn和d_obs形状不一致"
			residual = d_syn - d_obs
		else:
			residual = None
		# 广播残差
		residual = comm.bcast(residual, root=0)

		# 梯度（compute_gradient 内部含MPI规约；本地波场由正演返回的local_wavefield提供）
		grad_eps, grad_sig = compute_gradient(
			model_eps, model_sig, residual, source_list, receiver_list,
			dt, dx, dz, npml, freq, steps, sigma_required_gradient=True, wavefield_data=local_wavefield
		)

		if rank == 0:
			check_nan_inf(grad_eps, f"grad_eps (epoch {epoch})")
			if grad_sig is not None:
				check_nan_inf(grad_sig, f"grad_sig (epoch {epoch})")

		# Tikhonov正则梯度
		tikh_grad_eps = compute_tikhonov_gradient(model_eps.copy(), dx, dz, alpha_tikhonov_eps)
		tikh_grad_sig = compute_tikhonov_gradient(model_sig.copy(), dx, dz, alpha_tikhonov_sig)

		# RMSprop 更新 - epsilon
		g_eps = grad_eps.flatten()
		V = beta_eps * V + (1 - beta_eps) * (g_eps ** 2)
		update_eps = g_eps / (np.sqrt(V + 1e-8))
		max_update_eps = max(np.max(np.abs(update_eps)), 1e-12)
		update_eps_norm = update_eps / max_update_eps
		d_eps = -alpha_eps * update_eps_norm - tikh_grad_eps.flatten()
		model_eps = model_eps + d_eps.reshape(model_eps.shape)
		model_eps = np.maximum(model_eps, 1.0)

		# RMSprop 更新 - sigma（若返回了梯度）
		if grad_sig is not None:
			g_sig = grad_sig.flatten()
			U = beta_sig * U + (1 - beta_sig) * (g_sig ** 2)
			update_sig = g_sig / (np.sqrt(U + 1e-8))
			max_update_sig = max(np.max(np.abs(update_sig)), 1e-12)
			update_sig_norm = update_sig / max_update_sig
			d_sig = -alpha_sig * update_sig_norm - tikh_grad_sig.flatten()
			model_sig = model_sig + d_sig.reshape(model_sig.shape)
			model_sig = np.maximum(model_sig, 0.0)

		# 可视化/记录
		if rank == 0 and should_save_epoch(epoch):
			plt.figure(); plt.imshow(model_eps, cmap='jet', vmin=0, vmax=10); plt.title(f'epsilon model (epoch {epoch})'); plt.colorbar(); plt.savefig(f'{path}model_eps_epoch{epoch}.png'); plt.close()
			plt.figure(); plt.imshow(model_sig, cmap='jet', vmin=0, vmax=1e-3); plt.title(f'sigma model (epoch {epoch})'); plt.colorbar(); plt.savefig(f'{path}model_sig_epoch{epoch}.png'); plt.close()

		# 记录损失
		if rank == 0:
			loss = np.linalg.norm(residual)
			loss_list.append(loss)
			plt.figure(); plt.plot(loss_list); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('FWI Loss Curve'); plt.savefig(f'{path}loss_curve.png'); plt.close()

		comm.Barrier()
		if rank == 0:
			print(f"  Epoch {epoch+1}/{epochs} 耗时: {time.time() - epoch_start:.2f}s")

	# 最终保存
	if rank == 0:
		plt.figure(); plt.imshow(model_eps, cmap='jet'); plt.title('Final epsilon model'); plt.colorbar(); plt.savefig(f'{path}final_model_eps.png'); plt.close()
		plt.figure(); plt.imshow(model_sig, cmap='jet'); plt.title('Final sigma model'); plt.colorbar(); plt.savefig(f'{path}final_model_sig.png'); plt.close()
		print("训练完成！结果已保存。")


if __name__ == "__main__":
	main()
