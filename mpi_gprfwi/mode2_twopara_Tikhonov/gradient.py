import numpy as np
import sys
import os
from Time_loop import reverse_time_loop
from Add_CPML import Add_CPML

# 添加MPI支持
try:
	from mpi4py import MPI
	MPI_AVAILABLE = True
except ImportError:
	MPI_AVAILABLE = False
	print("Warning: mpi4py not available, running in serial mode")

def compute_laplacian_2d(field, dx, dz):
	"""
	计算二维拉普拉斯算子 ∇²f = ∂²f/∂x² + ∂²f/∂z²
	使用中心差分格式
	"""
	nx, nz = field.shape
	laplacian = np.zeros_like(field)
	# 内部点
	laplacian[1:-1, 1:-1] = (
		(field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx**2) +
		(field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / (dz**2)
	)
	# 边界与角点（简化一阶处理）
	laplacian[0, 1:-1] = ((field[1, 1:-1] - field[0, 1:-1]) / (dx**2) + (field[0, 2:] - 2*field[0, 1:-1] + field[0, :-2]) / (dz**2))
	laplacian[-1, 1:-1] = ((field[-1, 1:-1] - field[-2, 1:-1]) / (dx**2) + (field[-1, 2:] - 2*field[-1, 1:-1] + field[-1, :-2]) / (dz**2))
	laplacian[1:-1, 0] = ((field[2:, 0] - 2*field[1:-1, 0] + field[:-2, 0]) / (dx**2) + (field[1:-1, 1] - field[1:-1, 0]) / (dz**2))
	laplacian[1:-1, -1] = ((field[2:, -1] - 2*field[1:-1, -1] + field[:-2, -1]) / (dx**2) + (field[1:-1, -1] - field[1:-1, -2]) / (dz**2))
	laplacian[0, 0] = (field[1, 0] - field[0, 0]) / (dx**2) + (field[0, 1] - field[0, 0]) / (dz**2)
	laplacian[0, -1] = (field[1, -1] - field[0, -1]) / (dx**2) + (field[0, -1] - field[0, -2]) / (dz**2)
	laplacian[-1, 0] = (field[-1, 0] - field[-2, 0]) / (dx**2) + (field[-1, 1] - field[-1, 0]) / (dz**2)
	laplacian[-1, -1] = (field[-1, -1] - field[-2, -1]) / (dx**2) + (field[-1, -1] - field[-1, -2]) / (dz**2)
	return laplacian

def compute_tikhonov_gradient(model, dx, dz, alpha_tikhonov):
	"""
	计算二阶Tikhonov正则项的梯度: ∇R(m) = α * ∇²(∇²m)
	"""
	laplacian = compute_laplacian_2d(model, dx, dz)
	tikhonov_grad = compute_laplacian_2d(laplacian, dx, dz)
	max_abs = np.max(np.abs(tikhonov_grad))
	if max_abs > 0:
		tikhonov_grad = tikhonov_grad / max_abs
	return alpha_tikhonov * tikhonov_grad

def compute_gradient_mpi(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=False, wavefield_data=None, global_start_idx=0):
	"""
	MPI并行版本的梯度计算（与forward切分一致，wavefield_data应为本地shots顺序）
	返回: grad_eps, grad_sig(可为None)
	"""
	if not MPI_AVAILABLE:
		return compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data)
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	xl, zl = epsilon.shape
	n_shots = len(source_list)
	# 与forward一致的切分
	shots_per_proc = n_shots // size
	remainder = n_shots % size
	if rank < remainder:
		start_idx = rank * (shots_per_proc + 1)
		end_idx = start_idx + shots_per_proc + 1
	else:
		start_idx = rank * shots_per_proc + remainder
		end_idx = start_idx + shots_per_proc
	local_source_list = source_list[start_idx:end_idx]
	if residual.shape[0] == n_shots:
		residual_local = residual[start_idx:end_idx, :, :]
	else:
		residual_local = residual
	# 本地梯度
	local_grad_eps = np.zeros_like(epsilon)
	local_grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
	ep0 = 8.841941282883074e-12
	# 扩展模型
	epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
	sigma_extended = np.ones((xl + 2*npml, zl + 2*npml))
	mu_extended = np.ones((xl + 2*npml, zl + 2*npml))
	epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
	sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
	epsilon_extended[:npml, :] = epsilon_extended[npml, :]
	epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
	epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
	epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
	sigma_extended[:npml, :] = sigma_extended[npml, :]
	sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
	sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
	sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
	cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), dx, dz, dt)
	for local_shot_idx, _ in enumerate(local_source_list):
		if wavefield_data is None or local_shot_idx >= len(wavefield_data):
			print("Warning: wavefield_data not provided or mismatched; skip this shot in gradient.")
			continue
		forward_wavefield = np.array(wavefield_data[local_shot_idx])
		forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
		shot_residual = residual_local[local_shot_idx, :, :]
		reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), cpml_params, steps, receiver_list, shot_residual)
		adjoint_list = [np.array(adj) for adj in reverse_loop_gen][::-1]
		for k in range(1, steps-1):
			adjoint_field = adjoint_list[k]
			local_grad_eps += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
			if sigma_required_gradient:
				if local_grad_sig is None:
					local_grad_sig = np.zeros_like(sigma)
				local_grad_sig += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
	# 聚合
	all_grad_eps = comm.gather(local_grad_eps, root=0)
	if sigma_required_gradient:
		all_grad_sig = comm.gather(local_grad_sig if local_grad_sig is not None else np.zeros_like(sigma), root=0)
	if rank == 0:
		grad_eps = np.zeros_like(epsilon)
		for g in all_grad_eps:
			grad_eps += g
		if sigma_required_gradient:
			grad_sig = np.zeros_like(sigma)
			for g in all_grad_sig:
				grad_sig += g
		else:
			grad_sig = None
		mask = np.ones_like(grad_eps)
		grad_eps *= mask
		if sigma_required_gradient and grad_sig is not None:
			grad_sig *= mask
	else:
		grad_eps = np.zeros_like(epsilon)
		grad_sig = np.zeros_like(sigma) if sigma_required_gradient else None
	grad_eps = comm.bcast(grad_eps, root=0)
	if sigma_required_gradient:
		grad_sig = comm.bcast(grad_sig, root=0)
		return grad_eps, grad_sig
	else:
		return grad_eps, None

def compute_gradient(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient=False, wavefield_data=None, global_start_idx=0):
	"""
	串行版本梯度计算；当MPI可用时委托到并行实现。
	"""
	if MPI_AVAILABLE and MPI.COMM_WORLD.Get_size() > 1:
		return compute_gradient_mpi(epsilon, sigma, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, sigma_required_gradient, wavefield_data, global_start_idx)
	xl, zl = epsilon.shape
	epn = np.zeros_like(epsilon)
	sgn = np.zeros_like(sigma) if sigma_required_gradient else None
	ep0 = 8.841941282883074e-12
	# 扩展模型
	epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
	sigma_extended = np.ones((xl + 2*npml, zl + 2*npml))
	mu_extended = np.ones((xl + 2*npml, zl + 2*npml))
	epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon
	sigma_extended[npml:npml+xl, npml:npml+zl] = sigma
	epsilon_extended[:npml, :] = epsilon_extended[npml, :]
	epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
	epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
	epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)
	sigma_extended[:npml, :] = sigma_extended[npml, :]
	sigma_extended[-npml:, :] = sigma_extended[-npml-1, :]
	sigma_extended[:, :npml] = sigma_extended[:, npml].reshape(-1, 1)
	sigma_extended[:, -npml:] = sigma_extended[:, -npml-1].reshape(-1, 1)
	cpml_params = Add_CPML(xl, zl, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), dx, dz, dt)
	for shot_idx, _ in enumerate(source_list):
		if wavefield_data is None or shot_idx >= len(wavefield_data):
			print("Warning: wavefield_data not provided or mismatched; skip this shot in gradient.")
			continue
		forward_wavefield = np.array(wavefield_data[shot_idx])
		forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2 * dt)
		shot_residual = residual[shot_idx, :, :]
		reverse_loop_gen = reverse_time_loop(xl, zl, dx, dz, dt, sigma_extended.copy(), epsilon_extended.copy(), mu_extended.copy(), cpml_params, steps, receiver_list, shot_residual)
		adjoint_list = [np.array(adj) for adj in reverse_loop_gen][::-1]
		for k in range(1, steps-1):
			adjoint_field = adjoint_list[k]
			epn += ep0 * adjoint_field[npml:npml+xl, npml:npml+zl] * forward_diff[k-1][npml:npml+xl, npml:npml+zl]
			if sigma_required_gradient:
				if sgn is None:
					sgn = np.zeros_like(sigma)
				sgn += adjoint_field[npml:npml+xl, npml:npml+zl] * forward_wavefield[k][npml:npml+xl, npml:npml+zl]
	mask = np.ones_like(epn)
	epn *= mask
	if sigma_required_gradient and sgn is not None:
		sgn *= mask
		return epn, sgn
	else:
		return epn, None