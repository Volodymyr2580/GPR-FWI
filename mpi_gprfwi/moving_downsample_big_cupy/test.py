import numpy as np
import cupy as cp
from utils.Time_loop import time_loop as time_loop_cpu
from utils.Time_loop_cupy import time_loop as time_loop_gpu
from utils.Add_CPML import Add_CPML

xl, zl = 100, 120
dx, dz = 0.02, 0.02
dt = 4e-11
npml = 10
steps = 400
# 简单模型
epsilon = np.ones((xl, zl))*3.0
sigma = np.ones((xl, zl))*1e-4
mu = np.ones((xl, zl))
# 扩展到含PML边界
epsilon_ext = np.pad(epsilon, ((npml,npml),(npml,npml)), mode='edge')
sigma_ext   = np.pad(sigma,   ((npml,npml),(npml,npml)), mode='edge')
mu_ext      = np.pad(mu,      ((npml,npml),(npml,npml)), mode='edge')
# CPML参数（NumPy）
cpml = Add_CPML(xl, zl, sigma_ext.copy(), epsilon_ext.copy(), mu_ext.copy(), dx, dz, dt)
# Ricker子波
f0 = 4e8
t  = np.arange(steps)*dt
f  = (1 - 2*(np.pi*f0*(t-6/f0))**2) * np.exp(- (np.pi*f0*(t-6/f0))**2)

src = (npml+5, npml+5)
rec = (npml+10, npml+60)

# CPU 版（NumPy）
gen_cpu = time_loop_cpu(xl, zl, dx, dz, dt, sigma_ext.copy(), epsilon_ext.copy(), mu_ext.copy(), cpml, f, steps, src, rec)
d_cpu = np.array([next(gen_cpu)[1] for _ in range(steps)])

# GPU 版（CuPy）
gen_gpu = time_loop_gpu(xl, zl, dx, dz, dt, cp.asarray(sigma_ext.copy()), cp.asarray(epsilon_ext.copy()), cp.asarray(mu_ext.copy()), cpml, f, steps, src, rec)
d_gpu = cp.asnumpy(cp.array([next(gen_gpu)[1] for _ in range(steps)]))

print("max abs diff:", np.max(np.abs(d_cpu - d_gpu)))
print("L2 rel error:", np.linalg.norm(d_cpu - d_gpu) / (np.linalg.norm(d_cpu) + 1e-12))