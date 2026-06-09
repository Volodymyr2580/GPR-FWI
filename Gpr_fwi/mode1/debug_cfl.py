import numpy as np

# 参数设置
xl, zl = 100, 200
dx, dz = 0.02, 0.02
dt = 4e-11
npml = 10
freq = 4e8
steps = 1000

# 修复后的模型参数
epsilon_true = np.ones((xl, zl))
epsilon_true[40:80, 80:120] = 2  # 异常体

# CFL条件检查
c = 3e8  # 光速
epsilon_max = epsilon_true.max()
print(f"epsilon_max: {epsilon_max}")

# 计算CFL条件
cfl_dt_max = 1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))
print(f"当前dt: {dt}")
print(f"最大允许dt: {cfl_dt_max}")
print(f"CFL条件满足: {dt < cfl_dt_max}")

# 检查震源参数
from Wavelet import ricker
t = np.arange(0, steps * dt, dt)
wavelet = ricker(t, freq)
print(f"Ricker子波最大值: {np.max(np.abs(wavelet))}")
print(f"Ricker子波最小值: {np.min(wavelet)}")

# 检查模型参数范围
print(f"epsilon范围: [{epsilon_true.min()}, {epsilon_true.max()}]")
print(f"sigma范围: [{1e-3}, {1e-3}]")

# 计算震源项的大小
ep0 = 8.841941282883074e-12
mu0 = 1.2566370614359173e-06

# 模拟CPML参数计算
epsilon_extended = np.ones((xl + 2*npml, zl + 2*npml))
epsilon_extended[npml:npml+xl, npml:npml+zl] = epsilon_true
epsilon_extended[:npml, :] = epsilon_extended[npml, :]
epsilon_extended[-npml:, :] = epsilon_extended[-npml-1, :]
epsilon_extended[:, :npml] = epsilon_extended[:, npml].reshape(-1, 1)
epsilon_extended[:, -npml:] = epsilon_extended[:, -npml-1].reshape(-1, 1)

sigma_extended = np.ones((xl + 2*npml, zl + 2*npml)) * 1e-3
sigma_extended[npml:npml+xl, npml:npml+zl] = 1e-3

# 计算cb参数
epsilon_extended *= ep0
ca = (1-sigma_extended*dt/2/epsilon_extended)/(1+sigma_extended*dt/2/epsilon_extended)
cb = 1/epsilon_extended/(1+sigma_extended*dt/2/epsilon_extended)

print(f"cb参数范围: [{cb.min()}, {cb.max()}]")

# 计算震源项
source_pos = (10 + npml, 10 + npml)  # 第一个源点
source_term = -cb[source_pos[0], source_pos[1]] * wavelet[0] * dt / dx / dz
print(f"震源项大小: {source_term}")

# 检查数值稳定性
print(f"dt/dx/dz: {dt/dx/dz}")
print(f"cb * dt/dx/dz: {cb[source_pos[0], source_pos[1]] * dt / dx / dz}")

# 检查波场更新的数值范围
print(f"ca参数范围: [{ca.min()}, {ca.max()}]")
print(f"ca * dt范围: [{ca.min() * dt}, {ca.max() * dt}]") 