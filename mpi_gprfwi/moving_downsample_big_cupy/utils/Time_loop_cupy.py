import cupy as cp
import numpy as np


def time_loop(xl,zl,dx,dz,dt,sigma,epsilon,mu,CPML_Params,f,k_max,source_site,ref_pos):
    ep0 = 8.841941282883074e-12
    mu0 = 1.2566370614359173e-06
    npml = CPML_Params.npml
    x_len = xl + 2*npml
    z_len = zl + 2*npml
    epsilon = cp.asarray(epsilon.copy())*ep0
    mu = cp.asarray(mu.copy())*mu0
    sigma = cp.asarray(sigma.copy())
    ca = cp.asarray(CPML_Params.ca)
    cb = cp.asarray(CPML_Params.cb)
    a_x = cp.asarray(CPML_Params.a_x)
    b_x = cp.asarray(CPML_Params.b_x)
    k_x = cp.asarray(CPML_Params.k_x)
    a_z = cp.asarray(CPML_Params.a_z)
    b_z = cp.asarray(CPML_Params.b_z)
    k_z = cp.asarray(CPML_Params.k_z)
    a_x_half = cp.asarray(CPML_Params.a_x_half)
    b_x_half = cp.asarray(CPML_Params.b_x_half)
    k_x_half = cp.asarray(CPML_Params.k_x_half)
    a_z_half = cp.asarray(CPML_Params.a_z_half)
    b_z_half = cp.asarray(CPML_Params.b_z_half)
    k_z_half = cp.asarray(CPML_Params.k_z_half)
    Ey = cp.zeros((x_len, z_len))
    Hz = cp.zeros((x_len, z_len))
    Hx = cp.zeros((x_len, z_len))
    memory_dEy_dx = cp.zeros((2*npml, z_len))
    memory_dEy_dz = cp.zeros((x_len, 2*npml))
    memory_dHz_dx = cp.zeros((2*npml, z_len))
    memory_dHx_dz = cp.zeros((x_len, 2*npml))
    f_arr = cp.asarray(f)
    sx, sz = source_site
    rx, rz = ref_pos
    for tt in range(int(k_max)):
        dEy_dx = cp.zeros_like(Ey)
        dEy_dx[1:-1, :] = (Ey[2:, :] - Ey[1:-1, :]) / dx
        dEy_dz = cp.zeros_like(Ey)
        dEy_dz[:, 1:-1] = (Ey[:, 2:] - Ey[:, 1:-1]) / dz

        iL = slice(0, npml)
        iR = slice(x_len - npml, x_len)
        iI = slice(npml, x_len - npml)
        jT = slice(0, npml)
        jB = slice(z_len - npml, z_len)
        jI = slice(npml, z_len - npml)

        mem_dx_L = memory_dEy_dx[:npml, :]
        mem_dx_R = memory_dEy_dx[npml:, :]
        mem_dz_T = memory_dEy_dz[:, :npml]
        mem_dz_B = memory_dEy_dz[:, npml:]

        mem_dx_L[:, jI] = (b_x_half[:npml, None] * mem_dx_L[:, jI]) + (a_x_half[:npml, None] * dEy_dx[iL, jI])
        adj_dx_L = (dEy_dx[iL, jI] / k_x_half[:npml, None]) + mem_dx_L[:, jI]
        Hz[iL, jI] += adj_dx_L * dt / mu[iL, jI]

        mem_dx_R[:, jI] = (b_x_half[-npml:, None] * mem_dx_R[:, jI]) + (a_x_half[-npml:, None] * dEy_dx[iR, jI])
        adj_dx_R = (dEy_dx[iR, jI] / k_x_half[-npml:, None]) + mem_dx_R[:, jI]
        Hz[iR, jI] += adj_dx_R * dt / mu[iR, jI]

        Hz[iI, jI] += dEy_dx[iI, jI] * dt / mu[iI, jI]

        mem_dz_T[iI, :] = (b_z_half[:npml][None, :] * mem_dz_T[iI, :]) + (a_z_half[:npml][None, :] * dEy_dz[iI, jT])
        adj_dz_T = (dEy_dz[iI, jT] / k_z_half[:npml][None, :]) + mem_dz_T[iI, :]
        Hx[iI, jT] -= adj_dz_T * dt / mu[iI, jT]

        mem_dz_B[iI, :] = (b_z_half[-npml:][None, :] * mem_dz_B[iI, :]) + (a_z_half[-npml:][None, :] * dEy_dz[iI, jB])
        adj_dz_B = (dEy_dz[iI, jB] / k_z_half[-npml:][None, :]) + mem_dz_B[iI, :]
        Hx[iI, jB] -= adj_dz_B * dt / mu[iI, jB]

        Hx[iI, jI] -= dEy_dz[iI, jI] * dt / mu[iI, jI]

        dHz_dx = cp.zeros_like(Hz)
        dHz_dx[1:-1, :] = (Hz[1:-1, :] - Hz[:-2, :]) / dx
        dHx_dz = cp.zeros_like(Hx)
        dHx_dz[:, 1:-1] = (Hx[:, 1:-1] - Hx[:, :-2]) / dz

        mem_Hdx_L = memory_dHz_dx[:npml, :]
        mem_Hdx_R = memory_dHz_dx[npml:, :]
        mem_Hdz_T = memory_dHx_dz[:, :npml]
        mem_Hdz_B = memory_dHx_dz[:, npml:]

        mem_Hdx_L[:, jI] = (b_x[:npml, None] * mem_Hdx_L[:, jI]) + (a_x[:npml, None] * dHz_dx[iL, jI])
        adj_Hdx_L = (dHz_dx[iL, jI] / k_x[:npml, None]) + mem_Hdx_L[:, jI]
        Ey[iL, jI] = ca[iL, jI] * Ey[iL, jI] + cb[iL, jI] * (adj_Hdx_L - dHx_dz[iL, jI]) * dt

        mem_Hdx_R[:, jI] = (b_x[-npml:, None] * mem_Hdx_R[:, jI]) + (a_x[-npml:, None] * dHz_dx[iR, jI])
        adj_Hdx_R = (dHz_dx[iR, jI] / k_x[-npml:, None]) + mem_Hdx_R[:, jI]
        Ey[iR, jI] = ca[iR, jI] * Ey[iR, jI] + cb[iR, jI] * (adj_Hdx_R - dHx_dz[iR, jI]) * dt

        mem_Hdz_T[iI, :] = (b_z[:npml][None, :] * mem_Hdz_T[iI, :]) + (a_z[:npml][None, :] * dHx_dz[iI, jT])
        adj_Hdz_T = (dHx_dz[iI, jT] / k_z[:npml][None, :]) + mem_Hdz_T[iI, :]
        Ey[iI, jT] = ca[iI, jT] * Ey[iI, jT] + cb[iI, jT] * (dHz_dx[iI, jT] - adj_Hdz_T) * dt

        mem_Hdz_B[iI, :] = (b_z[-npml:][None, :] * mem_Hdz_B[iI, :]) + (a_z[-npml:][None, :] * dHx_dz[iI, jB])
        adj_Hdz_B = (dHx_dz[iI, jB] / k_z[-npml:][None, :]) + mem_Hdz_B[iI, :]
        Ey[iI, jB] = ca[iI, jB] * Ey[iI, jB] + cb[iI, jB] * (dHz_dx[iI, jB] - adj_Hdz_B) * dt

        Ey[iI, jI] = ca[iI, jI] * Ey[iI, jI] + cb[iI, jI] * (dHz_dx[iI, jI] - dHx_dz[iI, jI]) * dt
        Ey[sx,sz] += -cb[sx,sz]*f_arr[tt]*dt/dx/dz
        yield Ey, Ey[rx,rz]

def reverse_time_loop(xl,zl,dx,dz,dt,sigma,epsilon,mu,CPML_Params,k_max,ref_pos,rhs_data):
    ep0 = 8.841941282883074e-12
    mu0 = 1.2566370614359173e-06
    npml = CPML_Params.npml
    x_len = xl + 2*npml
    z_len = zl + 2*npml
    epsilon = cp.asarray(epsilon.copy())*ep0
    mu = cp.asarray(mu.copy())*mu0
    sigma = cp.asarray(sigma.copy())
    ca = cp.asarray(CPML_Params.ca_r)
    cb = cp.asarray(CPML_Params.cb)
    a_x = cp.asarray(CPML_Params.a_x)
    b_x = cp.asarray(CPML_Params.b_x)
    k_x = cp.asarray(CPML_Params.k_x)
    a_z = cp.asarray(CPML_Params.a_z)
    b_z = cp.asarray(CPML_Params.b_z)
    k_z = cp.asarray(CPML_Params.k_z)
    a_x_half = cp.asarray(CPML_Params.a_x_half)
    b_x_half = cp.asarray(CPML_Params.b_x_half)
    k_x_half = cp.asarray(CPML_Params.k_x_half)
    a_z_half = cp.asarray(CPML_Params.a_z_half)
    b_z_half = cp.asarray(CPML_Params.b_z_half)
    k_z_half = cp.asarray(CPML_Params.k_z_half)
    Ey = cp.zeros((x_len, z_len))
    Hz = cp.zeros((x_len, z_len))
    Hx = cp.zeros((x_len, z_len))
    memory_dEy_dx = cp.zeros((2*npml, z_len))
    memory_dEy_dz = cp.zeros((x_len, 2*npml))
    memory_dHz_dx = cp.zeros((2*npml, z_len))
    memory_dHx_dz = cp.zeros((x_len, 2*npml))
    rhs = cp.asarray(rhs_data)
    rx, rz = ref_pos
    for tt in range(int(k_max)):
        Ey[rx,rz] -= rhs[int(k_max)-tt-1]
        dEy_dx = cp.zeros_like(Ey)
        dEy_dx[1:-1, :] = (Ey[2:, :] - Ey[1:-1, :]) / dx
        dEy_dz = cp.zeros_like(Ey)
        dEy_dz[:, 1:-1] = (Ey[:, 2:] - Ey[:, 1:-1]) / dz

        iL = slice(0, npml)
        iR = slice(x_len - npml, x_len)
        iI = slice(npml, x_len - npml)
        jT = slice(0, npml)
        jB = slice(z_len - npml, z_len)
        jI = slice(npml, z_len - npml)

        mem_dx_L = memory_dEy_dx[:npml, :]
        mem_dx_R = memory_dEy_dx[npml:, :]
        mem_dz_T = memory_dEy_dz[:, :npml]
        mem_dz_B = memory_dEy_dz[:, npml:]

        mem_dx_L[:, jI] = (b_x_half[:npml, None] * mem_dx_L[:, jI]) + (a_x_half[:npml, None] * dEy_dx[iL, jI])
        adj_dx_L = (dEy_dx[iL, jI] / k_x_half[:npml, None]) + mem_dx_L[:, jI]
        Hz[iL, jI] += adj_dx_L * dt / mu[iL, jI]

        mem_dx_R[:, jI] = (b_x_half[-npml:, None] * mem_dx_R[:, jI]) + (a_x_half[-npml:, None] * dEy_dx[iR, jI])
        adj_dx_R = (dEy_dx[iR, jI] / k_x_half[-npml:, None]) + mem_dx_R[:, jI]
        Hz[iR, jI] += adj_dx_R * dt / mu[iR, jI]

        Hz[iI, jI] += dEy_dx[iI, jI] * dt / mu[iI, jI]

        mem_dz_T[iI, :] = (b_z_half[:npml][None, :] * mem_dz_T[iI, :]) + (a_z_half[:npml][None, :] * dEy_dz[iI, jT])
        adj_dz_T = (dEy_dz[iI, jT] / k_z_half[:npml][None, :]) + mem_dz_T[iI, :]
        Hx[iI, jT] -= adj_dz_T * dt / mu[iI, jT]

        mem_dz_B[iI, :] = (b_z_half[-npml:][None, :] * mem_dz_B[iI, :]) + (a_z_half[-npml:][None, :] * dEy_dz[iI, jB])
        adj_dz_B = (dEy_dz[iI, jB] / k_z_half[-npml:][None, :]) + mem_dz_B[iI, :]
        Hx[iI, jB] -= adj_dz_B * dt / mu[iI, jB]

        Hx[iI, jI] -= dEy_dz[iI, jI] * dt / mu[iI, jI]

        dHz_dx = cp.zeros_like(Hz)
        dHz_dx[1:-1, :] = (Hz[1:-1, :] - Hz[:-2, :]) / dx
        dHx_dz = cp.zeros_like(Hx)
        dHx_dz[:, 1:-1] = (Hx[:, 1:-1] - Hx[:, :-2]) / dz

        mem_Hdx_L = memory_dHz_dx[:npml, :]
        mem_Hdx_R = memory_dHz_dx[npml:, :]
        mem_Hdz_T = memory_dHx_dz[:, :npml]
        mem_Hdz_B = memory_dHx_dz[:, npml:]

        mem_Hdx_L[:, jI] = (b_x[:npml, None] * mem_Hdx_L[:, jI]) + (a_x[:npml, None] * dHz_dx[iL, jI])
        adj_Hdx_L = (dHz_dx[iL, jI] / k_x[:npml, None]) + mem_Hdx_L[:, jI]
        Ey[iL, jI] = ca[iL, jI] * Ey[iL, jI] + cb[iL, jI] * (adj_Hdx_L - dHx_dz[iL, jI]) * dt

        mem_Hdx_R[:, jI] = (b_x[-npml:, None] * mem_Hdx_R[:, jI]) + (a_x[-npml:, None] * dHz_dx[iR, jI])
        adj_Hdx_R = (dHz_dx[iR, jI] / k_x[-npml:, None]) + mem_Hdx_R[:, jI]
        Ey[iR, jI] = ca[iR, jI] * Ey[iR, jI] + cb[iR, jI] * (adj_Hdx_R - dHx_dz[iR, jI]) * dt

        mem_Hdz_T[iI, :] = (b_z[:npml][None, :] * mem_Hdz_T[iI, :]) + (a_z[:npml][None, :] * dHx_dz[iI, jT])
        adj_Hdz_T = (dHx_dz[iI, jT] / k_z[:npml][None, :]) + mem_Hdz_T[iI, :]
        Ey[iI, jT] = ca[iI, jT] * Ey[iI, jT] + cb[iI, jT] * (dHz_dx[iI, jT] - adj_Hdz_T) * dt

        mem_Hdz_B[iI, :] = (b_z[-npml:][None, :] * mem_Hdz_B[iI, :]) + (a_z[-npml:][None, :] * dHx_dz[iI, jB])
        adj_Hdz_B = (dHx_dz[iI, jB] / k_z[-npml:][None, :]) + mem_Hdz_B[iI, :]
        Ey[iI, jB] = ca[iI, jB] * Ey[iI, jB] + cb[iI, jB] * (dHz_dx[iI, jB] - adj_Hdz_B) * dt

        Ey[iI, jI] = ca[iI, jI] * Ey[iI, jI] + cb[iI, jI] * (dHz_dx[iI, jI] - dHx_dz[iI, jI]) * dt
        yield Ey