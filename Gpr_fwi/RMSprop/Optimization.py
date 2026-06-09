#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:46:58 2018

@author: nephilim
"""
import numpy as np
import shutil
import os

from dataclasses import dataclass

@dataclass
class Parameters:
    # 模型参数
    xl: int = 100
    zl: int = 200
    dx: float = 0.02
    dz: float = 0.02
    k_max: int = 1000
    dt: float = 4e-11
    AirLayer: int = 3
    
    # 波形参数
    Freq: float = 4e8
    fs: float = None  # 将在__post_init__中设置
    
    # 多尺度参数
    MultiScale_Key: bool = True
    target_freq: float = None  # 将在运行时设置
    
    # 正则化参数
    Modified_Total_Variation_key: bool = True
    initWeight: float = 0.5
    
    # 源点和接收点
    source_site: list = None
    receiver_site: list = None
    random_source_site: list = None
    random_source_index: list = None
    
    # 源点选择参数
    Source_num_select: int = None  # 添加这个属性
    change_source: int = None      # 添加这个属性
    
    # 数据
    data: np.ndarray = None
    data_eps: np.ndarray = None
    data_sigma: np.ndarray = None
    Original_data: np.ndarray = None
    Air_True_Profile: np.ndarray = None
    Air_True_Profile_Original: np.ndarray = None
    
    def __post_init__(self):
        if self.fs is None:
            self.fs = 1/self.dt
    
class options():
    def __init__(self):
        pass

class Optimization(Parameters,options):
    def __init__(self,fh,iepsilon,isigma):
        super().__init__()
        self.fh=fh
        self.data_eps=iepsilon.flatten() #是传入模型参数的二维数组，flatten()是将二维数组展平为一维数组
        self.data_sigma=isigma.flatten()
        # 从传入的 para 对象复制所有属性
        if hasattr(fh.__globals__.get('para'), 'Source_num_select'):
            self.Source_num_select = fh.__globals__.get('para').Source_num_select
        if hasattr(fh.__globals__.get('para'), 'change_source'):
            self.change_source = fh.__globals__.get('para').change_source
        if hasattr(fh.__globals__.get('para'), 'target_freq'):
            self.target_freq = fh.__globals__.get('para').target_freq
        
    def optimization(self):
        # Set default values for options if they don't exist
        # 1. 设置默认参数
        options.maxiter = getattr(options, 'maxiter', 10) # 最大迭代次数
        options.maxit_ls = getattr(options, 'maxit_ls', 5) # 线搜索最大迭代次数
        #或者我们就不用quai-Newton Method，而是采用类似RMSprop，不需要计算Hessian矩阵，只需要计算梯度，然后更新模型参数，感觉会更加合理。没错！对的！RMSprop是对的！
        options.tol = getattr(options, 'tol', 1e-2) # 收敛容差
        options.c1 = getattr(options, 'c1', 1e-4)  # Wolfe条件参数1
        options.c2 = getattr(options, 'c2', 0.9) # Wolfe条件参数2
        options.ls_int = getattr(options, 'ls_int', 2) # 线搜索插值次数
        options.progTol = getattr(options, 'progTol', 1e-9) # 进度显示容差

        # 2. 创建保存模型的目录
        model_eps_dir = f'./{self.target_freq}Hz_imodel_eps_file_{self.Source_num_select}_{self.change_source}'
        model_sig_dir = f'./{self.target_freq}Hz_imodel_sig_file_{self.Source_num_select}_{self.change_source}'
        if os.path.exists(model_eps_dir):
            shutil.rmtree(model_eps_dir)
        if os.path.exists(model_sig_dir):
            shutil.rmtree(model_sig_dir)
        os.makedirs(model_eps_dir)
        os.makedirs(model_sig_dir)

        x = self.data_eps.copy()
        y = self.data_sigma.copy()
        x=self.__counts_eps(x)
        y=self.__counts_sigma(y)

        n = len(x)

        iter_ = 0 # 迭代指标
        alpha0 = 1.0 # 初始步长
        alpha = 1.0 # 当前步长
        # Print header for the optimization progress table
        print(f'{"iter":>5s}, {"eval":>6s}, {"step length":>15s}, {"function value":>15s}, {"||g_eps||_2":>15s}, {"||g_sigma||_2":>15s}\n')
        
        # Compute initial function and gradient
        #调用fh=misfit函数 计算得到当前模型参数正演结果的misfit和梯度
        f, g_eps,g_sigma = self.fh(x.reshape((self.xl + 20, -1)),y.reshape((self.xl + 20, -1)))

        f0 = f
        fevals = 1
        info = [[iter_, fevals, alpha0, f, np.linalg.norm(g_eps, 2),np.linalg.norm(g_sigma, 2)]]
        
        # Print initial values
        print(f'{iter_:5d}, {fevals:6d}, {alpha0:15.5e}, {f:15.5e}, {np.linalg.norm(g_eps, 2):15.5e}, {np.linalg.norm(g_sigma, 2):15.5e}\n')
        
        #设置RMSprop超参数
        beta_eps = 0.9
        beta_sigma = 0.9
        alpha_eps = 0.005
        alpha_sigma = 0.005
        V = np.zeros_like(x)
        U = np.zeros_like(y)
        #实现RMSprop算法
        for iter_ in range(options.maxiter):
            print(f"RMSprop主循环迭代: {iter_},当前f: {f:.4e}, ||g_eps||: {np.linalg.norm(g_eps):.4e}, ||g_sigma||: {np.linalg.norm(g_sigma):.4e}")
            f, g_eps,g_sigma = self.fh(x.reshape((self.xl + 20, -1)),y.reshape((self.xl + 20, -1)))
            g_eps = g_eps.flatten()
            g_sigma = g_sigma.flatten()

            V = beta_eps * V + (1 - beta_eps) * g_eps**2
            U = beta_sigma * U + (1 - beta_sigma) * g_sigma**2

            d_eps = -alpha_eps * g_eps / np.sqrt(V + 1e-8)
            d_sigma = -alpha_sigma * g_sigma / np.sqrt(U + 1e-8)

            x = x + d_eps
            y = y + d_sigma

            x = self.__counts_eps(x)
            y = self.__counts_sigma(y)

            
            # Save current model
            np.save(f'./{self.target_freq}Hz_imodel_eps_file_{self.Source_num_select}_{self.change_source}/{iter_}_imodel_eps.npy', x)
            np.save(f'./{self.target_freq}Hz_imodel_sig_file_{self.Source_num_select}_{self.change_source}/{iter_}_imodel_sig.npy', y)
            # Record progress
            info.append([iter_, fevals, alpha, f, np.linalg.norm(g_eps, 2),np.linalg.norm(g_sigma, 2)])
            
            # Print progress
            print(f'{iter_ + 1:5d}, {fevals:6d}, {alpha:15.5e}, {f:15.5e}, {np.linalg.norm(g_eps, 2):15.5e}, {np.linalg.norm(g_sigma, 2):15.5e}\n')
            fevals += 1
            print(f"  epsilon范围: {x.min():.4e} ~ {x.max():.4e}, sigma范围: {y.min():.4e} ~ {y.max():.4e}")
            # Check stopping conditions
            if f / f0 < options.tol:
                print('Function Value less than funTol\n')
                break
            if fevals >= 1000:
                print('Reached Maximum Number of Function Evaluations\n')
                break
            if iter_ == options.maxiter:
                print('Reached Maximum Number of Iterations\n')
                break

        return x,y, info
    
    

    #Wolfe线性搜索方法
    def __WolfeLineSearch(self, x, t, d, f, g, gtd, c1, c2, LS_interp, maxLS, fh):
        #这个代码中传入的x是长为2N的向量，t是初始变化步长，d也是长为2N的搜索方向向量，g是长为2N的梯度向量，gtd是标量
        # Initialize new values
        x_new = x + t * d
        #做参数范围限制
        x_new = self.__counts(x_new)
        
        # 计算新模型参数的正演结果
        N=self.data_eps.size
        x_new_eps = x_new[:N]
        x_new_sig = x_new[N:]   
        f_new, g_new_eps,g_new_sig = fh(x_new_eps.reshape((self.xl + 20, -1)),x_new_sig.reshape((self.xl + 20, -1)))
        g_new = np.concatenate((g_new_eps.flatten(),g_new_sig.flatten()))
        
        lsiter = 1
        gtd_new = np.dot(g_new, d)
        LSiter = 0 
        t_prev = 0.0
        f_prev, g_prev, gtd_prev = f, g, gtd
        done = 0

        while LSiter < maxLS:
            # Armijo condition
            if (f_new > f + c1 * t * gtd) or (LSiter > 1 and f_new >= f_prev):
                bracket = np.hstack((t_prev, t))
                bracketFval = np.hstack((f_prev, f_new))
                bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))
                break
            elif abs(gtd_new) <= -c2 * gtd:
                bracket, bracketFval, bracketGval, done = t, f_new, g_new, 1
                break
            elif gtd_new >= 0:
                bracket = np.hstack((t_prev, t))
                bracketFval = np.hstack((f_prev, f_new))
                bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))
                break

            # Update step size
            t_prev, f_prev, g_prev, gtd_prev = t, f_new, g_new, gtd_new
            minStep = t + 0.01 * (t - t_prev)
            maxStep = t * 10
            t = maxStep if LS_interp <= 1 else self.__polyinterp(
                np.array([[t_prev, f_prev, gtd_prev], [t, f_new, gtd_new]]),
                minStep, maxStep
            )

            # Compute new function and gradient values
            x_new = self.__counts(x + t * d)
            N=self.data_eps.size
            x_new_eps = x_new[:N]
            x_new_sig = x_new[N:]   
            f_new, g_new_eps,g_new_sig = fh(x_new_eps.reshape((self.xl + 20, -1)),x_new_sig.reshape((self.xl + 20, -1)))
            g_new = np.concatenate((g_new_eps.flatten(),g_new_sig.flatten()))

            lsiter += 1
            gtd_new = np.dot(g_new, d)
            LSiter += 1

            print(f"  线搜索循环: {LSiter}, 当前alpha: {t:.4e}, f_new: {f_new:.4e}, gtd_new: {gtd_new:.4e}")

        # If max line search iterations reached
        if LSiter == maxLS:
            bracket = np.hstack((0, t))
            bracketFval = np.hstack((f, f_new))
            bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))

        insufProgress = 0
        while not done and LSiter < maxLS:
            LOpos = np.argmin(bracketFval)
            f_LO = bracketFval[LOpos]
            HIpos = -LOpos + 1

            # Interpolation step
            if LS_interp <= 1:
                t = np.mean(bracket)
            else:
                t = self.__polyinterp(
                    np.array([[bracket[0], bracketFval[0], np.dot(bracketGval[:, 0], d)],
                              [bracket[1], bracketFval[1], np.dot(bracketGval[:, 1], d)]]))
                print('Lines Search Grad-Cubic Interpolation Iteration',LSiter,'alpha=',t)
            # Adjust step if insufficient progress
            if np.min((np.max(bracket) - t, t - np.min(bracket))) / (np.max(bracket) - np.min(bracket)) < 0.1:
                if insufProgress or t >= np.max(bracket) or t <= np.min(bracket):
                    t = np.max(bracket) - 0.1 * (np.max(bracket) - np.min(bracket)) \
                        if abs(t - np.max(bracket)) < abs(t - np.min(bracket)) \
                        else np.min(bracket) + 0.1 * (np.max(bracket) - np.min(bracket))
                    insufProgress = 0
                else:
                    insufProgress = 1
            else:
                insufProgress = 0

            # Update function and gradient values
            x_new = self.__counts(x + t * d)
            N=self.data_eps.size
            x_new_eps = x_new[:N]
            x_new_sig = x_new[N:]   
            f_new, g_new_eps,g_new_sig = fh(x_new_eps.reshape((self.xl + 20, -1)),x_new_sig.reshape((self.xl + 20, -1)))
            g_new = np.concatenate((g_new_eps.flatten(),g_new_sig.flatten()))
            lsiter += 1
            gtd_new = np.dot(g_new, d)
            LSiter += 1

            # Check conditions
            armijo = f_new < f + c1 * t * gtd
            if not armijo or f_new >= f_LO:
                bracket[HIpos], bracketFval[HIpos], bracketGval[:, HIpos] = t, f_new, g_new
            else:
                if abs(gtd_new) <= -c2 * gtd:
                    done = 1
                elif gtd_new * (bracket[HIpos] - bracket[LOpos]) >= 0:
                    bracket[HIpos], bracketFval[HIpos], bracketGval[:, HIpos] = bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos]
                bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos] = t, f_new, g_new

        # Final values
        if isinstance(bracket, np.ndarray):
            LOpos = np.argmin(bracketFval)
            t, f_new, g_new = bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos]
        else:
            t, f_new, g_new = bracket, bracketFval, bracketGval

        return t, f_new, g_new, lsiter

    
    def __polyinterp(self, points, *vargs):
        #三次插值计算最优步长
        # Set default bounds and plotting flag
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        
        xminBound = vargs[0] if len(vargs) >= 1 else xmin
        xmaxBound = vargs[1] if len(vargs) >= 2 else xmax

        # Determine minimum position
        minPos = np.argmin(points[:, 0])
        notMinPos = -minPos + 1

        # Handle edge case where the points have the same x value
        if (points[minPos, 0] - points[notMinPos, 0]) == 0:
            return (xmaxBound + xminBound) / 2

        # Calculate intermediate values
        d1 = (points[minPos, 2] + points[notMinPos, 2] -
              3 * (points[minPos, 1] - points[notMinPos, 1]) /
              (points[minPos, 0] - points[notMinPos, 0]))
        
        d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])

        # If d2 is real, calculate the interpolation
        if d2.imag == 0.0:
            t = points[notMinPos, 0] - (points[notMinPos, 0] - points[minPos, 0]) * \
                ((points[notMinPos, 2] + d2.real - d1) /
                 (points[notMinPos, 2] - points[minPos, 2] + 2 * d2.real))

            minPos = np.clip(t, xminBound, xmaxBound)
        else:
            # If d2 is not real, return the midpoint
            minPos = (xmaxBound + xminBound) / 2

        return minPos

    
    def __counts(self, x):
        N = self.data_eps.size
        x_eps = np.clip(x[:N], 1, 81)
        x_sig = np.clip(x[N:], 0, np.inf)
        return np.concatenate([x_eps, x_sig])
    def __counts_eps(self, x):
        #参数约束
        x = np.clip(x, 1, 81)
        return x
    def __counts_sigma(self, x):
        #参数约束
        x = np.clip(x, 0, np.inf)
        return x

        
        
        
        
        
        