#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:20:43 2021

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop
import Modified_TV
import MultiScale

def calculate_gradient(sigma,epsilon,mu,index,CPML_Params,para):
    # 参数说明：
    # sigma: 电导率
    # epsilon: 相对介电常数
    # mu: 磁导率
    # index: 当前处理的源点索引
    # CPML_Params: CPML边界条件参数
    # para: 全局参数对象

    #Get Forward Params
    k_max=para.k_max
    ep0 = 8.841941282883074e-12
    #生成迭代时间序列
    t=np.arange(k_max)*para.dt #时间窗总长t = 迭代步长*迭代步数
    f=Wavelet.ricker(t,para.Freq) #生成Ricker小波

    if para.MultiScale_Key:
        #多尺度则按照目标频率来滤波。输入的频率target_freq是截止频率，fs是采样频率
        f=MultiScale.apply_filter(f, para.fs, para.target_freq)
    #True Model Profile Data
    #这里的index是随机源的索引，所以只需要找到选中的随机源的数据进行比对求loss和residual即可

    # 获取观测数据
    data_obs=para.data[:,index]
    #Get Forward Data ----> <Generator>
    #正演得到当前模型的波场
    Forward_data=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                     sigma.copy(),epsilon.copy(),mu.copy(),CPML_Params,f,k_max,\
                                     para.source_site[index],para.receiver_site[index])
    #Get Generator Data
    For_data=[]
    idata=np.zeros(para.k_max)
    for idx in range(para.k_max):
        
        tmp=Forward_data.__next__()
        For_data.append(np.array(tmp[0]))
        idata[idx]=tmp[1]
    idata-=para.Air_True_Profile #去除空气层响应
    #Get Residual Data
    #residual = d_est - d_obs
    rhs_data=idata-data_obs
    #Get Reversion Data ----> <Generator>
    Reverse_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                     para.dt,sigma.copy(),epsilon.copy(),\
                                                     mu.copy(),CPML_Params,k_max,\
                                                     para.receiver_site[index],rhs_data)
    #Get Generator Data
    # 收集反演结果
    RT_data=[]
    for i in range(para.k_max):
        tmp=Reverse_data.__next__()
        #这么看tmp[0]应该是每一时间步上的波场
        RT_data.append(np.array(tmp[0]))
    # 反转数据（因为tmp[0]对应的是t_max
    RT_data.reverse()
    
    time_sum_eps=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))
    time_sum_sigma=np.zeros((para.xl+2*CPML_Params.npml,para.zl+2*CPML_Params.npml))

    # 1. 检查时间步长
    if para.dt <= 0:
        raise ValueError("时间步长必须大于0")
    # 2. 检查波场数据
    for k in range(1,k_max-1):
        u1 = For_data[k+1]
        us = For_data[k]
        u0 = For_data[k-1]
        p1 = RT_data[k]
        
        
        # 3. 分步计算避免溢出
        # 计算时间差分
        
        # 计算梯度贡献
        eps_contribution = p1 * (u1 - u0) /para.dt/2
        sigma_contribution = p1 * us
        
        
        # 累加梯度
        time_sum_eps += eps_contribution
        time_sum_sigma += sigma_contribution
        
    
    # 4. 检查最终梯度
    g_eps = ep0 * time_sum_eps
    g_sigma = time_sum_sigma
    
    # 去除空气层
    g_eps[:10+para.AirLayer,:] = 0
    g_sigma[:10+para.AirLayer,:] = 0
    
    
    return rhs_data.flatten(),g_eps.flatten(),g_sigma.flatten()

#Calculate Modified-Total-Variation
def calculate_mtv_model(epsilon):
    u_epsilon=Modified_TV.denoising_2D_TV(epsilon)
    return u_epsilon

def calculate_mtv_penalty_data(epsilon,u_epsilon):
    epsilon_rhs=epsilon-u_epsilon
    f_epsilon=0.5*np.linalg.norm(epsilon_rhs.flatten(),2)**2
    g_epsilon=2*(epsilon-u_epsilon)
    return f_epsilon,g_epsilon.flatten()


#用到的fh函数
def misfit(sigma,epsilon,para): 
    mu=np.ones((para.xl+20,para.zl+20))
    start_time=time.time()  
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,sigma.copy(),epsilon.copy(),mu.copy(),para.dx,para.dz,para.dt)

    g_eps=0.0
    g_sigma=0.0
    rhs=[]
    pool=Pool(processes=64)
    res_l=[]
    
    for index,value in zip(para.random_source_index,para.random_source_site):
        #将进程池中的进程分配给每个随机源，并行计算梯度
        res=pool.apply_async(calculate_gradient,args=(sigma.copy(),epsilon.copy(),mu.copy(),index,CPML_Params,para))
        res_l.append(res)
    pool.close()# 不再接受新任务
    pool.join()# 等待所有任务完成

    for res in res_l:
        result = res.get()
        rhs.append(result[0])  # <--- 这一行是关键
        g_eps_temp = np.nan_to_num(result[1])
        g_sigma_temp = np.nan_to_num(result[2])
        
            
        g_eps += g_eps_temp
        g_sigma += g_sigma_temp
    
    rhs=np.array(rhs)# 将残差列表转换为numpy数组

    f=0.5*np.linalg.norm(rhs.flatten(),2)**2 #.flatten()将数组展平为一维数组, 然后计算L2范数的一半 ---f，相当于目标函数值了。

    
    #Get Modified Toltal Variation
    #原先梯度加上MTV正则化约束的惩罚
    if para.Modified_Total_Variation_key:
        mtv_time=time.time()
        u_epsilon=calculate_mtv_model(epsilon.copy())
        f_mtv_penalty_epsilon,g_mtv_penalty_epsilon=calculate_mtv_penalty_data(epsilon,u_epsilon)
        print('mtv function elapsed time is %s seconds !'%str(time.time()-mtv_time))
    
        print('''****fd=%s,g=%s,f_mtv_penalty_epsilon=%s,g_mtv_penalty_epsilon=%s'''\
              %(f,np.linalg.norm(g_eps,2),f_mtv_penalty_epsilon,np.linalg.norm(g_mtv_penalty_epsilon,2)))
            #Update Lambda
        lambda_=(np.linalg.norm(g_eps.flatten(),2))/(np.linalg.norm(g_mtv_penalty_epsilon.flatten(),2))*para.initWeight
        
        f+=lambda_*f_mtv_penalty_epsilon
        g_eps+=lambda_*g_mtv_penalty_epsilon
        
        
    pool.terminate() 
#    print('**********',lambda_,'**********')
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    g_eps = g_eps.reshape(para.xl+20,-1)
    g_sigma = g_sigma.reshape(para.xl+20,-1)
    # 在返回前检查最终梯度
    if np.any(np.isnan(g_eps)) or np.any(np.isinf(g_eps)):
        print("Warning: NaN or Inf in final g_eps")
        g_eps = np.nan_to_num(g_eps)
    #返回的是misfit函数值和梯度值
    return f,g_eps,g_sigma