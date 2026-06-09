#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:44:55 2021

@author: nephilim
"""

import Forward2D
import AirForward2D
import Calculate_Gradient_NoSave_Pool
import numpy as np
import time
from Optimization import Parameters,options,Optimization
from pathlib import Path
from matplotlib import pyplot,cm
from skimage import transform
import RandomSource
import MultiScale

def expand_list(lst):
    '''
    相当于把原先的迭代次数给展开每一项
    '''
    expanded_list=[] 
    for item in lst:
        if isinstance(item, list):
            n = item[-2]   # 获取列表倒数第二个元素作为展开次数
            for idx in range(n):
                new_item = item.copy()  # 复制原列表
                new_item.append(idx)    # 添加迭代索引
                expanded_list.append(new_item)
        else:
            expanded_list.append(item)
    return expanded_list


if __name__=='__main__':  
    start_time=time.time() 

    para = Parameters()
    # Model Params
    para.xl=100 # x方向网格数（不含CPML）
    para.zl=200 # z方向网格数（不含CPML） 这里定义的z在画出时是横向的（相当于直觉上的x轴，或者可以认为坐标轴是从长方形的左上角延展的，z轴朝右，x轴朝下）
    para.dx=0.02 # x方向网格间距
    para.dz=0.02 # z方向网格间距

    
    para.k_max=1000 # 最大迭代次数？
    
    
    para.dt=4e-11 # 时间步长
    para.AirLayer=3 # 空气层厚度
    # Ricker wavelet main frequence
    para.Freq=4e8 # Ricker子波主频
        
    para.MultiScale_Key=True # 启用多尺度
    para.fs=1/para.dt # 采样频率
    Target_freq=[4e8] # 多尺度频率列表
    
    para.Modified_Total_Variation_key=False # 启用TV正则化
    para.initWeight=0.5 # 正则化权重
    
    # True Model
    epsilon=np.ones((para.xl+20,para.zl+20)) #说明参数部分是不包含CPML层的 CPML上下左右各10个单元
    epsilon[100:,:]=1
    epsilon[40:80,80:120]=2
    CPML=10
    # 扩展CPML边界
    epsilon[:CPML,:]=epsilon[CPML,:] # 将第10行的值复制到0-9行
    epsilon[-CPML:,:]=epsilon[-CPML-1,:] # 将倒数第11行的值复制到最后10行
    epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1)) # 将第10列的值复制到0-9列
    epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1)) # 将倒数第11列的值复制到最后10列
    
    #设置相对介电常数参数
    epsilon[:10+para.AirLayer,:]=1 #这里空气层是加在x方向，
    
    #初始化电导率
    sigma=np.ones((para.xl+20,para.zl+20))*1e-3
    # 扩展CPML边界
    sigma[:CPML,:]=sigma[CPML,:] # 将第10行的值复制到0-9行
    sigma[-CPML:,:]=sigma[-CPML-1,:] # 将倒数第11行的值复制到最后10行
    sigma[:,:CPML]=sigma[:,CPML].reshape((len(sigma[:,CPML]),-1)) # 将第10列的值复制到0-9列
    sigma[:,-CPML:]=sigma[:,-CPML-1].reshape((len(sigma[:,-CPML-1]),-1)) # 将倒数第11列的值复制到最后10列
    sigma[:10+para.AirLayer,:]=0
    
    np.save('true_sigma.npy',sigma)
    np.save('true_epsilon.npy',epsilon)
    #检查是否满足CFL稳定性条件
    fenmu = 3e8/np.sqrt(epsilon.max()) *np.sqrt(1/para.dx**2+1/para.dz**2)
    cfl_condition = para.dt < 1 / fenmu
    print(f"CFL条件: {cfl_condition}")
    #设置相对磁导率参数
    mu=np.ones((para.xl+20,para.zl+20))
    
    # Source Position
    # 设置源点和接收点位置
    para.source_site=[]
    para.receiver_site=[]
    for index in range(10,210,2):
        #源点和接收器在x=10处，z方向每隔2个单元设置一个源点和接收器，一共100个源点和接收器
        para.source_site.append((10,index)) #在第10行 即CPML边界处，横向每2个单元设置一个source和一个receiver
        para.receiver_site.append((10,index))
            
    # Get True Model Data
    # 进行正演模拟获取观测数据，这里用到了Forward2D.py中的Forward_2D函数
    print('Start Forward to get observation data...')
    Forward2D.Forward_2D(sigma.copy(),epsilon.copy(),mu.copy(),para)
        
    print('Forward Done !')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))

    # Save Profile Data
    # 加载正演结果
    #Original_data是正演的record观测数据
    para.Original_data=np.load('./%sHz_forward_data_file/record.npy'%para.Freq)
    # 计算空气层响应并去除
    print('Start Air Forward to get air profile...')
    para.Air_True_Profile_Original=AirForward2D.Forward_2D(np.zeros_like(sigma),np.ones_like(epsilon),np.ones_like(mu),para.Freq,para)
    print('Air Forward Done !')

    np.save('Air_True_Profile_Original.npy',para.Air_True_Profile_Original)

    para.Original_data=para.Original_data-np.tile(para.Air_True_Profile_Original,(para.Original_data.shape[1],1)).T

    np.save('Original_data.npy',para.Original_data)



    # Anonymous function for Gradient Calculate
    #x:epsilon,y:sigma
    fh=lambda x,y:Calculate_Gradient_NoSave_Pool.misfit(y,x,para)  
    
    # FWI Parameters
    FWI_Params=[]
    Init_Params=[]
    # Random source & Change number
    # 随机源参数设置
    Source_num_segments=1     # 源点分段数
    Source_num_start=100      # 起始源点数
    #Source_num_end=22      # 结束源点数
    Source_Segments,Source_base_size=RandomSource.split_data(para.source_site, num_segments=Source_num_segments)#分成1段

    Iteration=1 # 每个频率的迭代次数
    Source_num_select_list=np.unique(np.linspace(Source_num_start,Source_base_size,Iteration).astype('int'))
    print(f'Source_num_select_list: {Source_num_select_list}')
    Source_num_select_Record=Source_num_start

    #对每个频段，迭代次数和每次迭代更新的源点数（？）
    Change_source_num_list=[1,] 
    Change_Iteration_list=[200,] 
    # 多尺度参数设置
    for multi_freq in Target_freq:
        for idx in range(Iteration):
            FWI_Params.append([multi_freq, # 当前频率
                               Source_num_select_list[idx], # 使用的源点数量
                               Change_source_num_list[idx], # 每次更换的源点数
                               Change_Iteration_list[idx] # 迭代次数
                               ]) 
    #最终生成的FWI_Params的每一个元素都是一个列表，包含当前频率，源点数量，每次更换的源点数，迭代次数
    
    # FWI_Params=[[4e8,np.int64(22),5,5]]
    
    
    
    FWI_Params=expand_list(FWI_Params)
    Init_Params=FWI_Params[:-1] # 取除最后一组外的所有参数
    Init_Params.insert(0,'Init') # 在开头插入'Init'标记，用于初始模型
    # Init_Params=expand_list(Init_Params)
    
    print('FWI Parameters:')
    print(FWI_Params)
    print('Init Parameters:')
    print(Init_Params)
    
    # Starting...
    FWI_INFO=[] # 存储反演信息
    for idx_FWI in np.arange(len(Init_Params)):
        # 设置当前频率和源点参数
        para.target_freq=FWI_Params[idx_FWI][0] # 当前使用的频率
        para.Source_num_select=FWI_Params[idx_FWI][1] # 选择的源点数量
        para.change_source=FWI_Params[idx_FWI][4] # 源点更新索引

        # 多尺度处理
        if para.target_freq==para.Freq: # 如果是最高频率
            para.MultiScale_Key=False # 关闭多尺度
            para.data=para.Original_data # 使用原始数据
            para.Air_True_Profile=para.Air_True_Profile_Original
        else: # 如果是低频
            para.MultiScale_Key=True  # 开启多尺度
            para.data=np.zeros_like(para.Original_data)
            # 对每列数据进行频率滤波
            for idx_data_col in np.arange(para.data.shape[1]):
                para.data[:,idx_data_col]=MultiScale.apply_filter(
                    para.Original_data[:,idx_data_col],
                    para.fs,
                    para.target_freq
                    )
            # 对空气层响应也进行相同的滤波
            para.Air_True_Profile=MultiScale.apply_filter(
                para.Air_True_Profile_Original,
                para.fs,
                para.target_freq
                )
        
        # 随机选择源点
        para.random_source_site=RandomSource.random_selection_from_segments(
            Source_Segments,
              num_to_select=para.Source_num_select #10,22
            )
        print(f'para.random_source_site: {para.random_source_site}')
        para.random_source_index=[para.source_site.index(element) for element in para.random_source_site] # 获取若干随机源点在source_site中的索引
        # 初始模型设置
        if Init_Params[idx_FWI]=='Init':# 如果是第一次迭代
            # 创建线性递增的初始模型
            iepsilon=np.ones((para.xl+20,para.zl+20))*6
            iepsilon[100:,:]=1
            # 添加CPML边界
            iepsilon[:CPML,:]=iepsilon[CPML,:]
            iepsilon[-CPML:,:]=iepsilon[-CPML-1,:]
            iepsilon[:,:CPML]=iepsilon[:,CPML].reshape((len(iepsilon[:,CPML]),-1))
            iepsilon[:,-CPML:]=iepsilon[:,-CPML-1].reshape((len(iepsilon[:,-CPML-1]),-1))
            # 设置空气层
            iepsilon[:10+para.AirLayer,:]=1
            
            isigma = np.ones((para.xl+20, para.zl+20)) * 1e-3
            isigma[:10+para.AirLayer,:]=0

            np.save('Initial_model_eps.npy',iepsilon)
            np.save('Initial_model_sig.npy',isigma)

        else:# 使用上一次的反演结果
            # 加载上一次的模型结果
            eps_path='./%sHz_imodel_eps_file_%s_%s'%(
                Init_Params[idx_FWI][0],
                Init_Params[idx_FWI][1],
                Init_Params[idx_FWI][4]
                )
            file_num=int(len(list(Path(eps_path).iterdir())))-1
            data_eps=np.load('./%sHz_imodel_eps_file_%s_%s/%s_imodel.npy'%(
                Init_Params[idx_FWI][0],
                Init_Params[idx_FWI][1],
                Init_Params[idx_FWI][4],
                file_num
                ))
            iepsilon=data_eps.reshape((para.xl+20,-1))

            sig_path='./%sHz_imodel_sig_file_%s_%s'%(
                Init_Params[idx_FWI][0],
                Init_Params[idx_FWI][1],
                Init_Params[idx_FWI][4]
                )
            file_num=int(len(list(Path(sig_path).iterdir())))-1
            data_sig=np.load('./%sHz_imodel_sig_file_%s_%s/%s_imodel.npy'%(
                Init_Params[idx_FWI][0],
                Init_Params[idx_FWI][1],
                Init_Params[idx_FWI][4],
                file_num
                ))
            isigma=data_sig.reshape((para.xl+20,-1))
                

                
        # # Test Gradient
        # f,g=Calculate_Gradient_NoSave_Pool.misfit(sigma.copy(),iepsilon.copy(),para)
        # pyplot.figure()
        # pyplot.imshow(g.reshape((120,-1)))
        
        # Options Params
        # 设置优化参数并执行优化
        options.maxiter=FWI_Params[idx_FWI][3] # 设置最大迭代次数
        Optimization_=Optimization(fh,iepsilon.copy(),isigma.copy()) # 创建优化器
        # 复制当前参数
        Optimization_.target_freq = para.target_freq
        Optimization_.Source_num_select = para.Source_num_select
        Optimization_.change_source = para.change_source
        
        #options.beta = 0.05

        result_eps,result_sig,info=Optimization_.optimization() # 执行优化
        FWI_INFO.append(info)# 保存优化信息
        # # 绘制当前反演结果
        # pyplot.figure()
        # pyplot.imshow(result_eps.reshape((para.xl+20,-1)),cmap=cm.jet)
        # pyplot.colorbar()
         
    # Plot Error Data 绘制误差曲线
    pyplot.figure()
    data_=[]
    for info in FWI_INFO:
        for info_ in info:
            data_.append(info_[3])# 提取误差信息
    # pyplot.plot(data_)
    # pyplot.yscale('log') # 使用对数坐标
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
    np.save('Loss.npy',data_) # 保存误差数据
    
    # 保存反演历史
    with open('history.txt','w') as fid:
        for dd in FWI_INFO:
            fid.write(str(dd))
            fid.write('\n')
            
    with open('history.txt', 'r') as fid:
        read_list = [eval(line.strip()) for line in fid if line.strip()]
