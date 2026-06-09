#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 06:31:58 2024

@author: nephilim
"""

import numpy as np
import random

def split_data(data, num_segments):
    '''
    输入：源点列表，段数
    输出：源点列表分成多个段，每段大小尽量均匀
    '''
    
    arr_len = len(data) # 获取源点列表总长度
    base_size = int(np.ceil(arr_len / num_segments)) # 计算每段的基本大小，向上取整
    remainder = arr_len % base_size # 计算余数

    segments = [] # 用于存储分段后的结
    start = 0 # 记录每段的起始位置
    for i in range(num_segments):
        # 如果是前num_segments-1段，使用base_size
        # 最后一段使用remainder
        size = base_size if i < num_segments else remainder
        segments.append(data[start:start + size])
        start += size

    return segments,base_size

def random_selection_from_segments(segments, num_to_select=1):# 输入参数:
    # segments: 分段后的源点列表
    # num_to_select: 每段要选择的源点数量，默认为1

    last_segment = segments[-1] # 获取最后一段
    selected_elements = [] # 存储选中的源点
    
    # 确定最后一段要选择的源点数量
    # 如果最后一段长度小于num_to_select，则全部选择
    remaining_to_select = len(last_segment) if len(last_segment)<num_to_select else num_to_select


    # 处理除最后一段外的所有段
    for segment in segments[:-1]:
        # 从每段中随机选择num_to_select个源点
        selected_element = random.sample(list(segment), num_to_select)
        # 按照原始顺序排序选中的源点
        selected_element.sort(key=lambda x: list(segment).index(x))
        # 将选中的源点添加到结果列表
        selected_elements.extend(selected_element)
    
    # 处理最后一段
    segment=segments[-1]
    # 从最后一段中随机选择remaining_to_select个源点
    selected_element = random.sample(list(segment), remaining_to_select)
    # 按照原始顺序排序选中的源点
    selected_element.sort(key=lambda x: list(segment).index(x))
    # 将选中的源点添加到结果列表
    selected_elements.extend(selected_element)
    
    return selected_elements