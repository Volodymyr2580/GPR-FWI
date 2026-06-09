#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:26:22 2024

@author: nephilim
"""

import numpy as np
from scipy.signal import firwin,lfilter

def design_fir_filter(cutoff, fs, numtaps):
    # 参数说明：
    # cutoff: 截止频率
    # fs: 采样频率
    # numtaps: 滤波器阶数
    cutoff = cutoff
    # firwin: 使用汉明窗设计低通FIR滤波器
    return firwin(numtaps, cutoff, window='hamming', fs=fs)

def apply_filter(data, fs, cutoff):
    # 参数说明：
    # data: 输入数据
    # fs: 采样频率
    # cutoff: 截止频率
    # 计算滤波器阶数
    numtaps = int(1 * (fs / (cutoff)))
    # 设计低通FIR滤波器
    fir_coeff = design_fir_filter(cutoff, fs, numtaps)
    # 应用滤波器
    filtered_data = lfilter(fir_coeff, 1.0, data)
    return filtered_data

def normalize_amplitude(original_data, filtered_data):
    # 计算归一化系数
    scale_factor = np.max(np.abs(original_data)) / np.max(np.abs(filtered_data))
    # 归一化
    return filtered_data * scale_factor