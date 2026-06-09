# MPI并行计算版本使用说明

## 概述

本项目已经成功集成了MPI4py和OpenMPI，实现了GPR电磁波双参数FWI的并行计算。主要并行化的部分包括：

1. **正演模拟并行化** (`forward.py`): 多个shot_idx并行计算
2. **梯度计算并行化** (`gradient.py`): 多个shot_idx并行计算
3. **自动并行检测**: 程序会自动检测MPI环境并选择合适的计算模式

## 系统要求

### 必需软件
- Python 3.6+
- OpenMPI 或 MPICH
- mpi4py Python包

### 安装依赖

```bash
# 安装OpenMPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin libopenmpi-dev

# 安装OpenMPI (CentOS/RHEL)
sudo yum install openmpi openmpi-devel

# 安装mpi4py
pip install mpi4py

# 或者使用conda
conda install mpi4py
```

## 使用方法

### 1. 基本运行

```bash
# 使用4个进程运行
mpirun -np 4 python3 run_mpi.py

# 或者使用提供的脚本
chmod +x run_mpi.sh
./run_mpi.sh 4
```

### 2. 进程数选择

- **推荐进程数**: 建议使用2-16个进程
- **最大进程数**: 不应超过shot总数
- **最佳性能**: 通常进程数等于CPU核心数时性能最佳

### 3. 运行示例

```bash
# 使用2个进程
./run_mpi.sh 2

# 使用8个进程
./run_mpi.sh 8

# 使用16个进程
./run_mpi.sh 16
```

## 并行化原理

### 数据分配策略

程序采用**负载均衡**的数据分配策略：

1. **均匀分配**: 每个进程处理大致相同数量的shots
2. **余数处理**: 前`remainder`个进程多处理一个shot
3. **动态分配**: 根据进程数自动调整分配

### 并行计算流程

```
1. 主进程(rank=0)初始化模型和参数
2. 所有进程同步等待
3. 每个进程处理分配的shots:
   - 正演模拟
   - 梯度计算
4. 收集所有进程的结果
5. 主进程合并结果并保存
```

### 通信模式

- **点对点通信**: 使用`comm.gather()`收集结果
- **广播通信**: 使用`comm.bcast()`分发残差数据
- **同步点**: 使用`comm.Barrier()`确保进程同步

## 性能优化建议

### 1. 进程数优化

```python
# 在run_mpi.py中调整
n_shots = 16  # 总炮数
# 建议: 进程数 ≤ n_shots
# 最佳: 进程数 = CPU核心数
```

### 2. 内存优化

- 每个进程独立处理shots，避免内存冲突
- 波场数据按需保存，减少内存占用
- 使用numpy数组的copy()避免引用问题

### 3. 负载均衡

- 程序自动处理负载均衡
- 每个进程处理相似数量的shots
- 支持非整除的进程数分配

## 故障排除

### 常见问题

1. **mpirun命令未找到**
   ```bash
   # 安装OpenMPI
   sudo apt-get install openmpi-bin
   ```

2. **mpi4py导入错误**
   ```bash
   # 安装mpi4py
   pip install mpi4py
   ```

3. **进程数过多**
   - 确保进程数不超过shot总数
   - 检查系统资源限制

4. **内存不足**
   - 减少模型尺寸
   - 减少时间步数
   - 减少shot数量

### 调试模式

```bash
# 启用MPI调试信息
mpirun -np 4 --verbose python3 run_mpi.py

# 查看进程分配
mpirun -np 4 --display-map python3 run_mpi.py
```

## 性能测试

### 测试环境
- CPU: 多核处理器
- 内存: 建议8GB+
- 模型: 100x100网格
- Shots: 16个

### 预期性能提升
- **2进程**: 1.5-2.0x加速
- **4进程**: 2.5-3.5x加速
- **8进程**: 4.0-6.0x加速
- **16进程**: 6.0-10.0x加速

*注意: 实际性能取决于硬件配置和模型复杂度*

## 扩展功能

### 1. 自定义并行策略

可以在`forward_model_mpi`和`compute_gradient_mpi`中修改数据分配策略：

```python
# 自定义分配策略
def custom_distribution(n_shots, size):
    # 实现自定义的负载均衡策略
    pass
```

### 2. 混合并行

结合OpenMP和MPI实现更细粒度的并行化：

```python
# 在时间循环中添加OpenMP并行
# 需要安装numba或使用Cython
```

### 3. GPU加速

集成CUDA或OpenCL实现GPU加速：

```python
# 使用cupy或pycuda
import cupy as cp
```

## 联系和支持

如有问题或建议，请检查：
1. MPI环境配置
2. Python依赖安装
3. 系统资源限制
4. 模型参数设置

---

**注意**: 本并行版本完全兼容原有的串行代码，当MPI不可用时会自动回退到串行模式。
