#!/bin/bash

# MPI并行计算运行脚本
# 使用方法: ./run_mpi.sh [进程数]

# 设置默认进程数
NP=${1:-4}

echo "Starting MPI parallel computation with $NP processes..."
echo "=================================================="

# 检查MPI是否可用
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI or MPICH."
    exit 1
fi

# 检查Python和mpi4py是否可用
python3 -c "import mpi4py" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: mpi4py not found. Please install it with: pip install mpi4py"
    exit 1
fi

# 运行MPI程序
echo "Running: mpirun -np $NP python3 run_mpi.py"
echo "=================================================="

mpirun -np $NP python3 run_mpi.py

echo "=================================================="
echo "MPI computation completed!"
