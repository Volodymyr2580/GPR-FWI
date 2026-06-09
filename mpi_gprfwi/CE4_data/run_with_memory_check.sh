#!/bin/bash

# MPI GPR FWI 内存优化运行脚本
# 根据系统内存自动选择MPI进程数

echo "=== MPI GPR FWI 内存优化运行脚本 ==="

# 检查系统内存
TOTAL_MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
AVAILABLE_MEMORY_GB=$(free -g | awk 'NR==2{print $7}')

echo "系统总内存: ${TOTAL_MEMORY_GB} GB"
echo "可用内存: ${AVAILABLE_MEMORY_GB} GB"

# 根据内存大小推荐MPI进程数
if [ $AVAILABLE_MEMORY_GB -ge 32 ]; then
    RECOMMENDED_PROCS=16
    echo "推荐使用 16 个MPI进程"
elif [ $AVAILABLE_MEMORY_GB -ge 16 ]; then
    RECOMMENDED_PROCS=8
    echo "推荐使用 8 个MPI进程"
elif [ $AVAILABLE_MEMORY_GB -ge 8 ]; then
    RECOMMENDED_PROCS=4
    echo "推荐使用 4 个MPI进程"
elif [ $AVAILABLE_MEMORY_GB -ge 4 ]; then
    RECOMMENDED_PROCS=2
    echo "推荐使用 2 个MPI进程"
else
    RECOMMENDED_PROCS=1
    echo "内存不足，推荐使用 1 个进程（串行运行）"
fi

echo ""
echo "可用选项："
echo "1. 使用推荐配置 (${RECOMMENDED_PROCS} 进程)"
echo "2. 手动指定进程数"
echo "3. 运行内存监控"
echo "4. 退出"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "使用推荐配置: mpirun -n ${RECOMMENDED_PROCS} python main.py"
        mpirun -n $RECOMMENDED_PROCS python main.py
        ;;
    2)
        read -p "请输入MPI进程数: " num_procs
        echo "运行命令: mpirun -n ${num_procs} python main.py"
        mpirun -n $num_procs python main.py
        ;;
    3)
        echo "启动内存监控..."
        python memory_monitor.py
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

