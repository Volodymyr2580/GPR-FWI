# 快速开始指南

## 修复权限问题

```bash
# 进入目录
cd ~/swz/mpi_gprfwi/CE4_data

# 给脚本添加执行权限（注意：不要包含 $ 符号）
chmod +x run_with_memory_check.sh
```

## 运行方式

### 方式1：使用内存监控脚本（推荐）
```bash
# 先运行内存分析
python memory_monitor.py

# 这会显示：
# - 系统总内存和可用内存
# - 估算的内存需求
# - 是否足够运行
```

### 方式2：手动指定MPI进程数
```bash
# 根据您的系统内存选择合适的进程数：
# - 如果有 32GB+ 内存，可以用：
mpirun -n 8 python main.py

# - 如果有 16GB 内存，建议用：
mpirun -n 4 python main.py

# - 如果有 8GB 内存，建议用：
mpirun -n 2 python main.py

# - 如果内存不足，使用串行：
python main.py
```

### 方式3：使用智能运行脚本
```bash
# 脚本会根据系统内存自动推荐进程数
./run_with_memory_check.sh
```

## 常见命令错误

### 错误示例
```bash
$ chmod +x run_with_memory_check.sh    # ❌ 错误：多了 $ 符号
$: command not found
```

### 正确示例
```bash
chmod +x run_with_memory_check.sh      # ✓ 正确：直接输入命令
```

**注意**：`$` 是命令提示符，不是命令的一部分，复制命令时不要包含它。

## 监控内存使用

程序运行时会自动输出内存监控信息：
```
[内存监控 Epoch 1 开始] 进程内存使用: 2048.3 MB (2.00 GB)
[系统内存] 总内存: 16.0 GB, 可用: 8.2 GB, 使用率: 48.7%
[Forward内存监控 正演开始] 进程内存使用: 2100.5 MB (2.05 GB)
```

如果看到内存使用率超过90%，建议：
1. 减少MPI进程数
2. 停止其他占用内存的程序
3. 考虑减小模型参数

## 完整运行流程

```bash
# 1. 进入目录
cd ~/swz/mpi_gprfwi/CE4_data

# 2. 先检查内存（可选但推荐）
python memory_monitor.py

# 3. 根据内存情况运行
mpirun -n 4 python main.py

# 4. 如果遇到内存问题，减少进程数重试
mpirun -n 2 python main.py
```

## 检查结果

结果保存在 `~/swz/mpi_gprfwi/CE4_data/results/` 目录下。


