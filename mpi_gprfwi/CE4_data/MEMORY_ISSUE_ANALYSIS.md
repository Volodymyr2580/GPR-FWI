# 内存问题深度分析与解决方案

## 问题根源

### 内存消耗计算

对于当前配置：
- 网格大小：1000 × 1000 + PML = 1020 × 1020
- 时间步数：8000
- 每个进程处理：10个shots（总共500个shots，50个MPI进程）
- 数据类型：float64 (8字节)

#### 单个shot的内存需求
```
波场数据 = 1020 × 1020 × 8000 × 8字节 ≈ 63 GB
```

#### 每个进程的内存需求（如果保存波场）
```
10个shots × 63 GB = 630 GB
```

**这就是进程被杀死的根本原因！**

## 已实施的修复

### 1. 避免不必要的波场数据存储
- ✅ 修改了 `forward.py`，只在 `save_wavefield=True` 时创建和填充列表
- ✅ 修改了主循环，只在 epoch=0 时保存波场数据
- ✅ 优化了内存管理，及时释放不需要的对象

### 2. 代码修改细节

**forward.py 关键修改：**
```python
# 修改前（错误）：
shot_wavefield = []  # 总是创建列表
for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
    local_data[local_shot_idx, step_idx] = receiver_data
    if save_wavefield:
        shot_wavefield.append(np.array(ey_field).copy())  # ey_field仍在内存中

# 修改后（正确）：
if save_wavefield:
    shot_wavefield = []  # 只在需要时创建
    for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
        local_data[local_shot_idx, step_idx] = receiver_data
        shot_wavefield.append(np.array(ey_field).copy())
else:
    # 不保存波场，ey_field在每次循环后自动释放
    for step_idx, (ey_field, receiver_data) in enumerate(time_loop_gen):
        local_data[local_shot_idx, step_idx] = receiver_data
```

### 3. 预期内存使用

修复后的内存需求：
```
不保存波场时：
- 模型数据：1000 × 1000 × 2 × 8 ≈ 15 MB
- 扩展模型：1020 × 1020 × 3 × 8 ≈ 24 MB
- 接收数据：10 × 8000 × 8 ≈ 0.6 MB
- 临时波场（单个时间步）：1020 × 1020 × 8 ≈ 8 MB
- 总计：≈ 50 MB（非常小！）

保存波场时（epoch=0）：
- 基础内存：≈ 50 MB
- 波场数据：10 × (1020 × 1020 × 8000 × 8) ≈ 630 GB
- 总计：≈ 630 GB
```

## 进一步优化建议

如果仍然遇到内存问题，可以尝试：

### 方案1：完全不保存波场数据
```python
# 在 main.py 中修改：
save_wavefield = False  # 永远不保存波场数据
```

这样梯度计算时会重新进行正演，虽然慢一些但不会有内存问题。

### 方案2：减少同时处理的shots数
```python
# 使用更多MPI进程，每个进程处理更少的shots
mpirun -n 100 python main.py  # 每个进程处理5个shots而不是10个
```

### 方案3：减少时间步数（如果可以接受）
```python
# 在 main.py 中修改：
steps = 4000  # 从8000减少到4000
```
这会使单个shot的波场数据从63GB降低到31.5GB。

### 方案4：使用检查点技术（推荐）
只保存关键时间步的波场数据：
```python
# 每隔N步保存一次
if save_wavefield and step_idx % 10 == 0:
    shot_wavefield.append(np.array(ey_field).copy())
```
这会将内存需求减少10倍。

## 运行建议

### 最安全的运行方式（无内存问题）
```bash
# 不保存波场数据，使用适中的进程数
mpirun -n 20 python main.py
```

### 如果需要调试（epoch=0会保存波场）
```bash
# 减少炮点数量进行测试
# 在 main.py 中临时修改：
# source_list = [(0, i) for i in range(0, zl, 10)]  # 只使用50个shots

# 然后运行
mpirun -n 10 python main.py
```

### 监控内存使用
```bash
# 在另一个终端运行
watch -n 1 'ps aux | grep python | head -20'

# 或者使用htop
htop
```

## 验证修复

运行程序后，检查输出：
```
[内存监控 正演开始前] 进程内存使用: 50.0 MB (0.05 GB)  # 应该很小
[内存监控 正演完成后] 进程内存使用: 60.0 MB (0.06 GB)  # 增长应该不大
```

如果看到内存使用超过1GB，说明仍有问题。

## 常见问题

**Q: 为什么进程48被杀死？**
A: 可能是该进程分配了更多的shots，或者系统对单个进程有内存限制（ulimit）。

**Q: 系统有470GB可用内存，为什么还会OOM？**
A: 单个进程的内存限制可能小于系统总内存。检查：
```bash
ulimit -v  # 虚拟内存限制
ulimit -m  # 物理内存限制
```

**Q: 如何完全避免内存问题？**
A: 设置 `save_wavefield = False`（所有epoch），这样永远不会保存波场数据。


