# 二阶Tikhonov正则化在FWI中的应用

## 概述

本实现为全波形反演（FWI）添加了二阶Tikhonov正则化项，以提高反演的稳定性和获得更平滑的模型参数。

## 数学原理

### 目标函数

添加Tikhonov正则化后，FWI的目标函数变为：

```
J(m) = J_data(m) + J_reg(m)
```

其中：
- `J_data(m)` 是数据拟合项：`(1/2) * ||d_syn - d_obs||²`
- `J_reg(m)` 是Tikhonov正则化项：`(α/2) * ||∇²m||²`

### 梯度计算

总梯度为：
```
∇J(m) = ∇J_data(m) + ∇J_reg(m)
```

其中正则化梯度为：
```
∇J_reg(m) = α * ∇²(∇²m)
```

## 实现细节

### 1. 拉普拉斯算子计算

在 `gradient.py` 中实现了 `compute_laplacian_2d()` 函数：

```python
def compute_laplacian_2d(field, dx, dz):
    """
    计算二维拉普拉斯算子 ∇²f = ∂²f/∂x² + ∂²f/∂z²
    使用中心差分格式
    """
```

- 内部点使用中心差分格式
- 边界点使用一阶差分格式以避免边界效应

### 2. Tikhonov梯度计算

```python
def compute_tikhonov_gradient(model, dx, dz, alpha_tikhonov=1e-6):
    """
    计算二阶Tikhonov正则项的梯度
    正则项: R(m) = (α/2) * ||∇²m||²
    梯度: ∇R(m) = α * ∇²(∇²m)
    """
```

### 3. 修改后的梯度计算函数

`compute_gradient()` 函数现在接受额外的参数：
- `alpha_tikhonov_eps`: epsilon的Tikhonov正则化参数
- `alpha_tikhonov_sig`: sigma的Tikhonov正则化参数

## 使用方法

### 1. 设置正则化参数

在 `main.py` 中设置正则化参数：

```python
# Tikhonov正则化参数
alpha_tikhonov_eps = 1e-6  # epsilon的Tikhonov正则化参数
alpha_tikhonov_sig = 1e-6  # sigma的Tikhonov正则化参数
```

### 2. 调用梯度计算

```python
grad_eps, grad_sig = compute_gradient(
    model_eps, model_sig, residual, source_list, receiver_list, 
    dt, dx, dz, npml, freq, steps, sigma_required_gradient, 
    wavefield_data, alpha_tikhonov_eps, alpha_tikhonov_sig
)
```

## 参数调优建议

### 正则化参数选择

1. **初始值**: 建议从 `1e-6` 开始
2. **调整策略**: 
   - 如果模型过于平滑，减小参数
   - 如果反演不稳定，增大参数
3. **自适应调整**: 可以在迭代过程中逐渐减小正则化参数

### 参数范围

- `alpha_tikhonov_eps`: 通常在 `1e-8` 到 `1e-4` 之间
- `alpha_tikhonov_sig`: 通常在 `1e-8` 到 `1e-4` 之间

## 测试

运行测试脚本验证实现：

```bash
python test_tikhonov.py
```

这将生成：
- `test_tikhonov_gradient.png`: 显示拉普拉斯算子和Tikhonov梯度
- `test_regularization_effect.png`: 显示不同正则化参数的效果

## 优势

1. **稳定性**: 减少反演过程中的噪声放大
2. **平滑性**: 获得更平滑的模型参数
3. **收敛性**: 改善反演的收敛性能
4. **鲁棒性**: 对噪声和初始模型更鲁棒

## 注意事项

1. **过度平滑**: 过大的正则化参数可能导致模型过度平滑
2. **边界效应**: 边界处理可能影响边界附近的梯度计算
3. **计算成本**: 增加了额外的计算开销（四阶导数计算）

## 参考文献

1. Tikhonov, A. N., & Arsenin, V. Y. (1977). Solutions of ill-posed problems.
2. Tarantola, A. (2005). Inverse problem theory and methods for model parameter estimation. 