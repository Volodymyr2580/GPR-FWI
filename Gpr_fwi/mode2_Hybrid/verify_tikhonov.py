import numpy as np
import matplotlib.pyplot as plt
from gradient import compute_laplacian_2d, compute_tikhonov_gradient

def build_laplacian_matrix(nx, nz, dx, dz):
    """
    构建拉普拉斯算子的矩阵形式
    对于二维网格 (nx, nz)，构建 (nx*nz) x (nx*nz) 的拉普拉斯矩阵
    使用零梯度边界条件（Neumann边界条件）
    """
    N = nx * nz
    L = np.zeros((N, N))
    
    for i in range(nx):
        for j in range(nz):
            idx = i * nz + j  # 当前点的索引
            
            # 中心差分格式
            # x方向二阶导数: (f[i+1,j] - 2*f[i,j] + f[i-1,j]) / dx^2
            # z方向二阶导数: (f[i,j+1] - 2*f[i,j] + f[i,j-1]) / dz^2
            
            # 对角线元素 (中心点)
            L[idx, idx] = -2/dx**2 - 2/dz**2
            
            # x方向邻居
            if i > 0:  # 左邻居
                L[idx, (i-1)*nz + j] = 1/dx**2
            if i < nx-1:  # 右邻居
                L[idx, (i+1)*nz + j] = 1/dx**2
                
            # z方向邻居
            if j > 0:  # 上邻居
                L[idx, i*nz + (j-1)] = 1/dz**2
            if j < nz-1:  # 下邻居
                L[idx, i*nz + (j+1)] = 1/dz**2
    
    # 边界条件处理 - 零梯度条件
    # 对于边界点，我们修改矩阵以反映零梯度条件
    # 这相当于在边界外添加虚拟点，使得边界点的梯度为零
    
    # 左边界 (i=0): 虚拟点 f[-1,j] = f[1,j]
    for j in range(nz):
        idx = 0 * nz + j
        if j > 0 and j < nz-1:  # 非角点
            L[idx, idx] = -2/dx**2 - 2/dz**2
            L[idx, 1*nz + j] = 2/dx**2  # 右邻居贡献加倍
            L[idx, 0*nz + (j-1)] = 1/dz**2  # 上邻居
            L[idx, 0*nz + (j+1)] = 1/dz**2  # 下邻居
    
    # 右边界 (i=nx-1): 虚拟点 f[nx,j] = f[nx-2,j]
    for j in range(nz):
        idx = (nx-1) * nz + j
        if j > 0 and j < nz-1:  # 非角点
            L[idx, idx] = -2/dx**2 - 2/dz**2
            L[idx, (nx-2)*nz + j] = 2/dx**2  # 左邻居贡献加倍
            L[idx, (nx-1)*nz + (j-1)] = 1/dz**2  # 上邻居
            L[idx, (nx-1)*nz + (j+1)] = 1/dz**2  # 下邻居
    
    # 上边界 (j=0): 虚拟点 f[i,-1] = f[i,1]
    for i in range(nx):
        idx = i * nz + 0
        if i > 0 and i < nx-1:  # 非角点
            L[idx, idx] = -2/dx**2 - 2/dz**2
            L[idx, i*nz + 1] = 2/dz**2  # 下邻居贡献加倍
            L[idx, (i-1)*nz + 0] = 1/dx**2  # 左邻居
            L[idx, (i+1)*nz + 0] = 1/dx**2  # 右邻居
    
    # 下边界 (j=nz-1): 虚拟点 f[i,nz] = f[i,nz-2]
    for i in range(nx):
        idx = i * nz + (nz-1)
        if i > 0 and i < nx-1:  # 非角点
            L[idx, idx] = -2/dx**2 - 2/dz**2
            L[idx, i*nz + (nz-2)] = 2/dz**2  # 上邻居贡献加倍
            L[idx, (i-1)*nz + (nz-1)] = 1/dx**2  # 左邻居
            L[idx, (i+1)*nz + (nz-1)] = 1/dx**2  # 右邻居
    
    # 四个角点的特殊处理
    # 左上角 (0,0)
    L[0, 0] = -2/dx**2 - 2/dz**2
    L[0, 1*nz + 0] = 2/dx**2  # 右邻居贡献加倍
    L[0, 0*nz + 1] = 2/dz**2  # 下邻居贡献加倍
    
    # 右上角 (nx-1,0)
    L[(nx-1)*nz + 0, (nx-1)*nz + 0] = -2/dx**2 - 2/dz**2
    L[(nx-1)*nz + 0, (nx-2)*nz + 0] = 2/dx**2  # 左邻居贡献加倍
    L[(nx-1)*nz + 0, (nx-1)*nz + 1] = 2/dz**2  # 下邻居贡献加倍
    
    # 左下角 (0,nz-1)
    L[0*nz + (nz-1), 0*nz + (nz-1)] = -2/dx**2 - 2/dz**2
    L[0*nz + (nz-1), 1*nz + (nz-1)] = 2/dx**2  # 右邻居贡献加倍
    L[0*nz + (nz-1), 0*nz + (nz-2)] = 2/dz**2  # 上邻居贡献加倍
    
    # 右下角 (nx-1,nz-1)
    L[(nx-1)*nz + (nz-1), (nx-1)*nz + (nz-1)] = -2/dx**2 - 2/dz**2
    L[(nx-1)*nz + (nz-1), (nx-2)*nz + (nz-1)] = 2/dx**2  # 左邻居贡献加倍
    L[(nx-1)*nz + (nz-1), (nx-1)*nz + (nz-2)] = 2/dz**2  # 上邻居贡献加倍
    
    return L

def compute_tikhonov_matrix_method(model, dx, dz, alpha_tikhonov=1e-6):
    """
    使用矩阵方法计算Tikhonov梯度
    ∇R(m) = α * L^T * L * m
    """
    nx, nz = model.shape
    N = nx * nz
    
    # 构建拉普拉斯矩阵
    L = build_laplacian_matrix(nx, nz, dx, dz)
    
    # 将模型展平为向量
    m_vec = model.flatten()
    
    # 计算 L * m
    Lm = L @ m_vec
    
    # 计算 L^T * (L * m)
    tikhonov_grad_vec = alpha_tikhonov * (L.T @ Lm)
    
    # 重塑回二维
    tikhonov_grad = tikhonov_grad_vec.reshape(nx, nz)
    
    return tikhonov_grad

def compare_methods():
    """
    比较两种Tikhonov梯度计算方法
    """
    # 创建测试模型
    nx, nz = 20, 30  # 使用较小的尺寸以便可视化
    dx, dz = 0.02, 0.02
    
    # 创建高斯模型
    x = np.linspace(0, (nx-1)*dx, nx)
    z = np.linspace(0, (nz-1)*dz, nz)
    X, Z = np.meshgrid(x, z, indexing='ij')
    
    center_x, center_z = nx//2 * dx, nz//2 * dz
    sigma_gauss = 0.1
    test_model = np.exp(-((X - center_x)**2 + (Z - center_z)**2) / (2 * sigma_gauss**2))
    
    # 添加一些噪声
    np.random.seed(42)
    test_model += np.random.randn(nx, nz) * 0.01
    
    alpha_tikhonov = 1e-6
    
    # 方法1: 连续两次拉普拉斯算子
    print("计算连续拉普拉斯算子方法...")
    tikhonov_grad_continuous = compute_tikhonov_gradient(test_model, dx, dz, alpha_tikhonov)
    
    # 方法2: 矩阵方法
    print("计算矩阵方法...")
    tikhonov_grad_matrix = compute_tikhonov_matrix_method(test_model, dx, dz, alpha_tikhonov)
    
    # 比较结果
    diff = tikhonov_grad_continuous - tikhonov_grad_matrix
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    
    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")
    print(f"相对误差: {max_diff/np.max(np.abs(tikhonov_grad_continuous)):.2e}")
    
    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始模型
    im0 = axes[0, 0].imshow(test_model, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('原始模型')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 连续拉普拉斯方法
    im1 = axes[0, 1].imshow(tikhonov_grad_continuous, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('连续拉普拉斯方法')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 矩阵方法
    im2 = axes[0, 2].imshow(tikhonov_grad_matrix, cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('矩阵方法')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 差异
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title(f'差异 (max={max_diff:.2e})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 拉普拉斯算子
    laplacian = compute_laplacian_2d(test_model, dx, dz)
    im4 = axes[1, 1].imshow(laplacian, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('拉普拉斯算子 ∇²m')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 拉普拉斯算子的拉普拉斯算子
    laplacian_2 = compute_laplacian_2d(laplacian, dx, dz)
    im5 = axes[1, 2].imshow(laplacian_2, cmap='RdBu_r', aspect='auto')
    axes[1, 2].set_title('∇²(∇²m)')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('tikhonov_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 验证矩阵的对称性
    L = build_laplacian_matrix(nx, nz, dx, dz)
    is_symmetric = np.allclose(L, L.T, atol=1e-10)
    print(f"拉普拉斯矩阵是否对称: {is_symmetric}")
    
    # 验证 L^T * L 的对称性
    LTL = L.T @ L
    is_LTL_symmetric = np.allclose(LTL, LTL.T, atol=1e-10)
    print(f"L^T * L 是否对称: {is_LTL_symmetric}")
    
    return test_model, tikhonov_grad_continuous, tikhonov_grad_matrix, diff

def test_boundary_conditions():
    """
    测试边界条件处理
    """
    nx, nz = 10, 10
    dx, dz = 0.02, 0.02
    
    # 创建简单的测试模型
    test_model = np.ones((nx, nz))
    test_model[3:7, 3:7] = 2.0  # 中心区域为2
    
    print("测试边界条件...")
    print("原始模型:")
    print(test_model)
    
    # 计算拉普拉斯算子
    laplacian = compute_laplacian_2d(test_model, dx, dz)
    print("\n拉普拉斯算子:")
    print(laplacian)
    
    # 计算Tikhonov梯度
    tikhonov_grad = compute_tikhonov_gradient(test_model, dx, dz, 1e-6)
    print("\nTikhonov梯度:")
    print(tikhonov_grad)
    
    return test_model, laplacian, tikhonov_grad

if __name__ == "__main__":
    print("验证Tikhonov梯度计算的正确性...")
    
    # 比较两种方法
    test_model, grad_continuous, grad_matrix, diff = compare_methods()
    
    # 测试边界条件
    print("\n" + "="*50)
    test_boundary_conditions()
    
    print("\n验证完成！") 