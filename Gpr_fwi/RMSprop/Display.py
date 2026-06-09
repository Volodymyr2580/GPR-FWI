import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

eps_dir = './400000000.0Hz_imodel_eps_file_100_0'
sig_dir = './400000000.0Hz_imodel_sig_file_100_0'

# 创建result文件夹（如果不存在）
if not os.path.exists('result'):
    os.makedirs('result')

for idx in range(200):
    eps_file = os.path.join(eps_dir, '{}_imodel_eps.npy'.format(idx))
    sig_file = os.path.join(sig_dir,'{}_imodel_sig.npy'.format(idx))

    eps = np.load(eps_file)
    sig = np.load(sig_file)
    eps = eps.reshape(120,220)
    sig = sig.reshape(120,220)
    # 创建新的图形
    plt.figure(figsize=(12, 5))
    
    # 绘制eps
    plt.subplot(1, 2, 1)
    plt.imshow(eps, cmap='gray_r', aspect='auto')  # 使用反转的灰度色标
    plt.title('介电常数 (eps)')
    plt.colorbar(label='介电常数')
    plt.xlabel('x')
    plt.ylabel('z')
    
    # 绘制sig
    plt.subplot(1, 2, 2)
    plt.imshow(sig, cmap='gray_r', aspect='auto')  # 使用反转的灰度色标
    plt.title('电导率 (sig)')
    plt.colorbar(label='电导率')
    plt.xlabel('x')
    plt.ylabel('z')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join('result', f'model_{idx}.png'))
    plt.close()

#draw initial model
initial_eps = np.load('./initial_model_eps.npy')
initial_sig = np.load('./initial_model_sig.npy')

# 创建新的图形
plt.figure(figsize=(12, 5))

# 绘制初始介电常数模型
plt.subplot(1, 2, 1)
plt.imshow(initial_eps, cmap='jet')
plt.colorbar(label='介电常数')
plt.title('初始介电常数模型')
plt.xlabel('x')
plt.ylabel('z')

# 绘制初始电导率模型 
plt.subplot(1, 2, 2)
plt.imshow(initial_sig, cmap='jet')
plt.colorbar(label='电导率 (S/m)')
plt.title('初始电导率模型')
plt.xlabel('x')
plt.ylabel('z')

# 调整子图间距
plt.tight_layout()

# 保存图像
plt.savefig('initial_model.png', dpi=300)
plt.close()

#draw true model
epsilon_=np.load('LayerModel.npy')
epsilon=np.zeros((120,220)) #说明参数部分是不包含CPML层的 CPML上下左右各10个单元
epsilon[10:-10,10:-10]=epsilon_
CPML=10
# 扩展CPML边界
epsilon[:CPML,:]=epsilon[CPML,:] # 将第10行的值复制到0-9行
epsilon[-CPML:,:]=epsilon[-CPML-1,:] # 将倒数第11行的值复制到最后10行
epsilon[:,:CPML]=epsilon[:,CPML].reshape((len(epsilon[:,CPML]),-1)) # 将第10列的值复制到0-9列
epsilon[:,-CPML:]=epsilon[:,-CPML-1].reshape((len(epsilon[:,-CPML-1]),-1)) # 将倒数第11列的值复制到最后10列
    
    #设置相对介电常数参数
epsilon_=epsilon.copy()
epsilon[:10+3,:]=1 #这里空气层是加在x方向，
    
    #初始化电导率
sigma_=np.load('true_sigma.npy')
sigma=np.ones((120,220))*1e-3
sigma[10:-10,10:-10]=sigma_
    # 扩展CPML边界
sigma[:CPML,:]=sigma[CPML,:] # 将第10行的值复制到0-9行
sigma[-CPML:,:]=sigma[-CPML-1,:] # 将倒数第11行的值复制到最后10行
sigma[:,:CPML]=sigma[:,CPML].reshape((len(sigma[:,CPML]),-1)) # 将第10列的值复制到0-9列
sigma[:,-CPML:]=sigma[:,-CPML-1].reshape((len(sigma[:,-CPML-1]),-1)) # 将倒数第11列的值复制到最后10列
sigma[:10+3,:]=0
# 创建新的图形
plt.figure(figsize=(12, 5))

# 绘制初始介电常数模型
plt.subplot(1, 2, 1)
plt.imshow(epsilon, cmap='gray_r')
plt.colorbar(label='介电常数')
plt.title('真实介电常数模型')
plt.xlabel('x')
plt.ylabel('z')

# 绘制初始电导率模型 
plt.subplot(1, 2, 2)
plt.imshow(sigma, cmap='gray_r')
plt.colorbar(label='电导率 (S/m)')
plt.title('真实电导率模型')
plt.xlabel('x')
plt.ylabel('z')

# 调整子图间距
plt.tight_layout()

# 保存图像
plt.savefig('true_model.png', dpi=300)
plt.close()

#draw loss
loss = np.load('./loss.npy')
plt.figure(figsize=(12,10))
# 移除固定纵横比,让图形自然展开
plt.plot(np.arange(len(loss))+1,loss,'b.-.',label='Ture Model')
ax=plt.gca()
ax.set_ylabel('Value of Objective Function', fontsize=12)
ax.set_xlabel('Iteration', fontsize=12) 
ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
plt.grid(linestyle='--')
plt.savefig('FWILoss.png',dpi=1000)



    

    

