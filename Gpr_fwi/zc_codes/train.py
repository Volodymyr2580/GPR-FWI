# 基础库
import time
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PyTorch相关
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# 其他第三方库
from scipy.ndimage import gaussian_filter
import scipy.io
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
from deepwave import scalar

# 自定义模块
from data_loader import DataLoad_Train
from unet import UNet
from utils import SaveTrainResults, get_noise
from Plot2D import plot2d
# --------------------------
# 1.1. 辅助函数定义
# --------------------------
def grad_hook(grad):
    """梯度处理钩子函数"""
    grad[:, :34] = grad[:, :34] * 0
    max_abs_value = torch.max(torch.abs(grad)).item()  # 计算最大绝对值
    if max_abs_value != 0:  # 防止除以0
        grad /= max_abs_value  # 归一化
    return grad

def hook_illum(grad, grad_illum):
    """光照梯度钩子函数"""
    def hook(grad):
        if grad_illum.grad is not None:
            divisor_grad = torch.where(grad_illum.grad == 0,
                                     torch.ones_like(grad_illum.grad),
                                     grad_illum.grad)
            return grad / divisor_grad
        return grad
    return hook

def padded_laplacian(x):
    """带填充的拉普拉斯算子"""
    padded = replicate_pad(x, (1, 1, 1, 1))  # 镜像填充
    laplacian = padded[:-2, :-2] + padded[:-2, 2:] + \
                padded[2:, :-2] + padded[2:, 2:] - \
                4 * padded[1:-1, 1:-1]
    return laplacian

# --------------------------
# 1.2. 自定义类和损失函数
# --------------------------
class SSIM_Loss(SSIM):
    """自定义SSIM损失函数"""
    def forward(self, img1, img2):
        return super(SSIM_Loss, self).forward(img1, img2)

class MyDataset(data_utils.Dataset):
    """自定义数据集类"""
    def __init__(self, seismic_data, source_amplitudes, source_locations, receiver_locations):
        self.seismic_data = seismic_data
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations
        
    def __getitem__(self, item):
        return (self.seismic_data[item], 
                self.source_amplitudes[item], 
                self.source_locations[item], 
                self.receiver_locations[item])
    
    def __len__(self):
        return len(self.seismic_data)

# --------------------------
# 1.3. 工具函数
# --------------------------
def get_parameter_number(net):
    """计算网络参数量"""
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def same_seeds(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def replicate_pad(x, padding):
    # x: 输入张量 (height, width)
    # padding: (left, right, top, bottom)
    left, right, top, bottom = padding
    
    # 初始化一个全零张量，尺寸为填充后的尺寸
    padded = torch.zeros(x.size(0) + top + bottom, x.size(1) + left + right, device=x.device)
    
    # 将原始输入张量复制到填充后的张量的中心
    padded[top:top + x.size(0), left:left + x.size(1)] = x
    
    # 填充左边
    if left > 0:
        padded[top:top + x.size(0), :left] = x[:, 0].unsqueeze(1).expand(-1, left)
    
    # 填充右边
    if right > 0:
        padded[top:top + x.size(0), -right:] = x[:, -1].unsqueeze(1).expand(-1, right)
    
    # 填充顶部
    if top > 0:
        padded[:top, left:left + x.size(1)] = x[0, :].unsqueeze(0).expand(top, -1)
    
    # 填充底部
    if bottom > 0:
        padded[-bottom:, left:left + x.size(1)] = x[-1, :].unsqueeze(0).expand(bottom, -1)
    
    # 填充四个角
    if left > 0 and top > 0:
        padded[:top, :left] = x[0, 0].unsqueeze(0).unsqueeze(1).expand(top, left)
    if right > 0 and top > 0:
        padded[:top, -right:] = x[0, -1].unsqueeze(0).unsqueeze(1).expand(top, right)
    if left > 0 and bottom > 0:
        padded[-bottom:, :left] = x[-1, 0].unsqueeze(0).unsqueeze(1).expand(bottom, left)
    if right > 0 and bottom > 0:
        padded[-bottom:, -right:] = x[-1, -1].unsqueeze(0).unsqueeze(1).expand(bottom, right)
    
    return padded
    
# --------------------------
# 2.1. 基础张量导入以及参数设置
# --------------------------
same_seeds(1234)


# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device('cuda:0')
torch.set_default_dtype(torch.float)

BatchSize = 4
number_of_shots = 20
dx = 20
dt = 0.005
freq = 4
x_size = 460
z_size = 150
l_size = 30
minv = 1500
maxv = 5500
minr = 1000
maxr = 3000

seismic_data, seismic_data_noise, data_dim, model_dim, source_amplitudes, x_s, x_r, v_true ,rho_true= DataLoad_Train(number_of_shots)
data_path = '../data/'
# --------------------------
# 2.2. 真实模型（及初始模型设定）
# --------------------------
v_layer = torch.zeros(x_size,l_size)+1500
rho_layer = torch.zeros(x_size,l_size)+1000
zeros = torch.zeros_like(v_layer).to(device)
v_layer = v_layer.to(device)
rho_layer = rho_layer.to(device)

v_true = torch.from_file(data_path+'modelv1.bin',
                    size=x_size*z_size).reshape(x_size, z_size).to(device)
rho_true = torch.from_file(data_path+'modelrho1.bin',
                    size=x_size*z_size).reshape(x_size, z_size).to(device)
v_input = (torch.tensor(1/gaussian_filter(1/v_true.cpu().numpy(), 5)))
rho_input = (torch.tensor(1/gaussian_filter(1/rho_true.cpu().numpy(), 5)))

v_mig = torch.zeros_like(v_true)
v_mig = torch.cat((zeros,v_mig),dim=1)

v_true = torch.cat((v_layer,v_true),dim=1)
v_true1 = v_true.detach().requires_grad_(True)


rho_mig = torch.zeros_like(rho_true)
rho_mig = torch.cat((zeros,rho_mig),dim=1)

rho_true = torch.cat((rho_layer,rho_true),dim = 1)
rho_true1 = rho_true.detach().requires_grad_(True)

illum = torch.zeros_like(v_true1).requires_grad_(True)

seismic_data_size = seismic_data.size()
# --------------------------
# 2.3. 固定网络初始输入和参数相同
# --------------------------
# #噪声输入
mean = seismic_data_noise.mean()
std = seismic_data_noise.std()
gaussian_tensor1 = torch.normal(mean, 1, seismic_data_noise.size())

mean = rho_input.mean()
std = rho_input.std()
#gaussian_tensor2 = torch.normal(mean, std, rho_input.size())
#模型输入
net1_input = v_input
net2_input = rho_input

x = F.pad(net1_input, (5, 5, 2, 2)) 

x = x.unsqueeze(0).unsqueeze(0).repeat(1, number_of_shots, 1, 1)

## Step 3: 调整宽度到160（双线性插值）-> [1, 9, 464, 160]
# x = F.interpolate(
#    x, 
#    size=(464, 160),  
#    mode="bilinear",   
#    align_corners=False 
# )
net1_input = x 

x = F.pad(net2_input, (5, 5, 2, 2))

x = x.unsqueeze(0).unsqueeze(0).repeat(1, number_of_shots, 1, 1)

## Step 3: 调整宽度到160（双线性插值）-> [1, 9, 464, 160]
# x = F.interpolate(
#    x, 
#    size=(464, 160),    
#    mode="bilinear",    
#    align_corners=False 
# )
net2_input = x 

# 分批次训练的数据采样器

train_loader = data_utils.DataLoader(MyDataset(seismic_data_noise, source_amplitudes, x_s, x_r),
                                        batch_size=BatchSize,
                                        shuffle=True, drop_last=False)
net_1 = UNet()
net_2 = UNet()

path_same_init = 'same_init/'

if not os.path.exists(path_same_init):
    os.makedirs(path_same_init, exist_ok=True)

plt.figure()
plt.imshow(net1_input.mean(dim=[0,1], keepdim=False).detach().cpu().numpy().T, aspect='auto' , cmap = 'jet', vmax=maxv, vmin=minv)
plt.title("net1_input")
plt.colorbar()
plt.tight_layout()
plt.savefig(path_same_init + 'net1_input.png')
plt.close()

plt.figure()
plt.imshow(net2_input.mean(dim=[0,1], keepdim=False).detach().cpu().numpy().T, aspect='auto' , cmap = 'jet', vmax=maxr, vmin=minr)
plt.title("net2_input")
plt.colorbar()
plt.tight_layout()
plt.savefig(path_same_init + 'net2_input.png')
plt.close()

net1_input.detach().cpu().numpy().tofile(path_same_init +  'net1_input.bin')
net2_input.detach().cpu().numpy().tofile(path_same_init +  'net2_input.bin')
torch.save(net_1.state_dict(),path_same_init + 'net1_init.pkl')
torch.save(net_2.state_dict(),path_same_init + 'net2_init.pkl')



# --------------------------
# 3. 不同学习率 for循环
# --------------------------
for alpha in [1e-2]: 
    path = 'result_3_sigmoid_/latent16_vlr_1e-3_freesurface_4layertrue_nonorm_gausssmooth20input_rholr'+ str(alpha) + '_Fpad_2randn_another2/'
    pathi = []
    pathi.append(path + 'batch0/')
    pathi.append(path + 'batch1/')
    pathi.append(path + 'batch2/')
    pathi.append(path + 'batch3/')
    pathi.append(path + 'batch4/')

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    else:
        print('路径已存在！请检查参数以及说明')
        sys.exit(0)

    for i in range(5):
        if not os.path.exists(pathi[i]):
            os.makedirs(pathi[i], exist_ok=True)
        else:
            print('路径已存在！请检查参数以及说明')
            sys.exit(0)

    # --------------------------
    # 3.1. 网络参数载入
    # --------------------------
    # 初始化网络
    net_1 = UNet()
    net_2 = UNet()
    net_1.load_state_dict(torch.load(path_same_init + 'net1_init.pkl'))
    net_2.load_state_dict(torch.load(path_same_init + 'net2_init.pkl'))

    net_1.to(device)
    net_2.to(device)

    # --------------------------
    # 3.2. 优化过程参数设置
    # --------------------------
    optimizer = torch.optim.Adam([
    {'params': net_1.parameters(), 'lr': 1e-3},
    {'params': net_2.parameters(), 'lr': alpha}
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[800, 1600,2400], gamma=0.5
    )
    loss_fn = torch.nn.MSELoss()
    num_epochs = 3001
    # --------------------------
    # 3.3. 衡量指标存储
    # --------------------------
    loss_data_file = path + 'loss_data.txt'
    loss_modelv_file = path + 'loss_modelv.txt'
    loss_modelrho_file = path + 'loss_modelrho.txt'
    mean_outputs_file = path + 'mean_outputs.txt'


    # --------------------------
    # 3.4. 绘图坐标设置
    # --------------------------
    pixel_height, pixel_width = v_true.cpu().numpy().T.shape  
    # 一单位网格间距代表的实际距离（m）  
    grid_spacing_m = 20
    # 想要在刻度上显示的间隔（m）  
    tick_interval_m1 = 2000  
    tick_interval_m2 = 1000 
    # 计算每隔多少个像素设置一个刻度  
    tick_interval_pixels_x = tick_interval_m1 // grid_spacing_m  
    tick_interval_pixels_y = tick_interval_m2 // grid_spacing_m  
    # 设置x轴刻度  
    x_ticks = np.arange(0, pixel_width, tick_interval_pixels_x)  
    x_ticklabels = [f'{i * grid_spacing_m}' for i in x_ticks]  
    # 设置y轴刻度（注意y轴是图像的垂直方向，所以是高度）  
    y_ticks = np.arange(0, pixel_height, tick_interval_pixels_y)  
    y_ticklabels = [f'{i * grid_spacing_m}' for i in y_ticks]  


    v_max = 5500
    v_min = 1500
    rho_max = 2562
    rho_min = 1000

    for epoch in range(num_epochs) :
        net_1.train()
        net_2.train()
        epoch_loss = 0
        total_samples = 0
        for i, (s_d_i, s_a_i, x_s_i, x_r_i) in enumerate(train_loader):
            batch_size_now = s_d_i.size(0)  # 当前 batch 的样本数
            optimizer.zero_grad()
            if illum.grad is not None: illum.grad.zero_()
            out1, net1 = net_1(net1_input.to(device))
            out2, net2 = net_2(net2_input.to(device))
            [o1, o2, o3, o4] = out1
            [r1, r2, r3, r4] = out2
            net1.retain_grad()
            net2.retain_grad()

            net1_ = nn.functional.sigmoid(net1)
            net2_ = nn.functional.sigmoid(net2)


            net1_o = net1_ * (v_max - v_min) + v_min
            net2_o = net2_ * (rho_max - rho_min) + rho_min


            v = torch.cat((v_layer,net1_o),dim=1)
            rho = torch.cat((rho_layer,net2_o),dim=1)

            v.retain_grad()
            rho.retain_grad()

            outputsv = v + v_mig
            outputsv = torch.clamp(outputsv, min=minv, max=maxv)
            outputsv[:, l_size:(l_size + 4)] = v_true[:, l_size:(l_size + 4)]

            outputsrho = rho+rho_mig
            outputsrho = torch.clamp(outputsrho, min=minr, max=maxr)
            outputsrho[:, l_size:(l_size + 4)] = rho_true[:, l_size:(l_size + 4)]

            v0 = outputsv.clone()
            rho0 = outputsrho.clone()
            o10 = o1.clone()
            o20 = o2.clone()
            o30 = o3.clone()
            net10 = net1.clone()
            r10 = r1.clone()
            r20 = r2.clone()
            r30 = r3.clone()
            net20 = net2.clone()
            if (epoch % 200 == 0 or epoch in [1,2,3,5,10,50,100]):

                plt.figure()
                plt.imshow(o10.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("o1")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_o1_epoch' + str(i) + '.png')
                plt.close()
                o10.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_o1_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(o20.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("o2")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_o2_epoch' + str(i) + '.png')
                plt.close()
                o20.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_o2_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(o30.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("o3")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_o3_epoch' + str(i) + '.png')
                plt.close()
                o30.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_o3_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(net10.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("o4")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_o4_epoch' + str(i) + '.png')
                plt.close()
                net10.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_o4_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(r10.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("r1")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_r1_epoch' + str(i) + '.png')
                plt.close()
                r10.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_r1_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(r20.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("r2")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_r2_epoch' + str(i) + '.png')
                plt.close()
                r20.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_r2_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(r30.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("r3")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_r3_epoch' + str(i) + '.png')
                plt.close()
                r30.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_r3_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(net20.detach().cpu().numpy().T, aspect='auto')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("r4")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_r4_epoch' + str(i) + '.png')
                plt.close()
                net20.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_r4_epoch' + str(i) + '.bin')


                plt.figure()
                plt.imshow(v0.detach().cpu().numpy().T, aspect='auto' , cmap = 'jet', vmax=maxv, vmin=minv)
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("V")
                plt.colorbar(label='m/s')
                plt.xticks(x_ticks, x_ticklabels)  
                plt.yticks(y_ticks, y_ticklabels) 
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_outputsv_epoch' + str(i) + '.png')
                plt.close()
                v0.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_outputsv_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(rho0.detach().cpu().numpy().T, aspect='auto', cmap = 'jet', vmax=maxr, vmin=minr)
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("Rho")
                plt.colorbar(label='kg/m^3')
                plt.xticks(x_ticks, x_ticklabels)  
                plt.yticks(y_ticks, y_ticklabels) 
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + '_outputsrho_epoch' + str(i) + '.png')
                plt.close()
                rho0.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_outputsrho_epoch' + str(i) + '.bin')

            #illum.retain_grad()
            out = scalar(
                outputsv, outputsrho, dx, dt, illum, 
                source_amplitudes=s_a_i.to(device),
                source_locations=x_s_i.to(device),
                receiver_locations=x_r_i.to(device),
                max_vel=5500,
                pml_freq=freq,
                accuracy=8,
                pml_width=[40, 40, 40, 40]
            )[-1]

            rho_laplacian = padded_laplacian(net2_o)
            beta = 0
            loss_laplacian = beta * loss_fn(rho_laplacian, torch.zeros_like(rho_laplacian))
            loss = loss_fn(out,s_d_i.to(device))# + loss_laplacian

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            #hook_illum_v = v.register_hook(hook_illum(v, illum))
            #hook_illum_rho = rho.register_hook(hook_illum(rho, illum))
            hook1 = v.register_hook(grad_hook)
            hook3 = rho.register_hook(grad_hook)

            loss.backward()

            #hook_illum_v.remove()
            #hook_illum_rho.remove()
            hook1.remove()
            hook3.remove()
            

            if net1.grad is not None: 
                grad_o4 = net1.grad.clone()
                grad_v = v.grad.clone()
            if net2.grad is not None: 
                grad_r4 = net2.grad.clone()
                grad_rho = rho.grad.clone()
            #grad_illum = illum.grad.clone()
            
            optimizer.step()
            #scheduler.step()
            epoch_loss += loss.item() * batch_size_now
            total_samples += batch_size_now


            out1, net11 = net_1(net1_input.to(device))
            out2, net22 = net_2(net2_input.to(device))
            [o1, o2, o3, o4] = out1
            [r1, r2, r3, r4] = out2
            net1_ = nn.functional.sigmoid(net11)
            net2_ = nn.functional.sigmoid(net22)
            net1_o = net1_ * (v_max - v_min) + v_min
            net2_o = net2_ * (rho_max - rho_min) + rho_min
            v = torch.cat((v_layer,net1_o),dim=1)
            rho = torch.cat((rho_layer,net2_o),dim=1)

            outputsv2 = v + v_mig
            outputsv2 = torch.clamp(outputsv2, min=minv, max=maxv)
            outputsv2[:, l_size:(l_size + 4)] = v_true[:, l_size:(l_size + 4)]

            outputsrho2 = rho+rho_mig
            outputsrho2 = torch.clamp(outputsrho2, min=minr, max=maxr)
            outputsrho2[:, l_size:(l_size + 4)] = rho_true[:, l_size:(l_size + 4)]

            updatev = outputsv2 - outputsv
            updaterho = outputsrho2 - outputsrho

            if epoch % 200 == 0 and i == 0:
                torch.save(net_1.state_dict(),pathi[i] + str(epoch) + 'net1.pkl')
                torch.save(net_2.state_dict(),pathi[i] + str(epoch) + 'net2.pkl')
            if (epoch % 200 == 0 or epoch in [1,2,3,5,10,50,100]):

                #plt.figure()
                #plt.imshow(grad_illum[1:-1, l_size:-1].detach().cpu().numpy().T, aspect='auto', cmap='gray')
                #plt.xlabel("x/m")
                #plt.ylabel("z/m")
                #plt.title("illum")
                #plt.colorbar()
                #plt.tight_layout()
                #plt.savefig(pathi[i] + str(epoch) + 'illum_epoch' + str(i) + '.png')
                #plt.close()
                #grad_illum.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_illum_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(updatev.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("V")
                plt.colorbar(label='m/s')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + 'updatev_epoch' + str(i) + '.png')
                plt.close()
                updatev.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_updatev_epoch' + str(i) + '.bin')

                plt.figure()
                plt.imshow(updaterho.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                plt.xlabel("x/m")
                plt.ylabel("z/m")
                plt.title("rho")
                plt.colorbar(label='kg/m^3')
                plt.tight_layout()
                plt.savefig(pathi[i] + str(epoch) + 'updaterho_epoch' + str(i) + '.png')
                plt.close()
                updaterho.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_updaterho_epoch' + str(i) + '.bin')

                if net1.grad is not None: 
                    plt.figure()
                    plt.imshow(grad_v.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                    plt.xlabel("x/m")
                    plt.ylabel("z/m")
                    plt.title("V")
                    plt.colorbar(label='m/s')
                    plt.tight_layout()
                    plt.savefig(pathi[i] + str(epoch) + 'grad_v_epoch' + str(i) + '.png')
                    plt.close()
                    grad_v.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_grad_v_epoch' + str(i) + '.bin')

                    plt.figure()
                    plt.imshow(grad_o4.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                    plt.xlabel("x/m")
                    plt.ylabel("z/m")
                    plt.title("V")
                    plt.colorbar(label='m/s')
                    plt.tight_layout()
                    plt.savefig(pathi[i] + str(epoch) + 'grad_o4_epoch' + str(i) + '.png')
                    plt.close()
                    grad_o4.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_grad_o4_epoch' + str(i) + '.bin')

                if net2.grad is not None: 
                    plt.figure()
                    plt.imshow(grad_rho.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                    plt.xlabel("x/m")
                    plt.ylabel("z/m")
                    plt.title("Rho")
                    plt.colorbar(label='kg/m^3')
                    plt.tight_layout()
                    plt.savefig(pathi[i] + str(epoch) + 'grad_rho_epoch' + str(i) + '.png')
                    plt.close()
                    grad_rho.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_grad_rho_epoch' + str(i) + '.bin')

                    plt.figure()
                    plt.imshow(grad_r4.detach().cpu().numpy().T, aspect='auto', cmap='gray')
                    plt.xlabel("x/m")
                    plt.ylabel("z/m")
                    plt.title("Rho")
                    plt.colorbar(label='kg/m^3')
                    plt.tight_layout()
                    plt.savefig(pathi[i] + str(epoch) + 'grad_r4_epoch' + str(i) + '.png')
                    plt.close()
                    grad_r4.detach().cpu().numpy().T.tofile(pathi[i] + str(epoch) + '_grad_r4_epoch' + str(i) + '.bin')

            if i==0:
                loss_modelv = loss_fn(outputsv2, v_true) / loss_fn(v_true, torch.zeros_like(v_true))
                loss_modelrho = loss_fn(outputsrho2, rho_true) / loss_fn(rho_true, torch.zeros_like(rho_true))


                epoch_loss = epoch_loss / loss_fn(seismic_data.to(device), 
                    torch.zeros_like(seismic_data.to(device)))


                if epoch == 0:
                    loss_modelv_sum = loss_modelv.detach().cpu()
                    loss_modelrho_sum = loss_modelrho.detach().cpu()
                    loss_data_sum = epoch_loss.detach().cpu()
                else:
                    loss_modelv_sum = np.append(loss_modelv_sum, loss_modelv.detach().cpu())
                    loss_modelrho_sum = np.append(loss_modelrho_sum, loss_modelrho.detach().cpu())
                    loss_data_sum = np.append(loss_data_sum, epoch_loss.detach().cpu())

                with open(loss_data_file, 'a') as fr:
                    fr.write(str(epoch_loss.detach().cpu().numpy()) + '\n')
                with open(loss_modelv_file, 'a') as fr:
                    fr.write(str(loss_modelv.detach().cpu().numpy()) + '\n')
                with open(loss_modelrho_file, 'a') as fr:
                    fr.write(str(loss_modelrho.detach().cpu().numpy()) + '\n')

        








    data = {}
    data['loss_data'] = loss_data_sum
    scipy.io.savemat(path + 'DataLoss_add2.mat', data)

    loss_data_sum = loss_data_sum / loss_data_sum[0]
    loss_modelv_sum = loss_modelv_sum / loss_modelv_sum[0]
    loss_modelrho_sum = loss_modelrho_sum / loss_modelrho_sum[0]

    fig = plt.figure(figsize=(10.5, 3.5), dpi=300)
    plt.plot(np.arange(len(loss_data_sum)), loss_data_sum, label='data loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(bottom=0, top = 1)
    # plt.ylim(-0.05,1.05)
    plt.title('Data loss', fontsize=15)
    plt.savefig(path + 'data_loss.jpg', dpi=300, transparent=True, bbox_inches='tight')

    plt.figure()
    fig = plt.figure(figsize=(10.5, 3.5), dpi=300)
    plt.plot(np.arange(len(loss_modelv_sum)), loss_modelv_sum, label='model loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(bottom=0, top = 1)
    # plt.ylim(-0.05,1.05)
    plt.title('modelv loss', fontsize=15)
    plt.savefig(path + 'modelv_loss.jpg', dpi=300, transparent=True, bbox_inches='tight')

    plt.figure()
    fig = plt.figure(figsize=(10.5, 3.5), dpi=300)
    plt.plot(np.arange(len(loss_modelrho_sum)), loss_modelrho_sum, label='model loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(bottom=0, top = 1)
    # plt.ylim(-0.05,1.05)
    plt.title('modelrho loss', fontsize=15)
    plt.savefig(path + 'modelrho_loss.jpg', dpi=300, transparent=True, bbox_inches='tight')



