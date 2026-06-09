import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from forward import forward_model
from gradient import compute_gradient
from scipy.ndimage import gaussian_filter, zoom
import os
import shutil
import time
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
except Exception:
    class _DummyComm:
        def Barrier(self):
            pass
        def bcast(self, x, root=0):
            return x
        def gather(self, x, root=0):
            return [x]
        def Get_rank(self):
            return 0
        def Get_size(self):
            return 1
    comm = _DummyComm()
    rank, size = 0, 1
    MPI_AVAILABLE = False

def main(optimizer=None, opt_params=None, epochs=5001):
    if rank == 0:
        if os.path.exists('results_marmousi'):
            shutil.rmtree('results_marmousi')
        os.makedirs('results_marmousi', exist_ok=True)

    # 网格参数
    xl, zl = 100, 200  # 深度x横向长度
    dx, dz = 0.02, 0.02
    dt = 4e-11
    npml = 10
    steps = 1000
    
    # 震源参数
    freq = 4e8
    
    #真实模型参数
    marmousi_model = np.fromfile('modelv1.bin', dtype=np.float32)
    orig_xl, orig_zl = 150, 460
    marmousi_model = marmousi_model.reshape(orig_zl, orig_xl).T
    zoom_factors = (xl / orig_xl, zl / orig_zl)
    marmousi_model = zoom(marmousi_model, zoom_factors, order=1)
    eps_max = 6
    marmousi_max = marmousi_model.max()
    epsilon_true = marmousi_model/marmousi_max*eps_max
    # 将epsilon_true从(150,460)扩展为(160,460)，在最浅层上方添加十行
    epsilon_top_row = epsilon_true[0, :].copy()  # 复制最浅层的一行
    epsilon_top_rows = np.tile(epsilon_top_row, (10, 1))
    epsilon_true = np.vstack([epsilon_top_rows, epsilon_true])  # 在顶部添加十行
    print(f"epsilon_true 的维度: {epsilon_true.shape}")
    sigma_true = np.zeros_like(epsilon_true)

    # 保存真实模型参数
    # if rank==0:
    #     np.save('results/epsilon_true.npy', epsilon_true)
    # np.save('results/sigma_true.npy', sigma_true)

    # 画图保存
    if rank == 0:
        plt.figure()
        plt.imshow(epsilon_true, aspect='auto', cmap='jet',vmin=1,vmax=6)
        plt.title('Epsilon True')
        plt.colorbar()
        plt.savefig('results_marmousi/epsilon_true.png')
        plt.close()


    # 初始模型参数
    epsilon0 = 1/gaussian_filter(1/epsilon_true, 5)
    sigma0 = np.zeros_like(epsilon0)

    # 画图保存
    if rank == 0:
        plt.figure()
        plt.imshow(epsilon0, aspect='auto', cmap='jet',vmin=1,vmax=6)
        plt.title('Epsilon Initial')
        plt.colorbar()
        plt.savefig('results_marmousi/epsilon_initial.png')
        plt.close()

    # 检查CFL条件
    if rank==0:
        c = 3e8  # 光速
        epsilon_max = epsilon_true.max()
        cfl_condition = dt < 1 / (c * np.sqrt(1/dx**2 + 1/dz**2) / np.sqrt(epsilon_max))
        
        print(f"CFL条件检查: {cfl_condition}")

    source_list = [(0, i) for i in range(0, zl, 5)]
    receiver_list = [(0, i) for i in range(0, zl, 2)]

    local_d_obs = forward_model(epsilon_true, sigma_true, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=False)
    d_obs_list = comm.gather(local_d_obs, root=0)
    if rank == 0:
        d_obs = np.concatenate(d_obs_list, axis=0) if len(d_obs_list) > 0 else np.zeros((0, len(receiver_list), steps), dtype=np.float32)
        plt.figure(); plt.imshow(d_obs.reshape(len(source_list)*len(receiver_list), steps).T, aspect='auto', cmap='seismic'); plt.colorbar(); plt.title('Observed data'); plt.savefig('results_marmousi/d_obs.png'); plt.close()
    else:
        d_obs = None
    d_obs = comm.bcast(d_obs, root=0)

    class SGD:
        def __init__(self, lr=0.01):
            self.lr = lr
        def step(self, P, G):
            for i in range(len(P)):
                P[i] = P[i] - self.lr * G[i]

    class Adam:
        def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
            self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.m=None; self.v=None; self.t=0
        def step(self, P, G):
            if self.m is None:
                self.m=[np.zeros_like(p) for p in P]; self.v=[np.zeros_like(p) for p in P]
            self.t += 1
            for i in range(len(P)):
                self.m[i] = self.b1*self.m[i] + (1-self.b1)*G[i]
                self.v[i] = self.b2*self.v[i] + (1-self.b2)*(G[i]*G[i])
                mhat = self.m[i]/(1-self.b1**self.t)
                vhat = self.v[i]/(1-self.b2**self.t)
                P[i] = P[i] - self.lr * mhat / (np.sqrt(vhat)+self.eps)

    class RMSprop:
        def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
            self.lr=lr; self.beta=beta; self.eps=eps; self.v=None
        def step(self, P, G):
            if self.v is None:
                self.v=[np.zeros_like(p) for p in P]
            for i in range(len(P)):
                self.v[i] = self.beta*self.v[i] + (1-self.beta)*(G[i]*G[i])
                P[i] = P[i] - self.lr * G[i]/(np.sqrt(self.v[i]) + self.eps)

    class Muon:
        def __init__(self, lr=0.01, mu=0.95, ns=5, eps=1e-8):
            self.lr=lr; self.mu=mu; self.ns=ns; self.eps=eps; self.V=None
        def _spec(self, G):
            if G.ndim < 2:
                return np.linalg.norm(G) + self.eps
            n = G.shape[1]
            v = np.random.randn(n)
            for _ in range(self.ns):
                v = G.T @ (G @ v)
                nv = np.linalg.norm(v) + 1e-12
                v = v / nv
            w = G @ v
            return np.linalg.norm(w) + self.eps
        def step(self, P, G):
            if self.V is None:
                self.V=[np.zeros_like(p) for p in P]
            for i in range(len(P)):
                g = G[i]
                s = self._spec(g)
                u = g / s
                v = self.mu*self.V[i] - self.lr*u
                self.V[i] = v; P[i] = P[i] + v

    def create_optimizer(name, params):
        if name=='SGD':
            return SGD(**(params or {}))
        if name=='Adam':
            return Adam(**(params or {}))
        if name=='RMSprop':
            return RMSprop(**(params or {}))
        if name=='Muon':
            return Muon(**(params or {}))
        raise ValueError('未知优化器')

    def save_model_fig(arr, path, vmin=1, vmax=6, title=''):
        plt.figure()
        plt.imshow(arr, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.colorbar()
        plt.savefig(path)
        plt.close()

    def render_animation(frames_dir, output_path):
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')], key=lambda x: int(x.split('epoch')[-1].split('.png')[0]))
        if not files:
            return
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        img = plt.imread(os.path.join(frames_dir, files[0]))
        im = ax.imshow(img)
        ax.axis('off')
        def update(i):
            im.set_data(plt.imread(os.path.join(frames_dir, files[i])))
            return [im]
        ani = animation.FuncAnimation(fig, update, frames=len(files), interval=50, blit=True)
        try:
            writer = FFMpegWriter(fps=10)
            ani.save(output_path, writer=writer)
        except Exception:
            try:
                writer = PillowWriter(fps=10)
                ani.save(output_path.replace('.mp4','.gif'), writer=writer)
            except Exception:
                print('动画生成失败')
        plt.close(fig)

    def run_once(name, params):
        exp_name = name
        if name == 'Muon' and isinstance(params, dict) and params.get('lr') == 0.01:
            exp_name = '0.01Muon'
        base = os.path.join('results_marmousi', exp_name)
        if rank == 0:
            os.makedirs(base, exist_ok=True)
            save_model_fig(epsilon_true, os.path.join(base, 'epsilon_true.png'), vmin=1, vmax=6, title='Epsilon True')
            save_model_fig(epsilon0, os.path.join(base, 'epsilon_initial.png'), vmin=1, vmax=6, title='Epsilon Initial')
            np.save(os.path.join(base, 'epsilon_true.npy'), epsilon_true)
            start_time = time.time()
            print(f'开始 {exp_name} 训练，epochs={epochs}')
        model_eps = epsilon0.copy()
        model_sig = sigma0.copy()
        loss_list = []
        opt = create_optimizer(name, params)
        frames_dir = os.path.join(base, 'frames')
        if rank == 0:
            os.makedirs(frames_dir, exist_ok=True)
        for epoch in range(epochs):
            local_d_syn, local_wavefield = forward_model(model_eps, model_sig, source_list, receiver_list, dt, dx, dz, npml, freq, steps, save_wavefield=True)
            all_local_d_syn = comm.gather(local_d_syn, root=0)
            if rank == 0:
                d_syn = np.zeros_like(d_obs)
                data_idx = 0
                for proc_d in all_local_d_syn:
                    if proc_d is not None and proc_d.shape[0] > 0:
                        n_local = proc_d.shape[0]
                        d_syn[data_idx:data_idx+n_local] = proc_d
                        data_idx += n_local
                residual = d_syn - d_obs
            else:
                residual = None
            residual = comm.bcast(residual, root=0)
            grad_eps, grad_sig = compute_gradient(model_eps, model_sig, residual, source_list, receiver_list, dt, dx, dz, npml, freq, steps, False, local_wavefield)
            eps_max = np.max(grad_eps)
            grad_eps = grad_eps / eps_max
            P=[model_eps]
            G=[grad_eps]
            opt.step(P, G)
            model_eps = P[0]
            model_eps = np.maximum(model_eps, 1.0)
            if rank == 0:
                loss_list.append(np.linalg.norm(residual))
                print(f'{name} 进度: {epoch+1}/{epochs}')
                if epoch % 2 == 0:
                    save_model_fig(model_eps, os.path.join(frames_dir, f'model_eps_epoch{epoch}.png'), vmin=1, vmax=6, title=f'{name} epoch {epoch}')
            comm.Barrier()
        if rank == 0:
            np.save(os.path.join(base, 'final_model_eps.npy'), model_eps)
            save_model_fig(model_eps, os.path.join(base, 'final_model_eps.png'), vmin=1, vmax=6, title=f'{exp_name} Final')
            plt.figure()
            plt.plot(loss_list)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{exp_name} Loss 曲线')
            plt.savefig(os.path.join(base, 'loss_curve.png'))
            plt.close()
            np.save(os.path.join(base, 'loss_curve.npy'), np.array(loss_list))
            render_animation(frames_dir, os.path.join(base, 'model_eps_video.mp4'))
            total_time = time.time() - start_time
            print(f'{exp_name} 2000个epoch总耗时: {total_time:.2f}s')

    if optimizer is None:
        defaults = {
            'SGD': {'lr': 0.01},
            'Adam': {'lr': 0.01, 'b1': 0.9, 'b2': 0.999, 'eps': 1e-8},
            'RMSprop': {'lr': 0.01, 'beta': 0.9, 'eps': 1e-8},
            'Muon': {'lr': 0.01, 'mu': 0.95, 'ns': 5, 'eps': 1e-8}
        }
        for k, v in defaults.items():
            if rank == 0:
                print(f'运行 {k}...')
            run_once(k, v)
    else:
        if rank == 0:
            print(f'运行 {optimizer}...')
        run_once(optimizer, opt_params or {})
    if rank == 0:
        print('FWI完成！结果保存在results_marmousi文件夹中。')

if __name__ == '__main__':
    main('Muon', {'lr': 0.01, 'mu': 0.95, 'ns': 5, 'eps': 1e-8})
