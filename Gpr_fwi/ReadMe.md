我将在这个工作文件夹中重新改写这套程序。

原先的程序有些冗余和复杂，使用起来并不方便。

我期望的一套程序流程非常trivial:

1.输入（需要用户自定义）：
    - 网格信息 xl,zl,dx,dz; k_max, dt; PML层 npml
    - 震源函数信息 Ricker子波主频率 f
    - 真实模型信息: epsilon, sigma
    - 总迭代epoch数 n_epoch
    - 初始模型信息: epsilon0, sigma0
2. Workflow
第一步，计算真实模型对应的观测数据d_obs
然后开始迭代，从初始模型开始逐步更新。直到收敛
不需要Optimization.py，Modified_TV.py,MultiScale.py,RandomSource.py，回归最基本的FWI流程。
并且在计算正演时，不需要考虑空气层，也不在模型顶部添加空气层，删去AirForward2D.py的有关操作，只需要进行Forward2D
不需要使用fh函数化输入的epsilon和sigma，把计算梯度的有关操作放入主函数中
需要调整整体的变量输入输出逻辑关系，我希望能够有一个输入是每一炮对应的检波器个数和坐标信息，然后time_loop需要返回对应的观测数据（这里可以简化为两种模式——模式1，自激自发，1个炮对应1个检波器；模式2，正常排列，每一炮对应相同数量的检波器，其位置信息也相同）前者相当于做了一个B-scan，后者则是每一炮都能得到一个数据剖面。
