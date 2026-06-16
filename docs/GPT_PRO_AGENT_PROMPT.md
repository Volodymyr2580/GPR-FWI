# GPT Pro Agent Prompt

下面这份提示词适合交给 GPT Pro 代理模式。它假设代理可以查看 GitHub 仓库，但不能直接访问你的本地 PDF。若要让代理阅读 PDF 正文，请同时上传 `GPR_references/` 中的核心 PDF 或压缩包。

```text
你是一名熟悉电磁波理论、Ground Penetrating Radar (GPR)、Full Waveform Inversion (FWI)、Maxwell 方程组、FDTD、伴随状态法、数值优化和多参数反演的研究型代理。你的任务是帮助我从第一性原理出发，重新建构 GPR 数据 FWI 的理论认识，并写出第一部分汇报材料。

你可以查看我的 GitHub 仓库：

https://github.com/Volodymyr2580/GPR-FWI

请先阅读仓库中的项目说明和索引文件：

- README.md
- docs/PROJECT_INVENTORY.md
- docs/DATA_AND_GIT_POLICY.md
- docs/GPT_PRO_REFERENCE_WORKFLOW.md
- docs/GPR_REFERENCE_INDEX.md

重要说明：

- GitHub 仓库里不包含本地 PDF 正文。
- 本地 reference 目录是 `E:\sci_research\GPR\GPR_references`，但你作为云端代理不能直接访问这个路径。
- 我会在对话中上传核心 PDF 或压缩包。请优先读取我上传的 PDF。
- 如果某篇文献只在索引中出现、但没有上传 PDF，请不要编造其内容。你可以基于题名和公开可查资料做“待确认”的初步判断，但必须明确标注。

本轮总目标：

基于我已经整理的 reference 文献和你扩展检索到的文献，完成“第一部分：GPR 数据 FWI 的数学理论与方法综述”。

第一部分包括两大块：

A. 数学理论部分：
从 GPR 方程的数学出发，完整推导 GPR data FWI 的梯度公式。

B. 反演方法和技术细节部分：
系统整理已有文献中的 GPR-FWI 反演方法，包括怎么反演、目标函数、反演域、正则化、照明补偿、单参数/双参数反演、crosstalk 处理，以及实验模型设置。

请按以下阶段执行。

第一阶段：项目和本地文献索引理解

1. 阅读 GitHub 仓库中的项目说明，理解当前研究线：
   - `隐式FWI/`：implicit multiparameter FWI reproduction scaffold
   - `Gpr_fwi/`：早期 traditional、Tikhonov、hybrid、UNet、autoencoder 实验
   - `mpi_gprfwi/`：MPI/GPU 和大模型实验
   - `marmousi_paper/gpr-inversion/`：更干净的迁移目标
   - `Fast-GPR-FWI/`：Fast-GPR-FWI 复现与反馈材料，如果仓库中可见
2. 阅读 `docs/GPR_REFERENCE_INDEX.md`，建立本地文献清单。
3. 根据我上传的 PDF，建立文献条目表。每篇文献至少记录：
   - 文件名
   - 题名
   - 作者
   - 年份
   - 期刊/会议/来源
   - DOI 或 URL
   - 研究对象：crosshole GPR / surface GPR / borehole GPR / synthetic / field data
   - 反演域：time-domain / frequency-domain / Laplace-domain / mixed
   - 物理方程：Maxwell equations / scalar wave equation / vector wave equation / FDTD
   - 反演参数：relative permittivity, conductivity, wave velocity, attenuation, source wavelet 等
   - 单参数还是双参数/多参数
   - 目标函数
   - 梯度计算方法：adjoint-state / finite difference sensitivity / automatic differentiation / neural-network surrogate 等
   - 正则化或约束
   - 是否讨论 crosstalk
   - 是否讨论 illumination compensation
   - 实验模型和数据设置
   - 对我当前研究的价值

第二阶段：扩展文献调研

请基于本地文献索引和上传 PDF 继续扩展调研，重点搜索经典工作和 2021 年以来的新工作。

重点关键词：

- ground penetrating radar full waveform inversion
- GPR FWI time domain
- crosshole GPR full waveform inversion
- multiparameter GPR FWI permittivity conductivity
- dual parameter full waveform inversion GPR
- GPR FWI crosstalk permittivity conductivity
- GPR FWI illumination compensation
- adjoint-state method Maxwell equations GPR FWI
- FDTD GPR full waveform inversion
- source independent GPR FWI
- envelope inversion GPR
- total variation regularization GPR FWI
- Tikhonov regularization GPR FWI
- implicit neural representation GPR FWI
- neural network reparameterization full waveform inversion GPR

扩展调研要求：

1. 优先使用同行评议论文、出版社页面、arXiv、作者主页或可引用技术报告。
2. 不要只依赖搜索摘要。
3. 对每篇新文献记录完整引用信息和链接。
4. 如果新文献与本地索引重复，请合并条目。
5. 特别关注 time-domain GPR-FWI。
6. 频域、Laplace 域、机器学习/隐式表示方法也要整理，但要说明它们和时域方法的区别。

第三阶段：数学理论推导

请从第一性原理推导 GPR data FWI 的梯度公式。不要直接跳到结论。

1. 从 Maxwell 方程组开始：

\[
\nabla \times \mathbf{E} = - \mu \frac{\partial \mathbf{H}}{\partial t}
\]

\[
\nabla \times \mathbf{H} = \sigma \mathbf{E} + \epsilon \frac{\partial \mathbf{E}}{\partial t} + \mathbf{J}_s
\]

说明各物理量：

- \(\mathbf{E}\)：electric field
- \(\mathbf{H}\)：magnetic field
- \(\epsilon = \epsilon_0 \epsilon_r\)：permittivity
- \(\mu = \mu_0 \mu_r\)：permeability，GPR 中通常近似 \(\mu_r = 1\)
- \(\sigma\)：electric conductivity
- \(\mathbf{J}_s\)：source current density

解释为什么 GPR-FWI 常以 \(\epsilon_r\) 和 \(\sigma\) 为主要反演参数。

2. 推导或说明时域二阶电场方程：

\[
\nabla \times \mu^{-1} \nabla \times \mathbf{E}
+ \sigma \frac{\partial \mathbf{E}}{\partial t}
+ \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}
= - \frac{\partial \mathbf{J}_s}{\partial t}
\]

并说明：

- 什么情况下可以使用二维近似？
- 什么情况下可以使用 scalar approximation？
- FDTD 正演如何离散 Maxwell 方程？
- source、receiver、PML/absorbing boundary 如何进入正演问题？
- 观测数据如何表示：

\[
\mathbf{d}_{syn} = \mathbf{P}\mathbf{u}(\mathbf{m})
\]

其中 \(\mathbf{P}\) 是接收算子，\(\mathbf{u}\) 是波场，\(\mathbf{m}\) 是模型参数。

3. 定义 FWI 优化问题：

单参数：

\[
\mathbf{m} = \epsilon_r
\]

或：

\[
\mathbf{m} = v
\]

双参数：

\[
\mathbf{m} = (\epsilon_r, \sigma)
\]

基础 waveform L2 misfit：

\[
\Phi(\mathbf{m}) =
\frac{1}{2}
\sum_s
\left\|
\mathbf{P}\mathbf{u}_s(\mathbf{m}) - \mathbf{d}^{obs}_s
\right\|_2^2
\]

然后整理文献中的目标函数：

- waveform L2 misfit
- normalized waveform misfit
- envelope misfit
- phase/amplitude separated objective
- source-independent objective
- frequency-domain objective
- Laplace-domain objective
- robust loss
- optimal transport 或 Wasserstein-type misfit，如果 GPR 文献中有相关应用
- deep-learning surrogate 或 implicit representation 中的 objective

4. 用伴随状态法完整推导梯度。

设正演约束为：

\[
\mathcal{F}(\mathbf{u}, \mathbf{m}) = 0
\]

构造 Lagrangian：

\[
\mathcal{L}(\mathbf{u}, \mathbf{m}, \boldsymbol{\lambda})
=
\Phi(\mathbf{u}, \mathbf{m})
+
\int_0^T
\langle
\boldsymbol{\lambda},
\mathcal{F}(\mathbf{u}, \mathbf{m})
\rangle
dt
\]

需要推导：

- 对 \(\mathbf{u}\) 求变分得到 adjoint equation
- adjoint source 如何由 data residual 产生
- 为什么 adjoint wavefield 是反向时间传播
- 对 \(\epsilon\) 和 \(\sigma\) 求变分得到梯度
- 连续公式和离散实现之间的关系
- 梯度符号与 Lagrangian 约定有关，需说明符号约定

在二阶电场形式下，若正演算子写为：

\[
\mathcal{F}(\mathbf{E}; \epsilon, \sigma)
=
\nabla \times \mu^{-1} \nabla \times \mathbf{E}
+ \sigma \frac{\partial \mathbf{E}}{\partial t}
+ \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}
- \mathbf{f}
= 0
\]

请推导类似形式：

\[
\frac{\partial \Phi}{\partial \epsilon}
=
\int_0^T
\boldsymbol{\lambda}(t)
\cdot
\frac{\partial^2 \mathbf{E}(t)}{\partial t^2}
dt
\]

\[
\frac{\partial \Phi}{\partial \sigma}
=
\int_0^T
\boldsymbol{\lambda}(t)
\cdot
\frac{\partial \mathbf{E}(t)}{\partial t}
dt
\]

如果符号应为负号，请根据你的 Lagrangian 约定解释清楚。

写出 relative permittivity 的链式法则：

\[
\epsilon = \epsilon_0 \epsilon_r
\]

\[
\frac{\partial \Phi}{\partial \epsilon_r}
=
\epsilon_0
\frac{\partial \Phi}{\partial \epsilon}
\]

如果使用 velocity \(v = 1/\sqrt{\mu \epsilon}\)，也请推导 \(\partial \Phi / \partial v\) 和 \(\partial \Phi / \partial \epsilon\) 的转换关系。

5. 解释离散 FDTD 梯度实现：

- forward wavefield 需要保存哪些量？
- adjoint wavefield 如何注入残差？
- 梯度累积为什么是 forward wavefield 和 adjoint wavefield 的时间相关成像条件？
- Yee grid / staggered grid 对变量位置有什么影响？
- PML 区域的梯度通常如何处理？
- 多炮数据如何累加梯度？
- mini-batch / stochastic FWI 在 GPR 中是否常见？
- 梯度预处理、平滑、归一化、illumination compensation 如何做？

第四阶段：反演方法和技术细节综述

请围绕 GPR-FWI 文献整理以下问题。

1. 怎么反演？

整理：

- steepest descent
- conjugate gradient
- L-BFGS
- Gauss-Newton
- truncated Newton
- Adam / RMSprop
- stochastic optimization
- neural-network reparameterization
- implicit neural representation
- hybrid physics + learning 方法

对每种方法说明使用场景、优点、缺点、对初始模型的依赖、计算成本、对噪声和 cycle skipping 的敏感性。

2. 在什么域反演？

重点关注 time-domain GPR-FWI，同时比较：

- time-domain FWI
- frequency-domain FWI
- Laplace-domain FWI
- time-frequency / multiscale strategy

说明 GPR 数据中低频、带宽、天线波形、衰减对反演域选择的影响。

3. 目标函数。

形成表格：

| Paper | Domain | Objective | Parameters | Regularization | Main reason |
| --- | --- | --- | --- | --- | --- |

解释 L2 waveform misfit、normalized/correlation objective、source-independent FWI、envelope objective、robust objective 等。

4. 正则化和先验。

整理：

- Tikhonov regularization
- total variation
- model smoothing
- bound constraints
- structural regularization
- petrophysical constraints
- fractal constraints
- neural-network parameterization
- implicit neural representation
- multi-parameter coupling constraints

说明数学形式、物理意义和可能副作用。

5. Illumination compensation。

重点解释：

- GPR-FWI 中 illumination 不均匀来自哪里？
- 近源、近接收器、高衰减区域、深部区域为什么梯度尺度不一致？
- Hessian diagonal approximation / pseudo-Hessian 如何用于梯度归一化？
- source-receiver coverage 如何影响反演分辨率？
- illumination compensation 和 gradient preconditioning 的区别与联系。
- 文献中有哪些具体实现。

6. 单参数与双参数/多参数反演。

特别关注：

- 只反演 \(\epsilon_r\) 或 velocity 的方法
- 只反演 \(\sigma\) 或 attenuation 的方法
- 同时反演 \(\epsilon_r\) 和 \(\sigma\) 的方法
- 反演 \(\epsilon_r, \sigma, source wavelet\) 的方法

解释：

- \(\epsilon_r\) 主要控制传播速度/相位
- \(\sigma\) 主要控制振幅衰减和扩散
- 两者为什么会发生 trade-off 和 crosstalk
- 什么数据类型对 \(\sigma\) 更敏感？
- 什么频率范围或 offset 范围有利于区分两个参数？

7. 双参数反演中的 crosstalk 问题。

重点展开：

- crosstalk 的数学来源：Jacobian/sensitivity kernels 不正交，Hessian 非对角块强
- \(\epsilon_r\) 和 \(\sigma\) sensitivity kernel 的相似和不同
- crosstalk 在图像上表现为什么
- 错误 conductivity 如何污染 permittivity inversion
- 错误 permittivity 如何导致 conductivity artifact
- sequential inversion、alternating inversion、hierarchical inversion 是否能缓解
- parameter scaling、gradient balancing、Hessian preconditioning 是否能缓解
- regularization、structure coupling、bound constraints 是否能缓解
- deep-learning 或 implicit representation 是否真的解决 crosstalk，还是只是引入新的先验

8. 实验模型设置。

形成表格：

| Paper | Geometry | Model size | Grid spacing | Source wavelet | Frequency band | Receiver layout | Parameters inverted | Initial model | Noise | Main result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

重点关注：

- crosshole GPR 模型
- surface GPR 模型
- synthetic cylinder/anomaly 模型
- layered model
- alluvial aquifer / high-porosity layer
- Marmousi / Overthrust 类复杂模型迁移到 GPR-FWI 的合理性
- 单参数和双参数实验各自怎么设置
- 初始模型是平滑真值、常数模型、层状模型还是其他先验

第五阶段：最终输出

请生成以下 Markdown 文件内容：

1. `GPR_FWI_literature_matrix.md`
   - 本地上传文献和扩展文献的文献矩阵。

2. `GPR_FWI_core_paper_notes.md`
   - 10 到 20 篇核心文献精读笔记。
   - 每篇包含 Citation、Problem、Governing equation、Objective function、Inversion parameters、Gradient / sensitivity method、Regularization / preconditioning、Experiment setup、Key contribution、Limitations、How it helps my project。

3. `Part1_GPR_FWI_theory_and_methods.md`
   - 第一部分汇报初稿。
   - 建议结构：
     # 第一部分：GPR 数据 FWI 的数学理论与方法综述
     ## 1. 研究问题：为什么 GPR-FWI 是一个非线性 PDE 约束优化问题
     ## 2. 从 Maxwell 方程到 GPR 正演模型
     ## 3. GPR 数据、观测算子和 FWI 目标函数
     ## 4. 伴随状态法推导 GPR-FWI 梯度
     ## 5. 单参数反演：permittivity / velocity
     ## 6. 双参数反演：permittivity-conductivity
     ## 7. Crosstalk 的数学来源与处理方法
     ## 8. Time-domain GPR-FWI 的技术路线
     ## 9. 目标函数、正则化和照明补偿
     ## 10. 典型实验模型设置
     ## 11. 对我后续实验设计的启发
     ## 12. 待确认问题和下一步文献补充

4. `GPR_FWI_formula_summary.md`
   - Maxwell 方程、二阶电场方程、目标函数、Lagrangian、adjoint equation、\(\epsilon\) 梯度、\(\epsilon_r\) 梯度、\(\sigma\) 梯度、多炮梯度累加、正则化项梯度、参数转换关系。

5. `Part1_to_experiment_recommendations.md`
   - 后续实验路线：
     - 最小可验证实验
     - crosstalk 诊断实验
     - illumination compensation 实验
     - 目标函数对比实验
     - 正则化实验

第六阶段：质量要求

1. 理论推导必须自洽。
2. 每个核心公式都要说明符号含义。
3. 每个文献结论都要给出引用，不要无来源断言。
4. 区分 GPR FWI 和 seismic FWI：
   - seismic 通常是 acoustic/elastic wave equation
   - GPR 是 electromagnetic Maxwell equations
   - GPR 中 conductivity 带来衰减和振幅敏感性
   - GPR 天线、source wavelet、near-field effect 更重要
5. 重点关注 time-domain GPR-FWI。
6. 频域和 Laplace 域做比较，但不要喧宾夺主。
7. 对双参数反演和 crosstalk 要写得足够深入。
8. 对每个技术名词都给出直观解释和数学解释。
9. 明确区分“已由上传 PDF 证实”“由公开资料确认”“待确认”。
10. 最后给出执行摘要：
    - 已读哪些本地上传文献
    - 扩展了哪些文献
    - 当前理论链条是否完整
    - 哪些地方仍需人工确认
    - 下一步最值得做的实验是什么

请现在开始。先读取 GitHub 仓库说明和文献索引，再读取我上传的 PDF，然后扩展文献调研，最后生成上述 Markdown 内容。
```
