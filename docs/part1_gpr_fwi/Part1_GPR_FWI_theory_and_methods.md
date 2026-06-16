# 第一部分：GPR 数据 FWI 的数学理论与方法综述

版本说明：本文是第一部分汇报的可用草稿，已经覆盖从 Maxwell 方程、GPR 正演、FWI 目标函数、伴随梯度，到单/双参数反演、crosstalk、目标函数、正则化、照明补偿和后续实验设计的主线。当前版本的目标是建立清晰理论框架；逐篇文献页码、公式编号和完整扩展综述仍需后续校对。

## 0. 执行摘要

GPR-FWI 可以被理解为一个由 Maxwell 方程约束的非线性优化问题。地下介质参数 \(\mathbf{m}\)，例如相对介电常数 \(\epsilon_r\) 和电导率 \(\sigma\)，并不直接生成数据，而是先通过电磁波正演生成波场 \(\mathbf{u}(\mathbf{m})\)，再由接收算子 \(\mathbf{P}\) 采样为合成雷达数据。因此，GPR-FWI 的核心链条是：

\[
\mathbf{m}
\rightarrow
\mathbf{u}(\mathbf{m})
\rightarrow
\mathbf{P}\mathbf{u}(\mathbf{m})
\rightarrow
\Phi(\mathbf{m}).
\]

第一性原理上，GPR-FWI 和地震 FWI 的共同点是“用完整波形约束地下参数”；差异在于 GPR 的物理基础是电磁波 Maxwell 方程，而不是声波/弹性波方程。因此，GPR 中 \(\epsilon_r\)、\(\sigma\)、source wavelet、天线耦合、近场效应和介质衰减都比普通声波 FWI 更突出。

从梯度推导看，伴随状态法提供了避免显式构造完整 Jacobian 的办法。直观地说，正演波场告诉我们“哪里被震源照亮”，伴随波场告诉我们“哪里能解释接收端残差”，二者在时间上的相关累加构成模型梯度。对二阶电场方程而言，\(\epsilon\) 梯度与 adjoint field 和 \(\partial_{tt}\mathbf{E}\) 相关，\(\sigma\) 梯度与 adjoint field 和 \(\partial_t\mathbf{E}\) 相关；在 Meles et al. (2012) 的一阶/FDTD sensitivity 表达中，\(\epsilon\) sensitivity 与 forward electric field 的时间导数相关，而 \(\sigma\) sensitivity 与 forward electric field 本身相关。两种写法的导数阶数差异来自方程形式和变量选择，后续需要在离散实现中进一步核对。

从反演方法看，单参数 \(\epsilon_r\) 反演适合作为最小可验证实验，因为它主要控制速度和相位；\(\sigma\) 主要影响振幅和衰减，但也容易与 source wavelet、几何扩散和噪声混淆。双参数 \((\epsilon_r,\sigma)\) 反演的关键困难是 crosstalk：两个参数的 sensitivity kernels 不正交，Hessian 的非对角块会让一个参数的误差映射到另一个参数中。Lavoue et al. (2014) 的频域双参数研究说明，parameter scaling 和 conductivity regularization 对稳定双参数反演至关重要。

从目标函数看，L2 waveform misfit 是最直接的选择，但容易 cycle skipping，也容易受 source wavelet 和振幅误差影响。Source-independent、envelope、Laplace-domain、frequency-domain multiscale、optimal-transport-to-least-squares switching 等方法，本质上都是在改变“残差如何被定义”和“残差如何变成 adjoint source”。这些方法并不替代物理正演，而是在不同反演阶段改善目标函数的几何形状。

后续实验不应直接跳到最复杂的 IFWI 或 Marmousi/Overthrust 模型，而应按照“最小闭环 -> 梯度验证 -> 单参数 -> 双参数 -> crosstalk 诊断 -> illumination compensation -> 目标函数/正则化对比 -> 网络重参数化”的顺序逐步打开复杂度。

## 0.1 证据边界

本文当前结论来自三类来源：

- `PDF-formula-checked`：已经从本地 PDF 中定位到公式或方法段落，例如 Meles et al. (2012)、Lavoue et al. (2014)、Liu et al. (2022)、Meng et al. (2019)。
- `PDF-skimmed`：已经核实标题、摘要、关键方法页或结论页，但尚未完整精读全文，例如 Busch et al. (2012)、Sun et al. (2024)、Ernst et al. (2007)、Hunziker et al. (2025)。
- `local-doc-checked`：来自本地 README/progress/migration map，用于连接已有实验线，但不能当作外部文献结论。

更细的证据状态见 `source_status.md`。

## 0.2 术语说明

- GPR：Ground Penetrating Radar，探地雷达。
- FWI：Full Waveform Inversion，全波形反演。
- Time-domain：时域，直接在时间序列上正演和匹配波形。
- Frequency-domain：频域，选择若干频率分量进行建模和反演。
- Laplace-domain：Laplace 域，常用于提取平滑/长波长信息，为时域 FWI 提供初始模型。
- Adjoint-state method：伴随状态法，用一次正演和一次伴随传播高效计算梯度。
- Crosstalk：多参数反演中一个参数的误差被另一个参数吸收或误解释。
- Illumination compensation：照明补偿，用来修正不同区域因 source-receiver 覆盖不同而导致的梯度尺度不均。

## 1. 研究问题：为什么 GPR-FWI 是一个非线性 PDE 约束优化问题

GPR 全波形反演（GPR-FWI）的目标是：在已知发射源、接收器记录和电磁波正演机制的条件下，反推出地下介质参数。本文重点关注两个最常见的参数：相对介电常数 \(\epsilon_r\) 和电导率 \(\sigma\)。前者主要控制电磁波速度和相位，后者主要控制介质损耗、振幅衰减和波形拖尾。

从第一性原理看，GPR-FWI 不是普通的曲线拟合问题，而是一个由偏微分方程约束的优化问题（PDE-constrained optimization）。参数 \(\mathbf{m}\) 不能直接代入一个显式函数得到数据；它必须先进入 Maxwell 方程，产生电磁波场，再由接收器采样成雷达记录：

\[
\mathbf{m}
\rightarrow
\mathbf{u}(\mathbf{m})
\rightarrow
\mathbf{P}\mathbf{u}(\mathbf{m})
\rightarrow
\Phi(\mathbf{m}).
\]

这条链路中的每一环都有明确含义：

- \(\mathbf{m}\)：地下模型参数，可以是单参数 \(\epsilon_r\)，也可以是双参数 \((\epsilon_r,\sigma)\)。
- \(\mathbf{u}\)：由 Maxwell 方程生成的电磁波场，包含电场和/或磁场分量。
- \(\mathbf{P}\)：观测算子，用来从全空间波场中取出接收器位置和分量上的时间序列。
- \(\Phi\)：目标函数，用来度量合成数据和观测数据之间的差异。

这个问题的非线性主要来自两个层面：

1. 波场 \(\mathbf{u}(\mathbf{m})\) 对介质参数的响应是非线性的。即使 Maxwell 方程对波场本身是线性的，改变介质参数后，传播路径、相位、反射和衰减都会整体改变。
2. 多参数反演时，\(\epsilon_r\) 和 \(\sigma\) 对数据的影响并不正交。一个参数造成的误差可能被另一个参数部分吸收，这就是双参数 GPR-FWI 中常见的 crosstalk。

与地震 FWI 相比，GPR-FWI 共享“用完整波形约束地下模型”的反演思想，但物理方程不同。地震 FWI 通常以声波或弹性波方程为核心；GPR-FWI 则以电磁波 Maxwell 方程为核心。因此，GPR 中的电导率、source wavelet、天线耦合、近场效应、频散和衰减问题更加突出，也更容易影响反演稳定性。

## 2. 从 Maxwell 方程到 GPR 正演模型

在各向同性介质中，GPR 时域正演可以从 Maxwell 方程组出发：

\[
\nabla \times \mathbf{E}
=
-\mu \frac{\partial \mathbf{H}}{\partial t},
\]

\[
\nabla \times \mathbf{H}
=
\sigma \mathbf{E}
+
\epsilon \frac{\partial \mathbf{E}}{\partial t}
+
\mathbf{J}_s .
\]

其中 \(\mathbf{E}\) 是电场，\(\mathbf{H}\) 是磁场，\(\mathbf{J}_s\) 是外加源项，\(\epsilon=\epsilon_0\epsilon_r\)，\(\mu=\mu_0\mu_r\)。在多数近地表 GPR 场景中，介质可以近似为非磁性介质，即 \(\mu_r \approx 1\)。这使得主要未知量集中在 \(\epsilon_r\) 和 \(\sigma\) 上。

消去磁场 \(\mathbf{H}\) 后，可以得到一个常用的二阶电场方程：

\[
\nabla \times \mu^{-1} \nabla \times \mathbf{E}
+
\sigma \frac{\partial \mathbf{E}}{\partial t}
+
\epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}
=
-\frac{\partial \mathbf{J}_s}{\partial t}.
\]

这个形式非常适合理解两个参数的物理分工：

- \(\epsilon\) 乘在二阶时间导数项上，直接影响波速 \(v \approx 1/\sqrt{\mu\epsilon}\)，因此强烈控制走时、相位和反射界面位置。
- \(\sigma\) 乘在一阶时间导数项上，对能量耗散和振幅衰减更加敏感，因此更容易与 source wavelet、天线响应和几何扩散混淆。

在数值实现中，GPR 正演常用 FDTD（finite-difference time-domain）。FDTD 可以理解为把空间和时间切成网格，在每个时间步交替更新电场和磁场。经典 Yee grid 会把不同电磁场分量放在交错位置，从而更自然地离散 curl 算子。Meles et al. (2012)、Ernst et al. (2007) 等 GPR-FWI 工作都直接或间接依赖这种时域 Maxwell 正演框架。

## 3. GPR 数据、观测算子和 FWI 目标函数

对第 \(s\) 个发射源，正演得到的波场记为 \(\mathbf{u}_s(\mathbf{m})\)。接收器并不会观测整个计算区域中的波场，而只记录若干位置、若干分量、若干时间采样点上的信号。因此，合成数据可写为：

\[
\mathbf{d}^{syn}_s
=
\mathbf{P}\mathbf{u}_s(\mathbf{m}).
\]

最基础的全波形最小二乘目标函数为：

\[
\Phi(\mathbf{m})
=
\frac{1}{2}
\sum_s
\left\|
\mathbf{P}\mathbf{u}_s(\mathbf{m})
-
\mathbf{d}^{obs}_s
\right\|_2^2.
\]

其中残差定义为：

\[
\mathbf{r}_s
=
\mathbf{P}\mathbf{u}_s(\mathbf{m})
-
\mathbf{d}^{obs}_s.
\]

这个目标函数直观、容易实现，也是许多 GPR-FWI 实验的起点。不过，它的几何形状并不总是友好：

- 如果初始模型较差，合成波形和观测波形相差超过半个周期，L2 失配可能把错误的波峰和波谷配对，形成 cycle skipping。
- 振幅误差不一定只来自 \(\sigma\)，也可能来自 source wavelet、天线耦合、几何扩散、边界吸收或噪声。
- 在双参数反演中，\(\epsilon_r\) 和 \(\sigma\) 的灵敏度核可能相互重叠，一个参数的误差会被另一个参数“解释掉”，形成 crosstalk。

因此，文献中出现了 normalized objective、source-independent objective、envelope objective、Laplace-domain objective、frequency-domain multiscale objective、optimal-transport distance 等变体。它们的共同目标不是改变 Maxwell 正演本身，而是改变残差的定义方式，让优化问题在早期迭代中更不容易被错误相位、错误震源或错误振幅牵着走。

## 4. 伴随状态法与 GPR-FWI 梯度

如果直接对每个模型网格点逐一扰动并重新正演，计算梯度的成本会随参数维度线性增长，几乎不可接受。伴随状态法的价值就在这里：它把“很多次参数扰动正演”转化为“每个 shot 一次正演加一次伴随传播”。

设正演约束为：

\[
\mathcal{F}(\mathbf{u},\mathbf{m})=0.
\]

构造 Lagrangian：

\[
\mathcal{L}(\mathbf{u},\mathbf{m},\boldsymbol{\lambda})
=
\Phi(\mathbf{u},\mathbf{m})
+
\int_0^T
\langle
\boldsymbol{\lambda},
\mathcal{F}(\mathbf{u},\mathbf{m})
\rangle
dt .
\]

对波场变量 \(\mathbf{u}\) 求变分并令其为零，可以得到伴随方程（adjoint equation）。数据残差在接收器位置作为伴随源注入，并沿时间反向传播。这样，正演波场表示“源如何照亮地下”，伴随波场表示“数据残差如何回传到地下”，两者的时空相关给出模型参数的更新方向。

若采用二阶电场算子：

\[
\mathcal{F}(\mathbf{E};\epsilon,\sigma)
=
\nabla \times \mu^{-1}\nabla\times\mathbf{E}
+
\sigma\partial_t\mathbf{E}
+
\epsilon\partial_{tt}\mathbf{E}
-
\mathbf{f},
\]

则对参数的扰动满足：

\[
\delta_\epsilon \mathcal{F}
=
\delta\epsilon\,\partial_{tt}\mathbf{E},
\]

\[
\delta_\sigma \mathcal{F}
=
\delta\sigma\,\partial_t\mathbf{E}.
\]

因此，忽略符号约定和边界项后，连续形式的梯度可写成如下相关型表达：

\[
\frac{\partial \Phi}{\partial \epsilon}
\propto
\int_0^T
\boldsymbol{\lambda}(t)\cdot
\partial_{tt}\mathbf{E}(t)
\,dt,
\]

\[
\frac{\partial \Phi}{\partial \sigma}
\propto
\int_0^T
\boldsymbol{\lambda}(t)\cdot
\partial_t\mathbf{E}(t)
\,dt.
\]

这里的正负号取决于 Lagrangian 的符号约定、残差定义以及伴随方程的写法。在实际代码中，不能只凭连续公式判断正负号，必须通过有限差分梯度检查验证。

如果模型参数使用相对介电常数 \(\epsilon_r\)，还需要链式法则：

\[
\frac{\partial \Phi}{\partial \epsilon_r}
=
\epsilon_0
\frac{\partial \Phi}{\partial \epsilon}.
\]

Meles et al. (2012) 给出了一个更贴近 FDTD sensitivity 实现的视角：\(\epsilon\) sensitivity 由 adjoint receiver wavefield 与 forward electric field 的时间导数相关得到，而 \(\sigma\) sensitivity 由 adjoint receiver wavefield 与 forward electric field 本身相关得到。它和上面的二阶电场推导在“相关成像条件”这一点上是一致的；时间导数阶数的差异来自所选方程形式、状态变量和离散化方式。这个差异已经在 `formula_summary.md` 中标记为后续需要重点核对的实现细节。

## 5. 单参数反演：先把问题降到可理解

单参数反演是理解 GPR-FWI 的第一层台阶。它通常只更新 \(\epsilon_r\)，或更新与 \(\epsilon_r\) 等价的速度参数。这样做的好处是问题更可诊断：数据残差只能通过一个参数通道来解释，错误来源更容易定位。

如果只反演 \(\epsilon_r\)，核心物理直觉非常清楚：

\[
v=\frac{1}{\sqrt{\mu\epsilon}}
=
\frac{1}{\sqrt{\mu_0\epsilon_0\epsilon_r}}
\quad(\mu_r\approx 1).
\]

也就是说，\(\epsilon_r\) 越大，电磁波速度越低，走时越长。于是 \(\epsilon_r\) 的错误主要表现为走时错误、相位错位和反射界面位置偏移。对 L2 waveform objective 而言，这类相位错误一旦超过半个周期，就可能触发 cycle skipping。

因此，单参数 \(\epsilon_r\) 反演最适合作为第一性原理验证实验：

1. 固定 \(\sigma\)，只让 \(\epsilon_r\) 更新。
2. 用有限差分梯度检查确认 adjoint gradient 的方向和尺度。
3. 观察目标函数是否下降、合成波形是否逐步对齐观测波形。
4. 再逐步加入平滑、照明补偿、更复杂目标函数和更复杂模型。

如果只反演 \(\sigma\)，问题通常更困难。原因是 \(\sigma\) 对数据的主要影响是振幅衰减和波形拖尾，而振幅还会受到 source wavelet、天线耦合、几何扩散、噪声和边界吸收影响。因此，\(\sigma\) 反演更容易把非介质因素误解释为电导率结构。

从实验设计角度看，更稳妥的顺序是：

1. 先做单参数 \(\epsilon_r\) 反演，验证相位/走时通道。
2. 再固定真值或可信的 \(\epsilon_r\)，只反演 \(\sigma\)，测试振幅/衰减通道。
3. 最后进入 \((\epsilon_r,\sigma)\) 双参数同步反演。

这个顺序不是为了回避复杂性，而是为了让每一步失败时都有明确解释。若单参数梯度都无法通过检查，直接进入双参数或神经网络重参数化只会把错误藏得更深。

## 6. 双参数反演与 Crosstalk

双参数 GPR-FWI 的目标是同时恢复：

\[
\mathbf{m}=(\epsilon_r,\sigma).
\]

理想情况下，\(\epsilon_r\) 解释速度和相位，\(\sigma\) 解释衰减和振幅。但真实数据中二者不会干净分工。在线性化意义下，数据扰动可以写成：

\[
\delta\mathbf{d}
\approx
J_{\epsilon_r}\delta\epsilon_r
+
J_\sigma\delta\sigma.
\]

如果 \(J_{\epsilon_r}\) 和 \(J_\sigma\) 的作用方向不够独立，同一个数据残差既可以被 \(\delta\epsilon_r\) 解释，也可以被 \(\delta\sigma\) 解释，crosstalk 就会出现。它不是某个优化器的小毛病，而是多参数反演的结构性病态。

用 Gauss-Newton 的块 Hessian 看得更清楚：

\[
\begin{bmatrix}
J_{\epsilon_r}^T J_{\epsilon_r}
&
J_{\epsilon_r}^T J_\sigma
\\
J_\sigma^T J_{\epsilon_r}
&
J_\sigma^T J_\sigma
\end{bmatrix}.
\]

对角块描述每个参数自己的可分辨性，非对角块描述两个参数之间的耦合。非对角块越强，\(\epsilon_r\) 和 \(\sigma\) 的 trade-off 越严重。此时，data misfit 下降并不等价于两个参数都恢复正确。

Lavoue et al. (2014) 给出了一个很有启发性的频域双参数例子。他们在 2D frequency-domain GPR-FWI 中同时反演 permittivity 和 conductivity，并指出：

- GPR 数据通常对 permittivity 更敏感。
- conductivity 也会影响振幅和相位，因此不能简单忽略。
- 先反演 permittivity、再反演 conductivity 的 cascaded strategy 可能失败，因为第一步中的 permittivity 误差会系统性映射成 conductivity artifacts。
- 同时反演通常更合理，但必须处理参数尺度、灵敏度差异和正则化。

他们使用参数缩放来调节两个参数的相对更新尺度：

\[
(\epsilon_r,\sigma_r/\beta),
\]

其中 \(\beta\) 控制 conductivity 相对于 permittivity 的更新权重。直观理解是：

- \(\beta<1\)：压低 conductivity 更新，让反演先更依赖 permittivity。
- \(\beta>1\)：放大 conductivity 更新，但容易引入 conductivity 振荡和伪影。

这对后续实验非常重要：双参数反演不能只看最终 data misfit。不同 \(\beta\) 可能得到相近的数据拟合，却给出完全不同的 conductivity 模型。换句话说，数据拟合好不等于参数恢复可信。

Lavoue et al. 还加入 conductivity 的 Tikhonov regularization：

\[
C(\mathbf{m})
=
C_D(\mathbf{m})
+
\lambda C_M(\mathbf{m}),
\]

\[
C_M(\mathbf{m})
=
\frac{1}{2}
\sigma_r^T D\sigma_r,
\]

其中 \(D\) 与 Laplacian smoothing 相关。这个正则化的作用不是“让结果变好看”，而是抑制 conductivity 中用于补偿错误低波数结构的高波数伪影。

这里需要区分两个概念：

- Parameter scaling 改变参数空间的相对权重，主要是引导反演路径。
- Regularization 改变目标函数，主要是约束模型结构。

在双参数 GPR-FWI 中，两者经常需要同时存在。只靠 L-BFGS-B 这样的 quasi-Newton 方法并不一定能自动解决参数尺度和 crosstalk，因为近似 Hessian 未必足够准确地平衡不同参数类型。更可靠的做法是把双参数实验拆成一组诊断：分别检查单参数灵敏度、参数缩放、正则化强度、照明补偿，以及错误初始模型下的误差转移方向。

## 7. Time-Domain GPR-FWI 技术路线

时域 GPR-FWI 的优势是物理过程直观：给定一个时域 source wavelet，用 FDTD 正向推进 Maxwell 方程，接收器记录完整 radargram，再把数据残差作为伴随源反向传播。它非常适合解释“为什么梯度是 forward wavefield 和 adjoint wavefield 的相关”。

一个标准 time-domain adjoint FWI 闭环可以写成：

1. 给定当前模型 \(\mathbf{m}_k=(\epsilon_r,\sigma)\)。
2. 对每个 source 做正演，得到 \(\mathbf{u}_s(\mathbf{m}_k)\)。
3. 用接收算子取出合成数据：

\[
\mathbf{d}^{syn}_s=\mathbf{P}\mathbf{u}_s(\mathbf{m}_k).
\]

4. 计算 residual：

\[
\mathbf{r}_s=\mathbf{d}^{syn}_s-\mathbf{d}^{obs}_s.
\]

5. 把 residual 作为 adjoint source 从 receiver 位置反向传播。
6. 在每个网格点累加 forward wavefield 和 adjoint wavefield 的时间相关，形成 \(\epsilon_r\) 和 \(\sigma\) 梯度。
7. 对梯度做必要的 scaling、smoothing、masking 或 illumination compensation。
8. 用 steepest descent、conjugate gradient、L-BFGS、Adam/RMSprop 等优化器更新模型。
9. 重复直到 data misfit、模型变化量或验证指标满足停止条件。

这里的关键不是“反向传播”这个词本身，而是 adjoint wavefield 的物理含义：它是由接收端残差激发出来、沿时间反向传播的误差信号。正演波场告诉我们某个位置是否被 source 照亮，伴随波场告诉我们该位置是否能解释接收端残差。两者相乘并在时间上累加，就得到该位置对目标函数的贡献。

在时域实现中，以下细节会直接决定公式能不能落到代码里。

第一，forward wavefield 的保存。梯度需要 forward wavefield 和 adjoint wavefield 在同一时间对应，所以最直接的做法是保存全部正演波场。但这会消耗大量内存。常见替代方案包括 checkpointing、只保存关键时间片、或在伴随传播阶段重新正演恢复波场。

第二，Yee grid 或 staggered grid 的变量位置。电场、磁场、\(\epsilon\)、\(\sigma\) 可能不在完全相同的网格位置。梯度累加时必须确认参数更新位置和场量插值方式，否则会出现数值错位：公式看起来正确，更新却落在错误位置。

第三，PML 区域。PML 是为了吸收边界反射而引入的人工区域。通常不希望反演 PML 参数，也不希望 PML 内的强数值效应污染物理模型，所以梯度常需要在 PML 区域置零或施加 mask。

第四，多炮累加。完整梯度是所有 source 的贡献之和：

\[
\nabla \Phi(\mathbf{m})
=
\sum_s
\nabla \Phi_s(\mathbf{m}).
\]

如果 source 很多，可以使用 mini-batch 或 stochastic source encoding，但这会引入梯度噪声，需要更谨慎的步长和收敛判断。

第五，步长和尺度。GPR 中 \(\epsilon_r\) 和 \(\sigma\) 的量纲差异很大，且 \(\sigma\) 梯度常比 \(\epsilon_r\) 更不稳定。双参数更新时必须考虑 parameter scaling，否则优化器可能把残差错误地压到 conductivity 上。

第六，梯度验证。每次改变方程形式、源项定义、残差符号、参数化方式或网格插值方式后，都应重新做 finite-difference gradient check。对初学者来说，这一步很像给推导和代码之间搭一座桥：它不证明反演一定成功，但能排除“方向写反了、尺度差太多、参数位置错了”这类最致命错误。

## 8. 目标函数、正则化和照明补偿

### 8.1 目标函数

目标函数决定“什么样的数据差异被认为重要”。最基本的选择是 L2 waveform misfit：

\[
\Phi_{L2}(\mathbf{m})
=
\frac{1}{2}
\sum_s
\|\mathbf{P}\mathbf{u}_s(\mathbf{m})-\mathbf{d}^{obs}_s\|_2^2.
\]

它的优点是简单、梯度推导清楚、和 adjoint-state method 配合自然。缺点也很明显：当合成波形和观测波形相差超过半个周期时，L2 objective 可能把错误相位当成正确下降方向，形成 cycle skipping。

Normalized 或 correlation-based objective 的目标是降低 source amplitude、receiver gain 或几何扩散对振幅的影响。它们通常更关注波形形状或相对相似度，而不是绝对振幅。这对 GPR 很有意义，因为 GPR 的振幅除了介质参数之外，还受天线耦合、源波形、近场效应和仪器响应影响。

Source-independent objective 试图绕开“source wavelet 不准确”的问题。Liu et al. (2022) 使用 cross-convolution 的思路，把 modeled trace 和 observed reference trace、observed trace 和 modeled reference trace 组合起来，使目标函数中两项含有相同的 source wavelet 因子。这样，即使 source wavelet 不完全准确，反演仍然可能推进。

Envelope objective 用 Hilbert transform 构造包络：

\[
E_{env}(t)=\sqrt{E(t)^2+H(E(t))^2}.
\]

包络更强调振幅包络和大尺度到时信息，通常比原始振荡波形更不容易陷入局部极小值。Liu et al. (2022) 把 source-independent 思想和 envelope objective 结合，用于 cross-hole GPR。需要谨慎的是，convolution、cross-correlation 和 envelope transform 本身也会引入新的非线性，所以它不是无条件优于 L2，而是适合 source uncertainty 和 cycle skipping 比较严重的情形。

Frequency-domain objective 通常选择若干频率分量来反演。优点是可以从低频到高频做 multiscale，也可以减少每次反演使用的数据量。Lavoue et al. (2014) 说明，双参数 GPR-FWI 中 permittivity 和 conductivity 对数据的相对影响随频率变化，因此 broad frequency bandwidth 对同步恢复两者很重要。

Laplace-domain objective 会强调信号早期和低频/平滑成分，常用于缓解 cycle skipping 或低频缺失。它可以作为 time-domain 和 frequency-domain 之外的补充路线，但本报告后续仍以 time-domain 为主线。

Meng et al. (2019) 给出了一个直接面向 cross-hole radar 的例子：在 Laplace 域中使用 logarithmic objective 反演 \(\epsilon\) 和 \(\sigma\)，主要目的不是替代 time-domain FWI，而是为 time-domain FWI 提供比 ray-based inversion 更平滑、更合适的初始模型。

Optimal-transport objective 是另一个处理 cycle skipping 的方向。Hunziker et al. (2025) 在 crosshole GPR-FWI 中采用先 OT 后 LS 的策略：早期用 OT 的宽吸引域靠近正确模型，后期切换到 LS 以获得更明确的局部收敛。这种策略提醒我们，目标函数可以按反演阶段切换，而不是从头到尾固定一个 misfit。

因此，目标函数可以按用途粗略分成三类：L2 负责最基本的局部波形拟合；envelope、Laplace 和 OT 负责扩大早期迭代的吸引域；source-independent 和 normalized/correlation objective 负责降低源波形和振幅标定误差的影响。

### 8.2 正则化

正则化的作用是把“只拟合数据”变成“在合理模型集合中拟合数据”。一个通用写法是：

\[
\Phi_R(\mathbf{m})
=
\Phi_D(\mathbf{m})
+
\alpha R(\mathbf{m}).
\]

Tikhonov regularization 常用于抑制模型过度振荡：

\[
R(\mathbf{m})
=
\frac{1}{2}
\|\mathbf{L}(\mathbf{m}-\mathbf{m}_{ref})\|_2^2.
\]

如果 \(\mathbf{L}\) 是 identity，它约束模型不要偏离参考模型太多；如果 \(\mathbf{L}\) 是 gradient 或 Laplacian，它约束模型平滑。Lavoue et al. (2014) 对 conductivity 使用了与 Laplacian 相关的 Tikhonov 项，因为 conductivity 更容易产生高波数伪影。

Total variation regularization 更偏向保边平滑：

\[
R_{TV}(\mathbf{m})
=
\int
\sqrt{|\nabla \mathbf{m}|^2+\eta^2}
d\mathbf{x}.
\]

它适合有块状异常体或层状界面的模型，但可能带来 staircasing，也可能过度偏向分段常数结构。

Bound constraints 也很重要。例如 \(\epsilon_r\) 和 \(\sigma\) 都应在物理合理范围内。L-BFGS-B 的 `B` 就表示 bound-constrained，它可以在优化过程中保持参数不跑出给定上下界。

Neural-network parameterization 或 implicit representation 可以看成一种隐式正则化。Sun et al. (2024) 的 IFWI 用 neural network 表示多参数模型，并利用 neural network 的 frequency principle，使模型倾向于先恢复低频/大尺度结构，再恢复高频细节。但它不是从数学上消灭 crosstalk，而是改变了模型空间和优化路径，所以需要用对照实验验证它到底缓解了什么。

选择正则化时要避免一个常见误区：目标函数下降、模型更平滑，并不自动意味着物理参数更真实。正则化强度 \(\alpha\) 本身应该成为实验变量，通过同一模型、同一初始条件、同一目标函数下的对比来判断。

### 8.3 Illumination Compensation

Illumination 指的是模型中不同区域被 source-receiver 系统“看见”的程度。即使梯度公式完全正确，照明不均也会让更新集中在近源、近接收器或强波场区域。GPR-FWI 中 illumination 不均匀主要来自：

- source 和 receiver 几何覆盖有限。
- 近源、近接收器波场很强，深部或遮挡区域波场弱。
- conductivity 衰减会让深部或远 offset 信号变弱。
- surface-to-surface acquisition 比 crosshole acquisition 更容易出现照明盲区。

Meles et al. (2012) 的 sensitivity/resolution 分析说明，同样的数据 residual 对不同区域的约束能力是不一样的。近源强波场可能主导梯度，使反演更关注已经被强照亮的位置，而不是物理上最需要修正的位置。

常见处理方式包括：

1. Gradient smoothing：降低局部尖峰，减少更新振荡。
2. Depth/time gain 或 trace weighting：平衡早到强信号和晚到弱信号，但必须小心不要同时放大噪声。
3. Pseudo-Hessian 或 diagonal Hessian normalization：

\[
\tilde{g}(\mathbf{x})
=
\frac{g(\mathbf{x})}
{H_{diag}(\mathbf{x})+\eta}.
\]

4. Source-receiver illumination mask：对不可分辨区域降低更新权重。
5. Acquisition design：增加角度覆盖，例如 crosshole、多侧观测或更密集 receiver。

Illumination compensation 和 regularization 不一样。Regularization 说的是“模型应该长什么样”，illumination compensation 说的是“梯度在不同位置的可信度和尺度是否公平”。前者约束模型结构，后者预处理更新方向。

在后续实验中，照明补偿至少应该用三张图诊断：原始梯度、补偿后的梯度、以及 forward/adjoint wavefield energy 或 pseudo-Hessian diagonal。这样才能判断补偿是在修正几何照明，还是只是在把噪声和边界伪影放大。

## 9. 典型实验模型设置

文献中的 GPR-FWI 实验大致可以分成四类：crosshole、on-ground/surface-to-surface、简单异常体模型、复杂 benchmark 模型。它们各自回答的问题不同。

Crosshole GPR 的优点是 source 和 receiver 分布在两个钻孔中，照明角度更丰富，直达波和透射波对介质速度/介电常数比较敏感。Meles et al. (2012) 的 sensitivity/resolution 分析和 Liu et al. (2022) 的 source-independent envelope objective 都和 cross-hole 设置紧密相关。它适合研究：

- time-domain adjoint gradient。
- acquisition geometry 对 illumination 的影响。
- source-independent objective 对 source wavelet uncertainty 的缓解。
- \(\epsilon_r/\sigma\) 双参数同步更新。

On-ground 或 surface-to-surface GPR 更接近很多工程场景，但反演更病态。source 和 receiver 都在地表附近，深部和侧向结构照明不足，conductivity 更容易出现非唯一性。Lavoue et al. (2014) 的频域多偏移实验说明，在 surface-to-surface acquisition 中，即使 data misfit 很接近，不同 parameter scaling 也可能给出差异很大的 conductivity 模型。因此这类实验适合研究：

- partial illumination。
- parameter scaling。
- conductivity regularization。
- frequency sampling strategy。
- data misfit 与 model correctness 的不一致。

简单异常体模型，例如圆柱体、十字形异常体、层状模型，是最适合做第一性原理验证的实验。它们不一定最真实，但最容易解释。推荐用它们检查：

- forward solver 是否稳定。
- adjoint gradient 是否正确。
- 单参数 \(\epsilon_r\) 是否能恢复相位结构。
- 错误 \(\sigma\) 是否污染 \(\epsilon_r\)。
- 错误 \(\epsilon_r\) 是否导致 \(\sigma\) artifact。

复杂 benchmark，例如 Marmousi 或 Overthrust 类模型，可以测试算法在强横向变化、多尺度结构和复杂照明下的表现。但它们不适合一开始就用来判断梯度公式是否正确，因为失败原因太多：初始模型、频率、步长、PML、正则化、优化器和照明都可能混在一起。

结合当前本地项目，已有实验线可以映射到下面几个理论问题。

| Local line | Role | Theory link |
| --- | --- | --- |
| `Gpr_fwi/` baseline rewrite notes | 回归最基本 FWI 流程 | forward data、residual、gradient、update 的最小闭环 |
| `Gpr_fwi/mode2_Tikhonov/` | 二阶 Tikhonov 正则化 | \(J=J_{data}+J_{reg}\)，\(\nabla^2(\nabla^2 m)\)，平滑与过平滑 |
| `Gpr_fwi/mode2_unet/` | 双参数 UNet 重参数化 | neural parameterization、\(\epsilon/\sigma\) 双网络、物理范围约束 |
| `Gpr_fwi/mode2_Hybrid/` | hybrid optimization variant, details pending | 传统梯度与网络先验结合，需进一步确认具体实现 |
| `Gpr_fwi/mode1_unet_fix_parallel/` | MPI shot parallelization | 多炮梯度累加、source 级并行 |
| `Fast-GPR-FWI/` | CUDA kernel + PyTorch 双参数加速 | 高性能正演/梯度、autograd 集成、双参数 inversion |
| `隐式FWI/` | IFWI/dropout-IFWI 复现 | implicit representation、frequency principle、dropout 正则化、illumination diagnosis |
| `marmousi_paper/gpr-inversion/` | clean migration target | 配置化实验矩阵、Marmousi/Overthrust、eps-only/sig-only/twopara/eps-then-sig |

这些本地实验给出一个很自然的后续路线：不要直接跳到最复杂的 IFWI 或 Marmousi，而是先构造最小可验证链条，再逐步打开复杂度。

推荐实验阶梯是：

1. 简单模型 + 单参数 \(\epsilon_r\) + L2 waveform objective。
2. 同一模型 + finite-difference gradient check。
3. 固定 \(\sigma\) 只反演 \(\epsilon_r\)，再固定 \(\epsilon_r\) 只反演 \(\sigma\)。
4. 同时反演 \(\epsilon_r,\sigma\)，观察 crosstalk。
5. 加入 parameter scaling 和 Tikhonov/TV 正则化。
6. 加入 illumination compensation 或 pseudo-Hessian normalization。
7. 比较 L2、normalized、source-independent、envelope objective。
8. 最后再比较 traditional parameter grid、UNet reparameterization、implicit representation。

这样设计的好处是，每一步只改变一个核心因素。对初学者来说，这叫 control variable，也就是“控制变量”：如果实验结果变了，我们能知道主要是哪个因素造成的，而不是所有东西一起变化后无法解释。

## References Used So Far

- Meles et al. 2012, IEEE TGRS, DOI: `10.1109/TGRS.2011.2170078`.
- Busch et al. 2012, Geophysics, DOI: `10.1190/GEO2012-0045.1`.
- Lavoue et al. 2014, GJI, DOI: `10.1093/gji/ggt528`.
- Ernst et al. 2007, IEEE TGRS.
- Meng et al. 2019, Remote Sensing, DOI: `10.3390/rs11161839`.
- Liu et al. 2022, Remote Sensing, DOI: `10.3390/rs14194878`.
- Sun et al. 2024, GJI, DOI: `10.1093/gji/ggae420`.
- Hunziker et al. 2025, Journal of Applied Geophysics.
