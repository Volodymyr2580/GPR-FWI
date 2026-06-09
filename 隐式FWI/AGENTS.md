# AGENTS.md

本目录用于复现论文 `Implicit multiparameter full waveform inversion of multioffset ground penetrating radar data` 中的 IFWI 和 dropout-IFWI。

当前实现以 GPR IFWI 论文为复现目标，并借鉴 CVPR 2024 FR-INR 开源仓库提供的可靠坐标网络架构标准：

- 参考仓库：`FR-INR/`，来自 `https://github.com/CVL-UESTC/FR-INR.git`。
- 本项目代码不直接修改参考仓库；复现代码放在 `src/ifwi_gpr/`。
- FR-INR 只作为网络架构参考和实现标准：坐标 `(x,z)` 输入网络，输出 `epsilon_r` 和 `sigma`。
- GPR 正演/梯度暂时通过 PyTorch autograd bridge 调用参考 CPU/MPI 求解器。

## 安全规则

- 禁止批量删除文件或目录。
- 不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件。
- 如果需要批量清理实验结果，停止操作并让用户手动处理或明确确认下一步。

## 文档和编码

- 中文 Markdown、TXT、说明文档默认使用 UTF-8。
- PowerShell 读取中文文件时使用 `Get-Content -Encoding UTF8 "文件路径"`。
- 写中文说明文档时确保 UTF-8 编码。
- 需要持续维护 `progress.md`：每次实验记录 config、命令、关键结果与产物目录；`plan.md` 只放阶段规划与验收标准。

## Git 和网络

- 执行访问 GitHub 的 Git 命令时，不依赖用户全局代理。
- 优先对单次 Git 命令临时清空代理，例如：
  - `git -c http.proxy= -c https.proxy= clone <repo-url>`
  - `git -c http.proxy= -c https.proxy= fetch`
  - `git -c http.proxy= -c https.proxy= pull`
- 不主动修改全局 Git 代理配置。

## 当前复现边界

- 当前阶段只复现 IFWI 和 dropout-IFWI。
- 不复现传统 FWI、多尺度 FWI、UNet 反演等对比框架。
- 当前网络架构参考：
  - `E:\sci_research\GPR\隐式FWI\FR-INR`
  - `E:\sci_research\GPR\隐式FWI\src\ifwi_gpr\networks\fr_inr.py`
- 当前 GPR 正演/梯度桥接参考：
  - `E:\sci_research\GPR\marmousi_paper\gpr-inversion\src\gpr_inversion\experiments\overthrust\mode2_eps_then_sig\forward.py`
  - `E:\sci_research\GPR\marmousi_paper\gpr-inversion\src\gpr_inversion\experiments\overthrust\mode2_eps_then_sig\gradient.py`
- 不修改参考工程源码；本目录只通过桥接层调用或后续复制必要接口。
- 不再使用 JAX 环境作为主线；当前主线是 PyTorch 版 GPR IFWI/dropout-IFWI。

## 初学者解释要求

用户是前后端开发初学者。遇到以下内容时，需要用通俗语言解释背景、原因和影响：

- PowerShell、Git 代理、编码、依赖安装。
- Python package、配置文件、命令行入口。
- GPR 正演、梯度、FWI、IFWI、dropout、GPU、PyTorch、FR-INR。

简要概念：

- GPR 正演：给定地下介电常数和电导率，模拟雷达接收数据。
- FWI：让模拟数据逼近观测数据，从而反推地下参数。
- IFWI：不直接更新每个网格点，而是训练坐标神经网络 `Nθ(x,z)` 生成地下参数图。
- dropout-IFWI：训练时随机关闭一部分神经元，用来减轻网络过度拟合高频噪声。
- FR-INR：一种改进的隐式神经表示方法，用固定傅里叶基重参数化网络权重，让网络更容易表达高频细节。
