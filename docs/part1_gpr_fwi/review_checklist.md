# Review Checklist

本文是下一轮或人工校对时使用的检查清单。

## Main Report

- [x] 所有章节已有正文，无 `To be expanded` 空段落。
- [x] Maxwell 方程、目标函数、Lagrangian、梯度形式已出现。
- [x] 单参数和双参数反演已有基本解释。
- [x] Crosstalk 已用 Jacobian/Hessian block 解释。
- [x] Time-domain workflow 已写成可执行闭环。
- [x] Objective / regularization / illumination 已有方法地图。
- [x] 本地实验线已映射到理论问题。
- [x] 为已 formula-checked 的关键文献补充轻量 PDF 页码锚点。
- [ ] 将轻量 PDF 页码锚点升级为人工核对后的 section/page/equation 引用。
- [x] 对 Meles et al. (2012) 做第一轮 section/equation 线索升级。
- [x] 对 Lavoue et al. (2014) 做第一轮 section/equation 线索升级。
- [x] 对 Liu et al. (2022) 做第一轮 section/equation 线索升级。
- [x] 补充更多经典 time-domain crosshole GPR-FWI 文献。
- [x] 补充 Laplace-domain 和 OT objective 的具体公式或实验结论。
- [x] 把 currently English-heavy sections 统一成更适合中文汇报的语气。
- [x] 添加中文执行摘要、证据边界和术语说明。
- [x] 润色 Sections 1-4，使研究问题、正演模型、目标函数和伴随梯度形成连续叙述。
- [x] 继续润色 Sections 5-8，使单/双参数、crosstalk、time-domain workflow 和方法地图达到同一汇报语气。
- [x] 润色 Section 9，使典型实验模型和本地实验线映射更像可执行路线图。
- [x] 完成最终一致性 QA，并记录到 `final_qa.md`。

## Formula Summary

- [x] Maxwell 方程。
- [x] 二阶电场方程。
- [x] L2 waveform objective。
- [x] Lagrangian。
- [x] \(\epsilon\)、\(\epsilon_r\)、\(\sigma\) 梯度。
- [x] velocity-permittivity chain rule。
- [x] Meles first-order/FDTD sensitivity view。
- [x] Liu source-independent envelope objective。
- [x] Lavoue frequency-domain biparameter formulas。
- [x] multi-source gradient accumulation。
- [x] pseudo-Hessian / illumination normalization。
- [ ] 核对二阶电场推导和一阶 FDTD sensitivity 的时间导数阶数差异。
- [ ] 为每个公式补“使用条件”和“离散实现注意事项”。
- [x] 与主报告 Section 5-8 的新表述同步，补充单参数/双参数诊断实验中的公式使用提醒。
- [x] 补充 finite-difference gradient check 和 illumination compensation 诊断图提醒。

## Literature Matrix

- [x] 本地 reference 已初步分类。
- [x] 五篇核心 PDF 已从 `not_read` 更新为 `skimmed`。
- [x] 将更多 `important` 文献更新到 `skimmed/read`。
- [ ] 补充 DOI/URL、authors、journal、year 字段。
- [ ] 明确哪些文献属于 time-domain、frequency-domain、Laplace-domain。

## Experiment Recommendations

- [x] 最小可验证实验。
- [x] Crosstalk 诊断实验。
- [x] Illumination compensation 实验。
- [x] Objective function 对比实验。
- [x] Regularization 实验。
- [x] Network / implicit representation 对比实验。
- [x] 指定最适合落地的本地代码路径。
- [x] 给出第一个 gradient check 的具体 config 草案。
- [ ] 将 config 草案转化为实际 `marmousi_paper/gpr-inversion/configs/` YAML 文件和最小 runner/test。
- [ ] 进入代码实现前确认任务边界：继续文献/报告，还是转入 runnable config。

## Git / Data Safety

- [x] 未提交 `GPR_references/` PDF。
- [x] 未提交实验输出数组或结果图。
- [x] 未提交未跟踪的 `Fast-GPR-FWI/`。
- [x] `docs/part1_gpr_fwi/` 已单独提交。
