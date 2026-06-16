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
- [ ] 每个关键文献结论补充更细引用位置，例如 section/page/equation。
- [x] 补充更多经典 time-domain crosshole GPR-FWI 文献。
- [x] 补充 Laplace-domain 和 OT objective 的具体公式或实验结论。
- [ ] 把 currently English-heavy sections 统一成更适合中文汇报的语气。

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
- [ ] 指定最适合落地的本地代码路径。
- [ ] 给出第一个 gradient check 的具体 config 草案。

## Git / Data Safety

- [x] 未提交 `GPR_references/` PDF。
- [x] 未提交实验输出数组或结果图。
- [x] 未提交未跟踪的 `Fast-GPR-FWI/`。
- [x] `docs/part1_gpr_fwi/` 已单独提交。
