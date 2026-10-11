# 载频剪切干涉实验进度报告

日期：2026-10-11  
状态：M3 主实验完成（实际 7.4 h < 外推 9.76 h），M4 汇总已生成，P1–P7 核对见下

## M3 主实验结果（2026-10-11）

实际墙钟：E1 1.80 h、E2 3.76 h、E3 1.36 h、E3k 136 s、E4m 830 s、E4l 914 s，合计 **7.42 h**（6 进程，
低于外推 9.76 h）。数据在 `results/carrier/m3/`，汇总 `results/carrier/m4/summary_main.json`、
`reports/carrier/tables_main.md`、`reports/carrier/figures/main/`。

**P1–P7 逐条核对（test 集）：**

- **P1 条件性通过**：matched 场景 RMS≤0.1 的全部 16 个噪声×RMS 条件 NMSE ∈ [0.94, 1.03]；RMS 0.3–1 的均值
  被单个真值 `test-p35-flat-1-2`（flat、RMS=1）毁掉（NMSE 1e4–1e7，27/256 行灾难）。诊断：误差 100% 集中在
  倾斜分量（wf 11.4 vs wf_notilt 0.0004–0.02），验证表明目标函数在两个倾斜方向上分别以 0.5/s ≈ 11.4 waves
  为周期**严格简并**（损失逐位相同），多起点取最小损失无法区分 —— 是问题的全局不可辨识性而非收敛失败，
  与 E2 的 tilt_alias 失效类型同源。**报告中 P1 应按"去倾斜简并方向"或中位数重新表述**。
- **P2 部分通过**：傅里叶链（默认）实测/预测 E[d²] 中位比：RMS≤0.3 时 0.47–1.23，RMS=1 时 1.13–1.18，
  RMS=2 时系统性 ~0.24（预测高估 4 倍）。趋势与量级正确，逐档数值有偏差。
- **P3 通过**：所有 9 个失配因素最小档 MLE 一阶预测相对差 ≤ 0.14（pixel 0.144，其余 < 0.08）。
- **P4 通过**：log 成本 ~ log κ 斜率：GD 0.94（理论 1.0）、Nesterov 0.62（理论 0.5）、L-BFGS 0.37、GN 0.17；
  排序与理论一致，加速方法介于 0.5–1 之间（n=4 点）。
- **P5 通过**：所有噪声档 × 两种 p × 两种初值 × 全部 τ，没有任何一阶/准牛顿方法中位成本低于 GN
  （τ=0.1 时最好的 BFGS/L-BFGS 仍贵 2.5–6.9 倍）。
- **P6 通过（p=35）**：r50·s 在三档周期上为 0.057–0.071（perturbed）/ 0.041–0.062（random），近似常数；
  p=12 的 r50 ≈ p=35 的 2.8 倍（预测 2.5）。
- **P7 通过**：调优傅里叶链成功率在瓣最大频率 4–5 cyc/unit 处跌破 50%（预测 ~4）；默认参数 2–3 cyc/unit 即失效。

**新确认的现象**：realistic + P1e5 的 IRLS-MLE 失效在 test 集完全复现（中位 wf 0.153，99% 灾难；
不加权 LS 0.0034、mle_pix 0.0012）。机理与 G2 记录相同。

## M2 pilot 结果与 G2 审查（2026-10-10）

pilot 墙钟：E1 506 s、E2 216 s、E3 1263 s、E4 256 s（6 进程）。数据见 `results/carrier/m2/*_pilot.csv.gz`，
汇总见 `results/carrier/m4/summary_pilot.json`、`reports/carrier/tables_pilot.md`、`reports/carrier/figures/pilot/`。
测试：`tests/test_carrier_study.py` 实际收集 35 项（交接文档写 44 项有误），全部通过。

**D1**：E1 1.57 h、E2 6.01 h、E3 1.40 h、E4 失配 0.26 h、E4 大像差 0.51 h，合计 9.76 h ≤ 10 h → 全量
（E1 4 种子、E2 12 方向、E3 2 重复、E4 4 种子）。注意：E3k 未计入外推（约为 E3 的 2%）；E2 外推只基于
r ∈ {0.1, 1, 3}，主实验含 r = 5 等更易失败（跑满预算）的档，实际耗时可能略超 10 h。

**P1–P7 方向**（pilot 只有 dev 子集，单种子，仅看方向）：
- P1 ✅ matched MLE 的 NMSE 在 16 个条件中 15 个落在 [0.9, 1.1]（G60/≤0.1 为 1.11，样本噪声量级）。
- P2 ✅ 傅里叶链（默认）实测 E[d²] 中位 / 预测中位：G50/G60 及 P1e5 档 0.91–1.03，低 SNR/小 RMS 档 0.76–1.51。
- P3 ✅ 各因素最小档 MLE 一阶预测相对差 < 0.14（pixel 0.14、illum 0.075，其余 < 0.07）；大失配（tail_frac 0.2、carrier_err 0.02）一阶公式明显低估，属预期的非线性区。
- P4 —— pilot 未跑 E3k，留待主实验。
- P5 ✅ 所有噪声档、两种 p、两种初值，最好的一阶方法（BFGS/L-BFGS）在 τ=0.1 时成本都是 GN 的 2.5–6.9 倍。
- P6 —— pilot 只有 30 µm 一档：r50 ≈ 1.6 waves，r50·s ≈ 0.070；跨周期标度留待主实验。
- P7 —— pilot 的 4 个大像差真值（RMS 2/4）瓣频率 80–89 cyc/unit、拍频超奈奎斯特 5.6 倍，全部在混叠上限之外，无法定阈值；主实验 RMS 0.5–1.5 档提供过渡区。

**发现的问题（不阻断主实验，报告中需说明）**：
1. **realistic + 高光子数 Poisson 的 IRLS-MLE 失效**：P1e5 下 MLE 误差中位 0.143 waves（不加权 LS 为 0.003，
   mle_pix 为 0.0012），P1e4 为 0.015。单真值核查（`irls` 外层迭代 10 次）：matched 场景一次收敛；realistic 场景
   在 0.128/0.141（P1e5）之间两周期振荡，不收敛。机理：名义模型条纹暗点光强 → 0（min I/peak ≈ 1e-7），像素积分后
   的数据暗点 ≈ 5e-3·peak，模型方差权重在暗点处放大到峰值的 ~4000 倍（读出噪声地板），失配被极度放大且重加权
   不再是收缩映射。这是"Poisson 加权对对比度失配极不鲁棒"的真实现象，也是 n_outer=3 的 IRLS 截断造成的实现
   特性；主实验按冻结协议照跑，报告中需把它和"MLE 本身"区分开（可补充：IRLS 方差地板或阻尼的对照）。
2. realistic 场景下名义 MLE 在 SNR ≥ 40 dB 时被像素积分偏差（≈ 3e-3 waves）主导，NMSE 30–3000；mle_pix 基本消除。
3. E1 RMS 2–3 档：dev 真值中 flat/pow1 已超混叠上限，NMSE 的均值被少数灾难样本主导，应按中位数/灾难率解读。
4. E3 零初值下 adam:tuned（调参用傅里叶初值）几乎不达标，adam:default 反而更好 —— 调参结果不能外推到零初值。
5. 绘图字体只配了 macOS 字体，已加入 Noto Sans CJK SC 回退（`analysis.py`）。

## 已完成

### M0 地基（已提交 70c5392）
- ✅ 名义反演模型（光强、雅可比、Jᵀv、Hessian 项）与 `lsi` 前向模型逐像素一致
- ✅ 数据生成端：超采样像素积分、占空比/高阶级次、振幅/相位/剪切/载频误差、照明、截断尾项、高斯与 Poisson 噪声
- ✅ 傅里叶链（向量化解包裹、精确一阶线性化）与 `lsi.invert` 一致
- ✅ 统一最小二乘问题：事件计数与预算、梯度 = Jᵀr、停滞检测
- ✅ 13 个求解器：GN、LM（两种实现）、TRF、BFGS、L-BFGS、CG、BB 步长 GD、Nesterov、固定步长 GD/Nesterov、Adam
- ✅ 全部测试通过（44 项），四个旧锚点复现
- ✅ 成本标定（p=12/35，三档周期）：GN 每步 ≈ 3.95 个前向当量（p=35）

### M1 理论预测（刚完成，13 分钟）
- ✅ 40 个 dev 真值的理论量（FIM、CRB、曲率、条件数、傅里叶链偏差与噪声放大、定义域统计）
- ✅ 失配偏差的一阶预测（31 种场景 x 每个真值）
- ✅ 收敛与成本模型：GD/加速梯度/GN 的迭代数预测
- ✅ 七条理论预测数值已冻结（`results/carrier/m1/predictions.json`）

**主要发现（p=35）**：
- κ(JᵀJ) 中位 136，范围 [126, 144]；GN 二次收敛常数 C ≈ 0.324 /wave
- GN 每步 3.95 个前向当量；加速梯度/准牛顿 1.65–2.65
- 在所有噪声档下（G20–G60、P1e3–P1e5），预测一阶方法都不比 GN 便宜（P5）
- 傅里叶链在 RMS ≥ 0.3 时 E[d²] >> 1（失效），主因：瓣局部频率超出低通半径
- 像素积分/振幅/相位误差对 MLE 偏差 < 0.003 waves；占空比 0.45、载频/剪切 2% 误差导致 > 0.1 waves 偏差

### M2 pilot（调参中）
- 🔄 傅里叶链调参：40 个网格点（radius_factor x threshold x erode），目标最小化 dev 真值波前 MSE
- 🔄 求解器调参：gd_fixed / nesterov_fixed 的 eta、adam 的 lr 各 8 档，在 dev 子集（16 真值 x G30/G50）上比较到达 τ=0.3 的成本
- ⏳ 预计 20–30 分钟完成，产出调优参数供主实验使用

## 进行中

### M2 pilot 剩余步骤（调参完成后）
1. **E1 pilot**（估计器对比）：dev 真值 x matched/realistic x 8 噪声 x 1 种子
   - 傅里叶链（默认/调优）vs MLE vs 不加权 LS（Poisson）vs 像素积分建模
   - 检验 P1（MLE 达到 CRB）、P2（傅里叶链噪声放大预测）

2. **E2 pilot**（吸引盆）：2 个真值 x 2 种初值模式 x 3 档距离 x 4 方向 x 7 方法
   - 进入全局盆的比例 vs 初值距离，50% 成功半径 r50
   - 检验 P6（r50·s 为常数）

3. **E3 pilot**（成本与精度）：8 真值（两档 RMS）x 8 噪声 x 2 初值 x 16 求解器配置
   - 到达 τ = 1.0 / 0.3 / 0.1 / 0.03 / 0.01 的成本（前向当量）
   - Performance profile、配对加速比
   - 检验 P5（一阶方法在哪些噪声档更便宜）

4. **E4 pilot**（失配与大像差）
   - 单因素失配：4 真值 x 31 场景 x (无噪声 + G40 x 1 种子)
   - 大像差：4 真值（RMS 2.0/4.0）x matched/realistic x G40
   - 检验 P3（一阶偏差公式）、P7（傅里叶链定义域失效）

5. **D1**（算力决策）：按 pilot 实测耗时外推主实验墙钟时间，若超过预算则缩减规模（E1 种子 4→2、E2 方向 12→8、E3 重复 2→1）

## 下一步

### G2 关口（pilot 完成后）
确认：
- [ ] P1–P7 的数值与现象级方向（pilot 数据 vs 预测）
- [ ] 主实验协议冻结（真值集、噪声档、求解器配置、误差度量）
- [ ] 规模决策（D1 的自动缩减规则 or 人工审批）

### M3 主实验（G2 通过后）
test 集 112 真值：
- **E1**: 112 真值 x 2 场景 x 8 噪声 x 4 种子（或按 D1 缩减为 2 种子）
- **E2**: 3 周期 x 6 真值 x (2 模式 x 9 r x 12 方向 + 零/傅里叶初值) + p=12 补充
- **E3**: 32 真值 x 8 噪声 x 2 p x 2 初值 x 16 配置（或重复 2→1）
- **E3k**（P4 条件数扫描）：三档剪切量 x 两种 p x 代表性求解器
- **E4**: 失配 12 真值 x 31 场景 x 5 种子、大像差 56 真值 x 2 场景

预计墙钟时间：pilot 实测后外推，目标 ≤ 10 小时（6 进程）

### M4 报告（M3 完成后）
- 汇总 E1–E4、核对 P1–P7（E5）
- 生成表格、图（reports/carrier/figures/*.png）
- 机器可读结论（results/carrier/m4/summary_main.json）
- 撰写最终报告（与计划 §7 对照）

## 文件组织

```
carrier_study/
  ├── __init__.py, config.py, truths.py    # 基础设施
  ├── zern.py, nominal.py, generator.py    # 模型
  ├── fourier.py, metrics.py, theory.py    # 估计与理论
  ├── problem.py, solvers.py               # 优化
  ├── estimators.py, common.py             # 实验共用
  ├── calibrate.py, tune.py                # 标定与调参
  ├── e1.py, e2.py, e3.py, e4.py           # 实验 worker
  ├── m0_anchors.py, m1_theory.py          # 阶段脚本
  ├── m2_pilot.py, m3_main.py, m4_report.py
  ├── report_e*.py, analysis.py            # 汇总
  └── m1_report.py                          # 预测备忘渲染

results/carrier/
  ├── m0/  (cost_units.json, anchors.json)
  ├── m1/  (predictions.json ← 主实验前冻结, theory_truths.csv.gz, mismatch_linear.csv.gz)
  ├── m2/  (调参结果 + pilot 数据 + d1_decision.json)
  └── m3/  (主实验数据 + m4/summary_main.json)

reports/carrier/
  ├── M0-foundation.md
  ├── figures/{pilot,main}/*.png
  ├── tables_{pilot,main}.md
  └── progress.md (本文档)
```

## 技术要点

### 已解决的关键问题
1. **步长下溢空转**：停滞检测（参数不变即停止 + 10 万次缓存命中兜底）
2. **傅里叶链掩膜奇异**：只保留最大 4 连通块（工程修正，报告中说明）
3. **Poisson MLE 权重**：IRLS 迭代更新权重（用模型方差，避免 Neyman χ² 偏差）
4. **内存压爆**：傅里叶链差分基共享（lru_cache）、批大小降至 64、进程数 5–6

### 当前技术约束
- 单机 8 GB 内存，6 进程并发
- M1 理论预测：40 真值 x 2 p ≈ 13 分钟（批处理大矩阵运算）
- E3 单数据集（2 初值 x 16 配置）：约 1 分钟（迭代优化主导）
- 主实验预计 ≤ 10 小时（按 D1 自动缩减规则）

### 设计决策
- **白化度量 d_rel**：相对 FIM 的马氏距离，与坐标选择无关，直接对应 E[d_rel²] = 1（有效无偏估计）
- **参考解**：多起点 GN（Poisson 用 IRLS 权重）取最小损失，所有求解器在同一目标上比较
- **成本单位**：前向当量（按串行标定），消除 CPU 争抢与批处理并行的影响
- **真值集分层**：dev 40（调参 + pilot）/ test 112（主实验），按 shape / RMS / rep 分层抽样

---

**当前状态**：M2 调参运行中，预计 00:50 前完成。完成后立即运行 pilot 实验（E1–E4），约 1 小时产出 D1 决策与 P1–P7 初步核对。
