# 载频剪切干涉实验进度报告

日期：2025-10-10  
状态：M2 pilot 调参中，预计 30 分钟内完成

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
