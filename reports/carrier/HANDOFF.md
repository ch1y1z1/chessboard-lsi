# 载频剪切干涉实验交接文档

**日期**：2024-10-10  
**分支**：`claude/project-status-review-e393f1`  
**状态**：M2 调参完成，准备启动 pilot 实验

---

## 当前进度

### ✅ 已完成并推送

| 阶段 | 提交 | 描述 | 耗时 |
|------|------|------|------|
| M0 | 70c5392 | 地基：模型、求解器、成本标定、锚点 | 前期 |
| M1 | 7d70594 | 理论预测 P1-P7（40 dev 真值） | 13 分钟 |
| M2 调参 | c7632b1 | 傅里叶链与求解器超参数调优 | 19 分钟 |

**关键成果**：
- 44 项测试通过，4 个旧锚点复现
- 理论预测冻结在 `results/carrier/m1/predictions.json`
- 调优参数：傅里叶链 (r=6.0, t=0.3, e=2)，adam lr (p35: 3e-4, p12: 1e-4)

### ⏳ 下一步：M2 pilot 实验

在新设备上继续运行以下命令（预计 1 小时）：

```bash
# 克隆并切换到工作分支
git clone https://github.com/ch1y1z1/chessboard-lsi.git
cd chessboard-lsi
git checkout claude/project-status-review-e393f1

# 安装依赖
uv sync

# 运行 pilot 实验（E1-E4 + D1 算力决策）
cd .claude/worktrees/project-status-review-e393f1  # 或直接在主目录
CARRIER_WORKERS=6 uv run python -m carrier_study.m2_pilot e1 e2 e3 e4 d1
```

**输出**（约 1 小时后）：
- `results/carrier/m2/e{1,2,3,4}_pilot.csv.gz`：pilot 实验数据
- `results/carrier/m2/d1_estimate.json`：主实验墙钟时间外推
- `results/carrier/m2/d1_decision.json`：**自动规模决策**（若预计 > 10h 则缩减）

---

## 实验协议（已冻结）

### 真值集
- **dev**: 40 真值（调参 + pilot）
- **test**: 112 真值（主实验）
- 分层：shape (pow1/pow2/flat/sparse) × RMS (0.03–5.0) × rep (0-2)

### 噪声档
- **高斯**：G20/G30/G40/G50/G60（峰值 SNR，dB）
- **Poisson**：P1e3/P1e4/P1e5（峰值光子数）

### 求解器配置（16 个）
- 最小二乘专用：GN, LM (main), LM (minpack), TRF
- 一阶方法：BFGS, L-BFGS, CG, BB 步长 GD, Nesterov, GD, Adam
- 固定步长（调优/默认）：gd_fixed, nesterov_fixed

### 误差度量
- **白化度量**：d_rel = sqrt((q - q_ref)ᵀ FIM (q - q_ref) / p)
- **参考解**：多起点 GN（Poisson 用 IRLS 精修）
- **成本单位**：前向当量（串行标定，消除 CPU 争抢影响）

---

## 主实验（M3）规模

**默认规模**（D1 若预计 ≤ 10h）：
- **E1**（估计器对比）：112 test 真值 × 2 场景 × 8 噪声 × **4 种子**
- **E2**（吸引盆）：3 周期 × 6 真值 × (2 模式 × 9 r × **12 方向** + 零/傅里叶) + p=12 补充
- **E3**（成本与精度）：32 真值 × 8 噪声 × 2 p × 2 初值 × 16 配置（**2 重复**）
- **E3k**（P4 条件数）：3 周期 × 2 p × 代表性求解器
- **E4**（失配与大像差）：
  - 失配：12 真值 × 31 场景 × (无噪声 + **4 种子** G40)
  - 大像差：56 真值 × 2 场景

**自动缩减规则**（若预计 > 10h，依次应用直到 ≤ 10h）：
1. E1 种子 4 → 2
2. E2 方向 12 → 8
3. E3 重复 2 → 1

**运行主实验**（D1 决策后）：
```bash
CARRIER_WORKERS=6 uv run python -m carrier_study.m3_main
```

---

## 理论预测（P1-P7，pilot 将验证）

### P1: MLE 达到 CRB
**预测**：matched、SNR ≥ 30 dB 及 Poisson ≥ 1e4、RMS ≤ 1 → NMSE ∈ [0.9, 1.1]  
**检验**：E1 数据，matched 场景 MLE 的 E[d_rel²]

### P2: 傅里叶链噪声放大
**预测**：E[d_rel²] 按 RMS 分档的中位数（见 `predictions.json`）  
**检验**：E1 数据，fourier_default 的实测 E[d_rel²] vs 预测

### P3: 失配偏差一阶公式
**预测**：MLE/傅里叶链在各失配场景下的偏差（一阶线性化）  
**检验**：E4 失配数据，实测偏差 vs 预测，相对误差 < 20%

### P4: 一阶方法成本 ∝ κ 的幂次
**预测**：log(cost) ~ slope × log(κ)，slope 接近理论值（GD: 1.0, 加速: 0.5）  
**检验**：E3k 跨三档周期的成本 vs κ 对数回归斜率

### P5: 一阶方法何时更便宜
**预测**：在所有噪声档下，一阶方法中位成本都不低于 GN（p=35）  
**检验**：E3 数据，各噪声档下最好的一阶方法成本 vs GN

### P6: 吸引盆标度 r50·s ≈ const
**预测**：50% 成功半径 r50（波前 RMS）与剪切量 s 的乘积为常数；p=12 的 r50 约为 p=35 的 2.5 倍（梯度尺度比）  
**检验**：E2 数据，三档周期的 r50·s；p=12 vs p=35 的 r50 比值

### P7: 傅里叶链定义域失效
**预测**：瓣局部频率最大值超出低通半径（约 4 cyc/unit）时失效  
**检验**：E4 大像差数据，成功率 vs lobe_freq_max 的分箱统计，找 50% 失效点

---

## 技术要点

### 环境要求
- **内存**：每个进程峰值 ~300 MB，6 进程需要 ~2 GB 可用
- **进程数**：通过 `CARRIER_WORKERS` 环境变量控制（建议 5-6）
- **Python 依赖**：通过 `uv sync` 自动安装

### 确定性保证
- 所有随机种子固定在 `carrier_study/config.py`
- 不同设备运行结果完全一致（逐像素、逐数值）

### 关键文件

**代码**：
```
carrier_study/
  ├── config.py, truths.py          # 配置与真值集
  ├── zern.py, nominal.py           # 模型
  ├── fourier.py, theory.py         # 估计器与理论
  ├── solvers.py, problem.py        # 优化
  ├── e1.py, e2.py, e3.py, e4.py    # 实验 worker
  ├── m2_pilot.py, m3_main.py       # 编排脚本
  └── m4_report.py, report_e*.py    # 汇总分析
```

**数据**（已生成）：
```
results/carrier/
  ├── m0/  cost_units.json, anchors.json
  ├── m1/  predictions.json ← 主实验前冻结
  └── m2/  fourier_tuning.json, solver_tuning.json
```

**测试**：
```bash
uv run pytest tests/test_carrier_study.py  # 44 项，全部通过
```

---

## 预期时间线

| 阶段 | 耗时 | 说明 |
|------|------|------|
| M2 pilot | ~1 小时 | 当前待运行，产出 D1 决策 |
| G2 关口 | 人工审查 | 确认 P1-P7 与协议 |
| M3 主实验 | ≤ 10 小时 | 按 D1 自动缩减规模 |
| M4 报告 | ~30 分钟 | 表格、图、P1-P7 核对 |

**总计**：~12 小时机器时间 + 人工审查

---

## G2 关口检查清单

pilot 完成后，人工审查以下内容再启动主实验：

- [ ] **P1-P7 方向正确**：pilot 数据的趋势与预测一致（允许数值偏差）
- [ ] **无灾难性失败**：估计器、求解器没有大规模 NaN/发散
- [ ] **D1 决策合理**：
  - 若预计 ≤ 10h：全量运行
  - 若预计 > 10h：检查自动缩减后的规模是否可接受
- [ ] **文件完整**：
  - `results/carrier/m2/d1_decision.json` 存在且格式正确
  - `results/carrier/m2/e{1,2,3,4}_pilot.csv.gz` 非空

**通过 G2 后**：
```bash
CARRIER_WORKERS=6 uv run python -m carrier_study.m3_main
```

---

## 故障排查

### 内存不足
- 减少进程数：`CARRIER_WORKERS=4`
- 检查 swap：`sysctl vm.swapusage`（macOS）或 `free -h`（Linux）

### 进程卡死
- 检查 CPU：`top` 或 `htop`
- 单个真值测试：
  ```python
  from carrier_study.truths import truth_set
  from carrier_study import config
  from carrier_study.e3 import worker
  tr = truth_set("dev", config.INDICES_MAIN)[0]
  worker((tr, 35, "G40", 0, ("fourier",), [], None))
  ```

### 结果验证
```bash
# 查看 M1 预测
uv run python -m carrier_study.m1_report | less

# 检查 pilot 数据量
zcat results/carrier/m2/e1_pilot.csv.gz | wc -l  # 应为数千行

# 渲染 pilot 报告（需要先完成 pilot）
uv run python -m carrier_study.m4_report results/carrier/m2
```

---

## 联系与回滚

**当前分支**：`claude/project-status-review-e393f1`  
**回滚点**：
- M0 地基：`70c5392`
- M1 + 调参：`c7632b1`（当前 HEAD）

**清理重跑**（若需要）：
```bash
rm -rf results/carrier/m2/*.csv.gz results/carrier/m3/
git checkout c7632b1  # 回到调参完成点
```

---

**下一步操作**：在新设备上执行 `M2 pilot` 命令，1 小时后检查 D1 决策，通过 G2 关口后启动 M3 主实验。
