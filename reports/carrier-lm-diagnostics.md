# 载频单帧 LM 曲率与实现差异诊断（T5）

日期：2026-10-09　分支：`analysis/carrier-lm-curvature-diagnostics-20261009`
基线：`experiment/single-frame-optimizers` @ `2c6e7c0`

## 0. 范围、口径与数据溯源

- 对象：单帧、已知空间载频、原始光强反演（`StackedIntensityProblem`，
  indices Z2–Z13，n=128，period=30 µm，即旧 `medium-carrier` 基准几何）。
- 输入数据（只读，未改动）：
  `/home/user/.lody/repos/github---ch1y1z1---chessboard-lsi/worktrees/cd15d640-c386-4307-a7c4-5966543b462f/output/medium-carrier/`
  —— 1400 行 runs.csv、trajectories.jsonl.gz、dataset.npz。
  旧 metadata 记 `revision=2d54a75` 且 `working_tree_status` 非空；
  本轮校验其 `source/` 快照逐文件 sha256 与基线 `2c6e7c0` 一致
  （lsi/*.py、benchmark_modulated.py 无差异），故旧数据可按基线源码解释。
  dataset.npz 中 `modulations` 与 `ForwardModel.carrier_phases(carrier_f0)`
  逐元素一致（atol 1e-12），载频口径确认。
- 统一诊断口径：全像素、q = scale⊙c（RMS 参数坐标）、
  r = (I−I_obs)/√M、loss = ‖r‖²/2；精确 Hessian
  H = JᵀJ + Σᵢ rᵢ∇²rᵢ（`analysis/carrier_curvature/hessian.py`）。
- 真值仅作离线评价（波前误差、到 ±truth 距离）；近真值初值仅用于局部
  机制诊断。所有近真值/真值点计算不作为部署证据。
- 导数验证（自有，`test_derivatives.py` 全部通过）：
  解析 J 对强度差分 rtol≤1e-4；精确 H 方向有限差分相对误差 ≤1e-5
  （实测端点处 1e-16–1e-11，初值大残差点 ~1e-11）；
  scaling='none'↔'rms' 链式变换一致；`problem_curvature` 内部对
  `problem.evaluate` 的 r/J 做逐元素断言（rtol 1e-10）。
- 并发机器上不计时、不排名；所有数值为功能/机制诊断。
- λ/ρ 轨迹由受控 instrumentation 捕获：monkeypatch 模块级
  `_solve_damped` 记录每次阻尼子问题调用，ρ 由评价历史按调用序重建，
  `finally` 中恢复；未复制或改写任何 LM 实现。
  SciPy `least_squares(method='lm')` 的 MINPACK 内部 λ 无公开 API，
  仅记录评价序列与终止状态（已标注为可观测性缺口）。

## 1. Part 1：三实现 LM 比较（6 场景）

场景 (case, init)：(0,0)、(0,6)、(5,2)、(10,8)、(14,4)、(17,9)，
覆盖 truth_rms 0.01–1.0 与零/非零初值。
原始记录见 `part1_lm_comparison.json` / `part1_lambda_rho.csv`。

### 表 A：原生默认配置（各实现自己的口径）

| 场景 | 主线 LM（`lsi.lm`，4096 黄金采样，c 坐标，cost=‖f‖²） | benchmark LM（`optimizers._local`，全像素，q 坐标，loss=‖r‖²/2，预算 500） | SciPy lm（scaling='none' 归一化残差，默认 ftol=xtol=gtol=1e-8） |
|---|---|---|---|
| (0,0) | conv=True, 5 iter, cost 1.45e-27, wfe 2.9e-16 | gradient_tolerance, 4 iter, loss 3.26e-25, wfe 2.7e-12 | xtol(status3), nfev4, wfe 1.3e-16 |
| (0,6) | conv=True, 6 iter, wfe ~1e-16 | gradient_tolerance, 4 iter, loss 7.2e-21, wfe 3.7e-10 | status3, wfe ~1e-16 |
| (5,2) | conv, 5 iter, wfe 5.0e-16 | grad_tol, 4 iter, wfe 7.8e-12 | status3, wfe 1.8e-16 |
| (10,8) | conv, 6 iter, wfe 2.1e-16 | grad_tol, 5 iter, wfe 1.1e-12 | status3, wfe 1.1e-16 |
| (14,4) | conv, 6 iter, wfe 2.1e-16 | grad_tol, 5 iter, wfe 3.1e-12 | status3, wfe 8.9e-17 |
| (17,9) | conv, 6 iter, wfe 3.0e-13 | grad_tol, 6 iter, wfe 6.6e-14 | status3, wfe 1.3e-16 |

读法：三实现均收敛到同一极小点（±truth，符号对齐后 wfe ≤~1e-10），
但**停在不同终点**。benchmark LM 的 loss 残量（如 7.2e-21 →
intensity_rms≈1.2e-10）恰好压在旧严格门槛 1e-10 上方；主线 LM 因更紧的
原生容差多走 1–2 步到达 ~1e-27。SciPy lm 的 MINPACK 终止于 xtol。

### 表 B：统一口径（同像素集/原始残差/c 坐标/同 c0）

像素=全 16384，目标=min ‖f(c)‖²（f=I−I_obs 未归一化），c0=同一场景初值。
benchmark LM 经 `scaling='none'` 的 `StackedIntensityProblem` 接入
（其残差为 f/√M，q=c）。

| 场景 | 主线 vs benchmark 前缀一致性 | benchmark 默认容差（n_iter） | benchmark 宽容差续跑（n_iter） |
|---|---|---|---|
| (0,0) | λ、ρ、迭代点逐位一致（max_Δλ=0, max_Δx=0） | 4（loss 3.3e-25） | 5（loss 1.4e-31） |
| (0,6) | 同上，一致 | 4（7.2e-21） | 5（2.2e-29） |
| (5,2) | 同上 | 4（2.8e-24） | 5（2.2e-31） |
| (10,8) | 同上 | 5（5.8e-26） | 6（8.1e-32） |
| (14,4) | 同上 | 5（5.4e-25） | 6（8.9e-32） |
| (17,9) | 同上（末步仅停机语义差，见下） | 6（3.7e-28） | 7（1.7e-31） |

- **数学一致性结论**：在共同前缀内两实现逐步产生**逐位相同**的
  阻尼步、λ 序列与接受/拒绝决策。残差整体缩放（1/√M）按尺度不变性
  不改变 Nielsen-ρ 与 λ 更新——数值上验证为完全一致（非近似）。
- 唯一前缀分歧（case 17 末步）：主线在算完阻尼步后先做 g/x 容差预检
  （未评价 trial 即停），benchmark 总是先评价 trial——是停机检查点
  位置差异，不是数值分歧。
- SciPy lm（同原始 f/J、c0）：走 MINPACK lmder 自有阻尼/对角缩放路径，
  评价序列不同，但 4–5 次评价内落到同一极小点邻域
  （端点距主线 incumbent ≤~1e-10 coeff 量级，各场景 wfe ≤1e-16–1e-10）。

### 逐项差异清单（实现层）

| 项 | 主线 `lsi.lm` | benchmark `optimizers` | SciPy `least_squares('lm')` |
|---|---|---|---|
| 残差定义 | f=I−I_obs，未归一化；cost=‖f‖² | r=f/√M；loss=‖r‖²/2（≡ ½mean(f²)） | 调用方给定；cost=½‖f‖² |
| 参数坐标 | 原始 c（waves） | q=scale⊙c（RMS，可 'none'） | 调用方坐标 |
| 像素集 | 4096 黄金低差异采样（确定性） | 全部 mask 像素（默认全图） | 调用方给定 |
| g 范数语义 | ‖g‖₂（raw c 坐标）≤1e-12 | ‖g‖∞（scaled q 坐标）≤1e-10 | MINPACK 正交性度量（fvec 与 J 列夹角余弦）≤gtol |
| xtol 语义 | ‖δ‖≤xtol(‖x‖+xtol)，**先检后评价** | ‖step‖≤xtol(xtol+‖x_old‖)，接受步后检 | MINPACK 相对步长条件 |
| ftol 语义 | 接受步相对下降 ≤1e-14·cost | 接受步 0≤Δloss≤1e-12·loss | MINPACK 平方和相对变化 |
| 预算 | _MAX_ITER=120，λ∈[1e-12,1e10] | max_forward/jacobian=500，max_iter=2000，λ同界 | max_nfev（默认 200·n） |
| 评价缓存 | 无 | _Evaluation 单点缓存（含 J），预算计数真实 | 内部 nfev/njev |
| 阻尼 | Marquardt diag(JᵀJ)+Nielsen，增广最小二乘 | 同一 `_solve_damped` 与 Nielsen 更新 | MINPACK lmder 阻尼+隐式对角缩放，不读 x_scale |
| λ/ρ 可观测性 | 可经 instrumentation 全量重建 | 同左 | 内部 λ 不可读（公开 API） |

**关键语义陷阱（量化）**：benchmark 的 g 经 1/M 缩放，gtol=1e-10(L∞)
对应 raw 域 ‖g‖∞≲1.6e-6，而主线停在 ‖g‖₂≤1e-12（raw c 坐标）——
两套量纲下有效阈值差约 6 个数量级，这正是其早停 1 步的直接原因；
旧"严格失败"行（如 lm 0/6 irms=1.20e-10）实际是**同一极小点、
少一步精度**，不是不同终点也不是恢复失败。

### λ/ρ 轨迹（统一口径主线记录，benchmark 逐位相同）

全部 6 场景：λ 自 1e-3 起按 Nielsen 因子 ~1/3 单调下降，
拒绝步 0 次；末次阻尼步仅用于 g/x 容差预检。
即：在载频问题良态曲率下 LM **实际运行于 GN 主导区**，阻尼只在前
几步提供保守步长；λ→0 说明 GN 曲率本身在起作用，不能据此推断
LM 的全局化机制有多大贡献（本数据上其拒绝路径未被触发）。

## 2. Part 2：端点/轨迹点曲率诊断（20 场景）

场景选择与结果全量：`part2_curvature_diagnostics.json`、
`part2_points.csv`。每场景 ≤3 个代表点（init/mid/final）。

### 2.1 谱画像（全部场景一致）

- σ(J)：最大 ~0.74，最小 ~0.142，**cond(J)≈5.2**，与场景无关
  （truth_rms 0.01–1.0、任何迭代点）——载频雅可比天然良态，
  与 `zero_jacobian_max=0.043`（c=0 处 J 非零，区别于无调制退化）
  一致。
- 精确 H：λ_min≈0.020、λ_max≈0.55，cond(H)≈27，
  **全部 24 个采样点严格正定**（含噪端点亦正定；无负曲率证据）。
- 残差二阶项 ‖Σrᵢ∇²rᵢ‖/‖JᵀJ‖：随 ‖r‖ 线性收缩——
  大残差初值点最高 ~0.30（loss~8e-3），momentum 终点 ~1e-4–1e-3，
  收敛端点 ~1e-11 以下。GN 近似在端点邻域极好；二阶项只在
  高残差过渡区可测。

### 2.2 端点分类（证据标签，非判定）

| 类别 | 场景 | 证据 | 标签 |
|---|---|---|---|
| 梯度容差停 + 严格失败（边界） | lm/gn/trf (0,6)(0,7)(4,5)(7,0)(17,4)(17,1) | g_inf 3e-11–9e-11 ≤gtol；H 正定；扰动 72 次 0 命中；沿 −g/最小特征向量方向 loss 上升 | stopped_gradient + local_min_candidate（与 truth 距 ~1e-10，**同极小点少一步**，非独立终点） |
| momentum 预算耗尽 | (0,8)(1,5)(1,2)(0,7)(0,0)(16,1) | g_inf 2e-4–8e-3 ≫gtol；随机扰动命中 8–28/72；−g 方向全半径下降；trust-exact 2–3 步直达 truth（wfe~1e-14） | **descent_found（有证据的停机不足）**；非局部点、非符号吸引子 |
| gd/adam 预算端点 | gd(0,8) g=1.6e-5 有下降；adam(0,8) g=3.8e-8、扰动无下降 | gd=停机不足；adam=近驻点但未达 gtol、邻域无下降证据（loss 1.25e-14, wfe 7.4e-7） | gd: descent_found；adam: 证据不足（介于停机不足与驻点之间） |
| LM/GN/TRF/BFGS 严格通过 | 其余场景 | 同上，g_inf≤~1e-15–5e-12 | stopped_gradient + local_min_candidate（= truth 极小点） |

负号对齐说明：旧数据 6 行 momentum `truth_sign=-1` 仅表示当前点
离 −truth 更近；端点有明确下降方向且 trust-exact 全部回到 +truth
极小点——证实这些是**路径中途的预算截断点**，不能称符号解收敛。

### 2.3 阻尼/接受率

- LM（重跑 instrumented，同预算 500）：全部场景 λ 单调 ×1/3、
  0 拒绝（同 Part 1）；incumbent 改善次数=接受步数。
- GN：Armijo 回溯使每轮迭代多耗评价（events/improvements≈7/3 等），
  与 LM 同达梯度容差。
- momentum/gd/adam：250 接受步 / 500 前向评价（预算恒耗尽），
  无容差停机路径（momentum/adam 设计上跳过 ftol）。

## 3. Part 3：小噪声二阶项 + trust-exact（分表，非噪声 MC）

### 3.1 小高斯噪声下的残差二阶项（1 场景=case0，σ=max/10^(SNR/20)）

| SNR | 真值点 S/GN | 真值点 g_inf | 拟合端点 S/GN | 端点 h_min | 端点 wfe |
|---|---|---|---|---|---|
| ∞（无噪） | 6e-18 | 1e-18 | 5e-14 | 0.0202 | 2.7e-12 |
| 60 dB（2 seeds） | 2.0e-4 / 2.6e-4 | ~1e-5 | ~2e-4 | 0.0201 | ~1.3e-4 |
| 40 dB（2 seeds） | 2.0e-3 / 2.6e-3 | ~1e-4 | ~2e-3 | 0.0201 | ~1.3e-3 |

含噪时真值点不再是极小点（g≠0），残差二阶项相对量级按噪声幅度
线性放大但仍 ≪JᵀJ（40 dB 下 ~2e-3）；端点仍正定，极小点偏移
≈噪声投影量级（wfe~σ_波前）。GN 近似在含噪小残差区仍成立；
此表仅为机制标定，不构成噪声性能结论。

### 3.2 trust-exact（精确 Hessian 信赖域，≤6 场景，诊断用途）

从 6 个 momentum 停滞端点启动（含 sign=-1 两行）：
全部 2–3 步内收敛到 +truth 极小点（end wfe 1e-14–1e-15，
min_eig≈0.020，μ→0 即纯 Newton 步）。

解释边界：这证明**端点邻域不是局部极小、精确曲率给出立即可行
下降**；不构成该点"非局部极小"的全局证明，也不说明 trust-exact
作为求解器优于 LM（其每步成本与 LM 不同，未作效率比较）。
本轮未观察到负曲率或鞍点结构，trust-exact 的负曲率路径未被触发。

## 4. 结论：假设核对

**支持（有数值证据）**
- H1 两自研 LM 数学等价：统一口径下 λ/ρ/迭代点逐位一致；
  差异只在归一化单位、容差绝对量纲与停机检查点位置。
- H2 "严格失败"行=阈值边缘现象：gtol 停在同极小点少一步处，
  irms 距门槛仅 ~20%；非不同终点、非恢复失败。
- H3 momentum 预算耗尽=停机不足：扰动/−g 方向有下降、
  trust-exact 即达；负号对齐行同理（路径中途点）。
- H4 载频问题良态：cond(J)≈5.2、H 恒正定（40 采样点），
  LM 的 λ→0 无拒绝——GN 曲率主导，全局化机制未被动用。
- H5 残差二阶项小残差律：‖ΣrH‖/‖JᵀJ‖ 随 ‖r‖ 线性
  （0.3→1e-3→1e-11；含噪 40dB 升至 ~2e-3）。

**不支持（证据否定）**
- 端点存在负曲率/鞍点解释 momentum 停滞：40 点全正定，否定。
- LM 优势主要来自全局化/拒绝路径：本数据上拒绝路径未触发，否定。
- −truth 是独立收敛吸引子（对本采样范围）：所有负号对齐端点
  均有下降且可达 +truth，否定（不推及未采样区域）。

**证据不足**
- adam 端点 (0,8)：g=3.8e-8 高于 gtol 但邻域无下降——"停机不足"
  与"近驻点"两解释并存；需按评价次数细粒度轨迹再判。
- 二阶项在高残差/大初值偏差区的结构（本轮只覆盖 ‖r‖≤~0.1 区段）。
- SciPy lm 的 MINPACK 内部阻尼路径不可观测，无法逐机制对齐。

## 5. 下一轮需要的日志/接口

1. LM 类方法在 `_local` 记录逐步 (λ, ρ, accepted)——本轮已验证
   可通过 `_solve_damped` spy 重建，建议 T2 在 history 增事件类型
   而非靠 instrumentation。
2. 评价事件区分"接受点/拒绝 trial/缓存命中"，便于接受率统计
   （目前需靠 q 相等性推断）。
3. gtol 建议同时记录 raw 与 scaled 两种量纲的 ‖g‖，避免跨实现
   比较时的量纲陷阱（本轮已量化：差 ~4 个数量级有效阈值）。
4. 停机语义统一化需求：主线"先检后评价"与 benchmark"先评价后检"
   差一步评价，建议在 T1/T2 整合时明确单一约定。
5. 若后续要诊断鞍点/逃逸：需要 (i) 更大初值偏差场景 (ii) H 谱
   轨迹日志；trust-exact 工具可复用（`diagnostics.trust_exact`）。

## 6. 复现与产物

- 代码：`analysis/carrier_curvature/`（common/hessian/instrument/
  diagnostics/run_part1/run_part2/test_derivatives）。
- 运行：`python analysis/carrier_curvature/test_derivatives.py`、
  `run_part1.py`、`run_part2.py`（全程 <5 min，BLAS 线程=1）。
- 产物根：`/home/user/research-artifacts/chessboard-lsi/carrier-research-20261009/T5/`
  （part1_lm_comparison.json、part1_lambda_rho.csv、
  part2_curvature_diagnostics.json、part2_points.csv、
  part3_noise_second_order.json、part3_trust_exact.json、
  SHA256SUMS.txt）。
- 接口说明：本任务只新增 analysis/ 与 reports/；只读调用 lsi.*；
  未改公共源码、旧数据与他人分支。`problem_curvature` 依赖
  `problem.cache/mods/blocks/scale/normalizer` 现有结构，
  与 T1 的 `value_gradient`/`frame_intensity_jtr` 接口兼容方向一致，
  但当前实现独立装配，整合时建议迁到 T1 的问题接口上。
