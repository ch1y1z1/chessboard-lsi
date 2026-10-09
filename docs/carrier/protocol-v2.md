# carrier-protocol-v2：单帧载频反演协议与接口契约

日期：2026-10-09，T1（协议 / problem / harness 接口 owner）。配套机器可读
草案：`configs/carrier-research/protocol-v2.json`；可运行配置与小型
fixture：`configs/carrier-research/{dev24,main100,smoke}.json`。

适用范围：单帧、已知空间载频（`SystemConfig.carrier_f0`，不独立扫 f0/s）、
原始光强直接反演。真值只出现在独立评价层（`lsi/metrics.py` 与报告），不
参与初始化、选解、停止与超参调试。额外相移帧与无载频结果不进入主结论。

## 1. 评价层：符号策略与误差域（`lsi/metrics.py`）

```python
recovery_metrics(coeffs, truth, indices, pupil_basis, shear, *, sign="free")
```

- `sign="free"`（默认，旧协议）：允许一个整体 ± 变号取最优。无载频观测
  下 `I(-c)=I(c)`，符号物理不可观测，必须保留。
- `sign="fixed"`（载频协议）：符号固定为 `+1`。载频观测打破 ± 简并，
  符号翻转是物理错误；`truth_sign` 恒为 1。
- 显式 `sign=±1` 固定为给定符号。

输出三个互不替代的域（均为 runs.csv 列，向后兼容旧列名）：

| 域 | 列 | 语义 |
| --- | --- | --- |
| raw（主指标） | `wavefront_rms_error` | 光瞳上差分波前的采样 std：**只去 piston**（减去采样均值，量记录在 `piston_removed`），**保留 tilt** |
| | `max_coeff_error`, `truth_sign` | 逐系数最大误差；所取符号 |
| alias（倾斜周期等价域） | `equivalent_wavefront_rms_error`, `equivalent_max_coeff_error`, `equivalent_truth_sign`, `tilt_alias` | 允许 Z2/Z3 按 `1/shear` 整数周期平移后取最优；只作诊断。别名解不能被默默排除，也不计入 raw 成功 |
| pm（历史兼容） | `historical_coeff_error_pm` | 始终按自由 ± 变号计算，只用于与旧输出对照 |
| | `sign_policy` | 行级记录所用策略（"free"/"fixed"） |

研究工程门槛为 1e-3 waves（`wavefront_tolerance`，`engineering_success`
列）；`success_wf_1e-02/1e-03/1e-05/1e-07` 列给出研究精度曲线，不构成应用
性能承诺。`equivalent_engineering_success` 为别名域诊断列，不并入成功判定。

## 2. problem 接口（`lsi/problem.py`）

`IntensityProblem` / `StackedIntensityProblem` 保持原有 `evaluate` 签名；
新增：

```python
loss, g_q = problem.value_gradient(q)
```

- `loss = 0.5 * mean((I - I_obs)^2)`，对所有有效像素/帧以 `M = normalizer**2`
  归一化；与 `evaluate` 返回的 `r@r/2` 数值一致（测试锁定）。
- `g_q = jtr_c / (M * scale)`，即按 `q_j = scale_j * c_j` 链式变换后的
  归一化梯度坐标。

能力标识（实例属性，可 JSON 序列化进 metadata）：

- `problem.supports_native_gradient`：bool。
- `problem.gradient_backend`：`{"name", "function", "explicit_jacobian"}`。

**给 T2 的成本记账约定**：`value_gradient` 的 native 路径记 1 次前向 +
1 次梯度（不产生显式 J）；`explicit_jacobian_fallback` 路径一次调用内部
含完整雅可比计算，调用方须记 1 次前向 + 1 次 J，不得冒称 native。
`gradient_backend["explicit_jacobian"]` 为 True 时即此情形。GN/LM/TRF
继续走显式 `evaluate(jacobian=True)`；旧问题对象行为不变。

**给 T3 的接口契约**（尚未整合时自动回退，见 `lsi/problem.py` 顶部
try-import）：

```python
# lsi/gradient.py
def frame_intensity_jtr(cache, coeffs, target, modulation=None):
    """cache: ForwardModel.zernike_samples 的产物（采样像素版）；
    coeffs: 原始未归一化 Zernike 系数（waves），顺序对应 indices；
    target: 已在 cache 像素上采样的观测强度向量；
    modulation: None / (n_orders,) 标量相移 / (n_orders, n_rows) 已按
        rows 采样的空间相位（与 _frame_and_jacobian 的约定一致）；
    返回 (intensity, jtr_coeff)：一帧模型光强与 (dI/dc)^T (I-target)，
    全部为原始、未归一化 coeff 坐标。"""
```

`lsi/problem.py` 在模块导入时探测该函数；存在则 `gradient_backend.name =
"native"`，否则 `"explicit_jacobian_fallback"` 并如实标记
`explicit_jacobian=True`。T3 只需新增 `lsi/gradient.py` 与自身测试。

## 3. 初值类（`benchmark_modulated.py`）

`ModulatedConfig.initial_classes`（非 None 时取代旧 `include_zero` +
`initial_rms` 池）支持三类，可组合：

| 类 | 语义 |
| --- | --- |
| `zero` | 全零初值（退化诊断点；汇总时不并入主要恢复率） |
| `random` | 独立于 truth 的共享随机方向，`initial_random_rms`（默认 0.1 wave），单初值，全部 case/方法共用同一向量 |
| `fourier` | 同帧载频解调初值（`lsi.invert.fourier_to_wavefront`），逐 case 由观测帧生成；**只读观测帧**，失败退化为零初值并在 `initial_provenance` 列记录 `fourier_failed:<ExcType>` |

所有初值路径不读取 truth；`metadata.initials_read_truth=false`。行级列：
`initial_class`、`initial_provenance`、`initial_rms`（fourier 类记录该
case 实际初值的波前 RMS）。

## 4. 噪声链（`NoiseConfig`）

`ModulatedConfig.noise` 为 `NoiseConfig` 参数字典；每 case 观测由全部
方法/初值共享。单位语义：

- 前向光强 `I = |E|^2` 为无量纲线性强度单位（未标定物理通量）。
- `model="gaussian"`：`I_obs = I + N(0, σ)`；`snr_db` 采用振幅约定
  `σ = max(I_clean) · 10^(-snr_db/20)`（与旧基准一致），或直接给强度单位
  `sigma`。二者只能给一。
- `model="camera"`（电子域）：`λ = clip(I,0)·peak_signal_e/max(I_clean)`
  为期望电子数，`e ~ Poisson(λ) + N(0, readout_sigma_e)`；可选满阱截断
  `full_well_e`、ADU 量化（`gain_e_per_adu` 与 `bit_depth` 须同时给出），
  最后按同一标度折算回强度单位。**不提供任何默认实测相机参数**；
  `peak_signal_e` 必须显式指定。
- `negative`: `"clip"`（默认，负值截 0，物理但有偏）/ `"keep"`（保留负值，
  零均值但允许非物理强度）。含噪 case 的 `strict_success` 留空。
- `bad_pixel_fraction`：固定坏点比例，从拟合 mask 中排除（非插值填充）；
  `dataset.npz` 存 `masks`，`cases[*].n_valid_pixels` 记录有效像素数。

种子流（互不重叠）：`truth_seed` / `initial_seed` / `noise_seed` 独立；
噪声逐 case 派生 `[noise_seed, truth_id, repeat]`，通道 1 像素噪声、
通道 2 坏点位置。`initial_seed` 变化不影响观测帧（测试锁定）。

## 5. 冻结研究设计（`configs/carrier-research/`）

| 配置 | 用途 | truth 数 |
| --- | --- | --- |
| `dev24-{dense,sparse}.json` | 验证集：调参与方法开发 | 3 RMS (0.03/0.3/1) × 2 kind × 4 = 24 |
| `main100-{dense,sparse}.json` | 盲测主测试，冻结后运行 | 5 RMS (0.01/0.03/0.1/0.3/1) × 2 kind × 10 = 100 |
| `smoke.json` | 接口 fixture：1 truth × 3 初值类，含 40 dB 高斯噪声与小预算 | 1 |

种子流见 `protocol-v2.json` 的 `seed_streams`（dev/main/dense/sparse/
initial/noise/tuning 互不重叠）。待标定字段（不编造数值）：`max_seconds`
绝对时间预算、`max_gradient`、各方法原生停止条件的公共数值终点标定。
统计口径：独立 truth 是统计单位；噪声/初值为组内重复；方法比较用配对
差值/比值 CI，不以两组 CI 重叠判显著。

## 6. 运行 schema（runs.csv / metadata.json / trajectories.jsonl.gz）

`metadata.json` 新增：`evaluation_version`、`sign_policy`、`noise`
（NoiseConfig 全量）、`gradient_backend`、`initials_read_truth`、
`seed_streams`、`schema`（阶段/事件/状态枚举与失败策略）。
`schema.phases = ["precompute","init","solve","restart"]`；
`schema.cache_states = ["cold"]`（当前每次 solve 冷启动计数，共享
precompute 见 `precompute_seconds`）；`schema.trajectory_events` 当前为
`["evaluation"]`（扩展事件类型由 T2 预算接口定义，须向后兼容）。

runs.csv 新增列：`noise_model`、`initial_class`、`initial_provenance`、
`sign_policy`、`evaluation_version`、`cache_state`、`piston_removed`、
`equivalent_engineering_success`、`success_wf_*`、`error`。
失败与超时：求解器异常落盘为 `status="solver_exception"`、`error` 记
异常描述、指标列 NaN、`loss=inf`（不参与 incumbent 选择）；预算/超时
由 `status` 列（`forward_budget`/`jacobian_budget`/`time_budget` 等）记录。
源溯源：`revision` + `working_tree_status` + `source_sha256` 快照联合
使用，不凭单一 revision 推定源码。

兼容性：旧 `ModulatedConfig` JSON 无新增字段也可运行（`sign_policy` 默认
auto：carrier→fixed，phase_shift→free；噪声 none）。旧列名与取值语义
不变，新增列均为附加列。

## 7. 交接 fixture

- `configs/carrier-research/smoke.json`：`python benchmark_modulated.py
  --config configs/carrier-research/smoke.json --output <新目录>`，
  约秒级，覆盖三类初值、高斯噪声、固定符号评价全链路。
- `tests/test_carrier_protocol.py`：符号策略、tilt 域、归一化梯度方向
  差分、后端标识、噪声语义、种子流独立性、旧路径兼容、非法参数。
- T2 接入点：`problem.value_gradient` + `gradient_backend`（第 2 节）；
  T3：`frame_intensity_jtr` 签名（第 2 节）；T5：曲率分析直接复用
  `problem.cache`/`pupil_basis` 与本文件坐标约定。
