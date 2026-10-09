# 单帧 J^T r 直算接口（T3 交付）

`lsi/gradient.py` 提供 `frame_intensity_jtr`，在不物化 M×p 雅可比的情况下，
与 `lm._frame_and_jacobian` 的显式路径逐像素同数学地返回模型光强与
`J^T (I - target)`。供 T1 在 `IntensityProblem` 上暴露 native value/gradient
接口，供 T2 的一阶/拟牛顿方法走非物化梯度路径；GN/LM/TRF 仍需要显式 J。

## 接口契约（T1 接入用）

```python
frame_intensity_jtr(cache, coeffs, target, modulation=None)
    -> (intensity, jtr_coeff)
```

| 参数 | 形状 | 说明 |
|------|------|------|
| `cache` | list of `(k, amp, inside, Z)` | `ForwardModel.zernike_samples(indices, rows)` 的产物；`inside` 为 (M,) 布尔，`Z` 为 (p, M) float64 |
| `coeffs` | (p,) float64 | 原始 Zernike 系数（waves），与 `indices` 对齐 |
| `target` | (M,) float64 | 采样像素上的观测光强 |
| `modulation` | None / (K,) / (K, M) | 每级附加相位：标量（光栅相移）或已按 rows 采样的空间相位（载频 ramp），与 `_frame_and_jacobian` 逐帧调制形式一致 |

返回：

* `intensity` — (M,) float64，与显式路径逐位一致（相同运算相同顺序）；
* `jtr_coeff` — (p,) float64，**原始系数坐标**下的 `J^T (I-target)`，
  **不按 M 归一化，不含 q-坐标链式因子**。

坐标/归一化换算（由 T1 统一执行，本接口不做）：

* `IntensityProblem` 定义 `F(q) = ½ mean(r²)`、`q_j = scale_j · c_j`，
  故 `g_q = jtr_coeff / (M · scale)`（逐元素除），`loss = r·r/(2M)`。
* 多帧 `StackedIntensityProblem`：对每帧调用本接口后求和，
  `g_q = Σ_frames jtr_frame / (M_total · scale)`，M_total 为全部帧的
  有效像素总数；每帧可携带各自 modulation。

预算语义建议（T2 执行）：一次 `frame_intensity_jtr` = 一次前向评价 +
一次梯度；不消耗 jacobian 预算。显式 J 回退路径必须照常计
`n_jacobian`，且不得冒称 native 梯度。

fixture：`carrier-research-20261009/T3/fixture.npz`（由
`analysis/carrier_gradient_check.py --output <dir>` 生成）包含
n=64 载频、12 参数、固定坏点采样下的 `cache` 分量
（`inside`/`basis`/`amplitudes`/`order_list`/`rows`/`modulation`）、
`coeffs`、`target` 与期望输出 `expected_intensity`/`expected_jtr`/
`explicit_jacobian`，可用于 T1 整合的独立回归。

## 推导

```
E_m   = Σ_k e_{k,m},  e_k = inside_k · A_k · exp(i φ_k),
φ_k   = 2π (c @ Z_k) + mod_k,  I = |E|²,  r = I - target

(J^T r)_j = Σ_m r_m · 2 Re(conj(E_m) · Σ_k 2πi e_{k,m} Z_k[j,m])
         = -4π Σ_k ( Z_k @ w_k^im )_j,
         w_k^im = r · Im(conj(E) e_k) = r·(E_re·e_k,im − E_im·e_k,re)
```

实现分两遍：第一遍逐光束算 `e_k` 并累加 `E`，物化 `e_k` 列表供第二遍
复用（避免重复 `coeffs @ Z` 与 `exp`）；第二遍逐光束
`acc += Z_k @ w_k^im`（实 dgemv），`jtr = -4π·acc`。

注意：`Z @ 复向量` 会使 NumPy 把 (p,M) 的 Z 隐式复制为 complex128
交给 zgemv，产生 p·M·16 字节隐藏临时（p=35、M=65536 时约 37 MB），
恰好抵消不物化 J 的意义；因此第二遍只取虚部做纯实数 matvec。

## 验证（`tests/test_carrier_gradient.py` + `analysis/carrier_gradient_check.py`）

对照：`_frame_and_jacobian` 显式 J，再算 `J.T @ (I - target)`。

覆盖矩阵：n∈{64,128,256}、p∈{12,20,35}（indices=2..p+1）、
采样∈{全图、固定坏点剔除、黄金比 4096 子采样}、
调制∈{无、每级标量相移、空间载频}、
系数 RMS∈{0.1…3.0} waves、残差∈{近零（model+1e-6）、大且有偏}。

结果（float64，最坏情形）：

* `intensity` 与显式路径**逐位一致**（max |ΔI| = 0）；
* `jtr` 最大相对差 **1.24e-14**（分母 max(1, ‖jtr_ref‖_∞)），
  全部情形 ≤1e-14 量级——差异仅来自求和顺序，符合舍入预期；
* 方向差分 `|jtr·v − (L(c+hv)−L(c−hv))/2h| ≤ 1e-6·scale`，h=1e-7·(1+‖c‖)；
* r=0 时 ‖jtr‖∞ ≤ 1e-12·max(1, ‖J‖∞)，即浮点噪声水平。

## 内存

理论持久峰值 ≈ `(64 + 16K)·M + O(p)` 字节（E 16M、e_k 列表 16K·M、
I/r 各 8M、逐光束临时约 48M/24M），对照显式路径额外物化
dE(16pM)+J(8pM)=24pM 字节；p=35、M=65536 时直算约 10 MB、
显式约多 55 MB。

实测（NumPy≥1.23 的缓冲区经 tracemalloc domain 追踪，ru_maxrss 增量
因分配器缓存不可靠仅作参考）：M=65512、p=35、K=5 时 tracemalloc
峰值 9.5 MB，与理论计数一致；最初复数 matvec 版本因 Z 隐式复制
实测 44 MB，已消除。

`tests/` 为回归断言；`analysis/carrier_gradient_check.py` 输出
完整矩阵到 `verification.json`（T3 产物目录）。

## 计时（仅诊断，非结论）

`analysis/carrier_gradient_check.py --timing N` 在同机对
`_frame_and_jacobian`+`J.T@r` 与本接口做 N 次重复计时，BLAS 线程
固定为 1。本轮在 5 任务并行的共享机器上运行，**数字不进性能结论**；
正式标定需独占条件。当前诊断值（n=256/p=35/全图/载频，7 次）：
显式 min≈156 ms、直算 min≈37 ms——方向一致但只是 sanity 证据，
Flop 量级两者相当，差异主要来自 J 的物化与回读。

## 不做的事

* 不改 `lm.py`/`problem.py`/`optimizers.py`，显式 J 路径保留为对照；
* 不做归一化、q-坐标变换或跨帧求和（T1 职责）；
* 不承诺 flop/耗时优势；未做独占计时标定。
