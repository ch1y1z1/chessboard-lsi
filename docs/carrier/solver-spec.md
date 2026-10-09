# 求解器规范（T2）：单帧载频反演的九法主比较

`lsi/optimizers.py` 在同一观测目标 `F(q)=½‖r(q)‖²`（`r` 为 M-归一化
光强残差，`q_j = scale_j·c_j` 为 RMS 参数坐标）上提供九个主比较方法：

    gd, nesterov, adam, nlcg, bfgs, lbfgs, gn, lm, trf

`PRIMARY_METHODS` 即该九法。`momentum`（固定步长带动量）只保留为旧配置
兼容/诊断用途（`LEGACY_METHODS`），不进入主比较。

所有方法 float64，共享观测与初值，按观测损失返回已评价过的最佳点
（incumbent）；任何线搜索试探、拒绝步都计入真实评价轨迹，不冒充已接受
迭代。停止状态词汇见文末。

## 统一评价器 `_Evaluation`（唯一模型接入点）

实际计算计数（缓存命中不计数也不写轨迹）：

| 计数器 | 含义 |
|---|---|
| `n_forward` | 每次真实光强计算（显式 `evaluate` 或 native 融合通道） |
| `n_jacobian` | 每次 Jacobian 级导数计算：显式 J 生成，或 native 融合 Jᵀr |
| `n_gradient` | 每次交付的梯度向量（显式 Jᵀr 乘积或融合 Jᵀr） |
| `n_fused_gradient` | 其中经 native 融合通道交付的次数 |

请求计数 `n_function_requests` / `n_gradient_requests` /
`n_jacobian_requests` 与实际计算分列。`max_forward` 与 `max_jacobian`
作用于实际计算计数；`max_iter` 是方法自身迭代数（SciPy 法按库返回）。

轨迹每个事件含 `request`（function/gradient/jacobian——求解器要什么）、
`kind`（residual/jacobian/native_gradient——实际算什么）、`backend`、
计数快照、`loss`、`best_loss`、`gradient_inf`、`q`、可选算法注释
（`alpha`/`lam`/`rho`/`armijo_accepted`/`accepted`/`restart`）。
`n_forward` 对非有限评价也计数（计算确实被尝试）；`history` 只记录
成功提交的有限评价。

## 缓存与升级

缓存按量分层（`r / j / loss / g` 各自独立）：

- 同点已有量直接复用，不计数；
- 残差点请求 J、或显式 J 点请求梯度：缺失量触发一次新真实计算
  （升级如实计入 forward/J 预算）；
- native 融合通道只返回 `(loss, g)`：它不产出残差向量，之后的
  `residual`/`jacobian` 请求必须做额外显式计算并如实计数——
  不会错误命中不存在的残差缓存；
- 反向地，一次融合 Jᵀr 内部是同量级的 dE 累积（Jacobian 级工作量），
  计入 `n_jacobian`/`max_jacobian`，保证两后端预算可比；
- 仅存 `loss/g` 的 native 结果不会被当作残差/雅可比缓存。

## native 梯度契约（T1/T3 接口）

问题对象若声明 `supports_native_gradient=True` 并实现
`value_gradient(q) -> (loss, g_q)`（T1 包装 T3 的 `frame_intensity_jtr`）：

- gd / nesterov / adam / nlcg / bfgs / lbfgs 走该融合通道；
- gn / lm / trf 始终用显式 `problem.evaluate(q, jacobian=True)`；
- 无能力标识的旧问题对象保持显式 J 路径，不冒称 native，
  J 成本照常计入；
- `gradient_backend` 元数据透传到 `OptimizationResult.gradient_backend`
  及每个轨迹事件；
- 只提供 `value_gradient` 的问题：残差请求升级为一次 J 级融合评价
  （如实计数）；`residual`/`jacobian` 显式请求返回
  `unsupported_problem` 状态而非伪造数据。

## 各方法与超参

共享容差（`q` 坐标下）：`gtol` 是 `‖g‖∞` 梯度停止；`xtol` 是相对步长
停止；`ftol` 是单步损失相对下降停止（仅对有下降保障的方法即时生效）。
预算：`max_iter` / `max_forward` / `max_jacobian` / `max_seconds`。

| 方法 | 实现 | 独立超参（默认值，来源） |
|---|---|---|
| `gd` | 负梯度 + Armijo 回溯；首步 `learning_rate`，其后 **BB1** 谱步长 α=sᵀy/yᵀy；`sᵀy≤0`（非凸）或 yy≤0 时退回上一接受步长保证 α>0；`gd_step="constant"` 退化为固定步长 | `learning_rate`（0.01，旧配置值）、`gd_step`（"bb1"，Barzilai–Borwein 1988）、`armijo`（1e-4，常用值）、`max_backtracks`（30）、`step_grow`（2.0，见注） |
| `nesterov` | FISTA 式加速：前瞻点 `y` 处取负梯度步（Armijo 保障相对 `y` 充分下降），动量参数 `t` 递推；`nesterov_restart`：`"function"`（O'Donoghue–Candès 2015 自适应重启，`f(x⁺)>f(x)` 重置 t=1）/`"gradient"`（`g·Δx>0`）/`"none"`；接受步长按 `step_grow` 放大试探 | `learning_rate`（初始步长）、`nesterov_restart`（"function"）、`step_grow`（2.0）、`patience`（100，平台期早停） |
| `adam` | 偏差修正 Adam（Kingma–Ba 2015），固定 `learning_rate`；`patience` 内 best_loss 无 `ftol` 相对改进则 `plateau_tolerance` 早停（可审计，不再默认跑满预算） | `learning_rate`（0.01，旧配置值；论文常用 1e-3）、`beta1`/`beta2`/`epsilon`（0.9/0.999/1e-8，论文默认）、`patience`（100，本工程启发值） |
| `nlcg` | PR+（Polak–Ribière⁺，`β=max(0, gᵀ(g-g⁻)/g⁻ᵀg⁻)`，β=0 自动回最速下降）；`gᵀd≥0` 或非有限方向显式重启为 `-g`；Armijo 线搜索失败再以 `-g` 兜底一次 | `learning_rate`（首步试探）、`step_grow`、`armijo`、`max_backtracks` |
| `bfgs` | `scipy.optimize.minimize(method="BFGS")` + 解析梯度 | `gtol`/`maxiter` 直接映射 SciPy 选项 |
| `lbfgs` | `scipy` `L-BFGS-B`，**`bounds=None`**：与主比较同一无约束问题，不把边界差异混为算法收益 | `lbfgs_memory`=maxcor（10，SciPy 默认）、`ftol`/`gtol`/`maxiter`/`maxfun=max_forward` 映射 |
| `gn` | 阻尼 GN：`lstsq(J,-r)` 方向，先全步长，Armijo 回溯；`gᵀd≥0` 报 `non_descent_direction` | 仅共享容差/线搜索参数 |
| `lm` | Marquardt 阻尼 + Nielsen λ 更新，增广最小二乘解阻尼步（与旧实现同一数学目标）；轨迹注释 `lam`/`rho`/`accepted` | `lambda0`（1e-3，旧实现沿用值） |
| `trf` | `scipy.optimize.least_squares(method="trf")`，`x_scale=1.0` | `gtol`/`ftol`/`xtol`/`max_nfev=max_forward` 映射 |
| `momentum` | 旧固定步长带动量（diagnostic-only） | `learning_rate`、`momentum`（0.9，旧配置值）、`patience` |

`step_grow`（默认 2.0）只作用于"接受步后的下一步试探初值"，是常见的
回溯线搜索增长惯例（每次迭代至多放大一倍），不改变接受判据。
`patience`（默认 100 迭代，`None` 禁用）是 best_loss 的平台期窗口，
阈值 `ftol` 相对改进；100 取"远大于振荡周期、远小于默认预算"的
启发值，非调优结果。`_ALPHA_MIN/MAX=(1e-16,1e12)` 仅为排除数值病态的
试探步裁剪。

## 停止状态

`gradient_tolerance`（‖g‖∞≤gtol）、`step_tolerance`（相对步长≤xtol）、
`loss_tolerance`（单步相对下降≤ftol）、`plateau_tolerance`（平台期
窗口耗尽）、`loss_and_step_tolerance`（trf）、`iteration_budget`、
`forward_budget`、`jacobian_budget`、`time_budget`、
`line_search_failed`（Armijo 耗尽回溯）、`non_descent_direction`、
`damping_limit`（λ≥1e10）、`nonfinite_parameters`、`nonfinite_evaluation`、
`precision_loss`（scipy BFGS）、`scipy_failure`、`unsupported_problem`。
`stopped_by_tolerance = status.endswith("tolerance")`；停止≠恢复成功。

## 已知限制

- 一阶线搜索为 Armijo（非强 Wolfe）；nlcg 的 PR+ 在非精确线搜索下
  依赖其内置重启与非下降回退保证。
- 未实现精确 Hessian 全规模法；GN/LM 用 JᵀJ 高斯-牛顿曲率，λ→0 即
  GN 步，不构成局部极小性证明。
- SciPy 方法（bfgs/lbfgs/trf）内部线性代数与试探不经评价器，预算经
  统一计数强制；其 `n_iter` 为库返回值（trf 不报告）。
- 并发环境下 `elapsed_seconds` 仅记录不作性能比较；数值效率结论须以
  forward/J/gradient 计数为准。
- `n_forward` 计真实计算尝试（含随后判为非有限的评价），`history`
  仅含成功提交的有限评价点。

## 验证

`tests/test_carrier_optimizers.py`：良态二次与 Rosenbrock 非线性 toy 的
解析恢复、方向导数、请求/计算计数、缓存升级不变量、预算耗尽 incumbent
一致性、native 融合通道契约（含残差不冒名、显式 J 法绕行、
unsupported_problem）、九法同初值同目标的小型载频 fixture。均为正确性
验证，不构成性能排名。
