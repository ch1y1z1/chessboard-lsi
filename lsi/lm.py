"""Levenberg–Marquardt 对非线性前向模型的直接反演。

前向模型 I = |sum_ab A_ab exp(i[2 pi W(x+as, y+bs) + delta_ab])|^2 对
波前 Zernike 系数 c 是非线性的。不做干涉图线性化（相移/傅里叶路线），
而是直接最小化光强残差：

    min_c  || I_meas - I_model(c) ||^2

LM 迭代（Marquardt 阻尼 + Nielsen 更新）：

    (J^T J + lambda diag(J^T J)) delta = -J^T f,   c <- c + delta
    rho = (F(c) - F(c + delta)) / delta^T (lambda D delta - J^T f)

其中 F(c) = ||f(c)||^2（不是 1/2||f||^2），分子分母因子一致。

阻尼步用增广最小二乘求解而不是显式法方程，避免 cond(J)^2 的数值损失：

    [ J           ] delta ~= [ -f ]
    [ sqrt(lam D) ]          [  0 ]

雅可比是解析的：dE/dc_j = sum_ab A_ab e^{i phi_ab} i 2 pi Z_j(x+as, y+bs)，
dI/dc_j = 2 Re{ E* dE/dc_j }，每次迭代一次级次叠加，无有限差分。

相比论文流程的优势：不解包裹、不做剪切区判定、无调制度变号问题、
单帧载频图即可反演；光强仿射参数（增益 + 背景）可用变量投影一并估计。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .forward import ForwardModel
from .zernike import check_indices

__all__ = [
    "LMConfig",
    "LMResult",
    "levenberg_marquardt",
    "fit_wavefront_from_frames",
    "fit_wavefront_from_carrier_frame",
    "multistart_fit",
]


@dataclass
class LMConfig:
    max_iter: int = 120
    lambda0: float = 1e-3        # 初始阻尼
    lambda_min: float = 1e-12
    lambda_max: float = 1e10
    nu0: float = 2.0             # 拒绝步时 lambda 的增长因子初值
    ftol: float = 1e-14          # 代价相对变化收敛阈
    xtol: float = 1e-12          # 步长收敛阈
    gtol: float = 1e-12          # 梯度范数收敛阈
    verbose: bool = False
    #: 在迭代内用变量投影估计仿射光强模型 alpha*I_model + beta
    fit_scale_background: bool = False


@dataclass
class LMResult:
    x: np.ndarray                # 拟合的 Zernike 系数
    cost: float                  # ||f||^2
    history: dict = field(default_factory=dict)
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    scale: float = 1.0           # 拟合的光强增益（变量投影）
    background: float = 0.0      # 拟合的背景
    cond: float = np.nan         # 解处雅可比的 2-范数条件数（可辨识性）
    rms_residual: float = 0.0


def _solve_damped(
    J: np.ndarray, f: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """增广最小二乘求阻尼步，返回 (delta, g = J^T f)。"""
    diag = np.clip(np.diag(J.T @ J), 1e-30, None)
    g = J.T @ f
    A = np.vstack([J, np.diag(np.sqrt(lam * diag))])
    delta = np.linalg.lstsq(A, np.concatenate([-f, np.zeros(J.shape[1])]), rcond=None)[0]
    return delta, g, diag


def levenberg_marquardt(
    residual_and_jac: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0: Sequence[float],
    config: LMConfig | None = None,
    *,
    scale_background: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float, float]
    ]
    | None = None,
) -> LMResult:
    """最小化 ||f(x)||^2；``residual_and_jac(x) -> (f, J)`` 用解析雅可比。

    ``scale_background`` 若给出，把原始 (f, J) 映射为变量投影后的
    (f, J, alpha, beta)（仿射光强模型的精简残差/雅可比）。
    """
    cfg = config or LMConfig()
    if cfg.fit_scale_background and scale_background is None:
        raise ValueError(
            "LMConfig(fit_scale_background=True) 需要 scale_background 回调"
        )
    x = np.asarray(x0, dtype=float).copy()

    def evaluate(p):
        f_p, J_p = residual_and_jac(p)
        if f_p.size == 0:
            raise ValueError("残差为空：没有可用观测")
        a_p, b_p = 1.0, 0.0
        if scale_background is not None:
            f_p, J_p, a_p, b_p = scale_background(f_p, J_p)
        return f_p, J_p, a_p, b_p

    f, J, alpha, beta = evaluate(x)
    cost = float(f @ f)
    lam, nu = cfg.lambda0, cfg.nu0
    history = {"cost": [], "lambda": [], "grad_norm": [], "rho": []}
    converged, msg, n_iter = False, "max_iter", 0

    for it in range(cfg.max_iter):
        n_iter = it + 1
        delta, g, diag = _solve_damped(J, f, lam)
        history["cost"].append(cost)
        history["lambda"].append(lam)
        history["grad_norm"].append(float(np.linalg.norm(g)))

        if np.linalg.norm(g) <= cfg.gtol:
            converged, msg = True, "gradient tolerance"
            break
        if np.linalg.norm(delta) <= cfg.xtol * (np.linalg.norm(x) + cfg.xtol):
            converged, msg = True, "step tolerance"
            break

        f_new, J_new, a_new, b_new = evaluate(x + delta)
        cost_new = float(f_new @ f_new)
        # Nielsen 增益比：实际下降 / 阻尼模型预测下降
        # cost = ||f||^2 约定下预测下降 = delta^T (lam D delta - g)
        predicted = float(delta @ (lam * diag * delta - g))
        rho = (cost - cost_new) / predicted if predicted > 0 else -1.0
        history["rho"].append(rho)

        if rho > 0.0:
            x, f, J, cost = x + delta, f_new, J_new, cost_new
            alpha, beta = a_new, b_new
            lam = float(np.clip(
                lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3),
                cfg.lambda_min, cfg.lambda_max,
            ))
            nu = cfg.nu0
            if abs(history["cost"][-1] - cost) <= cfg.ftol * max(cost, 1e-30):
                converged, msg = True, "cost tolerance"
                break
        else:
            lam = float(np.clip(lam * nu, cfg.lambda_min, cfg.lambda_max))
            nu *= 2.0
            if lam >= cfg.lambda_max:
                msg = "lambda overflow"
                break
        if cfg.verbose:
            print(f"  iter {n_iter:3d}  cost={cost:.6e}  lambda={lam:.2e}  rho={rho:+.3f}")

    return LMResult(
        x=x,
        cost=cost,
        history=history,
        n_iter=n_iter,
        converged=converged,
        message=msg,
        scale=float(alpha),
        background=float(beta),
        cond=float(np.linalg.cond(J)) if J.size else np.nan,
        rms_residual=float(np.sqrt(cost / max(f.size, 1))),
    )


# --------------------------------------------------------------------------- #
# 从光强帧拟合 Zernike 系数
# --------------------------------------------------------------------------- #
def _frame_and_jacobian(
    cache: list[tuple[int, complex, np.ndarray, np.ndarray]],
    coeffs: np.ndarray,
    deltas: np.ndarray | None,
    carriers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """一帧的模型光强 I(c) 与雅可比 dI/dc（在采样像素上）。

    E = sum_k A_k e^{i phi_k},  phi_k = 2 pi Z_k c + delta_k + carrier_k
    dE/dc_j = i 2 pi sum_k A_k e^{i phi_k} Z_{k,j}
    dI/dc_j = 2 Re( conj(E) * dE/dc_j )
    """
    n_rows = cache[0][2].size
    n_terms = coeffs.size
    E = np.zeros(n_rows, dtype=complex)
    dE = np.zeros((n_terms, n_rows), dtype=complex)
    for k, amp, inside, Z in cache:
        phase = 2.0 * np.pi * (coeffs @ Z)
        if deltas is not None:
            phase = phase + deltas[k]
        if carriers is not None:
            phase = phase + carriers[k]
        e = np.where(inside, amp * np.exp(1j * phase), 0.0)
        E += e
        dE += (2j * np.pi) * e[None, :] * Z
    I = np.abs(E) ** 2
    J = 2.0 * np.real(np.conj(E)[None, :] * dE).T  # (n_rows, n_terms)
    return I, J


def _reduce_scale_background(
    model: np.ndarray, meas: np.ndarray, J_model: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """变量投影：从残差中消去仿射参数 (a, b)，模型为 a*model + b。

    返回精简残差 f = a*model + b - meas、精确精简雅可比及拟合的 (a, b)。
    (a, b) 随非线性参数每步重新拟合，故雅可比不是简单的投影
    (I - Q Q^T)(a J_model)：对 stationarity 条件 A^T f = 0 求导得

        du = (A^T A)^-1 (-dA^T f - A^T dA u),   df = dA u + A du

    在奇异向量基下求 (A^T A)^-1，避免显式构造法方程矩阵（cond 平方），
    同时检测 [model, 1] 秩亏（增益/背景不可分）。
    """
    A = np.stack([model, np.ones_like(model)], axis=1)
    if A.shape[0] <= 2:
        raise ValueError(
            "变量投影需要至少 3 个观测：m <= 2 时 a*model+b 总能精确穿过"
        )
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = np.finfo(float).eps * max(A.shape) * s[0]
    if s[-1] <= tol:
        raise ValueError("增益与背景不可区分：nuisance 基 [model, 1] 秩亏")
    u = Vt.T @ ((U.T @ meas) / s)
    a, b = float(u[0]), float(u[1])
    f = A @ u - meas

    J = np.empty_like(J_model)
    for j in range(J_model.shape[1]):
        dA = np.column_stack([J_model[:, j], np.zeros_like(model)])
        du = Vt.T @ ((Vt @ (-dA.T @ f - A.T @ (dA @ u))) / s**2)
        J[:, j] = dA @ u + A @ du
    return f, J, a, b


def fit_wavefront_from_frames(
    forward: ForwardModel,
    indices: Sequence[int],
    frames: Sequence[np.ndarray],
    deltas: Sequence[np.ndarray | None] | None = None,
    carriers: Sequence[np.ndarray | None] | None = None,
    *,
    x0: Sequence[float] | None = None,
    config: LMConfig | None = None,
    samples: int | None = 4096,
    seed: int = 0,
) -> LMResult:
    """从光强帧（相移序列或单帧载频图）LM 拟合 Zernike 系数。

    ``deltas`` / ``carriers`` 每帧一项，描述已知的逐级相位调制；载频帧
    对应 ``deltas=None``。``indices`` 为拟合的 Fringe 序号（Z1 平移不可
    观测，不应包含）。``samples`` 为每帧随机采样像素数，None 表示全图。
    """
    indices = check_indices(indices)
    if 1 in indices:
        raise ValueError("Z1 平移对光强不可观测，请从 indices 中去掉")
    if samples is not None and int(samples) <= 0:
        raise ValueError("samples 必须是正整数（None 表示全图）")
    frames = [np.asarray(fr, dtype=float) for fr in frames]
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("frames 至少需要一帧")
    for i, fr in enumerate(frames):
        if fr.shape != forward.shape:
            raise ValueError(
                f"frames[{i}] 形状必须是 {forward.shape}，得到 {fr.shape}"
            )
    deltas = list(deltas) if deltas is not None else [None] * n_frames
    carriers = list(carriers) if carriers is not None else [None] * n_frames
    if len(deltas) != n_frames or len(carriers) != n_frames:
        raise ValueError("deltas/carriers 必须与 frames 一一对应")
    n_orders = len(forward.order_list)
    for i, d in enumerate(deltas):
        if d is not None:
            d = np.asarray(d, dtype=float)
            if d.shape != (n_orders,) or not np.all(np.isfinite(d)):
                raise ValueError(
                    f"deltas[{i}] 必须是长度为 {n_orders} 的有限数组"
                )
            deltas[i] = d
    for i, c in enumerate(carriers):
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape != (n_orders, *forward.shape) or not np.all(np.isfinite(c)):
                raise ValueError(
                    f"carriers[{i}] 形状必须是 ({n_orders}, {forward.shape[0]}, "
                    f"{forward.shape[1]}) 且元素有限，得到 {c.shape}"
                )
            carriers[i] = c

    n_pix = forward.shape[0] * forward.shape[1]
    if samples is not None and samples < n_pix:
        rows = np.sort(np.random.default_rng(seed).choice(n_pix, samples, replace=False))
    else:
        rows = np.arange(n_pix)
    # 变量投影额外消去 (a, b) 两个自由度：精简残差空间维数至多 m - 2
    n_free = len(indices) + (2 if (config and config.fit_scale_background) else 0)
    if n_frames * rows.size < n_free:
        raise ValueError(
            f"观测数 {n_frames * rows.size} 少于待拟合自由度 {n_free}：欠定"
        )
    meas = np.concatenate([fr.ravel()[rows] for fr in frames])
    carriers_s = [
        None if c is None else np.asarray(c).reshape(len(forward.order_list), -1)[:, rows]
        for c in carriers
    ]
    # 每级移位坐标/光瞳/Z 基只算一次
    cache = forward.zernike_samples(indices, rows)

    def residual_and_jac(c):
        f_parts, j_parts = [], []
        for i_frame in range(n_frames):
            I, J = _frame_and_jacobian(
                cache, c, deltas[i_frame], carriers_s[i_frame]
            )
            f_parts.append(I - meas[i_frame * rows.size : (i_frame + 1) * rows.size])
            j_parts.append(J)
        return np.concatenate(f_parts), np.vstack(j_parts)

    scale_fn = None
    if (config or LMConfig()).fit_scale_background:
        def scale_fn(f_raw, J_raw):
            return _reduce_scale_background(f_raw + meas, meas, J_raw)

    x0 = np.zeros(len(indices)) if x0 is None else np.asarray(x0, dtype=float)
    return levenberg_marquardt(residual_and_jac, x0, config, scale_background=scale_fn)


def fit_wavefront_from_carrier_frame(
    forward: ForwardModel,
    indices: Sequence[int],
    image: np.ndarray,
    *,
    f0: float | None = None,
    **kwargs,
) -> LMResult:
    """单帧载频干涉图的 LM 拟合（不相移、不解调、不解包裹）。"""
    f0 = forward.config.carrier_f0 if f0 is None else float(f0)
    carriers = [forward.carrier_phases(f0)]
    return fit_wavefront_from_frames(
        forward, indices, [image], [None], carriers, **kwargs
    )


def multistart_fit(
    forward: ForwardModel,
    indices: Sequence[int],
    frames,
    deltas=None,
    carriers=None,
    *,
    term: int = 0,
    values: Sequence[float] = (),
    coarse_iter: int = 25,
    config: LMConfig | None = None,
    **kwargs,
) -> LMResult:
    """对某一个系数做粗扫描取最优起点再精化（非线性最小二乘只局部收敛）。

    大像差（多条纹）代价面有多个极小；对轻微像差用零起点的
    ``fit_wavefront_from_frames`` 即可。
    """
    cfg = config or LMConfig()
    n_terms = len(indices)
    best_x0, best_cost = np.zeros(n_terms), np.inf
    coarse = replace(cfg, max_iter=coarse_iter)
    for v in values:
        x0 = np.zeros(n_terms)
        x0[term] = float(v)
        res = fit_wavefront_from_frames(
            forward, indices, frames, deltas, carriers,
            x0=x0, config=coarse, **kwargs,
        )
        if res.cost < best_cost:
            best_cost, best_x0 = res.cost, res.x.copy()
    return fit_wavefront_from_frames(
        forward, indices, frames, deltas, carriers,
        x0=best_x0, config=cfg, **kwargs,
    )
