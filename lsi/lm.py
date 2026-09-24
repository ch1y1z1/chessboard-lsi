"""Levenberg–Marquardt 对非线性前向模型的直接反演（第三条路线）。

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
单帧载频图即可反演。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .model import ForwardModel

_MAX_ITER = 120          # 最大迭代次数
_LAMBDA0 = 1e-3          # 初始阻尼
_NU0 = 2.0               # 拒绝步时 lambda 的增长因子初值
_FTOL = 1e-14            # 代价相对变化收敛阈
_XTOL = 1e-12            # 步长收敛阈
_GTOL = 1e-12            # 梯度范数收敛阈
_LAMBDA_MIN, _LAMBDA_MAX = 1e-12, 1e10


@dataclass
class LMResult:
    indices: np.ndarray
    coeffs: np.ndarray                 # 拟合的 Zernike 系数（waves）
    cost: float                        # ||f||^2
    n_iter: int = 0
    rms_residual: float = 0.0

    def as_dict(self) -> dict[int, float]:
        return {int(j): float(c) for j, c in zip(self.indices, self.coeffs)}


def _solve_damped(
    J: np.ndarray, f: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """增广最小二乘求阻尼步，返回 (delta, g = J^T f, diag(J^T J))。"""
    diag = np.clip(np.diag(J.T @ J), 1e-30, None)
    g = J.T @ f
    A = np.vstack([J, np.diag(np.sqrt(lam * diag))])
    delta = np.linalg.lstsq(A, np.concatenate([-f, np.zeros(J.shape[1])]), rcond=None)[0]
    return delta, g, diag


def levenberg_marquardt(
    residual_and_jac: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0: Sequence[float],
) -> tuple[np.ndarray, float, int, float]:
    """最小化 ||f(x)||^2；``residual_and_jac(x) -> (f, J)`` 用解析雅可比。

    返回 (x, cost, n_iter, rms_residual)。
    """
    x = np.asarray(x0, dtype=float).copy()

    f, J = residual_and_jac(x)
    cost = float(f @ f)
    lam, nu = _LAMBDA0, _NU0
    n_iter = 0

    for it in range(_MAX_ITER):
        n_iter = it + 1
        delta, g, diag = _solve_damped(J, f, lam)

        if np.linalg.norm(g) <= _GTOL:
            break
        if np.linalg.norm(delta) <= _XTOL * (np.linalg.norm(x) + _XTOL):
            break

        f_new, J_new = residual_and_jac(x + delta)
        cost_new = float(f_new @ f_new)
        # Nielsen 增益比：实际下降 / 阻尼模型预测下降
        # cost = ||f||^2 约定下预测下降 = delta^T (lam D delta - g)
        predicted = float(delta @ (lam * diag * delta - g))
        rho = (cost - cost_new) / predicted if predicted > 0 else -1.0

        if rho > 0.0:
            drop = cost - cost_new
            x, f, J, cost = x + delta, f_new, J_new, cost_new
            lam = float(np.clip(
                lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3),
                _LAMBDA_MIN, _LAMBDA_MAX,
            ))
            nu = _NU0
            if abs(drop) <= _FTOL * max(cost, 1e-30):
                break
        else:
            lam = float(np.clip(lam * nu, _LAMBDA_MIN, _LAMBDA_MAX))
            nu *= 2.0
            if lam >= _LAMBDA_MAX:
                break

    return x, cost, n_iter, float(np.sqrt(cost / max(f.size, 1)))


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


def fit_wavefront_from_frames(
    forward: ForwardModel,
    indices: Sequence[int],
    frames: Sequence[np.ndarray],
    deltas: Sequence[np.ndarray | None] | None = None,
    carriers: Sequence[np.ndarray | None] | None = None,
    *,
    samples: int | None = 4096,
    seed: int = 0,
) -> LMResult:
    """从光强帧（相移序列或单帧载频图）LM 拟合 Zernike 系数，从零初值起步。

    ``deltas`` / ``carriers`` 每帧一项，描述已知的逐级相位调制；载频帧
    对应 ``deltas=None``。``indices`` 为拟合的 Fringe 序号（Z1 平移不可
    观测，不应包含）。``samples`` 为每帧随机采样像素数，None 表示全图。
    """
    frames = [np.asarray(fr, dtype=float) for fr in frames]
    n_frames = len(frames)
    deltas = list(deltas) if deltas is not None else [None] * n_frames
    carriers = list(carriers) if carriers is not None else [None] * n_frames

    n_pix = forward.shape[0] * forward.shape[1]
    if samples is not None and samples < n_pix:
        rows = np.sort(np.random.default_rng(seed).choice(n_pix, samples, replace=False))
    else:
        rows = np.arange(n_pix)
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

    coeffs, cost, n_iter, rms = levenberg_marquardt(
        residual_and_jac, np.zeros(len(indices))
    )
    return LMResult(
        indices=np.asarray(indices),
        coeffs=coeffs,
        cost=cost,
        n_iter=n_iter,
        rms_residual=rms,
    )


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
