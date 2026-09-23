"""差分 Zernike 最小二乘波前重构（论文式 2-26 ... 2-31）。

测得的两个方向差分波前

    dW_x = W(x+s, y) - W(x-s, y),   dW_y = W(x, y+s) - W(x, y-s)

展开为差分 Zernike 基的线性组合 ΔW = ΔZ c，最小二乘解

    c = (ΔZ^T ΔZ)^{-1} ΔZ^T ΔW

``weights`` 为逐像素权重（如调制度），用于加权最小二乘。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .zernike import differential_zernike_matrix, wavefront


@dataclass
class ZernikeFit:
    indices: np.ndarray
    coeffs: np.ndarray                     # 拟合系数（waves）
    rms_residual: float = 0.0
    cond: float = np.nan

    def as_dict(self) -> dict[int, float]:
        return {int(j): float(c) for j, c in zip(self.indices, self.coeffs)}


def fit_differential_zernike(
    dW_x: np.ndarray,
    dW_y: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    s: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    indices: Sequence[int] = tuple(range(2, 17)),
    weights_x: np.ndarray | None = None,
    weights_y: np.ndarray | None = None,
) -> ZernikeFit:
    """最小二乘差分 Zernike 重构（式 2-28 ... 2-31）。

    Z1（平移）没有差分信号，不应出现在 ``indices`` 中。
    """
    if 1 in indices:
        raise ValueError("Z1 平移没有差分信号，请从 indices 中去掉")

    rows, rhs, wts = [], [], []
    for direction, d, mask, wgt in (
        ("x", dW_x, mask_x, weights_x),
        ("y", dW_y, mask_y, weights_y),
    ):
        mask = np.asarray(mask, dtype=bool) & np.isfinite(d)
        Z = differential_zernike_matrix(indices, x[mask], y[mask], s, direction)
        rows.append(Z)
        rhs.append(np.asarray(d, dtype=float)[mask])
        wts.append(np.sqrt(np.clip(wgt[mask], 0.0, None)) if wgt is not None
                   else np.ones(int(mask.sum())))
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    w = np.concatenate(wts)

    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    residual = A @ sol - b
    return ZernikeFit(
        indices=np.asarray(indices),
        coeffs=sol,
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        cond=float(np.linalg.cond(A * w[:, None])),
    )


def wavefront_on_grid(
    coeffs: Sequence[float],
    indices: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    pupil: np.ndarray | None = None,
) -> np.ndarray:
    W = wavefront(coeffs, indices, x, y)
    return np.where(pupil, W, np.nan) if pupil is not None else W
