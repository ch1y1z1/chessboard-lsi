"""差分 Zernike 最小二乘波前重构（论文式 2-26 ... 2-31）。

测得的两个方向差分波前

    dW_x = W(x+s, y) - W(x-s, y),   dW_y = W(x, y+s) - W(x, y-s)

展开为差分 Zernike 基的线性组合 ΔW = ΔZ c，最小二乘解

    c = (ΔZ^T ΔZ)^{-1} ΔZ^T ΔW

两个可选修正：

``fit_offsets``
    真实棋盘光栅 A_00 > 0 而 A_10 < 0、A_01 > 0，x 对与 y 对的解调相位
    相差半条纹。该常数是双方向上各自独立的冗余自由度，每方向估计一个
    可保持系数无偏（注意：它与 tilt 列共线，tilt 系数随之失去意义）。

``weights``
    逐像素权重（如调制度）用于加权最小二乘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .zernike import differential_zernike_matrix, wavefront

__all__ = ["ZernikeFit", "fit_differential_zernike", "wavefront_on_grid"]


@dataclass
class ZernikeFit:
    indices: np.ndarray
    coeffs: np.ndarray                     # 拟合系数（waves，不含偏移）
    offsets: dict[str, float] = field(default_factory=dict)
    residual: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_residual: float = 0.0
    cond: float = np.nan
    rank: int = 0

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
    fit_offsets: bool = False,
    known_offsets: dict[str, float] | None = None,
    weights_x: np.ndarray | None = None,
    weights_y: np.ndarray | None = None,
) -> ZernikeFit:
    """最小二乘差分 Zernike 重构（式 2-28 ... 2-31）。

    ``known_offsets``（键 "x"/"y"，单位与 dW 相同，即波长）在拟合前从
    数据中扣除；对物理棋盘可用 ``ForwardModel.demodulation_offset(d)/pi``。
    """
    indices = np.asarray(list(indices), dtype=int)

    rows, rhs, wts, tags = [], [], [], []
    for direction, d, mask, wgt in (
        ("x", dW_x, mask_x, weights_x),
        ("y", dW_y, mask_y, weights_y),
    ):
        if d is None:
            continue
        mask = np.asarray(mask, dtype=bool) & np.isfinite(d)
        Z = differential_zernike_matrix(indices, x[mask], y[mask], s, direction)
        rows.append(Z)
        rhs.append(np.asarray(d, dtype=float)[mask])
        wts.append(np.sqrt(np.clip(wgt[mask], 0.0, None)) if wgt is not None
                   else np.ones(int(mask.sum())))
        tags.append(direction)
    if not rows:
        raise ValueError("差分拟合没有有效数据")

    if fit_offsets:
        # 每个方向一个独立的常数列（其余块为零）
        n_off = len(rows)
        blocks = []
        for k, Z in enumerate(rows):
            off = np.zeros((Z.shape[0], n_off))
            off[:, k] = 1.0
            blocks.append(np.hstack([Z, off]))
        rows = blocks
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    w = np.concatenate(wts)
    if known_offsets:
        # 在装 b 之前按方向扣除已知常数
        offset = np.concatenate([
            np.full(len(r), known_offsets.get(t, 0.0)) for r, t in zip(rhs, tags)
        ])
        b = b - offset

    sol, _, rank, _ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    coeffs = sol[: len(indices)]
    offsets = (
        {t: float(v) for t, v in zip(tags, sol[len(indices):])}
        if fit_offsets else {}
    )
    residual = A @ sol - b
    return ZernikeFit(
        indices=indices,
        coeffs=coeffs,
        offsets=offsets,
        residual=residual,
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        cond=float(np.linalg.cond(A * w[:, None])),
        rank=int(rank),
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
