"""通用 Zernike 与 RMS 归一化。

Z1..Z36 直接沿用 ``lsi.model.FRINGE_MODES`` 的 (n, m, kind)；截断尾项
取 Fringe 第 6、7 组（n + m = 12、14），这些模式都不在 Z1..Z36 内。

单位圆上 R_n^m(rho) * {cos, sin}(m theta) 的 RMS（面积平均）为
    m = 0:  1 / sqrt(n + 1)
    m > 0:  1 / sqrt(2 (n + 1))
"""

from __future__ import annotations

from math import factorial
from typing import Sequence

import numpy as np

from lsi.model import FRINGE_MODES


def fringe_group(g: int) -> list[tuple[int, int, str]]:
    """Fringe 第 g 组（n + m = 2g）的模式表，组内 m 递减。"""
    out = []
    for m in range(g, -1, -1):
        n = 2 * g - m
        if m == 0:
            out.append((n, 0, "radial"))
        else:
            out.append((n, m, "cos"))
            out.append((n, m, "sin"))
    return out


def fringe_nm(j: int) -> tuple[int, int, str]:
    if 1 <= j <= len(FRINGE_MODES):
        n, m, kind, _ = FRINGE_MODES[j - 1]
        return n, m, kind
    raise ValueError(f"Fringe 序号 {j} 超出 1..{len(FRINGE_MODES)}")


#: 截断尾项模式（Z36 以外）：Fringe 第 6、7 组
TAIL_MODES: tuple[tuple[int, int, str], ...] = tuple(fringe_group(6) + fringe_group(7))


def zernike_nm(n: int, m: int, kind: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """未归一化实 Zernike R_n^m * {cos, sin}(m theta)，与 lsi.model.zernike 同式。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho2 = x * x + y * y
    kmax = (n - m) // 2
    q = np.zeros_like(rho2)
    for s in range(kmax + 1):
        coeff = (
            (-1.0) ** s * factorial(n - s)
            / (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s))
        )
        q = q + coeff * rho2 ** (kmax - s)
    if m == 0:
        return q
    z = (x + 1j * y) ** m
    return q * (np.real(z) if kind == "cos" else np.imag(z))


def rms_factor(n: int, m: int) -> float:
    return 1.0 / np.sqrt((n + 1.0) * (1.0 if m == 0 else 2.0))


def rms_factors(indices: Sequence[int]) -> np.ndarray:
    return np.array([rms_factor(*fringe_nm(j)[:2]) for j in indices])


def radial_orders(indices: Sequence[int]) -> np.ndarray:
    return np.array([fringe_nm(j)[0] for j in indices])


def basis(indices: Sequence[int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(..., p) 的 Fringe 基矩阵（c 坐标）。"""
    return np.stack([zernike_nm(*fringe_nm(j), x, y) for j in indices], axis=-1)


def basis_q(indices: Sequence[int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """q 坐标的基：Z_j / rms_j，单位圆上正交归一（面积平均）。"""
    return basis(indices, x, y) / rms_factors(indices)


def tail_basis_q(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        [zernike_nm(n, m, k, x, y) / rms_factor(n, m) for n, m, k in TAIL_MODES], axis=-1
    )


def to_c(q: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(q, float) / rms_factors(indices)


def to_q(c: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(c, float) * rms_factors(indices)
