"""Zernike 多项式（论文采用的 Fringe/Wyant 排序）。

论文表 2-5 列出前 16 项差分 Zernike 多项式 dZ(x,y) = Z(x+s,y) - Z(x-s,y)，
反推可知其基底是经典（未归一化）实 Zernike 集：

    Z_j(rho, theta) = R_n^|m|(rho) * { cos(m theta)  (偶 j)
                                      sin(m theta)  (奇 j) }

Fringe/Wyant 排序（1 起）：

     j  (n,m)      j  (n,m)         j  (n,m)
     1  (0,0)      6  (2,2)sin      11  (3,3)sin
     2  (1,1)cos   7  (3,1)cos      12  (4,2)cos
     3  (1,1)sin   8  (3,1)sin      13  (4,2)sin
     4  (2,0)      9  (4,0)         14  (5,1)cos
     5  (2,2)cos  10  (3,3)cos      15  (5,1)sin
                                     16  (6,0)

径向多项式（论文式 2-25）：

    R_n^m(rho) = sum_s (-1)^s (n-s)! / [s! ((n+m)/2-s)! ((n-m)/2-s)!] rho^(n-2s)

数值实现用分解 Z = Q(rho^2) * Re/Im[(x + i y)^m]，在原点稳定且不需要 atan2。
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial, isqrt
from typing import Sequence

import numpy as np

__all__ = [
    "FRINGE_MODES",
    "fringe_index",
    "zernike",
    "zernike_matrix",
    "differential_zernike",
    "differential_zernike_matrix",
    "wavefront",
]

#: (n, |m|, kind, 像差名称)，下标从 1 开始。Fringe 排序中 m=4 对在 Z17/Z18，
#: Z25、Z36 分别是 8 级、10 级径向球差。
FRINGE_MODES: tuple[tuple[int, int, str, str], ...] = (
    (0, 0, "radial", "piston"),
    (1, 1, "cos", "x tilt"), (1, 1, "sin", "y tilt"),
    (2, 0, "radial", "defocus"),
    (2, 2, "cos", "primary astigmatism"), (2, 2, "sin", "primary astigmatism"),
    (3, 1, "cos", "primary coma"), (3, 1, "sin", "primary coma"),
    (4, 0, "radial", "primary spherical"),
    (3, 3, "cos", "primary trefoil"), (3, 3, "sin", "primary trefoil"),
    (4, 2, "cos", "secondary astigmatism"), (4, 2, "sin", "secondary astigmatism"),
    (5, 1, "cos", "secondary coma"), (5, 1, "sin", "secondary coma"),
    (6, 0, "radial", "secondary spherical"),
    (4, 4, "cos", "primary quadrafoil"), (4, 4, "sin", "primary quadrafoil"),
    (5, 3, "cos", "secondary trefoil"), (5, 3, "sin", "secondary trefoil"),
    (6, 2, "cos", "tertiary astigmatism"), (6, 2, "sin", "tertiary astigmatism"),
    (7, 1, "cos", "tertiary coma"), (7, 1, "sin", "tertiary coma"),
    (8, 0, "radial", "tertiary spherical"),
    (5, 5, "cos", "primary pentafoil"), (5, 5, "sin", "primary pentafoil"),
    (6, 4, "cos", "secondary quadrafoil"), (6, 4, "sin", "secondary quadrafoil"),
    (7, 3, "cos", "tertiary trefoil"), (7, 3, "sin", "tertiary trefoil"),
    (8, 2, "cos", "quaternary astigmatism"), (8, 2, "sin", "quaternary astigmatism"),
    (9, 1, "cos", "quaternary coma"), (9, 1, "sin", "quaternary coma"),
    (10, 0, "radial", "quaternary spherical"),
)


def fringe_index(j: int) -> tuple[int, int, str]:
    """1 起 Fringe/Wyant 序号 j -> (n, |m|, 'cos'|'sin'|'radial')。"""
    j = int(j)
    if j < 1:
        raise ValueError("Zernike 序号从 1 开始")
    if j <= len(FRINGE_MODES):
        n, m, kind, _ = FRINGE_MODES[j - 1]
        return n, m, kind
    # j > 36 的延续：Fringe 序号按平方分组，组 p 含角阶 p-1 ... 1 的
    # cos/sin 对，径向项落在 j = p^2。
    p = isqrt(j - 1) + 1
    offset = j - (p - 1) ** 2 - 1
    m = p - 1 - offset // 2
    if m == 0:
        return 2 * (p - 1), 0, "radial"
    return 2 * (p - 1) - m, m, "cos" if offset % 2 == 0 else "sin"


@lru_cache(maxsize=None)
def _q_coeffs(n: int, m: int) -> tuple[tuple[int, float], ...]:
    """R_n^m = Q(rho^2) * rho^m 中 Q 的系数，返回 (rho2 的幂, 系数) 对。"""
    if (n - m) % 2 or m > n:
        return ()
    kmax = (n - m) // 2
    return tuple(
        (
            kmax - s,
            (-1.0) ** s * factorial(n - s)
            / (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)),
        )
        for s in range(kmax + 1)
    )


def zernike(j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """单位圆上的实 Zernike 多项式 Z_j(x, y)。"""
    n, m, kind = fringe_index(j)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho2 = x * x + y * y
    q = np.zeros_like(rho2)
    for power, coeff in _q_coeffs(n, m):
        q = q + coeff * rho2**power
    if m == 0:
        return q
    # (x + i y)^m = rho^m (cos mθ + i sin mθ)
    trig = np.real((x + 1j * y) ** m) if kind == "cos" else np.imag((x + 1j * y) ** m)
    return q * trig


def zernike_matrix(
    indices: Sequence[int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """(n_pixels, n_terms) 的 Zernike 基矩阵。"""
    return np.stack([zernike(j, x, y) for j in indices], axis=-1)


def differential_zernike(
    j: int, x: np.ndarray, y: np.ndarray, s: float, direction: str
) -> np.ndarray:
    """双边差分 Zernike（论文式 2-26/2-27）。

        ΔZx_j = Z_j(x+s, y) - Z_j(x-s, y)
        ΔZy_j = Z_j(x, y+s) - Z_j(x, y-s)
    """
    if direction == "x":
        return zernike(j, x + s, y) - zernike(j, x - s, y)
    if direction == "y":
        return zernike(j, x, y + s) - zernike(j, x, y - s)
    raise ValueError("direction 必须是 'x' 或 'y'")


def differential_zernike_matrix(
    indices: Sequence[int], x: np.ndarray, y: np.ndarray, s: float, direction: str
) -> np.ndarray:
    return np.stack(
        [differential_zernike(j, x, y, s, direction) for j in indices], axis=-1
    )


def wavefront(
    coeffs: Sequence[float], indices: Sequence[int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """W(x, y) = sum_j c_j Z_j（系数单位：波长）。"""
    W = np.zeros_like(np.asarray(x, dtype=float))
    for c, j in zip(coeffs, indices):
        if c != 0.0:
            W = W + c * zernike(j, x, y)
    return W
