"""Zernike polynomials in the index convention actually used by the
reference dissertation.

The dissertation (表 2-5) lists the first 16 *differential* Zernike
polynomials ``dZ(x,y) = Z(x+s,y) - Z(x-s,y)``.  Inverting that table shows
the underlying basis is the classical (un-normalized) real Zernike set

    Z_j(rho,theta) = R_n^|m|(rho) * { cos(m theta)  (even j)
                                     sin(m theta)  (odd  j) }

with the ordering (Noll-like sequence, but the cos/sin pair of j=7/8 is
swapped with respect to Noll):

     j  (n,m)      j  (n,m)         j  (n,m)
     1  (0,0)      6  (2,2)sin      11  (3,3)sin
     2  (1,1)cos    7  (3,1)cos      12  (4,2)cos
     3  (1,1)sin    8  (3,1)sin      13  (4,2)sin
     4  (2,0)       9  (4,0)         14  (5,1)cos
     5  (2,2)cos   10  (3,3)cos      15  (5,1)sin
                                     16  (6,0)

The radial polynomial is

    R_n^m(rho) = sum_{s=0}^{(n-m)/2} (-1)^s (n-s)! /
                 [ s! ((n+m)/2 - s)! ((n-m)/2 - s)! ] rho^(n-2s)

Numerically we use the factorization ``Z = Q(rho^2) * Re/Im[(x+iy)^m]``
which is stable at the origin and avoids any explicit atan2.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PAPER_ORDER",
    "noll_like_index",
    "radial_polynomial",
    "zernike_value",
    "zernike_matrix",
    "zernike_terms",
    "differential_zernike",
    "differential_zernike_matrix",
    "as_wavefront",
]

#: (n, m, kind) with kind in {"cos", "sin", "radial"}; 1-based index table.
PAPER_ORDER: tuple[tuple[int, int, str], ...] = (
    (0, 0, "radial"),   # 1
    (1, 1, "cos"),      # 2
    (1, 1, "sin"),      # 3
    (2, 0, "radial"),   # 4
    (2, 2, "cos"),      # 5
    (2, 2, "sin"),      # 6
    (3, 1, "cos"),      # 7   (dissertation's "Z7 coma", cos-type)
    (3, 1, "sin"),      # 8
    (4, 0, "radial"),   # 9
    (3, 3, "cos"),      # 10
    (3, 3, "sin"),      # 11
    (4, 2, "cos"),      # 12
    (4, 2, "sin"),      # 13
    (5, 1, "cos"),      # 14
    (5, 1, "sin"),      # 15
    (6, 0, "radial"),   # 16
)

_NAMES = {
    "radial": "R_n^0",
    "cos": "coef",
    "sin": "coef",
}


def noll_like_index(j: int) -> tuple[int, int, str]:
    """Return ``(n, m, kind)`` for 1-based index ``j`` (dissertation order)."""
    if j < 1:
        raise ValueError("Zernike index is 1-based")
    if j <= len(PAPER_ORDER):
        return PAPER_ORDER[j - 1]
    return _extend_to(j)[j - 1]


@lru_cache(maxsize=None)
def _extend_to(j: int) -> tuple[tuple[int, int, str], ...]:
    """Extend the index table with the standard Noll-like sequence."""
    table = list(PAPER_ORDER)
    n = 6
    while len(table) < j:
        n += 1
        for m in range(n % 2, n + 1, 2):
            if m == 0:
                table.append((n, 0, "radial"))
            else:
                table.append((n, m, "cos"))
                table.append((n, m, "sin"))
    return tuple(table)


@lru_cache(maxsize=None)
def _q_coeffs(n: int, m: int) -> tuple[tuple[int, float], ...]:
    """Coefficients of ``Q`` with ``R_n^m = Q(rho^2) * rho^m``.

    Returns pairs ``(power_of_rho2, coefficient)``.
    """
    m = abs(m)
    if (n - m) % 2 or m > n:
        return ()
    from math import factorial

    kmax = (n - m) // 2
    out = []
    for s in range(kmax + 1):
        c = (-1) ** s * factorial(n - s)
        c /= (
            factorial(s)
            * factorial((n + m) // 2 - s)
            * factorial((n - m) // 2 - s)
        )
        out.append((kmax - s, float(c)))
    return tuple(out)


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """``R_n^m(rho)`` for ``n >= 0``, ``|m| <= n``, ``n - |m|`` even."""
    rho = np.asarray(rho, dtype=float)
    rho2 = rho**2
    out = np.zeros_like(rho2)
    for power, coeff in _q_coeffs(n, m):
        out = out + coeff * rho2**power
    m = abs(m)
    return out * rho**m if m else out


def zernike_value(j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Real Zernike polynomial ``Z_j(x, y)`` on the unit disk."""
    n, m, kind = noll_like_index(int(j))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho2 = x * x + y * y
    q = np.zeros_like(rho2)
    for power, coeff in _q_coeffs(n, m):
        q = q + coeff * rho2**power
    if m == 0:
        return q
    # (x + i y)^m -> rho^m * (cos m theta + i sin m theta)
    z = (x + 1j * y) ** m
    trig = np.real(z) if kind == "cos" else np.imag(z)
    return q * trig


def zernike_matrix(
    indices: Sequence[int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Matrix ``(n_pixels, n_terms)`` of ``Z_j`` evaluated at ``(x, y)``."""
    cols = [zernike_value(j, x, y) for j in indices]
    return np.stack(cols, axis=-1)


def zernike_terms(jmax: int) -> list[int]:
    return list(range(1, jmax + 1))


def differential_zernike(
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    shear: float,
    direction: str = "x",
    model: str = "two_sided",
) -> np.ndarray:
    """Differential Zernike polynomial (dissertation eq. 2-26/2-27).

    ``model="two_sided"``  : ``Z(x+s, y) - Z(x-s, y)``      (0 / +-1 beams)
    ``model="one_sided"``  : ``Z(x+s, y) - Z(x, y)``        (0 / +1 beam only,
                             i.e. what an isolated ``+f0`` carrier lobe sees)
    ``model="one_sided_doubled"``: ``2 [Z(x+s) - Z(x)]``    (one-sided lobe
                             interpreted with the dissertation's "x2" rule)
    """
    if direction == "x":
        if model == "two_sided":
            return zernike_value(j, x + shear, y) - zernike_value(j, x - shear, y)
        if model == "one_sided":
            return zernike_value(j, x + shear, y) - zernike_value(j, x, y)
        if model == "one_sided_doubled":
            return 2.0 * (zernike_value(j, x + shear, y) - zernike_value(j, x, y))
    elif direction == "y":
        if model == "two_sided":
            return zernike_value(j, x, y + shear) - zernike_value(j, x, y - shear)
        if model == "one_sided":
            return zernike_value(j, x, y + shear) - zernike_value(j, x, y)
        if model == "one_sided_doubled":
            return 2.0 * (zernike_value(j, x, y + shear) - zernike_value(j, x, y))
    else:
        raise ValueError("direction must be 'x' or 'y'")
    raise ValueError(f"unknown difference model {model!r}")


def differential_zernike_matrix(
    indices: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    shear: float,
    direction: str,
    model: str = "two_sided",
) -> np.ndarray:
    cols = [
        differential_zernike(j, x, y, shear, direction, model) for j in indices
    ]
    return np.stack(cols, axis=-1)


def as_wavefront(
    coeffs: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Evaluate ``W = sum_j c_j Z_j`` on the grid (coefficients in waves)."""
    coeffs = np.asarray(coeffs, dtype=float)
    if indices is None:
        indices = range(1, len(coeffs) + 1)
    W = np.zeros_like(np.asarray(x, dtype=float))
    for c, j in zip(coeffs, indices):
        if c != 0.0:
            W = W + c * zernike_value(j, x, y)
    return W