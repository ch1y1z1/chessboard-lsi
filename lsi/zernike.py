"""Zernike polynomials in the Fringe/Wyant convention used by the dissertation.

The dissertation (表 2-5) lists the first 16 *differential* Zernike
polynomials ``dZ(x,y) = Z(x+s,y) - Z(x-s,y)``.  Inverting that table shows
the underlying basis is the classical (un-normalized) real Zernike set

    Z_j(rho,theta) = R_n^|m|(rho) * { cos(m theta)  (even j)
                                     sin(m theta)  (odd  j) }

with the Fringe/Wyant ordering:

     j  (n,m)      j  (n,m)         j  (n,m)
     1  (0,0)      6  (2,2)sin      11  (3,3)sin
     2  (1,1)cos    7  (3,1)cos      12  (4,2)cos
     3  (1,1)sin    8  (3,1)sin      13  (4,2)sin
     4  (2,0)       9  (4,0)         14  (5,1)cos
     5  (2,2)cos   10  (3,3)cos      15  (5,1)sin
                                     16  (6,0)

Unlike Noll ordering, Fringe ordering then continues with the ``m=4`` pair
at Z17/Z18 and reserves Z25 and Z36 for the next radial spherical terms.
The complete named table through Z36 below is deliberately explicit because
those numbers carry physical aberration meanings in chapters 4 and 5.

The radial polynomial is

    R_n^m(rho) = sum_{s=0}^{(n-m)/2} (-1)^s (n-s)! /
                 [ s! ((n+m)/2 - s)! ((n-m)/2 - s)! ] rho^(n-2s)

Numerically we use the factorization ``Z = Q(rho^2) * Re/Im[(x+iy)^m]``
which is stable at the origin and avoids any explicit atan2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "ZernikeMode",
    "FRINGE_MODES",
    "PAPER_ORDER",
    "fringe_index",
    "noll_like_index",
    "radial_polynomial",
    "zernike_value",
    "zernike_matrix",
    "zernike_terms",
    "differential_zernike",
    "differential_zernike_matrix",
    "as_wavefront",
]

@dataclass(frozen=True)
class ZernikeMode:
    """One real Fringe/Wyant Zernike mode."""

    n: int
    m: int
    kind: str
    aberration_name: str


#: Explicit Fringe/Wyant table used by the dissertation, indexed from one.
FRINGE_MODES: tuple[ZernikeMode, ...] = (
    ZernikeMode(0, 0, "radial", "piston"),                    # 1
    ZernikeMode(1, 1, "cos", "x tilt"),                      # 2
    ZernikeMode(1, 1, "sin", "y tilt"),                      # 3
    ZernikeMode(2, 0, "radial", "defocus"),                  # 4
    ZernikeMode(2, 2, "cos", "primary astigmatism"),         # 5
    ZernikeMode(2, 2, "sin", "primary astigmatism"),         # 6
    ZernikeMode(3, 1, "cos", "primary coma"),                # 7
    ZernikeMode(3, 1, "sin", "primary coma"),                # 8
    ZernikeMode(4, 0, "radial", "primary spherical"),        # 9
    ZernikeMode(3, 3, "cos", "primary trefoil"),             # 10
    ZernikeMode(3, 3, "sin", "primary trefoil"),             # 11
    ZernikeMode(4, 2, "cos", "secondary astigmatism"),       # 12
    ZernikeMode(4, 2, "sin", "secondary astigmatism"),       # 13
    ZernikeMode(5, 1, "cos", "secondary coma"),              # 14
    ZernikeMode(5, 1, "sin", "secondary coma"),              # 15
    ZernikeMode(6, 0, "radial", "secondary spherical"),      # 16
    ZernikeMode(4, 4, "cos", "primary quadrafoil"),          # 17
    ZernikeMode(4, 4, "sin", "primary quadrafoil"),          # 18
    ZernikeMode(5, 3, "cos", "secondary trefoil"),           # 19
    ZernikeMode(5, 3, "sin", "secondary trefoil"),           # 20
    ZernikeMode(6, 2, "cos", "tertiary astigmatism"),        # 21
    ZernikeMode(6, 2, "sin", "tertiary astigmatism"),        # 22
    ZernikeMode(7, 1, "cos", "tertiary coma"),               # 23
    ZernikeMode(7, 1, "sin", "tertiary coma"),               # 24
    ZernikeMode(8, 0, "radial", "tertiary spherical"),       # 25
    ZernikeMode(5, 5, "cos", "primary pentafoil"),           # 26
    ZernikeMode(5, 5, "sin", "primary pentafoil"),           # 27
    ZernikeMode(6, 4, "cos", "secondary quadrafoil"),        # 28
    ZernikeMode(6, 4, "sin", "secondary quadrafoil"),        # 29
    ZernikeMode(7, 3, "cos", "tertiary trefoil"),            # 30
    ZernikeMode(7, 3, "sin", "tertiary trefoil"),            # 31
    ZernikeMode(8, 2, "cos", "quaternary astigmatism"),      # 32
    ZernikeMode(8, 2, "sin", "quaternary astigmatism"),      # 33
    ZernikeMode(9, 1, "cos", "quaternary coma"),             # 34
    ZernikeMode(9, 1, "sin", "quaternary coma"),             # 35
    ZernikeMode(10, 0, "radial", "quaternary spherical"),    # 36
)

# Backward-compatible triples for callers that used the old public constant.
PAPER_ORDER: tuple[tuple[int, int, str], ...] = tuple(
    (mode.n, mode.m, mode.kind) for mode in FRINGE_MODES
)


def fringe_index(j: int) -> tuple[int, int, str]:
    """Return ``(n, |m|, kind)`` for one-based Fringe/Wyant index ``j``."""
    if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
        raise ValueError(f"Zernike index must be an integer, got {j!r}")
    j = int(j)
    if j < 1:
        raise ValueError("Zernike index is 1-based")
    if j <= len(PAPER_ORDER):
        return PAPER_ORDER[j - 1]

    # Fringe indices form square groups.  Group p contains azimuthal orders
    # p-1 ... 1 (cos/sin pairs), followed by its radial term at index p**2.
    p = isqrt(j - 1) + 1
    offset = j - (p - 1) ** 2 - 1
    pair = offset // 2
    m = p - 1 - pair
    if m == 0:
        return 2 * (p - 1), 0, "radial"
    n = 2 * (p - 1) - m
    return n, m, "cos" if offset % 2 == 0 else "sin"


def noll_like_index(j: int) -> tuple[int, int, str]:
    """Compatibility alias for :func:`fringe_index`.

    The historical name was inaccurate: the dissertation uses Fringe/Wyant,
    not Noll, ordering.
    """
    return fringe_index(j)


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
    n, m, kind = fringe_index(j)
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