"""45-degree-rotated amplitude chessboard grating: diffraction orders.

Two ways to obtain the order amplitudes:

``analytic_orders``
    50 % duty-cycle closed form for the ideal rotated chessboard:
    ``A_00 = 1/2``, ``A_ab = -2 / [pi^2 (a^2 - b^2)]`` for ``a + b`` odd,
    everything else zero.  (Equivalently ``A_mn = -2/(pi^2 m n)`` in the
    grating frame.)  This reproduces 表2-3 of the dissertation exactly:
    ``|A_11| = 0.2026`` (efficiency 4.11 %), ``|A_13| = 0.0675`` (0.46 %),
    ``|A_33| = 0.0225`` (0.05 %), ``|A_15| = 0.0405`` (0.16 %),
    ``|A_55| = 0.0081`` (6.6e-3 %), and Parseval gives
    ``sum |A|^2 = 1/4 + 1/4 = 1/2`` = mean transmittance.

``bitmap_orders``
    Build a sampled unit cell of the *physical* (unrotated) chessboard
    grating with arbitrary duty cycle / offset (and a rotation by a multiple
    of 90 degrees, the only angles the sampled order lattice can express), FFT
    it, keep the (m, n) harmonics and map them onto the detector-frame indices
    ``(a, b) = ((m+n)/2, (n-m)/2)``.  This is what supports the grating
    manufacturing-error analysis of chapter 4 of the dissertation
    (占空比误差 / 图形偏移误差).

The order amplitude convention matches the field model in ``forward.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import warnings

from .config import _as_int_array

__all__ = [
    "OrderSet",
    "analytic_orders",
    "bitmap_orders",
    "diffraction_efficiency",
    "far_field_amplitude",
]


@dataclass(frozen=True)
class OrderSet:
    """Set of diffraction orders in detector-frame coordinates.

    Attributes
    ----------
    ab : (K, 2) float array
        Effective order indices ``(a, b)``; order ``(a, b)`` samples the
        wavefront at ``(x + a*s, y + b*s)``.  Integer orders describe the ideal
        checkerboard; half-integer orders are retained because a relative
        pattern-placement error creates mixed-parity grating harmonics.
    amp : (K,) complex array
        Complex amplitude of each order (normalized so that the DC order is
        1/2 for the ideal 50 % duty chessboard).
    parity : (K,) int array
        The original ``(m, n)``-frame order index ``m + n`` etc. (kept for
        bookkeeping / diffraction-efficiency studies).
    """

    ab: np.ndarray
    amp: np.ndarray
    parity: np.ndarray | None = None

    def __post_init__(self) -> None:
        raw_ab = np.asarray(self.ab)
        if raw_ab.dtype == bool:
            raise ValueError("ab must contain numeric diffraction orders")
        try:
            ab = raw_ab.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("ab must contain numeric diffraction orders") from exc
        amp = np.asarray(self.amp, dtype=complex)
        if ab.ndim != 2 or ab.shape[1:] != (2,):
            raise ValueError(f"ab must have shape (K, 2), got {ab.shape}")
        if not np.all(np.isfinite(ab)):
            raise ValueError("ab must contain only finite values")
        if not np.allclose(2.0 * ab, np.rint(2.0 * ab), atol=1e-12):
            raise ValueError("ab entries must be integer or half-integer orders")
        ab = np.rint(2.0 * ab) / 2.0
        if amp.ndim != 1 or amp.shape[0] != ab.shape[0]:
            raise ValueError(
                f"amp must have shape ({ab.shape[0]},), got {amp.shape}"
            )
        if not np.all(np.isfinite(amp)):
            raise ValueError("amp must contain only finite values")
        if len({tuple(row) for row in ab.tolist()}) != len(ab):
            raise ValueError("ab must not contain duplicate diffraction orders")
        parity = self.parity
        if parity is not None:
            parity = _as_int_array(parity, "parity", ndim=1)
            if parity.shape != (len(ab),):
                raise ValueError(
                    f"parity must have shape ({len(ab)},), got {parity.shape}"
                )
        object.__setattr__(self, "ab", ab)
        object.__setattr__(self, "amp", amp)
        object.__setattr__(self, "parity", parity)

    def __len__(self) -> int:
        return len(self.ab)

    def indices(self) -> list[tuple[int | float, int | float]]:
        def canonical(value):
            integer = int(np.rint(value))
            return integer if np.isclose(value, integer) else float(value)

        return [tuple(canonical(v) for v in row) for row in self.ab]

    def select(
        self, orders: Iterable[tuple[int | float, int | float]]
    ) -> "OrderSet":
        want = {tuple(o) for o in orders}
        keep = [i for i, o in enumerate(self.indices()) if o in want]
        return OrderSet(self.ab[keep], self.amp[keep])

    def with_orders(
        self, orders: Iterable[tuple[int | float, int | float]]
    ) -> "OrderSet":
        """Same as :meth:`select` but tolerates missing orders (amp 0)."""
        want = [tuple(o) for o in orders]
        amps = []
        for o in want:
            idx = [i for i, oo in enumerate(self.indices()) if oo == o]
            amps.append(self.amp[idx[0]] if idx else 0.0 + 0.0j)
        return OrderSet(
            np.array(want, dtype=float).reshape(-1, 2),
            np.array(amps, dtype=complex),
        )

    def efficiencies(self) -> dict[tuple[int | float, int | float], float]:
        return {
            o: float(abs(a) ** 2) for o, a in zip(self.indices(), self.amp)
        }


def analytic_orders(
    max_index: int = 3,
    *,
    include_zero: bool = True,
    duty: float = 0.5,
) -> OrderSet:
    """Closed-form order amplitudes of the (rotated) chessboard grating.

    In the grating frame the transmittance is
    ``t = (1 + s(u) s(v)) / 2`` with ``s`` the +-1 square wave of duty ``d``,
    whose Fourier coefficients are ``S_k = (1 - exp(-2 i pi k d)) / (i pi k)``.
    Hence

        A_mn = 0.5 S_m S_n
             = -0.5 (1 - e^{-2 i pi m d})(1 - e^{-2 i pi n d}) / (pi^2 m n)
             = 2 sin(pi m d) sin(pi n d) e^{-i pi (m + n) d} / (pi^2 m n)

    The phase factor ``e^{-i pi (m+n) d} = e^{-2 i pi a d}`` is the grating
    origin (a shift of the pattern by ``d`` periods along ``x``); it is what
    makes the duty-dependent amplitudes reduce *exactly* to the real values of
    表2-3 at the ideal 50 % duty,

        A_00 = 1/2,   A_mn = -2/(pi^2 m n)   for odd m, n  (表2-3).

    For a duty error the amplitudes stay on the same branch (no sign jump at
    ``d = 1/2``); ``arg A_10`` and ``arg A_01`` move in opposite directions
    proportionally to ``2 pi (d - 1/2)``, which is the half-fringe constant
    that a two-sided shearing interferogram cannot separate from tilt.

    At ``d = 1/2`` only odd/odd harmonics survive.  Away from 50 %, even and
    mixed-parity products are retained as well; the latter map to half-integer
    detector coordinates and can reach a material fraction of a first order.

    Parameters
    ----------
    max_index:
        Keep orders with ``max(|a|, |b|) <= max_index`` in the detector frame.
    duty:
        Transparent fraction of each checker cell.  Integer and half-integer
        detector orders are retained.
    """
    if isinstance(max_index, bool) or not isinstance(max_index, (int, np.integer)):
        raise ValueError(f"max_index must be a non-negative integer, got {max_index!r}")
    max_index = int(max_index)
    if max_index < 0:
        raise ValueError(f"max_index must be non-negative, got {max_index!r}")
    duty = float(duty)
    if not np.isfinite(duty) or not 0.0 < duty < 1.0:
        raise ValueError(f"duty must lie strictly between 0 and 1, got {duty!r}")

    def square_wave_coeff(k: int) -> complex:
        if k == 0:
            return complex(2.0 * duty - 1.0)
        return -np.expm1(-2j * np.pi * k * duty) / (1j * np.pi * k)

    ab, amp = [], []
    harmonic_limit = 2 * max_index
    for m in range(-harmonic_limit, harmonic_limit + 1):
        for n in range(-harmonic_limit, harmonic_limit + 1):
            if (m, n) == (0, 0):
                continue
            a = (m + n) / 2.0
            b = (n - m) / 2.0
            if max(abs(a), abs(b)) > max_index:
                continue
            if duty == 0.5:
                # At exactly 50 % only odd/odd harmonics survive.  Keep the
                # closed form so the tabulated real amplitudes stay exact.
                if m % 2 == 0 or n % 2 == 0:
                    continue
                denom = m * n
                val = -2.0 / (np.pi**2 * denom)
            else:
                # Off 50 %, even, zero and mixed-parity harmonics are physical.
                val = 0.5 * square_wave_coeff(m) * square_wave_coeff(n)
            ab.append((a, b))
            amp.append(val)
    if include_zero:
        ab.insert(0, (0, 0))
        amp.insert(0, (1.0 + (2.0 * duty - 1.0) ** 2) / 2.0)
    return OrderSet(
        np.array(ab, dtype=float).reshape(-1, 2),
        np.array(amp, dtype=complex),
    )


def bitmap_orders(
    harmonic_cell: int = 64,
    *,
    max_index: int = 3,
    duty: float = 0.5,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    rotation_deg: float = 0.0,
    edge="ideal",
) -> OrderSet:
    """Order amplitudes of a *bitmapped* chessboard grating (FFT based).

    Parameters
    ----------
    harmonic_cell:
        Resolution of the sampled grating period (samples per period).
    duty:
        Transparent fraction of each checker cell (0.5 = ideal).
    offset_x, offset_y:
        Relative lithographic placement error of the second (diagonally
        opposite) transparent checker sub-cell, in units of the period.  This
        is the dissertation's 图形偏移误差: unlike a global translation it
        changes diffraction efficiencies and creates mixed-parity harmonics.
    origin_x, origin_y:
        Global translation of the complete grating, in units of the period.
        By the Fourier shift theorem this changes order phases but not their
        efficiencies or support.
    rotation_deg:
        Rotation of the sampled pattern with respect to the ``u = x``,
        ``v = y`` frame of :func:`analytic_orders`, in degrees.

        **Only multiples of 90 are representable by the sampled order set.**
        The window holds exactly one grating period, so the pattern has to
        stay commensurate with the sampling lattice for the FFT bins to *be*
        the detector-frame orders ``(a, b)``.  A chessboard maps onto itself
        under a 90-degree rotation about a cell centre, so those angles are
        exact (90 and 270 additionally move the pattern origin, which changes
        only the order phase).  Any other angle
        rotates the reciprocal lattice off the integer bins: the energy then
        leaks across frequencies and the ``m, n`` both-odd selection keeps
        only a minority of it (18 % at 45 degrees, where the *strongest*
        orders are the axial bins that get dropped), so the returned
        amplitudes do not describe the rotated grating.  Those angles raise
        :class:`NotImplementedError` instead of returning that subset.  Use
        :func:`far_field_amplitude` for the far field of an arbitrarily
        rotated grating.
    edge:
        ``"ideal"`` or ``"soft"`` (2-pixel linear transition, mimicking
        finite lithographic resolution).
    """
    if isinstance(harmonic_cell, bool) or not isinstance(
        harmonic_cell, (int, np.integer)
    ):
        raise ValueError(
            f"harmonic_cell must be an integer, got {harmonic_cell!r}"
        )
    n = int(harmonic_cell)
    if n < 2:
        raise ValueError("harmonic_cell must be at least 2 samples per period")
    if isinstance(max_index, bool) or not isinstance(max_index, (int, np.integer)):
        raise ValueError(f"max_index must be a non-negative integer, got {max_index!r}")
    max_index = int(max_index)
    if max_index < 0:
        raise ValueError(f"max_index must be non-negative, got {max_index!r}")
    if 2 * max_index >= n / 2:
        raise ValueError(
            f"harmonic_cell={n} cannot resolve detector orders through "
            f"max_index={max_index}; need harmonic_cell > {4 * max_index}"
        )
    if edge not in ("ideal", "soft"):
        raise ValueError("edge must be 'ideal' or 'soft'")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must lie strictly between 0 and 1")
    placement = {}
    for name, value in (
        ("offset_x", offset_x),
        ("offset_y", offset_y),
        ("origin_x", origin_x),
        ("origin_y", origin_y),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite, got {value!r}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        placement[name] = value
    offset_x = placement["offset_x"]
    offset_y = placement["offset_y"]
    origin_x = placement["origin_x"]
    origin_y = placement["origin_y"]
    rot = float(rotation_deg)
    if not np.isfinite(rot):
        raise ValueError(f"rotation_deg must be finite, got {rotation_deg!r}")
    if not np.isclose(rot % 90.0, 0.0, atol=1e-9):
        raise NotImplementedError(
            f"bitmap_orders cannot represent rotation_deg={rotation_deg!r}: "
            "only multiples of 90 degrees keep the sampled pattern "
            "commensurate with the unit-cell FFT whose integer bins are the "
            "detector-frame orders (a, b).  Other angles scatter the energy "
            "off every order and would return a small, arbitrary subset of "
            "it; see far_field_amplitude() for a rotated far field."
        )
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u - origin_x, v - origin_y, indexing="xy")
    if rot % 360.0:
        theta = np.deg2rad(rot)
        U, V = (
            np.cos(theta) * U + np.sin(theta) * V,
            -np.sin(theta) * U + np.cos(theta) * V,
        )
    # One period is the union of two diagonally opposite transparent sub-cells.
    # A placement error moves only the second sub-cell; translating U and V
    # together would move the entire grating and could never create the
    # coordinate-axis odd orders reported in dissertation table 4-2.
    u0 = np.mod(U, 1.0)
    v0 = np.mod(V, 1.0)
    first = (u0 < duty) & (v0 < duty)
    second = (
        (np.mod(U - offset_x, 1.0) >= duty)
        & (np.mod(V - offset_y, 1.0) >= duty)
    )
    t = (first | second).astype(float)
    if edge == "soft":
        from scipy.ndimage import uniform_filter

        t = uniform_filter(t, size=3, mode="wrap")
    F = np.fft.fft2(t) / t.size
    freqs = np.fft.fftfreq(n, d=1.0 / n)   # cycles per grating period

    ab, amp = [], []
    tol = 64.0 * np.finfo(float).eps * max(1.0, abs(F[0, 0]))
    for k_index in range(-2 * max_index, 2 * max_index + 1):
        for l_index in range(-2 * max_index, 2 * max_index + 1):
            if (k_index, l_index) == (0, 0):
                continue
            i = int(np.argmin(np.abs(freqs - k_index)))
            j = int(np.argmin(np.abs(freqs - l_index)))
            val = F[j, i]
            a = (k_index + l_index) / 2.0
            b = (l_index - k_index) / 2.0
            if max(abs(a), abs(b)) > max_index:
                continue
            if abs(val) <= tol:
                continue
            ab.append((a, b))
            amp.append(val)
    # constant (DC) term
    ab.insert(0, (0, 0))
    amp.insert(0, F[0, 0])
    # merge duplicates (a,b) that arise from the k,l <-> symmetric cells
    merged: dict[tuple[int, int], complex] = {}
    for (a, b), val in zip(ab, amp):
        merged[(a, b)] = merged.get((a, b), 0.0 + 0.0j) + val
    items = sorted(merged.items())
    ab_arr = np.array([k for k, _ in items], dtype=float)
    amp_arr = np.array([v for _, v in items], dtype=complex)
    return OrderSet(ab_arr, amp_arr)


def diffraction_efficiency(order_set: OrderSet) -> dict[str, float]:
    """Diffraction efficiency of the +-1 orders relative to the DC term."""
    eff = order_set.efficiencies()
    dc = eff.get((0, 0), 0.0)
    first = eff.get((1, 0), 0.0) + eff.get((-1, 0), 0.0)
    first += eff.get((0, 1), 0.0) + eff.get((0, -1), 0.0)
    return {
        "dc": dc,
        "first_order_total": first,
        "first_over_dc": first / dc if dc else float("inf"),
        "all": eff,
    }


def far_field_amplitude(
    grating: str,
    n: int = 512,
    orders: int = 12,
    *,
    duty: float = 0.5,
    rotation_deg: float = 0.0,
    extent: float = 1.0,
):
    """Far-field (Fraunhofer) amplitude distribution of a grating.

    Reproduces the chapter 2 comparison between a 1-D amplitude grating, a
    crossed (2-D) grating and the 45-degree rotated chessboard (the rotated
    case needs a wider window, see ``extent`` below): it shows why the
    chessboard is preferred (only odd orders, higher first-order efficiency,
    two orthogonal shears from one element).

    The grating is sampled over a square window of ``extent`` grating periods
    per side with ``n`` samples, so the far field is resolved on a spectral
    grid of pitch ``1 / extent`` in units of ``1 / p`` (``p`` = grating
    period).  The default ``extent = 1`` samples exactly one period, which
    gives the exact line spectrum of the periodic grating: every order lands
    on a spectral sample and ``A`` vanishes elsewhere.  A wider window lowers
    the order range resolved by Nyquist but spreads each order into the sinc
    profile of the finite aperture; an integer width avoids the extra
    leakage of a window that holds a fractional number of periods.
    ``rotation_deg`` rotates the pattern, and with it the far field.  A
    rotated pattern is incommensurate with a single-period window, so it is
    only resolved with a wider window: ``extent = 25`` shows the 45-degree
    rotated chessboard's first orders as sinc-sampled peaks on the
    ``(0, +-sqrt(2))`` and ``(+-sqrt(2), 0)`` axes, while the default
    single-period window reports the broadband spectrum of the
    incommensurate window content.  For ``duty = 0.5`` the first orders
    carry ``2 / pi`` of the DC amplitude for the 1-D and crossed gratings and
    ``4 / pi^2`` for the chessboard's diagonal ``(+-1, +-1)`` orders, which is
    also why the chessboard has *only* diagonal orders (no axial
    ``(odd, 0)`` / ``(0, odd)``): one element delivers both shear
    directions.

    Returns ``(fx, fy, A)`` with the coordinates in units of ``1 / p``, cropped
    to ``|fx|, |fy| <= orders``, so the default call returns the 25 x 25 map
    of orders -12 ... 12.  The crop cannot reach beyond the Nyquist order
    ``n / (2 * extent)`` of the sampled window; a warning is issued when
    ``orders`` asks for more.  ``A`` is normalised to its global maximum.
    """
    from numpy.fft import fft2, fftshift

    m = int(n)
    if m < 2:
        raise ValueError("n must be at least 2 samples")
    if orders <= 0:
        raise ValueError("orders must be positive")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must lie strictly between 0 and 1")
    if grating not in ("grating1d", "crossed", "chessboard"):
        raise ValueError(
            "grating must be 'grating1d', 'crossed' or 'chessboard'"
        )
    if extent <= 0:
        raise ValueError("extent must be positive")
    width = float(extent)
    nyquist = m / (2.0 * width)
    if orders > nyquist:
        warnings.warn(
            f"orders={orders} exceeds the Nyquist order {nyquist:.3f} of a"
            f" {width:g}-period window, so the returned map is not cropped",
            stacklevel=2,
        )
    if rotation_deg and not np.isclose(float(rotation_deg) % 90.0, 0.0, atol=1e-9):
        # A rotated pattern is incommensurate with the sampling window: its
        # orders sit at non-lattice positions and are only resolved once the
        # window is wide enough to give each of them its own sinc peak.
        # Returning the broadband spectrum of a narrow window without saying
        # so invites reading the noise as the order pattern.
        if width < 4.0:
            warnings.warn(
                f"rotation_deg={rotation_deg:g} with extent={width:g}: a "
                "rotated pattern is incommensurate with a narrow window, so "
                "the returned spectrum is the broadband content of the window "
                "rather than resolved orders.  Use a wider window "
                "(extent >= 4) or a multiple of 90 degrees.",
                stacklevel=2,
            )

    dx = width / m                      # sample pitch, in grating periods
    u = (np.arange(m) + 0.5) * dx
    U, V = np.meshgrid(u, u, indexing="xy")
    if rotation_deg:
        theta = np.deg2rad(rotation_deg)
        U, V = (
            np.cos(theta) * U + np.sin(theta) * V,
            -np.sin(theta) * U + np.cos(theta) * V,
        )
    if grating == "grating1d":
        t = (np.mod(U, 1.0) < duty).astype(float)
    elif grating == "crossed":
        t = (np.mod(U, 1.0) < duty).astype(float) * (
            np.mod(V, 1.0) < duty
        ).astype(float)
    else:                                 # chessboard
        su = 2 * (np.mod(U, 1.0) < duty) - 1
        sv = 2 * (np.mod(V, 1.0) < duty) - 1
        t = (1 + su * sv) / 2.0
    A = np.abs(fftshift(fft2(t)))
    peak = float(A.max())
    if peak:
        A /= peak
    f = fftshift(np.fft.fftfreq(m, d=dx))   # diffraction orders, units of 1/p
    keep = np.abs(f) <= orders
    fx, fy = np.meshgrid(f[keep], f[keep], indexing="xy")
    A = A[np.ix_(keep, keep)]
    return fx, fy, A
