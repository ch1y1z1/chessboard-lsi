"""Forward model of the 45-degree-rotated chessboard-grating shearing
interferometer (dissertation chapter 2.3 / 2.4).

Model
-----
The grating splits the incident field into diffraction orders ``(a, b)`` of
the *detector* frame.  With the order amplitudes ``A_ab`` (see
``grating.py``) and the exit pupil ``P`` (unit disk), the complex amplitude
on the detector is

    E(x,y) = sum_{a,b} A_ab * P(x+a s, y+b s)
                     * exp{ i 2*pi*[ W(x+a s, y+b s) + a*phi_x + b*phi_y
                                        + f0*(a*x + b*y) ] }

    I(x,y) = |E(x,y)|^2

where ``W`` is the wavefront under test in waves, ``s`` the shear ratio and
``phi_x, phi_y`` the grating phase-shift terms:

    phase-shift mode :  phi_x = t_x,  phi_y = t_y   (units of 2*pi)
    Fourier mode     :  2*pi*f0*(a x + b y) is the carrier, t_x = t_y = 0

Phase-shift law (dissertation eq. 2-10).  Translating the grating along the
detector x direction by ``dx`` shifts order ``(a,b)`` by

    delta_ab = 2*pi*(a * t_x + b * t_y),     t_x = sqrt(2) dx / p

i.e. the phase shift is proportional to the diffraction order, and moving the
grating by ``p/sqrt(2)`` (``t_x = 1``) advances the (+1,+1) order by exactly
2*pi while leaving the (+1,-1) order untouched -- exactly the statement of
section 2.3.1 of the dissertation.

Why the 5-beam superposition is *the* model
-------------------------------------------
Each order carries its own (translated) pupil, so the 2/3/4/5-beam overlap
regions of figures 2-8 / 2-9 appear automatically: no hand-made region
bookkeeping is required.  ``paper_region_intensity`` re-implements the
dissertation's explicit 4-beam and 5-beam formulas (2-12) ... (2-15) so that
the superposition model can be validated against them (see
``tests/test_forward_vs_paper.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import warnings

from .config import Grid, SystemConfig, check_direction
from .grating import OrderSet, analytic_orders
from .zernike import zernike_value

__all__ = [
    "ZernikeWavefront",
    "ForwardModel",
    "paper_region_intensity",
    "add_noise",
    "DEFAULT_ORDERS_5",
    "DEFAULT_ORDERS_9",
]

DEFAULT_ORDERS_5: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)
DEFAULT_ORDERS_9: tuple[tuple[int, int], ...] = DEFAULT_ORDERS_5 + (
    (2, 1), (2, -1), (-2, 1), (-2, -1),
)


# --------------------------------------------------------------------------- #
# wavefront container
# --------------------------------------------------------------------------- #
@dataclass
class ZernikeWavefront:
    """``W = sum_j c_j Z_j`` with the dissertation's Zernike convention.

    ``coeffs[j]`` is the coefficient *in waves* of Zernike index
    ``indices[j]`` (1-based).  Index 1 is piston and is normally kept at 0.
    """

    coeffs: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        self.coeffs = np.atleast_1d(np.asarray(self.coeffs, dtype=float))
        self.indices = np.atleast_1d(np.asarray(self.indices, dtype=int))
        if len(self.coeffs) != len(self.indices):
            raise ValueError("coeffs and indices must have the same length")

    # -- evaluation ----------------------------------------------------- #
    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(x, dtype=float))
        for c, j in zip(self.coeffs, self.indices):
            if c:
                out = out + c * zernike_value(int(j), x, y)
        return out

    def terms(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``(n_terms, *shape)`` array of ``Z_j`` at ``(x, y)``."""
        return np.stack(
            [zernike_value(int(j), x, y) for j in self.indices], axis=0
        )

    @property
    def n_terms(self) -> int:
        return len(self.coeffs)

    def copy_with(self, coeffs: Sequence[float]) -> "ZernikeWavefront":
        return ZernikeWavefront(np.asarray(coeffs, dtype=float), self.indices)


class _CallableWavefront:
    """Adapter for an arbitrary ``W(x, y)`` callable (used in tests)."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self._fn = fn

    def w(self, x, y):
        return self._fn(x, y)


# --------------------------------------------------------------------------- #
# forward model
# --------------------------------------------------------------------------- #
class ForwardModel:
    """Order-superposition forward model of the chessboard LSI."""

    def __init__(
        self,
        config: SystemConfig | None = None,
        orders: OrderSet | Iterable[tuple[int, int]] = DEFAULT_ORDERS_5,
        *,
        pupil: np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        normalize_orders: bool = False,
    ) -> None:
        self.config = config or SystemConfig()
        self.grid: Grid = self.config.grid
        if isinstance(orders, OrderSet):
            self.orders = orders
        else:
            base = analytic_orders(max_index=3)
            self.orders = base.with_orders(orders)
        if normalize_orders:
            amp = self.orders.amp
            scale = float(np.sqrt(np.sum(np.abs(amp) ** 2)))
            if not np.isfinite(scale) or scale == 0.0:
                raise ValueError(
                    "normalize_orders=True needs at least one non-zero order "
                    "amplitude; this order set has none (a missing order keeps "
                    "amplitude 0, see OrderSet.with_orders)."
                )
            self.orders = OrderSet(self.orders.ab, amp / scale)
        self._pupil, self._pupil_fn = self._prepare_pupil(pupil)
        self._x, self._y = self.grid.coords()

    # ---------------------------------------------------------------- info
    @property
    def shape(self) -> tuple[int, int]:
        return (self.grid.n, self.grid.n)

    @property
    def s(self) -> float:
        return self.config.s

    @property
    def indices(self) -> list[tuple[int, int]]:
        return self.orders.indices()

    @property
    def amplitudes(self) -> np.ndarray:
        return self.orders.amp

    # ------------------------------------------------------------- precompute
    def _prepare_pupil(self, pupil) -> np.ndarray | None:
        """Validate the user pupil and return ``(array, callable)``.

        ``pupil`` may be

        ``None``
            The analytic unit disk ``x**2 + y**2 <= 1`` is used (the default).
        a callable ``f(x, y) -> mask``
            Evaluated at the *shifted* pupil coordinates, so the order pupils
            are analytic and a non-integer shear introduces no error at all.
            This is the exact path for a non-circular aperture; the default
            unit disk is the special case ``f = lambda x, y: x*x + y*y <= 1``.
            The returned mask is used as given when boolean and thresholded at
            ``> 0.5`` otherwise.
        an array
            Must be sampled on the model grid (``grid.shape``).  Boolean arrays
            are used as given, any other array is treated as a hard aperture
            via ``> 0.5``.  The order pupils of an array are obtained by a
            nearest-neighbour *index* shift, which is exact only when the shear
            is an integer number of pixels, so the non-integer case is reported
            here (once) rather than on every ``order_geometry`` call.  Note
            that this snapping cannot be fixed by interpolating the array: a
            sampled binary mask has already lost its sub-pixel edge position,
            so pass a callable if that precision matters.
        """
        if pupil is None:
            return None, None
        if callable(pupil):
            return None, pupil
        arr = np.asarray(pupil)
        if arr.shape != self.shape:
            raise ValueError(
                f"pupil must have shape {self.shape} to match the grid, got "
                f"{arr.shape}"
            )
        px = self.s / self.grid.dx
        if not np.isclose(px, np.rint(px)):
            warnings.warn(
                f"shear={self.s:g} is {px:.3f} pixels, not an integer: the "
                "shear-shifted copies of a custom pupil are snapped to the "
                "grid, so the order pupils (and every region mask derived "
                "from them) carry a sub-pixel edge error.  Pass a callable "
                "pupil instead to get the analytic order pupils.",
                stacklevel=2,
            )
        if arr.dtype == bool:
            return arr, None
        return arr > 0.5, None

    def _pupil_mask(self, a: int, b: int, xs: np.ndarray, ys: np.ndarray):
        """Pupil transmission mask of order ``(a, b)``.

        Without a user pupil this is the unit disk, evaluated at the order's
        shifted coordinates.  With a callable pupil it is that callable
        evaluated at the same shifted coordinates, so the geometry stays exact
        for a non-integer shear.  With an array pupil the array is resampled by
        a nearest-neighbour index shift, which is exact when ``a * s`` is an
        integer multiple of the grid pitch and snaps to the grid otherwise.
        """
        if self._pupil_fn is not None:
            out = np.asarray(self._pupil_fn(xs, ys))
            if out.shape != np.shape(xs):
                raise ValueError(
                    f"pupil callable must return an array with the shape of "
                    f"its inputs {np.shape(xs)}, got {out.shape}"
                )
            return out if out.dtype == bool else out > 0.5
        if self._pupil is None:
            return (xs * xs + ys * ys) <= 1.0
        n = self.grid.n
        dx = self.grid.dx
        # Round the displacement once, then translate the whole sampled mask.
        # Rounding ``j + 0.5`` per pixel uses IEEE ties-to-even and alternates
        # between the two neighbours, which duplicates/skips pixels instead of
        # producing a rigid nearest-neighbour shift.
        col_shift = int(np.rint(a * self.s / dx))
        row_shift = int(np.rint(b * self.s / dx))
        jj = np.arange(n)[None, :] + col_shift
        ii = np.arange(n)[:, None] + row_shift
        ok = (jj >= 0) & (jj < n) & (ii >= 0) & (ii < n)
        out = np.zeros((n, n), dtype=bool)
        sampled = self._pupil[
            np.clip(ii, 0, n - 1), np.clip(jj, 0, n - 1)
        ]
        out[ok] = sampled[ok]
        return out

    def order_geometry(self):
        """Return per-order shifted coordinates, pupil masks and (a, b)."""
        out = []
        for k, (a, b) in enumerate(self.indices):
            xs = self._x + a * self.s
            ys = self._y + b * self.s
            inside = self._pupil_mask(a, b, xs, ys)
            out.append((a, b, xs, ys, inside))
        return out

    def aperture(self) -> np.ndarray:
        """Aperture support of the zero order, on ``grid.shape``.

        This is the unit disk unless a custom pupil was passed to the
        constructor.  The demodulation stages have to use *this* mask rather
        than a hard-coded unit circle, otherwise a custom-pupil forward model
        and the pipeline that consumes its frames disagree about which pixels
        carry signal.
        """
        # Delegating to the zero-order pupil mask keeps aperture() and
        # order_support(0, 0) identical by construction for every pupil kind.
        return self._pupil_mask(0, 0, self._x, self._y)

    @property
    def has_custom_pupil(self) -> bool:
        """Whether a user pupil was supplied (the unit disk is then not used)."""
        return self._pupil is not None or self._pupil_fn is not None

    @property
    def pupil_definition(self):
        """The user pupil as supplied: a callable, a boolean array, or ``None``.

        ``None`` means the analytic unit disk, i.e. the default pupil.  This
        returns the *definition*, not the sampled mask -- handing the array from
        :meth:`aperture` to a consumer that shifts it would downgrade a callable
        pupil to whole-pixel snapping for no reason.
        """
        if self._pupil_fn is not None:
            return self._pupil_fn
        return self._pupil

    def order_support(self, a: int, b: int) -> np.ndarray:
        """Pupil mask of order ``(a, b)``, or all-False if it is not present."""
        for aa, bb, _, _, inside in self.order_geometry():
            if (aa, bb) == (a, b):
                return inside
        return np.zeros(self.shape, dtype=bool)

    # ------------------------------------------------------------- phase law
    def phase_shift_deltas(self, t_x: float = 0.0, t_y: float = 0.0) -> np.ndarray:
        """Per-order phase shift (radians) for grating translation ``(t_x,t_y)``."""
        ab = self.orders.ab.astype(float)
        return 2.0 * np.pi * (ab[:, 0] * t_x + ab[:, 1] * t_y)

    def carrier_phases(self, x, y, f0: float) -> np.ndarray:
        """Per-order carrier phase (radians) ``2*pi*f0*(a x + b y)``."""
        ab = self.orders.ab.astype(float)
        return 2.0 * np.pi * f0 * (ab[:, 0, None, None] * x + ab[:, 1, None, None] * y)

    # ------------------------------------------------------------- field
    def field(
        self,
        wf,
        deltas: np.ndarray | None = None,
        carriers: np.ndarray | None = None,
    ) -> np.ndarray:
        """Complex amplitude on the detector."""
        E = np.zeros(self.shape, dtype=complex)
        for k, (a, b, xs, ys, inside) in enumerate(self.order_geometry()):
            amp = self.orders.amp[k]
            if amp == 0.0:
                continue
            ph = 2.0 * np.pi * wf.w(xs, ys)
            if deltas is not None:
                ph = ph + deltas[k]
            if carriers is not None:
                ph = ph + carriers[k]
            E = E + np.where(inside, amp * np.exp(1j * ph), 0.0)
        return E

    def intensity(self, wf, deltas=None, carriers=None) -> np.ndarray:
        E = self.field(wf, deltas=deltas, carriers=carriers)
        return np.abs(E) ** 2

    def demodulation_offset(self, direction: str = "x") -> float:
        """Constant phase carried by the demodulated frequency-1 interferogram.

        With amplitudes ``A_0`` and ``A_1`` the two beats (zero order with the
        ``+1`` and with the ``-1`` order) combine into

            D = 2 A_0 A_1 exp(i pi [W(x+s)-W(x-s)]) cos(Gamma)

        so the demodulated phase equals ``pi * dW`` shifted by the *argument*
        of the amplitude product ``A_0 A_1`` (``A_0`` is real positive here, so
        this is also ``arg(conj(A_0) A_1)``).  For the ideal 50 % duty that
        constant is exactly ``0`` or ``pi``; for a duty error it drifts
        smoothly with the grating phase (``arg A_10 = pi - 2 pi (d - 1/2)``,
        ``arg A_01 = 0``), which is why the duty-cycle error of section 4.2.2
        moves the demodulated constant and nothing else.  An additional sign
        flip appears wherever the modulation ``cos(Gamma)`` changes sign --
        the limitation discussed in section 2.3.2 of the dissertation.

        For the physical chessboard ``(A_00, A_10, A_01) =
        (+1/2, -2/pi^2, +2/pi^2)`` this gives ``pi`` for the x pair and ``0``
        for the y pair: the two shear pairs sit half a fringe apart.

        Caveat: a constant phase is indistinguishable from a tilt in a
        two-sided shearing measurement, and the unwrapping branch leaves a
        further ``2 pi k`` free.  This value is therefore *prior information*
        taken from the grating model, not something the interferogram alone
        can supply.
        """
        check_direction(direction)
        ab = {tuple(o): i for i, o in enumerate(self.indices)}
        zero = ab.get((0, 0))
        key = (1, 0) if direction == "x" else (0, 1)
        other = ab.get(key)
        if zero is None or other is None:
            return 0.0
        a0, a1 = self.orders.amp[zero], self.orders.amp[other]
        if a0 == 0 or a1 == 0:
            return 0.0
        return float(np.angle(a0 * a1))

    # ------------------------------------------------- phase-shift frames
    def phase_shift_frames(
        self,
        wf,
        direction: str = "x",
        n_steps: int | None = None,
    ) -> np.ndarray:
        """``(n_steps, n, n)`` intensity frames for a grating phase shift.

        The step is ``t = i / n_steps`` (i = 0 ... n_steps-1), i.e. the
        first-order phase step is ``2*pi/n_steps`` as required by the N-step
        least-squares algorithm of dissertation eq. (2-20).
        """
        check_direction(direction)
        n_steps = n_steps or self.config.phase_steps
        frames = []
        for i in range(n_steps):
            t = i / float(n_steps)
            deltas = (
                self.phase_shift_deltas(t, 0.0)
                if direction == "x"
                else self.phase_shift_deltas(0.0, t)
            )
            frames.append(self.intensity(wf, deltas=deltas))
        return np.stack(frames, axis=0)

    def phase_shift_stack(self, wf, n_steps: int | None = None):
        """Return ``(frames_x, frames_y)`` for the two shear directions."""
        n_steps = n_steps or self.config.phase_steps
        return (
            self.phase_shift_frames(wf, "x", n_steps),
            self.phase_shift_frames(wf, "y", n_steps),
        )

    def phase_shift_delta_table(self, n_steps: int | None = None, direction="x"):
        """The per-step phase applied to the (+1, 0) (or (0, +1)) order."""
        check_direction(direction)
        n_steps = n_steps or self.config.phase_steps
        return np.array([2.0 * np.pi * i / n_steps for i in range(n_steps)])

    # ------------------------------------------------------ Fourier mode
    def ft_mode_frame(self, wf, f0: float | None = None) -> np.ndarray:
        """Single carrier-mode frame (dissertation 2.4.1)."""
        f0 = self.config.carrier_f0 if f0 is None else f0
        f0 = float(f0)
        nyquist = 1.0 / (2.0 * self.grid.dx)
        if not np.isfinite(f0) or f0 <= 0.0:
            raise ValueError(f"f0 must be a positive finite frequency, got {f0!r}")
        if f0 >= nyquist:
            raise ValueError(
                f"f0={f0:g} is not below the grid Nyquist frequency "
                f"{nyquist:g}; increase grid.n or reduce f0"
            )
        carriers = self.carrier_phases(self._x, self._y, f0)
        return self.intensity(wf, carriers=carriers)

    # ------------------------------------------- analytic Jacobian pieces
    def terms_cache(
        self, wf_proto: ZernikeWavefront, rows: np.ndarray
    ) -> "OrderTermsCache":
        """Precompute per-order shifted coordinates, masks and Z_j samples."""
        return OrderTermsCache(self, wf_proto, rows)

    def model_and_jacobian(
        self,
        coeffs: np.ndarray,
        wf_proto: ZernikeWavefront,
        delta_list: Sequence[np.ndarray | None],
        carrier_list: Sequence[np.ndarray | None] | None = None,
        *,
        rows: np.ndarray | None = None,
        cache: "OrderTermsCache" | None = None,
    ):
        """Intensity stack and its Jacobian w.r.t. the fitted coefficients.

        ``coeffs`` are the *fitted* coefficients; ``wf_proto.indices`` defines
        which Zernike terms are fitted.  Returns ``(I_stack, J)`` with
        ``I_stack`` of shape ``(n_frames, n_samples)`` and ``J`` of shape
        ``(n_frames * n_samples, n_terms)``.
        """
        if rows is None:
            rows = np.arange(self.shape[0] * self.shape[1])
        if cache is None:
            cache = self.terms_cache(wf_proto, rows)
        n_terms = cache.n_terms
        n_rows = len(rows)

        I_out = np.empty((len(delta_list), n_rows))
        J_out = np.empty((len(delta_list) * n_rows, n_terms))
        for i_frame, deltas in enumerate(delta_list):
            carriers = None if carrier_list is None else carrier_list[i_frame]
            E = np.zeros(n_rows, dtype=complex)
            dE = np.zeros((n_terms, n_rows), dtype=complex)
            # ``cache.order_indices`` maps a *cache* position to the original
            # position in ``self.orders``; every other cache attribute is
            # indexed in the compressed cache space (orders with zero amplitude
            # are dropped from it).  Iterating the original indices and using
            # them on ``cache.z``/``cache.factor`` mixed the two spaces and
            # raised IndexError as soon as a zero-amplitude order was not last.
            for i_order in range(len(cache.order_indices)):
                fac = cache.factor(i_order, coeffs, deltas, carriers)
                E = E + fac
                dE = dE + (1j * 2.0 * np.pi) * cache.z[i_order] * fac[None, :]
            I = np.abs(E) ** 2
            dI = 2.0 * np.real(np.conj(E)[None, :] * dE)
            I_out[i_frame] = I
            J_out[i_frame * n_rows : (i_frame + 1) * n_rows] = dI.T
        return I_out.reshape(-1), J_out

    def sample_grid_rows(self, samples: int | None = None, rng=None) -> np.ndarray:
        """Random subset of detector samples (flat indices)."""
        n_tot = self.shape[0] * self.shape[1]
        if samples is None or samples >= n_tot:
            return np.arange(n_tot)
        rng = np.random.default_rng(0) if rng is None else rng
        return np.sort(rng.choice(n_tot, size=samples, replace=False))


class OrderTermsCache:
    """Per-order sampled geometry and Zernike basis (built once for LM).

    Index space: every list attribute (``ab``, ``amp``, ``inside``, ``z``) is
    in *compressed* cache space --- orders whose amplitude is zero are dropped.
    ``order_indices[i]`` gives the position of cache entry ``i`` in
    ``ForwardModel.orders``, and is the only thing to use when indexing
    ``deltas``/``carriers``.  ``factor`` follows that convention already.
    """

    def __init__(
        self, forward: ForwardModel, wf_proto: ZernikeWavefront, rows: np.ndarray
    ) -> None:
        self.shape = forward.shape
        self.order_indices: list[int] = []
        self.ab: list[tuple[int, int]] = []
        self.amp: list[complex] = []
        self.inside: list[np.ndarray] = []
        self.z: list[np.ndarray] = []
        for k, (a, b, xs, ys, inside) in enumerate(forward.order_geometry()):
            if forward.orders.amp[k] == 0.0:
                continue
            xs_f = xs.reshape(-1)[rows]
            ys_f = ys.reshape(-1)[rows]
            ins = inside.reshape(-1)[rows]
            Z = wf_proto.terms(xs_f, ys_f)  # (n_terms, n_rows)
            self.order_indices.append(k)
            self.ab.append((a, b))
            self.amp.append(forward.orders.amp[k])
            self.inside.append(ins)
            self.z.append(Z)
        self.n_terms = wf_proto.n_terms

    def factor(self, i_order: int, coeffs, deltas, carriers):
        ins = self.inside[i_order]
        W = np.tensordot(coeffs, self.z[i_order], axes=(0, 0))
        ph = 2.0 * np.pi * W
        if deltas is not None:
            d = deltas[self.order_indices[i_order]]
            if d is not None:
                ph = ph + d
        if carriers is not None:
            ph = ph + carriers[self.order_indices[i_order]]
        return np.where(ins, self.amp[i_order] * np.exp(1j * ph), 0.0)


# --------------------------------------------------------------------------- #
# the dissertation's explicit region formulas (for validation)
# --------------------------------------------------------------------------- #
def _samples(W: Callable, x, y, s: float) -> dict[str, np.ndarray]:
    return {
        "0": W(x, y),
        "xp": W(x + s, y),
        "xm": W(x - s, y),
        "yp": W(x, y + s),
        "ym": W(x, y - s),
    }


def paper_region_intensity(
    region: str,
    W: Callable,
    x: np.ndarray,
    y: np.ndarray,
    shear: float,
    delta: float,
    A0: float = 0.5,
    A1: float | None = None,
) -> np.ndarray:
    """Dissertation eqs. (2-12) ... (2-15), transcribed literally.

    ``region`` is one of

    ``"x1"``  eq. (2-12): 0 + (+1,+1) + (-1,+1) + (-1,-1)
    ``"x2"``  eq. (2-13): 0 + (+1,+1) + (-1,-1) + (+1,-1)
    ``"x5"``  eq. (2-14): 0 + all four first orders (five beams)
    ``"y1"``  eq. (2-15): 0 + (+1,+1) + (-1,+1) + (+1,-1)
    ``"y2"``  eq. (2-16): 0 + (+1,+1) + (-1,-1) + (-1,+1)

    The dissertation uses one common first-order amplitude ``A_{+1,+1}``;
    the exact chessboard amplitudes ``(A_10, A_01) = (-2/pi^2, +2/pi^2)``
    therefore correspond to ``A1 = 2/pi^2`` with a sign convention fixed by
    ``sign`` in the caller (see the validation test).  Distances/arguments
    follow the paper: ``s`` is the shear, ``delta`` the phase shift.
    """
    s = shear
    w = _samples(W, x, y, s)
    A1 = 2.0 / np.pi**2 if A1 is None else A1
    two_pi = 2.0 * np.pi
    half_sum_x = (w["xp"] + w["xm"]) / 2.0
    half_diff_x = (w["xp"] - w["xm"]) / 2.0
    half_sum_y = (w["yp"] + w["ym"]) / 2.0
    half_diff_y = (w["yp"] - w["ym"]) / 2.0

    if region in ("x1", "x2"):
        other = "yp" if region == "x1" else "ym"
        I = A0**2 + 3.0 * A1**2
        I = I + 2.0 * A0 * A1 * np.cos(two_pi * (w[other] - w["0"]))
        I = I + 4.0 * A0 * A1 * np.cos(two_pi * (half_sum_x - w["0"])) * np.cos(
            two_pi * half_diff_x + delta
        )
        I = I + 4.0 * A1**2 * np.cos(
            two_pi * (half_sum_x - w[other])
        ) * np.cos(two_pi * half_diff_x + delta)
        I = I + 2.0 * A1**2 * np.cos(two_pi * 2.0 * half_diff_x + 2.0 * delta)
        return I

    if region in ("y1", "y2"):
        other = "xp" if region == "y1" else "xm"
        I = A0**2 + 3.0 * A1**2
        I = I + 2.0 * A0 * A1 * np.cos(two_pi * (w[other] - w["0"]))
        I = I + 4.0 * A0 * A1 * np.cos(two_pi * (half_sum_y - w["0"])) * np.cos(
            two_pi * half_diff_y + delta
        )
        I = I + 4.0 * A1**2 * np.cos(
            two_pi * (half_sum_y - w[other])
        ) * np.cos(two_pi * half_diff_y + delta)
        I = I + 2.0 * A1**2 * np.cos(two_pi * 2.0 * half_diff_y + 2.0 * delta)
        return I

    if region == "x5":
        I = A0**2 + 4.0 * A1**2
        I = I + 2.0 * A0 * A1 * (
            np.cos(two_pi * (w["yp"] - w["0"])) + np.cos(two_pi * (w["ym"] - w["0"]))
        )
        I = I + 2.0 * A1**2 * np.cos(two_pi * (w["yp"] - w["ym"]))
        for coeff, shift in ((4.0 * A0 * A1, "0"), (4.0 * A1**2, "yp"),
                             (4.0 * A1**2, "ym")):
            I = I + coeff * np.cos(
                two_pi * (half_sum_x - w[shift])
            ) * np.cos(two_pi * half_diff_x + delta)
        I = I + 2.0 * A1**2 * np.cos(two_pi * 2.0 * half_diff_x + 2.0 * delta)
        return I

    if region == "y5":
        I = A0**2 + 4.0 * A1**2
        I = I + 2.0 * A0 * A1 * (
            np.cos(two_pi * (w["xp"] - w["0"])) + np.cos(two_pi * (w["xm"] - w["0"]))
        )
        I = I + 2.0 * A1**2 * np.cos(two_pi * (w["xp"] - w["xm"]))
        for shift in ("0", "xp", "xm"):
            I = I + 4.0 * A0 * A1 * np.cos(
                two_pi * (half_sum_y - w[shift])
            ) * np.cos(two_pi * half_diff_y + delta)
        I = I + 2.0 * A1**2 * np.cos(two_pi * 2.0 * half_diff_y + 2.0 * delta)
        return I

    raise ValueError(f"unknown region {region!r}")


# --------------------------------------------------------------------------- #
def add_noise(
    frames: np.ndarray, snr_db: float | None = None, seed: int | None = 0,
    poisson_scale: float | None = None,
) -> np.ndarray:
    """Add Gaussian (``snr_db``) and/or Poisson shot noise to intensity frames."""
    frames = np.asarray(frames, dtype=float)
    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain only finite values")
    if snr_db is None and poisson_scale is None:
        return frames
    if snr_db is not None:
        snr_db = float(snr_db)
        if not np.isfinite(snr_db):
            raise ValueError(f"snr_db must be finite, got {snr_db!r}")
    if poisson_scale is not None:
        poisson_scale = float(poisson_scale)
        if not np.isfinite(poisson_scale) or poisson_scale <= 0.0:
            raise ValueError(
                f"poisson_scale must be positive and finite, got {poisson_scale!r}"
            )
    rng = np.random.default_rng(seed)
    out = frames.copy()
    if poisson_scale is not None:
        out = rng.poisson(np.maximum(out, 0.0) * poisson_scale) / poisson_scale
    if snr_db is not None:
        peak = out.max()
        sigma = peak / (10.0 ** (snr_db / 20.0))
        out = out + rng.normal(0.0, sigma, size=out.shape)
    return out
