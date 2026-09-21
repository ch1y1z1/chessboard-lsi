"""System configuration for the 45-degree-rotated chessboard-grating
lateral shearing interferometer (LSI).

All lengths in the *normalized pupil* coordinate system unless stated
otherwise: the exit pupil is the unit disk (radius 1) on the grid
``[-extent, extent]^2``, and wavefronts are expressed in units of the
wavelength (i.e. "waves").

Every field is validated in ``__post_init__``.  The checks deliberately reject
a ``float`` where a count is expected (``n=64.0``, ``phase_steps=8.0``): the
old ``int(x) != x`` test accepted those and the failure surfaced much later as
an opaque ``TypeError`` from ``np.zeros``/``range``.

Physical parameters follow chapter 3 of the reference dissertation:

    相移模式 :  NA = 0.34, p = 18 um, lambda = 632.8 nm  ->  s = 0.0731
    傅里叶模式: NA = 0.34, p = 30 um, lambda = 632.8 nm  ->  s = 0.0439

(The dissertation text prints lambda = 633 nm; this code uses 632.8 nm
(He-Ne).  Both values give the same quoted shear ratios 0.0731 / 0.0439.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

__all__ = [
    "Grid",
    "SystemConfig",
    "check_difference_model",
    "check_direction",
    "check_offset_mode",
    "check_region_mode",
    "check_unwrap_method",
]


def _as_int(value, name: str, minimum: int = 1) -> int:
    """Return ``value`` as a real ``int``, or raise if it is not one.

    ``int(x) != x`` accepts any ``float`` that happens to be a whole number
    (``64.0``), which then propagates into ``shape`` / ``range`` and fails much
    later with a confusing ``TypeError``.  Booleans are ``int`` subclasses but
    are never a meaningful count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return value


def _as_int_array(
    value, name: str, *, ndim: int | None = None, minimum: int | None = None
) -> np.ndarray:
    """Return an integer ndarray without truncating floats or booleans."""
    array = np.asarray(value)
    if array.size == 0:
        array = array.astype(int)
    elif array.dtype == bool or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain only integers")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    array = array.astype(int, copy=False)
    if minimum is not None and np.any(array < minimum):
        raise ValueError(f"{name} entries must be at least {minimum}")
    return array


def _as_float(value, name: str, *, low: float, high: float | None = None,
              inclusive_low: bool = False, inclusive_high: bool = True) -> float:
    """Return ``value`` as a finite ``float`` inside the requested interval.

    ``inclusive_low`` / ``inclusive_high`` select between ``(low, high)``,
    ``(low, high]`` and ``[low, high)``; the interval built here is the one the
    error message prints, so the two can never disagree.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    below = value < low if inclusive_low else value <= low
    above = high is not None and (value > high if inclusive_high else value >= high)
    if below or above:
        left = "[" if inclusive_low else "("
        right = "]" if inclusive_high else ")"
        bound = f"{high:g}" if high is not None else "inf"
        raise ValueError(
            f"{name} must lie in {left}{low:g}, {bound}{right}, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class Grid:
    """Square sampling grid in normalized pupil coordinates."""

    n: int = 256
    extent: float = 1.10  # grid spans [-extent, extent] in both x and y

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _as_int(self.n, "n", minimum=1))
        object.__setattr__(
            self, "extent", _as_float(self.extent, "extent", low=0.0)
        )

    @property
    def dx(self) -> float:
        return 2.0 * self.extent / self.n

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def coords(self) -> Tuple[np.ndarray, np.ndarray]:
        v = (np.arange(self.n) - (self.n - 1) / 2.0) * self.dx
        x, y = np.meshgrid(v, v, indexing="xy")
        return x, y

    def radius(self) -> np.ndarray:
        x, y = self.coords()
        return np.hypot(x, y)

    def pupil(self) -> np.ndarray:
        """Unit-disk (exit pupil) indicator on the grid."""
        return self.radius() <= 1.0


@dataclass(frozen=True)
class SystemConfig:
    """Physical + numerical configuration for the scalar air-pupil model.

    Instances are immutable value objects so a :class:`ForwardModel` cannot
    retain cached geometry from one configuration while reading live values
    from a later mutation.  Use :func:`dataclasses.replace` to derive a variant.

    ``na`` is deliberately restricted to ``0 < NA <= 1``.  Supporting
    immersion ``NA > 1`` requires a refractive index, vector diffraction and
    high-NA pupil-coordinate corrections; merely relaxing the validator would
    incorrectly imply that those effects are modelled.
    """

    wavelength_nm: float = 632.8
    na: float = 0.34
    period_um: float = 18.0
    #: metric shear ratio (order displacement as a fraction of the pupil
    #: radius); if None it is derived from wavelength / NA / period.
    shear_ratio: float | None = None
    #: number of phase-shift frames per direction (dissertation uses 8)
    phase_steps: int = 8
    #: Talbot number m used in Fourier-transform (carrier) mode
    talbot_number: int = 1
    grid: Grid = field(default_factory=Grid)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wavelength_nm",
            _as_float(self.wavelength_nm, "wavelength_nm", low=0.0),
        )
        object.__setattr__(
            self, "na", _as_float(self.na, "na", low=0.0, high=1.0)
        )
        object.__setattr__(
            self,
            "period_um",
            _as_float(self.period_um, "period_um", low=0.0),
        )
        if self.shear_ratio is not None:
            # zero is excluded as well as one: ``carrier_f0 = m / (2 s)``
            # divides by it, and zero shear carries no differential information.
            object.__setattr__(
                self,
                "shear_ratio",
                _as_float(
                    self.shear_ratio,
                    "shear_ratio",
                    low=0.0,
                    high=1.0,
                    inclusive_high=False,
                ),
            )
        # ``lsq_phase_shift`` fits B + C cos d + S sin d, which needs three
        # frames; a smaller value here used to pass and fail only later.
        object.__setattr__(
            self,
            "phase_steps",
            _as_int(self.phase_steps, "phase_steps", minimum=3),
        )
        object.__setattr__(
            self,
            "talbot_number",
            _as_int(self.talbot_number, "talbot_number", minimum=1),
        )
        if not isinstance(self.grid, Grid):
            raise ValueError(f"grid must be a Grid, got {type(self.grid).__name__}")

    # ---------------------------------------------------------------- shear
    @property
    def shear_physical(self) -> float:
        """Shear ratio from first principles (dissertation eq. 4-5).

        ``S_r = lambda / (2 NA p)`` normalised to the pupil *diameter*; the
        45-degree rotation of the chessboard contributes the ``sqrt(2)``
        factor, so ``s = sqrt(2) lambda / (2 NA p)``, which reproduces the
        dissertation's 0.0731 / 0.0439.  Table 2-5 of the dissertation then
        uses this quantity directly as the displacement in unit-circle
        coordinates and this project follows that convention; note that a
        radius-normalised geometric derivation would give a displacement
        twice as large.
        """
        lam_um = self.wavelength_nm * 1e-3
        return np.sqrt(2.0) * lam_um / (2.0 * self.period_um * self.na)

    @property
    def s(self) -> float:
        """Shear ratio actually used (normalized order displacement)."""
        if self.shear_ratio is not None:
            return float(self.shear_ratio)
        return float(self.shear_physical)

    @property
    def carrier_f0(self) -> float:
        """Spatial carrier frequency (cycles per unit normalized coordinate).

        ``f0 = m / (2 s)`` with Talbot number ``m``.  This is one quarter of
        the dissertation's eq. (2-48) value ``f0 = 2 m / s`` (see
        :attr:`carrier_frequency_paper`): the full carrier would sit above
        the Nyquist frequency of the default grid, so the default is a
        deliberate sampling-driven choice and does not correspond to the
        dissertation's Talbot plane position.  The demodulation mathematics
        is identical for both carriers.
        """
        return self.talbot_number / (2.0 * self.s)

    @property
    def carrier_frequency_paper(self) -> float:
        """The dissertation's eq. (2-48) carrier ``f0 = 2 m / s``.

        Equals ``4 * carrier_f0``; usable only on grids fine enough to keep
        it (and the ``2 f0`` order-cross beat) below Nyquist.
        """
        return 2.0 * self.talbot_number / self.s

    # ------------------------------------------------------------ utility
    def describe(self) -> str:
        return (
            f"lambda={self.wavelength_nm:g} nm  NA={self.na:g}  "
            f"p={self.period_um:g} um  s={self.s:.5f}  "
            f"f0={self.carrier_f0:.3f} cyc/unit  "
            f"grid={self.grid.n}x{self.grid.n} (extent {self.grid.extent})"
        )


#: The two configurations simulated in chapter 3 of the dissertation.
PRESET_PHASE_SHIFT = SystemConfig()
PRESET_FOURIER = SystemConfig(period_um=30.0)


def preset_phase_shift() -> SystemConfig:
    return SystemConfig()


def preset_fourier() -> SystemConfig:
    return SystemConfig(period_um=30.0)


def check_direction(direction: str) -> str:
    """Validate a shear direction and return it unchanged.

    The public API accepts only ``"x"`` and ``"y"``.  The internal
    ``if direction == "x" else "y"`` chains silently treat every other string
    as ``"y"``, which hides typos at the caller.
    """
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")
    return direction


def check_difference_model(model: str) -> str:
    """Validate the differential model and return it unchanged.

    The demodulators convert the measured phase to a wavefront difference with
    ``dW = phase / factor``, where ``factor`` is ``2 pi`` only for
    ``"one_sided"`` and ``pi`` for the other two:

    =======================  =========================  ==========  ========
    model                    differential quantity      factor      dW
    =======================  =========================  ==========  ========
    ``"two_sided"``          ``Z(x+s) - Z(x-s)``        ``pi``      ``2 d``
    ``"one_sided"``          ``Z(x+s) - Z(x)``          ``2 pi``    ``d``
    ``"one_sided_doubled"``  ``2 [Z(x+s) - Z(x)]``      ``pi``      ``2 d``
    =======================  =========================  ==========  ========

    (``d`` is the one-sided difference in waves.)  The third name is the
    Zernike-level "x2" rule of the dissertation; it *is* consistent here,
    because the doubled model expects ``2 d`` and ``phase / pi`` supplies
    exactly that.  An unknown value used to fall through to ``pi`` silently,
    which is only correct for two of the three names.
    """
    if model not in ("one_sided", "two_sided", "one_sided_doubled"):
        raise ValueError(
            "difference_model must be 'one_sided', 'two_sided' or "
            f"'one_sided_doubled', got {model!r}"
        )
    return model


def check_offset_mode(mode: str) -> str:
    """Validate the constant-offset mode and return it unchanged.

    ``"estimate"`` also drives ``fit_offsets``, so a typo used to silently
    disable the estimation (or, for ``"model"``, silently skip the model
    correction).
    """
    if mode not in ("none", "model", "estimate"):
        raise ValueError(
            f"offset_mode must be 'none', 'model' or 'estimate', got {mode!r}"
        )
    return mode


def check_region_mode(mode: str) -> str:
    """Validate the shear-region mode and return it unchanged."""
    if mode not in ("analytic", "modulation"):
        raise ValueError(
            f"region_mode must be 'analytic' or 'modulation', got {mode!r}"
        )
    return mode


def check_unwrap_method(method: str) -> str:
    """Validate the unwrapping method name and return it unchanged."""
    if method not in ("seed", "poisson", "none"):
        raise ValueError(
            f"unwrap must be 'seed', 'poisson' or 'none', got {method!r}"
        )
    return method
