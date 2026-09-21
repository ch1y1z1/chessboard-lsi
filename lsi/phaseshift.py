"""Phase retrieval from intensity frames: phase-shift and Fourier methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import warnings

__all__ = [
    "PhaseShiftResult",
    "lsq_phase_shift",
    "extract_psd_phase",
    "modulation",
    "circle_fit",
    "shear_region_masks",
    "zero_order_center_radius",
]


# --------------------------------------------------------------------------- #
# N-step least-squares phase shift (dissertation eq. 2-20, eq. 3-1)
# --------------------------------------------------------------------------- #
@dataclass
class PhaseShiftResult:
    phase: np.ndarray          # wrapped phase (rad), in [-pi, pi)
    modulation: np.ndarray     # modulation amplitude (eq. 3-1)
    background: np.ndarray     # DC term
    frames: int


def lsq_phase_shift(
    frames: np.ndarray, deltas: np.ndarray | None = None
) -> PhaseShiftResult:
    """N-step least-squares phase-shift demodulation.

    Fits ``I_i = B + C cos(delta_i) + S sin(delta_i)`` per pixel with linear
    least squares, which is exact for arbitrary phase steps:

        psi = atan2(-S, C)      M = sqrt(C^2 + S^2)      B = intercept

    For the default uniform steps ``deltas = 2*pi*i/N`` the cos/sin basis is
    orthogonal, so this collapses to the classic closed form

        psi = atan2( -sum_i I_i sin delta_i , sum_i I_i cos delta_i )
        M   = (2/N) sqrt( (sum_i I_i cos delta_i)^2
                         + (sum_i I_i sin delta_i)^2 )
        B   = (1/N) sum_i I_i

    (dissertation eq. 2-20).  That closed form is *not* the least-squares
    solution once the steps are non-uniform or non-orthogonal, which is why
    the fit above is used instead: pass calibrated ``deltas`` to remove a
    phase-step calibration error from the demodulation.

    ``deltas`` must hold one step per frame (default ``2*pi*i/N``).
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim < 1:
        raise ValueError("frames must have at least one dimension")
    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain only finite values")
    n = frames.shape[0]
    if n < 3:
        raise ValueError(
            f"phase-shift demodulation needs at least 3 frames, got {n}"
        )
    if deltas is None:
        deltas = 2.0 * np.pi * np.arange(n) / n
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (n,):
        raise ValueError(f"deltas must have shape ({n},), got {deltas.shape}")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must be finite")

    design = np.column_stack([np.ones(n), np.cos(deltas), np.sin(deltas)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(
            "deltas do not span the offset/cos/sin basis, so the phase is not "
            "determined (e.g. repeated steps, or steps differing only by pi)"
        )
    coef = np.linalg.pinv(design) @ frames.reshape(n, -1)
    shape = frames.shape[1:]
    background = coef[0].reshape(shape)
    c_cos = coef[1].reshape(shape)
    s_sin = coef[2].reshape(shape)
    phase = np.arctan2(-s_sin, c_cos)
    mod = np.hypot(c_cos, s_sin)
    return PhaseShiftResult(
        phase=phase, modulation=mod, background=background, frames=n
    )


# --------------------------------------------------------------------------- #
# modulation (dissertation eq. 3-1) and circle fitting (eq. 3-2 ... 3-5)
# --------------------------------------------------------------------------- #
def modulation(frames: np.ndarray) -> np.ndarray:
    """Modulation degree map: ``(2/N) * |sum_i I_i exp(-i delta_i)|``."""
    return lsq_phase_shift(frames).modulation


def zero_order_center_radius(
    mod: np.ndarray,
    grid,
    *,
    threshold_frac: float = 0.4,
) -> tuple[float, float, np.ndarray]:
    """Centre/radius of the zero-order pupil from the modulation map.

    Steps (dissertation 3.1.1): threshold the modulation at
    ``threshold_frac * max``, keep the *outermost* edge points of the
    connected zero-order disc, then least-squares fit a circle.

    Returns ``(cx, cy, edge_mask)``.
    """
    from scipy import ndimage

    mask = mod > threshold_frac * mod.max()
    label, n_lab = ndimage.label(mask)
    if n_lab == 0:
        return 0.0, 0.0, mask
    sizes = ndimage.sum(mask, label, index=np.arange(1, n_lab + 1))
    k = int(np.argmax(sizes)) + 1
    blob = label == k
    # fill holes so the "outermost edge" is the pupil boundary
    blob = ndimage.binary_fill_holes(blob)
    edge = blob & ~ndimage.binary_erosion(blob)
    x, y = grid.coords()
    cx, cy, r = circle_fit(x[edge], y[edge])
    return cx, cy, edge


def circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = np.sqrt(sol[2] + cx**2 + cy**2)
    return float(cx), float(cy), float(r)


def _shift_mask(mask: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    """``out[i, j] = mask[i - drow, j - dcol]``, zero outside the grid."""
    out = np.zeros_like(mask)
    n0, n1 = mask.shape
    r0, r1 = max(drow, 0), min(n0 + drow, n0)
    c0, c1 = max(dcol, 0), min(n1 + dcol, n1)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = mask[r0 - drow : r1 - drow, c0 - dcol : c1 - dcol]
    return out


def shear_region_masks(
    grid,
    shear: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 1.0,
    aperture: np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """The four shear-interference regions.

    Following section 2.3.1 and figure 2-8, the "x" shear region is the
    overlap of the zero-order pupil with the pupils of the two x-shifted
    orders, i.e. the intersection of the circles centred at ``0``, ``+s x``
    and ``-s x``; likewise for "y".

    With ``aperture=None`` (default) the zero-order pupil is the unit disk,
    built analytically from ``center``/``radius``.  Passing a boolean
    ``aperture`` instead -- the same array
    :meth:`~lsi.forward.ForwardModel.aperture` returns -- makes the regions
    follow the *actual* exit pupil of the forward model rather than a hard
    unit circle, which matters as soon as the model is given a custom
    ``pupil``.  The shifted pupils are then translated by the shear with
    nearest-neighbour snapping, the same convention
    :meth:`~lsi.forward.ForwardModel._pupil_mask` uses, so the regions agree
    pixel for pixel with :meth:`~lsi.forward.ForwardModel.order_support`; a
    warning is issued when ``shear`` is not an integer number of pixels, as it
    also is once per model by
    :meth:`~lsi.forward.ForwardModel._prepare_pupil`.  ``center``/``radius``
    describe the fitted circle of the ``"modulation"`` region mode and cannot
    be combined with an explicit aperture.

    A *callable* ``f(x, y) -> mask`` is evaluated at the shifted coordinates
    instead, exactly as :meth:`~lsi.forward.ForwardModel._pupil_mask` does, so a
    non-circular aperture keeps exact sub-pixel edges and no warning applies.
    ``ForwardModel.pupil_definition`` returns whichever form was supplied.
    """
    x, y = grid.coords()

    if aperture is not None and (
        tuple(center) != (0.0, 0.0) or radius != 1.0
    ):
        raise ValueError(
            "center/radius describe the fitted-circle ('modulation') "
            "construction and cannot be combined with an explicit aperture"
        )

    if aperture is None:
        cx, cy = center

        def pupil(dx, dy):
            return (x - cx - dx) ** 2 + (y - cy - dy) ** 2 <= radius**2
    elif callable(aperture):
        def pupil(dx, dy):
            # Same convention as the model: the shifted pupil is the aperture
            # sampled at the coordinates the order reads from.
            out = np.asarray(aperture(x - dx, y - dy))
            if out.shape != grid.shape:
                raise ValueError(
                    f"aperture callable must return an array with the shape of "
                    f"the grid {grid.shape}, got {out.shape}"
                )
            return out if out.dtype == bool else out > 0.5
    else:
        # Same convention as ``ForwardModel._prepare_pupil``: a boolean array is
        # taken as given, anything else is a hard aperture thresholded at 0.5.
        ap = np.asarray(aperture)
        if ap.shape != grid.shape:
            raise ValueError(
                f"aperture must have shape {grid.shape}, got {ap.shape}"
            )
        ap = ap if ap.dtype == bool else ap > 0.5
        px = shear / grid.dx
        if not np.isclose(px, np.rint(px)):
            warnings.warn(
                f"shear={shear:g} is {px:.3f} pixels, not an integer: the "
                "shifted apertures are snapped to the grid, so the region "
                "edges carry a sub-pixel error",
                stacklevel=2,
            )

        def pupil(dx, dy):
            # Forward model convention: order (a, b) samples the exit pupil at
            # ``(x + a*s, y + b*s)``, i.e. ``ap[i + a*px]`` in index space, so
            # the shift is *negative* in the ``_shift_mask`` convention
            # (``out[i, j] = ap[i - drow, j - dcol]``).  The sign does not
            # change the returned intersections -- ``zero & P(+s) & P(-s)`` is
            # symmetric in ``s`` -- but it keeps this helper consistent with
            # ``ForwardModel._pupil_mask``.
            return _shift_mask(
                ap, -int(np.rint(dy / grid.dx)), -int(np.rint(dx / grid.dx))
            )

    zero = pupil(0.0, 0.0)
    mx = zero & pupil(shear, 0.0) & pupil(-shear, 0.0) & pupil(0.0, shear) & pupil(0.0, -shear)
    region_x = zero & pupil(shear, 0.0) & pupil(-shear, 0.0)
    region_y = zero & pupil(0.0, shear) & pupil(0.0, -shear)
    return {
        "zero": zero,
        "region_x": region_x,
        "region_y": region_y,
        "center5": mx,
        "region_x_only": region_x & ~mx,
        "region_y_only": region_y & ~mx,
    }


def extract_psd_phase(
    frames: np.ndarray,
    grid,
    config,
    *,
    direction: str = "x",
    threshold_frac: float = 0.4,
    region: np.ndarray | None = None,
) -> tuple[PhaseShiftResult, np.ndarray]:
    """Demodulate one phase-shift stack and return phase + shear-region mask."""
    res = lsq_phase_shift(frames)
    if region is None:
        mode = zero_order_center_radius(res.modulation, grid, threshold_frac=threshold_frac)
        masks = shear_region_masks(grid, config.s)
        region = masks[f"region_{direction}"]
    return res, region
