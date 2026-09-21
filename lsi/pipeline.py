"""End-to-end dissertation pipeline:

    I(x, y)  --(phase shift / Fourier transform)-->  dW  --(differential
    Zernike least squares)-->  W

Both routes share the same reconstruction stage; only the demodulation of
``I`` differs.  Everything is driven by a :class:`~lsi.forward.ForwardModel`
so that simulations stay consistent with the model used for reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import (
    check_difference_model,
    check_direction,
    check_offset_mode,
    check_region_mode,
    check_unwrap_method,
)
from .forward import ForwardModel
from .ftmode import LobeResult, demodulate_lobe
from .phaseshift import lsq_phase_shift, shear_region_masks, zero_order_center_radius
from .reconstruct import ZernikeFit, fit_differential_zernike
from .unwrap import unwrap_masked_poisson, unwrap_seed_growth

__all__ = [
    "DiffPhase",
    "demodulate_phase_shift",
    "demodulate_fourier",
    "offset_in_dW",
    "reconstruct",
    "phase_shift_to_wavefront",
    "fourier_to_wavefront",
]

DEFAULT_INDICES = tuple(range(2, 16))


# --------------------------------------------------------------------------- #
@dataclass
class DiffPhase:
    """Demodulated differential phase data for both shear directions."""

    dW: dict[str, np.ndarray]
    phase: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    modulation: dict[str, np.ndarray] = field(default_factory=dict)
    amplitude: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    lobes: dict[str, LobeResult] = field(default_factory=dict)

    def regions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mask["x"], self.mask["y"]


def offset_in_dW(fm: ForwardModel, direction: str, difference_model: str) -> float:
    """Predicted constant offset of a difference map, in waves."""
    check_direction(direction)
    check_difference_model(difference_model)
    off = fm.demodulation_offset(direction)
    if difference_model == "one_sided":
        return off / (2.0 * np.pi)
    return off / np.pi


def _require_nonempty(mask: np.ndarray, what: str) -> np.ndarray:
    """Reject an empty demodulation region instead of returning an empty fit."""
    if not np.any(mask):
        raise ValueError(
            f"{what} is empty: the demodulation found no usable pixels. Check "
            "that the model aperture and shear match the data (a custom "
            "'pupil' that does not overlap its own shear-shifted copies gives "
            "no interference region)."
        )
    return mask


# --------------------------------------------------------------------------- #
def demodulate_phase_shift(
    fm: ForwardModel,
    frames_x: np.ndarray,
    frames_y: np.ndarray,
    *,
    n_steps: int | None = None,
    deltas_x: Sequence[float] | None = None,
    deltas_y: Sequence[float] | None = None,
    region_mode: str = "analytic",
    unwrap: str = "seed",
    threshold_frac: float = 0.4,
    remove_offset: bool = True,
    erode_px: int = 0,
) -> DiffPhase:
    """N-step phase-shift demodulation + shear-region extraction.

    ``region_mode="analytic"``
        shear regions from the known pupil (0-order disc intersected with the
        +-1 order discs, i.e. the dissertation's "circle offset by the shear"
        construction).
    ``region_mode="modulation"``
        the dissertation's experimental procedure: threshold the modulation,
        take the outermost edge of the zero-order disc, least-squares fit a
        circle (eqs. 3-1 ... 3-5) and offset that circle by the shear.

    ``n_steps`` is a cross-check against the data: it must equal the number of
    frames supplied, and the metadata records the number of frames actually
    demodulated.  ``deltas_x`` / ``deltas_y`` pass calibrated phase steps to
    the least-squares demodulation (default ``2*pi*i/N``); supplying the
    measured steps is the only way to keep a phase-step calibration error out
    of the demodulated phase.  ``erode_px`` shrinks both shear regions by that
    many pixels before unwrapping, which rejects the boundary pixels where
    the demodulated phase is unreliable.
    """
    cfg = fm.config
    frames_x = np.asarray(frames_x, dtype=float)
    frames_y = np.asarray(frames_y, dtype=float)
    check_region_mode(region_mode)
    check_unwrap_method(unwrap)
    n_frames = int(frames_x.shape[0])
    if frames_y.shape[0] != n_frames:
        raise ValueError(
            "frames_x and frames_y must hold the same number of phase steps, "
            f"got {n_frames} and {frames_y.shape[0]}"
        )
    if n_steps is not None and int(n_steps) != n_frames:
        raise ValueError(
            f"n_steps={n_steps} does not match the {n_frames} frames supplied"
        )
    for name, d in (("deltas_x", deltas_x), ("deltas_y", deltas_y)):
        if d is not None and len(d) != n_frames:
            raise ValueError(
                f"{name} holds {len(d)} steps but {n_frames} frames were "
                "supplied"
            )
    res_x = lsq_phase_shift(frames_x, deltas_x)
    res_y = lsq_phase_shift(frames_y, deltas_y)

    # The regions have to follow the exit pupil actually used by the forward
    # model.  Without a custom pupil the analytic unit-disk construction below
    # is exact and sub-pixel accurate, so it stays the default; with one, the
    # shifted pupils must be taken from the model or the regions would claim
    # signal outside the aperture.  The *definition* is passed through, not the
    # sampled mask, so a callable pupil keeps its exact edges here too.
    aperture = fm.pupil_definition
    if region_mode == "modulation":
        if aperture is not None:
            raise ValueError(
                "region_mode='modulation' fits a single circle, so it cannot "
                "describe a non-circular custom 'pupil'; use "
                "region_mode='analytic' with a custom pupil"
            )
        _, _, edge = zero_order_center_radius(res_x.modulation, cfg.grid, threshold_frac=threshold_frac)
        x, y = cfg.grid.coords()
        from .phaseshift import circle_fit

        cx, cy, r = circle_fit(x[edge], y[edge])
        masks = shear_region_masks(cfg.grid, cfg.s, center=(cx, cy), radius=r)
        meta = {"circle": (cx, cy, r)}
    else:
        masks = shear_region_masks(cfg.grid, cfg.s, aperture=aperture)
        meta = {}

    for key in ("region_x", "region_y"):
        _require_nonempty(masks[key], f"the {key[7:]} shear region")

    if erode_px:
        from scipy import ndimage

        r = int(erode_px)
        if r < 1:
            raise ValueError("erode_px must be a positive number of pixels")
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        structure = (xx * xx + yy * yy) <= r * r
        for key in ("region_x", "region_y"):
            eroded = ndimage.binary_erosion(masks[key], structure=structure)
            if not eroded.any():
                raise ValueError(
                    f"erode_px={r} erodes the whole {key} shear region away"
                )
            masks[key] = eroded
        meta["erode_px"] = r

    dW, phase, mod, amplitude = {}, {}, {}, {}
    for res, direction in ((res_x, "x"), (res_y, "y")):
        mask = masks[f"region_{direction}"]
        model_offset = fm.demodulation_offset(direction)
        # Remove the known grating phase on the unit circle *before*
        # unwrapping.  The ideal x-order offset is exactly pi, so anchoring the
        # raw wrapped phase first makes an arbitrarily small noise perturbation
        # choose between +pi and -pi and can shift the whole result by -2*pi.
        wrapped = np.angle(np.exp(1j * (res.phase - model_offset)))
        if unwrap == "seed":
            ph = unwrap_seed_growth(wrapped, mask)
        elif unwrap == "poisson":
            ph = unwrap_masked_poisson(wrapped, mask)
        elif unwrap == "none":
            ph = wrapped.copy()
        # Gauge convention: every unwrapping method is pinned to the wrapped
        # value at the seed pixel (the mask pixel closest to the pupil
        # centre).  Path following does this by construction; the two
        # least-squares solvers fix an arbitrary constant instead (zero mean
        # over the region) and need it imposed.  Without this, the phase
        # carries an extra multiple of 2 pi, which the differential-Zernike
        # fit can only absorb as a tilt.
        ys, xs = np.nonzero(mask)
        kk = int(np.argmin((ys - cfg.grid.n / 2.0) ** 2 + (xs - cfg.grid.n / 2.0) ** 2))
        ph = ph + (wrapped[ys[kk], xs[kk]] - ph[ys[kk], xs[kk]])
        if not remove_offset:
            # Preserve the public "raw phase" option, but add the model offset
            # back only after unwrapping so its exact pi value cannot choose the
            # wrong wrapped branch.
            ph = ph + model_offset
        dW[direction] = ph / np.pi
        phase[direction] = ph
        mod[direction] = res.modulation
        amplitude[direction] = res.modulation
    return DiffPhase(
        dW=dW,
        phase=phase,
        mask={"x": masks["region_x"], "y": masks["region_y"]},
        modulation=mod,
        amplitude=amplitude,
        meta={**meta, "n_steps": n_frames, "route": "phase_shift"},
    )


def demodulate_fourier(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    direction: str = "x",
    f0: float | None = None,
    threshold_frac: float = 0.5,
    difference_model: str = "one_sided",
    window_radius: float | None = None,
    remove_offset: bool = True,
    erode_px: int = 4,
    **lobe_kwargs,
) -> tuple[np.ndarray, np.ndarray, LobeResult]:
    """Single-frame carrier demodulation for one shear direction.

    Returns ``(dW, mask, lobe)`` with the phase converted according to
    ``difference_model`` (``phase / 2 pi`` for a one-sided difference,
    ``phase / pi`` for a two-sided one).
    """
    check_direction(direction)
    check_difference_model(difference_model)
    offset = fm.demodulation_offset(direction) if remove_offset else 0.0
    lobe = demodulate_lobe(
        image, fm.config.grid, direction=direction,
        f0=fm.config.carrier_f0 if f0 is None else f0,
        phase_offset=offset, **lobe_kwargs,
    )
    # support of the isolated lobe: intersection of the 0 and +-1 pupils, taken
    # from the model so that a custom 'pupil' is honored (a hard-coded unit
    # circle would claim signal the aperture never passed).
    a, b = (1, 0) if direction == "x" else (0, 1)
    support = fm.order_support(0, 0) & fm.order_support(a, b)
    amp = lobe.amplitude
    mask = (amp > threshold_frac * amp.max()) & support
    if erode_px > 0:
        # The demodulated lobe is distorted in a ring at the region border
        # (the filter kernel mixes in the dark area).  Eroding the mask by a
        # few kernel widths removes that bias, which otherwise projects onto
        # the low-order Zernike terms.
        from scipy.ndimage import binary_erosion

        mask = binary_erosion(mask, iterations=int(erode_px))
    _require_nonempty(mask, f"the {direction} demodulation mask")
    factor = 2.0 * np.pi if difference_model == "one_sided" else np.pi
    dW = lobe.phase / factor
    return dW, mask, lobe


# --------------------------------------------------------------------------- #
def reconstruct(
    fm: ForwardModel,
    diff: DiffPhase,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    difference_model: str = "two_sided",
    offset_mode: str = "none",
    weight_by_modulation: bool = True,
) -> ZernikeFit:
    """Differential-Zernike least squares on demodulated data.

    ``offset_mode``
        ``"none"``     default: the demodulator already removed the
                       model-predicted half-fringe constant,
        ``"model"``    subtract the grating-model prediction here (use when
                       the demodulation was run with ``remove_offset=False``),
        ``"estimate"`` estimate one free constant per direction -- only useful
                       when the beam amplitudes are unknown; note that such a
                       constant is collinear with the tilt column, so the tilt
                       coefficients lose their meaning (see
                       ``fit_differential_zernike``).
    """
    x, y = fm.config.grid.coords()
    check_offset_mode(offset_mode)
    check_difference_model(difference_model)
    known = None
    if offset_mode == "model":
        known = {
            d: fm.demodulation_offset(d) / (np.pi if difference_model != "one_sided" else 2 * np.pi)
            for d in ("x", "y")
        }
    wx = wy = None
    if weight_by_modulation and diff.modulation:
        wx = np.clip(np.abs(diff.modulation["x"]), 1e-6, None)
        wy = np.clip(np.abs(diff.modulation["y"]), 1e-6, None)
    return fit_differential_zernike(
        diff.dW["x"], diff.dW["y"], diff.mask["x"], diff.mask["y"],
        fm.s, x, y, indices=indices,
        fit_offsets=(offset_mode == "estimate"),
        known_offsets=known,
        weights_x=wx, weights_y=wy,
        difference_model=difference_model,
    )


def phase_shift_to_wavefront(
    fm: ForwardModel, frames_x, frames_y, *, indices=DEFAULT_INDICES,
    offset_mode: str = "none", **kwargs,
) -> tuple[ZernikeFit, DiffPhase]:
    diff = demodulate_phase_shift(fm, frames_x, frames_y, **kwargs)
    return reconstruct(fm, diff, indices=indices, offset_mode=offset_mode), diff


def fourier_to_wavefront(
    fm: ForwardModel, image: np.ndarray, *, indices=DEFAULT_INDICES,
    difference_model: str = "one_sided", offset_mode: str = "none", **kwargs,
) -> tuple[ZernikeFit, DiffPhase]:
    dWx, mask_x, lobe_x = demodulate_fourier(fm, image, direction="x",
                                             difference_model=difference_model, **kwargs)
    dWy, mask_y, lobe_y = demodulate_fourier(fm, image, direction="y",
                                             difference_model=difference_model, **kwargs)
    diff = DiffPhase(
        dW={"x": dWx, "y": dWy},
        phase={"x": lobe_x.phase, "y": lobe_y.phase},
        mask={"x": mask_x, "y": mask_y},
        amplitude={"x": lobe_x.amplitude, "y": lobe_y.amplitude},
        lobes={"x": lobe_x, "y": lobe_y},
        meta={"route": "fourier", "difference_model": difference_model},
    )
    return reconstruct(fm, diff, indices=indices, difference_model=difference_model,
                       offset_mode=offset_mode), diff
