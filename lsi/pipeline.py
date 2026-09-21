"""End-to-end dissertation pipeline:

    I(x, y)  --(phase shift / Fourier transform)-->  dW  --(differential
    Zernike least squares)-->  W

Both routes share the same reconstruction stage; only the demodulation of
``I`` differs.  Everything is driven by a :class:`~lsi.forward.ForwardModel`
so that simulations stay consistent with the model used for reconstruction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import (
    _as_float,
    _as_int,
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

DEFAULT_INDICES = tuple(range(2, 17))


# --------------------------------------------------------------------------- #
@dataclass
class DiffPhase:
    """Demodulated differential phase data for both shear directions.

    ``phase`` is the final unwrapped phase used to form ``dW``.
    ``wrapped_phase`` is the branch-safe wrapped map after removal of the known
    grating offset; it remains offset-corrected even when ``phase`` requests
    the raw constant via ``remove_offset=False``.

    ``difference_model`` is part of the data contract: reconstruction defaults
    to it and rejects an explicitly conflicting model.  ``confidence`` is the
    route-specific signal strength used for weighted reconstruction (phase-
    shift modulation or Fourier-lobe amplitude).
    """

    dW: dict[str, np.ndarray]
    phase: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    difference_model: str
    wrapped_phase: dict[str, np.ndarray] = field(default_factory=dict)
    modulation: dict[str, np.ndarray] = field(default_factory=dict)
    amplitude: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    lobes: dict[str, LobeResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.difference_model = check_difference_model(self.difference_model)

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


def _require_connected(mask: np.ndarray, what: str) -> np.ndarray:
    """Reject independent phase gauges that the reconstruction cannot model."""
    from scipy.ndimage import label

    _, n_components = label(np.asarray(mask, dtype=bool))
    if n_components > 1:
        raise ValueError(
            f"{what} has {n_components} disconnected components. Each component "
            "has an independent phase gauge, but differential-Zernike "
            "reconstruction models only one gauge per direction; reconstruct "
            "the components separately or supply a connected mask."
        )
    return mask


def _fourier_pair_orders(
    fm: ForwardModel,
    direction: str,
    difference_model: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validate the zero/first-order beats represented by a Fourier model."""
    positive = (1, 0) if direction == "x" else (0, 1)
    negative = (-positive[0], -positive[1])
    amplitudes = dict(zip(fm.indices, fm.amplitudes))
    a0 = amplitudes.get((0, 0), 0.0j)
    a_plus = amplitudes.get(positive, 0.0j)
    a_minus = amplitudes.get(negative, 0.0j)
    beat_plus = a_plus * np.conj(a0)
    beat_minus = a0 * np.conj(a_minus)
    scale = max(abs(beat_plus), abs(beat_minus), 1.0)
    tol = 1e-12 * scale

    if abs(beat_plus) <= tol:
        raise ValueError(
            f"the {direction} +f0 lobe needs non-zero (0, 0) and "
            f"{positive} diffraction orders"
        )
    if difference_model == "two_sided":
        if abs(beat_minus) <= tol:
            raise ValueError(
                f"difference_model='two_sided' needs the symmetric {negative} "
                f"order in the {direction} +f0 lobe"
            )
        if not np.isclose(beat_plus, beat_minus, rtol=1e-10, atol=tol):
            raise ValueError(
                f"the {direction} +f0 beat coefficients are asymmetric, so "
                "their summed phase is not an exact two-sided difference"
            )
    elif abs(beat_minus) > tol:
        raise ValueError(
            f"difference_model={difference_model!r} requires the opposite "
            f"{negative} order to be optically removed; it contributes to the "
            f"same {direction} +f0 lobe in this ForwardModel"
        )
    return positive, negative


def _unwrap_phase(wrapped: np.ndarray, mask: np.ndarray, method: str, grid) -> np.ndarray:
    """Unwrap one phase map and pin its additive gauge at the pupil centre."""
    if method == "seed":
        phase = unwrap_seed_growth(wrapped, mask)
    elif method == "poisson":
        phase = unwrap_masked_poisson(wrapped, mask)
    elif method == "none":
        phase = wrapped.copy()
    ys, xs = np.nonzero(mask)
    k = int(
        np.argmin(
            (ys - grid.n / 2.0) ** 2 + (xs - grid.n / 2.0) ** 2
        )
    )
    return phase + (wrapped[ys[k], xs[k]] - phase[ys[k], xs[k]])


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
    for name, frames in (("frames_x", frames_x), ("frames_y", frames_y)):
        if frames.ndim != 3 or frames.shape[1:] != fm.shape:
            raise ValueError(
                f"{name} must have shape (N, {fm.shape[0]}, {fm.shape[1]}), "
                f"got {frames.shape}"
            )
    check_region_mode(region_mode)
    check_unwrap_method(unwrap)
    n_frames = int(frames_x.shape[0])
    if frames_y.shape[0] != n_frames:
        raise ValueError(
            "frames_x and frames_y must hold the same number of phase steps, "
            f"got {n_frames} and {frames_y.shape[0]}"
        )
    if n_steps is not None:
        n_steps = _as_int(n_steps, "n_steps", minimum=3)
        if n_steps != n_frames:
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
        circle = zero_order_center_radius(
            res_x.modulation, cfg.grid, threshold_frac=threshold_frac
        )
        masks = shear_region_masks(
            cfg.grid,
            cfg.s,
            center=(circle.cx, circle.cy),
            radius=circle.radius,
        )
        meta = {"circle": (circle.cx, circle.cy, circle.radius)}
    else:
        masks = shear_region_masks(cfg.grid, cfg.s, aperture=aperture)
        meta = {}

    for key in ("region_x", "region_y"):
        _require_nonempty(masks[key], f"the {key[7:]} shear region")

    erode_px = _as_int(erode_px, "erode_px", minimum=0)
    if erode_px:
        from scipy import ndimage

        r = erode_px
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

    for key in ("region_x", "region_y"):
        _require_connected(masks[key], f"the {key[7:]} shear region")

    dW, phase, wrapped_phase, mod, amplitude = {}, {}, {}, {}, {}
    for res, direction in ((res_x, "x"), (res_y, "y")):
        mask = masks[f"region_{direction}"]
        model_offset = fm.demodulation_offset(direction)
        # Remove the known grating phase on the unit circle *before*
        # unwrapping.  The ideal x-order offset is exactly pi, so anchoring the
        # raw wrapped phase first makes an arbitrarily small noise perturbation
        # choose between +pi and -pi and can shift the whole result by -2*pi.
        wrapped = np.angle(np.exp(1j * (res.phase - model_offset)))
        ph = _unwrap_phase(wrapped, mask, unwrap, cfg.grid)
        if not remove_offset:
            # Preserve the public "raw phase" option, but add the model offset
            # back only after unwrapping so its exact pi value cannot choose the
            # wrong wrapped branch.
            ph = ph + model_offset
        dW[direction] = ph / np.pi
        phase[direction] = ph
        wrapped_phase[direction] = wrapped
        mod[direction] = res.modulation
        amplitude[direction] = res.modulation
    return DiffPhase(
        dW=dW,
        phase=phase,
        difference_model="two_sided",
        wrapped_phase=wrapped_phase,
        mask={"x": masks["region_x"], "y": masks["region_y"]},
        modulation=mod,
        amplitude=amplitude,
        confidence=mod,
        meta={**meta, "n_steps": n_frames, "route": "phase_shift"},
    )


def demodulate_fourier(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    direction: str = "x",
    f0: float | None = None,
    threshold_frac: float = 0.5,
    difference_model: str = "two_sided",
    window_radius: float | None = None,
    remove_offset: bool = True,
    erode_px: int = 4,
    unwrap: str = "seed",
    **lobe_kwargs,
) -> tuple[np.ndarray, np.ndarray, LobeResult]:
    """Single-frame carrier demodulation for one shear direction.

    Returns ``(dW, mask, lobe)`` with the unwrapped phase converted according to
    ``difference_model`` (``phase / 2 pi`` for a one-sided difference,
    ``phase / pi`` for a two-sided one).  ``lobe.phase`` remains the wrapped
    phase after removal of the model offset.  ``lobe.unwrapped_phase`` records
    the final phase used for ``dW``: when ``remove_offset=False`` the model
    offset is added back only *after* unwrapping, avoiding the ideal
    chessboard's ``+-pi`` branch cut.

    With the model's symmetric ``+-1`` orders, the same carrier lobe contains
    both zero/first-order beats and its phase is the dissertation's two-sided
    difference.  A one-sided model is valid only when the opposite first order
    has physically been removed from ``fm.orders``.
    """
    check_direction(direction)
    check_difference_model(difference_model)
    check_unwrap_method(unwrap)
    positive, negative = _fourier_pair_orders(
        fm, direction, difference_model
    )
    threshold_frac = _as_float(
        threshold_frac,
        "threshold_frac",
        low=0.0,
        high=1.0,
        inclusive_low=True,
        inclusive_high=False,
    )
    erode_px = _as_int(erode_px, "erode_px", minimum=0)
    model_offset = fm.demodulation_offset(direction)
    lobe = demodulate_lobe(
        image, fm.config.grid, direction=direction,
        f0=fm.config.carrier_f0 if f0 is None else f0,
        phase_offset=model_offset, **lobe_kwargs,
    )
    # Support of the isolated lobe, taken from the model so that a custom pupil
    # is honored.  A two-sided reading needs all three pupils; near an edge
    # where one symmetric first order is absent, the same carrier becomes
    # one-sided and cannot be fitted with a two-sided Zernike basis.
    a, b = positive
    support = fm.order_support(0, 0) & fm.order_support(a, b)
    if difference_model == "two_sided":
        support &= fm.order_support(*negative)
    _require_nonempty(support, f"the physical {direction} shear support")
    amp = lobe.amplitude
    peak = float(np.max(amp[support]))
    mask = (amp > threshold_frac * peak) & support
    if erode_px > 0:
        # The demodulated lobe is distorted in a ring at the region border
        # (the filter kernel mixes in the dark area).  Eroding the mask by a
        # few kernel widths removes that bias, which otherwise projects onto
        # the low-order Zernike terms.
        from scipy.ndimage import binary_erosion

        mask = binary_erosion(mask, iterations=erode_px)
    _require_nonempty(mask, f"the {direction} demodulation mask")
    _require_connected(mask, f"the {direction} demodulation mask")
    phase = _unwrap_phase(lobe.phase, mask, unwrap, fm.config.grid)
    if not remove_offset:
        phase = phase + model_offset
    lobe.unwrapped_phase = phase
    factor = 2.0 * np.pi if difference_model == "one_sided" else np.pi
    dW = phase / factor
    return dW, mask, lobe


# --------------------------------------------------------------------------- #
def reconstruct(
    fm: ForwardModel,
    diff: DiffPhase,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    difference_model: str | None = None,
    offset_mode: str = "none",
    weight_by_confidence: bool = True,
    weight_by_modulation: bool | None = None,
) -> ZernikeFit:
    """Differential-Zernike least squares on demodulated data.

    ``difference_model`` defaults to the model carried by ``diff``.  An
    explicit conflicting value is rejected rather than silently fitting the
    data with the wrong differential-Zernike basis.

    ``weight_by_confidence=True`` uses the route-specific signal strength:
    phase-shift modulation or Fourier lobe amplitude.

    ``weight_by_modulation`` is the deprecated name of
    ``weight_by_confidence`` and is retained for API compatibility.

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
    if weight_by_modulation is not None:
        warnings.warn(
            "weight_by_modulation is deprecated; use weight_by_confidence",
            DeprecationWarning,
            stacklevel=2,
        )
        weight_by_confidence = bool(weight_by_modulation)
    if difference_model is None:
        difference_model = diff.difference_model
    else:
        check_difference_model(difference_model)
        if difference_model != diff.difference_model:
            raise ValueError(
                f"difference_model={difference_model!r} conflicts with "
                f"DiffPhase.difference_model={diff.difference_model!r}"
            )
    check_difference_model(difference_model)
    for direction, mask in diff.mask.items():
        _require_nonempty(mask, f"the {direction} reconstruction mask")
        _require_connected(mask, f"the {direction} reconstruction mask")
    known = None
    if offset_mode == "model":
        known = {
            d: fm.demodulation_offset(d) / (np.pi if difference_model != "one_sided" else 2 * np.pi)
            for d in ("x", "y")
        }
    wx = wy = None
    if weight_by_confidence and diff.confidence:
        wx = np.clip(np.abs(diff.confidence["x"]), 1e-6, None)
        wy = np.clip(np.abs(diff.confidence["y"]), 1e-6, None)
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
    difference_model: str = "two_sided", offset_mode: str = "none", **kwargs,
) -> tuple[ZernikeFit, DiffPhase]:
    dWx, mask_x, lobe_x = demodulate_fourier(fm, image, direction="x",
                                             difference_model=difference_model, **kwargs)
    dWy, mask_y, lobe_y = demodulate_fourier(fm, image, direction="y",
                                             difference_model=difference_model, **kwargs)
    diff = DiffPhase(
        dW={"x": dWx, "y": dWy},
        phase={
            "x": lobe_x.unwrapped_phase,
            "y": lobe_y.unwrapped_phase,
        },
        difference_model=difference_model,
        wrapped_phase={"x": lobe_x.phase, "y": lobe_y.phase},
        mask={"x": mask_x, "y": mask_y},
        amplitude={"x": lobe_x.amplitude, "y": lobe_y.amplitude},
        confidence={"x": lobe_x.amplitude, "y": lobe_y.amplitude},
        lobes={"x": lobe_x, "y": lobe_y},
        meta={"route": "fourier"},
    )
    return reconstruct(
        fm, diff, indices=indices, offset_mode=offset_mode
    ), diff
