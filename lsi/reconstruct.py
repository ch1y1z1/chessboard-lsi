"""Differential-Zernike least-squares wavefront reconstruction.

Dissertation eqs. (2-26) ... (2-31): the measured differential wavefronts

    dW_x = W(x+s, y) - W(x-s, y),     dW_y = W(x, y+s) - W(x, y-s)

are written as linear combinations of *differential* Zernike polynomials

    dW = Zmatrix * c ,     c = (Z^T Z)^-1 Z^T dW

with the differential Zernike basis of 表 2-5.

Two refinements that make the estimator exact for a physically faithful
forward model:

``fit_offsets``
    The real amplitude chessboard has ``A_00 > 0`` while ``A_10 < 0`` and
    ``A_01 > 0``, so the demodulated phase of the x-pair and the y-pair carry
    *different* constant offsets (a half fringe).  Those constants are
    unobservable nuisance parameters; estimating one per difference map keeps
    the Zernike coefficients unbiased.

``weights``
    Optional per-pixel weights (e.g. modulation) for weighted least squares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import warnings

from .config import _as_int_array, check_difference_model
from .zernike import differential_zernike_matrix

__all__ = ["ZernikeFit", "fit_differential_zernike", "wavefront_on_grid"]


@dataclass
class ZernikeFit:
    indices: np.ndarray
    coeffs: np.ndarray            # fitted coefficients (waves), offsets excluded
    offsets: dict[str, float] = field(default_factory=dict)
    residual: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_residual: float = 0.0
    max_abs_residual: float = 0.0
    cond: float = np.nan
    rank: int = 0
    singular_values: np.ndarray = field(default_factory=lambda: np.array([]))
    n_rows: int = 0

    def wavefront(self, x, y, pupil=None) -> np.ndarray:
        return wavefront_on_grid(self.coeffs, self.indices, x, y, pupil)

    def as_dict(self) -> dict:
        return {int(j): float(c) for j, c in zip(self.indices, self.coeffs)}


def _build_system(
    dW: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    shear: float,
    indices: Sequence[int],
    fit_offsets: bool,
    weights: dict[str, np.ndarray] | None,
    x: np.ndarray,
    y: np.ndarray,
    difference_model: str = "two_sided",
):
    rows, rhs, wts, tags = [], [], [], []
    for direction in ("x", "y"):
        d = dW.get(direction)
        if d is None:
            continue
        mask = masks[direction] & np.isfinite(d)
        if not np.any(mask):
            continue
        Z = differential_zernike_matrix(
            indices, x[mask], y[mask], shear, direction, difference_model
        )
        rows.append(Z)
        rhs.append(d[mask])
        if weights is not None and direction in weights:
            w = np.sqrt(np.clip(weights[direction][mask], 0.0, None))
        else:
            w = np.ones(mask.sum())
        wts.append(w)
        tags.append(direction)
    if not rows:
        raise ValueError("no valid data for the differential fit")
    n_off = len(tags) if fit_offsets else 0
    if fit_offsets:
        # One constant offset *per direction*: every direction block carries
        # its own column, zero in the other blocks.  A single shared column
        # cannot represent two independent direction-dependent biases.
        blocks = []
        for k, Z in enumerate(rows):
            off = np.zeros((Z.shape[0], n_off))
            off[:, k] = 1.0
            blocks.append(np.hstack([Z, off]))
        rows = blocks
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    w = np.concatenate(wts)
    return A, b, w, n_off, tags


def fit_differential_zernike(
    dW_x: np.ndarray | None,
    dW_y: np.ndarray | None,
    mask_x: np.ndarray | None,
    mask_y: np.ndarray | None,
    shear: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    indices: Sequence[int] = tuple(range(2, 17)),
    fit_offsets: bool = False,
    known_offsets: dict[str, float] | None = None,
    weights_x: np.ndarray | None = None,
    weights_y: np.ndarray | None = None,
    difference_model: str = "two_sided",
) -> ZernikeFit:
    """Least-squares differential-Zernike reconstruction (eqs. 2-28 ... 2-31).

    ``known_offsets`` (keys ``"x"``/``"y"``, in the same units as ``dW``, i.e.
    waves) are subtracted from the data before the fit; use
    ``ForwardModel.demodulation_offset(direction)/pi`` for the physical
    chessboard.  ``fit_offsets`` instead *estimates* one constant per
    direction -- but such a constant is exactly collinear with the tilt
    column (``dZx(j=2) = 2s`` is constant), so with ``fit_offsets=True`` the
    tilt coefficients become meaningless unless the beam amplitudes are
    genuinely unknown.
    """
    check_difference_model(difference_model)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError(
            f"x and y must have the same non-empty shape, got {x.shape} and {y.shape}"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    if not np.isfinite(shear) or shear <= 0.0:
        raise ValueError(f"shear must be positive and finite, got {shear!r}")

    indices = _as_int_array(list(indices), "indices", ndim=1, minimum=1)
    if indices.size == 0:
        raise ValueError("indices must contain at least one fitted mode")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("indices must not contain duplicates")
    if 1 in indices:
        raise ValueError(
            "piston Z1 has zero differential signal; remove it from indices"
        )

    def data_array(value, name, *, boolean=False):
        if value is None:
            return None
        array = np.asarray(value, dtype=bool if boolean else float)
        if array.shape != x.shape:
            raise ValueError(f"{name} must have shape {x.shape}, got {array.shape}")
        return array

    dW_x = data_array(dW_x, "dW_x")
    dW_y = data_array(dW_y, "dW_y")
    mask_x = data_array(mask_x, "mask_x", boolean=True)
    mask_y = data_array(mask_y, "mask_y", boolean=True)
    weights_x = data_array(weights_x, "weights_x")
    weights_y = data_array(weights_y, "weights_y")
    for name, weight in (("weights_x", weights_x), ("weights_y", weights_y)):
        if weight is not None and (
            not np.all(np.isfinite(weight)) or np.any(weight < 0.0)
        ):
            raise ValueError(f"{name} must contain finite non-negative values")
    if known_offsets is not None:
        unknown = set(known_offsets) - {"x", "y"}
        if unknown:
            raise ValueError(f"unknown known_offsets directions: {sorted(unknown)}")
        if not all(np.isfinite(value) for value in known_offsets.values()):
            raise ValueError("known_offsets values must be finite")

    dW, masks = {}, {}
    if dW_x is not None:
        dW["x"] = dW_x
        masks["x"] = (
            np.ones_like(dW_x, dtype=bool) if mask_x is None else mask_x
        )
    if dW_y is not None:
        dW["y"] = dW_y
        masks["y"] = (
            np.ones_like(dW_y, dtype=bool) if mask_y is None else mask_y
        )
    if known_offsets:
        for direction in ("x", "y"):
            if direction in known_offsets and direction in dW:
                dW[direction] = dW[direction] - known_offsets[direction]
    weights = {}
    if weights_x is not None:
        weights["x"] = weights_x
    if weights_y is not None:
        weights["y"] = weights_y

    A, b, w, n_off, tags = _build_system(
        dW, masks, shear, indices, fit_offsets, weights or None, x, y,
        difference_model,
    )
    Aw = A * w[:, None]
    bw = b * w
    if not np.any(w > 0.0):
        raise ValueError("all effective reconstruction weights are zero")
    sol, _, rank, singular_values = np.linalg.lstsq(Aw, bw, rcond=None)
    if rank < Aw.shape[1]:
        warnings.warn(
            "the weighted differential design matrix is rank deficient: "
            f"rank={rank}, parameters={Aw.shape[1]}; the reported coefficients "
            "are a minimum-norm solution and are not all identifiable",
            stacklevel=2,
        )
    coeffs = sol[: len(indices)]
    offsets = {}
    if n_off:
        off_vals = sol[len(indices) :]
        for tag, val in zip(tags, off_vals):
            offsets[tag] = float(val)

    resid = Aw @ sol - bw
    cond = float(np.linalg.cond(Aw))
    return ZernikeFit(
        indices=indices,
        coeffs=coeffs,
        offsets=offsets,
        residual=resid,
        rms_residual=float(np.sqrt(np.mean(resid**2))),
        max_abs_residual=float(np.max(np.abs(resid))),
        cond=cond,
        rank=int(rank),
        singular_values=singular_values,
        n_rows=int(Aw.shape[0]),
    )


def wavefront_on_grid(
    coeffs: Sequence[float],
    indices: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    pupil: np.ndarray | None = None,
) -> np.ndarray:
    from .zernike import as_wavefront

    W = as_wavefront(coeffs, x, y, indices=indices)
    if pupil is not None:
        W = np.where(pupil, W, np.nan)
    return W