"""06 - invert a wavefront that was not generated from Zernike coefficients.

The freeform truth below is an analytic surface made from two localized Gaussian
features, a broad smooth feature, and a mid-spatial-frequency ripple.  Zernike
polynomials are used only after the detector frames have been generated:

1. project the sampled truth onto Z1...Z36 to establish the best possible
   36-mode wave-domain approximation (the truncation floor);
2. fit Z2...Z36 directly from 8+8 intensity frames with LM;
3. compare the LM wavefront both with the original freeform truth and with the
   best 36-mode projection.

Z1 is included in the oracle projection but not in LM: piston is unobservable
from intensity.  Every map comparison therefore removes the mean difference
inside the pupil.

Run:  python3 scripts/06_freeform_wavefront.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.lm import LMConfig, fit_wavefront_from_frames
from lsi.pipeline import fourier_to_wavefront, phase_shift_to_wavefront
from lsi.plotting import imshow, new_fig, savefig
from lsi.zernike import as_wavefront, zernike_matrix

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)

ALL_INDICES = np.arange(1, 37)
OBSERVABLE_INDICES = np.arange(2, 37)


class FreeformWavefront:
    """Smooth synthetic surface defined independently of the Zernike basis."""

    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        local_positive = 0.20 * np.exp(
            -((x - 0.28) ** 2 / 0.18**2 + (y + 0.12) ** 2 / 0.24**2)
        )
        local_negative = -0.15 * np.exp(
            -((x + 0.32) ** 2 / 0.25**2 + (y - 0.24) ** 2 / 0.16**2)
        )
        broad = 0.06 * np.exp(
            -((x + 0.05) ** 2 + (y + 0.10) ** 2) / 0.65**2
        )
        ripple = (
            0.025
            * np.sin(5.0 * np.pi * x + 0.4)
            * np.cos(4.0 * np.pi * y - 0.2)
        )
        return local_positive + local_negative + broad + ripple


def align_piston(estimate: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    """Align an intensity-unobservable piston to ``reference``."""
    return estimate - np.mean((estimate - reference)[mask])


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[mask] ** 2)))


def max_abs(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.abs(values[mask])))


def pv(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.ptp(values[mask]))


cfg = SystemConfig(grid=Grid(n=128, extent=1.10), period_um=30.0)
fm = ForwardModel(cfg)
truth = FreeformWavefront()
x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_true = truth.w(x, y)
W_true_zero_piston = align_piston(W_true, np.zeros_like(W_true), pupil)

# Independent wave-domain oracle: the least-squares Z1...Z36 projection of the
# already-created freeform surface.  This is not used to generate the frames.
Z36 = zernike_matrix(ALL_INDICES, x[pupil], y[pupil])
oracle_coeffs, *_ = np.linalg.lstsq(Z36, W_true[pupil], rcond=None)
W_oracle = as_wavefront(oracle_coeffs, x, y, indices=ALL_INDICES)
W_oracle_aligned = align_piston(W_oracle, W_true, pupil)
projection_error = W_oracle_aligned - W_true

# Generate measurements from the original freeform object, not from W_oracle.
frames_x = fm.phase_shift_frames(truth, "x", 8)
frames_y = fm.phase_shift_frames(truth, "y", 8)
frames = np.concatenate([np.asarray(frames_x), np.asarray(frames_y)], axis=0)
deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]

prototype = ZernikeWavefront(
    np.zeros(len(OBSERVABLE_INDICES)),
    OBSERVABLE_INDICES,
)
result = fit_wavefront_from_frames(
    fm,
    prototype,
    frames,
    deltas,
    samples=6000,
    config=LMConfig(max_iter=100),
)
W_lm = as_wavefront(result.x, x, y, indices=OBSERVABLE_INDICES)
W_lm_aligned = align_piston(W_lm, W_true, pupil)
W_lm_to_oracle = align_piston(W_lm, W_oracle, pupil)
lm_total_error = W_lm_aligned - W_true
lm_algorithmic_error = W_lm_to_oracle - W_oracle

# Apply both dissertation pipelines to the same freeform truth, with the same
# Z2...Z36 reconstruction basis used by LM.
phase_fit, _ = phase_shift_to_wavefront(
    fm,
    frames_x,
    frames_y,
    indices=OBSERVABLE_INDICES,
)
W_phase = as_wavefront(
    phase_fit.coeffs, x, y, indices=OBSERVABLE_INDICES
)
W_phase_aligned = align_piston(W_phase, W_true, pupil)
W_phase_to_oracle = align_piston(W_phase, W_oracle, pupil)
phase_total_error = W_phase_aligned - W_true
phase_algorithmic_error = W_phase_to_oracle - W_oracle

carrier_frame = fm.ft_mode_frame(truth)
fourier_fit, _ = fourier_to_wavefront(
    fm,
    carrier_frame,
    indices=OBSERVABLE_INDICES,
)
W_fourier = as_wavefront(
    fourier_fit.coeffs, x, y, indices=OBSERVABLE_INDICES
)
W_fourier_aligned = align_piston(W_fourier, W_true, pupil)
W_fourier_to_oracle = align_piston(W_fourier, W_oracle, pupil)
fourier_total_error = W_fourier_aligned - W_true
fourier_algorithmic_error = W_fourier_to_oracle - W_oracle

# Control: when the data really lie in the same 36-mode subspace, the same LM
# implementation should recover the observable oracle coefficients exactly.
projected_truth = ZernikeWavefront(oracle_coeffs[1:], OBSERVABLE_INDICES)
control_x = fm.phase_shift_frames(projected_truth, "x", 8)
control_y = fm.phase_shift_frames(projected_truth, "y", 8)
control_result = fit_wavefront_from_frames(
    fm,
    prototype,
    np.concatenate([np.asarray(control_x), np.asarray(control_y)], axis=0),
    deltas,
    samples=6000,
    config=LMConfig(max_iter=100),
)

ranked_modes = sorted(
    (
        {
            "index": int(j),
            "lm_coefficient": float(c_lm),
            "oracle_coefficient": float(c_oracle),
            "difference": float(c_lm - c_oracle),
        }
        for j, c_lm, c_oracle in zip(
            OBSERVABLE_INDICES, result.x, oracle_coeffs[1:]
        )
    ),
    key=lambda item: abs(item["lm_coefficient"]),
    reverse=True,
)

REPORT = {
    "wavefront_source": (
        "two localized Gaussians + one broad Gaussian + sinusoidal ripple; "
        "no Zernike coefficients are used to generate the measurements"
    ),
    "grid": {"n": cfg.grid.n, "extent": cfg.grid.extent},
    "data": {
        "phase_shift_frames": 16,
        "fourier_frames": 1,
        "lm_samples_per_frame": 6000,
    },
    "truth": {
        "pv_wave": pv(W_true, pupil),
        "rms_after_piston_alignment_wave": rms(W_true_zero_piston, pupil),
    },
    "best_36_mode_projection": {
        "rms_error_wave": rms(projection_error, pupil),
        "max_abs_error_wave": max_abs(projection_error, pupil),
    },
    "lm_z2_to_z36": {
        "n_parameters": len(OBSERVABLE_INDICES),
        "n_iter": result.n_iter,
        "converged": result.converged,
        "message": result.message,
        "rank": result.rank,
        "condition_number": result.cond,
        "intensity_rms_residual": result.rms_residual,
        "rms_error_to_original_wave": rms(lm_total_error, pupil),
        "max_abs_error_to_original_wave": max_abs(lm_total_error, pupil),
        "rms_error_to_best_projection_wave": rms(lm_algorithmic_error, pupil),
        "max_abs_error_to_best_projection_wave": max_abs(
            lm_algorithmic_error, pupil
        ),
        "max_coefficient_difference_from_projection_wave": float(
            np.max(np.abs(result.x - oracle_coeffs[1:]))
        ),
    },
    "same_subspace_control": {
        "n_iter": control_result.n_iter,
        "max_coefficient_error_wave": float(
            np.max(np.abs(control_result.x - oracle_coeffs[1:]))
        ),
    },
    "method_comparison": [
        {
            "method": "phase_shift_differential_zernike",
            "frames": 16,
            "n_parameters": len(OBSERVABLE_INDICES),
            "rank": phase_fit.rank,
            "condition_number": phase_fit.cond,
            "rms_error_to_original_wave": rms(phase_total_error, pupil),
            "max_abs_error_to_original_wave": max_abs(
                phase_total_error, pupil
            ),
            "rms_error_to_best_projection_wave": rms(
                phase_algorithmic_error, pupil
            ),
            "max_abs_error_to_best_projection_wave": max_abs(
                phase_algorithmic_error, pupil
            ),
            "max_coefficient_difference_from_projection_wave": float(
                np.max(np.abs(phase_fit.coeffs - oracle_coeffs[1:]))
            ),
        },
        {
            "method": "fourier_differential_zernike",
            "frames": 1,
            "n_parameters": len(OBSERVABLE_INDICES),
            "rank": fourier_fit.rank,
            "condition_number": fourier_fit.cond,
            "rms_error_to_original_wave": rms(fourier_total_error, pupil),
            "max_abs_error_to_original_wave": max_abs(
                fourier_total_error, pupil
            ),
            "rms_error_to_best_projection_wave": rms(
                fourier_algorithmic_error, pupil
            ),
            "max_abs_error_to_best_projection_wave": max_abs(
                fourier_algorithmic_error, pupil
            ),
            "max_coefficient_difference_from_projection_wave": float(
                np.max(np.abs(fourier_fit.coeffs - oracle_coeffs[1:]))
            ),
        },
        {
            "method": "lm_intensity",
            "frames": 16,
            "n_parameters": len(OBSERVABLE_INDICES),
            "rank": result.rank,
            "condition_number": result.cond,
            "rms_error_to_original_wave": rms(lm_total_error, pupil),
            "max_abs_error_to_original_wave": max_abs(
                lm_total_error, pupil
            ),
            "rms_error_to_best_projection_wave": rms(
                lm_algorithmic_error, pupil
            ),
            "max_abs_error_to_best_projection_wave": max_abs(
                lm_algorithmic_error, pupil
            ),
            "max_coefficient_difference_from_projection_wave": float(
                np.max(np.abs(result.x - oracle_coeffs[1:]))
            ),
        },
    ],
    "largest_recovered_modes": ranked_modes[:10],
}

print("=" * 78)
print("Freeform truth -> intensity frames -> LM fit with Z2...Z36")
print("=" * 78)
print("  truth source       : Gaussian features + sinusoidal ripple (not Zernike)")
print(f"  truth PV / RMS     : {REPORT['truth']['pv_wave']:.5f} / "
      f"{REPORT['truth']['rms_after_piston_alignment_wave']:.5f} wave")
print("  best Z1...Z36 projection (irreducible truncation floor):")
print(f"    RMS / max error  : "
      f"{REPORT['best_36_mode_projection']['rms_error_wave']:.5f} / "
      f"{REPORT['best_36_mode_projection']['max_abs_error_wave']:.5f} wave")
print(f"  LM Z2...Z36       : {result.n_iter} iterations, rank {result.rank}/35, "
      f"cond(J)={result.cond:.2f}")
print(f"    vs original RMS  : "
      f"{REPORT['lm_z2_to_z36']['rms_error_to_original_wave']:.5f} wave")
print(f"    vs projection RMS: "
      f"{REPORT['lm_z2_to_z36']['rms_error_to_best_projection_wave']:.5f} wave")
print(f"  same-subspace control max coefficient error: "
      f"{REPORT['same_subspace_control']['max_coefficient_error_wave']:.2e} wave")
print()
print("  method comparison (all reconstruct Z2...Z36):")
print("  method          frames  RMS vs truth  max vs truth  RMS vs projection")
for row in REPORT["method_comparison"]:
    label = {
        "phase_shift_differential_zernike": "phase shift",
        "fourier_differential_zernike": "Fourier",
        "lm_intensity": "LM",
    }[row["method"]]
    print(
        f"  {label:12s}  {row['frames']:6d}  "
        f"{row['rms_error_to_original_wave']:12.5f}  "
        f"{row['max_abs_error_to_original_wave']:12.5f}  "
        f"{row['rms_error_to_best_projection_wave']:17.5f}"
    )

fig, axes = new_fig(3, 3, figsize=(15, 13))
imshow(
    axes[0, 0],
    np.where(pupil, W_true_zero_piston, np.nan),
    cfg.grid,
    title="freeform truth, piston removed (waves)",
)
imshow(
    axes[0, 1],
    np.where(pupil, W_oracle_aligned, np.nan),
    cfg.grid,
    title="best Z1...Z36 projection (waves)",
)
imshow(
    axes[0, 2],
    np.where(pupil, projection_error, np.nan),
    cfg.grid,
    title="projection truncation error (waves)",
)
imshow(
    axes[1, 0],
    np.where(pupil, W_phase_aligned, np.nan),
    cfg.grid,
    title="phase-shift reconstruction, Z2...Z36",
)
imshow(
    axes[1, 1],
    np.where(pupil, W_fourier_aligned, np.nan),
    cfg.grid,
    title="Fourier reconstruction, Z2...Z36",
)
imshow(
    axes[1, 2],
    np.where(pupil, W_lm_aligned, np.nan),
    cfg.grid,
    title="LM reconstruction, Z2...Z36 (waves)",
)
imshow(
    axes[2, 0],
    np.where(pupil, phase_total_error, np.nan),
    cfg.grid,
    title="phase shift minus freeform truth",
)
imshow(
    axes[2, 1],
    np.where(pupil, fourier_total_error, np.nan),
    cfg.grid,
    title="Fourier minus freeform truth",
)
imshow(
    axes[2, 2],
    np.where(pupil, lm_total_error, np.nan),
    cfg.grid,
    title="LM minus freeform truth (waves)",
)
savefig(fig, "06_freeform_wavefront.png")

with open(os.path.join(OUT, "06_freeform_wavefront.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/06_freeform_wavefront.png and .json")

