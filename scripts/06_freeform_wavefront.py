"""06 - 反演一个并非由 Zernike 系数生成的波前。

下面的自由曲面真值由两个局部高斯、一个宽缓特征和一段中频波纹解析
构造。Zernike 多项式只在探测器帧生成之后才登场：

1. 把采样真值投影到 Z1...Z36，得到 36 模波前域最优逼近（截断下限）；
2. 用 LM 直接从 8+8 帧光强拟合 Z2...Z36；
3. 把 LM 波前分别与原始自由真值和最优 36 模投影比较。

Z1 进入 oracle 投影但不进 LM：光强对平移不可观。所有图比较都在光瞳内
先去掉均值差。

运行：  python3 scripts/06_freeform_wavefront.py
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
from lsi.metrics import pv, rms
from lsi.pipeline import fourier_to_wavefront, phase_shift_to_wavefront
from lsi.plotting import imshow, new_fig, savefig
from lsi.zernike import wavefront, zernike_matrix

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)

ALL_INDICES = np.arange(1, 37)
OBSERVABLE_INDICES = np.arange(2, 37)


class FreeformWavefront:
    """独立于 Zernike 基定义的光滑合成曲面。"""

    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        local_positive = 0.20 * np.exp(
            -((x - 0.28) ** 2 / 0.18**2 + (y + 0.12) ** 2 / 0.24**2)
        )
        local_negative = -0.15 * np.exp(
            -((x + 0.32) ** 2 / 0.25**2 + (y - 0.24) ** 2 / 0.16**2)
        )
        broad = 0.06 * np.exp(-((x + 0.05) ** 2 + (y + 0.10) ** 2) / 0.65**2)
        ripple = 0.025 * np.sin(5.0 * np.pi * x + 0.4) * np.cos(4.0 * np.pi * y - 0.2)
        return local_positive + local_negative + broad + ripple


def align_piston(estimate: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    """把强度不可观的平移对齐到 ``reference``。"""
    return estimate - np.mean((estimate - reference)[mask])


def masked_rms(values, mask):
    return float(np.sqrt(np.mean(values[mask] ** 2)))


def masked_max(values, mask):
    return float(np.max(np.abs(values[mask])))


cfg = SystemConfig(grid=Grid(n=128, extent=1.10), period_um=30.0)
fm = ForwardModel(cfg)
truth = FreeformWavefront()
x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_true = truth.w(x, y)
W_true_zero_piston = align_piston(W_true, np.zeros_like(W_true), pupil)

# 波前域 oracle：对已生成自由曲面做 Z1...Z36 最小二乘投影（不参与生成数据）
Z36 = zernike_matrix(ALL_INDICES, x[pupil], y[pupil])
oracle_coeffs, *_ = np.linalg.lstsq(Z36, W_true[pupil], rcond=None)
W_oracle = wavefront(oracle_coeffs, ALL_INDICES, x, y)
W_oracle_aligned = align_piston(W_oracle, W_true, pupil)
projection_error = W_oracle_aligned - W_true

# 测量帧由原始自由曲面对象生成，而非 W_oracle
frames_x = fm.phase_shift_frames(truth, "x", 8)
frames_y = fm.phase_shift_frames(truth, "y", 8)
frames = np.concatenate([np.asarray(frames_x), np.asarray(frames_y)], axis=0)
deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]

result = fit_wavefront_from_frames(
    fm, OBSERVABLE_INDICES, frames, deltas,
    samples=6000, config=LMConfig(max_iter=100),
)
W_lm = wavefront(result.x, OBSERVABLE_INDICES, x, y)
W_lm_aligned = align_piston(W_lm, W_true, pupil)
lm_total_error = W_lm_aligned - W_true
lm_algorithmic_error = align_piston(W_lm, W_oracle, pupil) - W_oracle

# 同一自由真值过两条论文路线，重构基同为 Z2...Z36
phase_fit, _ = phase_shift_to_wavefront(fm, frames_x, frames_y,
                                        indices=OBSERVABLE_INDICES)
W_phase = wavefront(phase_fit.coeffs, phase_fit.indices, x, y)
W_phase_aligned = align_piston(W_phase, W_true, pupil)
phase_total_error = W_phase_aligned - W_true
phase_algorithmic_error = align_piston(W_phase, W_oracle, pupil) - W_oracle

carrier_frame = fm.carrier_frame(truth)
fourier_fit, _ = fourier_to_wavefront(fm, carrier_frame,
                                      indices=OBSERVABLE_INDICES)
W_fourier = wavefront(fourier_fit.coeffs, fourier_fit.indices, x, y)
W_fourier_aligned = align_piston(W_fourier, W_true, pupil)
fourier_total_error = W_fourier_aligned - W_true
fourier_algorithmic_error = align_piston(W_fourier, W_oracle, pupil) - W_oracle

# 对照：数据真落在同一 36 模子空间时，同一 LM 应精确恢复可观系数
projected_truth = ZernikeWavefront(oracle_coeffs[1:], OBSERVABLE_INDICES)
control = np.concatenate(
    [fm.phase_shift_frames(projected_truth, "x", 8),
     fm.phase_shift_frames(projected_truth, "y", 8)], axis=0)
control_result = fit_wavefront_from_frames(
    fm, OBSERVABLE_INDICES, control, deltas,
    samples=6000, config=LMConfig(max_iter=100),
)

ranked_modes = sorted(
    ({"index": int(j), "lm": float(c_lm), "oracle": float(c_or),
      "difference": float(c_lm - c_or)}
     for j, c_lm, c_or in zip(OBSERVABLE_INDICES, result.x, oracle_coeffs[1:])),
    key=lambda item: abs(item["lm"]), reverse=True,
)


def method_row(name, n_frames, coeffs, indices, W_aligned, W_to_oracle_err, cond):
    return {
        "method": name, "frames": n_frames, "n_parameters": len(indices),
        "condition_number": cond,
        "rms_error_to_original_wave": masked_rms(W_aligned - W_true, pupil),
        "max_abs_error_to_original_wave": masked_max(W_aligned - W_true, pupil),
        "rms_error_to_best_projection_wave": masked_rms(W_to_oracle_err, pupil),
        "max_coefficient_difference_from_projection_wave": float(
            np.max(np.abs(coeffs - oracle_coeffs[1:]))
        ),
    }


REPORT = {
    "wavefront_source": (
        "two localized Gaussians + one broad Gaussian + sinusoidal ripple; "
        "no Zernike coefficients are used to generate the measurements"
    ),
    "grid": {"n": cfg.grid.n, "extent": cfg.grid.extent},
    "truth": {
        "pv_wave": pv(W_true, pupil),
        "rms_after_piston_alignment_wave": masked_rms(W_true_zero_piston, pupil),
    },
    "best_36_mode_projection": {
        "rms_error_wave": masked_rms(projection_error, pupil),
        "max_abs_error_wave": masked_max(projection_error, pupil),
    },
    "lm_z2_to_z36": {
        "n_iter": result.n_iter, "converged": result.converged,
        "message": result.message, "condition_number": result.cond,
        "intensity_rms_residual": result.rms_residual,
        "rms_error_to_original_wave": masked_rms(lm_total_error, pupil),
        "max_abs_error_to_original_wave": masked_max(lm_total_error, pupil),
        "rms_error_to_best_projection_wave": masked_rms(lm_algorithmic_error, pupil),
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
        method_row("phase_shift_differential_zernike", 16,
                   phase_fit.coeffs, OBSERVABLE_INDICES,
                   W_phase_aligned, phase_algorithmic_error, phase_fit.cond),
        method_row("fourier_differential_zernike", 1,
                   fourier_fit.coeffs, OBSERVABLE_INDICES,
                   W_fourier_aligned, fourier_algorithmic_error, fourier_fit.cond),
        method_row("lm_intensity", 16,
                   result.x, OBSERVABLE_INDICES,
                   W_lm_aligned, lm_algorithmic_error, result.cond),
    ],
    "largest_recovered_modes": ranked_modes[:10],
}

print("=" * 78)
print("自由曲面真值 -> 光强帧 -> Z2...Z36 拟合")
print("=" * 78)
print(f"  真值 PV / RMS      : {REPORT['truth']['pv_wave']:.5f} / "
      f"{REPORT['truth']['rms_after_piston_alignment_wave']:.5f} wave")
print("  Z1...Z36 最优投影（不可约截断下限）:")
print(f"    RMS / max 误差   : "
      f"{REPORT['best_36_mode_projection']['rms_error_wave']:.5f} / "
      f"{REPORT['best_36_mode_projection']['max_abs_error_wave']:.5f} wave")
print(f"  LM Z2...Z36       : {result.n_iter} 次迭代, cond(J)={result.cond:.2f}")
print(f"    对真值 RMS       : "
      f"{REPORT['lm_z2_to_z36']['rms_error_to_original_wave']:.5f} wave")
print(f"    对投影 RMS       : "
      f"{REPORT['lm_z2_to_z36']['rms_error_to_best_projection_wave']:.5f} wave")
print(f"  同子空间对照最大系数误差: "
      f"{REPORT['same_subspace_control']['max_coefficient_error_wave']:.2e} wave")
print()
print("  方法比较（均重构 Z2...Z36）:")
print("  方法          帧数   RMS vs 真值   max vs 真值   RMS vs 投影")
for row in REPORT["method_comparison"]:
    label = {"phase_shift_differential_zernike": "phase shift",
             "fourier_differential_zernike": "Fourier",
             "lm_intensity": "LM"}[row["method"]]
    print(f"  {label:12s}  {row['frames']:5d}  "
          f"{row['rms_error_to_original_wave']:12.5f}  "
          f"{row['max_abs_error_to_original_wave']:12.5f}  "
          f"{row['rms_error_to_best_projection_wave']:17.5f}")

fig, axes = new_fig(3, 3, figsize=(15, 13))
imshow(axes[0, 0], np.where(pupil, W_true_zero_piston, np.nan), cfg.grid,
       title="freeform truth, piston removed (waves)")
imshow(axes[0, 1], np.where(pupil, W_oracle_aligned, np.nan), cfg.grid,
       title="best Z1...Z36 projection (waves)")
imshow(axes[0, 2], np.where(pupil, projection_error, np.nan), cfg.grid,
       title="projection truncation error (waves)")
imshow(axes[1, 0], np.where(pupil, W_phase_aligned, np.nan), cfg.grid,
       title="phase-shift reconstruction, Z2...Z36")
imshow(axes[1, 1], np.where(pupil, W_fourier_aligned, np.nan), cfg.grid,
       title="Fourier reconstruction, Z2...Z36")
imshow(axes[1, 2], np.where(pupil, W_lm_aligned, np.nan), cfg.grid,
       title="LM reconstruction, Z2...Z36 (waves)")
imshow(axes[2, 0], np.where(pupil, phase_total_error, np.nan), cfg.grid,
       title="phase shift minus freeform truth")
imshow(axes[2, 1], np.where(pupil, fourier_total_error, np.nan), cfg.grid,
       title="Fourier minus freeform truth")
imshow(axes[2, 2], np.where(pupil, lm_total_error, np.nan), cfg.grid,
       title="LM minus freeform truth (waves)")
savefig(fig, "06_freeform_wavefront.png")

with open(os.path.join(OUT, "06_freeform_wavefront.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/06_freeform_wavefront.png and .json")
