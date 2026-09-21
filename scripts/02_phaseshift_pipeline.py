"""02 - dissertation route A: multi-step phase shifting.

    I(x,y)  --8-step phase-shift demodulation-->  wrapped dW
            --region selection-->  --unwrap-->  dW  --differential Zernike-->  W

Also demonstrates two things the dissertation's own route has to live with:

* the half-fringe offset of the chessboard (the x pair and the y pair sit
  half a fringe apart) -- a constant that is collinear with tilt,
* the break-down for large aberrations, where the modulation
  ``4 A0 A1 cos(pi[...])`` changes sign inside a shear region (2.3.2).

Run:  python3 scripts/02_phaseshift_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.metrics import coefficient_error_metrics, pv, rms, wavefront_error
from lsi.phaseshift import lsq_phase_shift, zero_order_center_radius
from lsi.pipeline import demodulate_phase_shift, phase_shift_to_wavefront
from lsi.plotting import imshow, new_fig, savefig
from lsi.reconstruct import wavefront_on_grid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(fit) -> dict:
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


# --------------------------------------------------------------------------- #
section("1. Single Zernike term: W = Z7 (coma), 1 wave, 8-step phase shift")

cfg = SystemConfig(grid=Grid(n=128, extent=1.10))
fm = ForwardModel(cfg)
truth = ZernikeWavefront(np.array([1.0]), np.array([7]))
t0 = time.time()
fx = fm.phase_shift_frames(truth, "x", 8)
fy = fm.phase_shift_frames(truth, "y", 8)
fit, diff = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
dt = time.time() - t0
tab = table(fit)
print(f"  {cfg.describe()}")
print(f"  {len(INDICES)} Zernike terms fitted from {2*8} frames in {dt*1e3:.0f} ms")
print("  recovered coefficients (waves):")
for j, v in tab.items():
    flag = "   <-- input" if j == 7 else ""
    print(f"    Z{j:2d} : {v:+.16e}{flag}")
print(f"  max |coefficient error| : {max(abs(v - (1.0 if j==7 else 0.0)) for j, v in tab.items()):.3e} wave")
print(f"  max |differential residual| : {fit.max_abs_residual:.3e} wave")
print(f"  condition number of the LSQ system : {fit.cond:.1f}")
REPORT["single_term"] = {"fit": tab, "max_residual": float(fit.max_abs_residual),
                         "cond": float(fit.cond), "time_ms": dt * 1e3}

# --------------------------------------------------------------------------- #
section("2. PV / RMS against the dissertation (its 表3-2/3-4: PV 1.985, RMS 0.354)")

x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y, pupil)
print(f"  sampled PV  = {pv(W_fit, pupil):.4f} wave   (paper 1.985)")
print(f"  sampled RMS = {rms(W_fit, pupil):.4f} wave   (paper 0.354)")
fine = SystemConfig(grid=Grid(n=512, extent=1.10))
Wf = ZernikeWavefront(np.array([1.0]), np.array([7])).w(*fine.grid.coords())
print(f"  analytic basis PV = {pv(Wf, fine.grid.pupil()):.4f}, RMS = {rms(Wf, fine.grid.pupil()):.5f} "
      f"(Z7 = (3rho^3-2rho)cos(theta): PV = 2 exactly, RMS = 1/sqrt(8))")
REPORT["pv_rms"] = {"pv_sampled": pv(W_fit, pupil), "rms_sampled": rms(W_fit, pupil)}

# --------------------------------------------------------------------------- #
section("3. Shear-region selection from the modulation (3.1.1, eqs. 3-1 ... 3-5)")

res = lsq_phase_shift(fx)
from lsi.phaseshift import circle_fit

cx0, cy0, edge = zero_order_center_radius(res.modulation, cfg.grid, threshold_frac=0.4)
x_, y_ = cfg.grid.coords()
cx, cy, radius = circle_fit(x_[edge], y_[edge])
fit_m, diff_m = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES, region_mode="modulation")
print(f"  thresholded edge pixels  : {int(edge.sum())}")
print(f"  fitted zero-order circle : centre = ({cx:+.5f}, {cy:+.5f}), radius = {radius:.5f} "
      f"(true values 0, 0, 1.000)")
print(f"  analytic region pixels   : {int(diff.mask['x'].sum())}")
print(f"  circle-fit region pixels : {int(diff_m.mask['x'].sum())}")
print(f"  region overlap (IoU)     : {float((diff.mask['x'] & diff_m.mask['x']).sum() / (diff.mask['x'] | diff_m.mask['x']).sum()):.4f}")
print(f"  Z7 from circle-fit region: {table(fit_m)[7]:+.10f}   (analytic region: {tab[7]:+.10f})")
REPORT["circle_fit"] = {"cx": cx, "cy": cy, "r": radius,
                        "iou_with_analytic": float((diff.mask['x'] & diff_m.mask['x']).sum()
                                                   / (diff.mask['x'] | diff_m.mask['x']).sum())}

# --------------------------------------------------------------------------- #
section("4. Mixed aberration + noise sensitivity")

truth_mix = ZernikeWavefront(np.array([0.0, 0.0, 0.22, -0.05, 0.11, 0.31, -0.12, 0.07, 0.04]),
                             np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
fx = fm.phase_shift_frames(truth_mix, "x", 8)
fy = fm.phase_shift_frames(truth_mix, "y", 8)
fit_mix, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
tab_mix = table(fit_mix)
worst = coefficient_error_metrics(
    tab_mix, truth_mix.indices, truth_mix.coeffs
)["max_error_all_modes"]
print(f"  mixed wavefront: max |coefficient error| = {worst:.3e} wave "
      f"({len(INDICES)} fitted modes)")

snr_list = [60, 50, 40, 30, 20, 10]
errs = []
for snr in snr_list:
    fx = add_noise(fm.phase_shift_frames(truth_mix, "x", 8), snr_db=snr, seed=1)
    fy = add_noise(fm.phase_shift_frames(truth_mix, "y", 8), snr_db=snr, seed=2)
    f, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES, unwrap="poisson")
    t = table(f)
    errs.append(coefficient_error_metrics(
        t, truth_mix.indices, truth_mix.coeffs
    )["max_error_all_modes"])
    print(f"  SNR {snr:3d} dB : max |coefficient error| = {errs[-1]:.4f} wave")
REPORT["noise"] = {"snr_db": snr_list, "max_coeff_error": errs}

fig, ax = new_fig(1, 1, figsize=(6.5, 4.5))
ax.semilogy(snr_list, np.maximum(errs, 1e-6), "o-")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("max |Zernike coefficient error| (wave)")
ax.set_title("phase-shift route: noise sensitivity")
ax.grid(alpha=0.3)
ax.invert_xaxis()
savefig(fig, "02_noise_sweep.png")

# --------------------------------------------------------------------------- #
section("5. The half-fringe offset: indistinguishable from tilt")

fm0 = ForwardModel(cfg)
fd = demodulate_phase_shift(fm0, fm0.phase_shift_frames(truth, "x", 8),
                            fm0.phase_shift_frames(truth, "y", 8), remove_offset=False)
print(f"  demodulated phase of the x pair carries a constant of "
      f"{fm0.demodulation_offset('x'):.6f} rad = half a fringe")
print(f"  demodulated phase of the y pair carries "
      f"{fm0.demodulation_offset('y'):.6f} rad")
from lsi.pipeline import reconstruct

for mode in ("none", "model", "estimate"):
    if mode == "none":
        d = demodulate_phase_shift(
            fm0,
            fm0.phase_shift_frames(truth, "x", 8),
            fm0.phase_shift_frames(truth, "y", 8),
        )
        f = reconstruct(fm0, d, indices=INDICES, offset_mode="none")
    elif mode == "model":
        d = demodulate_phase_shift(
            fm0,
            fm0.phase_shift_frames(truth, "x", 8),
            fm0.phase_shift_frames(truth, "y", 8),
            remove_offset=False,
        )
        f = reconstruct(fm0, d, indices=INDICES, offset_mode="model")
    else:
        d = demodulate_phase_shift(
            fm0,
            fm0.phase_shift_frames(truth, "x", 8),
            fm0.phase_shift_frames(truth, "y", 8),
            remove_offset=False,
        )
        f = reconstruct(fm0, d, indices=INDICES, offset_mode="estimate")
    t = table(f)
    print(f"  offset handling '{mode:8s}': Z2 = {t[2]:+9.4f}, Z7 = {t[7]:.6f}, cond = {f.cond:.1e}")
print(f"  (with s = {cfg.s:.5f}, a one-wave constant in dW_x maps onto Z2 as 1/(2s) = {1/(2*cfg.s):.2f})")

# --------------------------------------------------------------------------- #
section("6. Known limitation: large aberration breaks the demodulated phase")

big = ZernikeWavefront(np.array([6.0]), np.array([7]))
fb = ForwardModel(cfg)
db = demodulate_phase_shift(fb, fb.phase_shift_frames(big, "x", 8),
                            fb.phase_shift_frames(big, "y", 8))
from lsi.unwrap import wrap

x, y = cfg.grid.coords()
pred = np.pi * (big.w(x + cfg.s, y) - big.w(x - cfg.s, y))
resid = wrap(db.phase["x"] - pred)[db.mask["x"]]
frac = float(np.mean(np.abs(wrap(resid - np.angle(np.mean(np.exp(1j * resid))))) > 1.0))
print(f"  W = 6 x Z7 : {100*frac:.1f} % of the x-region pixels are more than 1 rad off a")
print("  *single* constant -> the modulation term changes sign inside the region and the")
print("  demodulated phase picks up jumps of pi; a global offset cannot repair it.")
fit_big, _ = phase_shift_to_wavefront(fb, fb.phase_shift_frames(big, "x", 8),
                                      fb.phase_shift_frames(big, "y", 8), indices=INDICES)
print(f"  this route returns Z7 = {table(fit_big)[7]:.4f} instead of 6.0 "
      f"(see script 04: LM has no such problem)")
print("  mechanism (README 4.1): the demodulation route is exact (<1e-9) for a single")
print("  Z7 up to ~3 waves; past ~3.1 waves the leaked (0,+-1) order drives the")
print("  modulation |Z| through a null, the wrapped phase picks up 4 vortices and")
print("  unwrapping drops whole waves (1 wave at 3.5, 2 waves at >= 4).  The")
print("  coefficient deficit therefore depends on grid/mask/sampling (~5.6-5.8 here),")
print("  unlike the exact loss bookkeeping of the differential itself.")
REPORT["large_aberration"] = {"frac_bad_pixels": frac, "Z7_phase_shift": table(fit_big)[7]}

# --------------------------------------------------------------------------- #
section("7. Figures")

d = demodulate_phase_shift(fm, fm.phase_shift_frames(truth, "x", 8),
                           fm.phase_shift_frames(truth, "y", 8))
res_x = lsq_phase_shift(fm.phase_shift_frames(truth, "x", 8))
fig, axes = new_fig(2, 3, figsize=(15, 9))
imshow(axes[0, 0], fm.phase_shift_frames(truth, "x", 8)[0], cfg.grid, title="frame t=0 (x scan)")
imshow(axes[0, 1], res_x.modulation, cfg.grid, title="modulation |A| (zero-order disc)")
imshow(axes[0, 2], res_x.phase, cfg.grid, title="wrapped demodulated phase (rad)")
imshow(axes[1, 0], np.where(d.mask["x"], d.phase["x"], np.nan), cfg.grid,
       title="unwrapped dW_x (rad)")
imshow(axes[1, 1], W_fit, cfg.grid, title="reconstructed W (waves)")
imshow(axes[1, 2], W_fit - truth.w(x, y), cfg.grid, title="reconstruction error (waves)")
savefig(fig, "02_phaseshift_pipeline.png")

with open(os.path.join(OUT, "02_phaseshift_pipeline.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/02_phaseshift_pipeline.png and .json")