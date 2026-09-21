"""01 - forward model of the 2-D chessboard shearing interferometer.

Reproduces, step by step, chapter 2 of the dissertation:

* diffraction orders of the 45 deg rotated amplitude chessboard grating
  (表2-3: 0 order 25 %, each +-1 order 4.11 %, crossed grating 2.53 %),
* the phase-shift law of 2.3.1 (period 8 frames along +x),
* the superposition forward model ``I = |sum A exp(i[2 pi W + delta])|^2``
  against the explicit 4-beam / 5-beam region formulas (2-12) ... (2-16).

Run:  python3 scripts/01_forward_model.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig, preset_fourier, preset_phase_shift
from lsi.forward import ForwardModel, ZernikeWavefront, paper_region_intensity
from lsi.ftmode import spectrum
from lsi.grating import OrderSet, analytic_orders, bitmap_orders, diffraction_efficiency
from lsi.metrics import pv, rms
from lsi.plotting import imshow, new_fig, savefig
from lsi.zernike import zernike_value

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
REPORT: dict = {}


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
section("1. Diffraction orders of the chessboard grating (表2-3)")

ideal = analytic_orders(max_index=25)
eff = diffraction_efficiency(ideal)
first = eff["first_order_total"] * 100 / 4
crossed = (0.5 / np.pi) ** 2 * 100
print(f"  0 order            : {eff['dc']*100:.2f} %   (paper 25 %)")
print(f"  each +-1 order     : {first:.2f} %   (paper 4.11 %)")
print(f"  sum over all orders: {sum(eff['all'].values()):.4f}   (Parseval; analytically 0.5)")
print(f"  crossed 2-D grating: {crossed:.2f} %   (paper 2.53 %)")
print("  closed form |A| = 2/(pi^2 |a^2-b^2|):")
for a, b in [(1, 0), (0, 1), (2, 1), (-2, -1)]:
    print(f"    (a,b)=({a:+d},{b:+d})  |A| = {abs(ideal.with_orders([(a, b)]).amp[0]):.6f}")
REPORT["grating"] = {
    "dc_pct": eff["dc"] * 100,
    "first_order_each_pct": first,
    "crossed_pct": crossed,
    "parseval": float(sum(eff["all"].values())),
}

bit = bitmap_orders(harmonic_cell=120, max_index=3)
print("\n  full bitmap FFT (duty 0.5), |A| of the first orders:")
for k in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
    print(f"    (a,b)=({k[0]:+d},{k[1]:+d})  |A| = {abs(bit.with_orders([k]).amp[0]):.6f}")

# --------------------------------------------------------------------------- #
section("2. Superposition model vs the paper's region formulas (2-12) ... (2-16)")

cfg = SystemConfig(grid=Grid(n=160, extent=1.10))
s = cfg.s
A0, A1 = 0.5, 2.0 / np.pi**2


def order_set(orders) -> OrderSet:
    return OrderSet(
        np.array(orders, dtype=int),
        np.array([A0 if tuple(o) == (0, 0) else A1 for o in orders], dtype=complex),
    )


class SmoothWavefront:
    """Non-separable test wavefront (coma + defocus + astig + local bump)."""

    def w(self, x, y):
        return (
            0.45 * zernike_value(7, x, y)
            - 0.20 * zernike_value(4, x, y)
            + 0.15 * zernike_value(11, x, y)
            + 0.10 * zernike_value(5, x, y)
            + 0.12 * np.exp(-3.0 * (x**2 + y**2)) * np.cos(3.0 * y)
        )


wf = SmoothWavefront()
x, y = cfg.grid.coords()
regions = {
    "x1 (beams 0,+-1_x,+1_y)": ([(0, 0), (1, 0), (-1, 0), (0, 1)], "x1"),
    "x2 (beams 0,+-1_x,-1_y)": ([(0, 0), (1, 0), (-1, 0), (0, -1)], "x2"),
    "x5 (all five beams)": ([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], "x5"),
    "y1 (beams 0,+-1_y,+1_x)": ([(0, 0), (0, 1), (0, -1), (1, 0)], "y1"),
}
for label, (orders, tag) in regions.items():
    fm = ForwardModel(cfg, order_set(orders))
    for t in (0.0, 0.125):
        # x-regions are compared with an x-scan, y-regions with a y-scan
        deltas = fm.phase_shift_deltas(t, 0.0) if tag.startswith("x") else fm.phase_shift_deltas(0.0, t)
        I_model = fm.intensity(wf, deltas=deltas)
        mask = np.ones(cfg.grid.shape, dtype=bool)
        for a, b in orders:
            mask &= (x + a * s) ** 2 + (y + b * s) ** 2 <= 1.0
        I_paper = paper_region_intensity(tag, wf.w, x, y, s, 2 * np.pi * t, A0=A0, A1=A1)
        err = np.abs(I_model[mask] - I_paper[mask]).max() / np.abs(I_paper[mask]).max()
        print(f"  {label:26s} t={t:5.3f}  max relative difference = {err:.2e}")
        REPORT.setdefault("region_agreement", []).append(
            {"region": label, "t": t, "rel_err": float(err)}
        )

# --------------------------------------------------------------------------- #
section("3. Phase-shift law of 2.3.1 (period 8 frames along +x)")

cfg_ps = preset_phase_shift()
fm_ps = ForwardModel(cfg_ps)
truth = ZernikeWavefront(np.array([1.0]), np.array([7]))
frames = fm_ps.phase_shift_frames(truth, "x", 8)
print("  8-step x-scan: p2v of each frame and the delta of the (+1,0) order")
for k, frame in enumerate(frames):
    d = fm_ps.phase_shift_deltas(k / 8, 0.0)[fm_ps.indices.index((1, 0))]
    print(f"    frame {k}: p2v = {np.ptp(frame):.5f}   delta(+1,0) = {d:+.4f} rad")
print("  |frame_0 - frame_8| would be the same grating position again "
      "(period of the phase-shift law).")
REPORT["phase_shift_law"] = {"period_frames": 8}

# --------------------------------------------------------------------------- #
section("4. Forward-model output for W = Z7 (1 wave): maps and spectrum")

cfg_ft = preset_fourier()
fm_ft = ForwardModel(cfg_ft)
frame_ft = fm_ft.ft_mode_frame(truth)
frames4 = fm_ps.phase_shift_frames(truth, "x", 4)

x_ps, y_ps = cfg_ps.grid.coords()
pupil = cfg_ps.grid.pupil()
W_true = truth.w(x_ps, y_ps)
print(f"  phase-shift config : {cfg_ps.describe()}")
print(f"  Fourier config     : {cfg_ft.describe()}")
print(f"  input W            : PV = {pv(W_true, pupil):.4f} wave, RMS = {rms(W_true, pupil):.4f} wave")
print(f"  (theory for Z7=1: PV = 2.000, RMS = 0.35355; the sampled values above "
      f"depend on grid resolution)")

fig, axes = new_fig(2, 4, figsize=(16, 7.5))
for k in range(4):
    imshow(axes[0, k], frames4[k], cfg_ps.grid, title=f"phase-shift frame t={k}/4")
imshow(axes[1, 0], frame_ft, cfg_ft.grid, title="Fourier mode: single frame")
imshow(axes[1, 1], np.log10(np.abs(spectrum(frame_ft)) + 1e-9), cfg_ft.grid,
       title=f"log|spectrum|, f0 = {cfg_ft.carrier_f0:.2f} cyc/unit")
imshow(axes[1, 2], 2 * np.pi * (truth.w(x_ps + cfg_ps.s, y_ps) - truth.w(x_ps - cfg_ps.s, y_ps)),
       cfg_ps.grid, title="true 2-sided dW_x (rad)")
imshow(axes[1, 3], W_true, cfg_ps.grid, title="input W (waves)")
savefig(fig, "01_forward.png")
print(f"\n  -> wrote {OUT}/01_forward.png and 01_forward_model.json")

with open(os.path.join(OUT, "01_forward_model.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)