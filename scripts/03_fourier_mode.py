"""03 - dissertation route B: single-frame Fourier transform (spatial carrier).

    I(x,y)  --carrier f0 = m/2s-->  spectral lobe  --window-->  c = A e^{i psi}
            --arg-->  dW  --differential Zernike-->  W

The script also verifies the physical content of the ``+f0`` lobe.  With both
symmetric first orders present, the lobe contains the two beats ``E+ E0*`` and
``E0 E-*`` and therefore carries the dissertation's two-sided difference
``W(x+s)-W(x-s)``.

Run:  python3 scripts/03_fourier_mode.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.ftmode import demodulate_lobe, spectrum

from lsi.pipeline import demodulate_fourier, fourier_to_wavefront
from lsi.plotting import imshow, new_fig, savefig

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(fit):
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


section("1. Carrier frequency vs grating period (f0 = m / 2s, s = p / (sqrt(2) f))")
print("  the shear is set by the *pitch* p of the 45 deg grating; a longer pitch gives")
print("  a smaller shear and therefore a denser carrier:")
rows = []
for p_um in (18.0, 30.0, 45.0, 60.0):
    cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=p_um, na=0.34, talbot_number=1)
    print("    p = %5.1f um -> s = %.5f, f0 = %6.2f cyc/unit (%.2f fringes across the pupil), "
          "fringe spacing = %.2f px" % (p_um, cfg.s, cfg.carrier_f0, cfg.carrier_f0 * 2.0,
                                  cfg.grid.n / (2.0 * cfg.carrier_f0 * 2 * cfg.grid.extent)))
    rows.append({"p_um": p_um, "s": cfg.s, "f0": cfg.carrier_f0})
REPORT["periods"] = rows

cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
fm = ForwardModel(cfg)
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6]), np.array([2, 3, 4, 5, 6, 7]))
print(f"  working configuration: {cfg.describe()}")

# --------------------------------------------------------------------------- #
section("2. Demodulation of one frame: the +f0 lobe is two-sided")

I = fm.ft_mode_frame(truth)
dWx, mask_x, lobe_x = demodulate_fourier(fm, I, direction="x")
dWy, mask_y, lobe_y = demodulate_fourier(fm, I, direction="y")
x, y = cfg.grid.coords()

two = truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y)
wrong_one_doubled = 2.0 * (truth.w(x + cfg.s, y) - truth.w(x, y))
inner = mask_x & (np.hypot(x, y) < 0.7)
print(f"  lobe detected at {lobe_x.peak_freq} cyc/unit (nominal f0 = {cfg.carrier_f0:.3f})")
print(f"  region pixels: {int(mask_x.sum())} (x), {int(mask_y.sum())} (y)")
print(f"  two-sided reading  : rms(dW - [W(x+s)-W(x-s)]) = "
      f"{np.sqrt(np.mean((dWx - two)[inner] ** 2)):.3e} wave")
print(f"  wrong one-sided    : rms(dW - 2[W(x+s)-W(x)]) = "
      f"{np.sqrt(np.mean((dWx - wrong_one_doubled)[inner] ** 2)):.3e} wave")
h = 1e-4
W_xx = (truth.w(x + h, y) - 2 * truth.w(x, y) + truth.w(x - h, y)) / h**2
print(f"  wrong-model gap    : max |2[W(x+s)-W(x)] - [W(x+s)-W(x-s)]| = "
      f"{np.abs(wrong_one_doubled - two)[inner].max():.4f} wave")
print(f"                       predicted s^2 W_xx       = {np.abs(cfg.s**2*W_xx)[inner].max():.4f} wave")
REPORT["model_difference"] = {
    "two_sided_rms": float(np.sqrt(np.mean((dWx - two)[inner] ** 2))),
    "wrong_one_sided_doubled_rms": float(
        np.sqrt(np.mean((dWx - wrong_one_doubled)[inner] ** 2))
    ),
    "max_wrong_model_gap": float(
        np.abs(wrong_one_doubled - two)[inner].max()
    ),
}

# --------------------------------------------------------------------------- #
section("3. Reconstruction from one carrier frame (and with noise)")

fit, diff = fourier_to_wavefront(fm, I, indices=INDICES)
tab = table(fit)
print("  recovered coefficients:",
      ", ".join(f"Z{j}={v:+.4f}" for j, v in tab.items() if abs(v) > 1e-3))
print(f"  Z7 = {tab[7]:+.4f}  (input 0.6),  max |error over all terms| = "
      f"{max(abs(v - (0.6 if j == 7 else 0.0)) for j, v in tab.items()):.4f} wave")
print(f"  max |differential residual| = {fit.max_abs_residual:.4f} wave")

errs = []
snrs = [60, 50, 40, 30, 20]
for snr in snrs:
    In = add_noise(fm.ft_mode_frame(truth), snr_db=snr, seed=7)
    f, _ = fourier_to_wavefront(fm, In, indices=INDICES)
    e = abs(table(f)[7] - 0.6)
    errs.append(e)
    print(f"  SNR {snr:3d} dB : |Z7 - 0.6| = {e:.4f} wave")
REPORT["noise"] = {"snr_db": snrs, "Z7_abs_error": errs}

# --------------------------------------------------------------------------- #
section("4. Choice of demodulator (all with the same physical phase reference)")

x_, y_ = cfg.grid.coords()
two_x = truth.w(x_ + cfg.s, y_) - truth.w(x_ - cfg.s, y_)
inner_x = demodulate_fourier(fm, I, direction="x")[1] & (np.hypot(x_, y_) < 0.7)
print("  method                       rms(dW - two-sided)   Z7")
demodulator_rows = []
for sig in (0.04, 0.06, 0.10):
    dW, mk, L = demodulate_fourier(fm, I, direction="x",
                                   method="local", sigma_units=sig)
    f, _ = fourier_to_wavefront(fm, I, indices=INDICES,
                                method="local", sigma_units=sig)
    r = dW[inner_x] - two_x[inner_x]
    error = float(np.sqrt(np.mean(r**2)))
    recovered = table(f)[7]
    demodulator_rows.append({
        "method": "local",
        "sigma_units": sig,
        "dW_rms": error,
        "Z7": recovered,
    })
    print(f"  local, sigma = {sig:.2f}          {error:.3e} wave      {recovered:.4f}")
for rad in (5.5, 6.6, 8.8):
    dW, mk, L = demodulate_fourier(fm, I, direction="x",
                                   method="lowpass", window_radius=rad)
    f, _ = fourier_to_wavefront(fm, I, indices=INDICES,
                                method="lowpass", window_radius=rad)
    r = dW[inner_x] - two_x[inner_x]
    error = float(np.sqrt(np.mean(r**2)))
    recovered = table(f)[7]
    demodulator_rows.append({
        "method": "lowpass",
        "window_radius_px": rad,
        "dW_rms": error,
        "Z7": recovered,
    })
    print(f"  lowpass, radius = {rad:4.1f} px  {error:.3e} wave      {recovered:.4f}")
print("  -> both keep the phase reference at each pixel; 'lowpass' (band-limited,")
print("     default) has the smaller bias, the Gaussian 'local' filter is smoother.")
REPORT["demodulator_sensitivity"] = demodulator_rows

# --------------------------------------------------------------------------- #
section("5. The dissertation's eq. (2-48) carrier: f0 = 2m/s")

# The project default carrier is m/(2s), i.e. one quarter of the value the
# dissertation derives for the Talbot plane (eq. 2-48 gives f0 = 2m/s = 45.6
# cyc/unit for s = 0.0439).  That full carrier needs a finer grid: the frame
# also contains the (+-1)x(-+1) beat at 2 f0 = 91.2 cyc/unit, which must stay
# below the grid Nyquist n/(4*extent); n > 8*1.10*45.6 = 401, so 512 is used.
cfg_paper = SystemConfig(grid=Grid(n=512, extent=1.10), period_um=30.0)
f0_paper = cfg_paper.carrier_frequency_paper
fm_paper = ForwardModel(cfg_paper)
print(f"  default carrier f0 = m/(2s) = {cfg_paper.carrier_f0:.2f} cyc/unit; "
      f"paper eq. (2-48) f0 = 2m/s = {f0_paper:.2f} cyc/unit = 4x that")
print(f"  grid 512x512 (extent 1.10): Nyquist {512 / (4 * 1.10):.1f} cyc/unit > "
      f"2 f0 = {2 * f0_paper:.1f}; a 256 grid would alias the 2 f0 beat")
truth_paper = ZernikeWavefront(np.array([0.5]), np.array([7]))
I_paper = fm_paper.ft_mode_frame(truth_paper, f0=f0_paper)
fit_paper, _ = fourier_to_wavefront(fm_paper, I_paper, f0=f0_paper,
                                    indices=INDICES)
tab_paper = table(fit_paper)
max_err_paper = max(
    abs(v - (0.5 if j == 7 else 0.0)) for j, v in tab_paper.items()
)
print("  recovered coefficients:",
      ", ".join(f"Z{j}={v:+.4f}" for j, v in tab_paper.items() if abs(v) > 1e-3))
print(f"  Z7 = {tab_paper[7]:+.4f}  (input 0.5),  max |error over all terms| = "
      f"{max_err_paper:.4f} wave")
print("  -> the demodulation mathematics is identical at both carriers; the")
print("     4x default is a sampling-driven choice for the default grid, not a")
print("     statement about the dissertation's Talbot-plane position.")
REPORT["carrier_paper_eq_2_48"] = {
    "grid_n": cfg_paper.grid.n,
    "f0_paper": f0_paper,
    "f0_default": cfg_paper.carrier_f0,
    "Z7_recovered": tab_paper[7],
    "max_coef_error": float(max_err_paper),
}

# --------------------------------------------------------------------------- #
section("6. Figures")

truth_ft = ZernikeWavefront(np.array([0.8]), np.array([7]))
fm_c = ForwardModel(SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0))
I = fm_c.ft_mode_frame(truth_ft)
dWx, mx, lx = demodulate_fourier(fm_c, I, direction="x")
dWy, my, ly = demodulate_fourier(fm_c, I, direction="y")
fit, _ = fourier_to_wavefront(fm_c, I, indices=INDICES)
from lsi.reconstruct import wavefront_on_grid

x, y = fm_c.config.grid.coords()
W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y)

fig, axes = new_fig(2, 3, figsize=(15, 9))
imshow(axes[0, 0], I, fm_c.config.grid, title="single frame, carrier f0 = %.2f cyc/unit" % fm_c.config.carrier_f0)
imshow(axes[0, 1], np.log10(np.abs(spectrum(I)) + 1e-9), fm_c.config.grid, title="log|spectrum| with +-f0 lobes")
imshow(axes[0, 2], lx.amplitude, fm_c.config.grid, title="|c(x,y)| of the +f0 lobe")
imshow(axes[1, 0], lx.phase, fm_c.config.grid, title="wrapped differential phase psi_x (rad)")
imshow(axes[1, 1], W_fit, fm_c.config.grid, title="W from one frame (waves)")
imshow(axes[1, 2], W_fit - truth_ft.w(x, y), fm_c.config.grid, title="error (waves)")
savefig(fig, "03_fourier_mode.png")

with open(os.path.join(OUT, "03_fourier_mode.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/03_fourier_mode.png and .json")