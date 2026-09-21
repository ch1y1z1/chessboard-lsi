"""03 - dissertation route B: single-frame Fourier transform (spatial carrier).

    I(x,y)  --carrier f0 = m/2s-->  spectral lobe  --window-->  c = A e^{i psi}
            --arg-->  dW  --differential Zernike-->  W

The script also quantifies the model question: the isolated ``+f0`` lobe carries
the *one-sided* difference ``W(x+s)-W(x)``, while the dissertation's eq. (2-42)
reads it as the two-sided difference ``W(x+s)-W(x-s)``; the two differ by
``s^2/2 W_xx``, which is a genuine model error of order ``s^2``.

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
section("2. Demodulation of one frame: lobe, wrapped phase, one-sided vs two-sided")

I = fm.ft_mode_frame(truth)
dWx, mask_x, lobe_x = demodulate_fourier(fm, I, direction="x", difference_model="one_sided")
dWy, mask_y, lobe_y = demodulate_fourier(fm, I, direction="y", difference_model="one_sided")
x, y = cfg.grid.coords()

one = truth.w(x + cfg.s, y) - truth.w(x, y)
two_half = 0.5 * (truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y))
inner = mask_x & (np.hypot(x, y) < 0.7)
print(f"  lobe detected at {lobe_x.peak_freq} cyc/unit (nominal f0 = {cfg.carrier_f0:.3f})")
print(f"  region pixels: {int(mask_x.sum())} (x), {int(mask_y.sum())} (y)")
print(f"  one-sided reading  : rms(dW - [W(x+s)-W(x)]) = "
      f"{np.sqrt(np.mean((dWx - one)[inner] ** 2)):.3e} wave")
print(f"  two-sided reading  : rms(dW - [W(x+s)-W(x-s)]/2) = "
      f"{np.sqrt(np.mean((dWx - two_half)[inner] ** 2)):.3e} wave")
h = 1e-4
W_xx = (truth.w(x + h, y) - 2 * truth.w(x, y) + truth.w(x - h, y)) / h**2
print(f"  model difference   : max |[W(x+s)-W(x)] - [W(x+s)-W(x-s)]/2| = "
      f"{np.abs(one - two_half)[inner].max():.4f} wave")
print(f"                       predicted s^2/2 W_xx     = {np.abs(0.5*cfg.s**2*W_xx)[inner].max():.4f} wave")
REPORT["model_difference"] = {
    "one_sided_rms": float(np.sqrt(np.mean((dWx - one)[inner] ** 2))),
    "two_sided_rms": float(np.sqrt(np.mean((dWx - two_half)[inner] ** 2))),
    "max_model_gap": float(np.abs(one - two_half)[inner].max()),
}

# --------------------------------------------------------------------------- #
section("3. Reconstruction from one carrier frame (and with noise)")

fit, diff = fourier_to_wavefront(fm, I, indices=INDICES, difference_model="one_sided")
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
    f, _ = fourier_to_wavefront(fm, In, indices=INDICES, difference_model="one_sided")
    e = abs(table(f)[7] - 0.6)
    errs.append(e)
    print(f"  SNR {snr:3d} dB : |Z7 - 0.6| = {e:.4f} wave")
REPORT["noise"] = {"snr_db": snrs, "Z7_abs_error": errs}

# --------------------------------------------------------------------------- #
section("4. Choice of demodulator (all with the same physical phase reference)")

x_, y_ = cfg.grid.coords()
one_x = truth.w(x_ + cfg.s, y_) - truth.w(x_, y_)
inner_x = demodulate_fourier(fm, I, direction="x", difference_model="one_sided")[1] & (np.hypot(x_, y_) < 0.7)
print("  method                       rms(dW - one-sided)   Z7")
for sig in (0.04, 0.06, 0.10):
    dW, mk, L = demodulate_fourier(fm, I, direction="x", difference_model="one_sided",
                                   method="local", sigma_units=sig)
    f, _ = fourier_to_wavefront(fm, I, indices=INDICES, difference_model="one_sided",
                                method="local", sigma_units=sig)
    r = dW[inner_x] - one_x[inner_x]
    print(f"  local, sigma = {sig:.2f}          {np.sqrt(np.mean(r**2)):.3e} wave      {table(f)[7]:.4f}")
for rad in (5.5, 6.6, 8.8):
    dW, mk, L = demodulate_fourier(fm, I, direction="x", difference_model="one_sided",
                                   method="lowpass", window_radius=rad)
    f, _ = fourier_to_wavefront(fm, I, indices=INDICES, difference_model="one_sided",
                                method="lowpass", window_radius=rad)
    r = dW[inner_x] - one_x[inner_x]
    print(f"  lowpass, radius = {rad:4.1f} px  {np.sqrt(np.mean(r**2)):.3e} wave      {table(f)[7]:.4f}")
print("  -> both keep the phase reference at each pixel; 'lowpass' (band-limited,")
print("     default) has the smaller bias, the Gaussian 'local' filter is smoother.")

# --------------------------------------------------------------------------- #
section("5. Figures")

truth_ft = ZernikeWavefront(np.array([0.8]), np.array([7]))
fm_c = ForwardModel(SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0))
I = fm_c.ft_mode_frame(truth_ft)
dWx, mx, lx = demodulate_fourier(fm_c, I, direction="x", difference_model="one_sided")
dWy, my, ly = demodulate_fourier(fm_c, I, direction="y", difference_model="one_sided")
fit, _ = fourier_to_wavefront(fm_c, I, indices=INDICES, difference_model="one_sided")
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