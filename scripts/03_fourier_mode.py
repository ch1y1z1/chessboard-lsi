"""03 - 论文路线 B：单帧傅里叶变换（空间载频）。

    I(x,y)  --载频 f0 = m/2s-->  频谱瓣  --带通-->  c = A e^{i psi}
            --arg-->  dW  --差分 Zernike-->  W

同时验证 +f0 瓣的物理内容：两个对称一级都在时，瓣内含 E+ E0* 与
E0 E-* 两个拍频，因此携带论文的双边差分 W(x+s) - W(x-s)。

运行：  python3 scripts/03_fourier_mode.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.ftmode import spectrum
from lsi.pipeline import demodulate_fourier, fourier_to_wavefront
from lsi.plotting import imshow, new_fig, savefig
from lsi.reconstruct import wavefront_on_grid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(fit):
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


section("1. 载频 vs 光栅周期（f0 = m / 2s）")
print("  剪切量由 45° 光栅的周期 p 决定；周期越大剪切越小，载频越密：")
rows = []
for p_um in (18.0, 30.0, 45.0, 60.0):
    cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=p_um, na=0.34, talbot_number=1)
    print("    p = %5.1f um -> s = %.5f, f0 = %6.2f cyc/unit (%.2f 条纹/光瞳), "
          "条纹间距 = %.2f px" % (p_um, cfg.s, cfg.carrier_f0, cfg.carrier_f0 * 2.0,
                                  cfg.grid.n / (2.0 * cfg.carrier_f0 * 2 * cfg.grid.extent)))
    rows.append({"p_um": p_um, "s": cfg.s, "f0": cfg.carrier_f0})
REPORT["periods"] = rows

cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
fm = ForwardModel(cfg)
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6]), np.array([2, 3, 4, 5, 6, 7]))
print(f"  工作配置: {cfg.describe()}")

# --------------------------------------------------------------------------- #
section("2. 单帧解调：+f0 瓣是双边差分")

I = fm.carrier_frame(truth)
dWx, mask_x, lobe_x = demodulate_fourier(fm, I, direction="x")
dWy, mask_y, lobe_y = demodulate_fourier(fm, I, direction="y")
x, y = cfg.grid.coords()

two = truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y)
wrong_one_doubled = 2.0 * (truth.w(x + cfg.s, y) - truth.w(x, y))
inner = mask_x & (np.hypot(x, y) < 0.7)
print(f"  瓣峰位于 {lobe_x.peak_freq} cyc/unit（名义 f0 = {cfg.carrier_f0:.3f}）")
print(f"  区域像素: {int(mask_x.sum())} (x), {int(mask_y.sum())} (y)")
print(f"  双边读数   : rms(dW - [W(x+s)-W(x-s)]) = "
      f"{np.sqrt(np.mean((dWx - two)[inner] ** 2)):.3e} wave")
print(f"  错误单边   : rms(dW - 2[W(x+s)-W(x)]) = "
      f"{np.sqrt(np.mean((dWx - wrong_one_doubled)[inner] ** 2)):.3e} wave")
h = 1e-4
W_xx = (truth.w(x + h, y) - 2 * truth.w(x, y) + truth.w(x - h, y)) / h**2
print(f"  错误模型差距 : max |2[W(x+s)-W(x)] - [W(x+s)-W(x-s)]| = "
      f"{np.abs(wrong_one_doubled - two)[inner].max():.4f} wave")
print(f"               预测 s^2 W_xx = {np.abs(cfg.s**2*W_xx)[inner].max():.4f} wave")
REPORT["model_difference"] = {
    "two_sided_rms": float(np.sqrt(np.mean((dWx - two)[inner] ** 2))),
    "wrong_one_sided_doubled_rms": float(
        np.sqrt(np.mean((dWx - wrong_one_doubled)[inner] ** 2))
    ),
}

# --------------------------------------------------------------------------- #
section("3. 单载频帧重构（及含噪情形）")

fit, diff = fourier_to_wavefront(fm, I, indices=INDICES)
tab = table(fit)
print("  恢复系数:",
      ", ".join(f"Z{j}={v:+.4f}" for j, v in tab.items() if abs(v) > 1e-3))
print(f"  Z7 = {tab[7]:+.4f}（输入 0.6），最大 |全项误差| = "
      f"{max(abs(v - (0.6 if j == 7 else 0.0)) for j, v in tab.items()):.4f} wave")
print(f"  差分残差 rms = {fit.rms_residual:.4f} wave")

errs = []
snrs = [60, 50, 40, 30, 20]
for snr in snrs:
    In = add_noise(fm.carrier_frame(truth), snr_db=snr, seed=7)
    f, _ = fourier_to_wavefront(fm, In, indices=INDICES)
    e = abs(table(f)[7] - 0.6)
    errs.append(e)
    print(f"  SNR {snr:3d} dB : |Z7 - 0.6| = {e:.4f} wave")
REPORT["noise"] = {"snr_db": snrs, "Z7_abs_error": errs}

# --------------------------------------------------------------------------- #
section("4. 低通窗半径的影响")

two_x = truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y)
inner_x = demodulate_fourier(fm, I, direction="x")[1] & (np.hypot(x, y) < 0.7)
print("  window_radius (px)     rms(dW - 双边)     Z7")
rows = []
for rad in (5.5, 6.6, 8.8):
    dW, mk, L = demodulate_fourier(fm, I, direction="x", window_radius=rad)
    f, _ = fourier_to_wavefront(fm, I, indices=INDICES, window_radius=rad)
    error = float(np.sqrt(np.mean((dW[inner_x] - two_x[inner_x]) ** 2)))
    print(f"  {rad:16.1f}     {error:.3e} wave   {table(f)[7]:.4f}")
    rows.append({"window_radius_px": rad, "dW_rms": error, "Z7": table(f)[7]})
REPORT["window_sensitivity"] = rows

# --------------------------------------------------------------------------- #
section("5. 论文式 (2-48) 的载频 f0 = 2m/s")

# 项目默认载频为 m/(2s)，即论文在泰伯位置推导值（式 2-48 给出
# f0 = 2m/s = 45.6 cyc/unit，s = 0.0439）的四分之一。完整载频需要更细
# 网格：帧中还含 (±1)x(∓1) 的 2f0 = 91.2 cyc/unit 拍频，必须低于网格
# 奈奎斯特 n/(4*extent)；n > 8*1.10*45.6 = 401，故取 512。
cfg_paper = SystemConfig(grid=Grid(n=512, extent=1.10), period_um=30.0)
f0_paper = cfg_paper.carrier_f0_paper
fm_paper = ForwardModel(cfg_paper)
print(f"  默认载频 f0 = m/(2s) = {cfg_paper.carrier_f0:.2f} cyc/unit；"
      f"论文式 (2-48) f0 = 2m/s = {f0_paper:.2f} cyc/unit = 4 倍")
print(f"  网格 512x512 (extent 1.10)：奈奎斯特 {512 / (4 * 1.10):.1f} cyc/unit > "
      f"2 f0 = {2 * f0_paper:.1f}；256 网格会使 2 f0 拍频混叠")
truth_paper = ZernikeWavefront(np.array([0.5]), np.array([7]))
I_paper = fm_paper.carrier_frame(truth_paper, f0=f0_paper)
fit_paper, _ = fourier_to_wavefront(fm_paper, I_paper, f0=f0_paper, indices=INDICES)
tab_paper = table(fit_paper)
max_err_paper = max(abs(v - (0.5 if j == 7 else 0.0)) for j, v in tab_paper.items())
print("  恢复系数:",
      ", ".join(f"Z{j}={v:+.4f}" for j, v in tab_paper.items() if abs(v) > 1e-3))
print(f"  Z7 = {tab_paper[7]:+.4f}（输入 0.5），最大 |全项误差| = {max_err_paper:.4f} wave")
REPORT["carrier_paper_eq_2_48"] = {
    "grid_n": cfg_paper.grid.n,
    "f0_paper": f0_paper,
    "f0_default": cfg_paper.carrier_f0,
    "Z7_recovered": tab_paper[7],
    "max_coef_error": float(max_err_paper),
}

# --------------------------------------------------------------------------- #
section("6. 图")

truth_ft = ZernikeWavefront(np.array([0.8]), np.array([7]))
fm_c = ForwardModel(SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0))
I = fm_c.carrier_frame(truth_ft)
dWx, mx, lx = demodulate_fourier(fm_c, I, direction="x")
dWy, my, ly = demodulate_fourier(fm_c, I, direction="y")
fit, _ = fourier_to_wavefront(fm_c, I, indices=INDICES)

x, y = fm_c.grid.coords()
W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y)

fig, axes = new_fig(2, 3, figsize=(15, 9))
imshow(axes[0, 0], I, fm_c.grid, title="single frame, carrier f0 = %.2f cyc/unit" % fm_c.config.carrier_f0)
imshow(axes[0, 1], np.log10(np.abs(spectrum(I)) + 1e-9), fm_c.grid, title="log|spectrum| with +-f0 lobes")
imshow(axes[0, 2], lx.amplitude, fm_c.grid, title="|c(x,y)| of the +f0 lobe")
imshow(axes[1, 0], lx.phase, fm_c.grid, title="wrapped differential phase psi_x (rad)")
imshow(axes[1, 1], W_fit, fm_c.grid, title="W from one frame (waves)")
imshow(axes[1, 2], W_fit - truth_ft.w(x, y), fm_c.grid, title="error (waves)")
savefig(fig, "03_fourier_mode.png")

with open(os.path.join(OUT, "03_fourier_mode.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/03_fourier_mode.png and .json")
