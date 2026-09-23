"""02 - 论文路线 A：多步相移。

    I(x,y)  --8 步相移解调-->  缠绕 dW
            --区域选择-->  --解包裹-->  dW  --差分 Zernike-->  W

同时演示论文路线必须面对的两件事：

* 棋盘光栅的半条纹偏移（x 对与 y 对相差半条纹）——与 tilt 共线的常数，
* 大像差时的失效：调制度 4 A0 A1 cos(pi[...]) 在剪切区内变号（2.3.2）。

运行：  python3 scripts/02_phaseshift_pipeline.py
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
from lsi.metrics import coefficient_error_metrics, pv, rms
from lsi.phaseshift import find_pupil_circle, lsq_phase_shift
from lsi.pipeline import demodulate_phase_shift, phase_shift_to_wavefront, reconstruct
from lsi.plotting import imshow, new_fig, savefig
from lsi.reconstruct import wavefront_on_grid
from lsi.unwrap import wrap

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(fit) -> dict:
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


# --------------------------------------------------------------------------- #
section("1. 单 Zernike 项：W = Z7（彗差），1 波长，8 步相移")

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
print(f"  从 {2*8} 帧拟合 {len(INDICES)} 个 Zernike 项，耗时 {dt*1e3:.0f} ms")
print("  恢复的系数（waves）：")
for j, v in tab.items():
    flag = "   <-- 输入" if j == 7 else ""
    print(f"    Z{j:2d} : {v:+.16e}{flag}")
print(f"  最大 |系数误差| : {max(abs(v - (1.0 if j==7 else 0.0)) for j, v in tab.items()):.3e} wave")
print(f"  最大 |差分残差| : {fit.rms_residual:.3e} wave (rms)")
print(f"  最小二乘系统条件数 : {fit.cond:.1f}")
REPORT["single_term"] = {"fit": tab, "rms_residual": float(fit.rms_residual),
                         "cond": float(fit.cond), "time_ms": dt * 1e3}

# --------------------------------------------------------------------------- #
section("2. PV / RMS 对照论文（表 3-2/3-4：PV 1.985，RMS 0.354）")

x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y, pupil)
print(f"  采样 PV  = {pv(W_fit, pupil):.4f} wave   (论文 1.985)")
print(f"  采样 RMS = {rms(W_fit, pupil):.4f} wave   (论文 0.354)")
fine = SystemConfig(grid=Grid(n=512, extent=1.10))
Wf = ZernikeWavefront(np.array([1.0]), np.array([7])).w(*fine.grid.coords())
print(f"  解析基 PV = {pv(Wf, fine.grid.pupil()):.4f}, RMS = {rms(Wf, fine.grid.pupil()):.5f} "
      f"(Z7 = (3rho^3-2rho)cos(theta): PV = 2 精确, RMS = 1/sqrt(8))")
REPORT["pv_rms"] = {"pv_sampled": pv(W_fit, pupil), "rms_sampled": rms(W_fit, pupil)}

# --------------------------------------------------------------------------- #
section("3. 由调制度判定剪切区域（3.1.1，式 3-1 ... 3-5）")

res = lsq_phase_shift(fx)
cx, cy, radius = find_pupil_circle(res.modulation, cfg.grid, threshold_frac=0.4)
fit_m, diff_m = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES, region_mode="modulation")
print(f"  拟合的零级圆 : 圆心 = ({cx:+.5f}, {cy:+.5f}), 半径 = {radius:.5f} "
      f"(真值 0, 0, 1.000)")
print(f"  解析区域像素     : {int(diff.mask['x'].sum())}")
print(f"  圆拟合区域像素   : {int(diff_m.mask['x'].sum())}")
iou = float((diff.mask['x'] & diff_m.mask['x']).sum() / (diff.mask['x'] | diff_m.mask['x']).sum())
print(f"  区域重叠 (IoU)   : {iou:.4f}")
print(f"  圆拟合区域的 Z7  : {table(fit_m)[7]:+.10f}   (解析区域: {tab[7]:+.10f})")
REPORT["circle_fit"] = {"cx": cx, "cy": cy, "r": radius, "iou_with_analytic": iou}

# --------------------------------------------------------------------------- #
section("4. 混合像差 + 噪声敏感性")

truth_mix = ZernikeWavefront(np.array([0.0, 0.0, 0.22, -0.05, 0.11, 0.31, -0.12, 0.07, 0.04]),
                             np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
fx = fm.phase_shift_frames(truth_mix, "x", 8)
fy = fm.phase_shift_frames(truth_mix, "y", 8)
fit_mix, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
tab_mix = table(fit_mix)
worst = coefficient_error_metrics(tab_mix, truth_mix.indices, truth_mix.coeffs)["max_error_all_modes"]
print(f"  混合波前: 最大 |系数误差| = {worst:.3e} wave （{len(INDICES)} 个拟合模式）")

snr_list = [60, 50, 40, 30, 20, 10]
errs = []
for snr in snr_list:
    fx = add_noise(fm.phase_shift_frames(truth_mix, "x", 8), snr_db=snr, seed=1)
    fy = add_noise(fm.phase_shift_frames(truth_mix, "y", 8), snr_db=snr, seed=2)
    f, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
    errs.append(coefficient_error_metrics(table(f), truth_mix.indices, truth_mix.coeffs)["max_error_all_modes"])
    print(f"  SNR {snr:3d} dB : 最大 |系数误差| = {errs[-1]:.4f} wave")
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
section("5. 半条纹偏移：与 tilt 不可区分")

fd = demodulate_phase_shift(fm, fm.phase_shift_frames(truth, "x", 8),
                            fm.phase_shift_frames(truth, "y", 8), remove_offset=False)
print(f"  x 对的解调相位携带常数 {fm.demodulation_offset('x'):.6f} rad = 半条纹")
print(f"  y 对的解调相位携带常数 {fm.demodulation_offset('y'):.6f} rad")

for mode in ("raw", "model", "estimate"):
    d = demodulate_phase_shift(
        fm, fm.phase_shift_frames(truth, "x", 8), fm.phase_shift_frames(truth, "y", 8),
        remove_offset=False,
    )
    f = reconstruct(
        fm, d, indices=INDICES,
        offset_mode="none" if mode == "raw" else mode,
        # raw 分支关掉整数波规范修正，否则半条纹恰好是整数个波长被自动吸收
        resolve_tilt_gauge=(mode != "raw"),
    )
    t = table(f)
    print(f"  偏移处理 '{mode:8s}': Z2 = {t[2]:+9.4f}, Z7 = {t[7]:.6f}, cond = {f.cond:.1e}")
print(f"  (s = {cfg.s:.5f} 时，dW_x 中一波长的常数映射到 Z2 为 1/(2s) = {1/(2*cfg.s):.2f})")

# --------------------------------------------------------------------------- #
section("6. 已知限制：大像差破坏解调相位")

big = ZernikeWavefront(np.array([6.0]), np.array([7]))
db = demodulate_phase_shift(fm, fm.phase_shift_frames(big, "x", 8),
                            fm.phase_shift_frames(big, "y", 8))

pred = np.pi * (big.w(x + cfg.s, y) - big.w(x - cfg.s, y))
resid = wrap(db.phase["x"] - pred)[db.mask["x"]]
frac = float(np.mean(np.abs(wrap(resid - np.angle(np.mean(np.exp(1j * resid))))) > 1.0))
print(f"  W = 6 x Z7 : x 区域 {100*frac:.1f} % 的像素偏离 *单一* 常数超过 1 rad；")
print("  调制度项在区域内变号，解调相位出现 pi 跳变，全局偏移无法修复。")
fit_big, _ = phase_shift_to_wavefront(fm, fm.phase_shift_frames(big, "x", 8),
                                      fm.phase_shift_frames(big, "y", 8), indices=INDICES)
print(f"  该路线返回 Z7 = {table(fit_big)[7]:.4f} 而非 6.0 "
      f"(见脚本 04：LM 无此问题)")
print("  机理：对单个 Z7，解调路线在 ~3 波长内精确 (<1e-9)；超过 ~3.1 波长后")
print("  (±1,0)·(0,∓1)* 交叉项把调制度 |Z| 带过零点，缠绕相位出现涡旋，")
print("  解包裹丢失整波（3.5 波长丢 1 波，>= 4 丢 2 波）。")
REPORT["large_aberration"] = {"frac_bad_pixels": frac, "Z7_phase_shift": table(fit_big)[7]}

# --------------------------------------------------------------------------- #
section("7. 图")

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
