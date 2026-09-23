"""04 - Levenberg–Marquardt 直接非线性反演。

前向模型对波前是非线性的，

    I_k(i) = | sum_m A_m exp( i [ 2 pi W(x_i + s a_m, y_i + s b_m) + delta_km ] ) |^2,

因此可以 *不经* 相移、*不经* 解调、*不经* 解包裹直接求系数：

    c = argmin_c  sum_k || I_k - I_k(c) ||^2      用 LM（Marquardt 阻尼）。

本脚本展示：

1. 由 8+8 相移帧做 LM（与解调路线对照），
2. 由 *单帧* 载频干涉图做 LM，
3. 未知条纹对比度时的 LM（增益 + 背景一起拟合），
4. 噪声鲁棒性，
5. 大像差（6 波长）——解调路线失效处，
6. 收敛历史。

运行：  python3 scripts/04_lm_inverse.py
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
from lsi.lm import (
    LMConfig,
    fit_wavefront_from_carrier_frame,
    fit_wavefront_from_frames,
    multistart_fit,
)
from lsi.metrics import coefficient_error_metrics, coefficient_errors, rms
from lsi.pipeline import fourier_to_wavefront, phase_shift_to_wavefront
from lsi.plotting import imshow, new_fig, savefig
from lsi.reconstruct import wavefront_on_grid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(res):
    return {int(j): float(v) for j, v in zip(INDICES, res.x)}


def errs(tab, truth):
    return coefficient_errors(tab, truth.indices, truth.coeffs)


def err_metrics(tab, truth):
    return coefficient_error_metrics(tab, truth.indices, truth.coeffs)


# --------------------------------------------------------------------------- #
section("1. 由 8 + 8 相移帧做 LM（96 x 96 光瞳采样）")

cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
fm = ForwardModel(cfg)
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.31, -0.12, 0.07, 0.42, 0.05]),
                        np.array([2, 3, 4, 5, 6, 7, 8]))
fx = fm.phase_shift_frames(truth, "x", 8)
fy = fm.phase_shift_frames(truth, "y", 8)
deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]
frames = np.concatenate([np.asarray(fx), np.asarray(fy)], axis=0)

t0 = time.time()
res = fit_wavefront_from_frames(fm, INDICES, frames, deltas, samples=6000)
dt = time.time() - t0
tab = table(res)
print(f"  {res.n_iter} 次 LM 迭代, {len(res.x)} 参数, {dt:.2f} s")
print(f"  最终 cost = {res.cost:.3e} (rms 残差 = {res.rms_residual:.3e})")
print("  系数误差 (wave):",
      ", ".join(f"Z{j}:{e:+.2e}" for j, e in sorted(errs(tab, truth).items())))
metrics = err_metrics(tab, truth)
print(f"  全部 {len(INDICES)} 个拟合模式的最大 |误差| = {metrics['max_error_all_modes']:.2e}")
print(f"  向零真值模式的最大泄漏 = {metrics['max_leakage_into_zero_modes']:.2e}")

fit_ps, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
tab_ps = {int(j): float(v) for j, v in zip(fit_ps.indices, fit_ps.coeffs)}
print(f"  对照：相移解调路线给出 {err_metrics(tab_ps, truth)['max_error_all_modes']:.2e}")
REPORT["lm_phaseshift"] = {"n_iter": res.n_iter, "time_s": dt, **metrics}

x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_true = truth.w(x, y)
W_lm = wavefront_on_grid(res.x, INDICES, x, y, pupil)
print(f"  波前图误差: rms = {rms(W_lm - W_true, pupil):.3e} wave "
      f"(输入 W rms = {rms(W_true, pupil):.4f} wave)")

# --------------------------------------------------------------------------- #
section("2. 由单帧载频干涉图做 LM")

cfg_ft = SystemConfig(grid=Grid(n=128, extent=1.10), period_um=30.0)
fm_ft = ForwardModel(cfg_ft)
truth_ft = ZernikeWavefront(np.array([0.8]), np.array([7]))
I = fm_ft.carrier_frame(truth_ft)
t0 = time.time()
res1 = fit_wavefront_from_carrier_frame(fm_ft, INDICES, I, samples=8000,
                                        config=LMConfig(max_iter=80))
dt1 = time.time() - t0
tab1 = table(res1)
print(f"  {res1.n_iter} 次迭代, {dt1:.2f} s, cost {res1.cost:.2e}")
print(f"  Z7 = {tab1[7]:.6f} (输入 0.8), Z4 = {tab1[4]:+.6f}, Z5 = {tab1[5]:+.6f}")
print(f"  最大 |系数误差| = "
      f"{max(abs(v - (0.8 if j == 7 else 0.0)) for j, v in tab1.items()):.2e}")
print("  -> 一张干涉图，不相移、不解调、不解包裹；")
print("     与同帧解调路线（脚本 03）对照。")
REPORT["lm_single_frame"] = {"n_iter": res1.n_iter, "time_s": dt1,
                             "Z7": tab1[7],
                             "max_err": float(max(abs(v - (0.8 if j == 7 else 0.0))
                                                  for j, v in tab1.items()))}

# --------------------------------------------------------------------------- #
section("3. 未知对比度与背景（增益 + 偏移一起拟合）")

frames_scaled = [2.3 * f + 0.17 for f in frames]
res_sb = fit_wavefront_from_frames(
    fm, INDICES, frames_scaled, deltas, samples=6000,
    config=LMConfig(max_iter=80, fit_scale_background=True),
)
print(f"  拟合增益 = {res_sb.scale:.6f} (真值 2.3), 背景 = {res_sb.background:+.6f} "
      f"(真值 0.17)")
print(f"  最大 |系数误差| = "
      f"{err_metrics(table(res_sb), truth)['max_error_all_modes']:.2e}")
REPORT["lm_scale_background"] = {
    "scale": res_sb.scale,
    "background": res_sb.background,
    **err_metrics(table(res_sb), truth),
}

# 同样的缩放数据过解调路线：仿射增益 + 背景在对称 atan2 解调中抵消
# （整 2pi 周期上 sin/cos 求和为零），对傅里叶路线只加一个直流瓣。
fit_ps_sb, _ = phase_shift_to_wavefront(
    fm, np.asarray(frames_scaled[:8]), np.asarray(frames_scaled[8:]),
    indices=INDICES)
m_ps_sb = err_metrics(
    {int(j): float(v) for j, v in zip(fit_ps_sb.indices, fit_ps_sb.coeffs)}, truth)
print(f"  相移路线处理缩放帧: 最大 |系数误差| = {m_ps_sb['max_error_all_modes']:.2e}")

I_sb = 2.3 * I + 0.17
fit_ft_sb, _ = fourier_to_wavefront(fm_ft, I_sb, indices=INDICES)
m_ft_sb = err_metrics(
    {int(j): float(v) for j, v in zip(fit_ft_sb.indices, fit_ft_sb.coeffs)}, truth_ft)
print(f"  傅里叶路线处理缩放载频帧: 最大 |系数误差| = {m_ft_sb['max_error_all_modes']:.2e}")

res_ft_sb = fit_wavefront_from_carrier_frame(
    fm_ft, INDICES, I_sb, samples=8000,
    config=LMConfig(max_iter=80, fit_scale_background=True))
m_lm_ft_sb = err_metrics(table(res_ft_sb), truth_ft)
print(f"  LM 处理缩放载频帧: 增益 = {res_ft_sb.scale:.6f}, "
      f"背景 = {res_ft_sb.background:+.6f}, "
      f"最大 |系数误差| = {m_lm_ft_sb['max_error_all_modes']:.2e}")
REPORT["scale_background_routes"] = {
    "phase_shift": m_ps_sb,
    "fourier": m_ft_sb,
    "lm_carrier": {"scale": res_ft_sb.scale, "background": res_ft_sb.background,
                   **m_lm_ft_sb},
}

# --------------------------------------------------------------------------- #
section("4. 噪声")

noise_snrs = (60, 50, 40, 30, 20)
noise_metrics = []
for snr in noise_snrs:
    noisy = add_noise(frames, snr_db=snr, seed=3)
    r = fit_wavefront_from_frames(fm, INDICES, noisy, deltas, samples=6000,
                                  config=LMConfig(max_iter=80))
    metric = err_metrics(table(r), truth)
    noise_metrics.append(metric)
    print(f"  SNR {snr:3d} dB : 最大全模式误差 = "
          f"{metric['max_error_all_modes']:.4f} wave "
          f"(零模式泄漏 {metric['max_leakage_into_zero_modes']:.4f})")
REPORT["lm_noise"] = {
    "snr_db": list(noise_snrs),
    "max_error_all_modes": [m["max_error_all_modes"] for m in noise_metrics],
    "max_leakage_into_zero_modes": [m["max_leakage_into_zero_modes"] for m in noise_metrics],
}

# --------------------------------------------------------------------------- #
section("5. 大像差（6 波长彗差）：LM vs 解调")

big = ZernikeWavefront(np.array([6.0]), np.array([7]))
fxb = fm.phase_shift_frames(big, "x", 8)
fyb = fm.phase_shift_frames(big, "y", 8)
res_big = multistart_fit(
    fm, INDICES, np.concatenate([np.asarray(fxb), np.asarray(fyb)], axis=0), deltas,
    term=5, values=np.arange(-8.0, 8.1, 1.0), samples=4000,
    config=LMConfig(max_iter=150), coarse_iter=20,
)
print(f"  LM（粗扫描 + 精化）  : Z7 = {table(res_big)[7]:.5f}   (输入 6.0)")
fit_big, diff_big = phase_shift_to_wavefront(fm, fxb, fyb, indices=INDICES)
fit_raw, diff_raw = phase_shift_to_wavefront(
    fm, fxb, fyb, indices=INDICES, resolve_tilt_gauge=False)


def _row(fit, diff):
    t = {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}
    return t, diff.meta["tilt_gauge"]


t_on, k_on = _row(fit_big, diff_big)
t_off, k_off = _row(fit_raw, diff_raw)
print(f"  解调路线              : Z7 = {t_on[7]:.5f}")
print(f"    tilt 规范修正 ON  : Z2 = {t_on[2]:+.4f}, Z3 = {t_on[3]:+.4f}, "
      f"k_x = {k_on['k_x']}, k_y = {k_on['k_y']}")
print(f"    tilt 规范修正 OFF : Z2 = {t_off[2]:+.4f}, Z3 = {t_off[3]:+.4f}, "
      f"k_x = {k_off['k_x']}, k_y = {k_off['k_y']}")
print("    未修正时的大 Z2 是解包裹规范：种子像素真实相位距其缠绕值超过")
print("    1 条纹，整张 dW 图差整数个波长，常数被 tilt 列吸收。")
print("    （区别于调制度过零涡旋 —— 后者还破坏 6 波长处的*形状*。）")
print("    LM 不受影响（无揭包裹步骤）。")
REPORT["lm_large"] = {"LM": table(res_big)[7], "phaseshift": t_on[7],
                      "phaseshift_Z2": t_on[2], "phaseshift_Z3": t_on[3],
                      "tilt_gauge_k_x": k_on["k_x"], "tilt_gauge_k_y": k_on["k_y"]}

# --------------------------------------------------------------------------- #
section("6. 图")

fig, axes = new_fig(2, 3, figsize=(15, 9))
imshow(axes[0, 0], frames[0], cfg.grid, title="input frame (t=0, x pair)")
imshow(axes[0, 1], W_true, cfg.grid, title="true W (waves)")
imshow(axes[0, 2], W_lm, cfg.grid, title="LM reconstruction (waves)")
imshow(axes[1, 0], W_lm - W_true, cfg.grid, title="LM error (waves)")
ax = axes[1, 1]
ax.semilogy(res.history["cost"], "o-", ms=3)
ax.set_xlabel("iteration"), ax.set_ylabel("cost")
ax.set_title("LM convergence (phase-shift frames)")
ax.grid(alpha=0.3)
ax = axes[1, 2]
ax.semilogy(res1.history["cost"], "s-", ms=3, color="C1")
ax.set_xlabel("iteration"), ax.set_ylabel("cost")
ax.set_title("LM convergence (single carrier frame)")
ax.grid(alpha=0.3)
savefig(fig, "04_lm_inverse.png")

with open(os.path.join(OUT, "04_lm_inverse.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/04_lm_inverse.png and .json")
