"""05 - 论文第四章的误差来源在前向模型中的体现。

* 光栅占空比误差（4.1.1）——改变级次复振幅：形状仍精确，常数相位被
  tilt 吸收；"enlarged" 占空比模型复现表 4-1；
* 相移步长标定误差与相位位置抖动（4.2）；
* 剪切量误差（区别于光栅周期误差）；
* 探测器噪声与相移步数。

运行：  python3 scripts/05_error_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.grating import chessboard_orders, diffraction_efficiency
from lsi.metrics import coefficient_error_metrics, coefficient_errors
from lsi.pipeline import demodulate_phase_shift, phase_shift_to_wavefront, reconstruct
from lsi.plotting import new_fig, savefig

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {
    "noise_definition": "peak_snr_db = 20 log10(max(intensity) / noise_std)"
}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def run(cfg, truth, *, n_steps=8, orders=None, fit_orders=None,
        step_scale_error_frac=0.0, noise_db=None, seed=0):
    """生成两个方向的相移序列并重构，返回系数表。"""
    fm = ForwardModel(cfg, orders) if orders is not None else ForwardModel(cfg)
    fm_fit = fm if fit_orders is None else ForwardModel(cfg, fit_orders)

    def frames(direction):
        tx, ty = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
        out = []
        for k in range(n_steps):
            t = k / n_steps * (1.0 + step_scale_error_frac)
            out.append(fm.intensity(truth, deltas=fm.phase_shift_deltas(tx * t, ty * t)))
        return np.array(out)

    fx, fy = frames("x"), frames("y")
    if noise_db is not None:
        fx = add_noise(fx, snr_db=noise_db, seed=seed)
        fy = add_noise(fy, snr_db=noise_db, seed=seed + 1)
    if fit_orders is not None:
        # 重构与生成使用不同级次：连解调常数项也按假定（名义）光栅取，
        # 偏移误差因此落进 tilt。
        diff = demodulate_phase_shift(fm_fit, fx, fy)
        fit = reconstruct(fm_fit, diff, indices=INDICES)
    else:
        fit, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def run_jitter(cfg, truth, n_steps, jitter_std_deg, seed=12345):
    fm = ForwardModel(cfg)
    rng = np.random.default_rng(seed)
    out = []
    for direction in ("x", "y"):
        tx, ty = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
        fr = [
            fm.intensity(
                truth,
                deltas=fm.phase_shift_deltas(
                    tx * (k / n_steps + jitter_std_deg / 360.0 * rng.standard_normal()),
                    ty * (k / n_steps + jitter_std_deg / 360.0 * rng.standard_normal()),
                ),
            )
            for k in range(n_steps)
        ]
        out.append(np.array(fr))
    fit, _ = phase_shift_to_wavefront(fm, out[0], out[1], indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def error_metrics(tab, truth, skip_tilt=True):
    return coefficient_error_metrics(
        tab, truth.indices, truth.coeffs,
        exclude_indices=(2, 3) if skip_tilt else (),
    )


def maxerr(tab, truth, skip_tilt=True):
    return error_metrics(tab, truth, skip_tilt)["max_error_all_modes"]


def tilterr(tab, truth):
    errors = coefficient_errors(tab, truth.indices, truth.coeffs)
    return max((abs(errors[j]) for j in (2, 3) if j in errors), default=0.0)


cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.22, 0.0, 0.31, -0.12]),
                         np.array([2, 3, 4, 6, 7, 8]))
print("  truth: " + ", ".join(f"Z{int(j)}={float(c):+.3f}"
                             for j, c in zip(truth.indices, truth.coeffs)))

# --------------------------------------------------------------------------- #
section("1. 光栅占空比误差（4.1.1）")

FIVE = {(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}
nominal = {k: v for k, v in chessboard_orders(max_index=3, duty=0.5).items() if k in FIVE}
print("  duty   A(0,0)  |A(1,0)|  arg A(1,0)  eff(4x+-1) | shape    tilt   | "
      "shape    tilt")
print("                              (rad)        (%)     | 匹配先验        | "
      "假设 50% 先验")
rows = []
for duty in (0.40, 0.45, 0.50, 0.55, 0.60):
    five = {k: v for k, v in chessboard_orders(max_index=3, duty=duty).items()
            if k in FIVE}
    dc = float(five[(0.0, 0.0)].real)
    a1 = five[(1.0, 0.0)]
    eff = 4.0 * abs(a1) ** 2
    tab = run(cfg, truth, orders=five)
    tab_no = run(cfg, truth, orders=five, fit_orders=nominal)
    e, tl = maxerr(tab, truth), tilterr(tab, truth)
    en, tln = maxerr(tab_no, truth), tilterr(tab_no, truth)
    # |a|,|b| <= 3 的全部级次（含占空比偏离 50% 时新出现的偶数级）
    tab_all = run(cfg, truth, orders=chessboard_orders(max_index=3, duty=duty))
    metric_all = error_metrics(tab_all, truth)
    rows.append({
        "duty": duty, "A00": dc, "A10": abs(a1), "eff_pct": eff * 100,
        "arg_A10": float(np.angle(a1)),
        "max_coef_error": e, "tilt_error": tl,
        "max_coef_error_nominal_prior": en, "tilt_error_nominal_prior": tln,
        "max_coef_error_all_orders": metric_all["max_error_all_modes"],
    })
    print(f"  {duty:.2f}   {dc:.4f}   {abs(a1):.5f}   {np.angle(a1):+8.4f}"
          f"   {eff*100:6.2f}   | {e:.2e} {tl:7.2f} | {en:.2e}  {tln:6.2f}")
print("  -> 占空比改变级次复振幅；只要解调所用振幅与真实光栅一致，波前形状")
print(f"     仍精确（<= {max(r['max_coef_error'] for r in rows):.1e} wave）。双边剪切")
print("     无法与 tilt 区分的常数相位按 arg A_10 = pi - 2 pi (d-1/2) 漂移；")
print(f"     假设理想 50% 光栅会把全部误差放进 tilt（最多 "
      f"{max(r['tilt_error_nominal_prior'] for r in rows):.1f} waves ~ 1/s）。")
print(f"  -> 全级次列显示高级次串扰：50% 占空比下也有 "
      f"{rows[2]['max_coef_error_all_orders']:.1e} wave。")
REPORT["duty"] = rows

print()
print("  论文式 (4-1) 的占空比模型（'enlarged'：两个透明方块同步长到 w）：")
print("  (0,2) 级以 O(delta) 出现，功率比 |(0,2)/(1,1)|^2 可对照表 4-1:")
print("  占空比误差  |(0,2)/(1,1)|^2   表4-1      A00 = 2w^2")
TABLE_4_1 = {1: 0.254, 2: 1.056, 4: 4.593, 5: 7.489}
enlarged_rows = []
for pct in (1, 2, 4, 5):
    w = 0.5 + pct / 100.0
    eff = diffraction_efficiency(
        chessboard_orders(max_index=3, duty=w, model="enlarged"))["all"]
    # 探测器坐标 (a,b)=(1,1) 对应光栅坐标 (m,n)=(0,2)
    ratio_pct = eff.get((1.0, 1.0), 0.0) / eff[(1.0, 0.0)] * 100.0
    a00 = abs(eff.get((0.0, 0.0), 0.0)) ** 0.5
    enlarged_rows.append({"duty_error_pct": pct,
                          "power_ratio_02_over_11_pct": float(ratio_pct),
                          "table_4_1_pct": TABLE_4_1[pct], "A00": a00})
    print(f"  {pct:6.0f} %   {ratio_pct:12.4f} %  {TABLE_4_1[pct]:8.3f} %  {a00:.4f}")
REPORT["duty_enlarged_table4_1"] = enlarged_rows

# --------------------------------------------------------------------------- #
section("2. 相移步长标定误差与相位位置抖动（4.2）")

JITTER_SEEDS = 20
rows = []
print("  步长误差 | 标定误差下的步数          | 位置抖动（deg rms, "
      f"{JITTER_SEEDS} 种子均值）")
print("      (%)   |  4 步          8 步       |  4 步               8 步")
for value in (0.0, 0.5, 1.0, 2.0, 5.0):
    scale_error = value / 100.0
    e4 = maxerr(run(cfg, truth, n_steps=4, step_scale_error_frac=scale_error), truth)
    e8 = maxerr(run(cfg, truth, n_steps=8, step_scale_error_frac=scale_error), truth)
    j4_all = np.array([maxerr(run_jitter(cfg, truth, 4, value, seed=7000 + i), truth)
                       for i in range(JITTER_SEEDS)])
    j8_all = np.array([maxerr(run_jitter(cfg, truth, 8, value, seed=9000 + i), truth)
                       for i in range(JITTER_SEEDS)])
    rows.append({
        "step_scale_error_pct": value, "phase_position_jitter_std_deg": value,
        "err_4step": e4, "err_8step": e8,
        "jitter_4step": float(j4_all.mean()), "jitter_8step": float(j8_all.mean()),
        "jitter_4step_std": float(j4_all.std()), "jitter_8step_std": float(j8_all.std()),
    })
    print(f"  {value:6.1f}    | {e4:10.3e}  {e8:10.3e}     | "
          f"{j4_all.mean():10.3e}+-{j4_all.std():8.2e}  "
          f"{j8_all.mean():10.3e}+-{j8_all.std():8.2e}")
print("  -> 左列为步长的分数标定误差；右列为逐帧绝对相位位置抖动")
print(f"     （deg rms），对 {JITTER_SEEDS} 个种子取均值——单种子不足以排序 4/8 步。")
REPORT["phase_shift_error"] = rows

# --------------------------------------------------------------------------- #
section("3. 剪切量误差（区别于光栅周期误差）")

rows = []
for eps in (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05):
    fm_fit = ForwardModel(replace(cfg, shear_ratio=cfg.s * (1.0 + eps)))
    fm = ForwardModel(cfg)
    fx = fm.phase_shift_frames(truth, "x", 8)
    fy = fm.phase_shift_frames(truth, "y", 8)
    diff = demodulate_phase_shift(fm, fx, fy)
    fit = reconstruct(fm_fit, diff, indices=INDICES)
    tab = {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}
    metric = error_metrics(tab, truth)
    e = metric["max_error_all_modes"]
    rel = np.linalg.norm([tab[j] for j in INDICES]) / np.linalg.norm(truth.coeffs)
    rows.append({"shear_rel_error": eps, "max_coef_error": e,
                 "coef_norm_ratio": float(rel),
                 "first_order_scale_prediction": 1.0 / (1.0 + eps)})
    print(f"  ds/s = {eps:+.3f} : max |系数误差| = {e:.4f} wave, "
          f"系数范数比 = {rel:.4f} (一阶预测 {1.0/(1.0+eps):.4f})")
print("  -> 一阶近似下主导系数按 1/(1+eps_s) 缩放；有限剪切与完整差分基")
print("     还允许模间耦合。光栅周期误差 eps_p 是另一变量：")
print("     eps_s = -eps_p/(1+eps_p)，因为 s 正比于 1/p。")
REPORT["shear_error"] = rows

# --------------------------------------------------------------------------- #
section("4. 探测器噪声与相移步数")

rows = []
for n in (4, 8, 12):
    e_clean = maxerr(run(cfg, truth, n_steps=n), truth)
    noisy = [maxerr(run(cfg, truth, n_steps=n, noise_db=30, seed=s_), truth)
             for s_ in range(5)]
    rows.append({"n_steps": n, "err_clean": e_clean,
                 "err_snr30_mean": float(np.mean(noisy))})
    print(f"  {n:2d} 步 : 无噪声 {e_clean:.3e} wave,  峰值 SNR 30 dB (5 种子) "
          f"{np.mean(noisy):.4f} +- {np.std(noisy):.4f} wave")
print("  -> 帧数增加以匹配滤波方式平均噪声。")
REPORT["steps"] = rows

# --------------------------------------------------------------------------- #
section("5. 图")

fig, axes = new_fig(2, 2, figsize=(11, 8.5))
d = [r["step_scale_error_pct"] for r in REPORT["phase_shift_error"]]
axes[0, 0].loglog(np.array(d) + 1e-3,
                  np.array([r["err_8step"] for r in REPORT["phase_shift_error"]]) + 1e-16,
                  "o-", label="8 step")
axes[0, 0].loglog(np.array(d) + 1e-3,
                  np.array([r["err_4step"] for r in REPORT["phase_shift_error"]]) + 1e-16,
                  "s-", label="4 step")
axes[0, 0].set_xlabel("phase-step scale error (%)")
axes[0, 0].set_ylabel("max |coeff error| (wave)")
axes[0, 0].legend(), axes[0, 0].grid(alpha=0.3), axes[0, 0].set_title("step error (4.2)")

sh = [r["shear_rel_error"] for r in REPORT["shear_error"]]
axes[0, 1].plot(sh, [r["max_coef_error"] for r in REPORT["shear_error"]], "o-")
axes[0, 1].set_xlabel("relative shear error")
axes[0, 1].set_ylabel("max |coeff error| (wave)")
axes[0, 1].grid(alpha=0.3), axes[0, 1].set_title("reconstruction shear error")

du = [r["duty"] for r in REPORT["duty"]]
axes[1, 0].plot(du, [r["eff_pct"] for r in REPORT["duty"]], "o-", color="tab:blue",
                label=r"$\pm1$ order efficiency")
axes[1, 0].set_xlabel("grating duty"), axes[1, 0].set_ylabel("+-1 order efficiency (%)")
ax_du = axes[1, 0].twinx()
ax_du.plot(du, [r["tilt_error_nominal_prior"] for r in REPORT["duty"]], "s--",
           color="tab:red", label="tilt error, 50 % prior assumed")
ax_du.set_ylabel("tilt error (wave)", color="tab:red")
axes[1, 0].legend(loc="lower left", fontsize=7), ax_du.legend(loc="lower right", fontsize=7)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_title("duty error (4.1.1): shape exact, tilt only")

ns = [r["n_steps"] for r in REPORT["steps"]]
axes[1, 1].semilogy(ns, [r["err_snr30_mean"] for r in REPORT["steps"]], "o-")
axes[1, 1].set_xlabel("phase steps")
axes[1, 1].set_ylabel("max |coeff error| (wave), peak SNR 30 dB")
axes[1, 1].grid(alpha=0.3), axes[1, 1].set_title("noise averaging vs steps")
savefig(fig, "05_error_analysis.png")

with open(os.path.join(OUT, "05_error_analysis.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/05_error_analysis.png and .json")
