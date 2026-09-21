"""05 - selected dissertation chapter-4 error sources in the forward model.

Studied here:

* grating duty cycle error (4.1.1) -- changes complex order coefficients,
* phase-step scale calibration error and phase-position jitter (4.2), with
  their fractional and degree units kept distinct,
* shear-ratio error -- separated from the inverse grating-period error,
* detector noise,
* number of phase steps.

Run:  python3 scripts/05_error_analysis.py
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.grating import analytic_orders, bitmap_orders
from lsi.metrics import coefficient_error_metrics, coefficient_errors
from lsi.pipeline import phase_shift_to_wavefront
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
        step_scale_error_frac=0.0, shear_rel_error=0.0, noise_db=None, seed=0):
    fm = ForwardModel(cfg, orders) if orders is not None else ForwardModel(cfg)
    fm_fit = fm if fit_orders is None else ForwardModel(cfg, fit_orders)
    def frames(direction):
        tx = 1.0 if direction == "x" else 0.0
        ty = 1.0 - tx
        out = []
        for k in range(n_steps):
            t = k / n_steps
            if step_scale_error_frac:
                # Fractional calibration error of the nominal phase increment:
                # every step is multiplied by the same 1 + epsilon.
                t = t * (1.0 + step_scale_error_frac)
            d = fm.phase_shift_deltas(tx * t, ty * t)
            out.append(fm.intensity(truth, deltas=d))
        return np.array(out)

    fx, fy = frames("x"), frames("y")
    if noise_db is not None:
        fx = add_noise(fx, snr_db=noise_db, seed=seed)
        fy = add_noise(fy, snr_db=noise_db, seed=seed + 1)
    if fit_orders is not None:
        from lsi.pipeline import demodulate_phase_shift, reconstruct

        diff = demodulate_phase_shift(fm_fit, fx, fy)
        fit = reconstruct(fm_fit, diff, indices=INDICES)
        return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}
    fit, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
    if shear_rel_error:
        # reconstruction uses the wrong shear: rebuild the differential data
        # with the true shear but fit with s_fit = s_true * (1 + eps_s).
        from lsi.pipeline import demodulate_phase_shift, reconstruct

        cfg_fit = replace(
            cfg, shear_ratio=cfg.s * (1.0 + shear_rel_error)
        )
        fm_fit = ForwardModel(cfg_fit)
        diff = demodulate_phase_shift(fm, fx, fy)
        fit = reconstruct(fm_fit, diff, indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def run_jitter(cfg, truth, n_steps, jitter_std_deg):
    fm = ForwardModel(cfg)
    rng = np.random.default_rng(12345)
    out = []
    for direction in ("x", "y"):
        tx = 1.0 if direction == "x" else 0.0
        ty = 1.0 - tx
        fr = []
        for k in range(n_steps):
            t = (
                k / n_steps
                + jitter_std_deg / 360.0 * rng.standard_normal()
            )
            fr.append(fm.intensity(truth, deltas=fm.phase_shift_deltas(tx * t, ty * t)))
        out.append(np.array(fr))
    fit, _ = phase_shift_to_wavefront(fm, np.asarray(out[0]), np.asarray(out[1]), indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def error_metrics(tab, truth, skip_tilt=True):
    """Errors over every fitted mode; tilt is optionally reported separately."""
    return coefficient_error_metrics(
        tab,
        truth.indices,
        truth.coeffs,
        exclude_indices=(2, 3) if skip_tilt else (),
    )


def maxerr(tab, truth, skip_tilt=True):
    return error_metrics(tab, truth, skip_tilt)["max_error_all_modes"]


def tilterr(tab, truth):
    errors = coefficient_errors(tab, truth.indices, truth.coeffs)
    return max((abs(errors[j]) for j in (2, 3) if j in errors), default=0.0)


cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.22, 0.0, 0.31, -0.12]), np.array([2, 3, 4, 6, 7, 8]))
print(f"  truth: " + ", ".join(f"Z{int(j)}={float(c):+.3f}" for j, c in zip(truth.indices, truth.coeffs)))

# --------------------------------------------------------------------------- #
section("1. Grating duty-cycle error (4.1.1)")
print("  amplitudes use the closed form A_mn = S_m S_n/2 of lsi.grating;")
print("  the first comparison keeps the same five beams as the nominal model:")
print()
print("  duty   A(0,0)  |A(1,0)|  arg A(1,0)  arg A(0,1)  eff(4x+-1) | shape    tilt   | shape    tilt")
print("                              (rad)        (rad)        (%)     | prior ok  prior ok| 50% prior assumed")
rows = []
FIVE = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
NOMINAL = analytic_orders(duty=0.5).with_orders(FIVE)
for duty in (0.40, 0.45, 0.50, 0.55, 0.60):
    five = analytic_orders(duty=duty).with_orders(FIVE)
    dc = float(five.with_orders([(0, 0)]).amp[0].real)
    a1 = five.with_orders([(1, 0)]).amp[0]
    a2 = five.with_orders([(0, 1)]).amp[0]
    eff = 4.0 * abs(a1) ** 2
    tab = run(cfg, truth, orders=five)
    tab_no = run(cfg, truth, orders=five, fit_orders=NOMINAL)
    metric, tl = error_metrics(tab, truth), tilterr(tab, truth)
    metric_no, tln = error_metrics(tab_no, truth), tilterr(tab_no, truth)
    e = metric["max_error_all_modes"]
    en = metric_no["max_error_all_modes"]
    # every integer/half-integer detector harmonic (not only the five beams),
    # including orders that appear away from 50 % duty -> higher-order crosstalk
    tab_all = run(cfg, truth, orders=analytic_orders(max_index=3, duty=duty))
    tiny = lambda z: 0.0 if abs(z) < 1e-12 else float(z)
    metric_all = error_metrics(tab_all, truth)
    rows.append({
        "duty": duty, "A00": dc, "A10": abs(a1), "eff_pct": eff * 100,
        "arg_A10": tiny(np.angle(a1)), "arg_A01": tiny(np.angle(a2)),
        "max_coef_error": e,
        "max_leakage": metric["max_leakage_into_zero_modes"],
        "tilt_error": tl,
        "max_coef_error_nominal_prior": en,
        "max_leakage_nominal_prior": metric_no["max_leakage_into_zero_modes"],
        "tilt_error_nominal_prior": tln,
        "max_coef_error_all_orders": metric_all["max_error_all_modes"],
        "max_leakage_all_orders": metric_all["max_leakage_into_zero_modes"],
        "tilt_error_all_orders": tilterr(tab_all, truth),
    })
    print(f"  {duty:.2f}   {dc:.4f}   {abs(a1):.5f}   {tiny(np.angle(a1)):+8.4f}    {tiny(np.angle(a2)):+8.4f}"
          f"   {eff*100:6.2f}   | {e:.2e} {tl:7.2f} | {en:.2e}  {tln:6.2f}")
print("  -> duty changes the complex order coefficients: efficiency and, under")
print("     this fixed-edge unit-cell convention, an order-dependent constant phase.")
print("     The recovered wavefront shape is exact")
print(f"     (<= {max(r['max_coef_error'] for r in rows):.1e} wave) whenever the amplitudes used for the")
print("     demodulation match the real grating.  The constant that a two-sided")
print("     shearing interferogram cannot separate from tilt moves as arg A_10 =")
print("     pi - 2 pi (d-1/2), arg A_01 = 0, so assuming an ideal 50 % grating puts")
print(f"     the whole error into tilt (up to {max(r['tilt_error_nominal_prior'] for r in rows):.1f} waves = 1/s;")
print("     the sign of the residual depends on where the 2 pi wrap falls).")
print("  -> the all-orders column (integer/half orders up to |a|,|b| <= 3) also shows")
print(f"     higher-order crosstalk: {rows[2]['max_coef_error_all_orders']:.1e} wave even at 50 % duty;")
print("     away from 50 %, newly non-zero integer and half-integer orders are included.")
REPORT["duty"] = rows

print()
print("  relative y-placement error creates y-axis odd harmonics while the")
print("  orthogonal x-axis harmonics remain extinguished:")
print("  offset     |A(m=0,n=1)|  |A(m=0,n=3)|  max x-axis odd amplitude")
placement_rows = []
for offset in (0.01, 0.02, 0.04, 0.05):
    orders = bitmap_orders(harmonic_cell=400, max_index=2, offset_y=offset)
    ay1 = abs(orders.with_orders([(0.5, 0.5)]).amp[0])
    ay3 = abs(orders.with_orders([(1.5, 1.5)]).amp[0])
    ax1 = abs(orders.with_orders([(0.5, -0.5)]).amp[0])
    ax3 = abs(orders.with_orders([(1.5, -1.5)]).amp[0])
    placement_rows.append(
        {
            "offset_y": offset,
            "abs_A_m0_n1": ay1,
            "abs_A_m0_n3": ay3,
            "max_abs_A_modd_n0": max(ax1, ax3),
        }
    )
    print(f"  {offset:6.3f}       {ay1:10.6f}      {ay3:10.6f}          {max(ax1, ax3):.2e}")
print("  -> this is a relative sub-cell displacement, not a global grating shift.")
REPORT["pattern_offset_y"] = placement_rows

# --------------------------------------------------------------------------- #
section("2. Phase-step scale error and phase-position jitter (4.2)")
rows = []
print("  scale error | mis-calibrated step size      | position jitter (same number)")
print("      (%)     |  4 steps       8 steps         |  4 steps       8 steps (deg rms)")
for value in (0.0, 0.5, 1.0, 2.0, 5.0):
    scale_error = value / 100.0
    e4 = maxerr(
        run(cfg, truth, n_steps=4, step_scale_error_frac=scale_error), truth
    )
    e8 = maxerr(
        run(cfg, truth, n_steps=8, step_scale_error_frac=scale_error), truth
    )
    j4 = maxerr(run_jitter(cfg, truth, 4, value), truth)
    j8 = maxerr(run_jitter(cfg, truth, 8, value), truth)
    rows.append({
        "step_scale_error_pct": value,
        "phase_position_jitter_std_deg": value,
        "err_4step": e4,
        "err_8step": e8,
        "jitter_4step": j4,
        "jitter_8step": j8,
    })
    print(f"  {value:6.1f}    | {e4:10.3e}  {e8:10.3e}     | {j4:10.3e}  {j8:10.3e}")
print("  -> the left columns use a fractional step-scale error; the right columns")
print("     use an absolute per-frame phase-position jitter in degrees rms.")
REPORT["phase_shift_error"] = rows

# --------------------------------------------------------------------------- #
section("3. Shear-ratio error (distinct from inverse grating-period error)")
rows = []
for eps in (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05):
    tab = run(cfg, truth, shear_rel_error=eps)
    metric = error_metrics(tab, truth)
    e = metric["max_error_all_modes"]
    rel = np.linalg.norm([tab[int(j)] for j in INDICES]) / np.linalg.norm([float(c) for c in truth.coeffs])
    rows.append({
        "shear_rel_error": eps,
        "max_coef_error": e,
        "max_error_nonzero_truth_modes": metric["max_error_nonzero_truth_modes"],
        "max_leakage_into_zero_modes": metric["max_leakage_into_zero_modes"],
        "coef_norm_ratio": float(rel),
        "first_order_scale_prediction": 1.0 / (1.0 + eps),
    })
    print(f"  ds/s = {eps:+.3f} : max |coeff error| = {e:.4f} wave, "
          f"leakage = {metric['max_leakage_into_zero_modes']:.4f}, "
          f"coefficient-norm ratio = {rel:.4f}")
print("  -> to first order the dominant coefficient follows 1/(1+eps_s), but")
print("     finite shear and the full differential basis also permit modal coupling.")
print("     A grating-period error eps_p is a different variable:")
print("     eps_s = -eps_p/(1+eps_p), because s is proportional to 1/p.")
REPORT["shear_error"] = rows

# --------------------------------------------------------------------------- #
section("4. Detector noise and number of phase steps")
rows = []
for n in (4, 8, 12):
    e_clean = maxerr(run(cfg, truth, n_steps=n), truth)
    noisy = [maxerr(run(cfg, truth, n_steps=n, noise_db=30, seed=s_), truth) for s_ in range(5)]
    rows.append({"n_steps": n, "err_clean": e_clean, "err_snr30_mean": float(np.mean(noisy))})
    print(f"  {n:2d} steps : clean {e_clean:.3e} wave,  peak SNR 30 dB (5 seeds) "
          f"{np.mean(noisy):.4f} +- {np.std(noisy):.4f} wave")
print("  -> averaging over the frames reduces the noise sensitivity with the number")
print("     of steps (the demodulation weights the frames as a matched filter).")
REPORT["steps"] = rows

# --------------------------------------------------------------------------- #
section("5. Figure")

fig, axes = new_fig(2, 2, figsize=(11, 8.5))
d = [r["step_scale_error_pct"] for r in REPORT["phase_shift_error"]]
axes[0, 0].loglog(np.array(d) + 1e-3, np.array([r["err_8step"] for r in REPORT["phase_shift_error"]]) + 1e-16, "o-", label="8 step")
axes[0, 0].loglog(np.array(d) + 1e-3, np.array([r["err_4step"] for r in REPORT["phase_shift_error"]]) + 1e-16, "s-", label="4 step")
axes[0, 0].set_xlabel("phase-step scale error (%)"), axes[0, 0].set_ylabel("max |coeff error| (wave)")
axes[0, 0].legend(), axes[0, 0].grid(alpha=0.3), axes[0, 0].set_title("step error (4.2)")

sh = [r["shear_rel_error"] for r in REPORT["shear_error"]]
axes[0, 1].plot(sh, [r["max_coef_error"] for r in REPORT["shear_error"]], "o-")
axes[0, 1].set_xlabel("relative shear error"), axes[0, 1].set_ylabel("max |coeff error| (wave)")
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
axes[1, 0].grid(alpha=0.3), axes[1, 0].set_title("duty error (4.1.1): shape exact, tilt only")

ns = [r["n_steps"] for r in REPORT["steps"]]
axes[1, 1].semilogy(ns, [r["err_snr30_mean"] for r in REPORT["steps"]], "o-")
axes[1, 1].set_xlabel("phase steps"), axes[1, 1].set_ylabel("max |coeff error| (wave), peak SNR 30 dB")
axes[1, 1].grid(alpha=0.3), axes[1, 1].set_title("noise averaging vs steps")
savefig(fig, "05_error_analysis.png")

with open(os.path.join(OUT, "05_error_analysis.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/05_error_analysis.png and .json")
