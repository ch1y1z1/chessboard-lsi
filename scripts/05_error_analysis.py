"""05 - error sources (dissertation chapter 4) reproduced in the forward model.

Studied here:

* grating duty cycle error (4.2.2) -- changes efficiency, not the phase,
* phase-shift step error delta (4.2.3 / the 4-step algorithm is much more
  sensitive than 8-step),
* shear (grating pitch) error -- the classic Zernike-fit scale error,
* detector noise,
* number of phase steps.

Run:  python3 scripts/05_error_analysis.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.grating import analytic_orders
from lsi.pipeline import phase_shift_to_wavefront
from lsi.plotting import new_fig, savefig

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
INDICES = tuple(range(2, 14))
REPORT: dict = {}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def run(cfg, truth, *, n_steps=8, orders=None, fit_orders=None, delta_error_deg=0.0,
        shear_error=0.0, noise_db=None, seed=0):
    fm = ForwardModel(cfg, orders) if orders is not None else ForwardModel(cfg)
    fm_fit = fm if fit_orders is None else ForwardModel(cfg, fit_orders)
    def frames(direction, jitter=False):
        tx = 1.0 if direction == "x" else 0.0
        ty = 1.0 - tx
        rng = np.random.default_rng(12345)
        out = []
        for k in range(n_steps):
            t = k / n_steps
            if delta_error_deg:
                if jitter:      # random step position error
                    t = t + np.deg2rad(delta_error_deg) / (2 * np.pi) * rng.standard_normal()
                else:           # mis-calibrated step size (linear drift)
                    t = t * (1.0 + np.deg2rad(delta_error_deg) / (2 * np.pi))
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
    # the shear is proportional to the grating pitch: a pitch error is a
    # relative shear error ds/s
    fm_fit = ForwardModel(SystemConfig(grid=cfg.grid, period_um=cfg.period_um * (1.0 + shear_error)))
    fit, _ = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
    if shear_error:
        # reconstruction uses the wrong shear: rebuild the differential data
        # with the nominal shear but fit with the perturbed one
        from lsi.pipeline import demodulate_phase_shift, reconstruct

        diff = demodulate_phase_shift(fm, fx, fy)
        fit = reconstruct(fm_fit, diff, indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def run_jitter(cfg, truth, n_steps, ddeg):
    fm = ForwardModel(cfg)
    rng = np.random.default_rng(12345)
    out = []
    for direction in ("x", "y"):
        tx = 1.0 if direction == "x" else 0.0
        ty = 1.0 - tx
        fr = []
        for k in range(n_steps):
            t = k / n_steps + np.deg2rad(ddeg) / (2 * np.pi) * rng.standard_normal()
            fr.append(fm.intensity(truth, deltas=fm.phase_shift_deltas(tx * t, ty * t)))
        out.append(np.array(fr))
    fit, _ = phase_shift_to_wavefront(fm, np.asarray(out[0]), np.asarray(out[1]), indices=INDICES)
    return {int(j): float(v) for j, v in zip(fit.indices, fit.coeffs)}


def maxerr(tab, truth, skip_tilt=True):
    """max |coefficient error|; the tilt terms (Z2, Z3) are reported separately
    because they are degenerate with the half-fringe constant of the grating."""
    terms = [(j, c) for j, c in zip(truth.indices, truth.coeffs)
             if not (skip_tilt and int(j) in (2, 3))]
    return max(abs(tab[int(j)] - float(c)) for j, c in terms)


def tilterr(tab, truth):
    out = 0.0
    for j, c in zip(truth.indices, truth.coeffs):
        if int(j) in (2, 3):
            out = max(out, abs(tab[int(j)] - float(c)))
    return out


cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
truth = ZernikeWavefront(np.array([0.0, 0.0, 0.22, 0.0, 0.31, -0.12]), np.array([2, 3, 4, 6, 7, 8]))
print(f"  truth: " + ", ".join(f"Z{int(j)}={float(c):+.3f}" for j, c in zip(truth.indices, truth.coeffs)))

# --------------------------------------------------------------------------- #
section("1. Grating duty-cycle error (4.2.2)")
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
    e, tl = maxerr(tab, truth), tilterr(tab, truth)
    en, tln = maxerr(tab_no, truth), tilterr(tab_no, truth)
    # every integer-lattice harmonic (not only the five beams), including the
    # even orders that appear away from 50 % duty -> higher-order crosstalk
    tab_all = run(cfg, truth, orders=analytic_orders(max_index=3, duty=duty))
    tiny = lambda z: 0.0 if abs(z) < 1e-12 else float(z)
    rows.append({"duty": duty, "A00": dc, "A10": abs(a1), "eff_pct": eff * 100,
                 "arg_A10": tiny(np.angle(a1)), "arg_A01": tiny(np.angle(a2)),
                 "max_coef_error": e, "tilt_error": tl,
                 "max_coef_error_nominal_prior": en, "tilt_error_nominal_prior": tln,
                 "max_coef_error_all_orders": maxerr(tab_all, truth),
                 "tilt_error_all_orders": tilterr(tab_all, truth)})
    print(f"  {duty:.2f}   {dc:.4f}   {abs(a1):.5f}   {tiny(np.angle(a1)):+8.4f}    {tiny(np.angle(a2)):+8.4f}"
          f"   {eff*100:6.2f}   | {e:.2e} {tl:7.2f} | {en:.2e}  {tln:6.2f}")
print("  -> the duty error changes only the *amplitudes*; the wavefront shape is exact")
print(f"     (<= {max(r['max_coef_error'] for r in rows):.1e} wave) whenever the amplitudes used for the")
print("     demodulation match the real grating.  The constant that a two-sided")
print("     shearing interferogram cannot separate from tilt moves as arg A_10 =")
print("     pi - 2 pi (d-1/2), arg A_01 = 0, so assuming an ideal 50 % grating puts")
print(f"     the whole error into tilt (up to {max(r['tilt_error_nominal_prior'] for r in rows):.1f} waves = 1/s;")
print("     the sign of the residual depends on where the 2 pi wrap falls).")
print("  -> the all-orders column (integer orders up to |a|,|b| <= 3) also shows")
print(f"     higher-order crosstalk: {rows[2]['max_coef_error_all_orders']:.1e} wave even at 50 % duty;")
print("     away from 50 %, newly non-zero even orders are included as well.")
REPORT["duty"] = rows

# --------------------------------------------------------------------------- #
section("2. Phase-shift step error (4.2.3)")
rows = []
print("  step error | mis-calibrated step size      | random step jitter")
print("     (deg)   |  4 steps       8 steps         |  4 steps       8 steps")
for ddeg in (0.0, 0.5, 1.0, 2.0, 5.0):
    e4 = maxerr(run(cfg, truth, n_steps=4, delta_error_deg=ddeg), truth)
    e8 = maxerr(run(cfg, truth, n_steps=8, delta_error_deg=ddeg), truth)
    j4 = maxerr(run(cfg, truth, n_steps=4, delta_error_deg=ddeg, orders=None), truth) if ddeg == 0 else         maxerr(run_jitter(cfg, truth, 4, ddeg), truth)
    j8 = maxerr(run_jitter(cfg, truth, 8, ddeg), truth)
    rows.append({"delta_deg": ddeg, "err_4step": e4, "err_8step": e8,
                 "jitter_4step": j4, "jitter_8step": j8})
    print(f"  {ddeg:6.1f}    | {e4:10.3e}  {e8:10.3e}     | {j4:10.3e}  {j8:10.3e}")
print("  -> both algorithms respond to a step-size mis-calibration at first order")
print("     (no advantage for more steps); random step jitter is comparable.")
REPORT["delta_error"] = rows

# --------------------------------------------------------------------------- #
section("3. Shear / grating-pitch error (scale error of the Zernike fit)")
rows = []
for eps in (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05):
    tab = run(cfg, truth, shear_error=eps)
    e = maxerr(tab, truth)
    rel = np.linalg.norm([tab[int(j)] for j in INDICES]) / np.linalg.norm([float(c) for c in truth.coeffs])
    rows.append({"shear_rel_error": eps, "max_coef_error": e, "coef_norm_ratio": float(rel)})
    print(f"  ds/s = {eps:+.3f} : max |coeff error| = {e:.4f} wave, "
          f"coefficient-norm ratio = {rel:.4f}")
print("  -> a shear error scales the whole coefficient vector (the shear is the")
print("     ruler of the measurement); the dominant term follows Z7 -> Z7(1+eps).")
REPORT["shear_error"] = rows

# --------------------------------------------------------------------------- #
section("4. Detector noise and number of phase steps")
rows = []
for n in (4, 8, 12):
    e_clean = maxerr(run(cfg, truth, n_steps=n), truth)
    noisy = [maxerr(run(cfg, truth, n_steps=n, noise_db=30, seed=s_), truth) for s_ in range(5)]
    rows.append({"n_steps": n, "err_clean": e_clean, "err_snr30_mean": float(np.mean(noisy))})
    print(f"  {n:2d} steps : clean {e_clean:.3e} wave,  SNR 30 dB (5 seeds) "
          f"{np.mean(noisy):.4f} +- {np.std(noisy):.4f} wave")
print("  -> averaging over the frames reduces the noise sensitivity with the number")
print("     of steps (the demodulation weights the frames as a matched filter).")
REPORT["steps"] = rows

# --------------------------------------------------------------------------- #
section("5. Figure")

fig, axes = new_fig(2, 2, figsize=(11, 8.5))
d = [r["delta_deg"] for r in REPORT["delta_error"]]
axes[0, 0].loglog(np.array(d) + 1e-3, np.array([r["err_8step"] for r in REPORT["delta_error"]]) + 1e-16, "o-", label="8 step")
axes[0, 0].loglog(np.array(d) + 1e-3, np.array([r["err_4step"] for r in REPORT["delta_error"]]) + 1e-16, "s-", label="4 step")
axes[0, 0].set_xlabel("phase-shift step error (deg)"), axes[0, 0].set_ylabel("max |coeff error| (wave)")
axes[0, 0].legend(), axes[0, 0].grid(alpha=0.3), axes[0, 0].set_title("step error (4.2.3)")

sh = [r["shear_rel_error"] for r in REPORT["shear_error"]]
axes[0, 1].plot(sh, [r["max_coef_error"] for r in REPORT["shear_error"]], "o-")
axes[0, 1].set_xlabel("relative shear error"), axes[0, 1].set_ylabel("max |coeff error| (wave)")
axes[0, 1].grid(alpha=0.3), axes[0, 1].set_title("shear (period) error")

du = [r["duty"] for r in REPORT["duty"]]
axes[1, 0].plot(du, [r["eff_pct"] for r in REPORT["duty"]], "o-", color="tab:blue",
                label=r"$\pm1$ order efficiency")
axes[1, 0].set_xlabel("grating duty"), axes[1, 0].set_ylabel("+-1 order efficiency (%)")
ax_du = axes[1, 0].twinx()
ax_du.plot(du, [r["tilt_error_nominal_prior"] for r in REPORT["duty"]], "s--",
           color="tab:red", label="tilt error, 50 % prior assumed")
ax_du.set_ylabel("tilt error (wave)", color="tab:red")
axes[1, 0].legend(loc="lower left", fontsize=7), ax_du.legend(loc="lower right", fontsize=7)
axes[1, 0].grid(alpha=0.3), axes[1, 0].set_title("duty error (4.2.2): shape exact, tilt only")

ns = [r["n_steps"] for r in REPORT["steps"]]
axes[1, 1].semilogy(ns, [r["err_snr30_mean"] for r in REPORT["steps"]], "o-")
axes[1, 1].set_xlabel("phase steps"), axes[1, 1].set_ylabel("max |coeff error| (wave), SNR 30 dB")
axes[1, 1].grid(alpha=0.3), axes[1, 1].set_title("noise averaging vs steps")
savefig(fig, "05_error_analysis.png")

with open(os.path.join(OUT, "05_error_analysis.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
print(f"  -> wrote {OUT}/05_error_analysis.png and .json")
