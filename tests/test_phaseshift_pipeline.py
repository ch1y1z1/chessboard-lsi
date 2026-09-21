"""End-to-end dissertation route:

    I(x,y) --(8-step phase shift)--> dW --(differential Zernike)--> W
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.metrics import coefficient_comparison, pv, rms, wavefront_error
from lsi.phaseshift import lsq_phase_shift, zero_order_center_radius
from lsi.pipeline import demodulate_phase_shift, phase_shift_to_wavefront, reconstruct
from lsi.reconstruct import wavefront_on_grid

CFG = SystemConfig(grid=Grid(n=128, extent=1.10))
INDICES = tuple(range(2, 14))          # piston excluded as in eq. (2-31)


def _coma():
    return ZernikeWavefront(np.array([1.0]), np.array([7]))


def _mixed():
    idx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    coeffs = np.array([0.0, 0.0, 0.22, -0.05, 0.11, 0.31, -0.12, 0.07, 0.04, 0.0, 0.0])
    return ZernikeWavefront(coeffs, idx)


def _run(truth, cfg=CFG, offset_mode="none", region_mode="analytic", n_steps=8,
         remove_offset=True):
    fm = ForwardModel(cfg)
    fx = fm.phase_shift_frames(truth, "x", n_steps)
    fy = fm.phase_shift_frames(truth, "y", n_steps)
    fit, diff = phase_shift_to_wavefront(
        fm, fx, fy, indices=INDICES, offset_mode=offset_mode,
        region_mode=region_mode, remove_offset=remove_offset,
    )
    return fm, fit, diff


# --------------------------------------------------------------------------- #
def test_recovers_single_coma_to_machine_precision():
    _, fit, _ = _run(_coma())
    table = fit.as_dict()
    assert table[7] == pytest.approx(1.0, abs=1e-10)
    assert max(abs(v) for k, v in table.items() if k != 7) < 1e-10
    assert fit.max_abs_residual < 1e-9


def test_recovers_mixed_aberration():
    _, fit, _ = _run(_mixed())
    truth = _mixed()
    got = fit.as_dict()
    for j, c in zip(truth.indices, truth.coeffs):
        assert got[int(j)] == pytest.approx(float(c), abs=1e-9), j


def test_wavefront_matches_truth_on_pupil():
    cfg = CFG
    truth = _mixed()
    _, fit, _ = _run(truth, cfg)
    x, y = cfg.grid.coords()
    pupil = cfg.grid.pupil()
    W_ref = truth.w(x, y)
    W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y)
    err = wavefront_error(W_fit, W_ref, pupil)
    assert err["max_abs"] < 1e-9


def test_pv_rms_against_dissertation():
    """Z7 = 1 wave: RMS 0.35355 (dissertation: 0.354) and PV = 2.0.

    ``Z7 = (3 rho^3 - 2 rho) cos(theta)`` reaches +1 at ``(1, 0)`` and -1 at
    ``(-1, 0)``, so PV = 2 exactly, RMS = 1/sqrt(8) = 0.353553.  The
    dissertation prints PV = 1.985 and RMS = 0.354: both agree with this
    basis once the finite grid (their 256-ish sampling) is taken into
    account.  On a 128 grid the sampled PV is 1.965.
    """
    cfg = CFG
    _, fit, _ = _run(_coma(), cfg)
    x, y = cfg.grid.coords()
    pupil = cfg.grid.pupil()
    W = wavefront_on_grid(fit.coeffs, fit.indices, x, y, pupil)
    assert rms(W, pupil) == pytest.approx(0.35355, abs=2e-3)
    assert pv(W, pupil) == pytest.approx(1.965, abs=5e-3)

    fine = SystemConfig(grid=Grid(n=512, extent=1.10))
    xf, yf = fine.grid.coords()
    from lsi.zernike import zernike_value

    Wf = zernike_value(7, xf, yf)
    pf = fine.grid.pupil()
    assert pv(Wf, pf) == pytest.approx(2.0, abs=5e-3)
    assert rms(Wf, pf) == pytest.approx(0.35355, abs=1e-3)


def test_modulation_circle_fit_region_selection():
    """The experimental region-selection procedure (modulation threshold +
    circle fit, eqs. 3-1 ... 3-5) must reproduce the analytic shear region."""
    cfg = CFG
    fm = ForwardModel(cfg)
    fx = fm.phase_shift_frames(_coma(), "x", 8)
    res = lsq_phase_shift(fx)
    cx, cy, edge = zero_order_center_radius(res.modulation, cfg.grid, threshold_frac=0.4)
    assert abs(cx) < 0.05 and abs(cy) < 0.05
    _, fit_a, _ = _run(_coma(), cfg, region_mode="analytic")
    _, fit_m, _ = _run(_coma(), cfg, region_mode="modulation")
    assert fit_m.as_dict()[7] == pytest.approx(1.0, abs=1e-6)
    assert fit_m.max_abs_residual < 1e-6
    assert fit_a.as_dict()[7] == pytest.approx(fit_m.as_dict()[7], abs=1e-6)


def test_tilt_is_degenerate_with_the_constant_offset():
    """A constant phase offset is indistinguishable from a tilt.

    If the half-fringe offset of the chessboard is ignored, the x-tilt
    absorbs it and comes out as 1/(2s) ~ 6.8 waves instead of 0.  The other
    coefficients are unaffected -- which is why the tilt has to be supplied
    from prior knowledge (or dropped) and the offset handled explicitly.
    """
    cfg = CFG
    _, fit_ok, _ = _run(_coma(), cfg)                      # offset removed
    _, fit_raw, _ = _run(_coma(), cfg, remove_offset=False)  # offset ignored
    # fitting the offset makes the design matrix rank deficient; the fit warns
    # instead of silently returning a min-norm answer
    with pytest.warns(UserWarning, match="rank deficient"):
        _, fit_est, _ = _run(_coma(), cfg, offset_mode="estimate")
    assert abs(fit_ok.as_dict()[2]) < 1e-9
    assert fit_raw.as_dict()[2] == pytest.approx(1.0 / (2 * cfg.s), rel=1e-6)
    # a free constant is exactly collinear with the tilt column: the system
    # becomes singular and the tilt is simply not determined by the data
    assert fit_est.cond > 1e6
    assert fit_ok.cond < 1e3
    for fit in (fit_ok, fit_raw, fit_est):
        assert fit.as_dict()[7] == pytest.approx(1.0, abs=1e-8)


def test_noise_sensitivity():
    cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
    fm = ForwardModel(cfg)
    truth = _coma()
    results = {}
    for snr in (60, 40, 30):
        fx = add_noise(fm.phase_shift_frames(truth, "x", 8), snr_db=snr, seed=1)
        fy = add_noise(fm.phase_shift_frames(truth, "y", 8), snr_db=snr, seed=2)
        fit, _ = phase_shift_to_wavefront(
            fm, fx, fy, indices=INDICES, offset_mode="model", unwrap="poisson"
        )
        results[snr] = fit.as_dict()[7]
    assert results[60] == pytest.approx(1.0, abs=0.02)
    assert results[40] == pytest.approx(1.0, abs=0.05)
    # monotone degradation
    assert abs(results[60] - 1) <= abs(results[30] - 1)


def test_phase_shift_step_count_variants():
    """The N-step algorithm must work for N = 4, 8, 12 (paper's alternative)."""
    cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
    for n_steps in (4, 8, 12):
        _, fit, _ = _run(_coma(), cfg, n_steps=n_steps)
        assert fit.as_dict()[7] == pytest.approx(1.0, abs=1e-8), n_steps


def test_modulation_flips_sign_for_large_aberration():
    """The dissertation's known limitation: for large aberration the
    modulation term 4 A0 A1 cos(pi[...]) changes sign inside the region, so a
    single global offset cannot fix the phase (the phase jumps by pi)."""
    cfg = CFG
    truth = ZernikeWavefront(np.array([6.0]), np.array([7]))
    fm = ForwardModel(cfg)
    diff = demodulate_phase_shift(
        fm,
        fm.phase_shift_frames(truth, "x", 8),
        fm.phase_shift_frames(truth, "y", 8),
    )
    x, y = cfg.grid.coords()
    pred = np.pi * (truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y))
    from lsi.unwrap import wrap

    resid = wrap(diff.phase["x"] - pred)[diff.mask["x"]]
    # a single constant cannot explain the residual any more
    frac = np.mean(np.abs(resid) > 1.0)
    assert frac > 0.05