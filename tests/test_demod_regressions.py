"""Regressions for phase-step demodulation, pipeline plumbing and unwrapping.

Covered former defects:

* ``lsq_phase_shift`` used the closed form ``sum(I cos) / sum(I sin)``, which
  is only the least-squares solution for uniform orthogonal steps, and
  ``demodulate_phase_shift`` never passed calibrated steps on;
* ``demodulate_phase_shift`` recorded ``n_steps`` in the metadata without
  checking it and ignored ``erode_px`` entirely;
* unwrapping left every component but the seed's untouched for a disconnected
  mask (and raised for an empty one).
"""

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.phaseshift import lsq_phase_shift
from lsi.pipeline import demodulate_phase_shift
from lsi.unwrap import unwrap_masked_poisson, unwrap_seed_growth


# --------------------------------------------------------------- phase steps
def test_uniform_steps_reproduce_the_classic_closed_form():
    n = 4
    deltas = 2 * np.pi * np.arange(n) / n
    rng = np.random.default_rng(0)
    frames = 10.0 + 3.0 * np.cos(0.9 + deltas)[:, None] + 0.05 * rng.normal(size=(n, 7))
    res = lsq_phase_shift(frames)
    s_cos = np.sum(frames * np.cos(deltas)[:, None], axis=0)
    s_sin = np.sum(frames * np.sin(deltas)[:, None], axis=0)
    assert np.allclose(res.phase, np.arctan2(-s_sin, s_cos))
    assert np.allclose(res.modulation, 2.0 / n * np.hypot(s_cos, s_sin))
    assert np.allclose(res.background, frames.mean(axis=0))


def test_non_uniform_steps_need_a_real_least_squares_fit():
    deltas = np.array([0.0, 0.9, 2.6, 4.2])
    psi, mod, background = 0.7, 2.5, 4.0
    frames = background + mod * np.cos(psi + deltas)[:, None]
    fit = lsq_phase_shift(frames, deltas=deltas)
    assert np.isclose(float(fit.phase[0]), psi, atol=1e-9)
    assert np.isclose(float(fit.modulation[0]), mod, atol=1e-9)
    assert np.isclose(float(fit.background[0]), background, atol=1e-9)
    # assuming uniform steps instead is measurably wrong (0.12 vs 0.70 rad)
    naive = lsq_phase_shift(frames)
    assert abs(float(naive.phase[0]) - psi) > 0.3


def test_calibrated_steps_remove_a_step_calibration_error():
    truth = 0.9
    measured = 2 * np.pi * np.arange(4) / 4 + 0.15
    frames = 1.0 + 3.0 * np.cos(truth + measured)[:, None]
    naive = lsq_phase_shift(frames)
    calib = lsq_phase_shift(frames, deltas=measured)
    assert abs(float(naive.phase[0]) - truth) > 0.02
    assert np.isclose(float(calib.phase[0]), truth, atol=1e-9)


def test_lsq_phase_shift_validation():
    with pytest.raises(ValueError):
        lsq_phase_shift(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        lsq_phase_shift(np.array(5.0))
    with pytest.raises(ValueError, match="shape"):
        lsq_phase_shift(np.zeros((4, 3)), deltas=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        lsq_phase_shift(np.zeros((4, 3)), deltas=[0.0, np.nan, 1.0, 2.0])
    with pytest.raises(ValueError, match="basis"):
        lsq_phase_shift(np.zeros((4, 3)), deltas=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="basis"):
        lsq_phase_shift(np.zeros((4, 3)), deltas=[0.0, np.pi, 0.0, np.pi])


# ----------------------------------------------------------------- pipeline
def _phase_shift_data(n=96, steps=8):
    cfg = SystemConfig(grid=Grid(n=n, extent=1.10), phase_steps=steps)
    fm = ForwardModel(cfg)
    wf = ZernikeWavefront(np.array([0.2, -0.1]), np.array([4, 6]))
    return fm, fm.phase_shift_frames(wf, "x", steps), fm.phase_shift_frames(wf, "y", steps)


def test_n_steps_is_cross_checked_against_the_data():
    fm, frames_x, frames_y = _phase_shift_data()
    res = demodulate_phase_shift(fm, frames_x, frames_y, n_steps=8)
    assert res.meta["n_steps"] == 8
    with pytest.raises(ValueError, match="n_steps"):
        demodulate_phase_shift(fm, frames_x, frames_y, n_steps=99)


def test_erode_px_shrinks_the_shear_regions():
    fm, frames_x, frames_y = _phase_shift_data()
    base = demodulate_phase_shift(fm, frames_x, frames_y, n_steps=8)
    eroded = demodulate_phase_shift(fm, frames_x, frames_y, n_steps=8, erode_px=4)
    assert eroded.meta["erode_px"] == 4
    for key in ("x", "y"):
        assert 0 < int(eroded.mask[key].sum()) < int(base.mask[key].sum())
    with pytest.raises(ValueError, match="erode"):
        demodulate_phase_shift(fm, frames_x, frames_y, n_steps=8, erode_px=200)


def test_calibrated_steps_reach_the_pipeline_demodulation():
    fm, frames_x, frames_y = _phase_shift_data()
    uniform = list(2 * np.pi * np.arange(8) / 8)
    base = demodulate_phase_shift(fm, frames_x, frames_y, n_steps=8)
    same = demodulate_phase_shift(
        fm, frames_x, frames_y, n_steps=8, deltas_x=uniform, deltas_y=uniform
    )
    assert np.allclose(base.dW["x"], same.dW["x"], equal_nan=True)
    wrong = list(np.array(uniform) + 0.12)
    other = demodulate_phase_shift(
        fm, frames_x, frames_y, n_steps=8, deltas_x=wrong, deltas_y=wrong
    )
    assert not np.allclose(base.dW["x"], other.dW["x"], equal_nan=True)


# ------------------------------------------------------------------ unwrap
def _ramp(shape=(40, 40)):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    truth = 0.3 * x - 0.2 * y
    return truth, np.angle(np.exp(1j * truth))


def _two_region_mask():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:18, 2:18] = True
    mask[22:38, 22:38] = True
    labels = np.zeros_like(mask, dtype=int)
    labels[2:18, 2:18] = 1
    labels[22:38, 22:38] = 2
    return mask, labels


@pytest.mark.parametrize("unwrap", [unwrap_masked_poisson, unwrap_seed_growth])
def test_disconnected_mask_is_unwrapped_completely(unwrap):
    truth, phi = _ramp()
    mask, labels = _two_region_mask()
    with pytest.warns(UserWarning, match="disconnected"):
        out = unwrap(phi, mask)
    assert np.isfinite(out[mask]).all()
    # each region is reconstructed exactly, up to its own unknown constant
    for label in (1, 2):
        sel = labels == label
        error = out[sel] - truth[sel]
        assert np.ptp(error) < 1e-9


@pytest.mark.parametrize("unwrap", [unwrap_masked_poisson, unwrap_seed_growth])
def test_single_component_mask_does_not_warn(unwrap):
    truth, phi = _ramp()
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = unwrap(phi, mask)
    # the only free parameter is the reference value of the unwrapped phase
    assert np.ptp(out[mask] - truth[mask]) < 1e-9
    assert np.isfinite(out[mask]).all()
    assert np.isnan(out[~mask]).all()


@pytest.mark.parametrize("unwrap", [unwrap_masked_poisson, unwrap_seed_growth])
def test_empty_mask_returns_all_nan(unwrap):
    truth, phi = _ramp()
    out = unwrap(phi, np.zeros_like(truth, dtype=bool))
    assert out.shape == truth.shape
    assert np.isnan(out).all()


def test_unwrap_validation():
    truth, phi = _ramp()
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    with pytest.raises(ValueError, match="2-D"):
        unwrap_masked_poisson(np.zeros(9), None)
    with pytest.raises(ValueError, match="shape"):
        unwrap_masked_poisson(phi, np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="method"):
        unwrap_masked_poisson(phi, mask, method="zzz")
    with pytest.raises(ValueError, match="seed"):
        unwrap_seed_growth(phi, mask, seed=(99, 0))
    with pytest.raises(ValueError, match="seed"):
        unwrap_seed_growth(phi, mask, seed=(0, 0))
    with pytest.raises(ValueError, match="shape"):
        unwrap_seed_growth(phi, np.ones((3, 3), dtype=bool))