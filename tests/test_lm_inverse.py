"""Levenberg-Marquardt inversion of the non-linear forward model."""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig, preset_fourier, preset_phase_shift
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.lm import (
    LMConfig,
    fit_wavefront_from_carrier_frame,
    fit_wavefront_from_frames,
    levenberg_marquardt,
    multistart_fit,
)
from lsi.metrics import wavefront_error
from lsi.reconstruct import wavefront_on_grid

INDICES = tuple(range(2, 14))


def _fm():
    cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
    return cfg, ForwardModel(cfg)


def _frames(fm, truth, n_steps=8):
    fx = fm.phase_shift_frames(truth, "x", n_steps)
    fy = fm.phase_shift_frames(truth, "y", n_steps)
    frames = list(fx) + list(fy)
    deltas = [fm.phase_shift_deltas(i / n_steps, 0.0) for i in range(n_steps)]
    deltas += [fm.phase_shift_deltas(0.0, i / n_steps) for i in range(n_steps)]
    return frames, deltas


# --------------------------------------------------------------------------- #
def test_analytic_jacobian_matches_finite_differences():
    cfg, fm = _fm()
    fm9 = ForwardModel(cfg)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    rows = np.arange(cfg.grid.n**2)[::37]
    deltas = [fm9.phase_shift_deltas(0.125, 0.0)]
    c = np.linspace(-0.1, 0.1, len(INDICES))
    I, J = fm9.model_and_jacobian(c, proto, deltas, rows=rows)
    eps = 1e-6
    J_fd = np.zeros_like(J)
    for k in range(len(INDICES)):
        cp = c.copy()
        cp[k] += eps
        Ip, _ = fm9.model_and_jacobian(cp, proto, deltas, rows=rows)
        J_fd[:, k] = (Ip - I) / eps
    assert np.max(np.abs(J - J_fd)) < 1e-4 * max(1.0, np.max(np.abs(J_fd)))


def test_lm_recovers_wavefront_from_phase_shift_frames():
    cfg, fm = _fm()
    truth = ZernikeWavefront(np.array([0.0, 0.0, 0.31, -0.12, 0.07, 0.42, 0.05]),
                             np.array([2, 3, 4, 5, 6, 7, 8]))
    frames, deltas = _frames(fm, truth)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    res = fit_wavefront_from_frames(
        fm, proto, frames, deltas, samples=6000,
        config=LMConfig(max_iter=60, verbose=False),
    )
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    expected = {
        int(j): float(c) for j, c in zip(truth.indices, truth.coeffs)
    }
    for j in INDICES:
        assert table[j] == pytest.approx(expected.get(j, 0.0), abs=2e-4), j
    assert res.converged or res.cost < 1e-12
    assert res.rank == res.n_parameters


def test_lm_works_from_a_single_carrier_frame():
    """No phase shifting, no unwrapping: one interferogram is enough."""
    cfg = preset_fourier()
    fm = ForwardModel(cfg)
    truth = ZernikeWavefront(np.array([0.8]), np.array([7]))
    I = fm.ft_mode_frame(truth)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    res = fit_wavefront_from_carrier_frame(
        fm, proto, I, samples=8000, config=LMConfig(max_iter=80),
    )
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    for j in INDICES:
        expected = 0.8 if j == 7 else 0.0
        assert table[j] == pytest.approx(expected, abs=5e-3), j


def test_lm_beats_phase_shift_route_for_large_aberration():
    """For a large aberration the demodulated phase flips inside the shear
    region (section 2.3.2 limitation); LM is unaffected because it never
    builds a phase."""
    cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
    fm = ForwardModel(cfg)
    truth = ZernikeWavefront(np.array([6.0]), np.array([7]))
    frames, deltas = _frames(fm, truth)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    # the cost landscape has several minima at 6 waves: coarse scan first
    res = multistart_fit(
        fm, proto, frames, deltas, term=5, values=np.arange(-8.0, 8.1, 1.0),
        samples=4000, config=LMConfig(max_iter=120), coarse_iter=20,
    )
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    assert table[7] == pytest.approx(6.0, abs=5e-3)

    # for comparison: the phase-shift/Zernike route on the same data
    from lsi.pipeline import phase_shift_to_wavefront

    fx = frames[:8]
    fy = frames[8:]
    fit, _ = phase_shift_to_wavefront(fm, np.array(fx), np.array(fy), indices=INDICES)
    assert abs(fit.as_dict()[7] - 6.0) > 0.05


def test_lm_with_scale_and_background():
    cfg, fm = _fm()
    truth = ZernikeWavefront(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
                             np.array([2, 3, 4, 5, 6, 7, 8]))
    frames, deltas = _frames(fm, truth)
    frames = [2.3 * f + 0.17 for f in frames]          # unknown gain/offset
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    res = fit_wavefront_from_frames(
        fm, proto, frames, deltas, samples=6000,
        config=LMConfig(max_iter=80, fit_scale_background=True),
    )
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    assert table[7] == pytest.approx(0.5, abs=5e-4)
    assert res.scale == pytest.approx(2.3, rel=1e-3)
    assert res.background == pytest.approx(0.17, abs=1e-3)


def test_lm_noise_robustness():
    cfg, fm = _fm()
    truth = ZernikeWavefront(np.array([0.6]), np.array([7]))
    frames, deltas = _frames(fm, truth)
    frames = add_noise(np.array(frames), snr_db=40, seed=11)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    res = fit_wavefront_from_frames(fm, proto, frames, deltas, samples=8000,
                                    config=LMConfig(max_iter=80))
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    assert table[7] == pytest.approx(0.6, abs=0.02)


def test_lm_from_poor_initial_guess():
    cfg, fm = _fm()
    truth = ZernikeWavefront(np.array([1.5]), np.array([7]))
    frames, deltas = _frames(fm, truth)
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    res = fit_wavefront_from_frames(fm, proto, frames, deltas, samples=8000,
                                    config=LMConfig(max_iter=150, lambda0=1e-2))
    table = {int(j): v for j, v in zip(INDICES, res.x)}
    assert table[7] == pytest.approx(1.5, abs=2e-3)


@pytest.mark.parametrize("term", [-1, len(INDICES), 1.5, True])
def test_multistart_rejects_an_invalid_term(term):
    _, fm = _fm()
    proto = ZernikeWavefront(np.zeros(len(INDICES)), np.array(INDICES))
    frame = fm.intensity(proto)

    with pytest.raises(ValueError, match="term"):
        multistart_fit(
            fm,
            proto,
            [frame],
            [None],
            term=term,
            values=[0.0],
            samples=100,
        )


def test_generic_levenberg_marquardt_on_rosenbrock():
    def resid(x):
        f = np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])
        J = np.array([[-20 * x[0], 10], [-1, 0]])
        return f, J

    res = levenberg_marquardt(resid, [0.0, 0.0], LMConfig(max_iter=200))
    assert res.x[0] == pytest.approx(1.0, abs=1e-8)
    assert res.x[1] == pytest.approx(1.0, abs=1e-8)
    hist = res.history["cost"]
    assert len(hist) >= 3