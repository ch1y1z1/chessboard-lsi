"""Fourier-transform (spatial carrier) route, dissertation section 2.4.1."""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig, preset_fourier, preset_phase_shift
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.ftmode import demodulate_lobe, find_carrier_peak, spectrum
from lsi.metrics import wavefront_error
from lsi.pipeline import demodulate_fourier, fourier_to_wavefront, offset_in_dW
from lsi.reconstruct import wavefront_on_grid


def _fm(cfg=None):
    cfg = cfg or preset_fourier()
    return cfg, ForwardModel(cfg)


def _truth(scale=0.6):
    return ZernikeWavefront(np.array([scale]), np.array([7]))


def test_carrier_peaks_are_where_theory_says():
    cfg, fm = _fm()
    I = fm.ft_mode_frame(_truth())
    spec = spectrum(I)
    n = cfg.grid.n
    L = 2 * cfg.grid.extent
    f0 = cfg.carrier_f0
    for direction, expected in (("x", (f0, 0.0)), ("y", (0.0, f0))):
        lobe = demodulate_lobe(I, cfg.grid, direction=direction, f0=f0)
        fx, fy = lobe.peak_freq
        assert fx == pytest.approx(expected[0], abs=1.5 / L)
        assert fy == pytest.approx(expected[1], abs=1.5 / L)
    # explicit peak search without prior knowledge
    i0, j0 = find_carrier_peak(spec, cfg.grid)
    assert max(abs(i0 - n // 2), abs(j0 - n // 2)) > 5


def test_one_sided_model_recovers_wavefront():
    cfg, fm = _fm()
    truth = _truth()
    I = fm.ft_mode_frame(truth)
    fit, diff = fourier_to_wavefront(fm, I, indices=tuple(range(2, 14)),
                                     difference_model="one_sided")
    table = fit.as_dict()
    # the windowed-FT demodulator carries an O(sigma^2) smoothing bias
    assert table[7] == pytest.approx(0.6, abs=0.02)
    assert max(abs(v) for k, v in table.items() if k not in (2, 3, 7)) < 0.06

    x, y = cfg.grid.coords()
    pupil = diff.mask["x"] & diff.mask["y"]
    W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y)
    err = wavefront_error(W_fit, truth.w(x, y), pupil)
    assert err["max_abs"] < 0.10


def test_one_sided_vs_two_sided_model_differ_by_s_squared_curvature():
    """The dissertation reads the isolated +f0 lobe as the two-sided
    difference (eq. 2-42).  The demodulated phase really carries the
    *one-sided* difference ``W(x+s)-W(x)``; the two readings differ by

        [W(x+s) - W(x)] - [W(x+s) - W(x-s)] / 2
            = s^2/2 W_xx + O(s^4),

    i.e. by an O(s^2) term that is a real model error, not noise."""
    cfg, fm = _fm()
    truth = _truth(0.6)
    x, y = cfg.grid.coords()
    s = cfg.s
    one = truth.w(x + s, y) - truth.w(x, y)
    two_half = 0.5 * (truth.w(x + s, y) - truth.w(x - s, y))
    diff = one - two_half
    h = 1e-4
    W_xx = (truth.w(x + h, y) - 2 * truth.w(x, y) + truth.w(x - h, y)) / h**2
    pred = 0.5 * s**2 * W_xx
    inner = np.hypot(x, y) < 0.7
    scale = np.abs(pred[inner]).max()
    assert scale > 1e-3                      # the model error is substantial
    assert np.abs(diff - pred)[inner].max() / scale < 1e-2


def test_nuisance_offsets_model_prediction():
    cfg, fm = _fm()
    assert offset_in_dW(fm, "x", "one_sided") == pytest.approx(0.5)
    assert offset_in_dW(fm, "y", "one_sided") == pytest.approx(0.0)
    assert offset_in_dW(fm, "x", "two_sided") == pytest.approx(1.0)


def test_noise_robustness():
    cfg, fm = _fm()
    truth = _truth()
    I = add_noise(fm.ft_mode_frame(truth), snr_db=40, seed=5)
    fit, _ = fourier_to_wavefront(fm, I, indices=tuple(range(2, 14)),
                                  difference_model="one_sided")
    assert fit.as_dict()[7] == pytest.approx(0.6, abs=0.05)


def test_ft_and_ps_models_use_the_same_forward_model():
    """Both routes must reconstruct the same wavefront from the same truth."""
    cfg_ps, fm_ps = preset_phase_shift(), None
    fm_ps = ForwardModel(cfg_ps)
    cfg_ft = preset_fourier()
    fm_ft = ForwardModel(cfg_ft)
    truth = ZernikeWavefront(np.array([0.0, 0.5]), np.array([2, 7]))

    from lsi.pipeline import phase_shift_to_wavefront

    fx = fm_ps.phase_shift_frames(truth, "x", 8)
    fy = fm_ps.phase_shift_frames(truth, "y", 8)
    fit_ps, _ = phase_shift_to_wavefront(fm_ps, fx, fy, indices=tuple(range(2, 14)))
    I = fm_ft.ft_mode_frame(truth)
    fit_ft, _ = fourier_to_wavefront(fm_ft, I, indices=tuple(range(2, 14)),
                                     difference_model="one_sided")
    assert fit_ps.as_dict()[7] == pytest.approx(0.5, abs=1e-6)
    assert fit_ft.as_dict()[7] == pytest.approx(0.5, abs=3e-2)
    assert fit_ft.as_dict()[7] == pytest.approx(fit_ps.as_dict()[7], abs=3e-2)