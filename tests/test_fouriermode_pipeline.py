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
from lsi.unwrap import wrap


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


def test_two_sided_model_recovers_wavefront():
    cfg, fm = _fm()
    truth = _truth()
    I = fm.ft_mode_frame(truth)
    fit, diff = fourier_to_wavefront(fm, I, indices=tuple(range(2, 14)))
    table = fit.as_dict()
    # The band-limited demodulator retains a small filter/edge bias.
    assert table[7] == pytest.approx(0.6, abs=0.02)
    assert max(abs(v) for k, v in table.items() if k not in (2, 3, 7)) < 0.03

    x, y = cfg.grid.coords()
    pupil = diff.mask["x"] & diff.mask["y"]
    W_fit = wavefront_on_grid(fit.coeffs, fit.indices, x, y)
    err = wavefront_error(W_fit, truth.w(x, y), pupil)
    assert err["max_abs"] < 0.10


@pytest.mark.parametrize("index", [7, 9, 16])
def test_extracted_complex_lobe_phase_is_two_sided_for_curved_modes(index):
    """Both symmetric first orders contribute to the same ``+f0`` lobe."""
    extent = 1.1
    cfg = SystemConfig(
        grid=Grid(n=256, extent=extent),
        # Put the carrier exactly on FFT bin 24 to isolate model error from
        # sub-pixel carrier leakage in this physics-oracle regression.
        shear_ratio=(2.0 * extent) / (2.0 * 24.0),
    )
    fm = ForwardModel(
        cfg, pupil=lambda x, y: np.ones_like(x, dtype=bool)
    )
    truth = ZernikeWavefront([0.2], [index])
    flat = ZernikeWavefront([0.0], [index])
    phase_offset = fm.demodulation_offset("x")
    reference = demodulate_lobe(
        fm.ft_mode_frame(flat),
        cfg.grid,
        direction="x",
        f0=cfg.carrier_f0,
        phase_offset=phase_offset,
        window_radius=8.0,
    )
    lobe = demodulate_lobe(
        fm.ft_mode_frame(truth),
        cfg.grid,
        direction="x",
        f0=cfg.carrier_f0,
        phase_offset=phase_offset,
        window_radius=8.0,
    )

    x, y = cfg.grid.coords()
    s = cfg.s
    measured = np.angle(lobe.complex_field * np.conj(reference.complex_field))
    two_sided = np.pi * (
        truth.w(x + s, y) - truth.w(x - s, y)
    )
    wrong_one_sided = 2.0 * np.pi * (
        truth.w(x + s, y) - truth.w(x, y)
    )
    inner = (np.hypot(x, y) < 0.6) & (
        lobe.amplitude > 0.1 * lobe.amplitude.max()
    )

    def circular_rms(prediction):
        error = wrap(measured - prediction)[inner]
        gauge = np.angle(np.mean(np.exp(1j * error)))
        error = wrap(error - gauge)
        return float(np.sqrt(np.mean(error**2)))

    two_sided_error = circular_rms(two_sided)
    one_sided_error = circular_rms(wrong_one_sided)
    assert two_sided_error < 0.004
    assert two_sided_error < 0.3 * one_sided_error


def test_nuisance_offsets_model_prediction():
    cfg, fm = _fm()
    assert offset_in_dW(fm, "x", "one_sided") == pytest.approx(0.5)
    assert offset_in_dW(fm, "y", "one_sided") == pytest.approx(0.0)
    assert offset_in_dW(fm, "x", "two_sided") == pytest.approx(1.0)


def test_noise_robustness():
    cfg, fm = _fm()
    truth = _truth()
    I = add_noise(fm.ft_mode_frame(truth), snr_db=40, seed=5)
    fit, _ = fourier_to_wavefront(fm, I, indices=tuple(range(2, 14)))
    assert fit.as_dict()[7] == pytest.approx(0.6, abs=0.05)


def test_paper_carrier_frequency_on_fine_grid():
    """Run the FT pipeline at the dissertation's eq. (2-48) carrier 2 m / s.

    The paper carrier needs ``f0 = 2/s = 45.6`` cyc/unit *and* keeps the
    ``(+-1)x(-+1)`` beat at ``2 f0 = 91.2`` cyc/unit below the grid Nyquist
    ``n / (4 extent)``; that needs ``n > 8 * 1.10 * 45.6 = 401``, so the test
    uses a 512 grid (the 256 default would alias the 2f0 term).
    """
    cfg = SystemConfig(grid=Grid(n=512, extent=1.10), period_um=30.0)
    f0 = cfg.carrier_frequency_paper
    assert f0 < cfg.grid.n / (4.0 * cfg.grid.extent)
    assert 2.0 * f0 < cfg.grid.n / (4.0 * cfg.grid.extent)
    fm = ForwardModel(cfg)
    truth = ZernikeWavefront(np.array([0.5]), np.array([7]))
    I = fm.ft_mode_frame(truth, f0=f0)
    fit, _ = fourier_to_wavefront(
        fm, I, f0=f0, indices=tuple(range(2, 14))
    )
    table = fit.as_dict()
    errors = [
        abs(table[j] - (0.5 if j == 7 else 0.0)) for j in table
    ]
    assert max(errors) < 1e-2


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
    fit_ft, _ = fourier_to_wavefront(fm_ft, I, indices=tuple(range(2, 14)))
    assert fit_ps.as_dict()[7] == pytest.approx(0.5, abs=1e-6)
    assert fit_ft.as_dict()[7] == pytest.approx(0.5, abs=3e-2)
    assert fit_ft.as_dict()[7] == pytest.approx(fit_ps.as_dict()[7], abs=3e-2)