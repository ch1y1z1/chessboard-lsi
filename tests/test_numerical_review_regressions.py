"""Regression tests for numerical issues found in the focused review."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront, add_noise
from lsi.ftmode import demodulate_lobe
from lsi.grating import analytic_orders, bitmap_orders
from lsi.lm import _solve_damped, fit_wavefront_from_frames
from lsi.phaseshift import lsq_phase_shift, shear_region_masks
from lsi.pipeline import fourier_to_wavefront, phase_shift_to_wavefront


def test_known_carrier_does_not_remove_physical_defocus_slope():
    cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
    fm = ForwardModel(cfg)

    for truth in (0.0, 0.2, 0.5):
        wf = ZernikeWavefront(np.array([truth]), np.array([4]))
        fit, _ = fourier_to_wavefront(
            fm, fm.ft_mode_frame(wf), indices=[4], method="lowpass", erode_px=0
        )
        assert fit.coeffs[0] == pytest.approx(truth, abs=2e-3)


def test_phase_shift_offset_is_removed_before_the_pi_branch_cut():
    cfg = SystemConfig(grid=Grid(n=64, extent=1.10))
    fm = ForwardModel(cfg)
    flat = ZernikeWavefront(np.zeros(2), np.array([2, 3]))
    fx = fm.phase_shift_frames(flat, "x")
    fy = fm.phase_shift_frames(flat, "y")

    # These seeds put the old x-direction seed on the -pi side.  Subtracting
    # +pi after unwrapping then created a -2pi map and a -1/s fake x tilt.
    for seed in (0, 1, 2, 3, 6, 7):
        fit, diff = phase_shift_to_wavefront(
            fm,
            add_noise(fx, snr_db=120, seed=seed),
            add_noise(fy, snr_db=120, seed=100 + seed),
            indices=[2, 3],
        )
        assert np.max(np.abs(fit.coeffs)) < 1e-5
        assert abs(np.nanmedian(diff.dW["x"][diff.mask["x"]])) < 1e-6


def test_raw_phase_offset_path_uses_the_same_stable_branch():
    cfg = SystemConfig(grid=Grid(n=64, extent=1.10))
    fm = ForwardModel(cfg)
    flat = ZernikeWavefront(np.zeros(2), np.array([2, 3]))
    fx = add_noise(fm.phase_shift_frames(flat, "x"), snr_db=120, seed=0)
    fy = add_noise(fm.phase_shift_frames(flat, "y"), snr_db=120, seed=100)

    fit, _ = phase_shift_to_wavefront(
        fm,
        fx,
        fy,
        indices=[2, 3],
        remove_offset=False,
        offset_mode="model",
    )
    assert np.max(np.abs(fit.coeffs)) < 1e-5


def test_lm_rejects_empty_observation_sets():
    cfg = SystemConfig(grid=Grid(n=16))
    fm = ForwardModel(cfg)
    proto = ZernikeWavefront(np.zeros(1), np.array([4]))
    image = fm.intensity(proto)

    with pytest.raises(ValueError, match="samples must be positive"):
        fit_wavefront_from_frames(fm, proto, [image], samples=0)
    with pytest.raises(ValueError, match="no observations"):
        fit_wavefront_from_frames(
            fm, proto, [image], samples=None, pixel_mask=np.zeros(fm.shape, bool)
        )


def test_carrier_above_nyquist_is_rejected():
    cfg = SystemConfig(grid=Grid(n=16, extent=1.10))
    fm = ForwardModel(cfg)
    wf = ZernikeWavefront(np.zeros(1), np.array([4]))
    assert cfg.carrier_f0 >= 1.0 / (2.0 * cfg.grid.dx)

    with pytest.raises(ValueError, match="Nyquist"):
        fm.ft_mode_frame(wf)
    with pytest.raises(ValueError, match="Nyquist"):
        demodulate_lobe(
            np.zeros(fm.shape), cfg.grid, direction="x", f0=cfg.carrier_f0
        )


def _square_wave_coefficient(k: int, duty: float) -> complex:
    if k == 0:
        return complex(2.0 * duty - 1.0)
    return (1.0 - np.exp(-2j * np.pi * k * duty)) / (1j * np.pi * k)


@pytest.mark.parametrize("duty", [0.4, 0.55, 0.6])
def test_off_duty_orders_keep_even_integer_harmonics(duty):
    # (m,n)=(2,0) maps to detector order (a,b)=(1,1).
    expected = 0.5 * _square_wave_coefficient(2, duty) * _square_wave_coefficient(0, duty)
    analytic = analytic_orders(max_index=2, duty=duty).with_orders([(1, 1)]).amp[0]
    bitmap = bitmap_orders(harmonic_cell=400, max_index=2, duty=duty).with_orders(
        [(1, 1)]
    ).amp[0]

    assert analytic == pytest.approx(expected, abs=1e-14)
    # Pixel-centre sampling carries an O(1/N) phase/reference error.
    assert bitmap == pytest.approx(expected, rel=0.02, abs=2e-4)
    assert abs(bitmap) > 0.0


def test_half_pixel_array_pupil_uses_one_rigid_shift_everywhere():
    grid = Grid(n=8, extent=1.0)
    cfg = SystemConfig(grid=grid, shear_ratio=0.5 * grid.dx)
    aperture = np.zeros(grid.shape, dtype=bool)
    aperture[:, 2:6] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, pupil=aperture)
        regions = shear_region_masks(grid, cfg.s, aperture=aperture)

    # np.rint(0.5) is one global ties-to-even decision (zero shift), not a
    # separate decision for every j+0.5 pixel coordinate.
    assert np.array_equal(fm.order_support(1, 0), aperture)
    expected = (
        fm.order_support(0, 0)
        & fm.order_support(1, 0)
        & fm.order_support(-1, 0)
    )
    assert np.array_equal(regions["region_x"], expected)


def test_damped_solver_does_not_square_the_jacobian_condition_number():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(200, 2)))
    v = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    J = q @ np.diag([1.0, 1e-8]) @ v.T
    target = v[:, 1]
    f = -J @ target

    step, _, _ = _solve_damped(J, f, 0.0)
    assert np.linalg.cond(J) > 1e7
    assert np.linalg.norm(step - target) < 1e-6


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_phase_shift_rejects_non_finite_frames(bad):
    with pytest.raises(ValueError, match="finite"):
        lsq_phase_shift(np.full((3, 2, 2), bad))


def test_noise_and_order_parameters_reject_non_finite_values():
    frames = np.ones((2, 2))
    with pytest.raises(ValueError, match="poisson_scale"):
        add_noise(frames, poisson_scale=0.0)
    with pytest.raises(ValueError, match="snr_db"):
        add_noise(frames, snr_db=np.nan)
    with pytest.raises(ValueError, match="duty"):
        analytic_orders(duty=np.nan)
