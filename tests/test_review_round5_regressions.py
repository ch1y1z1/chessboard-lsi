"""Regression tests for complex-amplitude and Fourier API invariants."""

from __future__ import annotations

import numpy as np
import pytest

import lsi.pipeline as pipeline_module
from lsi.config import Grid, SystemConfig
from lsi.forward import DEFAULT_ORDERS_5, ForwardModel, ZernikeWavefront
from lsi.ftmode import LobeResult, demodulate_lobe
from lsi.grating import OrderSet, analytic_orders, far_field_amplitude
from lsi.lm import LMConfig, levenberg_marquardt
from lsi.pipeline import (
    demodulate_fourier,
    fourier_to_wavefront,
    phase_shift_to_wavefront,
)


def test_global_order_phase_does_not_change_observables_or_reconstruction():
    config = SystemConfig(grid=Grid(n=64, extent=1.1))
    orders = analytic_orders(max_index=1).with_orders(DEFAULT_ORDERS_5)
    phase = np.exp(0.73j)
    shifted_orders = OrderSet(
        orders.ab,
        orders.amp * phase,
        parity=orders.parity,
    )
    base = ForwardModel(config, orders)
    shifted = ForwardModel(config, shifted_orders)
    wavefront = ZernikeWavefront([0.2, -0.1], [4, 7])

    assert np.allclose(base.intensity(wavefront), shifted.intensity(wavefront))
    for direction in ("x", "y"):
        delta = shifted.demodulation_offset(direction) - base.demodulation_offset(
            direction
        )
        assert np.angle(np.exp(1j * delta)) == pytest.approx(0.0, abs=1e-14)

    base_x = base.phase_shift_frames(wavefront, "x")
    base_y = base.phase_shift_frames(wavefront, "y")
    shifted_x = shifted.phase_shift_frames(wavefront, "x")
    shifted_y = shifted.phase_shift_frames(wavefront, "y")
    assert np.allclose(base_x, shifted_x)
    assert np.allclose(base_y, shifted_y)

    base_fit, _ = phase_shift_to_wavefront(
        base, base_x, base_y, indices=[4, 7]
    )
    shifted_fit, _ = phase_shift_to_wavefront(
        shifted, shifted_x, shifted_y, indices=[4, 7]
    )
    assert shifted_fit.coeffs == pytest.approx(base_fit.coeffs, abs=1e-12)


def test_fourier_raw_offset_is_added_only_after_unwrap():
    config = SystemConfig(period_um=30.0, grid=Grid(n=128, extent=1.1))
    model = ForwardModel(config)
    wavefront = ZernikeWavefront([0.25, 0.5], [4, 7])
    rng = np.random.default_rng(42)
    image = model.ft_mode_frame(wavefront)
    image = image + 1e-10 * rng.standard_normal(image.shape)

    corrected_fit, corrected = fourier_to_wavefront(
        model, image, indices=[4, 7], remove_offset=True
    )
    raw_fit, raw = fourier_to_wavefront(
        model,
        image,
        indices=[4, 7],
        remove_offset=False,
        offset_mode="model",
    )

    assert raw_fit.coeffs == pytest.approx(corrected_fit.coeffs, abs=1e-12)
    for direction in ("x", "y"):
        assert np.array_equal(raw.mask[direction], corrected.mask[direction])
        mask = raw.mask[direction]
        model_offset = model.demodulation_offset(direction)
        assert raw.phase[direction][mask] == pytest.approx(
            corrected.phase[direction][mask] + model_offset,
            abs=1e-12,
        )
        # Both wrapped maps are intentionally model-offset corrected before
        # unwrapping, even when the public result requests the raw constant.
        assert raw.wrapped_phase[direction][mask] == pytest.approx(
            corrected.wrapped_phase[direction][mask],
            abs=1e-12,
        )
        assert raw.lobes[direction].phase_offset == pytest.approx(model_offset)


def test_fourier_threshold_uses_peak_inside_physical_support(monkeypatch):
    model = ForwardModel(SystemConfig(grid=Grid(n=32, extent=1.1)))
    support = (
        model.order_support(0, 0)
        & model.order_support(1, 0)
        & model.order_support(-1, 0)
    )
    amplitude = np.where(support, 1.0, 100.0)

    def fake_lobe(image, grid, *, direction, phase_offset, **kwargs):
        field = amplitude.astype(complex)
        return LobeResult(
            phase=np.zeros(model.shape),
            amplitude=amplitude,
            complex_field=field,
            peak_index=(0, 0),
            peak_freq=(0.0, 0.0),
            direction=direction,
            phase_offset=phase_offset,
        )

    monkeypatch.setattr(pipeline_module, "demodulate_lobe", fake_lobe)
    _, mask, _ = demodulate_fourier(
        model,
        np.zeros(model.shape),
        direction="x",
        threshold_frac=0.5,
        erode_px=0,
        difference_model="two_sided",
    )
    assert np.array_equal(mask, support)


@pytest.mark.parametrize("threshold", [-0.1, 1.0, np.nan, True])
def test_fourier_threshold_is_strictly_validated(threshold):
    model = ForwardModel(SystemConfig(grid=Grid(n=24)))
    with pytest.raises(ValueError, match="threshold_frac"):
        demodulate_fourier(
            model,
            np.zeros(model.shape),
            threshold_frac=threshold,
        )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("sigma_units", {"method": "local", "sigma_units": 0.0}),
        ("sigma_units", {"method": "local", "sigma_units": np.nan}),
        ("window_radius", {"window_radius": -1.0}),
        ("window_radius", {"window_radius": True}),
        ("phase_offset", {"phase_offset": np.inf}),
    ],
)
def test_lobe_filter_parameters_are_validated_early(name, kwargs):
    grid = Grid(n=24)
    with pytest.raises(ValueError, match=name):
        demodulate_lobe(np.zeros(grid.shape), grid, **kwargs)


def test_generic_lm_requires_requested_variable_projection_callback():
    def residual_and_jac(x):
        return np.array([x[0] - 1.0]), np.ones((1, 1))

    with pytest.raises(ValueError, match="scale_background callback"):
        levenberg_marquardt(
            residual_and_jac,
            [0.0],
            LMConfig(fit_scale_background=True),
        )


def test_order_selection_preserves_available_parity_metadata():
    orders = OrderSet(
        [[0, 0], [1, 0]],
        [0.5, 0.2],
        parity=[10, 11],
    )
    assert orders.select([(1, 0)]).parity.tolist() == [11]
    assert orders.with_orders([(1, 0), (0, 0)]).parity.tolist() == [11, 10]
    assert orders.with_orders([(2, 0)]).parity is None


@pytest.mark.parametrize("n", [64.0, True])
def test_far_field_rejects_lossy_sample_counts(n):
    with pytest.raises(ValueError, match="n"):
        far_field_amplitude("chessboard", n=n)
