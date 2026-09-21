"""Regression tests for the paper-consistency and public-API review."""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import (
    DEFAULT_ORDERS_5,
    DEFAULT_ORDERS_9,
    ForwardModel,
    ZernikeWavefront,
)
from lsi.grating import OrderSet
from lsi.lm import fit_wavefront_from_frames
from lsi.phaseshift import extract_psd_phase, zero_order_center_radius
from lsi.pipeline import demodulate_fourier
from lsi.reconstruct import fit_differential_zernike


def test_nine_beam_paper_set_uses_only_the_33_third_order_family():
    assert DEFAULT_ORDERS_9 == DEFAULT_ORDERS_5 + (
        (3, 0),
        (-3, 0),
        (0, 3),
        (0, -3),
    )


def test_modulation_detection_controls_extract_psd_region():
    grid = Grid(n=96, extent=1.1)
    config = SystemConfig(grid=grid)
    x, y = grid.coords()
    modulation = np.exp(-((x - 0.18) ** 2 + (y + 0.11) ** 2) / 0.18)
    deltas = 2.0 * np.pi * np.arange(8) / 8
    frames = np.array([1.0 + modulation * np.cos(delta) for delta in deltas])

    broad_circle = zero_order_center_radius(modulation, grid, threshold_frac=0.2)
    narrow_circle = zero_order_center_radius(modulation, grid, threshold_frac=0.8)
    assert broad_circle.cx == pytest.approx(0.18, abs=grid.dx)
    assert broad_circle.cy == pytest.approx(-0.11, abs=grid.dx)
    assert broad_circle.radius > narrow_circle.radius

    _, broad = extract_psd_phase(
        frames, grid, config, threshold_frac=0.2, direction="x"
    )
    _, narrow = extract_psd_phase(
        frames, grid, config, threshold_frac=0.8, direction="x"
    )
    assert broad.sum() > narrow.sum()
    assert not np.array_equal(broad, narrow)


@pytest.mark.parametrize("n_steps", [0, False, 2, 8.0])
def test_phase_shift_generators_apply_strict_step_validation(n_steps):
    model = ForwardModel(SystemConfig(grid=Grid(n=24)))
    wavefront = ZernikeWavefront([0.0], [2])
    with pytest.raises(ValueError, match="n_steps"):
        model.phase_shift_frames(wavefront, n_steps=n_steps)
    with pytest.raises(ValueError, match="n_steps"):
        model.phase_shift_delta_table(n_steps=n_steps)


def test_intensity_lm_rejects_unobservable_piston():
    model = ForwardModel(SystemConfig(grid=Grid(n=24)))
    proto = ZernikeWavefront([0.0, 0.0], [1, 4])
    with pytest.raises(ValueError, match="piston Z1 is unobservable"):
        fit_wavefront_from_frames(model, proto, [])


@pytest.mark.parametrize("indices", [[7.9], [True], [4, 4]])
def test_wavefront_rejects_lossy_or_duplicate_indices(indices):
    with pytest.raises(ValueError):
        ZernikeWavefront(np.zeros(len(indices)), indices)


@pytest.mark.parametrize(
    ("ab", "amp"),
    [
        ([(0, 0), (1, 0)], [1.0]),
        ([(0, 0), (0, 0)], [1.0, 1.0]),
        ([(0.25, 0)], [1.0]),
        ([(0, 0)], [np.nan]),
    ],
)
def test_order_set_rejects_inconsistent_or_lossy_inputs(ab, amp):
    with pytest.raises(ValueError):
        OrderSet(ab, amp)


def test_differential_fit_rejects_broadcastable_shape_mismatch():
    grid = Grid(n=16)
    x, y = grid.coords()
    with pytest.raises(ValueError, match="dW_x must have shape"):
        fit_differential_zernike(
            np.zeros((16, 1)),
            None,
            None,
            None,
            0.1,
            x,
            y,
            indices=[4],
        )


@pytest.mark.parametrize("erode_px", [1.9, -1, True])
def test_fourier_demodulation_rejects_non_integer_erosion(erode_px):
    model = ForwardModel(SystemConfig(grid=Grid(n=24)))
    with pytest.raises(ValueError, match="erode_px"):
        demodulate_fourier(
            model,
            np.zeros(model.shape),
            erode_px=erode_px,
        )
