"""Regression tests for pipeline model contracts and phase gauges."""

from __future__ import annotations

import numpy as np
import pytest

import lsi.pipeline as pipeline_module
from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.pipeline import (
    DEFAULT_INDICES,
    DiffPhase,
    demodulate_phase_shift,
    fourier_to_wavefront,
    phase_shift_to_wavefront,
    reconstruct,
)


def test_high_level_default_includes_z16():
    assert DEFAULT_INDICES == tuple(range(2, 17))

    model = ForwardModel(SystemConfig(grid=Grid(n=64, extent=1.1)))
    wavefront = ZernikeWavefront([0.2], [16])
    frames_x = model.phase_shift_frames(wavefront, "x")
    frames_y = model.phase_shift_frames(wavefront, "y")
    fit, _ = phase_shift_to_wavefront(model, frames_x, frames_y)

    assert fit.indices.tolist() == list(range(2, 17))
    assert fit.as_dict()[16] == pytest.approx(0.2, abs=1e-10)


def _simple_diff(model: str = "one_sided") -> DiffPhase:
    shape = (8, 8)
    zeros = np.zeros(shape)
    mask = np.ones(shape, dtype=bool)
    confidence = np.ones(shape)
    return DiffPhase(
        dW={"x": zeros.copy(), "y": zeros.copy()},
        phase={"x": zeros.copy(), "y": zeros.copy()},
        mask={"x": mask.copy(), "y": mask.copy()},
        difference_model=model,
        confidence={"x": confidence.copy(), "y": confidence.copy()},
    )


def test_reconstruct_uses_diff_model_and_rejects_an_explicit_mismatch():
    model = ForwardModel(SystemConfig(grid=Grid(n=8, extent=1.1)))
    diff = _simple_diff("one_sided")

    fit = reconstruct(model, diff, indices=[2])
    assert fit.coeffs == pytest.approx([0.0])

    with pytest.raises(ValueError, match="conflicts with DiffPhase"):
        reconstruct(model, diff, indices=[2], difference_model="two_sided")

    with pytest.raises(ValueError, match="difference_model"):
        _simple_diff("one-side")


def test_route_specific_confidence_reaches_weighted_reconstruction(monkeypatch):
    model = ForwardModel(SystemConfig(grid=Grid(n=8, extent=1.1)))
    diff = _simple_diff("two_sided")
    diff.confidence["x"][0, 0] = 0.25
    diff.confidence["y"][0, 0] = 0.5
    captured = {}
    sentinel = object()

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        pipeline_module, "fit_differential_zernike", fake_fit
    )
    result = reconstruct(model, diff, indices=[2])

    assert result is sentinel
    assert captured["weights_x"] == pytest.approx(diff.confidence["x"])
    assert captured["weights_y"] == pytest.approx(diff.confidence["y"])


def test_old_weighting_keyword_remains_a_deprecated_alias(monkeypatch):
    model = ForwardModel(SystemConfig(grid=Grid(n=8, extent=1.1)))
    diff = _simple_diff("two_sided")
    captured = {}

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        pipeline_module, "fit_differential_zernike", fake_fit
    )
    with pytest.warns(DeprecationWarning, match="weight_by_confidence"):
        reconstruct(
            model, diff, indices=[2], weight_by_modulation=False
        )

    assert captured["weights_x"] is None
    assert captured["weights_y"] is None


def test_fourier_route_publishes_lobe_amplitude_as_confidence():
    model = ForwardModel(
        SystemConfig(period_um=30.0, grid=Grid(n=96, extent=1.1))
    )
    image = model.ft_mode_frame(ZernikeWavefront([0.3], [7]))
    _, diff = fourier_to_wavefront(model, image, indices=[7])

    assert diff.difference_model == "one_sided"
    for direction in ("x", "y"):
        assert diff.confidence[direction] is diff.amplitude[direction]
        assert np.max(diff.confidence[direction]) > 0.0


def test_reconstruct_rejects_disconnected_phase_gauges():
    model = ForwardModel(SystemConfig(grid=Grid(n=8, extent=1.1)))
    diff = _simple_diff("two_sided")
    disconnected = np.zeros((8, 8), dtype=bool)
    disconnected[1:3, 1:3] = True
    disconnected[5:7, 5:7] = True
    diff.mask = {"x": disconnected, "y": disconnected}

    with pytest.raises(ValueError, match="independent phase gauge"):
        reconstruct(model, diff, indices=[2])


def test_phase_shift_pipeline_rejects_a_segmented_pupil():
    config = SystemConfig(grid=Grid(n=64, extent=1.1))

    def segmented_pupil(x, y):
        left = (x + 0.45) ** 2 + y**2 <= 0.2**2
        right = (x - 0.45) ** 2 + y**2 <= 0.2**2
        return left | right

    model = ForwardModel(config, pupil=segmented_pupil)
    wavefront = ZernikeWavefront([0.1], [4])
    frames_x = model.phase_shift_frames(wavefront, "x")
    frames_y = model.phase_shift_frames(wavefront, "y")

    with pytest.raises(ValueError, match="independent phase gauge"):
        demodulate_phase_shift(model, frames_x, frames_y)
