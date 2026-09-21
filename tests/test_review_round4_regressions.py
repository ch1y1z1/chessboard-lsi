"""Regression tests for the numerical/API review after commit 01c6ddb."""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig, preset_fourier
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.lm import LMConfig
from lsi.metrics import coefficient_comparison
from lsi.pipeline import demodulate_phase_shift, fourier_to_wavefront
from lsi.reconstruct import fit_differential_zernike
from lsi.unwrap import unwrap_masked_poisson, unwrap_seed_growth
from lsi.zernike import as_wavefront


def test_fourier_pipeline_unwraps_large_differential_phase():
    model = ForwardModel(preset_fourier())
    image = model.ft_mode_frame(ZernikeWavefront([3.0], [7]))

    fit, diff = fourier_to_wavefront(model, image, indices=[7])
    wrapped_fit, _ = fourier_to_wavefront(
        model, image, indices=[7], unwrap="none"
    )

    assert fit.as_dict()[7] == pytest.approx(3.0, abs=0.03)
    assert wrapped_fit.as_dict()[7] < 2.0
    unwrapped_extrema = []
    for direction in ("x", "y"):
        mask = diff.mask[direction]
        assert np.max(np.abs(diff.wrapped_phase[direction][mask])) <= np.pi
        unwrapped_extrema.append(np.max(np.abs(diff.phase[direction][mask])))
        assert diff.lobes[direction].unwrapped_phase is diff.phase[direction]
    assert max(unwrapped_extrema) > np.pi


@pytest.mark.parametrize("unwrap", [unwrap_seed_growth, unwrap_masked_poisson])
def test_unwrap_rejects_nonfinite_phase_inside_mask(unwrap):
    phase = np.zeros((3, 3))
    phase[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite inside mask"):
        unwrap(phase, np.ones_like(phase, dtype=bool))


def test_jacobi_poisson_keeps_pixels_outside_mask_nan():
    phase = np.zeros((7, 7))
    mask = np.zeros_like(phase, dtype=bool)
    mask[2:5, 2:5] = True
    out = unwrap_masked_poisson(phase, mask, method="jacobi")
    assert np.isfinite(out[mask]).all()
    assert np.isnan(out[~mask]).all()


def test_rank_deficiency_is_reported_in_fit_result():
    x, y = Grid(n=16).coords()
    data = np.zeros_like(x)
    mask = np.ones_like(x, dtype=bool)
    with pytest.warns(UserWarning, match="rank deficient"):
        fit = fit_differential_zernike(
            data, None, mask, None, 0.1, x, y, indices=[3]
        )
    assert fit.rank == 0
    assert fit.rank < len(fit.coeffs)
    assert fit.singular_values.shape == (1,)


def test_differential_fit_rejects_all_zero_effective_weights():
    x, y = Grid(n=16).coords()
    data = np.zeros_like(x)
    mask = np.ones_like(x, dtype=bool)
    with pytest.raises(ValueError, match="all effective reconstruction weights"):
        fit_differential_zernike(
            data,
            None,
            mask,
            None,
            0.1,
            x,
            y,
            indices=[2],
            weights_x=np.zeros_like(x),
        )


def test_phase_shift_pipeline_uses_strict_step_and_frame_validation():
    model = ForwardModel(SystemConfig(grid=Grid(n=24)))
    wavefront = ZernikeWavefront([0.1], [4])
    frames_x = model.phase_shift_frames(wavefront, "x")
    frames_y = model.phase_shift_frames(wavefront, "y")

    with pytest.raises(ValueError, match="n_steps"):
        demodulate_phase_shift(model, frames_x, frames_y, n_steps=8.9)
    with pytest.raises(ValueError, match="frames_x must have shape"):
        demodulate_phase_shift(model, frames_x[:, :, :-1], frames_y)


def test_wavefront_helpers_reject_mismatched_coefficient_lengths():
    x, y = Grid(n=8).coords()
    with pytest.raises(ValueError, match="same length"):
        as_wavefront([0.1, 0.2], x, y, indices=[4])
    with pytest.raises(ValueError, match="same length"):
        coefficient_comparison([4], [0.1, 0.2], [0.1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": -1},
        {"max_iter": 8.0},
        {"lambda0": -1.0},
        {"lambda_min": 10.0, "lambda_max": 1.0},
        {"lambda0": 2.0, "lambda_max": 1.0},
        {"nu0": 1.0},
        {"ftol": -1.0},
        {"verbose": 1},
        {"fit_scale_background": 1},
    ],
)
def test_lm_config_rejects_invalid_solver_parameters(kwargs):
    with pytest.raises(ValueError):
        LMConfig(**kwargs)
