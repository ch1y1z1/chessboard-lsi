"""Differential-Zernike reconstruction result semantics."""

import numpy as np
import pytest

from lsi.config import Grid
from lsi.reconstruct import fit_differential_zernike
from lsi.zernike import differential_zernike_matrix


def test_fit_reports_physical_and_weighted_residuals_separately():
    grid = Grid(n=16, extent=1.1)
    x, y = grid.coords()
    mask = grid.pupil()
    shear = 0.08
    basis = differential_zernike_matrix(
        [4], x[mask], y[mask], shear, "x", "two_sided"
    )[:, 0]
    perturbation = np.linspace(-0.01, 0.02, mask.sum())
    d_w = np.full(grid.shape, np.nan)
    d_w[mask] = 0.2 * basis + perturbation
    weights = np.ones(grid.shape)
    weights[mask] = np.linspace(0.25, 4.0, mask.sum())

    fit = fit_differential_zernike(
        d_w,
        None,
        mask,
        None,
        shear,
        x,
        y,
        indices=[4],
        weights_x=weights,
    )

    expected_weighted = fit.residual * np.sqrt(weights[mask])
    assert fit.weighted_residual == pytest.approx(expected_weighted)
    assert fit.rms_residual == pytest.approx(
        np.sqrt(np.mean(fit.residual**2))
    )
    assert fit.max_abs_residual == pytest.approx(
        np.max(np.abs(fit.residual))
    )
    assert fit.weighted_rms_residual == pytest.approx(
        np.sqrt(np.mean(expected_weighted**2))
    )
    assert fit.weighted_max_abs_residual == pytest.approx(
        np.max(np.abs(expected_weighted))
    )
    assert not np.allclose(fit.residual, fit.weighted_residual)
