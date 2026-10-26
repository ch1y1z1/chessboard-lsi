from pathlib import Path
import json

import numpy as np
import pytest
from scipy.integrate import trapezoid

from lsi.basis import annular_zernike, mode_labels
from lsi.config import Scene, SystemConfig
from lsi.direct import fit_direct
from lsi.evaluate import wavefront_metrics
from lsi.forward import DiffractionModel
from lsi.fourier import FourierCalibration, FourierExtractor, FourierSettings, calibration_images


def test_frozen_pr16_clean_and_noisy_results(reference):
    system, _ = reference
    frozen = json.loads((Path(__file__).parent / 'data/refactor_baseline.json').read_text())
    model = DiffractionModel(system)
    truth = np.asarray(frozen['truth'])
    clean = model.evaluate(truth); clean /= clean.max()
    settings = FourierSettings(**frozen['settings'])
    cal = FourierCalibration.fit(FourierExtractor(system, settings),
                                calibration_images(model, frozen['calibration_amplitude']),
                                model.indices, frozen['calibration_amplitude'])
    for seed in (None, 42, 43, 44):
        image = clean if seed is None else clean + np.random.default_rng(seed).normal(0, .01, clean.shape)
        fourier = cal.reconstruct(FourierExtractor(system, settings), image)['coeffs_waves']
        direct = fit_direct(model, image, **frozen['direct'])['coeffs']
        for method, coefficients in [('fourier_calibrated', fourier), ('direct', direct)]:
            expected = next(r for r in frozen['rows'] if r['seed'] == seed and r['method'] == method)
            np.testing.assert_allclose(coefficients, expected['coeffs_waves'], rtol=1e-9, atol=1e-11)
            assert wavefront_metrics(model, coefficients, truth)['rmse_waves'] == pytest.approx(
                expected['rmse_waves'], rel=1e-9, abs=1e-11)


def test_physical_units_gain_background_and_mode_convention():
    assert [mode_labels()[j] for j in (7, 25, 11, 15)] == [(3, 1), (5, -5), (5, -1), (3, 3)]
    rho, theta = np.broadcast_arrays(np.linspace(.3845, 1, 500)[:, None],
                                     np.linspace(0, 2 * np.pi, 800, endpoint=False)[None, :])
    z = annular_zernike(3, 1, .3845, rho, theta)
    rms2 = trapezoid(np.mean(z*z, axis=1) * rho[:, 0], x=rho[:, 0]) * 2 / (1 - .3845**2)
    assert rms2 == pytest.approx(1, abs=2e-4)
    model = DiffractionModel(SystemConfig.profile('test'))
    truth = np.array([.04, -.025, .03, .035])
    raw = model.evaluate(truth)
    measured = 2.7 * raw / raw.max() + .03
    fit = fit_direct(model, measured, starts=2, screen_initials=3, samples=1500, max_nfev=35)
    np.testing.assert_allclose(fit['coeffs'], truth, atol=1e-7)
    assert fit['trials'][fit['best_trial']]['full_intensity_rms'] < 1e-8
    assert wavefront_metrics(model, -truth, truth)['rmse_waves'] > .05
    with pytest.raises(ValueError, match='absent'):
        Scene(indices=(7, 2), coefficients=(.1, .2)).coefficients_for((7, 25, 11, 15))
    assert not hasattr(model.config, 'coefficients') and not hasattr(model.config, 'c_Zernike')


def test_independent_fourier_calibration():
    model = DiffractionModel(SystemConfig.profile('test'))
    settings = FourierSettings(half_width=8, carrier_x=29, carrier_y=29, phase_sigma=1, regression='lstsq')
    cal = FourierCalibration.fit(FourierExtractor(model.config, settings),
                                calibration_images(model, 1e-4), model.indices, 1e-4)
    truth = np.array([.3, -.2, .4, .2]) * 1e-4
    fitted = cal.reconstruct(FourierExtractor(model.config, settings), model.evaluate(truth))['coeffs_waves']
    assert np.linalg.norm(fitted - truth) < .03 * np.linalg.norm(truth)
