from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.signal import czt
from scipy.integrate import trapezoid
from lsi.annular import annular_zernike, configured_coeffs, mode_labels
from lsi.config import DiffractionConfig
from lsi.propagation import CenteredCZT, DiffractionModel
from lsi.direct import fit_direct
from lsi.fourier import FourierExtractor, FourierSettings, FourierCalibration, calibration_images
from lsi.metrics import wavefront_metrics


def test_reference_mode_order_and_normalization():
    labels = mode_labels()
    assert [labels[j] for j in (7, 25, 11, 15)] == [(3, 1), (5, -5), (5, -1), (3, 3)]
    r = np.linspace(.3845, 1, 500)[:, None]
    theta = np.linspace(0, 2*np.pi, 800, endpoint=False)[None, :]
    rho, theta = np.broadcast_arrays(r, theta)
    for n, l in ((0, 0), (3, 1), (5, -5), (5, -1)):
        z = annular_zernike(n, l, .3845, rho, theta)
        integrand = np.mean(z*z, axis=1) * r[:, 0]
        integral = trapezoid(integrand, x=r[:, 0]) * 2 / (1-.3845**2)
        assert integral == pytest.approx(1, abs=2e-4)


def test_cached_czt_matches_reference_centering_for_batch():
    rng = np.random.default_rng(8)
    field = rng.normal(size=(3, 14, 14)) + 1j*rng.normal(size=(3, 14, 14))
    dx, f0, df, m = .013, -.9, .21, 18
    actual = CenteredCZT(14, m, dx, f0, df)(field)
    expected = field
    for axis in (-1, -2):
        expected = czt(expected, m=m, w=np.exp(-2j*np.pi*df*dx), a=np.exp(2j*np.pi*f0*dx), axis=axis)
        shape = [1]*3; shape[axis] = m
        expected *= (np.exp(2j*np.pi*(f0+np.arange(m)*df)*(14/2)*dx)*dx).reshape(shape)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_exact_separable_operator_with_tilts_shifts_and_sampled_jacobian():
    cfg = DiffractionConfig.profile('test').with_angle('grating', .37, 'diagonal').with_angle('camera', -.2, 'y')
    cfg = replace(cfg, shift_x_grating=2*cfg.dx_grating, shift_y_grating=-cfg.dx_grating,
                  shift_z_grating=1e-6)
    fast = DiffractionModel(cfg)
    oracle = DiffractionModel(cfg, backend='czt')
    c = np.array([.04,-.025,.03,.035])
    image, jac = oracle.evaluate(c, jacobian=True)
    actual, dactual = fast.evaluate(c, jacobian=True)
    assert np.linalg.norm(actual-image)/np.linalg.norm(image) < 1e-10
    assert np.linalg.norm(dactual-jac)/np.linalg.norm(jac) < 1e-10
    rows = np.random.default_rng(2).choice(cfg.n*cfg.n, 500, replace=False)
    sampled, dsampled = fast.evaluate(c, jacobian=True, rows=rows)
    np.testing.assert_allclose(sampled, image.ravel()[rows], rtol=1e-9, atol=1e-24)
    np.testing.assert_allclose(dsampled, jac.reshape(4,-1)[:,rows], rtol=1e-8, atol=1e-23)


def test_full_diffraction_jacobian():
    model = DiffractionModel(DiffractionConfig.profile('test'))
    c = np.array([.04, -.025, .03, .035])
    image, jac = model.evaluate(c, jacobian=True)
    np.testing.assert_allclose(image, model.evaluate(c), rtol=1e-12)
    for j in range(4):
        d = np.zeros(4); d[j] = 1e-6
        fd = (model.evaluate(c+d) - model.evaluate(c-d))/(2e-6)
        assert np.linalg.norm(fd-jac[j])/np.linalg.norm(fd) < 1e-7


def test_direct_recovers_independent_coefficients_and_exposure():
    model = DiffractionModel(DiffractionConfig.profile('test'))
    truth = np.array([.04, -.025, .03, .035])
    raw = model.evaluate(truth)
    measured = 2.7 * raw / raw.max() + .03
    result = fit_direct(model, measured, starts=2, samples=1500, max_nfev=35)
    np.testing.assert_allclose(result['coeffs'], truth, atol=1e-7)
    assert result['trials'][result['best_trial']]['full_intensity_rms'] < 1e-8
    assert len(result['trials']) == 2


def test_source_tilt_is_a_null_control_for_reference_point_sources():
    c = DiffractionConfig.profile('test')
    a = DiffractionModel(c).evaluate(np.zeros(4))
    b = DiffractionModel(c.with_angle('object', .5)).evaluate(np.zeros(4))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-25)
    camera = DiffractionModel(c.with_angle('camera', .5)).evaluate(np.zeros(4))
    assert np.linalg.norm(a-camera)/np.linalg.norm(a) > .01


def test_config_roundtrip_and_quick_has_all_64_sources(tmp_path):
    cfg = DiffractionConfig.profile('quick')
    p = tmp_path/'config.json'; p.write_text(json.dumps(cfg.to_dict()))
    assert DiffractionConfig.from_json(p) == cfg
    assert len(cfg.sources) == 64
    assert cfg.n*cfg.dx_cam == pytest.approx(.012)
    assert cfg.n*cfg.up_sample_grating*cfg.dx_grating == pytest.approx(40.2e-6)
    with pytest.raises(ValueError):
        replace(cfg, tilt_axis_x=0, tilt_axis_y=0)
    with pytest.raises(ValueError, match='absent from fit indices'):
        DiffractionModel(replace(cfg, c_Zernike=(.25,.15,.05,.2,.1)))


def test_fourier_independent_calibration_has_physical_units():
    model = DiffractionModel(DiffractionConfig.profile('test'))
    s = FourierSettings(half_width=8, carrier_x=29, carrier_y=29, phase_sigma=1, regression='lstsq')
    extractor = FourierExtractor(model.config, s)
    # This is a local linear response calibration. Test an unseen mixture
    # inside its small-signal regime, rather than assuming global linearity.
    amplitude = 1e-4
    images = calibration_images(model, amplitude)
    cal = FourierCalibration.fit(extractor, images, model.indices, amplitude)
    truth = np.array([.3, -.2, .4, .2]) * amplitude
    fitted = cal.reconstruct(extractor, model.evaluate(truth))['coeffs_waves']
    assert np.linalg.norm(fitted-truth) < .03 * np.linalg.norm(truth)
    assert np.linalg.matrix_rank(cal.response) == 4
    with pytest.raises(ValueError):
        FourierCalibration.fit(extractor, [images[0]]*9, model.indices, .005)


def test_metrics_do_not_allow_whole_wavefront_sign_flip():
    model = DiffractionModel(DiffractionConfig.profile('test'))
    truth = np.array([.04, -.025, .03, .035])
    assert wavefront_metrics(model, truth, truth)['rmse_waves'] == 0
    assert wavefront_metrics(model, -truth, truth)['rmse_waves'] > .05


def test_golden_reference_fields():
    fixture = np.load(Path(__file__).parent/'data/reference_fields.npz')
    cfg = DiffractionConfig(**json.loads(str(fixture['config'])))
    model = DiffractionModel(cfg)
    src = cfg.sources[0]
    np.testing.assert_allclose(model.source_field(src), fixture['source'], atol=1e-13)
    c = configured_coeffs(cfg)[list(model.indices)]
    H = model.pupil*np.exp(2j*np.pi*model.wavefront(c))
    objective = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fixture['source']))) * H)))
    np.testing.assert_allclose(objective, fixture['objective'], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(model.propagate_image(objective), fixture['camera'], rtol=1e-8, atol=1e-16)
    np.testing.assert_allclose(model.evaluate(c), fixture['intensity'], rtol=1e-8, atol=1e-25)
