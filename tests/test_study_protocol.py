import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from lsi.config import DiffractionConfig
from lsi.propagation import DiffractionModel
from lsi.fourier import FourierExtractor, FourierSettings
from lsi.direct import fit_direct
from lsi.study import main


def test_inverse_model_does_not_read_truth_coefficients():
    cfg = DiffractionConfig.profile('test')
    other = replace(cfg, c_Zernike=(3., -2., 4., 1., 0.))
    c = np.array([.04, -.025, .03, .035])
    original = DiffractionModel(cfg)
    inverse = DiffractionModel(other)
    np.testing.assert_allclose(inverse.evaluate(c), original.evaluate(c), atol=0, rtol=1e-13)
    obs = original.evaluate(c)
    obs /= obs.max()
    fitted = fit_direct(inverse, obs, starts=2, screen_initials=3, samples=1500, max_nfev=30)
    np.testing.assert_allclose(fitted['coeffs'], c, atol=1e-7)
    assert len(fitted['screened_initials']) == 3
    assert fitted['n_forward_fit'] > fitted['n_jacobian_fit']


def test_fourier_sideband_exact_reference_window():
    path = Path(__file__).parent/'data/reference_fields.npz'
    fixture = np.load(path)
    cfg = DiffractionConfig(**json.loads(str(fixture['config'])))
    settings = FourierSettings(half_width=8, carrier_x=29, carrier_y=29, phase_sigma=1)
    extraction = FourierExtractor(cfg, settings).extract(fixture['intensity'], details=True)
    expected = np.load(Path(__file__).parent/'data/reference_sidebands.npz')
    # Compare phasors to avoid the equivalent +pi/-pi branch endpoint.
    np.testing.assert_allclose(np.exp(1j*extraction['wrapped']), np.exp(1j*expected['wrapped']), atol=1e-10)


def test_comparison_cli_writes_full_protocol(tmp_path):
    main(['compare', '--profile', 'test', '--starts', '1', '--truth-rms', '.04',
          '--samples', '1500', '--max-nfev', '25', '--output', str(tmp_path)])
    rows = json.loads((tmp_path/'results.json').read_text())
    assert {r['method'] for r in rows} == {'direct','fourier_calibrated','fourier_original'}
    raw = next(r for r in rows if r['method'] == 'fourier_original')
    assert 'rmse_waves' not in raw
    direct = next(r for r in rows if r['method'] == 'direct')
    assert direct['rmse_waves'] < 1e-6
    validation = json.loads((tmp_path/'baseline_calibration.json').read_text())
    assert 'held_out_mixed_validation' in validation
    assert (tmp_path/'comparison.svg').exists()
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['source_count'] == 4 and manifest['source_sha256']
