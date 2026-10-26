import json

import numpy as np
import pytest

from lsi.cli import main
from lsi.config import SystemConfig
from lsi.execution import exclusive_study


def run_small(tmp_path, command, sweep):
    config = tmp_path / f'{command}.json'
    config.write_text(json.dumps({'profile': 'test', 'scene': {'rms': .04},
                                 'fit': {'starts': 1, 'samples': 1500, 'max_nfev': 25},
                                 'sweep': sweep}))
    output = tmp_path / command
    main([command, '--config', str(config), '--output', str(output)])
    return output, json.loads((output / 'results.json').read_text())


def test_compare_writes_reproducible_data(tmp_path):
    output, rows = run_small(tmp_path, 'compare', {})
    assert {r['method'] for r in rows} == {'direct', 'fourier_calibrated'}
    assert next(r for r in rows if r['method'] == 'direct')['rmse_waves'] < 1e-6
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['source_count'] == 4
    assert 'coefficients' not in manifest['system'] and manifest['scene']['coefficients']
    assert manifest['source_sha256'] and (output / 'source_snapshot/lsi/cli.py').exists()
    assert (output / 'results.csv').exists()
    with np.load(output / 'baseline_observation.npz') as observation:
        assert observation['observed'].shape == (80, 80)
    with pytest.raises(FileExistsError):
        main(['compare', '--config', str(tmp_path / 'compare.json'), '--output', str(output)])


def test_region_and_angle_protocols(tmp_path):
    _, regions = run_small(tmp_path, 'regions', {
        'widths': [8], 'offsets': [], 'shapes': [], 'aspects': [], 'erosions': [], 'thresholds': []})
    assert len(regions) == 3 and all(r['status'] == 'ok' for r in regions)
    assert regions[-1]['calibration_policy'] == 'frozen_nominal'
    _, angles = run_small(tmp_path, 'angles', {
        'components': ['camera'], 'axes': ['x'], 'angles': [.25],
        'center_modes': ['fixed', 'auto'], 'geometry': ['known', 'ignored']})
    assert len(angles) == 9 and len({(r['case'], r['method']) for r in angles}) == 9
    for row in angles:
        if row['method'] == 'fourier_calibrated':
            assert row['settings']['center_mode'] == row['calibration_settings']['center_mode']
    known_direct = next(r for r in angles if r['method'] == 'direct' and r['geometry'] == 'known')
    assert known_direct['rmse_waves'] < 1e-6


def test_no_implicit_large_or_parallel_run(tmp_path):
    with pytest.raises(SystemExit):
        main(['compare', '--profile', 'reference', '--output', str(tmp_path / 'large')])
    assert not (tmp_path / 'large').exists()
    with exclusive_study(tmp_path / 'lock'):
        with pytest.raises(RuntimeError, match='sequentially'):
            with exclusive_study(tmp_path / 'lock'):
                pass
    assert len(SystemConfig.profile('quick').sources) == 64
