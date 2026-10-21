"""Execution-policy checks do not allocate diffraction grids."""
import json
import os
import signal

import pytest

from lsi.config import DiffractionConfig
from lsi.execution import exclusive_study
from lsi.study import Study, main


@pytest.mark.parametrize('arguments', [[], ['--profile', 'reference'],
                                      ['--profile', 'retrieval'], ['--profile', 'quick']])
def test_large_or_implicit_runs_are_rejected_before_allocating(monkeypatch, tmp_path, arguments):
    def unexpected_model(*args, **kwargs):
        pytest.fail('a blocked command must not construct a model')
    monkeypatch.setattr(Study, 'make_model', unexpected_model)
    with pytest.raises(SystemExit) as error:
        main(['compare', *arguments, '--output', str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path/'manifest.json').exists()


def test_custom_config_cannot_bypass_grid_limit(tmp_path):
    config = tmp_path/'large.json'
    config.write_text(json.dumps(DiffractionConfig.profile('reference').to_dict()))
    with pytest.raises(SystemExit) as error:
        main(['compare', '--profile', 'test', '--config', str(config)])
    assert error.value.code == 2


def test_studies_cannot_overlap_and_lock_is_released(tmp_path):
    lock = tmp_path/'study.lock'
    with exclusive_study(lock):
        with pytest.raises(RuntimeError, match='sequentially'):
            with exclusive_study(lock):
                pytest.fail('overlapping run acquired the lock')
    with exclusive_study(lock):
        pass


def test_existing_results_are_not_overwritten(monkeypatch, tmp_path):
    saved = '{"status": "interrupted", "important": "partial results"}\n'
    (tmp_path/'manifest.json').write_text(saved)
    def unexpected_model(*args, **kwargs):
        pytest.fail('an existing run must not be overwritten')
    monkeypatch.setattr(Study, 'make_model', unexpected_model)
    with pytest.raises(SystemExit, match='choose a new --output'):
        main(['compare', '--profile', 'test', '--output', str(tmp_path)])
    assert (tmp_path/'manifest.json').read_text() == saved


def test_sigterm_during_initialization_is_recorded_without_model(monkeypatch, tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail('SIGTERM did not interrupt initialization')
    monkeypatch.setattr(Study, 'make_model', terminate)
    with pytest.raises(SystemExit, match='SIGTERM'):
        main(['compare', '--profile', 'test', '--output', str(tmp_path)])
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['status'] == 'interrupted'
    assert manifest['row_count'] == 0
    assert signal.getsignal(signal.SIGTERM) == previous
