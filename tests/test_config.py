"""Configuration value-object contracts."""

from dataclasses import FrozenInstanceError

import pytest

from lsi.config import Grid, PRESET_PHASE_SHIFT, SystemConfig


@pytest.mark.parametrize("name,value", [
    ("grid", Grid(n=32)),
    ("period_um", 20.0),
    ("shear_ratio", 0.1),
])
def test_system_config_is_immutable(name, value):
    config = SystemConfig()

    with pytest.raises(FrozenInstanceError):
        setattr(config, name, value)


def test_exported_preset_cannot_be_mutated():
    with pytest.raises(FrozenInstanceError):
        PRESET_PHASE_SHIFT.phase_steps = 4


def test_carrier_frequency_paper_is_four_times_the_default():
    """Dissertation eq. (2-48): f0 = 2 m / s; the project default is f0/4."""
    import numpy as np

    for cfg in (SystemConfig(), SystemConfig(period_um=30.0)):
        assert cfg.carrier_frequency_paper == pytest.approx(
            2.0 * cfg.talbot_number / cfg.s, rel=1e-15
        )
        assert cfg.carrier_frequency_paper == pytest.approx(
            4.0 * cfg.carrier_f0, rel=1e-15
        )
        assert np.isfinite(cfg.carrier_frequency_paper)
