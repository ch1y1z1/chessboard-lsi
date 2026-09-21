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
