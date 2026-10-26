from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

from lsi.config import SystemConfig


@pytest.fixture
def reference():
    """Independent, unmodified BUV fields; adapt only its historical config keys."""
    with np.load(Path(__file__).parent / 'data/reference_fields.npz') as saved:
        arrays = dict(saved)
    original = json.loads(str(arrays['config']))
    names = {f.name for f in fields(SystemConfig)}
    return SystemConfig(**{k: v for k, v in original.items() if k in names}), arrays
