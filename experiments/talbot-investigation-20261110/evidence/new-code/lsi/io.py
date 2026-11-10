"""Small, explicit result files for reproducible experiments."""
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import importlib.metadata
import csv
import json
from pathlib import Path
import platform
import resource
import subprocess

import numpy as np


def serializable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=serializable, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_arrays(path, **arrays):
    """Publish a complete archive only after compression has finished."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_rows(directory, rows):
    write_json(directory / 'results.json', rows)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / 'results.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, tuple, dict)) else v
                          for k, v in row.items()} for row in rows)
    temporary.replace(directory / 'results.csv')


def peak_memory_mib():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return size / (1024**2 if platform.system() == 'Darwin' else 1024)


def provenance():
    root = Path(__file__).parent
    sources = [*root.glob('*.py'), *root.parent.glob('tools/*.py')]
    for name in ('pyproject.toml', 'uv.lock'):
        sources.append(root.parent / name)
    return {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(),
        'dependencies': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'scipy', 'scikit-image', 'scikit-learn')},
        'units': 'object-pupil waves',
        'metric': 'fixed annular pupil; primary RMSE removes piston, both tilts and defocus',
        'noise': 'additive Gaussian, sigma=10**(-snr_db/20) after clean peak normalization',
        'source_sha256': {str(p.relative_to(root.parent)): sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        'tilt_model': 'reference phase ramp, not rotated-plane propagation',
    }
