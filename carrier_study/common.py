"""实验脚本共用：噪声档、失配因素表、并行与落盘工具。"""

from __future__ import annotations

import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .generator import Noise, Scene

#: E1/E3 噪声档：高斯 5 档（峰值 SNR）+ Poisson 3 档（峰值光电子数，读出 5 e-）
NOISES = (
    Noise("gauss", 20.0), Noise("gauss", 30.0), Noise("gauss", 40.0),
    Noise("gauss", 50.0), Noise("gauss", 60.0),
    Noise("poisson", 1e3), Noise("poisson", 1e4), Noise("poisson", 1e5),
)
NOISE_BY_LABEL = {nz.label: nz for nz in NOISES}

#: E4 单因素失配扫描（计划 6.2 节；在 M1 冻结）
MISMATCH_FACTORS: dict[str, tuple[str, tuple]] = {
    "pixel": ("supersample", (4,)),
    "amp_err": ("amp_err", (0.02, 0.05, 0.1, 0.2)),
    "phase_err": ("phase_err", (0.02, 0.05, 0.1, 0.2)),
    "duty": ("duty", (0.49, 0.48, 0.45)),
    "higher_orders": ("higher_orders", (0.03, 0.1, 0.3, 1.0)),
    "shear_err": ("shear_err", (0.002, 0.005, 0.01, 0.02)),
    "carrier_err": ("carrier_err", (0.002, 0.005, 0.01, 0.02)),
    "illum": ("illum", (0.05, 0.1, 0.2)),
    "tail_frac": ("tail_frac", (0.05, 0.1, 0.2)),
}


def mismatch_scenes(seed: int = 0) -> list[tuple[str, float, Scene]]:
    out = []
    for name, (field, levels) in MISMATCH_FACTORS.items():
        for lv in levels:
            out.append((name, float(lv), Scene(**{field: lv}, seed=seed)))
    return out


def workers() -> int:
    return int(os.environ.get("CARRIER_WORKERS", "6"))


def pmap(fn: Callable, items: Iterable, n: int | None = None, chunksize: int = 1) -> list:
    items = list(items)
    n = workers() if n is None else n
    if n <= 1:
        return [fn(it) for it in items]
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps([float(x) for x in np.ravel(v)])
    return v


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
