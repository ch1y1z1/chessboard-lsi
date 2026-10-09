"""阶段 0.3：串行单线程标定原语成本，换算为前向当量（F = 1）。

运行：  uv run python -m carrier_study.calibrate
输出：  results/carrier/m0/cost_units.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import config
from .nominal import NominalModel
from .truths import sample_q

OUT = Path("results/carrier/m0")


def _best(fn, repeat: int = 7, number: int = 5) -> float:
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t) / number)
    return float(np.median(times))


def calibrate(p: int, n: int = config.N_GRID, period: float = config.PERIOD_UM) -> dict:
    idx = config.indices_for(p)
    model = NominalModel(config.system(period, n), idx)
    q = sample_q("pow1", 0.3, idx, np.random.default_rng(0))
    I, st = model.forward(q)
    u = np.random.default_rng(1).normal(size=model.M)
    J = model.jacobian(st)
    r = u / np.sqrt(model.M)
    t = {
        "F": _best(lambda: model.forward(q)),
        "G": _best(lambda: model.jtv(st, u)),
        "J": _best(lambda: model.jacobian(st)),
        "NE": _best(lambda: cho_solve(cho_factor(J.T @ J), J.T @ r)),
        "LS": _best(lambda: np.linalg.lstsq(J, r, rcond=None)),
        "QR": _best(lambda: np.linalg.qr(J, mode="r")),
        "SVD": _best(lambda: np.linalg.svd(J, full_matrices=False)),
    }
    units = {k: v / t["F"] for k, v in t.items()}
    return {"p": p, "n": n, "period_um": period, "M": int(model.M), "seconds": t, "units": units}


def load_units(p: int, period: float = config.PERIOD_UM) -> dict:
    data = json.loads((OUT / "cost_units.json").read_text())
    for row in data["models"]:
        if row["p"] == p and row["period_um"] == period:
            return row["units"]
    raise KeyError((p, period))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [calibrate(p, period=per) for per in config.PERIODS_SHEAR_SCAN for p in (12, 35)]
    for row in rows:
        u = row["units"]
        print(f"p={row['p']:2d} period={row['period_um']:g}  M={row['M']}  "
              f"F={row['seconds']['F']*1e3:.2f} ms  "
              + "  ".join(f"{k}={v:.2f}" for k, v in u.items() if k != "F"))
    (OUT / "cost_units.json").write_text(json.dumps({"models": rows}, indent=2))


if __name__ == "__main__":
    main()
