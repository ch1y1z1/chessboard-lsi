"""结果汇总：按真值聚类的配对 bootstrap、各实验的汇总表与图。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

FIG = Path("reports/carrier/figures")


def setup_matplotlib():
    """Agg 后端 + 中文字体（macOS 自带 Hiragino / PingFang / Heiti）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def load_rows(path: Path) -> list[dict]:
    from .common import read_csv
    rows = read_csv(path)
    out = []
    for r in rows:
        out.append({k: (num(v) if k not in ("tid", "shape", "scene", "noise", "estimator", "config", "method",
                                              "status", "init", "mode", "basin", "factor", "fourier_params",
                                              "mle_best_start", "multistart_best_start", "curve") else v)
                    for k, v in r.items()})
    return out


def group(rows: Iterable[dict], *keys) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def cluster_bootstrap(values_by_cluster: dict, stat: Callable[[np.ndarray], float], n_boot: int = 4000,
                      seed: int = 0) -> tuple[float, float, float]:
    """按簇（真值）重采样；values_by_cluster: {簇: 数组}。返回 (点估计, 2.5%, 97.5%)。"""
    keys = list(values_by_cluster)
    arrs = [np.asarray(values_by_cluster[k], float) for k in keys]
    point = stat(np.concatenate(arrs))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(arrs), len(arrs))
        boots.append(stat(np.concatenate([arrs[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_ratio_ci(a_by_cluster: dict, b_by_cluster: dict, n_boot: int = 4000, seed: int = 0):
    """簇内取均值后的比值 mean(a)/mean(b) 的配对 bootstrap CI。"""
    keys = [k for k in a_by_cluster if k in b_by_cluster]
    A = np.array([np.mean(a_by_cluster[k]) for k in keys])
    B = np.array([np.mean(b_by_cluster[k]) for k in keys])
    point = A.mean() / B.mean()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(keys), len(keys))
        boots.append(A[i].mean() / B[i].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(keys)


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—" if x is None or np.isnan(x) else "∞"
    if abs(x) >= 1000 or (abs(x) < 1e-2 and x != 0):
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
