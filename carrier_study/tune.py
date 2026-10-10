"""dev 集调参：傅里叶链参数（每个噪声档）与可调求解器的步长 / 学习率（每个 p）。

傅里叶链：网格 radius_factor x threshold x erode，目标为 dev 真值上波前 MSE 最小
         （失败或误差 > 1 wave 记为 1 wave）。
求解器  ：gd_fixed / nesterov_fixed 的 eta、adam 的 lr，各 8 档；在 dev 子集（16 真值
         x G30/G50 x 傅里叶初值）上取"到达 tau = 0.3 的成本"中位数最小者
         （未到达记为无穷；并列时比较到达比例与末端 d_rel）。各方法调参消耗同样的配置数。
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from . import config
from .common import NOISES, pmap, write_csv, write_json
from .e3 import worker as e3_worker
from .estimators import chain_for, fourier_q, observe
from .fourier import FourierParams
from .generator import MATCHED, Generator
from .solvers import TUNABLE
from .truths import truth_set

OUT = Path("results/carrier/m2")
FOURIER_GRID = [FourierParams(r, t, e) for r, t, e in itertools.product((2.0, 3.0, 4.0, 5.0, 6.0), (0.3, 0.5), (2, 4))]


def _fourier_worker(task) -> list[dict]:
    tr, n_seeds = task
    idx = config.INDICES_MAIN
    cfg = config.system()
    clean = Generator(cfg, idx, MATCHED).clean(tr.q)
    chains = [chain_for(cfg, idx, fp) for fp in FOURIER_GRID]
    rows = []
    for nz in NOISES:
        for seed in range(n_seeds):
            obs = observe(clean, nz, config.rng(config.SEED_NOISE, 6, config.key(tr.tid), NOISES.index(nz), seed))
            for fp, ch in zip(FOURIER_GRID, chains):
                q = fourier_q(ch, obs.y)
                err = float(np.linalg.norm(q - tr.q)) if np.all(np.isfinite(q)) else np.inf
                rows.append({"tid": tr.tid, "rms": tr.rms, "shape": tr.shape, "noise": nz.label, "seed": seed,
                             "params": fp.label(), "wf": err})
    return rows


def tune_fourier(n_seeds: int = 2) -> dict:
    trs = truth_set("dev", config.INDICES_MAIN)
    rows = [r for rr in pmap(_fourier_worker, [(tr, n_seeds) for tr in trs]) for r in rr]
    write_csv(OUT / "fourier_tuning_runs.csv.gz", rows)
    best = {}
    for nz in NOISES:
        scores = {}
        for fp in FOURIER_GRID:
            e = [min(r["wf"], 1.0) for r in rows if r["noise"] == nz.label and r["params"] == fp.label()]
            scores[fp.label()] = float(np.mean(np.square(e)))
        lab = min(scores, key=scores.get)
        fp = next(f for f in FOURIER_GRID if f.label() == lab)
        best[nz.label] = {"radius_factor": fp.radius_factor, "threshold": fp.threshold, "erode": fp.erode,
                          "mse": scores[lab], "mse_default": scores[FourierParams().label()]}
    write_json(OUT / "fourier_tuning.json", best)
    return best


def load_fourier_tuning() -> dict[str, FourierParams]:
    import json
    data = json.loads((OUT / "fourier_tuning.json").read_text())
    return {k: FourierParams(v["radius_factor"], v["threshold"], int(v["erode"])) for k, v in data.items()}


def tune_solvers() -> dict:
    ft = load_fourier_tuning()
    trs = truth_set("dev", config.INDICES_MAIN, rms_levels=(0.03, 0.1, 0.3, 1.0), reps=1)
    trs12 = truth_set("dev", config.INDICES_CTRL, rms_levels=(0.03, 0.1, 0.3, 1.0), reps=1)
    configs = [(f"{m}:{name}={v:g}", m, {name: v}) for m, (name, _, grid) in TUNABLE.items() for v in grid]
    tasks = [(tr, p, nl, 0, ("fourier",), configs, ft[nl])
             for p, group in ((35, trs), (12, trs12)) for tr in group for nl in ("G30", "G50")]
    t0 = time.time()
    rows = [r for rr in pmap(e3_worker, tasks) for r in rr]
    for r in rows:
        r.pop("curve", None)
    write_csv(OUT / "solver_tuning_runs.csv.gz", rows)
    best = {}
    for p in (35, 12):
        best[p] = {}
        for m, (name, _, grid) in TUNABLE.items():
            stats = []
            for v in grid:
                rr = [r for r in rows if r["p"] == p and r["config"] == f"{m}:{name}={v:g}"]
                c = np.array([r["cost_t0.3"] for r in rr])
                stats.append((float(np.median(c)), -float(np.mean(np.isfinite(c))),
                              float(np.mean([r["final_d"] for r in rr])), v))
            stats.sort()
            best[p][m] = stats[0][3]
            best[p][f"{m}_table"] = [{"value": s[3], "median_cost_t0.3": s[0], "hit_frac": -s[1], "mean_final_d": s[2]}
                                     for s in sorted(stats, key=lambda z: z[3])]
    best["tuning_runs"] = len(rows)
    best["tuning_seconds"] = time.time() - t0
    best["tuning_cost_units_per_method"] = {
        m: float(sum(r["cost"] for r in rows if r["method"] == m)) for m in TUNABLE}
    write_json(OUT / "solver_tuning.json", best)
    return best


def load_solver_tuning(p: int) -> dict:
    import json
    data = json.loads((OUT / "solver_tuning.json").read_text())
    return {m: data[str(p)][m] for m in TUNABLE}
