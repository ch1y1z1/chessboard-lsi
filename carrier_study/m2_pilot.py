"""M2 pilot（dev 集）：调参 + 小规模 E1–E4 + 耗时外推（供 D1 算力决策）。

运行：  uv run python -m carrier_study.m2_pilot [steps...]
        steps 取自 fourier solvers e1 e2 e3 e4 d1，默认全部
输出：  results/carrier/m2/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .common import NOISES, pmap, workers, write_csv, write_json
from . import e1, e2, e3, e4
from .tune import OUT, load_fourier_tuning, load_solver_tuning, tune_fourier, tune_solvers
from .truths import truth_set

TIMING = OUT / "pilot_timing.json"


def _timing() -> dict:
    import json
    return json.loads(TIMING.read_text()) if TIMING.exists() else {}


def _save_timing(key: str, seconds: float, n_tasks: int, unit: str) -> None:
    t = _timing()
    t[key] = {"wall_seconds": seconds, "n_tasks": n_tasks, "workers": workers(),
              "cpu_seconds_per_task": seconds * workers() / max(n_tasks, 1), "unit": unit}
    write_json(TIMING, t)


def step_fourier():
    t0 = time.time()
    best = tune_fourier(n_seeds=2)
    _save_timing("fourier_tuning", time.time() - t0, 40, "dev truth (8 noises x 2 seeds x 20 configs)")
    for k, v in best.items():
        print(f"  {k}: r={v['radius_factor']} t={v['threshold']} e={v['erode']}  "
              f"MSE {v['mse']:.3g} (default {v['mse_default']:.3g})")


def step_solvers():
    t0 = time.time()
    best = tune_solvers()
    _save_timing("solver_tuning", time.time() - t0, best["tuning_runs"], "solver run")
    for p in (35, 12):
        print(f"  p={p}: " + ", ".join(f"{m}={best[p][m]:g}" for m in ("gd_fixed", "nesterov_fixed", "adam")))


def step_e1():
    ft = load_fourier_tuning()
    trs = truth_set("dev", config.INDICES_MAIN)
    tasks = [(tr, 35, sc, [n.label for n in NOISES], 1, ft) for tr in trs for sc in ("matched", "realistic")]
    t0 = time.time()
    out = pmap(e1.worker, tasks)
    _save_timing("e1", time.time() - t0, len(tasks), "truth x scene (8 noises x 1 seed)")
    rows = [r for rr, _ in out for r in rr]
    write_csv(OUT / "e1_pilot.csv.gz", rows)
    secs = [m["seconds"] for _, m in out]
    print(f"  E1 pilot: {len(rows)} rows, task seconds median {np.median(secs):.1f} max {np.max(secs):.1f}")


def step_e2():
    trs = [tr for tr in truth_set("dev", config.INDICES_MAIN) if tr.rep == 0 and
           (tr.shape, tr.rms) in (("pow2", 0.3), ("flat", 1.0))]
    adam_lr = load_solver_tuning(35)["adam"]
    tasks = [(tr, 30.0, 35, mode, r, 4, e2.METHODS, {"adam": adam_lr})
             for tr in trs for mode in ("perturbed", "random") for r in (0.1, 1.0, 3.0)]
    tasks += [(tr, 30.0, 35, mode, 0.0, 1, e2.METHODS, {"adam": adam_lr}) for tr in trs for mode in ("zero", "fourier")]
    t0 = time.time()
    rows = [r for rr in pmap(e2.worker, tasks) for r in rr]
    n_dir_runs = sum(4 for t in tasks if t[3] in ("perturbed", "random")) + sum(1 for t in tasks if t[3] in ("zero", "fourier"))
    _save_timing("e2", time.time() - t0, n_dir_runs, "initial point (7 methods)")
    write_csv(OUT / "e2_pilot.csv.gz", rows)
    print(f"  E2 pilot: {len(rows)} runs")


def step_e3():
    ft = load_fourier_tuning()
    tasks = []
    for p in (35, 12):
        tuned = load_solver_tuning(p)
        configs = e3.solver_configs(tuned)
        trs = [tr for tr in truth_set("dev", config.indices_for(p), rms_levels=(0.1, 1.0), reps=1)]
        tasks += [(tr, p, nz.label, 0, ("fourier", "zero"), configs, ft[nz.label]) for tr in trs for nz in NOISES]
    t0 = time.time()
    rows = [r for rr in pmap(e3.worker, tasks) for r in rr]
    _save_timing("e3", time.time() - t0, len(tasks), "dataset (2 inits x 16 configs)")
    for r in rows:
        r["curve"] = list(r["curve"])
    write_csv(OUT / "e3_pilot.csv.gz", rows)
    print(f"  E3 pilot: {len(rows)} runs from {len(tasks)} datasets")


def step_e4():
    ft = load_fourier_tuning()
    trs = [tr for tr in truth_set("dev", config.INDICES_MAIN) if tr.rep == 0 and tr.rms in (0.3, 1.0) and tr.shape in ("pow1", "sparse")]
    t0 = time.time()
    rows = [r for rr in pmap(e4.worker_mismatch, [(tr, 35, ft["G40"], 1) for tr in trs]) for r in rr]
    _save_timing("e4_mismatch", time.time() - t0, len(trs), "truth (31 scenes x (1 clean + 1 noisy))")
    write_csv(OUT / "e4_mismatch_pilot.csv.gz", rows)
    big = truth_set("dev", config.INDICES_MAIN, rms_levels=(2.0, 4.0), shapes=("pow1", "flat"), reps=1)
    t0 = time.time()
    rows = [r for rr in pmap(e4.worker_large, [(tr, 35, sc, ft["G40"]) for tr in big for sc in ("matched", "realistic")]) for r in rr]
    _save_timing("e4_large", time.time() - t0, 2 * len(big), "truth x scene")
    write_csv(OUT / "e4_large_pilot.csv.gz", rows)
    print(f"  E4 pilot: mismatch {len(trs)} truths, large {len(big)} truths")


# --------------------------------------------------------------------------- #
# D1：按 pilot 实测耗时外推主实验
# --------------------------------------------------------------------------- #
MAIN_PLAN = {
    "e1": {"tasks": 112 * 2, "scale_per_task": 4.0, "desc": "112 test truths x 2 scenes x 8 noises x 4 seeds"},
    "e2": {"tasks": 3 * 6 * (2 * 9 * 12 + 2) + 6 * 2 * 9 * 12, "scale_per_task": 1.0,
           "desc": "p=35: 3 periods x 6 truths x (2 modes x 9 r x 12 dirs + zero + fourier); p=12: 6 truths x 216"},
    "e3": {"tasks": 32 * 8 * 2, "scale_per_task": 1.0, "desc": "32 test truths x 8 noises x 2 p (2 inits x 16 configs)"},
    "e4_mismatch": {"tasks": 12, "scale_per_task": 5.0 / 2.0, "desc": "12 test truths x 31 scenes x (1 clean + 4 noisy)"},
    "e4_large": {"tasks": 56 * 2, "scale_per_task": 1.0, "desc": "56 truths (RMS 0.5-5) x 2 scenes"},
}


def step_d1():
    t = _timing()
    est = {}
    total = 0.0
    for k, plan in MAIN_PLAN.items():
        cpu = t[k]["cpu_seconds_per_task"] * plan["scale_per_task"] * plan["tasks"]
        wall = cpu / workers()
        est[k] = {"cpu_hours": cpu / 3600, "wall_hours_at_workers": wall / 3600, **plan}
        total += wall
    est["total_wall_hours"] = total / 3600
    est["workers"] = workers()
    write_json(OUT / "d1_estimate.json", est)
    for k, v in est.items():
        if isinstance(v, dict):
            print(f"  {k:12s} {v['wall_hours_at_workers']:.2f} h  ({v['desc']})")
    print(f"  total {est['total_wall_hours']:.2f} h with {workers()} workers")
    # 决策规则（用户授权"小规模测试后再定"，此处按预先写定的规则自动取舍）：
    # 全量 <= 10 h：按计划全量；否则依次缩减 E1 种子 4->2、E2 方向 12->8、E3 重复 2->1，直到 <= 10 h。
    decision = {"e1": {"seeds": 4}, "e2": {"dirs": 12}, "e3": {"reps": 2}, "e4": {"seeds": 4}}
    hours = {k: est[k]["wall_hours_at_workers"] for k in MAIN_PLAN}
    steps = [("e1", "seeds", 2, 0.5), ("e2", "dirs", 8, 8 / 12), ("e3", "reps", 1, 0.5)]
    for exp, field, val, factor in steps:
        if sum(hours.values()) <= 10.0:
            break
        decision[exp][field] = val
        hours[exp] *= factor
    decision["estimated_wall_hours"] = sum(hours.values())
    decision["budget_rule_hours"] = 10.0
    write_json(OUT / "d1_decision.json", decision)
    print(f"  decision {decision}")


STEPS = {"fourier": step_fourier, "solvers": step_solvers, "e1": step_e1, "e2": step_e2,
         "e3": step_e3, "e4": step_e4, "d1": step_d1}


def main() -> None:
    steps = sys.argv[1:] or list(STEPS)
    for s in steps:
        t0 = time.time()
        print(f"[{s}]", flush=True)
        STEPS[s]()
        print(f"[{s}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
