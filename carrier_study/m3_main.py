"""M3 主实验（test 集，协议在 G2 冻结、规模在 D1 决定）。

运行：  uv run python -m carrier_study.m3_main [steps...]
        steps 取自 e1 e2 e3 e3k e4m e4l，默认全部
输出：  results/carrier/m3/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config, e1, e2, e3, e4
from .common import NOISES, pmap, write_csv, write_json
from .tune import load_fourier_tuning, load_solver_tuning
from .truths import truth_set

OUT = Path("results/carrier/m3")
SCALE_FILE = Path("results/carrier/m2/d1_decision.json")


def scale() -> dict:
    return json.loads(SCALE_FILE.read_text())


def _log(msg: str) -> None:
    print(msg, flush=True)


def step_e1():
    sc = scale()["e1"]
    ft = load_fourier_tuning()
    trs = truth_set("test", config.INDICES_MAIN)
    tasks = [(tr, 35, s, [n.label for n in NOISES], sc["seeds"], ft) for tr in trs for s in ("matched", "realistic")]
    t0 = time.time()
    out = pmap(e1.worker, tasks)
    rows = [r for rr, _ in out for r in rr]
    write_csv(OUT / "e1.csv.gz", rows)
    dq = {f"{m['tid']}|{m['scene']}|{k}": v for _, m in out for k, v in m["dq"].items()}
    np.savez_compressed(OUT / "e1_dq.npz", **dq)
    _log(f"  e1 {len(rows)} rows in {time.time() - t0:.0f}s")


def step_e2():
    sc = scale()["e2"]
    adam_lr = load_solver_tuning(35)["adam"]
    trs = [tr for tr in truth_set("test", config.INDICES_MAIN, rms_levels=(0.1, 0.3, 1.0), shapes=("pow2", "flat"), reps=1)]
    tasks = [(tr, per, 35, mode, r, sc["dirs"], e2.METHODS, {"adam": adam_lr})
             for per in config.PERIODS_SHEAR_SCAN for tr in trs for mode in ("perturbed", "random") for r in e2.R_LEVELS]
    tasks += [(tr, per, 35, mode, 0.0, 1, e2.METHODS, {"adam": adam_lr})
              for per in config.PERIODS_SHEAR_SCAN for tr in trs for mode in ("zero", "fourier")]
    # p = 12（周期 30 µm）：检验吸引盆随拟合阶数的标度
    adam12 = load_solver_tuning(12)["adam"]
    trs12 = truth_set("test", config.INDICES_CTRL, rms_levels=(0.1, 0.3, 1.0), shapes=("pow2", "flat"), reps=1)
    tasks += [(tr, 30.0, 12, mode, r, sc["dirs"], e2.METHODS, {"adam": adam12})
              for tr in trs12 for mode in ("perturbed", "random") for r in e2.R_LEVELS]
    t0 = time.time()
    rows = [r for rr in pmap(e2.worker, tasks) for r in rr]
    write_csv(OUT / "e2.csv.gz", rows)
    _log(f"  e2 {len(rows)} runs in {time.time() - t0:.0f}s")


def step_e3():
    sc = scale()["e3"]
    ft = load_fourier_tuning()
    tasks = []
    for p in (35, 12):
        configs = e3.solver_configs(load_solver_tuning(p))
        trs = truth_set("test", config.indices_for(p), rms_levels=(0.03, 0.1, 0.3, 1.0), reps=sc["reps"])
        tasks += [(tr, p, nz.label, 0, ("fourier", "zero"), configs, ft[nz.label]) for tr in trs for nz in NOISES]
    t0 = time.time()
    rows = [r for rr in pmap(e3.worker, tasks) for r in rr]
    write_csv(OUT / "e3.csv.gz", rows)
    _log(f"  e3 {len(rows)} runs from {len(tasks)} datasets in {time.time() - t0:.0f}s")


def step_e3k():
    """P4 的条件数扫描：三档剪切量 x 两种 p，G40，傅里叶初值，代表性求解器。"""
    ft = load_fourier_tuning()
    tasks = []
    for p in (35, 12):
        tuned = load_solver_tuning(p)
        configs = [(c, m, prm) for c, m, prm in e3.solver_configs(tuned)
                   if c in ("gn", "lbfgs", "gd", "nesterov", "gd_fixed:tuned", "nesterov_fixed:tuned")]
        trs = truth_set("test", config.indices_for(p), rms_levels=(0.1, 0.3), reps=1)
        for per in config.PERIODS_SHEAR_SCAN:
            tasks += [(tr, p, "G40", 0, ("fourier",), configs, ft["G40"], per) for tr in trs]
    t0 = time.time()
    rows = [r for rr in pmap(e3.worker, tasks) for r in rr]
    write_csv(OUT / "e3_kappa.csv.gz", rows)
    _log(f"  e3k {len(rows)} runs in {time.time() - t0:.0f}s")


def step_e4m():
    sc = scale()["e4"]
    ft = load_fourier_tuning()
    trs = truth_set("test", config.INDICES_MAIN, rms_levels=(0.1, 0.3, 1.0), reps=1)
    t0 = time.time()
    rows = [r for rr in pmap(e4.worker_mismatch, [(tr, 35, ft["G40"], sc["seeds"]) for tr in trs]) for r in rr]
    write_csv(OUT / "e4_mismatch.csv.gz", rows)
    _log(f"  e4m {len(rows)} rows in {time.time() - t0:.0f}s")


def step_e4l():
    ft = load_fourier_tuning()
    trs = truth_set("test", config.INDICES_MAIN, rms_levels=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0), reps=2)
    t0 = time.time()
    rows = [r for rr in pmap(e4.worker_large, [(tr, 35, s, ft["G40"]) for tr in trs for s in ("matched", "realistic")]) for r in rr]
    write_csv(OUT / "e4_large.csv.gz", rows)
    _log(f"  e4l {len(rows)} rows in {time.time() - t0:.0f}s")


STEPS = {"e1": step_e1, "e2": step_e2, "e3": step_e3, "e3k": step_e3k, "e4m": step_e4m, "e4l": step_e4l}


def main() -> None:
    steps = sys.argv[1:] or list(STEPS)
    for s in steps:
        t0 = time.time()
        _log(f"[{s}] start")
        STEPS[s]()
        _log(f"[{s}] done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
