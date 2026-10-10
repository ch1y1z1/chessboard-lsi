"""E2 可靠性与吸引盆（无噪声 matched：景观问题，参考解即真值）。

初值 q0 = q* + r u（perturbed，围绕真值）或 q0 = r u（random，与真值无关），u 为 q 空间
单位球上的均匀随机方向；另测零初值与傅里叶初值。每次运行后用 GN 精修 50 步：
精修结果与 q* 的距离 < 1e-6 即判定"进入全局吸引盆"。这样判定的是求解器能否把
参数带进正确的盆，而不是它的末端精度。失败类型：符号（q ≈ -q*）、倾斜周期别名、其他。
"""

from __future__ import annotations

import numpy as np

from . import config
from .calibrate import load_units
from .estimators import chain_for, fourier_q
from .generator import Generator
from .nominal import NominalModel
from .problem import EarlyStop, LSProblem
from .solvers import run

R_LEVELS = (0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
METHODS = ("gn", "lm_minpack", "trf", "bfgs", "lbfgs", "nesterov", "adam")
BUDGET = 1500.0


def classify(q: np.ndarray, q_star: np.ndarray, s: float, indices) -> str:
    d = q - q_star
    if np.linalg.norm(d) < 1e-6:
        return "global"
    if np.linalg.norm(q + q_star) < 1e-4:
        return "sign"
    # 倾斜周期别名：Z2/Z3 的 c 系数按 1/s 周期等价（q 坐标乘 rms = 0.5）
    period_q = 0.5 / s
    tilt = [k for k, j in enumerate(indices) if j in (2, 3)]
    rest = [k for k in range(len(indices)) if k not in tilt]
    if np.linalg.norm(d[rest]) < 1e-4:
        frac = d[tilt] / period_q
        if np.all(np.abs(frac - np.round(frac)) < 1e-3):
            return "tilt_alias"
    return "other"


def worker(task) -> list[dict]:
    """task = (truth, period, p, mode, r, n_dirs, methods, tuned)"""
    tr, period, p, mode, r, n_dirs, methods, tuned = task
    idx = config.indices_for(p)
    cfg = config.system(period)
    model = NominalModel(cfg, idx)
    q_star = tr.q
    y = model.intensity(q_star)
    units = load_units(p, period)
    if mode in ("perturbed", "random"):
        rng = config.rng(config.SEED_INIT, 2, config.key(tr.tid), int(period), int(round(r * 1000)), 0 if mode == "perturbed" else 1)
        U = rng.normal(size=(n_dirs, p))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        inits = [(i, (q_star if mode == "perturbed" else 0.0) + r * U[i]) for i in range(n_dirs)]
    elif mode == "zero":
        inits = [(0, np.zeros(p))]
    else:  # fourier
        img = Generator(cfg, idx).clean(q_star)
        inits = [(0, fourier_q(chain_for(cfg, idx, None), img))]
    rows = []
    for di, q0 in inits:
        for m in methods:
            params = {}
            if m == "adam":
                params = {"lr": tuned.get("adam", 1e-3)} if tuned else {}

            def stop(q, loss, cost, improved, q_star=q_star):
                if improved and np.linalg.norm(q - q_star) < 1e-5:
                    raise EarlyStop

            P = LSProblem(model, y, units=units, budget=BUDGET, observer=stop)
            if not np.all(np.isfinite(q0)):
                rows.append({"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "period_um": period, "s": cfg.s,
                             "p": p, "mode": mode, "r": r, "dir": di, "method": m, "status": "init_failed",
                             "basin": "init_failed"})
                continue
            res = run(m, P, q0, **params)
            Pp = LSProblem(model, y)
            pol = run("gn", Pp, res["q"], max_iter=50)["q"]
            rows.append({"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "period_um": period, "s": cfg.s,
                         "p": p, "mode": mode, "r": r, "dir": di, "method": m, "status": res["status"],
                         "cost": res["cost"], "final_err": float(np.linalg.norm(res["q"] - q_star)),
                         "polished_err": float(np.linalg.norm(pol - q_star)),
                         "polished_loss": float(Pp.best_loss),
                         "basin": classify(pol, q_star, cfg.s, idx)})
    return rows
