"""E3 成本与精度（核心）：同一最小二乘目标上各求解器到达统计精度所需的真实成本。

每个数据集（真值 x p x 噪声 x 种子 x 初值）：
  1. 参考解 q_ref：多起点 GN（Poisson 为固定权重的加权目标，权重取 IRLS 收敛点的模型方差）
  2. 白化度量：d_rel(q) = sqrt((q - q_ref)^T FIM (q - q_ref) / p)，FIM 用真实噪声方差在 q_ref 处计算
  3. 每个求解器配置在预算内运行；按观测损失跟踪最佳点，记录其 d_rel 首次 <= tau 时的累计成本
     （前向当量，按串行标定的单位成本）；d_rel <= 0.01 时提前停止。
"""

from __future__ import annotations

import time

import numpy as np

from . import config
from .calibrate import load_units
from .common import NOISE_BY_LABEL
from .estimators import chain_for, fourier_q, irls, model_weights, mle, observe, random_inits
from .fourier import FourierParams
from .generator import MATCHED, Generator
from .metrics import Whitener
from .nominal import NominalModel
from .problem import EarlyStop, LSProblem
from .solvers import ALL_METHODS, TUNABLE, run

TAUS = (1.0, 0.3, 0.1, 0.03, 0.01)
BUDGET = 3000.0
CHECKPOINTS = np.geomspace(1.0, BUDGET, 41)


def solver_configs(tuned: dict | None) -> list[tuple[str, str, dict]]:
    """[(标签, 方法, 参数)]；可调方法同时给出默认值与调优值两个版本。"""
    out = []
    for m in ALL_METHODS:
        if m in TUNABLE:
            name, default, _ = TUNABLE[m]
            out.append((f"{m}:default", m, {name: default}))
            if tuned and m in tuned:
                out.append((f"{m}:tuned", m, {name: tuned[m]}))
        else:
            out.append((m, m, {}))
    return out


class _Tracker:
    def __init__(self, W: Whitener, q_ref: np.ndarray, q_true: np.ndarray, stop_at: float = 0.01):
        self.W, self.q_ref, self.q_true, self.stop_at = W, q_ref, q_true, stop_at
        self.hits: dict[float, float] = {}
        self.curve: list[tuple[float, float]] = []
        self.best_d = np.inf

    def __call__(self, q, loss, cost, improved):
        if not improved:
            return
        d = float(self.W.d_rel(q - self.q_ref))
        self.best_d = d
        self.curve.append((cost, d))
        for tau in TAUS:
            if tau not in self.hits and d <= tau:
                self.hits[tau] = cost
        if d <= self.stop_at:
            raise EarlyStop

    def at_checkpoints(self) -> list[float]:
        out, k, cur = [], 0, np.inf
        for c in CHECKPOINTS:
            while k < len(self.curve) and self.curve[k][0] <= c:
                cur = self.curve[k][1]
                k += 1
            out.append(cur)
        return out


def prepare(tr, p: int, noise_label: str, seed: int, fourier_params: FourierParams | None,
            period: float = config.PERIOD_UM) -> dict:
    idx = config.indices_for(p)
    cfg = config.system(period)
    model = NominalModel(cfg, idx)
    nz = NOISE_BY_LABEL[noise_label]
    clean = Generator(cfg, idx, MATCHED).clean(tr.q)
    obs = observe(clean, nz, config.rng(config.SEED_NOISE, 3, config.key(tr.tid), config.key(noise_label), seed))
    y = model.take(obs.y)
    q_four = fourier_q(chain_for(cfg, idx, fourier_params), obs.y)
    inits = {"fourier": q_four if np.all(np.isfinite(q_four)) else np.zeros(p), "zero": np.zeros(p)}
    inits.update(random_inits(p, (0.1, 0.3), config.rng(config.SEED_INIT, seed, 11)))
    m = mle(model, y, None, inits)
    w = None
    q_ref = m["q"]
    if nz.kind == "poisson":
        q_ref = irls(model, y, obs, q_ref)
        w = model_weights(model, q_ref, obs)
        P = LSProblem(model, y, w)
        q_ref = run("gn", P, q_ref, max_iter=200)["q"]
    # 参考解核验：梯度与精确 Hessian
    P = LSProblem(model, y, w)
    g = P.grad(q_ref)
    H = P.hessian(q_ref)
    _, st = model.forward(q_ref)
    J = model.jacobian(st)
    var = model.take(obs.var_true)
    W = Whitener((J / var[:, None]).T @ J)
    return {"model": model, "y": y, "w": w, "q_ref": q_ref, "W": W, "q_four": q_four,
            "q_true": tr.q, "mle_n_distinct": m["n_distinct"],
            "ref_grad_inf": float(np.abs(g).max()), "ref_hess_min_eig": float(np.linalg.eigvalsh(H)[0]),
            "ref_wf_err": float(np.linalg.norm(q_ref - tr.q))}


def worker(task) -> list[dict]:
    """task = (truth, p, noise_label, seed, init_kinds, configs, fourier_params[, period])"""
    tr, p, noise_label, seed, init_kinds, configs, fparams, *rest = task
    period = rest[0] if rest else config.PERIOD_UM
    data = prepare(tr, p, noise_label, seed, fparams, period)
    units = load_units(p, period)
    rows = []
    for init in init_kinds:
        q0 = data["q_four"] if init == "fourier" else np.zeros(p)
        if not np.all(np.isfinite(q0)):
            q0 = np.zeros(p)
        d0 = float(data["W"].d_rel(q0 - data["q_ref"]))
        for label, method, params in configs:
            tracker = _Tracker(data["W"], data["q_ref"], data["q_true"])
            P = LSProblem(data["model"], data["y"], data["w"], units=units, budget=BUDGET, observer=tracker)
            t0 = time.perf_counter()
            res = run(method, P, q0, **params)
            wall = time.perf_counter() - t0
            row = {"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "rep": tr.rep, "p": p, "period_um": period,
                   "noise": noise_label, "seed": seed, "init": init, "config": label, "method": method,
                   "status": res["status"], "cost": res["cost"], "wall_s": wall,
                   "n_points": res["n_points"], "d0": d0, "final_d": tracker.best_d,
                   "final_wf_err_true": float(np.linalg.norm(res["q"] - data["q_true"])) if res["q"] is not None else np.nan,
                   "mle_n_distinct": data["mle_n_distinct"], "ref_grad_inf": data["ref_grad_inf"],
                   "ref_hess_min_eig": data["ref_hess_min_eig"], "ref_wf_err": data["ref_wf_err"],
                   "crb_wf": data["W"].crb_wavefront_rms}
            for k, v in res["counts"].items():
                row[f"n_{k}"] = v
            for tau in TAUS:
                row[f"cost_t{tau:g}"] = tracker.hits.get(tau, np.inf)
            row["curve"] = tracker.at_checkpoints()
            rows.append(row)
    return rows
