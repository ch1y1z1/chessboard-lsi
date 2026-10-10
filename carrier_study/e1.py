"""E1 估计器对比：傅里叶链（默认 / 调优） vs 直接光强最小二乘（MLE）。

每个数据集（真值 x 场景 x 噪声 x 种子）记录：
    fourier_default, fourier_tuned   傅里叶链
    mle                              多起点 GN（傅里叶调优初值、零、两个随机初值）取最小损失
    ls_unweighted                    仅 Poisson：不加权最小二乘（检验 P1 的加权必要性）
    mle_pix                          仅 realistic：像素积分建模（S = 2）的 MLE
误差全部相对真值 q*；白化度量用真值处 FIM（真实噪声方差）。

运行（由 m2_pilot / m3_main 调用）。
"""

from __future__ import annotations

import time

import numpy as np

from . import config
from .common import NOISES
from .estimators import (
    IntegratedModel,
    chain_for,
    fim_for,
    fourier_q,
    irls,
    mle,
    observe,
    random_inits,
)
from .fourier import FourierParams
from .generator import MATCHED, REALISTIC, Generator
from .metrics import Whitener
from .nominal import NominalModel
from .problem import LSProblem
from .solvers import run

SCENES = {"matched": MATCHED, "realistic": REALISTIC}


def _row_metrics(dq: np.ndarray, W: Whitener, indices) -> dict:
    if not np.all(np.isfinite(dq)):
        return {"wf": np.nan, "wf_notilt": np.nan, "d2": np.nan}
    return {"wf": float(np.linalg.norm(dq)), "wf_notilt": float(np.linalg.norm(dq[2:])),
            "d2": float(W.d_rel(dq) ** 2)}


def worker(task) -> tuple[list[dict], dict]:
    """task = (truth, p, scene_name, noise_labels, n_seeds, tuned_params: {label: FourierParams})"""
    tr, p, scene_name, noise_labels, n_seeds, tuned = task
    t0 = time.time()
    idx = config.indices_for(p)
    cfg = config.system()
    model = NominalModel(cfg, idx)
    clean = Generator(cfg, idx, SCENES[scene_name]).clean(tr.q)
    chain_def = chain_for(cfg, idx, FourierParams())
    chains = {}
    pix_model = IntegratedModel(cfg, idx, S=2) if scene_name == "realistic" else None
    rows, dqs = [], {}
    for lab in noise_labels:
        nz = next(n for n in NOISES if n.label == lab)
        params = tuned.get(lab, FourierParams()) if tuned else FourierParams()
        key = params.label()
        if key not in chains:
            chains[key] = chain_for(cfg, idx, params)
        chain = chains[key]
        for seed in range(n_seeds):
            rng = config.rng(config.SEED_NOISE, config.key(tr.tid), NOISES.index(nz), seed,
                             0 if scene_name == "matched" else 1)
            obs = observe(clean, nz, rng)
            W = Whitener(fim_for(model, tr.q, obs))
            y = model.take(obs.y)
            est = {"fourier_default": fourier_q(chain_def, obs.y), "fourier_tuned": fourier_q(chain, obs.y)}
            inits = {"fourier": est["fourier_tuned"] if np.all(np.isfinite(est["fourier_tuned"])) else np.zeros(p),
                     "zero": np.zeros(p)}
            inits.update(random_inits(p, (0.1, 0.3), config.rng(config.SEED_INIT, seed, 7)))
            # 多起点在不加权目标上选解；Poisson 再做 IRLS 加权精修（近似 MLE）
            m = mle(model, y, None, inits)
            if nz.kind == "poisson":
                est["ls_unweighted"] = m["q"]
                est["mle"] = irls(model, y, obs, m["q"])
            else:
                est["mle"] = m["q"]
            if pix_model is not None:
                if nz.kind == "poisson":
                    est["mle_pix"] = irls(pix_model, pix_model.take(obs.y), obs, est["mle"])
                else:
                    P = LSProblem(pix_model, pix_model.take(obs.y), None)
                    est["mle_pix"] = run("gn", P, est["mle"], max_iter=200)["q"]
            base = {"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "rep": tr.rep, "p": p,
                    "scene": scene_name, "noise": lab, "seed": seed,
                    "crb_wf": W.crb_wavefront_rms, "fourier_params": key,
                    "mle_n_distinct": m["n_distinct"], "mle_n_agree": m["n_agree"],
                    "mle_best_start": m["best_start"],
                    "mle_loss_zero_minus_best": m["start_losses"]["zero"] - m["loss"],
                    "mle_loss_fourier_minus_best": m["start_losses"]["fourier"] - m["loss"]}
            for name, q in est.items():
                dq = q - tr.q
                r = dict(base, estimator=name)
                r.update(_row_metrics(dq, W, idx))
                rows.append(r)
                dqs[(lab, seed, name)] = dq
    meta = {"tid": tr.tid, "scene": scene_name, "seconds": time.time() - t0,
            "dq": {f"{k[0]}|{k[1]}|{k[2]}": v.astype(np.float32) for k, v in dqs.items()}}
    return rows, meta
