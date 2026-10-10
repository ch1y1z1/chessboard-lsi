"""E4 失配与适用边界。

A 部分（单因素失配）：每个真值 x 失配场景，比较傅里叶链（调优）、名义 MLE、
  像素积分建模的 MLE（mle_pix，S = 2）、带级次振幅 nuisance 的 MLE（mle_amp），
  无噪声（纯偏差）与 G40 x 4 种子；同时给出失配偏差的一阶预测（P3）。
B 部分（大像差扫描）：RMS 0.5–5 waves，matched / realistic，G40 一个种子；比较傅里叶链、
  不同初值的 GN、MINPACK LM、TRF、L-BFGS 以及多起点 MLE，并记录定义域统计（P7）。
"""

from __future__ import annotations

import numpy as np

from . import config
from .common import NOISE_BY_LABEL, mismatch_scenes
from .estimators import AmplitudeModel, IntegratedModel, chain_for, fourier_q, mle, observe, random_inits
from .fourier import FourierParams
from .generator import MATCHED, REALISTIC, Generator, Noise
from .metrics import Whitener
from .nominal import NominalModel
from .problem import LSProblem
from .solvers import run
from .theory import domain_stats
from .truths import tail_q

G40 = NOISE_BY_LABEL["G40"]


def _gn(model, y, q0, max_iter=200):
    return run("gn", LSProblem(model, y), q0, max_iter=max_iter)["q"]


def _metrics(prefix: str, dq: np.ndarray, W: Whitener) -> dict:
    if not np.all(np.isfinite(dq)):
        return {f"{prefix}_wf": np.nan, f"{prefix}_wf_notilt": np.nan, f"{prefix}_d2": np.nan}
    return {f"{prefix}_wf": float(np.linalg.norm(dq)), f"{prefix}_wf_notilt": float(np.linalg.norm(dq[2:])),
            f"{prefix}_d2": float(W.d_rel(dq) ** 2)}


def worker_mismatch(task) -> list[dict]:
    """task = (truth, p, fourier_params, n_seeds)"""
    tr, p, fparams, n_seeds = task
    idx = config.indices_for(p)
    cfg = config.system()
    model = NominalModel(cfg, idx)
    pix_model = IntegratedModel(cfg, idx, S=2)
    amp_model = AmplitudeModel(model)
    chain = chain_for(cfg, idx, fparams)
    q = tr.q
    img0 = Generator(cfg, idx, MATCHED).clean(q)
    lin = chain.linearize(img0)
    _, st = model.forward(q)
    J = model.jacobian(st)
    proj = np.linalg.solve(J.T @ J, J.T)
    sigma = img0.max() / 10.0 ** (G40.level / 20.0)
    W = Whitener(J.T @ J / sigma**2)
    rows = []
    scenes = [("matched", 0.0, MATCHED)] + mismatch_scenes(seed=config.SEED_MISMATCH)
    for name, level, scene in scenes:
        tq = None
        if name == "tail_frac":
            tq = tail_q(level * np.linalg.norm(q), config.rng(config.SEED_MISMATCH, 1, tr.rep))
        img = Generator(cfg, idx, scene).clean(q, tq)
        dI = img - img0
        pred_mle = proj @ model.take(dI)
        pred_four = (lin.q0 - q) + lin.apply(dI)[0]
        for seed in range(-1, n_seeds):
            if seed < 0:
                y_img = img
                noise = "none"
            else:
                obs = observe(img, G40, config.rng(config.SEED_NOISE, 4, config.key(tr.tid), config.key(name), int(level * 1e4), seed))
                y_img = obs.y
                noise = "G40"
            qf = fourier_q(chain, y_img)
            inits = {"fourier": qf if np.all(np.isfinite(qf)) else np.zeros(p), "zero": np.zeros(p)}
            m = mle(model, model.take(y_img), None, inits)
            q_mle = m["q"]
            q_pix = _gn(pix_model, pix_model.take(y_img), q_mle, max_iter=100)
            th = _gn(amp_model, amp_model.take(y_img), np.concatenate([q_mle, np.zeros(amp_model.K)]), max_iter=100)
            row = {"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "p": p, "factor": name, "level": level,
                   "noise": noise, "seed": seed, "crb_wf_G40": W.crb_wavefront_rms,
                   "mle_n_distinct": m["n_distinct"]}
            row.update(_metrics("fourier", qf - q, W))
            row.update(_metrics("mle", q_mle - q, W))
            row.update(_metrics("mle_pix", q_pix - q, W))
            row.update(_metrics("mle_amp", th[:p] - q, W))
            if seed < 0:
                row.update(_metrics("pred_mle", pred_mle, W))
                row.update(_metrics("pred_fourier", pred_four, W))
            rows.append(row)
    return rows


def worker_large(task) -> list[dict]:
    """task = (truth, p, scene_name, fourier_params)"""
    tr, p, scene_name, fparams = task
    idx = config.indices_for(p)
    cfg = config.system()
    model = NominalModel(cfg, idx)
    q = tr.q
    scene = MATCHED if scene_name == "matched" else REALISTIC
    clean = Generator(cfg, idx, scene).clean(q)
    obs = observe(clean, G40, config.rng(config.SEED_NOISE, 5, config.key(tr.tid), 0 if scene_name == "matched" else 1))
    y = model.take(obs.y)
    sigma = clean.max() / 10.0 ** (G40.level / 20.0)
    _, st = model.forward(q)
    J = model.jacobian(st)
    W = Whitener(J.T @ J / sigma**2)
    row = {"tid": tr.tid, "shape": tr.shape, "rms": tr.rms, "rep": tr.rep, "p": p, "scene": scene_name,
           "crb_wf": W.crb_wavefront_rms}
    row.update(domain_stats(cfg, idx, q))
    qf_def = fourier_q(chain_for(cfg, idx, FourierParams()), obs.y)
    qf = fourier_q(chain_for(cfg, idx, fparams), obs.y)
    row.update(_metrics("fourier_default", qf_def - q, W))
    row.update(_metrics("fourier_tuned", qf - q, W))
    q0f = qf if np.all(np.isfinite(qf)) else np.zeros(p)
    for m in ("gn", "lm_minpack", "trf", "lbfgs"):
        budget = 3000.0 if m == "lbfgs" else np.inf
        P = LSProblem(model, y, budget=budget)
        res = run(m, P, q0f, **({"max_iter": 300} if m == "gn" else {}))
        row.update(_metrics(f"{m}_from_fourier", res["q"] - q, W))
    row.update(_metrics("gn_from_zero", _gn(model, y, np.zeros(p), 300) - q, W))
    inits = {"fourier": q0f, "zero": np.zeros(p)}
    inits.update(random_inits(p, (0.3, 0.3, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0), config.rng(config.SEED_INIT, 5, config.key(tr.tid))))
    ms = mle(model, y, None, inits, max_iter=300)
    row.update(_metrics("mle_multistart", ms["q"] - q, W))
    row["multistart_n_distinct"] = ms["n_distinct"]
    row["multistart_best_start"] = ms["best_start"]
    # 真值处损失 vs 多起点最优损失：判断失败是"没找到"还是"目标函数本身的最小值不在真值附近"
    P = LSProblem(model, y)
    row["loss_at_truth_minus_best"] = float(P.loss(q) - ms["loss"])
    return [row]
