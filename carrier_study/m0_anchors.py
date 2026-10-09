"""阶段 0.5：用新框架复现四个旧锚点（p = 12，与旧实验同配置）。

    (a) 无噪声下直接拟合收敛到 1e-10 量级
    (b) 傅里叶链偏差地板（n = 256，演示真值）：Z2 +3.5e-3、Z7 +3.2e-3、Z10 +1.5e-3、Z4 +6.8e-4
    (c) 单真值 40 dB、64 种子：MLE 达到 CRB（eta ≈ 1），傅里叶链 eta ≈ 0.60
    (d) GN/LM 约 7–9 次前向收敛；GD 500 次前向远未达同等精度

运行：  uv run python -m carrier_study.m0_anchors
输出：  results/carrier/m0/anchors.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .calibrate import load_units
from .fourier import FourierChain
from .generator import Generator, Noise, add_noise
from .metrics import Whitener, fim
from .nominal import NominalModel
from .problem import LSProblem
from .solvers import run
from .truths import sample_q
from .zern import to_c, to_q

OUT = Path("results/carrier/m0")
IDX = config.INDICES_CTRL
DEMO_C = dict(zip([4, 5, 6, 7, 8], [0.31, -0.12, 0.07, 0.42, 0.05]))


def demo_q() -> np.ndarray:
    return to_q(np.array([DEMO_C.get(j, 0.0) for j in IDX]), IDX)


def anchor_a() -> dict:
    cfg = config.system(n=128)
    model = NominalModel(cfg, IDX)
    q = demo_q()
    img = Generator(cfg, IDX).clean(q)
    y = model.take(img)
    q_four, _ = FourierChain(cfg, IDX).estimate(img)
    out = {}
    for init_name, q0 in (("zero", np.zeros(len(IDX))), ("fourier", q_four)):
        for m in ("gn", "lm_main", "lm_minpack", "trf"):
            P = LSProblem(model, y)
            res = run(m, P, q0)
            out[f"{m}/{init_name}"] = {
                "max_coeff_err_waves": float(np.abs(to_c(res["q"] - q, IDX)).max()),
                "n_forward": res["counts"].get("F", 0),
                "status": res["status"],
            }
    return out


def anchor_b() -> dict:
    cfg = config.system(n=256)
    q = demo_q()
    img = Generator(cfg, IDX).clean(q)
    q_four, _ = FourierChain(cfg, IDX).estimate(img)
    err_c = to_c(q_four - q, IDX)
    return {f"Z{j}": float(e) for j, e in zip(IDX, err_c)}


def anchor_c(n_seeds: int = 64) -> dict:
    cfg = config.system(n=256)
    model = NominalModel(cfg, IDX)
    q = demo_q()
    img = Generator(cfg, IDX).clean(q)
    chain = FourierChain(cfg, IDX)
    noise = Noise("gauss", 40.0)
    sigma2 = (img.max() / 10.0 ** (40.0 / 20.0)) ** 2
    W = Whitener(fim(model, q, sigma2))
    est = {"mle": [], "fourier": []}
    for k in range(n_seeds):
        y, _, _ = add_noise(img, noise, config.rng(config.SEED_NOISE, 0, k))
        q_four, _ = chain.estimate(y)
        P = LSProblem(model, model.take(y))
        res = run("gn", P, q_four)
        est["mle"].append(res["q"])
        est["fourier"].append(q_four)
    out = {"sigma": float(np.sqrt(sigma2)), "crb_wavefront_rms": W.crb_wavefront_rms}
    for name, arr in est.items():
        arr = np.array(arr)
        dq = arr - q
        cov = np.cov(arr.T)
        eta = np.diag(W.crb) / np.diag(cov)
        out[name] = {
            "eta_median": float(np.median(eta)),
            "eta_per_coeff": eta.tolist(),
            "trace_ratio_var": float(np.trace(W.F @ cov) / W.p),
            "mean_d_rel2": float(np.mean(W.d_rel(dq) ** 2)),
            "bias_c_waves": to_c(dq.mean(axis=0), IDX).tolist(),
        }
    return out


def anchor_d() -> dict:
    cfg = config.system(n=128)
    model = NominalModel(cfg, IDX)
    units = load_units(12)
    rows = []
    for ti, rms in enumerate((0.01, 0.03, 0.1, 0.3, 1.0)):
        q = sample_q("flat", rms, IDX, config.rng(config.SEED_DEV, 99, ti))
        y = model.intensity(q)
        for ii, r0 in enumerate((0.03, 0.1, 0.3)):
            u = config.rng(config.SEED_INIT, 99, ii).normal(size=len(IDX))
            q0 = u / np.linalg.norm(u) * r0
            for m in ("gn", "lm_main", "lm_minpack", "trf", "gd", "bfgs"):
                first = {}

                def obs(qq, loss, cost, improved, q=q, first=first):
                    if improved and "hit" not in first and np.abs(to_c(qq - q, IDX)).max() < 1e-6:
                        first["hit"] = cost

                budget = 500.0 if m in ("gd", "bfgs") else np.inf
                P = LSProblem(model, y, units=units, budget=budget, observer=obs)
                res = run(m, P, q0)
                rows.append({
                    "method": m, "truth_rms": rms, "init_rms": r0,
                    "n_forward_total": res["counts"].get("F", 0),
                    "cost_to_1e-6": first.get("hit"),
                    "final_wf_err": float(np.linalg.norm(res["q"] - q)),
                    "status": res["status"],
                })
    summary = {}
    for m in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == m]
        hits = [r["cost_to_1e-6"] for r in rr if r["cost_to_1e-6"] is not None]
        summary[m] = {
            "n_runs": len(rr),
            "n_reached_1e-6": len(hits),
            "median_cost_to_1e-6": float(np.median(hits)) if hits else None,
            "median_forward_total": float(np.median([r["n_forward_total"] for r in rr])),
            "median_final_wf_err": float(np.median([r["final_wf_err"] for r in rr])),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"a": anchor_a(), "b": anchor_b()}
    print("(a)", json.dumps(res["a"], indent=1))
    print("(b)", {k: f"{v:+.2e}" for k, v in res["b"].items()})
    res["c"] = anchor_c()
    for k in ("mle", "fourier"):
        c = res["c"][k]
        print(f"(c) {k}: eta_median={c['eta_median']:.3f} trace_ratio_var={c['trace_ratio_var']:.3f} "
              f"mean_d_rel2={c['mean_d_rel2']:.3f}")
    res["d"] = anchor_d()
    print("(d)", json.dumps(res["d"]["summary"], indent=1))
    (OUT / "anchors.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
