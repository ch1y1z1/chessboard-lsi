"""E3 汇总：到达精度目标的成本（前向当量）、配对加速比、performance profile、P5 检验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import fmt, group, md_table, setup_matplotlib
from .common import NOISES

NOISE_ORDER = [n.label for n in NOISES]
ORDER = ("gn", "lm_main", "lm_minpack", "trf", "bfgs", "lbfgs", "cg", "gd_bb", "nesterov", "gd",
         "adam:tuned", "adam:default", "nesterov_fixed:tuned", "nesterov_fixed:default",
         "gd_fixed:tuned", "gd_fixed:default")
LSQ = ("gn", "lm_main", "lm_minpack", "trf")
FIRST = ("bfgs", "lbfgs", "cg", "gd_bb", "nesterov", "gd", "adam:tuned", "nesterov_fixed:tuned", "gd_fixed:tuned")


def _median_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    pt = float(np.median(vals))
    bs = [np.median(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, float(lo), float(hi)


def cost_table(rows: list[dict], tau: float) -> dict:
    """{(p, init, noise, config): (中位成本, lo, hi, 到达比例, n)}；未到达记为无穷。"""
    key = f"cost_t{tau:g}"
    out = {}
    for (p, init, noise, cfg), rr in group(rows, "p", "init", "noise", "config").items():
        c = np.array([r[key] for r in rr], float)
        pt, lo, hi = _median_ci(c)
        out[(p, init, noise, cfg)] = {"median": pt, "lo": lo, "hi": hi, "hit": float(np.mean(np.isfinite(c))), "n": len(c)}
    return out


def speedup(rows: list[dict], tau: float, ref: str = "gn") -> dict:
    """配对比值 cost(config) / cost(ref) 的几何均值与 bootstrap CI（簇 = 数据集）。"""
    key = f"cost_t{tau:g}"
    out = {}
    for (p, init, noise), rr in group(rows, "p", "init", "noise").items():
        by = {}
        for r in rr:
            by.setdefault(r["config"], {})[(r["tid"], r["seed"])] = r[key]
        base = by.get(ref, {})
        for cfg, d in by.items():
            ks = [k for k in d if k in base and np.isfinite(base[k])]
            if not ks:
                continue
            lr = np.array([np.log(d[k] / base[k]) if np.isfinite(d[k]) else np.log(1e6) for k in ks])
            rng = np.random.default_rng(1)
            bs = [np.mean(lr[rng.integers(0, len(lr), len(lr))]) for _ in range(2000)]
            out[(p, init, noise, cfg)] = (float(np.exp(lr.mean())), float(np.exp(np.percentile(bs, 2.5))),
                                          float(np.exp(np.percentile(bs, 97.5))), len(ks))
    return out


def profile(rows: list[dict], tau: float, p: int, init: str, configs, thetas=np.geomspace(1, 1000, 61)) -> dict:
    """Dolan–Moré performance profile：rho_m(theta) = P(cost_m / min cost <= theta)。"""
    key = f"cost_t{tau:g}"
    inst = {}
    for r in rows:
        if r["p"] == p and r["init"] == init and r["config"] in configs:
            inst.setdefault((r["tid"], r["noise"], r["seed"]), {})[r["config"]] = r[key]
    res = {c: [] for c in configs}
    for d in inst.values():
        best = min(d.values())
        if not np.isfinite(best):
            continue
        for c in configs:
            res[c].append(d.get(c, np.inf) / best)
    return {"theta": thetas.tolist(),
            **{c: [float(np.mean(np.array(v) <= t)) for t in thetas] for c, v in res.items() if v}}


def p5_check(tab: dict, p: int, init: str = "fourier") -> dict:
    """P5：是否存在噪声档使最好的一阶/准牛顿方法中位成本低于 GN。"""
    out = {}
    for noise in NOISE_ORDER:
        gn = tab.get((p, init, noise, "gn"))
        if not gn:
            continue
        firsts = {c: tab[(p, init, noise, c)]["median"] for c in FIRST if (p, init, noise, c) in tab}
        best = min(firsts, key=firsts.get)
        out[noise] = {"gn": gn["median"], "best_first": best, "best_first_cost": firsts[best],
                      "first_cheaper": bool(firsts[best] < gn["median"])}
    return out


def figure_cost(tab: dict, tau: float, init: str, path: Path) -> None:
    plt = setup_matplotlib()
    gauss = [n for n in NOISE_ORDER if n.startswith("G")]
    snr = [float(n[1:]) for n in gauss]
    show = ("gn", "lm_minpack", "trf", "bfgs", "lbfgs", "gd_bb", "nesterov", "adam:tuned", "gd")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, p in zip(axes, (35, 12)):
        for i, c in enumerate(show):
            ys = [tab.get((p, init, n, c), {}).get("median", np.nan) for n in gauss]
            ls = "-" if c in LSQ else "--"
            ax.plot(snr, ys, ls=ls, marker="o", ms=3, label=c, color=f"C{i}")
        ax.set_yscale("log")
        ax.set_xlabel("峰值 SNR (dB)")
        ax.set_title(f"p = {p}（Z2–Z{p + 1}），{'傅里叶初值' if init == 'fourier' else '零初值'}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(f"到达 d_rel ≤ {tau:g} 的成本（前向当量，中位数）")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_profile(prof: dict, path: Path, title: str) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    th = prof["theta"]
    for i, (c, v) in enumerate((k, v) for k, v in prof.items() if k != "theta"):
        ax.plot(th, v, label=c, ls="-" if c in LSQ else "--", color=f"C{i % 10}")
    ax.set_xscale("log")
    ax.set_xlabel("θ（相对最便宜方法的成本倍数）")
    ax.set_ylabel("实例比例")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_costs(tab: dict, p: int, init: str, noises=None) -> str:
    noises = noises or NOISE_ORDER
    cfgs = [c for c in ORDER if any((p, init, n, c) in tab for n in noises)]
    body = []
    for c in cfgs:
        cells = []
        for n in noises:
            v = tab.get((p, init, n, c))
            if v is None:
                cells.append("")
            else:
                s = fmt(v["median"])
                if v["hit"] < 1:
                    s += f"（{v['hit']:.0%}）"
                cells.append(s)
        body.append([c] + cells)
    return md_table(["求解器"] + list(noises), body)
