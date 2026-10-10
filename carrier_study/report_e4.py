"""E4 汇总：失配容差曲线、P3（一阶偏差公式）、大像差下各管线的成功率与 P7。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import fmt, group, md_table, setup_matplotlib

EST4 = ("fourier", "mle", "mle_pix", "mle_amp")
LARGE_EST = ("fourier_default", "fourier_tuned", "gn_from_fourier", "lm_minpack_from_fourier", "trf_from_fourier",
             "lbfgs_from_fourier", "gn_from_zero", "mle_multistart")


def bias_table(rows: list[dict], metric: str = "wf") -> dict:
    out = {}
    clean = [r for r in rows if r["noise"] == "none"]
    for (f, lv), rr in group(clean, "factor", "level").items():
        d = {e: float(np.nanmedian([r[f"{e}_{metric}"] for r in rr])) for e in EST4}
        d["pred_mle"] = float(np.nanmedian([r[f"pred_mle_{metric}"] for r in rr]))
        d["pred_fourier"] = float(np.nanmedian([r[f"pred_fourier_{metric}"] for r in rr]))
        d["n"] = len(rr)
        out[(f, lv)] = d
    return out


def noisy_table(rows: list[dict]) -> dict:
    """G40：各估计器 NMSE（含偏差）与"傅里叶优于 MLE"的真值比例。"""
    out = {}
    noisy = [r for r in rows if r["noise"] == "G40"]
    for (f, lv), rr in group(noisy, "factor", "level").items():
        d = {e: float(np.nanmean([r[f"{e}_d2"] for r in rr])) for e in EST4}
        per_truth = group(rr, "tid")
        better = [np.nanmean([x["fourier_wf"] ** 2 for x in g]) < np.nanmean([x["mle_wf"] ** 2 for x in g])
                  for g in per_truth.values()]
        d["frac_fourier_better"] = float(np.mean(better))
        out[(f, lv)] = d
    return out


def p3_check(rows: list[dict]) -> dict:
    """一阶公式 vs 实测（无噪声）MLE 偏差的相对差，按因素取最小档。"""
    clean = [r for r in rows if r["noise"] == "none" and r["factor"] != "matched"]
    out = {}
    for f, rr in group(clean, "factor").items():
        lv = min(r["level"] for r in rr)
        sub = [r for r in rr if r["level"] == lv]
        rel = [abs(r["pred_mle_wf"] - r["mle_wf"]) / r["mle_wf"] for r in sub if r["mle_wf"] > 0]
        relf = [abs(r["pred_fourier_wf"] - r["fourier_wf"]) / r["fourier_wf"] for r in sub
                if np.isfinite(r["fourier_wf"]) and r["fourier_wf"] > 0]
        out[f[0]] = {"level": lv, "median_rel_err_mle": float(np.median(rel)) if rel else np.nan,
                     "median_rel_err_fourier": float(np.median(relf)) if relf else np.nan,
                     "pass_mle": bool(np.median(rel) < 0.2) if rel else False}
    return out


def figure_mismatch(tab: dict, path: Path, metric_label: str) -> None:
    plt = setup_matplotlib()
    factors = [f for f in dict.fromkeys(k[0] for k in tab) if f not in ("matched", "pixel")]
    n = len(factors)
    cols = 4
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 3.0 * rows_))
    axes = np.atleast_1d(axes).ravel()
    names = {"fourier": "傅里叶链", "mle": "MLE", "mle_pix": "MLE+像素积分", "mle_amp": "MLE+振幅"}
    for ax, f in zip(axes, factors):
        lv = sorted(k[1] for k in tab if k[0] == f)
        for i, e in enumerate(EST4):
            ax.plot(lv, [tab[(f, l)][e] for l in lv], marker="o", ms=3, color=f"C{i}", label=names[e])
        ax.plot(lv, [tab[(f, l)]["pred_mle"] for l in lv], ls=":", color="C1", label="MLE 一阶预测")
        ax.plot(lv, [tab[(f, l)]["pred_fourier"] for l in lv], ls=":", color="C0", label="傅里叶一阶预测")
        base = tab.get(("matched", 0.0))
        if base:
            ax.axhline(base["fourier"], color="C0", lw=0.6, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel(metric_label)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def large_success(rows: list[dict]) -> dict:
    out = {}
    for (scene, rms), rr in group(rows, "scene", "rms").items():
        d = {}
        for e in LARGE_EST:
            ok = [np.isfinite(r[f"{e}_wf"]) and r[f"{e}_wf"] <= max(0.01, 5 * r["crb_wf"]) for r in rr]
            d[e] = float(np.mean(ok))
        d["n"] = len(rr)
        d["lobe_freq_max_median"] = float(np.median([r["lobe_freq_max"] for r in rr]))
        d["beat_over_nyquist_median"] = float(np.median([r["beat_over_nyquist"] for r in rr]))
        out[(scene, rms)] = d
    return out


def fourier_threshold(rows: list[dict], est: str = "fourier_tuned") -> dict:
    """傅里叶链成功率随 lobe_freq_max 的变化：分箱，找 50% 失效点。"""
    m = [r for r in rows if r["scene"] == "matched"]
    f = np.array([r["lobe_freq_max"] for r in m])
    ok = np.array([np.isfinite(r[f"{est}_wf"]) and r[f"{est}_wf"] <= max(0.01, 5 * r["crb_wf"]) for r in m])
    edges = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 25, 50])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "success": float(ok[sel].mean())})
    thr = None
    for b in bins:
        if b["success"] < 0.5:
            thr = b["lo"]
            break
    return {"bins": bins, "first_bin_below_half": thr}


def figure_large(rows: list[dict], path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    show = ("fourier_tuned", "gn_from_fourier", "lbfgs_from_fourier", "gn_from_zero", "mle_multistart")
    for ax, scene in zip(axes, ("matched", "realistic")):
        rr = [r for r in rows if r["scene"] == scene]
        for i, e in enumerate(show):
            x = [r["lobe_freq_max"] for r in rr]
            y = [r[f"{e}_wf"] if np.isfinite(r[f"{e}_wf"]) else 3.0 for r in rr]
            ax.scatter(x, y, s=10, color=f"C{i}", label=e, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("傅里叶瓣最大局部频率 ½|∇ΔW|（cyc/unit）")
        ax.set_title(scene)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("波前 RMS 误差（waves）")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_bias(tab: dict, metric: str) -> str:
    body = []
    for (f, lv), d in sorted(tab.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        body.append([f, fmt(lv), fmt(d["fourier"]), fmt(d["mle"]), fmt(d["mle_pix"]), fmt(d["mle_amp"]),
                     fmt(d["pred_fourier"]), fmt(d["pred_mle"])])
    return md_table(["因素", "幅度", "傅里叶", "MLE", "MLE+像素", "MLE+振幅", "傅里叶(预测)", "MLE(预测)"], body)
