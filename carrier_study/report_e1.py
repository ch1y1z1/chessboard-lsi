"""E1 汇总：估计器的 NMSE（相对 CRB）、配对 MSE 比、灾难率、P1/P2 检验与图。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import FIG, cluster_bootstrap, setup_matplotlib, fmt, group, md_table, paired_ratio_ci
from .common import NOISES

NOISE_ORDER = [n.label for n in NOISES]
EST = ("mle", "mle_pix", "ls_unweighted", "fourier_tuned", "fourier_default")
RMS_GROUPS = {"≤0.1": (0.0, 0.1), "0.3–1": (0.3, 1.0), "2–3": (2.0, 3.0)}


def rms_group(rms: float) -> str:
    for k, (lo, hi) in RMS_GROUPS.items():
        if lo <= rms <= hi + 1e-9:
            return k
    return "?"


def catastrophic(r: dict) -> bool:
    wf = r["wf"]
    return (not np.isfinite(wf)) or (wf > 0.05 and wf > 10 * r["crb_wf"])


def nmse_table(rows: list[dict]) -> dict:
    """{(scene, noise, rms_group, estimator): (NMSE, lo, hi, n_truths, cat_rate, median_wf)}"""
    out = {}
    for (scene, noise, est), rr in group(rows, "scene", "noise", "estimator").items():
        for gname in RMS_GROUPS:
            sub = [r for r in rr if rms_group(r["rms"]) == gname]
            if not sub:
                continue
            ok = [r for r in sub if np.isfinite(r["d2"])]
            byc = {tid: [r["d2"] for r in g] for (tid,), g in group(ok, "tid").items()}
            if byc:
                pt, lo, hi = cluster_bootstrap(byc, np.mean, n_boot=1000)
            else:
                pt = lo = hi = np.nan
            out[(scene, noise, gname, est)] = {
                "nmse": pt, "lo": lo, "hi": hi, "n_truths": len({r["tid"] for r in sub}),
                "cat_rate": float(np.mean([catastrophic(r) for r in sub])),
                "median_wf": float(np.nanmedian([r["wf"] for r in sub])),
                "median_wf_notilt": float(np.nanmedian([r["wf_notilt"] for r in sub])),
                "median_crb_wf": float(np.median([r["crb_wf"] for r in sub])),
            }
    return out


def paired_mse(rows: list[dict], a: str, b: str, metric: str = "wf") -> dict:
    """配对 MSE 比 a/b（簇 = 真值），灾难性失败的 a 计入（误差封顶 1 wave）。"""
    out = {}
    for (scene, noise), rr in group(rows, "scene", "noise").items():
        for gname in RMS_GROUPS:
            A, B = {}, {}
            for r in rr:
                if rms_group(r["rms"]) != gname:
                    continue
                v = r[metric]
                v = 1.0 if not np.isfinite(v) else min(v, 1.0)
                if r["estimator"] == a:
                    A.setdefault(r["tid"], []).append(v**2)
                elif r["estimator"] == b:
                    B.setdefault(r["tid"], []).append(v**2)
            if A and B:
                out[(scene, noise, gname)] = paired_ratio_ci(A, B, n_boot=2000)
    return out


def p1_check(tab: dict) -> dict:
    """P1：matched、MLE，SNR >= 30 dB 及 Poisson >= 1e4，RMS <= 1：NMSE 在 1 ± 0.1。"""
    res = {}
    for noise in NOISE_ORDER:
        for g in ("≤0.1", "0.3–1"):
            v = tab.get(("matched", noise, g, "mle"))
            if v:
                res[f"{noise}|{g}"] = {"nmse": v["nmse"], "ci": (v["lo"], v["hi"]),
                                      "in_band": bool(0.9 <= v["nmse"] <= 1.1)}
    return res


def figure_nmse(tab: dict, path: Path) -> None:
    plt = setup_matplotlib()
    gauss = [n for n in NOISE_ORDER if n.startswith("G")]
    snr = [float(n[1:]) for n in gauss]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"mle": ("C0", "-", "MLE（直接光强最小二乘）"), "mle_pix": ("C2", "-.", "MLE + 像素积分建模"),
              "fourier_tuned": ("C3", "--", "傅里叶链（调优）")}
    for ax, scene in zip(axes, ("matched", "realistic")):
        for est, (col, ls, lab) in styles.items():
            for gi, g in enumerate(("≤0.1", "0.3–1")):
                ys = [tab.get((scene, n, g, est), {}).get("nmse", np.nan) for n in gauss]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot(snr, ys, color=col, ls=ls, marker="os"[gi], alpha=1.0 if gi == 0 else 0.6,
                        label=f"{lab}, RMS {g}")
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("峰值 SNR (dB)")
        ax.set_title({"matched": "matched（同模型）", "realistic": "realistic（像素积分）"}[scene])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NMSE = E[d_rel²]（1 = 达到 CRB）")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(tab: dict, ratios: dict) -> str:
    lines = []
    for scene in ("matched", "realistic"):
        hdr = ["噪声", "RMS"] + [e for e in EST]
        body = []
        for noise in NOISE_ORDER:
            for g in RMS_GROUPS:
                cells = []
                for e in EST:
                    v = tab.get((scene, noise, g, e))
                    cells.append("" if v is None else f"{fmt(v['nmse'])}" + (f"（灾难 {v['cat_rate']:.0%}）" if v["cat_rate"] > 0 else ""))
                if any(cells):
                    body.append([noise, g] + cells)
        lines.append(f"**{scene}：NMSE = E[d_rel²]（1 = CRB）**\n\n" + md_table(hdr, body))
        hdr = ["噪声", "RMS", "MSE 比 傅里叶(调优)/MLE", "95% CI", "真值数"]
        body = [[n, g, fmt(v[0]), f"[{fmt(v[1])}, {fmt(v[2])}]", v[3]]
                for (sc, n, g), v in sorted(ratios.items(), key=lambda kv: (NOISE_ORDER.index(kv[0][1]), kv[0][2]))
                if sc == scene]
        lines.append(md_table(hdr, body))
    return "\n\n".join(lines)
