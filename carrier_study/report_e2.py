"""E2 汇总：进入全局吸引盆的比例 vs 初值距离、50% 成功半径 r50、P6（r50·s 为常数）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import fmt, group, md_table, setup_matplotlib


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def success_table(rows: list[dict]) -> dict:
    out = {}
    for (m, per, mode, r), rr in group([x for x in rows if x["mode"] in ("perturbed", "random")],
                                       "method", "period_um", "mode", "r").items():
        k = sum(x["basin"] == "global" for x in rr)
        lo, hi = wilson(k, len(rr))
        out[(m, per, mode, r)] = {"frac": k / len(rr), "lo": lo, "hi": hi, "n": len(rr),
                                  "fail_types": {t: sum(x["basin"] == t for x in rr) for t in ("sign", "tilt_alias", "other")}}
    return out


def r50(tab: dict, method: str, period: float, mode: str) -> float:
    rs = sorted({k[3] for k in tab if k[0] == method and k[1] == period and k[2] == mode})
    fr = [tab[(method, period, mode, r)]["frac"] for r in rs]
    for (r1, f1), (r2, f2) in zip(zip(rs, fr), zip(rs[1:], fr[1:])):
        if f1 >= 0.5 > f2:
            t = (f1 - 0.5) / (f1 - f2)
            return float(np.exp(np.log(r1) + t * (np.log(r2) - np.log(r1))))
    if fr and fr[-1] >= 0.5:
        return float("inf")
    return float(rs[0]) if rs else float("nan")


def special_inits(rows: list[dict]) -> dict:
    out = {}
    for (m, per, mode), rr in group([x for x in rows if x["mode"] in ("zero", "fourier")], "method", "period_um", "mode").items():
        out[(m, per, mode)] = (sum(x["basin"] == "global" for x in rr), len(rr))
    return out


def figure(tab: dict, methods, periods, path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, len(periods), figsize=(4.2 * len(periods), 4), sharey=True)
    for ax, per in zip(np.atleast_1d(axes), periods):
        for i, m in enumerate(methods):
            rs = sorted({k[3] for k in tab if k[0] == m and k[1] == per and k[2] == "perturbed"})
            ax.plot(rs, [tab[(m, per, "perturbed", r)]["frac"] for r in rs], marker="o", ms=3, label=m, color=f"C{i}",
                    ls="-" if m in ("gn", "lm_minpack", "trf") else "--")
        ax.set_xscale("log")
        ax.set_xlabel("初值距离 r（波前 RMS，waves）")
        ax.set_title(f"光栅周期 {per:g} µm")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("进入全局吸引盆的比例")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(tab: dict, rows: list[dict], methods, periods) -> str:
    s_of = {per: rows_s for per, rows_s in ((x["period_um"], x["s"]) for x in rows)}
    body = []
    for m in methods:
        cells = []
        for per in periods:
            for mode in ("perturbed", "random"):
                v = r50(tab, m, per, mode)
                cells.append(fmt(v))
        prods = [r50(tab, m, per, "perturbed") * s_of[per] for per in periods]
        cells.append(" / ".join(fmt(x) for x in prods))
        body.append([m] + cells)
    hdr = ["求解器"] + [f"{per:g} µm {md}" for per in periods for md in ("围绕真值", "随机")] + ["r50·s（围绕真值）"]
    return md_table(hdr, body)
