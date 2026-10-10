"""M3/M4：汇总 E1–E4、核对 P1–P7（E5），生成表格、图与机器可读的结论。

运行：  uv run python -m carrier_study.m4_report [results_dir]
        results_dir 默认 results/carrier/m3（pilot 用 results/carrier/m2 时读取 *_pilot 文件）
输出：  reports/carrier/figures/*.png、results/carrier/m4/summary.json、reports/carrier/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import report_e1, report_e2, report_e3, report_e4
from .analysis import FIG, fmt, load_rows, md_table, write_json

PRED = Path("results/carrier/m1/predictions.json")


def _f(d: Path, name: str) -> Path | None:
    for cand in (d / f"{name}.csv.gz", d / f"{name}_pilot.csv.gz"):
        if cand.exists():
            return cand
    return None


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/carrier/m3")
    tag = "pilot" if d.name == "m2" else "main"
    fig = FIG / tag
    pred = json.loads(PRED.read_text())
    out: dict = {"tag": tag}
    md: list[str] = []

    # ---------------- E1
    p = _f(d, "e1")
    if p:
        rows = load_rows(p)
        tab = report_e1.nmse_table(rows)
        ratios = report_e1.paired_mse(rows, "fourier_tuned", "mle")
        ratios_pix = report_e1.paired_mse([r for r in rows if r["scene"] == "realistic"], "fourier_tuned", "mle_pix")
        report_e1.figure_nmse(tab, fig / "e1_nmse.png")
        out["e1"] = {"p1": report_e1.p1_check(tab),
                     "nmse": {"|".join(map(str, k)): v for k, v in tab.items()},
                     "ratio_fourier_over_mle": {"|".join(map(str, k)): v for k, v in ratios.items()},
                     "ratio_fourier_over_mle_pix": {"|".join(map(str, k)): v for k, v in ratios_pix.items()}}
        # P2：傅里叶链默认参数的 E[d²]（matched）实测中位 vs 预测中位（按 RMS 档）
        p2 = {}
        P35 = pred["p35"]["noise"]
        for (scene, noise, est), rr in report_e1.group(rows, "scene", "noise", "estimator").items():
            if scene != "matched" or est != "fourier_default":
                continue
            for rms_key, pv in P35[noise]["pred_d2_fourier_by_rms"].items():
                vals = [r["d2"] for r in rr if abs(r["rms"] - float(rms_key)) < 1e-9 and np.isfinite(r["d2"])]
                if vals:
                    p2[f"{noise}|{rms_key}"] = {"pred": pv, "meas_median": float(np.median(vals)),
                                                "ratio": float(np.median(vals) / pv)}
        out["e1"]["p2"] = p2
        md.append("## E1 估计器对比\n\n" + report_e1.markdown(tab, ratios))

    # ---------------- E3
    p = _f(d, "e3")
    if p:
        rows = load_rows(p)
        out["e3"] = {}
        for tau in (1.0, 0.3, 0.1, 0.03):
            tab = report_e3.cost_table(rows, tau)
            out["e3"][f"t{tau:g}"] = {"cost": {"|".join(map(str, k)): v for k, v in tab.items()},
                                     "p5": {f"p{pp}|{init}": report_e3.p5_check(tab, pp, init)
                                            for pp in (35, 12) for init in ("fourier", "zero")}}
            if tau == 0.1:
                for init in ("fourier", "zero"):
                    report_e3.figure_cost(tab, tau, init, fig / f"e3_cost_{init}.png")
                    for pp in (35, 12):
                        md.append(f"## E3 到达 d_rel ≤ {tau:g} 的中位成本（p={pp}，{init} 初值）\n\n"
                                  + report_e3.markdown_costs(tab, pp, init))
        sp = report_e3.speedup(rows, 0.1)
        out["e3"]["speedup_vs_gn_t0.1"] = {"|".join(map(str, k)): v for k, v in sp.items()}
        cfgs = [c for c in report_e3.ORDER if c not in ("adam:default", "nesterov_fixed:default", "gd_fixed:default")]
        for pp in (35, 12):
            prof = report_e3.profile(rows, 0.1, pp, "fourier", cfgs)
            out["e3"][f"profile_p{pp}"] = prof
            report_e3.figure_profile(prof, fig / f"e3_profile_p{pp}.png", f"Performance profile（p={pp}，τ=0.1，傅里叶初值）")

    pk = _f(d, "e3_kappa")
    if pk:
        out["e3_kappa"] = kappa_scaling(load_rows(pk), pred)

    # ---------------- E2
    p = _f(d, "e2")
    if p:
        rows = load_rows(p)
        tab = report_e2.success_table(rows)
        methods = [m for m in ("gn", "lm_minpack", "trf", "bfgs", "lbfgs", "nesterov", "adam") if any(k[0] == m for k in tab)]
        for pp in sorted({r["p"] for r in rows}):
            rr = [r for r in rows if r["p"] == pp]
            t = report_e2.success_table(rr)
            periods = sorted({k[1] for k in t})
            report_e2.figure(t, methods, periods, fig / f"e2_basin_p{int(pp)}.png")
            out.setdefault("e2", {})[f"p{int(pp)}"] = {
                "r50": {f"{m}|{per:g}|{mode}": report_e2.r50(t, m, per, mode)
                        for m in methods for per in periods for mode in ("perturbed", "random")},
                "special": {"|".join(map(str, k)): v for k, v in report_e2.special_inits(rr).items()},
                "fail_types": {"|".join(map(str, k)): v["fail_types"] for k, v in t.items()},
            }
            md.append(f"## E2 50% 成功半径 r50（p={int(pp)}）\n\n" + report_e2.markdown(t, rr, methods, periods))

    # ---------------- E4
    p = _f(d, "e4_mismatch")
    if p:
        rows = load_rows(p)
        bt = report_e4.bias_table(rows, "wf")
        btn = report_e4.bias_table(rows, "wf_notilt")
        nt = report_e4.noisy_table(rows)
        report_e4.figure_mismatch(bt, fig / "e4_mismatch_bias.png", "偏差（波前 RMS，waves）")
        report_e4.figure_mismatch(btn, fig / "e4_mismatch_bias_notilt.png", "偏差（去倾斜，waves）")
        out["e4_mismatch"] = {"bias": {"|".join(map(str, k)): v for k, v in bt.items()},
                              "bias_notilt": {"|".join(map(str, k)): v for k, v in btn.items()},
                              "noisy": {"|".join(map(str, k)): v for k, v in nt.items()},
                              "p3": report_e4.p3_check(rows)}
        md.append("## E4 失配偏差（无噪声，中位，waves）\n\n" + report_e4.markdown_bias(bt, "wf"))
        md.append("## E4 失配偏差（去倾斜）\n\n" + report_e4.markdown_bias(btn, "wf_notilt"))
    p = _f(d, "e4_large")
    if p:
        rows = load_rows(p)
        ls = report_e4.large_success(rows)
        report_e4.figure_large(rows, fig / "e4_large.png")
        out["e4_large"] = {"success": {"|".join(map(str, k)): v for k, v in ls.items()},
                           "p7_tuned": report_e4.fourier_threshold(rows, "fourier_tuned"),
                           "p7_default": report_e4.fourier_threshold(rows, "fourier_default")}
        body = [[sc, fmt(rms), fmt(v["lobe_freq_max_median"]), fmt(v["beat_over_nyquist_median"])]
                + [f"{v[e]:.0%}" for e in report_e4.LARGE_EST] for (sc, rms), v in sorted(ls.items())]
        md.append("## E4 大像差：各管线成功率（误差 ≤ max(0.01, 5×CRB)）\n\n"
                  + md_table(["场景", "RMS", "瓣频率", "拍频/奈奎斯特"] + list(report_e4.LARGE_EST), body))

    tagdir = Path("results/carrier/m4")
    write_json(tagdir / f"summary_{tag}.json", out)
    Path(f"reports/carrier/tables_{tag}.md").write_text("\n\n".join(md) + "\n")
    print(f"wrote summary_{tag}.json and tables_{tag}.md")


def kappa_scaling(rows: list[dict], pred: dict) -> dict:
    """P4：一阶方法到达 tau = 0.1 的迭代数（成本 / 每步成本）随 κ 的对数斜率。"""
    import csv
    kap = {}
    for fn in ("shear_kappa.csv", "shear_kappa_p12.csv"):
        for r in csv.DictReader(open(f"results/carrier/m1/{fn}")):
            kap.setdefault((int(r["p"]), float(r["period_um"])), []).append(float(r["kappa_JtJ"]))
    out = {}
    for cfg in sorted({r["config"] for r in rows}):
        xs, ys = [], []
        for (pp, per), rr in report_e3.group([r for r in rows if r["config"] == cfg], "p", "period_um").items():
            c = np.array([r["cost_t0.1"] for r in rr], float)
            if (int(pp), per) not in kap or not np.all(np.isfinite(c)):
                continue
            xs.append(np.log(np.median(kap[(int(pp), per)])))
            ys.append(np.log(np.median(c)))
        if len(xs) >= 3:
            slope = float(np.polyfit(xs, ys, 1)[0])
            out[cfg] = {"slope_logcost_logkappa": slope, "n_points": len(xs),
                        "kappas": [float(np.exp(x)) for x in xs], "costs": [float(np.exp(y)) for y in ys]}
    return out


if __name__ == "__main__":
    main()
