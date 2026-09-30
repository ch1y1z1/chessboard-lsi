"""可辨识性谱系：帧设计 -> LM 反演解的唯一性（experiment.py 第 5 节的系统化）。

对每个帧设计：

    (a) 对称性度量 max|I(c*) - I(-c*)|（采样像素支撑上，逐帧取最大）
        —— 0 表示 ±W 全局简并对该帧组仍然精确成立
    (b) 固定随机方向初值（10 方向 x RMS{0.03,0.1,0.3} + 零初值，种子 42）
        跑 LM 拟合，收敛点分类为 +W / -W / 其他极小(按 1e-3 聚类) / 不收敛

输出（output/）：
    identifiability_summary.csv  设计 x 类别占比 + 对称性度量
    identifiability_runs.csv     逐次运行明细（初值与终值系数）
    identifiability_minima.csv   各设计"其他极小"的代表性系数向量
    identifiability.png          类别占比堆叠条形图 + 对称性度量

运行：  uv run python analysis/identifiability.py
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lsi.lm import fit_wavefront_from_carrier_frame, fit_wavefront_from_frames  # noqa: E402

from common import (  # noqa: E402
    CFG_SHIFT,
    INDICES,
    SAMPLES,
    TRUTH_C,
    TRUTH_IDX,
    build_designs,
    classify_convergence,
    cluster_points,
    design_symmetry_metric,
    initial_guesses,
    truth_vector,
)
from lsi.model import ForwardModel, zernike_wavefront  # noqa: E402

OUT = Path("output")
OUT.mkdir(exist_ok=True)

CLASSES = ("+W", "-W", "other", "nonconv")

DESCRIPTIONS = {
    "raw1": "single frame, no modulation",
    "dx_1/8": "single frame, x phase shift t=1/8",
    "dx_1/4": "single frame, x phase shift t=1/4",
    "dy_1/8": "single frame, y phase shift t=1/8",
    "dx+dy_1/8": "two frames, dx(1/8)+dy(1/8)",
    "dx4": "4 frames, dx(t=i/4), i=0..3",
    "dx8": "8 frames, dx(t=i/8), i=0..7",
    "dx8+dy8": "16 frames, dx+dy(t=i/8)",
    "carrier_p30": "single carrier frame (p=30um, f0=1/2s)",
}


def run_design(label: str, design: dict, inits, c_star: np.ndarray):
    """一个帧设计：对称性度量 + 全部初值的 LM 分类。"""
    fm = design["forward"]
    sym = design_symmetry_metric(design, c_star, INDICES, SAMPLES)

    def fit(c0):
        if design["carrier"]:
            return fit_wavefront_from_carrier_frame(
                fm, INDICES, design["frames"][0], samples=SAMPLES, x0=c0
            )
        return fit_wavefront_from_frames(
            fm, INDICES, design["frames"], design["modulations"],
            samples=SAMPLES, x0=c0,
        )

    runs = []
    for init_rms, dir_id, c0 in inits:
        res = fit(c0)
        cls = classify_convergence(res.coeffs, res.converged, c_star)
        err_p = float(np.max(np.abs(res.coeffs - c_star)))
        err_m = float(np.max(np.abs(res.coeffs + c_star)))
        runs.append(
            dict(
                design=label, init_rms=init_rms, direction=dir_id,
                converged=bool(res.converged), n_iter=res.n_iter,
                rms_residual=res.rms_residual, cost=res.cost,
                err_plus=err_p, err_minus=err_m, cls=cls,
                c0=c0, c_final=res.coeffs,
            )
        )
        print(
            f"    {label:>10s} rms={init_rms:4.2f} dir={dir_id:3d} "
            f"it={res.n_iter:3d} conv={int(res.converged)} "
            f"r={res.rms_residual:9.3e} e+={err_p:8.2e} e-={err_m:8.2e} {cls}",
            flush=True,
        )
    return sym, runs


def summarize(label: str, sym: float, n_frames: int, runs):
    """设计级汇总：类别占比 + 其他极小的聚类。"""
    multi = [r for r in runs if r["init_rms"] > 0]  # 只统计 30 个随机初值
    cnt = Counter(r["cls"] for r in multi)
    share = {k: cnt.get(k, 0) / len(multi) for k in CLASSES}

    # 其他极小按终值系数的 max 距离 < 1e-4 贪心聚类
    others = [r for r in multi if r["cls"] == "other"]
    clusters = {
        i: [others[k] for k in members]
        for i, (_, members) in enumerate(
            cluster_points([r["c_final"] for r in others])
        )
    }

    row = dict(
        design=label,
        description=DESCRIPTIONS[label],
        n_frames=n_frames,
        symmetry_max_dI=sym,
        n_runs=len(multi),
        **{f"share_{k.replace('+', 'p').replace('-', 'm')}": share[k]
           for k in CLASSES},
        n_other_clusters=len(clusters),
        med_iter= float(np.median([r["n_iter"] for r in multi])),
    )
    return row, clusters


def symmetry_scan() -> None:
    """单帧相移量 t 扫描：max|I(c*)-I(-c*)|，验证 ± 简并的结构条件。

    I(c, δ) ≡ I(-c, -δ)（A_ab 全实）⇒ δ ≢ -δ (mod 2π) 时破缺，
    t=1/2（δ=±π）自逆而恢复简并——写 output/symmetry_scan.csv。
    """
    fm = ForwardModel(CFG_SHIFT)
    truth = zernike_wavefront(TRUTH_C, TRUTH_IDX)
    c_star = truth_vector()
    rows = []
    print("  单帧对称性扫描：max|I(c*)-I(-c*)|")
    for direction in ("x", "y"):
        for t in (0.0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 1.0):
            d = (fm.phase_shift_deltas(t, 0.0) if direction == "x"
                 else fm.phase_shift_deltas(0.0, t))
            design = dict(forward=fm,
                          frames=[fm.intensity(truth, deltas=d)],
                          modulations=[d])
            sym = design_symmetry_metric(design, c_star, INDICES, SAMPLES)
            rows.append((direction, t, sym))
            print(f"    d{direction}(t={t:.3f}): {sym:.3e}")
    with open(OUT / "symmetry_scan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "t", "symmetry_max_dI"])
        w.writerows(rows)
    print(f"  wrote {OUT}/symmetry_scan.csv")


def main() -> None:
    c_star = truth_vector()
    inits = initial_guesses()
    designs = build_designs()

    symmetry_scan()

    print(f"拟合 {INDICES[0]}..{INDICES[-1]}（{len(INDICES)} 项），"
          f"每帧采样 {SAMPLES} 像素；初值 {len(inits)} 个/设计"
          f"（含 1 个零初值作参照，不计入占比统计）")

    all_runs, summary_rows, cluster_map = [], [], {}
    for label, design in designs.items():
        t0 = time.time()
        sym, runs = run_design(label, design, inits, c_star)
        row, clusters = summarize(label, sym, len(design["frames"]), runs)
        summary_rows.append(row)
        cluster_map[label] = clusters
        all_runs.extend(runs)
        print(f"  {label:>10s}: sym={sym:.3e}  +W {row['share_pW']:.2f} "
              f"-W {row['share_mW']:.2f} other {row['share_other']:.2f} "
              f"nonconv {row['share_nonconv']:.2f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------ CSV
    hdr = ["design", "description", "n_frames", "symmetry_max_dI", "n_runs",
           "share_pW", "share_mW", "share_other", "share_nonconv",
           "n_other_clusters", "med_iter"]
    with open(OUT / "identifiability_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(summary_rows)

    zcols = [f"Z{j}" for j in INDICES]
    with open(OUT / "identifiability_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "init_rms", "direction", "converged", "n_iter",
                    "rms_residual", "cost", "err_plus", "err_minus", "class"]
                   + [f"init_{c}" for c in zcols]
                   + [f"final_{c}" for c in zcols])
        for r in all_runs:
            w.writerow([r["design"], r["init_rms"], r["direction"],
                        int(r["converged"]), r["n_iter"],
                        f"{r['rms_residual']:.6e}", f"{r['cost']:.6e}",
                        f"{r['err_plus']:.6e}", f"{r['err_minus']:.6e}",
                        r["cls"]]
                       + [f"{v:.8e}" for v in r["c0"]]
                       + [f"{v:.8e}" for v in r["c_final"]])

    with open(OUT / "identifiability_minima.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "cluster", "count", "med_rms_residual",
                    "med_n_iter"] + zcols)
        for label, clusters in cluster_map.items():
            reps = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
            for i, (_, members) in enumerate(reps):
                med = np.median([m["c_final"] for m in members], axis=0)
                w.writerow([label, i, len(members),
                            f"{np.median([m['rms_residual'] for m in members]):.6e}",
                            f"{np.median([m['n_iter'] for m in members]):.0f}"]
                           + [f"{v:.6f}" for v in med])

    # ------------------------------------------------------------------ PNG
    colors = {"+W": "#2ca02c", "-W": "#9467bd",
              "other": "#ff7f0e", "nonconv": "#d62728"}
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ypos = np.arange(len(summary_rows))
    left = np.zeros(len(summary_rows))
    for cls in CLASSES:
        key = f"share_{cls.replace('+', 'p').replace('-', 'm')}"
        vals = np.array([r[key] for r in summary_rows])
        ax.barh(ypos, vals, left=left, color=colors[cls], label=cls)
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{r['design']}\n({r['n_frames']}f, dI={r['symmetry_max_dI']:.2f})"
         for r in summary_rows],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of 30 random-direction initializations")
    ax.set_title("LM basin outcome vs frame design "
                 "(truth Z4..Z8, fit Z2..Z13, n=128)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "identifiability.png", dpi=150)

    print(f"\nwrote {OUT}/identifiability_summary.csv, "
          f"identifiability_runs.csv, identifiability_minima.csv, "
          f"identifiability.png")


if __name__ == "__main__":
    main()
