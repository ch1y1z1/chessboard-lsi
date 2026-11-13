"""Read saved FT-region results; audit metrics and generate publication figures locally.

No propagation, calibration, optimization, or changes to original run results.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from lsi.basis import basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--font", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(args.font)
    font = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({
        "font.family": font, "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "figure.facecolor": "#fbfaf7",
        "axes.facecolor": "#fbfaf7", "text.color": "#243343",
        "axes.labelcolor": "#243343", "xtick.color": "#405061",
        "ytick.color": "#405061", "axes.edgecolor": "#bbc1c6",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#dedfdc", "grid.alpha": .7,
        "legend.frameon": False, "savefig.facecolor": "#fbfaf7",
        "axes.unicode_minus": False, "pdf.fonttype": 42,
    })
    colors = {"single": "#315e87", "multi64": "#b36644"}
    labels = {"single": "单源", "multi64": "64 源"}
    raw = json.loads((args.source / "sweep_results-snapshot.json").read_text())
    rows = [r for r in raw if not r.get("error")]
    errors = [r for r in raw if r.get("error")]
    index = {(r["kind"], r["group"], r["name"], r["condition"]): r for r in rows}
    conditions = ["clean"] + [f"40db_seed{s}" for s in range(41, 46)] + ["30db_seed42"]
    truth_seeds = [1807, 1808, 1809] + list(range(2001, 2011))
    conditions += [f"truth{s}" for s in truth_seeds]
    configs = list(dict.fromkeys((r["group"], r["name"]) for r in rows))
    expected = {(k, g, n, c) for k in colors for g, n in configs for c in conditions}
    assert len(configs) == 50 and len(index) == len(rows) == 2000
    assert set(index) == expected
    assert all(r["cal_rank"] == r["basis_rank"] == 35 for r in rows)
    assert all(np.isfinite(r["raw_rms_waves"]) for r in rows)
    resolved_errors = all(all((r["kind"], r["group"], r["name"], c) in index
                              for c in conditions) for r in errors)
    assert resolved_errors

    def row(k, g, n, c="clean"):
        return index[k, g, n, c]

    def value(k, g, n, c="clean"):
        return row(k, g, n, c)["raw_rms_waves"]

    def noises(k, g, n):
        return np.array([value(k, g, n, f"40db_seed{s}") for s in range(41, 46)])

    def truths(k, g, n):
        return np.array([value(k, g, n, f"truth{s}") for s in truth_seeds])

    def baseline(k, c="clean"):
        return value(k, "E0_baseline", "baseline", c)

    def finish(fig, name, title, subtitle):
        fig.suptitle(title, x=.055, y=.988, ha="left", fontsize=19, fontweight="bold")
        fig.text(.055, .925, subtitle, ha="left", fontsize=10.5, color="#637282")
        fig.savefig(figs / f"{name}.png", dpi=190, bbox_inches="tight")
        fig.savefig(figs / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = []
    for k in colors:
        for g, n in configs:
            r = row(k, g, n)
            nz = noises(k, g, n)
            ts = truths(k, g, n)
            ratios = ts / truths(k, "E0_baseline", "baseline")
            summary.append({
                "kind": k, "group": g, "name": n,
                "clean_rms_waves": r["raw_rms_waves"],
                "noise40_mean": nz.mean(), "noise40_sample_sd": nz.std(ddof=1),
                "noise30_single": value(k, g, n, "30db_seed42"),
                "clean_ratio": r["raw_rms_waves"] / baseline(k),
                "truth13_mean": ts.mean(), "truth13_median": np.median(ts),
                "truth13_ratio_mean": ratios.mean(),
                "truth13_ratio_min": ratios.min(), "truth13_ratio_max": ratios.max(),
                "truth13_improved_count": int((ratios < 1 - 1e-12).sum()),
                "cal_condition": r["cal_condition"],
                "effective_window_pixels": r["window_metrics"][0]["equivalent_noise_bins"],
                "window_weight": r["window_metrics"][0]["weight_sum"],
            })
    with (out / "configuration-summary.csv").open("w") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)

    # Independent reconstruction on the exact native pupil frequencies.
    c = json.loads((args.source / "config.json").read_text())
    dx_obj = c["super_lattice_period"] / c["N_lattice"]
    n_obj = c["N_lattice"] * c["N_super_lattice"]
    step = c["wavelength"] / (n_obj * dx_obj * c["NA_out"])
    extent = int(np.ceil(1 / step))
    freq = np.fft.fftshift(np.fft.fftfreq(n_obj, dx_obj))
    freq = freq[n_obj // 2 - extent:n_obj // 2 + extent + 1]
    xx, yy = np.meshgrid(freq, freq)
    rho = c["wavelength"] * np.hypot(xx, yy) / c["NA_out"]
    xy = freq * c["wavelength"] / c["NA_out"]
    epsilon = c["NA_in"] / c["NA_out"]
    pupil = (rho >= epsilon) & (rho <= 1)
    z = basis(tuple(range(1, 36)), rho, np.arctan2(yy, xx), epsilon, 5)
    assert pupil.sum() == 52428
    zv = z[:, pupil]
    coeff = np.array([np.array(r["coefficients"]) - r["truth"] for r in rows])
    recomputed = np.sqrt(np.mean((coeff @ zv) ** 2, axis=1))
    differences = abs(recomputed - [r["raw_rms_waves"] for r in rows])
    # Basis orthogonalization is sensitive to host NumPy/BLAS arithmetic.
    # Preserve the observed discrepancy instead of claiming bitwise parity.
    assert np.allclose(recomputed, [r["raw_rms_waves"] for r in rows],
                       rtol=1e-8, atol=1e-12)
    run08 = json.loads((args.source / "run08-summary.json").read_text())["results"]
    parity = []
    for k in colors:
        for cond in conditions[:7]:
            old = next(r for r in run08 if r["na_out"] == .02 and
                       r["source_kind"] == k and r["method"] == "ft" and r["condition"] == cond)
            parity.append(abs(old["raw_rms_waves"] - baseline(k, cond)))
    assert max(parity) < 1e-12
    audit = {
        "success_rows": len(rows), "unique_keys": len(index), "configurations": 50,
        "source_kinds": 2, "conditions_per_configuration": 20,
        "historical_errors": len(errors), "all_historical_errors_superseded": resolved_errors,
        "missing_keys": 0, "duplicate_success_keys": 0,
        "native_pupil_pixels": int(pupil.sum()),
        "independently_recomputed_rms_rows": len(rows),
        "max_rms_reconstruction_difference": float(differences.max()),
        "max_rms_reconstruction_relative_difference": float(
            np.max(differences / np.maximum(recomputed, 1e-300))),
        "reconstruction_tolerance": {"rtol": 1e-8, "atol_waves": 1e-12},
        "run08_parity_rows": 14, "run08_max_rms_difference": max(parity),
        "all_calibration_ranks": sorted({r["cal_rank"] for r in rows}),
        "all_basis_ranks": sorted({r["basis_rank"] for r in rows}),
        "scope": "saved-coefficient metric verification; no independent optical model",
    }

    # 01: experimental path and the distinction between fitting and evaluation.
    fig = plt.figure(figsize=(13, 5.8))
    ax = fig.add_axes([.04, .08, .92, .77])
    ax.axis("off")
    nodes = [
        ("固定观测", "同一系统 NA_out = 0.020\n单源 / 64 源\n复用 run-08 图像"),
        ("改变选区", "频域窗宽、位置、形状\n空间交集掩膜与平滑\n每次单项改变"),
        ("逐配置重新标定", "35 模式 × 正负幅度\n独立零波前复参考\n再拟合 35 项系数"),
        ("统一评价", "完整光瞳 raw RMS\n5 个噪声种子\n13 个独立真值"),
    ]
    for i, (title, txt) in enumerate(nodes):
        x = .01 + .252 * i
        ax.add_patch(FancyBboxPatch((x, .35), .225, .53,
                                    boxstyle="round,pad=.012", fc="#edf1f4", ec="#d3dce4"))
        ax.text(x + .112, .76, title, ha="center", fontsize=14, weight="bold")
        ax.text(x + .112, .56, txt, ha="center", va="center", fontsize=11, linespacing=1.8)
        if i < 3:
            ax.annotate("", (x + .252, .62), (x + .228, .62),
                        arrowprops=dict(arrowstyle="->", color="#526b80", lw=2))
    ax.text(.02, .15, "空间掩膜腐蚀：改变拟合使用的像素；最终仍在完整 52,428 像素光瞳评价。",
            fontsize=12, color="#315e87")
    ax.text(.02, .02, "本轮重写复用保存结果，不重新传播、标定或启动长实验。", fontsize=11, color="#657480")
    finish(fig, "01_research_route", "这次实验究竟比较了什么", "研究主线：同一观测、逐配置重标定、同一完整光瞳评价。")

    # 02: width response, noise means, and calibration conditioning.
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.8))
    fig.subplots_adjust(top=.80, bottom=.17, wspace=.33)
    widths = [.10, .15, .20, .25, .30, .35, .40, .45, .50]
    for k, ax in zip(colors, axs[:2]):
        names = [("E0_baseline", "baseline") if w == .4 else
                 ("E1_width", f"hw{w:.2f}") for w in widths]
        y = [value(k, g, n) * 1e4 for g, n in names]
        ax.plot(widths, y, "-o", color=colors[k], label="clean", lw=2.3, ms=5)
        means = [noises(k, g, n).mean() * 1e4 for g, n in names]
        sd = [noises(k, g, n).std(ddof=1) * 1e4 for g, n in names]
        ax.errorbar(widths, means, sd, fmt="s", mfc="none", ms=7,
                    color="#7f9690", label="40 dB 均值 ± 样本 SD", capsize=3)
        ax.axvline(.4, color="#8996a3", ls="--", lw=1)
        ax.set(title=labels[k], xlabel="名义半宽 / 基频", ylabel="完整光瞳 RMS（×10⁻⁴ waves）")
        ax.grid(axis="y")
        ax.legend(fontsize=8.5)
    for k in colors:
        vals = [row(k, "E0_baseline" if w == .4 else "E1_width",
                    "baseline" if w == .4 else f"hw{w:.2f}")["cal_condition"] for w in widths]
        axs[2].plot(widths, vals, "-o", color=colors[k], label=labels[k])
    axs[2].set(yscale="log", xlabel="名义半宽 / 基频", ylabel="标定响应矩阵条件数",
               title="过窄窗口的条件数升高")
    axs[2].grid(axis="y")
    axs[2].legend()
    finish(fig, "02_width_response", "窗宽影响明显，曲线并不单调",
           "主真值 clean 与五种子均值几乎重合；0.45 的主真值低点不能直接当作通用最优窗宽。")

    # 03: 13 truths in paired ratios; all truths displayed individually.
    selected = [("E1_width", "hw0.10", "半宽 0.10"), ("E1_width", "hw0.25", "半宽 0.25"),
                ("E1_width", "hw0.45", "半宽 0.45"), ("E3_shape", "circle_s0", "等面积圆窗"),
                ("E4_erosion", "er12", "掩膜腐蚀 12 px"), ("E4_phase_sigma", "ps20", "平滑 σ = 20"),
                ("E4_amplitude", "thr0.5", "幅度阈值 0.5")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    fig.subplots_adjust(top=.80, bottom=.13, left=.19, wspace=.11)
    for k, ax in zip(colors, axs):
        a = np.array([truths(k, g, n) / truths(k, "E0_baseline", "baseline")
                      for g, n, label in selected])
        im = ax.imshow(a, vmin=.7, vmax=1.3, cmap="RdBu_r", aspect="auto")
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.7, color="white" if abs(a[i,j] - 1) > .18 else "#263440")
        ax.set_xticks(range(13), [str(s) for s in truth_seeds], rotation=65, fontsize=8)
        ax.set_yticks(range(len(selected)), [x[2] for x in selected])
        ax.set_title(labels[k])
        ax.set_xlabel("独立真值的随机种子（clean）")
    fig.colorbar(im, ax=axs, fraction=.025, pad=.025, label="RMS / 同一真值的基线 RMS；1 = 不变")
    finish(fig, "03_truth_generalization", "主真值上的收益，换一个波前还成立吗",
           "每格与同一独立真值的默认窗比较；蓝色改善、红色变差。13 个真值均已参与本次分析，不能再称为未见测试集。")

    # 04: small offsets and the actual quantized fractional-window response.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.subplots_adjust(top=.79, bottom=.19, wspace=.28)
    for k in colors:
        x = np.arange(-3, 4)
        y = [value(k, "E2_center_x", f"ox{v:+d}") / baseline(k) for v in x]
        axs[0].plot(x, y, "-o", color=colors[k], label=labels[k], lw=2)
        sx = np.array([-1.5, -.5, .5, 1.5])
        sy = [value(k, "E2_center_subpx", f"sx{v:+.1f}") / baseline(k) for v in sx]
        axs[1].plot(sx, sy, "-o", color=colors[k], label=labels[k], lw=2)
    for ax in axs:
        ax.axhline(1, ls="--", color="#8a96a0", lw=1)
        ax.grid(axis="y")
        ax.set_ylabel("clean RMS / 基线 RMS")
        ax.legend()
    axs[0].set(xlabel="正 x 边带的径向窗偏移（bin）", title="可解释的小范围共窗重标定实验")
    axs[1].set(xlabel="名义分数偏移（bin）", title="实现退化：四个分数窗对应整数窗")
    axs[1].set_xticks([-1.5, -.5, .5, 1.5], ["−1.5\n→ −1", "−0.5\n→ 0", "+0.5\n→ +1", "+1.5\n→ +2"])
    finish(fig, "04_center_audit", "中心偏移有方向性，但现有实验覆盖有限",
           "只扫描 ±3 bin（约基频的 ±1.06%），且测量、参考和标定共窗；尚未完成测量与标定的中心失配实验。")

    # 05: erosion masks in camera coordinates, and full-pupil accuracy.
    geom = json.loads((args.source / "geometry.json").read_text())
    radius = geom["camera_pupil_radius_m"] / c["dx_cam"]
    sh = 54
    v = np.arange(-600, 601)
    mx, my = np.meshgrid(v, v)
    camera_rho = np.hypot(mx, my) / radius
    cp = (camera_rho >= epsilon) & (camera_rho <= 1)
    mask = cp & np.roll(cp, sh, axis=1)
    eroded = ndimage.binary_erosion(mask, iterations=12)
    audit["camera_mask_pixels_per_axis_baseline"] = int(mask.sum())
    audit["camera_mask_pixels_per_axis_er12"] = int(eroded.sum())
    audit["camera_mask_retained_fraction_er12"] = float(eroded.sum() / mask.sum())
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, left=.07, right=.97, top=.80, bottom=.16, wspace=.3)
    ax = fig.add_subplot(gs[0])
    mask_img = np.where(eroded, 2, np.where(mask, 1, np.nan))
    from matplotlib.colors import ListedColormap
    ax.imshow(mask_img, origin="lower", extent=[-600, 600, -600, 600],
              cmap=ListedColormap(["#dfb39a", "#7594aa"]), vmin=1, vmax=2)
    ax.set(xlabel="相机 x（px）", ylabel="相机 y（px）", title="x 剪切交集：棕色为腐蚀移除区域")
    ax = fig.add_subplot(gs[1])
    for k in colors:
        for mode, ls, marker in [("clean", "-", "o"), ("noise", "--", "s")]:
            vals = []
            for er in [0, 2, 4, 8, 12]:
                g, n = ("E0_baseline", "baseline") if er == 0 else ("E4_erosion", f"er{er}")
                vals.append((value(k, g, n) if mode == "clean" else noises(k, g, n).mean()) * 1e4)
            ax.plot([0, 2, 4, 8, 12], vals, ls + marker, color=colors[k],
                    label=f"{labels[k]} · {'clean' if mode == 'clean' else '40 dB 均值'}",
                    mfc=colors[k] if mode == "clean" else "none")
    ax.set(xlabel="交集掩膜腐蚀次数（4 邻域）", ylabel="完整光瞳 RMS（×10⁻⁴ waves）",
           title="腐蚀用于拟合，评价光瞳保持完整")
    ax.grid(axis="y")
    ax.legend(fontsize=9)
    finish(fig, "05_mask_erosion", "移除交集边界后，完整光瞳误差下降",
           f"er12 保留约 {100*eroded.sum()/mask.sum():.1f}% 拟合像素；腐蚀同时触及外缘、内遮拦及移位光瞳交集边界，不能单独归因于外缘。")

    # 06: equal-area shape comparison and effective bandwidth.
    names = ["rect_s0", "rect_s5", "circle_s0", "ellipse_s0", "gaussian_s0"]
    titles = ["硬矩形", "软矩形 σ=5", "等面积圆", "等面积椭圆", "高斯"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(top=.80, bottom=.20, wspace=.27)
    y = np.arange(5)
    for j, k in enumerate(colors):
        vals = [100 * (value(k, "E3_shape", n) / baseline(k) - 1) for n in names]
        axs[0].scatter(vals, y + (j - .5) * .15, s=65, color=colors[k], label=labels[k], zorder=3)
    axs[0].set_yticks(y, titles)
    axs[0].invert_yaxis()
    axs[0].axvline(0, color="#8a96a0", ls="--", lw=1)
    axs[0].set(xlabel="clean RMS 相对默认软矩形的变化（%）", title="主真值：圆窗改善约 8%—9%")
    axs[0].grid(axis="x")
    axs[0].legend()
    effective = [row("single", "E3_shape", n)["window_metrics"][0]["equivalent_noise_bins"] for n in names]
    axs[1].barh(y, np.array(effective) / 1e4, color=["#7594aa"] * 4 + ["#c78a67"])
    axs[1].set_yticks(y, titles)
    axs[1].invert_yaxis()
    axs[1].set(xlabel="有效噪声像素 N_eff（×10⁴ bin²）", title="等面积不等于等有效带宽")
    for i, v in enumerate(effective):
        axs[1].text(v / 1e4 + .1, i, f"{v:,.0f}", va="center", fontsize=9)
    axs[1].set_xlim(0, 11.6)
    finish(fig, "06_window_shapes", "窗形比较必须连同带宽一起看",
           "N_eff = (Σw)² / Σw²，为原代码保存的 equivalent_noise_bins；圆窗已按连续等面积构造，高斯有效噪声面积约为矩形两倍。")

    # 07: spatial error maps, exact same native pupil and fixed color scale.
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 10))
    fig.subplots_adjust(top=.84, bottom=.07, left=.07, right=.85, hspace=.27, wspace=.13)
    scale = max(np.max(abs(np.tensordot(np.array(row(k, g, n)["coefficients"]) -
                                       row(k, g, n)["truth"], z, axes=1)))
                for k in colors for g, n in [("E0_baseline", "baseline"), ("E4_erosion", "er12")]) * 1e4
    for i, k in enumerate(colors):
        for j, (g, n, label) in enumerate([("E0_baseline", "baseline", "默认"), ("E4_erosion", "er12", "腐蚀 12 px")]):
            r = row(k, g, n)
            e = np.tensordot(np.array(r["coefficients"]) - r["truth"], z, axes=1)
            im = axs[i, j].imshow(np.where(pupil, e * 1e4, np.nan), origin="lower",
                                  extent=[xy[0], xy[-1], xy[0], xy[-1]], cmap="RdBu_r",
                                  vmin=-scale, vmax=scale)
            axs[i, j].set_title(f"{labels[k]} · {label} · RMS={r['raw_rms_waves']*1e4:.3f}")
            axs[i, j].set(xlabel="归一化光瞳 x", ylabel="归一化光瞳 y")
            axs[i, j].set_xticks([-1, 0, 1])
            axs[i, j].set_yticks([-1, 0, 1])
    fig.colorbar(im, cax=fig.add_axes([.89, .17, .02, .59]), label="有符号波前误差（×10⁻⁴ waves）")
    finish(fig, "07_wavefront_error", "误差空间图：比较同一完整光瞳上的变化",
           "由保存的 35 项系数在原生光瞳网格重建；四图共用色标。数值复核通过不等于识别了误差的物理根因。")

    # 08: distinguish deterministic selection effects from five-seed noise variation.
    groups = [("E1_width", "窗宽"), ("E2_center_x", "x 中心偏移"),
              ("E3_shape", "窗形"), ("E4_erosion", "掩膜腐蚀"),
              ("E4_amplitude", "幅度阈值"), ("E4_phase_sigma", "相位平滑")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(top=.80, bottom=.19, left=.15, wspace=.17)
    spread_summary = {}
    for k, ax in zip(colors, axs):
        sd = noises(k, "E0_baseline", "baseline").std(ddof=1)
        records = []
        for i, (g, label) in enumerate(groups):
            ns = [n for gg, n in configs if gg == g]
            v = [value(k, g, n) for n in ns] + [baseline(k)]
            ratio = (max(v) - min(v)) / sd
            ax.barh(i, ratio, color=colors[k], alpha=.85)
            ax.text(ratio * 1.13, i, f"{ratio:,.0f}×", va="center", fontsize=10)
            records.append(dict(group=g, clean_range=max(v)-min(v),
                                range_over_baseline_noise_sd=ratio, max_over_min=max(v)/min(v)))
        spread_summary[k] = records
        ax.set_xscale("log")
        ax.set_xlim(1, 15000)
        ax.set_yticks(range(len(groups)), [x[1] for x in groups])
        ax.invert_yaxis()
        ax.set_title(f"{labels[k]}：基线 SD = {sd:.2e} waves")
        ax.set_xlabel("选区 clean 极差 / 基线五种子 RMS 的样本 SD")
        ax.grid(axis="x")
    finish(fig, "08_selection_vs_noise", "变化小，不代表低于噪声波动",
           "这里比较的是固定主真值、当前扫描范围内的参数效应；极差与样本 SD 的比值用于描述量级，不是显著性检验。")

    # 09: amplitude selection implementation and smoothing.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(top=.80, bottom=.19, wspace=.28)
    for k in colors:
        vals = [baseline(k)] + [value(k, "E4_amplitude", f"thr{t}") for t in [.1, .3, .5]]
        axs[0].plot([0, .1, .3, .5], np.array(vals) * 1e4, "-o", color=colors[k], label=labels[k])
        vals = [value(k, "E4_phase_sigma", "ps0"), value(k, "E4_phase_sigma", "ps5"),
                baseline(k), value(k, "E4_phase_sigma", "ps20")]
        axs[1].plot([0, 5, 10, 20], np.array(vals) * 1e4, "-o", color=colors[k], label=labels[k])
    for ax in axs:
        ax.set_ylabel("clean 完整光瞳 RMS（×10⁻⁴ waves）")
        ax.grid(axis="y")
        ax.legend()
    axs[0].set(title="当前幅度阈值实现：误差明显增加", xlabel="相对最大幅度的阈值")
    axs[1].set(title="相位平滑：较大 σ 有小幅收益", xlabel="高斯平滑 σ（相机 px）")
    finish(fig, "09_quality_controls", "幅度阈值与平滑，结论需要限定实现",
           "阈值代码只截取已有伪逆的列，没有对筛后设计矩阵重算伪逆；平滑则在掩膜外补零后进行。尚未做正确加权拟合与掩膜内归一化平滑。")

    # 10: same-observation LM reference and gross 2f failure.
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(top=.79, bottom=.23, wspace=.3)
    lm_stats = {}
    for j, k in enumerate(colors):
        nz = np.array([r["raw_rms_waves"] for r in run08
                       if r["na_out"] == .02 and r["source_kind"] == k and r["method"] == "lm"
                       and r["condition"].startswith("40db")])
        lm_stats[k] = {"mean": float(nz.mean()), "sample_sd": float(nz.std(ddof=1))}
        vals = [noises(k, "E0_baseline", "baseline").mean(),
                noises(k, "E4_erosion", "er12").mean(), nz.mean()]
        sds = [noises(k, "E0_baseline", "baseline").std(ddof=1),
               noises(k, "E4_erosion", "er12").std(ddof=1), nz.std(ddof=1)]
        axs[0].errorbar(np.arange(3) + (j - .5) * .14, np.array(vals) * 1e4,
                        np.array(sds) * 1e4, fmt="o-", color=colors[k], label=labels[k], capsize=4)
        ratios = [1, value(k, "E5c_sideband", "sideband_2f") / baseline(k)]
        axs[1].plot([0, 1], ratios, "-o", color=colors[k], label=labels[k])
        axs[1].text(1.025, ratios[1], f"{ratios[1]:.1f}×", color=colors[k], va="center")
    axs[0].set_xticks(range(3), ["默认 FT", "er12 FT", "原 run-08 LM"])
    axs[0].set(ylabel="40 dB RMS（×10⁻⁴ waves）", title="同观测的五种子均值 ± 样本 SD")
    axs[1].set_xticks([0, 1], ["基频 f", "误选 2f"])
    axs[1].set(yscale="log", ylabel="clean RMS / 默认 FT RMS", title="错误边带的整体流程失配")
    axs[1].set_xlim(-.1, 1.35)
    for ax in axs:
        ax.grid(axis="y")
        ax.legend()
    finish(fig, "10_lm_and_sideband", "选区改善后，主真值上的 FT–LM 差距仍在",
           "LM 复用 run-08，未重跑；2f 组沿用原相邻级次差分基和掩膜，不能据此断言任何正确建模的 2f 方法都失败。")

    audit["window_groups"] = spread_summary
    audit["lm40_reference"] = lm_stats
    audit["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.source / "sweep_results-snapshot.json",
                                       args.source / "run08-summary.json",
                                       args.source / "config.json",
                                       args.source / "geometry.json"]}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (out / "successful-results.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    (out / "historical-errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
