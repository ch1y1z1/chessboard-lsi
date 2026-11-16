"""Redraw existing run-01 results for reading inside a Notion report.

No propagation, calibration, fitting or parameter selection is performed here.
All scalar data come from the frozen successful-results view. Only four saved
coefficient vectors are evaluated to recreate the existing spatial error maps.
"""
import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.text import Text
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lsi.basis import basis

BLUE, ORANGE, INK = "#2268A2", "#C26B32", "#233244"
COLORS = {"single": BLUE, "multi64": ORANGE}
LABELS = {"single": "单源", "multi64": "64 源"}
MARKERS = {"single": "o", "multi64": "D"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-output", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--font", type=Path, required=True)
    a = ap.parse_args()
    (a.output / "figures").mkdir(parents=True, exist_ok=True)
    (a.output / "previews").mkdir(exist_ok=True)
    font_manager.fontManager.addfont(a.font)
    font = font_manager.FontProperties(fname=a.font).get_name()
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"], "font.size": 12,
        "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#A4B0BB",
        "xtick.color": "#415162", "ytick.color": "#415162",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E4E9ED", "grid.linewidth": .8, "grid.alpha": 1,
        "legend.frameon": False, "legend.fontsize": 11,
        "axes.unicode_minus": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white", "lines.linewidth": 2,
    })
    raw = json.loads((a.original_output / "successful-results.json").read_text())
    idx = {(r["kind"], r["group"], r["name"], r["condition"]): r for r in raw}
    assert len(idx) == len(raw) == 2000
    truth_ids = [1807, 1808, 1809] + list(range(2001, 2011))
    summary = list(csv.DictReader((a.original_output / "configuration-summary.csv").open()))
    cfg = json.loads((a.source / "config.json").read_text())
    geom = json.loads((a.source / "geometry.json").read_text())
    old_audit = json.loads((a.original_output / "audit.json").read_text())
    run08 = json.loads((a.source / "run08-summary.json").read_text())["results"]

    def record(k, g="E0_baseline", n="baseline", c="clean"):
        return idx[k, g, n, c]

    def val(k, g="E0_baseline", n="baseline", c="clean"):
        return record(k, g, n, c)["raw_rms_waves"]

    def noise(k, g="E0_baseline", n="baseline"):
        return np.array([val(k, g, n, f"40db_seed{s}") for s in range(41, 46)])

    def ratios(k, g, n):
        return np.array([val(k, g, n, f"truth{s}") /
                         val(k, c=f"truth{s}") for s in truth_ids])

    # Cross-check the complete scalar source used by this revision.
    for r in summary:
        k, g, n = r["kind"], r["group"], r["name"]
        rr = ratios(k, g, n)
        checks = {
            "clean_rms_waves": val(k, g, n),
            "noise40_mean": noise(k, g, n).mean(),
            "noise40_sample_sd": noise(k, g, n).std(ddof=1),
            "truth13_ratio_mean": rr.mean(),
            "truth13_ratio_min": rr.min(), "truth13_ratio_max": rr.max(),
            "truth13_improved_count": int((rr < 1 - 1e-12).sum()),
        }
        assert all(np.isclose(float(r[q]), v, rtol=1e-12, atol=1e-15)
                   for q, v in checks.items())

    figures, layout_audit = [], []
    pdf = PdfPages(a.output / "figures-revised.pdf")

    def canvas(title, subtitle, ratio="16:9"):
        fig = plt.figure(figsize=(10, 5.625 if ratio == "16:9" else 7.5))
        fig.text(.075, .956, title, fontsize=18, weight="bold", ha="left", va="top")
        fig.text(.075, .882 if ratio == "16:9" else .902, subtitle,
                 fontsize=11.5, color="#5B6B79", ha="left", va="top")
        return fig

    def axis(fig, left=.14, right=.95, bottom=.20, top=.75):
        return fig.add_axes([left, bottom, right-left, top-bottom])

    def legend(fig, ax, y=.817, ncol=2, x=.14):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y),
                   borderaxespad=0, ncol=ncol, columnspacing=1.8, handletextpad=.6)

    def line(ax, k, x, y, **kwargs):
        ax.plot(x, y, marker=MARKERS[k], color=COLORS[k], label=LABELS[k],
                ms=6, **kwargs)

    def finish(fig, name, title, caption, guide, data, number=None):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = []
        for t in fig.findobj(Text):
            if t.get_visible() and t.get_text().strip():
                bb = t.get_window_extent(renderer)
                if bb.width > 0 and bb.height > 0:
                    texts.append((t, bb))
        clipped, overlaps = [], []
        bounds = fig.bbox
        for t, b in texts:
            if b.x0 < -.5 or b.y0 < -.5 or b.x1 > bounds.x1+.5 or b.y1 > bounds.y1+.5:
                clipped.append(t.get_text())
        for (t1, b1), (t2, b2) in combinations(texts, 2):
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if dx > 2 and dy > 2:
                overlaps.append([t1.get_text(), t2.get_text()])
        entry = {"figure": name, "clipped_text": clipped, "text_overlaps": overlaps}
        layout_audit.append(entry)
        # Fixed canvas: never use bbox_inches="tight", which changes aspect ratio.
        path = a.output / "figures" / (name + ".png")
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        pdf.savefig(fig)
        from PIL import Image
        with Image.open(path) as im:
            width, height = im.size
            im.resize((900, round(height*900/width)),
                      Image.Resampling.LANCZOS).save(a.output / "previews" / path.name)
        figures.append({
            "name": name, "number": number, "title": title, "caption": caption,
            "guide": guide, "data": data, "width": width, "height": height,
            "aspect_ratio": "16:9" if height == 1350 else "4:3",
            "png": "figures/" + path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        plt.close(fig)

    # 1. Method: short boxes, arrows in explicit gutters, no axes to decode.
    title = "改变拟合选区，始终评价同一个完整光瞳"
    fig = canvas(title, "run-01 的比较方式：同一系统、逐项改变、每个配置重新标定。")
    ax = fig.add_axes([.055, .11, .89, .65]); ax.set_axis_off()
    boxes = [
        ("01  固定观测", "NA_out = 0.020\n单源 / 64 源"),
        ("02  改变选区", "频谱窗：宽度、中心、形状\n空间区：交集边界、平滑"),
        ("03  重新标定", "35 模式的正负扰动\n拟合 35 项波前系数"),
        ("04  统一评价", "完整 52,428 像素光瞳\n无噪声 / 噪声 / 随机真值"),
    ]
    width = .222
    for i, (head, body) in enumerate(boxes):
        x = i * .259
        ax.add_patch(FancyBboxPatch((x, .40), width, .51,
                     boxstyle="round,pad=0.006,rounding_size=0.015",
                     fc="#EDF4F9" if i == 1 else "#F5F7F9", ec="#CBD7E0",
                     transform=ax.transAxes, clip_on=False))
        ax.text(x+width/2, .78, head, ha="center", va="center", fontsize=12,
                weight="bold", transform=ax.transAxes)
        ax.text(x+width/2, .59, body, ha="center", va="center", fontsize=10.5,
                linespacing=1.8, transform=ax.transAxes)
        if i < 3:
            ax.annotate("", xy=(x+.252, .65), xytext=(x+width+.006, .65),
                        xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": "#6D8395", "lw": 1.8})
    ax.text(.5, .23, "拟合区域可以变；最终用于计算 RMS 的光瞳区域不变。",
            ha="center", va="center", color=BLUE, weight="bold", fontsize=13,
            transform=ax.transAxes)
    ax.text(.5, .07, "因此，腐蚀后的改善不能解释成删掉了评价中的高误差像素。",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    finish(fig, "01_method", title,
           "图 1｜四步流程：固定观测 → 改变选区 → 重新标定 → 同一完整光瞳评价。",
           "按箭头从左向右读。第二步改变参与特征提取或拟合的选区，第四步始终使用同一个完整光瞳；本图是流程示意，没有数据坐标轴。",
           {"evaluation_pixels": 52428, "fitted_modes": 35}, 1)

    # 2. Width: one result axis; keep conditioning in a separate supplement.
    widths = [.10, .15, .20, .25, .30, .35, .40, .45, .50]
    width_configs = [("E0_baseline", "baseline") if w == .4 else
                     ("E1_width", f"hw{w:.2f}") for w in widths]
    title = "主真值误差随窗宽变化，曲线并不单调"
    fig = canvas(title, "主真值 · 无噪声（clean）· 每个窗口重新标定")
    ax = axis(fig); data = {"x_nominal_halfwidth_over_carrier": widths}
    for k in COLORS:
        yy = [val(k, g, n)*1e4 for g, n in width_configs]
        line(ax, k, widths, yy); data[k] = yy
    ax.axvline(.4, color="#6E7B88", ls="--", lw=1.2)
    ax.set(xlabel="名义窗口半宽 / 基频 f（无量纲）", ylabel="全光瞳 RMS（10⁻⁴ waves）",
           xlim=(.085, .515), ylim=(3, 7))
    ax.set_xticks([.1, .2, .3, .4, .5]); ax.grid(axis="y"); legend(fig, ax)
    ax.text(.405, 6.75, "默认 0.40f", fontsize=10.5, color="#5B6B79", va="top")
    finish(fig, "02_width", title, "图 2｜主真值的窗宽响应；虚线为默认名义半宽 0.40f。",
           "横轴是窗口半宽除以基频 f，数值越大表示取更宽的边带。f≈282.09 个 FFT 频点（bin）；0.40f 对应实际半宽 112 bin，约 0.397f。纵轴是同一完整光瞳上的波前误差 RMS，刻度 4 表示 4×10⁻⁴ waves，越低越好。蓝圆为单源，橙菱形为 64 源；这里只画无噪声结果，不把近乎重合的含噪曲线叠在一起。40 dB 五种子统计见附件 CSV。",
           data, 2)

    title = "过窄窗口使标定条件数升高"
    fig = canvas(title, "补充图 S1 · 与图 2 使用完全相同的配置")
    ax = axis(fig); data = {}
    for k in COLORS:
        yy = [record(k, g, n)["cal_condition"] for g, n in width_configs]
        line(ax, k, widths, yy); data[k] = yy
    ax.set(xlabel="名义窗口半宽 / 基频 f", ylabel="标定响应矩阵条件数（无量纲）",
           yscale="log", ylim=(10, 200))
    ax.set_xticks([.1, .2, .3, .4, .5])
    ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="y"); legend(fig, ax)
    finish(fig, "S1_condition", title, "补充图 S1｜窗口半宽与标定响应矩阵条件数。",
           "横轴同图 2。纵轴为标定响应矩阵条件数，使用对数刻度，数值越大表示对扰动越敏感；满秩不保证精度。条件数和 RMS 同时改变不能识别单一因果。", data)

    # 3. Replace the illegible 182-number heatmap with paired ratio summaries.
    selected = [("E1_width", "hw0.25", "半宽 0.25f"),
                ("E1_width", "hw0.45", "半宽 0.45f"),
                ("E3_shape", "circle_s0", "等面积圆窗"),
                ("E4_erosion", "er12", "交集腐蚀 er12"),
                ("E4_phase_sigma", "ps20", "平滑 σ=20")]
    title = "候选设置的收益，随随机波前改变"
    fig = canvas(title, "13 个随机真值 · 点为配对误差比的均值，横线为最小值至最大值。", "4:3")
    ax = axis(fig, left=.26, right=.82, bottom=.17, top=.79); data = {}
    for j, k in enumerate(COLORS):
        entries = []
        for i, (g, n, _) in enumerate(selected):
            rr = ratios(k, g, n); y = i + (j-.5)*.25
            ax.plot([rr.min(), rr.max()], [y, y], color=COLORS[k], alpha=.55, lw=2)
            ax.scatter(rr.mean(), y, color=COLORS[k], marker=MARKERS[k], s=40,
                       label=LABELS[k] if i == 0 else None, zorder=3)
            count = int((rr < 1-1e-12).sum())
            ax.text(1.055, y, f"{count}/13", color=COLORS[k], va="center",
                    fontsize=10.5, transform=ax.get_yaxis_transform())
            entries.append({"configuration": n, "paired_ratios": rr.tolist(),
                            "mean_ratio": float(rr.mean()), "improved": count})
        data[k] = entries
    ax.axvline(1, color="#687988", ls="--", lw=1.2)
    ax.set(xlim=(.60, 1.12), ylim=(4.6, -.6),
           xlabel="同一真值的 RMS / 默认 RMS（1 = 不变）")
    ax.set_xticks([.6, .7, .8, .9, 1, 1.1])
    ax.set_yticks(range(5), [r[2] for r in selected])
    ax.text(1.055, 1.025, "改善数", fontsize=10.5, transform=ax.transAxes)
    ax.grid(axis="x"); legend(fig, ax, y=.85, x=.26)
    finish(fig, "03_truths", title,
           "图 3｜按同一随机真值配对比较：小于 1 为改善，大于 1 为变差；右列为改善的真值数量。",
           "纵轴列出五种候选设置；每行的蓝色与橙色分别代表单源与 64 源。横轴先对每个真值计算“该设置 RMS ÷ 该真值默认 RMS”，再汇总 13 个比值：点是算术均值，横线是完整最小—最大范围，不是误差条或置信区间。虚线 1 表示无变化，0.8 表示 RMS 下降 20%，1.05 表示上升 5%。横轴为线性刻度，从 0.60 起画；这五组的全部比值都在图示范围内。0.10f 窄窗与幅度阈值 0.5 的 13 个随机真值均变差，其完整数据见附件。",
           data, 3)

    title = "小范围 x 中心偏移仍能改变恢复误差"
    fig = canvas(title, "主真值 · 无噪声 · 测量、零参考和标定共用同一个窗口")
    ax = axis(fig); data = {"offset_bin": list(range(-3, 4))}
    for k in COLORS:
        yy = [val(k, "E2_center_x", f"ox{x:+d}")/val(k) for x in range(-3, 4)]
        line(ax, k, list(range(-3, 4)), yy); data[k] = yy
    ax.axhline(1, color="#6E7B88", ls="--", lw=1.2)
    ax.set(xlabel="正 x 边带沿 x 轴的窗口中心偏移（bin）",
           ylabel="无噪声 RMS / 默认 RMS", xlim=(-3.3, 3.3), ylim=(.955, 1.025))
    ax.set_xticks(range(-3, 4)); ax.grid(axis="y"); legend(fig, ax)
    finish(fig, "04_center", title,
           "图 4｜x 中心共窗偏移扫描；横虚线 1 表示默认窗口的误差水平。",
           "横轴为正 x 基频边带窗口相对默认位置的整数频点偏移；负值朝 DC 移动，正值远离 DC。±3 bin 约占载波的 ±1.06%。纵轴是该配置的 clean RMS 除以默认 clean RMS，0.98 表示误差比默认低 2%；纵轴从 0.955 起画，便于看小变化。此图没有测量—标定中心失配；名义亚像素组的退化问题仍见下表。",
           data, 4)

    # 5. Geometry and RMS are separate figures, each with its own coordinates.
    radius = geom["camera_pupil_radius_m"]/cfg["dx_cam"]
    v = np.arange(-600, 601); mx, my = np.meshgrid(v, v)
    rr = np.hypot(mx, my)/radius; eps = cfg["NA_in"]/cfg["NA_out"]
    cp = (rr >= eps) & (rr <= 1)
    mask = cp & np.roll(cp, 54, axis=1)
    eroded = ndimage.binary_erosion(mask, iterations=12)
    assert int(mask.sum()) == old_audit["camera_mask_pixels_per_axis_baseline"]
    assert int(eroded.sum()) == old_audit["camera_mask_pixels_per_axis_er12"]
    title = "er12 删除交集边界，保留内部拟合像素"
    fig = canvas(title, "x 剪切交集示意 · 这是相机坐标下的拟合掩膜，不是最终评价光瞳。", "4:3")
    ax = fig.add_axes([.17, .16, .65, .65])
    img = np.where(eroded, 1, np.where(mask, 2, 0))
    ax.imshow(img, origin="lower", extent=[-600.5, 600.5, -600.5, 600.5],
              cmap=ListedColormap(["#F5F7F9", "#82A9C8", ORANGE]), vmin=0, vmax=2,
              interpolation="nearest")
    ax.set(xlabel="相对光轴的相机 x 坐标（px）", ylabel="相对光轴的相机 y 坐标（px）")
    ax.set_xticks([-500, 0, 500]); ax.set_yticks([-500, 0, 500])
    fig.legend(handles=[Patch(color="#82A9C8", label="保留的拟合像素"),
                        Patch(color=ORANGE, label="er12 删除的边界像素")],
               loc="upper left", bbox_to_anchor=(.16, .87), ncol=2, borderaxespad=0)
    finish(fig, "05_mask", title,
           "图 5｜蓝色为 er12 保留的交集像素，橙色为删除的边界，浅灰区域不参与该轴拟合。",
           "横、纵轴是在相机平面以光轴为原点的像素坐标；此处的 px 是相机像素，不是图 2 的频谱 bin。图示 x 方向剪切 54 px 后两个环形光瞳的交集。er12 是四邻域二值腐蚀 12 次，每个剪切轴保留 642,282/692,778=92.71% 的交集像素。内遮拦、外边界及移位交集边界都受影响；浅灰区域不等同于最终评价时被删除的像素。",
           {"baseline_pixels": int(mask.sum()), "er12_pixels": int(eroded.sum()),
            "retained_fraction": float(eroded.sum()/mask.sum())}, 5)

    title = "增加交集腐蚀量，主真值的全光瞳误差下降"
    fig = canvas(title, "主真值 · 40 dB 五种子均值 ± 样本标准差 · 每个腐蚀量重新标定")
    ax = axis(fig); ers = [0, 2, 4, 8, 12]; data = {"erosion_iterations": ers}
    for k in COLORS:
        arrays = [noise(k) if er == 0 else noise(k, "E4_erosion", f"er{er}") for er in ers]
        means = np.array([x.mean() for x in arrays])*1e4
        sds = np.array([x.std(ddof=1) for x in arrays])*1e4
        ax.errorbar(ers, means, sds, fmt="-"+MARKERS[k], color=COLORS[k],
                    ms=6, capsize=4, label=LABELS[k])
        data[k] = {"mean": means.tolist(), "sample_sd": sds.tolist()}
    ax.set(xlabel="交集掩膜四邻域腐蚀次数（0 为默认）",
           ylabel="全光瞳 RMS（10⁻⁴ waves）", xlim=(-.5, 12.5), ylim=(2.2, 4.8))
    ax.set_xticks(ers); ax.grid(axis="y"); legend(fig, ax)
    finish(fig, "06_erosion", title,
           "图 6｜含噪主真值的腐蚀响应；点为五种子均值，竖线为 ±1 个样本标准差。",
           "横轴是四邻域腐蚀的迭代次数，数字越大表示移除更多交集边界像素；横轴 0 是未腐蚀的默认配置。纵轴仍在同一个完整原生光瞳上计算 RMS，越低越好。误差条是 seeds41—45 的样本标准差，不是置信区间；它们很小，部分被点遮住。该图只描述固定主真值，13 个随机真值的适用范围见图 3。",
           data, 6)

    # 7. Same four saved fields, common signed scale, no crowded panel titles.
    dx = cfg["super_lattice_period"]/cfg["N_lattice"]
    nn = cfg["N_lattice"]*cfg["N_super_lattice"]
    step = cfg["wavelength"]/(nn*dx*cfg["NA_out"])
    extent = int(np.ceil(1/step))
    freq = np.fft.fftshift(np.fft.fftfreq(nn, dx))[nn//2-extent:nn//2+extent+1]
    xx, yy = np.meshgrid(freq, freq)
    rho = cfg["wavelength"]*np.hypot(xx, yy)/cfg["NA_out"]
    xy = freq*cfg["wavelength"]/cfg["NA_out"]
    pupil = (rho >= eps) & (rho <= 1)
    zz = basis(tuple(range(1, 36)), rho, np.arctan2(yy, xx), eps, 5)
    assert pupil.sum() == 52428
    fields, data = {}, {}
    for k in COLORS:
        for g, n in [("E0_baseline", "baseline"), ("E4_erosion", "er12")]:
            r = record(k, g, n)
            e = np.tensordot(np.array(r["coefficients"])-r["truth"], zz, axes=1)
            assert np.isclose(np.sqrt(np.mean(e[pupil]**2)), r["raw_rms_waves"],
                              rtol=1e-8, atol=1e-12)
            fields[k, n] = e*1e4
            data[k+"_"+n] = {"rms_waves": r["raw_rms_waves"]}
    scale = max(np.max(abs(e[pupil])) for e in fields.values())
    title = "腐蚀后误差减小，角向结构仍然存在"
    fig = canvas(title, "主真值 · 无噪声 · 四幅图使用相同光瞳和同一个有符号色标。", "4:3")
    axs = fig.subplots(2, 2, gridspec_kw={"left": .12, "right": .85, "bottom": .12,
                                       "top": .80, "wspace": .34, "hspace": .50})
    for i, k in enumerate(COLORS):
        for j, n in enumerate(["baseline", "er12"]):
            ax = axs[i, j]
            im = ax.imshow(np.where(pupil, fields[k, n], np.nan), origin="lower",
                           extent=[xy[0], xy[-1], xy[0], xy[-1]], cmap="RdBu_r",
                           vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(LABELS[k]+" · "+("默认" if j == 0 else "er12"), fontsize=12, pad=6)
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.set(xlabel="归一化光瞳 x", ylabel="归一化光瞳 y")
            ax.xaxis.label.set_fontsize(10.5); ax.yaxis.label.set_fontsize(10.5)
    cb = fig.colorbar(im, cax=fig.add_axes([.89, .20, .018, .51]))
    cb.set_label("有符号波前误差（10⁻⁴ waves）", fontsize=11)
    finish(fig, "07_error_maps", title,
           "图 7｜行对应单源/64 源，列对应默认/er12；蓝负红正，白色接近零，四图共用色标。",
           "横轴为 λf_x/NA_out，纵轴为 λf_y/NA_out，均为无量纲归一化光瞳坐标；外圆半径为 1，内遮拦半径为 NA_in/NA_out。这里的坐标不是相机像素。颜色表示恢复波前减去真值的有符号误差，色条单位为 10⁻⁴ waves；光瞳外和内遮拦留白。共用色标覆盖四图的绝对最大误差，因此内部弱误差的颜色较浅；未截断高误差值。颜色接近零不代表局部 FT 相位没有误差。",
           {**data, "common_color_limit_1e4_waves": float(scale)}, 7)

    # 8/9. Shape benefit and bandwidth need distinct scales and separate figures.
    names = ["rect_s0", "rect_s5", "circle_s0", "ellipse_s0", "gaussian_s0"]
    titles = ["硬矩形", "软矩形 σ=5", "等面积圆窗", "等面积椭圆", "高斯窗"]
    title = "等面积圆窗降低了主真值误差"
    fig = canvas(title, "主真值 · 无噪声 · 相对于默认软矩形窗口的变化")
    ax = axis(fig, left=.23, top=.74); data = {}
    for j, k in enumerate(COLORS):
        yy = [100*(val(k, "E3_shape", n)/val(k)-1) for n in names]
        ax.scatter(yy, np.arange(5)+(j-.5)*.24, color=COLORS[k],
                   marker=MARKERS[k], s=55, label=LABELS[k], zorder=3)
        data[k] = yy
    ax.set_yticks(range(5), titles); ax.set_ylim(4.6, -.6)
    ax.axvline(0, color="#6E7B88", ls="--", lw=1.2)
    ax.set_xlabel("相对默认窗口的 RMS 变化（%）")
    ax.set_xlim(-12, 8); ax.set_xticks([-10, -5, 0, 5])
    ax.grid(axis="x"); legend(fig, ax, x=.23)
    finish(fig, "08_shapes", title,
           "图 8｜窗形的主真值收益：负值表示误差下降，零虚线为默认软矩形窗。",
           "纵轴是具体窗口实现，默认参照为软矩形 σ=2 bin；“软矩形 σ=5”是另一个配置。横轴为 100×(该窗口 clean RMS/默认 clean RMS−1)，单位 %；−8 表示误差下降 8%，越靠左越好。圆窗和椭圆窗按连续等面积构造，但各窗的有效带宽尚未完全配平，因此图中收益不能全部归因于窗形本身。",
           data, 8)

    title = "高斯窗的有效噪声像素数约为矩形两倍"
    fig = canvas(title, "窗口权重诊断 · 有效像素数越大不等于恢复精度越高")
    ax = axis(fig, left=.23, right=.91, top=.75)
    effective = [record("single")["window_metrics"][0]["equivalent_noise_bins"]]
    effective += [record("single", "E3_shape", n)["window_metrics"][0]["equivalent_noise_bins"]
                  for n in names]
    ax.barh(np.arange(6), np.array(effective)/1e4,
            color=["#82A9C8"]*5+[ORANGE], height=.60)
    ax.set_yticks(range(6), ["默认软矩形 σ=2"]+titles); ax.invert_yaxis()
    ax.set(xlabel="有效噪声像素数 N_eff（10⁴ 个频谱像素）", xlim=(0, 12))
    ax.grid(axis="x"); ax.set_axisbelow(True)
    for i, v in enumerate(effective):
        ax.text(v/1e4+.15, i, f"{v/1e4:.2f}", va="center", fontsize=11)
    finish(fig, "09_bandwidth", title,
           "图 9｜窗口的等效噪声像素数；柱尾数字与横轴同为 10⁴ 个频谱像素。",
           "纵轴是窗口实现；横轴为 N_eff=(Σw)²/Σw²，其中 w 是频谱窗口的逐像素权重。刻度 5 约等于 5 万个等权频谱像素的噪声面积，不是相机像素数，也不是 RMS。权重为二值时它才等于窗口像素数。该组只展示正 x 窗；两个源数使用同样的窗口。高斯约 10.04 万，其他形状约 5 万，因此这组并非严格等有效带宽比较。",
           {"window_names": ["default"]+names, "effective_pixels": effective}, 9)

    title = "当前幅度阈值实现使恢复误差增大"
    fig = canvas(title, "主真值 · 无噪声 · 结论限于截取旧伪逆列的现有实现")
    ax = axis(fig); data = {"relative_amplitude_threshold": [0, .1, .3, .5]}
    for k in COLORS:
        yy = [val(k)*1e4]+[val(k, "E4_amplitude", f"thr{t}")*1e4 for t in [.1, .3, .5]]
        line(ax, k, [0, .1, .3, .5], yy); data[k] = yy
    ax.set(xlabel="相对最大幅度的筛选阈值（0 为默认未筛选）",
           ylabel="全光瞳 RMS（10⁻⁴ waves）", xlim=(-.02, .52))
    ax.set_xticks([0, .1, .3, .5]); ax.grid(axis="y"); legend(fig, ax)
    finish(fig, "10_threshold", title,
           "图 10｜当前幅度阈值实现的主真值响应。",
           "横轴是相对该张图最大复场幅度的阈值，例如 0.3 表示以最大幅度的 30% 为界筛选；0 为不筛选的默认链路。纵轴为完整光瞳 clean RMS，越低越好。现有代码只截取旧伪逆的列，未对筛选后的设计矩阵重算伪逆；此图不能用于否定正确的重拟合或加权方法。",
           data, 10)

    title = "σ=20 优于默认 σ=10，尚不能称为最优"
    fig = canvas(title, "主真值 · 无噪声 · 掩膜外补零后进行高斯相位平滑")
    ax = axis(fig); sigmas = [0, 5, 10, 20]; data = {"sigma_camera_px": sigmas}
    for k in COLORS:
        yy = [(val(k) if s == 10 else val(k, "E4_phase_sigma", f"ps{s}"))*1e4
              for s in sigmas]
        line(ax, k, sigmas, yy); data[k] = yy
    ax.axvline(10, ls="--", color="#6E7B88", lw=1.2)
    ax.set(xlabel="高斯相位平滑标准差 σ（相机 px）",
           ylabel="全光瞳 RMS（10⁻⁴ waves）", xlim=(-1, 21))
    ax.set_xticks(sigmas); ax.grid(axis="y"); legend(fig, ax)
    finish(fig, "11_smoothing", title, "图 11｜相位平滑标准差的主真值响应；虚线为默认 σ=10。",
           "横轴为高斯平滑的标准差 σ，单位是相机像素；σ=0 表示关闭平滑，越大表示平滑范围越宽。纵轴为全光瞳 clean RMS，越低越好。该图测试的是掩膜外补零后的平滑，未比较掩膜内归一化平滑；最右端 σ=20 是已测端点，不能据此认定通用最优。",
           data, 11)

    title = "已测参数效应高于五种子噪声重复波动"
    fig = canvas(title, "固定主真值 · clean 极差 / 默认 40 dB 五种子 RMS 的样本标准差")
    ax = axis(fig, left=.23, top=.74); data = {}
    groups = [("E1_width", "窗宽"), ("E2_center_x", "x 中心偏移"),
              ("E3_shape", "窗形"), ("E4_erosion", "交集腐蚀"),
              ("E4_amplitude", "幅度阈值"), ("E4_phase_sigma", "相位平滑")]
    for j, k in enumerate(COLORS):
        entries = []
        for i, (g, _) in enumerate(groups):
            vals = [r["raw_rms_waves"] for r in raw if
                    r["kind"] == k and r["group"] == g and r["condition"] == "clean"]
            vals += [val(k)]
            q = (max(vals)-min(vals))/noise(k).std(ddof=1)
            y = i+(j-.5)*.26
            ax.scatter(q, y, color=COLORS[k], marker=MARKERS[k], s=45,
                       label=LABELS[k] if i == 0 else None, zorder=3)
            ax.text(q*1.12, y, f"{q:.1f}" if q < 20 else f"{q:.0f}",
                    color=COLORS[k], va="center", fontsize=10.5)
            entries.append({"group": g, "range_over_sd": q})
        data[k] = entries
    ax.set(xscale="log", xlim=(1, 1000), ylim=(5.6, -.6),
           xlabel="参数 clean 极差 / 基线噪声样本 SD（对数刻度）")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks(range(6), [x[1] for x in groups]); ax.grid(axis="x"); legend(fig, ax, x=.23)
    finish(fig, "12_vs_noise", title,
           "图 12｜参数扫描的确定性极差相对噪声重复波动的量级；点旁数字为该比值。",
           "纵轴是本次实际扫描的参数组，各组都计入默认点。横轴用该组 clean RMS 的最大值减最小值，再除以默认窗口 40 dB 五种子的 RMS 样本标准差。数值 10 表示扫描极差为噪声 SD 的 10 倍；横轴是对数刻度，1、10、100 等距。该比值依赖扫描范围，既不是显著性检验，也不是普遍重要性排名；幅度阈值还受已知实现缺陷影响。",
           data, 12)

    title = "er12 降低主真值误差，FT 仍高于 LM"
    fig = canvas(title, "同一观测 · 40 dB 五种子均值 ± 样本标准差 · LM 复用 run-08")
    ax = axis(fig, left=.23, top=.74); data = {}
    for j, k in enumerate(COLORS):
        lm = np.array([r["raw_rms_waves"] for r in run08 if r["na_out"] == .02 and
                       r["source_kind"] == k and r["method"] == "lm" and
                       r["condition"].startswith("40db")])
        arrays = [noise(k), noise(k, "E4_erosion", "er12"), lm]
        means = np.array([x.mean() for x in arrays])*1e4
        sds = np.array([x.std(ddof=1) for x in arrays])*1e4
        yy = np.arange(3)+(j-.5)*.28
        ax.errorbar(means, yy, xerr=sds, fmt=MARKERS[k], color=COLORS[k],
                    ms=6, capsize=4, label=LABELS[k])
        for x, y, sd in zip(means, yy, sds):
            ax.text(x+sd+.12, y, f"{x:.3f}", va="center", fontsize=10.5, color=COLORS[k])
        data[k] = {"mean": means.tolist(), "sample_sd": sds.tolist()}
    ax.set_yticks(range(3), ["默认 FT", "er12 FT", "run-08 LM"])
    ax.set(ylim=(2.55, -.55), xlim=(0, 5.25),
           xlabel="全光瞳 RMS（10⁻⁴ waves；越靠左越好）")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis="x"); legend(fig, ax, x=.23)
    finish(fig, "13_lm", title,
           "图 13｜主真值上的同观测方法对比；点为均值，横向误差条为 ±1 个样本 SD。",
           "纵轴列出三条恢复链路；横轴为完整光瞳 RMS，刻度 4 表示 4×10⁻⁴ waves，越靠左误差越低。点旁只标均值，完整均值±样本 SD 见下表。FT 与 LM 使用相同五个含噪观测；LM 直接复用 run-08。本图未包含 13 个随机真值的 LM 结果，不能将主真值上的倍数推广到全部波前。",
           data, 13)

    title = "误选 2f 并沿用原模型，会造成整体失配"
    fig = canvas(title, "主真值 · 无噪声 · 故障对照，未为 2f 重建差分基和掩膜")
    ax = axis(fig, left=.23, top=.74); data = {}
    for j, k in enumerate(COLORS):
        qq = [1., val(k, "E5c_sideband", "sideband_2f")/val(k)]
        yy = np.arange(2)+(j-.5)*.28
        ax.scatter(qq, yy, color=COLORS[k], marker=MARKERS[k], s=55, label=LABELS[k])
        for x, y in zip(qq, yy):
            ax.text(x*1.13, y, f"{x:.1f}×", color=COLORS[k], va="center", fontsize=11)
        data[k] = qq
    ax.set(yscale="linear", xscale="log", xlim=(.8, 90), ylim=(1.55, -.55),
           xlabel="clean RMS / 默认 FT RMS（对数刻度）")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks([0, 1], ["默认基频 f", "误选 2f"])
    ax.grid(axis="x"); legend(fig, ax, x=.23)
    finish(fig, "14_sideband", title,
           "图 14｜误选 2f 的故障对照；误差倍数相对默认基频链路计算。",
           "纵轴区分默认基频与误选 2f。横轴为该配置 clean RMS 除以默认 FT 的 clean RMS，1 是默认误差水平，10 表示误差为默认的 10 倍；横轴为对数刻度。2f 组仍沿用相邻级次的原差分基与掩膜，因此只能说明这次整体流程失配的代价，不能据此否定正确建模的 2f 方法。",
           data, 14)
    pdf.close()
    (a.output / "figure-manifest.json").write_text(json.dumps(figures, ensure_ascii=False, indent=2))
    (a.output / "layout-audit.json").write_text(json.dumps(layout_audit, ensure_ascii=False, indent=2))
    audit = {
        "input_rows": 2000, "scalar_summary_checks": len(summary)*7,
        "reconstructed_error_maps": 4, "pupil_pixels": int(pupil.sum()),
        "no_new_numerical_experiment": True,
        "figures": len(figures),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            a.original_output/"successful-results.json",
            a.original_output/"configuration-summary.csv",
            a.source/"config.json", a.source/"geometry.json", a.source/"run08-summary.json"]},
        "layout_collisions": sum(len(x["text_overlaps"]) for x in layout_audit),
        "clipped_text": sum(len(x["clipped_text"]) for x in layout_audit),
    }
    (a.output / "data-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["layout_collisions"] or audit["clipped_text"]:
        print("Inspect layout-audit.json; revise all reported text collisions before publishing.")


if __name__ == "__main__":
    main()
