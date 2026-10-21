#!/usr/bin/env python
"""Consolidate the serial follow-up studies into reviewable tables and figures.

The script only reads saved run directories; it never launches an experiment.
Raw observation/calibration arrays stay in ``output/``.  Compact numeric
summaries and figures are written under ``docs/results/followup`` so a reviewer
can check the claims without loading the large arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("output/diffraction/followup_20261021")
OUT = Path("docs/results/followup")
BASELINE = Path("work/handoff-20261021/output/diffraction/reference_compare/results.json")
RUN_NAMES = [
    "regions_width_frozen",
    "regions_width_recal",
    "regions_center",
    "regions_shape",
    "regions_mask",
    "regions_cross_angle_width",
    "angles_stage1",
    "angles_camera_0p5",
    "angles_grating_0p5",
    "angles_object_control",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_run(path: Path) -> dict | None:
    if not (path / "results.json").exists():
        return None
    rows = read_json(path / "results.json")
    manifest = read_json(path / "manifest.json") if (path / "manifest.json").exists() else {}
    return {"path": str(path), "rows": rows, "manifest": manifest,
            "files": sum(1 for p in path.rglob("*") if p.is_file()),
            "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file())}


def ok_rows(run: dict | None) -> list[dict]:
    return [row for row in (run or {}).get("rows", []) if row.get("status") == "ok"]


def finite_mean(values) -> float | None:
    array = np.asarray([value for value in values if value is not None], float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def resource(run: dict | None) -> dict:
    if not run:
        return {}
    m = run["manifest"]
    args = m.get("arguments", {})
    return {"seconds": m.get("seconds"), "peak_memory_mib": m.get("peak_memory_mib"),
            "row_count": m.get("row_count"), "status": m.get("status"),
            "workers": args.get("workers"), "batch_sources": args.get("batch_sources"),
            "files": run["files"], "bytes": run["bytes"]}


def collect_width(run: dict | None) -> dict:
    if not run:
        return {}
    widths = {}
    for row in ok_rows(run):
        if row.get("method") != "fourier_calibrated":
            continue
        case = row.get("case", "")
        if not case.startswith("width_"):
            continue
        settings = row.get("settings", {})
        widths[float(settings["half_width"])] = {
            "half_width": float(settings["half_width"]),
            "rmse_waves": row.get("rmse_waves"),
            "pv_waves": row.get("pv_waves"),
            "coverage": row.get("coverage"),
            "phase_residual_rms": row.get("phase_residual_rms"),
            "window_metrics": row.get("window_metrics"),
        }
    selection = {}
    if (Path(run["path"]) / "width_selection.json").exists():
        selection = read_json(Path(run["path"]) / "width_selection.json")
    selected_row = next((row for row in ok_rows(run)
                         if row.get("case") == "selected_width"), None)
    direct = next((row for row in ok_rows(run) if row.get("method") == "direct"), None)
    ordered = [widths[key] for key in sorted(widths)]
    best = min(ordered, key=lambda row: row["rmse_waves"]) if ordered else None
    return {"widths": ordered, "best_test_truth": best, "selection": selection,
            "selected_width_row": selected_row,
            "direct_baseline_rmse_waves": direct.get("rmse_waves") if direct else None}


def collect_center(run: dict | None) -> dict:
    if not run:
        return {}
    points = []
    for row in ok_rows(run):
        if row.get("method") != "fourier_calibrated" or row.get("scan") != "center":
            continue
        settings = row.get("settings", {})
        points.append({"x": float(settings["offset_x"]), "y": float(settings["offset_y"]),
                       "rmse_waves": row.get("rmse_waves"), "pv_waves": row.get("pv_waves"),
                       "coverage": row.get("coverage")})
    return {"points": points}


def collect_shape(run: dict | None) -> dict:
    if not run:
        return {}
    points = []
    for row in ok_rows(run):
        if row.get("method") != "fourier_calibrated":
            continue
        case = row.get("case", "")
        if not (case.startswith("shape_") or case.startswith("ellipse_aspect_")):
            continue
        settings = row.get("settings", {})
        metrics = row.get("window_metrics") or [{}, {}]
        points.append({"case": case, "shape": settings.get("shape"),
                       "aspect": float(settings.get("aspect", 1)),
                       "rmse_waves": row.get("rmse_waves"), "pv_waves": row.get("pv_waves"),
                       "coverage": row.get("coverage"),
                       "envelope_bins": finite_mean([m.get("equivalent_noise_bins")
                                                     for m in metrics]),
                       "weight_sum": finite_mean([m.get("weight_sum") for m in metrics])})
    return {"points": points}


def collect_mask(run: dict | None) -> dict:
    if not run:
        return {}
    erosion, threshold = [], []
    for row in ok_rows(run):
        if row.get("method") != "fourier_calibrated":
            continue
        case = row.get("case", "")
        settings = row.get("settings", {})
        metrics = row.get("window_metrics") or [{}, {}]
        item = {"case": case, "rmse_waves": row.get("rmse_waves"),
                "pv_waves": row.get("pv_waves"), "coverage": row.get("coverage"),
                "weight_sum": finite_mean([m.get("weight_sum") for m in metrics])}
        if case.startswith("erosion_"):
            item["erosion"] = int(settings.get("erosion", 0))
            erosion.append(item)
        elif case.startswith("amplitude_"):
            item["threshold"] = float(settings.get("amplitude_threshold", 0))
            threshold.append(item)
    return {"erosion": sorted(erosion, key=lambda row: row["erosion"]),
            "threshold": sorted(threshold, key=lambda row: row["threshold"])}


def collect_angles(runs: dict[str, dict | None]) -> dict:
    rows = []
    zero = []
    for run_name, run in runs.items():
        if not run_name.startswith("angles"):
            continue
        for row in ok_rows(run):
            component = row.get("component")
            if component is None:
                continue
            item = {"run": run_name, "case": row.get("case"), "method": row.get("method"),
                    "component": component, "axis": row.get("axis"),
                    "angle_deg": row.get("angle_deg"), "geometry": row.get("geometry"),
                    "rmse_waves": row.get("rmse_waves"), "pv_waves": row.get("pv_waves"),
                    "coverage": row.get("coverage"),
                    "image_change_relative_l2": row.get("image_change_relative_l2"),
                    "center_mode": (row.get("settings") or {}).get("center_mode", "none"),
                    "converged": row.get("converged"),
                    "status": row.get("status")}
            if component == "zero" or row.get("angle_deg") == 0:
                zero.append(item)
            else:
                rows.append(item)
    return {"rows": rows, "zero": zero}


def collect_methods(root: Path) -> dict:
    runs = {}
    for path in sorted(root.glob("*")):
        if not path.is_dir():
            continue
        if not (path.name.startswith("methods_") or path.name.startswith("noise_")
                or path.name.startswith("init_")):
            continue
        run = load_run(path)
        if run is None:
            continue
        args = run["manifest"].get("arguments", {})
        rows = ok_rows(run)
        fourier = next((row for row in rows if row.get("method") == "fourier_calibrated"), None)
        direct = next((row for row in rows if row.get("method") == "direct"), None)
        calibration = {}
        calibration_path = path / "baseline_calibration.json"
        if calibration_path.exists():
            calibration = read_json(calibration_path)
        trials = {}
        trial_path = path / "baseline_direct_trials.json"
        if trial_path.exists():
            trials = read_json(trial_path)
        trial_rows = []
        for trial in trials.get("trials", []):
            metrics = trial.get("metrics", {})
            trial_rows.append({"converged": trial.get("converged"),
                               "nfev": trial.get("nfev"),
                               "rmse_waves": metrics.get("rmse_waves"),
                               "coeffs": trial.get("coeffs")})
        runs[path.name] = {
            "truth_rms": args.get("truth_rms"), "snr_db": args.get("snr_db"),
            "seed": args.get("seed"), "starts": args.get("starts"),
            "screen_initials": args.get("screen_initials"),
            "fourier_rmse_waves": fourier.get("rmse_waves") if fourier else None,
            "direct_rmse_waves": direct.get("rmse_waves") if direct else None,
            "fourier_coverage": fourier.get("coverage") if fourier else None,
            "direct_full_intensity_rms": direct.get("full_intensity_rms") if direct else None,
            "calibration_condition": calibration.get("condition"),
            "calibration_validation_rmse_waves": (calibration.get("held_out_mixed_validation") or {}).get("rmse_waves"),
            "trials": trial_rows,
            "resource": resource(run),
        }
    return runs


def shape_label(point: dict) -> str:
    shape = point["shape"]
    if shape == "ellipse":
        return f"ellipse {point['aspect']:g}"
    return str(shape)


def plot_width(width_frozen: dict, width_recal: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for data, label, marker, color in ((width_frozen, "frozen nominal calibration", "o", "C0"),
                                       (width_recal, "per-window recalibration", "s", "C2")):
        points = data.get("widths", [])
        if points:
            ax.plot([p["half_width"] for p in points], [p["rmse_waves"] for p in points],
                    marker + "-", color=color, label=label)
    selection = width_frozen.get("selection", {})
    scores = selection.get("scores", [])
    if scores:
        ax.errorbar([row["half_width"] for row in scores],
                    [row["validation_rmse_waves"] for row in scores],
                    yerr=[row.get("validation_rmse_std_waves", 0) for row in scores],
                    fmt="D--", color="C3", alpha=.7, label="independent validation (mean±sd)")
    direct = width_frozen.get("direct_baseline_rmse_waves")
    if direct is not None:
        ax.axhline(direct, color="C1", ls=":", label="direct least squares")
    best = width_frozen.get("best_test_truth")
    if best:
        ax.axvline(best["half_width"], color="k", ls="--", lw=1, alpha=.5,
                   label="best on test truth (analysis only)")
    selected = selection.get("selected_width")
    if selected is not None:
        ax.axvline(selected, color="C3", ls=":", lw=1.5, label="validation-selected width")
    ax.set_xlabel("FFT window half-width / bins")
    ax.set_ylabel("Wavefront RMSE / waves")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"width_scan.{suffix}", dpi=160)
    plt.close(fig)


def plot_center(center: dict, path: Path) -> None:
    points = center.get("points", [])
    if not points:
        return
    xs = sorted({int(p["x"]) for p in points})
    ys = sorted({int(p["y"]) for p in points})
    grid = np.full((len(ys), len(xs)), np.nan)
    for point in points:
        grid[ys.index(int(point["y"])), xs.index(int(point["x"]))] = point["rmse_waves"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(grid, origin="lower", extent=(min(xs)-.5, max(xs)+.5, min(ys)-.5, max(ys)+.5),
                      cmap="viridis", aspect="equal")
    ax.set_xlabel("window centre offset x / bins")
    ax.set_ylabel("window centre offset y / bins")
    ax.set_title("Fourier wavefront RMSE / waves (frozen nominal calibration)")
    fig.colorbar(image, ax=ax, label="RMSE / waves")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"center_heatmap.{suffix}", dpi=160)
    plt.close(fig)


def plot_shape_mask(shape: dict, mask: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    points = shape.get("points", [])
    if points:
        labels = [shape_label(point) for point in points]
        values = [point["rmse_waves"] for point in points]
        axes[0].bar(range(len(labels)), values, color="C0")
        axes[0].set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("Wavefront RMSE / waves")
        axes[0].set_title("Window shape / aspect (equal-area)")
    erosion = mask.get("erosion", [])
    if erosion:
        axes[1].plot([row["erosion"] for row in erosion], [row["rmse_waves"] for row in erosion],
                     "o-", color="C2", label="RMSE")
        axes[1].set_xlabel("mask erosion / pixels")
        axes[1].set_ylabel("RMSE / waves", color="C2")
        axes[1].set_title("Spatial-mask erosion")
        twin = axes[1].twinx()
        twin.plot([row["erosion"] for row in erosion], [row["coverage"] for row in erosion],
                  "s--", color="C1", alpha=.7)
        twin.set_ylabel("coverage", color="C1")
    threshold = mask.get("threshold", [])
    if threshold:
        axes[2].plot([row["threshold"] for row in threshold], [row["rmse_waves"] for row in threshold],
                     "o-", color="C2")
        axes[2].set_xlabel("low-amplitude rejection threshold")
        axes[2].set_ylabel("RMSE / waves")
        axes[2].set_title("Low-amplitude rejection")
        twin = axes[2].twinx()
        twin.plot([row["threshold"] for row in threshold], [row["coverage"] for row in threshold],
                  "s--", color="C1", alpha=.7)
        twin.set_ylabel("coverage", color="C1")
    for ax in axes:
        ax.grid(True, alpha=.25)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"shape_mask.{suffix}", dpi=160)
    plt.close(fig)


def plot_angles(angles: dict, path: Path) -> None:
    rows = angles.get("rows", [])
    if not rows:
        return
    components = [c for c in ("camera", "grating", "object") if any(r["component"] == c for r in rows)]
    axes_names = [a for a in ("x", "y", "diagonal") if any(r["axis"] == a for r in rows)]
    fig, grid = plt.subplots(len(components), max(1, len(axes_names)),
                             squeeze=False, figsize=(5.6*max(1, len(axes_names)), 3.8*len(components)+1))
    for i, component in enumerate(components):
        for j, axis in enumerate(axes_names or ["x"]):
            ax = grid[i, j]
            selected = [r for r in rows if r["component"] == component and r["axis"] == axis]
            groups: dict[tuple, list[dict]] = {}
            for row in selected:
                key = (row["method"], row["geometry"], row["center_mode"])
                groups.setdefault(key, []).append(row)
            for (method, geometry, center), group in groups.items():
                group.sort(key=lambda row: row["angle_deg"])
                if method == "direct":
                    label, color, marker = f"direct / {geometry}", "C1", "s"
                else:
                    label = f"Fourier {center} / {geometry}"
                    color = "C0" if center == "fixed" else "C2"
                    marker = "o"
                ax.plot([r["angle_deg"] for r in group], [r["rmse_waves"] for r in group],
                        marker + ("--" if geometry == "ignored" else "-"),
                        color=color, label=label, ms=4)
            ax.set_title(f"{component} / {axis} axis")
            ax.set_xlabel("reference phase-ramp tilt / degree")
            ax.set_ylabel("RMSE / waves")
            ax.grid(True, alpha=.25)
    handles, labels = grid[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(path / f"angle_scan.{suffix}", dpi=160)
    plt.close(fig)


def plot_methods(methods: dict, path: Path) -> None:
    rms = [(v["truth_rms"], v) for v in methods.values()
           if v.get("truth_rms") is not None and v.get("snr_db") is None]
    if rms:
        rms.sort(key=lambda pair: pair[0])
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot([pair[0] for pair in rms],
                [pair[1]["fourier_rmse_waves"] for pair in rms], "o-", label="calibrated Fourier")
        ax.plot([pair[0] for pair in rms],
                [pair[1]["direct_rmse_waves"] for pair in rms], "s-", label="direct least squares")
        ax.set_xlabel("true wavefront RMS / waves")
        ax.set_ylabel("recovered wavefront RMSE / waves")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(path / f"methods_rms.{suffix}", dpi=160)
        plt.close(fig)
    noise = [(v["snr_db"], v) for v in methods.values() if v.get("snr_db") is not None]
    if noise:
        snrs = sorted({pair[0] for pair in noise})
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for snr in snrs:
            group = [pair[1] for pair in noise if pair[0] == snr]
            ax.scatter([snr]*len(group), [v["fourier_rmse_waves"] for v in group],
                       color="C0", marker="o", label="Fourier" if snr == snrs[0] else None)
            ax.scatter([snr]*len(group), [v["direct_rmse_waves"] for v in group],
                       color="C1", marker="s", label="direct" if snr == snrs[0] else None)
        ax.set_xlabel("SNR / dB (peak / noise sd)")
        ax.set_ylabel("RMSE / waves")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(path / f"methods_noise.{suffix}", dpi=160)
        plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.{digits}g}"
    return str(value)


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# 后续实验数值摘要", "",
             f"生成时间：{summary['generated_at']}", "",
             "本文件由 `scripts/analyze_followup.py` 从已保存的 `results.json` 生成；",
             "原始观测、标定数组和拟合轨迹保留在 `output/diffraction/followup_20261021`。", ""]
    baseline = summary.get("baseline", {})
    if baseline:
        lines += ["## 已有基线（交接数据）", "",
                  "| 方法 | RMSE / waves | 说明 |", "|---|---:|---|",
                  f"| 标定傅里叶 | {fmt(baseline.get('fourier_rmse_waves'))} | 原采样、四模式、0.1 waves、无噪声 |",
                  f"| 直接最小二乘 | {fmt(baseline.get('direct_rmse_waves'))} | 同模型、单一零初值 |", ""]
    width = summary.get("width", {})
    for key, title in (("frozen", "频谱窗半宽：冻结名义标定"),
                       ("recalibrated", "频谱窗半宽：逐窗重标定（独立诊断）")):
        points = width.get(key, {}).get("widths", [])
        if not points:
            continue
        lines += [f"## {title}", "", "| 半宽 / bin | RMSE / waves | PV / waves | 覆盖率 | 相位残差 |",
                  "|---:|---:|---:|---:|---:|"]
        for point in points:
            lines.append(f"| {point['half_width']:g} | {fmt(point['rmse_waves'])} | "
                         f"{fmt(point['pv_waves'])} | {fmt(point['coverage'])} | "
                         f"{fmt(point['phase_residual_rms'])} |")
        lines.append("")
    selection = width.get("frozen", {}).get("selection", {})
    if selection:
        lines += ["独立验证集选窗（不读取测试真值）：", "",
                  f"- 验证样本数：{selection.get('repeats')}，RMS：{selection.get('validation_rms')} waves；",
                  f"- 选中半宽：{fmt(selection.get('selected_width'))} bin。", ""]
    best = width.get("frozen", {}).get("best_test_truth")
    if best:
        lines += [f"测试真值上的最低误差半宽（仅分析下界，不用于生产选窗）：{best['half_width']:g} bin，"
                  f"RMSE {fmt(best['rmse_waves'])} waves。", ""]
    center = summary.get("center", {}).get("points", [])
    if center:
        best_center = min(center, key=lambda row: row["rmse_waves"])
        lines += ["## 二维窗中心偏移", "",
                  f"- 扫描 {len(center)} 个偏移（x,y 各 -3…3 bin）；",
                  f"- 测试图上最低点：x={best_center['x']:g}, y={best_center['y']:g}，"
                  f"RMSE {fmt(best_center['rmse_waves'])} waves（仅分析）。", ""]
    shape = summary.get("shape", {}).get("points", [])
    if shape:
        lines += ["## 窗形状与长宽比", "", "| 形状 | RMSE / waves | 等效噪声 bin | 覆盖率 |",
                  "|---|---:|---:|---:|"]
        for point in shape:
            lines.append(f"| {shape_label(point)} | {fmt(point['rmse_waves'])} | "
                         f"{fmt(point['envelope_bins'])} | {fmt(point['coverage'])} |")
        lines.append("")
    mask = summary.get("mask", {})
    if mask.get("erosion") or mask.get("threshold"):
        lines += ["## 空间有效区", "", "| 类型 | 参数 | RMSE / waves | 覆盖率 |",
                  "|---|---:|---:|---:|"]
        for point in mask.get("erosion", []):
            lines.append(f"| 边缘腐蚀 | {point['erosion']} px | {fmt(point['rmse_waves'])} | {fmt(point['coverage'])} |")
        for point in mask.get("threshold", []):
            lines.append(f"| 低幅剔除 | {point['threshold']:g} | {fmt(point['rmse_waves'])} | {fmt(point['coverage'])} |")
        lines.append("")
    angle_rows = summary.get("angles", {}).get("rows", [])
    if angle_rows:
        lines += ["## 部件倾斜", "", "| 部件 | 轴 | 角度/° | 几何 | 中心 | 方法 | RMSE / waves | 图像变化 |",
                  "|---|---|---:|---|---|---|---:|---:|"]
        for row in sorted(angle_rows, key=lambda r: (r["component"], r["axis"], r["angle_deg"],
                                                     r["geometry"], r["method"], r["center_mode"])):
            lines.append(f"| {row['component']} | {row['axis']} | {row['angle_deg']:g} | "
                         f"{row['geometry']} | {row['center_mode']} | {row['method']} | "
                         f"{fmt(row['rmse_waves'])} | {fmt(row.get('image_change_relative_l2'))} |")
        lines.append("")
    methods = summary.get("methods", {})
    if methods:
        lines += ["## 波前幅度、噪声与初始化", "",
                  "| 运行 | 真值 RMS | SNR/dB | 种子 | starts | 傅里叶 RMSE | 直接 RMSE | 收敛/总数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in sorted(methods):
            item = methods[name]
            trials = item.get("trials", [])
            converged = sum(1 for trial in trials if trial.get("converged"))
            lines.append(f"| {name} | {fmt(item.get('truth_rms'))} | {fmt(item.get('snr_db'))} | "
                         f"{fmt(item.get('seed'))} | {fmt(item.get('starts'))} | "
                         f"{fmt(item.get('fourier_rmse_waves'))} | {fmt(item.get('direct_rmse_waves'))} | "
                         f"{converged}/{len(trials)} |")
        lines.append("")
    resources = summary.get("resources", {})
    if resources:
        lines += ["## 资源与运行状态", "", "| 运行 | 状态 | 行数 | 秒 | 峰值 MiB | 文件数 | 输出 MiB |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name in sorted(resources):
            item = resources[name]
            lines.append(f"| {name} | {item.get('status')} | {item.get('row_count')} | "
                         f"{fmt(item.get('seconds'), 5)} | {fmt(item.get('peak_memory_mib'), 5)} | "
                         f"{item.get('files')} | {fmt((item.get('bytes') or 0)/2**20, 4)} |")
        lines.append("")
    lines += ["## 限制", "",
              "- 所有结果来自同一参考模型的仿真反演；生成与反演使用同一物理模型，不能据此给出仪器精度。",
              "- 傅里叶标定是小信号局部线性标定；大像差或超出标定范围的结论需单独验证。",
              "- 参考倾斜是输入场相位坡度近似，不是严格旋转平面传播；物面单像素源倾斜是零效应对照。",
              "- 独立验证选窗只使用零角度、无噪声的混合波前样本；生产规则尚未在真实数据上冻结。", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {name: load_run(args.root / name) for name in RUN_NAMES}
    methods = collect_methods(args.root)
    resources = {name: resource(run) for name, run in runs.items() if run}
    resources.update({f"methods:{name}": item["resource"] for name, item in methods.items()})
    baseline = {}
    if args.baseline.exists():
        rows = read_json(args.baseline)
        baseline = {
            "fourier_rmse_waves": next((r.get("rmse_waves") for r in rows
                                        if r.get("method") == "fourier_calibrated"), None),
            "direct_rmse_waves": next((r.get("rmse_waves") for r in rows
                                       if r.get("method") == "direct"), None),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(args.root),
        "baseline": baseline,
        "width": {"frozen": collect_width(runs["regions_width_frozen"]),
                  "recalibrated": collect_width(runs["regions_width_recal"])},
        "center": collect_center(runs["regions_center"]),
        "shape": collect_shape(runs["regions_shape"]),
        "mask": collect_mask(runs["regions_mask"]),
        "cross_angle_width": collect_width(runs["regions_cross_angle_width"]),
        "angles": collect_angles(runs),
        "methods": methods,
        "resources": resources,
        "missing_runs": [name for name, run in runs.items() if run is None],
    }

    write_json(args.out / "summary.json", summary)
    write_markdown(summary, args.out / "summary.md")
    plot_width(summary["width"]["frozen"], summary["width"]["recalibrated"], args.out)
    plot_center(summary["center"], args.out)
    plot_shape_mask(summary["shape"], summary["mask"], args.out)
    plot_angles(summary["angles"], args.out)
    plot_methods(methods, args.out)
    print("wrote", args.out)
    print("missing runs:", ", ".join(summary["missing_runs"]) or "none")


if __name__ == "__main__":
    main()
