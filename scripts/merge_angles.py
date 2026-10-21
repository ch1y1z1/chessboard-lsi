#!/usr/bin/env python
"""Merge the angle pilot and camera/grating batches into one reviewable table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(run: Path) -> list[dict]:
    return json.loads((run / "results.json").read_text())


def plot_angles(rows: list[dict], zero_rows: list[dict], path: Path) -> None:
    zero_direct = next(row for row in zero_rows if row["method"] == "direct")
    zero_fixed = next(row for row in zero_rows
                      if row["method"] == "fourier_calibrated"
                      and (row.get("settings") or {}).get("center_mode") == "fixed")
    zero_auto = next(row for row in zero_rows
                     if row["method"] == "fourier_calibrated"
                     and (row.get("settings") or {}).get("center_mode") == "auto")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    panels = [("camera", "x"), ("camera", "y"), ("grating", "x"), ("grating", "y")]
    styles = {
        ("known", "direct", None): ("C0", "o", "-"),
        ("known", "fourier_calibrated", "fixed"): ("C0", "s", "--"),
        ("known", "fourier_calibrated", "auto"): ("C0", "^", ":"),
        ("ignored", "direct", None): ("C3", "o", "-"),
        ("ignored", "fourier_calibrated", "fixed"): ("C3", "s", "--"),
        ("ignored", "fourier_calibrated", "auto"): ("C3", "^", ":"),
    }
    for ax, (component, axis) in zip(axes.flat, panels):
        for (geometry, method, center), (color, marker, linestyle) in styles.items():
            selected = [row for row in rows
                        if row.get("component") == component and row.get("axis") == axis
                        and row.get("geometry") == geometry and row["method"] == method
                        and (row.get("settings") or {}).get("center_mode") == center]
            if not selected:
                continue
            selected.sort(key=lambda row: row["angle_deg"])
            x = [row["angle_deg"] for row in selected]
            y = [row["rmse_waves"] for row in selected]
            if method == "direct":
                zero_row = zero_direct
            else:
                zero_row = zero_auto if center == "auto" else zero_fixed
            # Insert the zero-angle baseline in numerical order before drawing
            # connecting lines; prepending it would connect 0 -> negative -> positive.
            points = sorted(zip([0.0] + x, [zero_row["rmse_waves"]] + y))
            x, y = zip(*points)
            # Exact zero has no logarithm. Mask it instead of drawing a line
            # to Matplotlib's artificial clipping floor or inventing a value.
            y = np.ma.masked_less_equal(y, 0)
            label = ("direct" if method == "direct" else f"FT {center}") + f" / {geometry}"
            ax.semilogy(x, y, marker=marker, ls=linestyle, color=color, label=label, ms=4)
        ax.set_title(f"{component} / {axis} axis")
        ax.set_xlabel("tilt angle / degree")
        ax.set_ylabel("Wavefront RMSE / waves")
        ax.grid(True, which="both", alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("Exact zero RMSE is omitted on logarithmic axes", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .97))
    for suffix in ("png", "svg"):
        fig.savefig(path / f"angle_known_ignored.{suffix}", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
                        default=Path("output/diffraction/followup_20261021/angles_pilot_camera_x_pos025"))
    parser.add_argument("--camera", type=Path,
                        default=Path("output/diffraction/followup_20261021/angles_batch_camera_pm025"))
    parser.add_argument("--grating", type=Path,
                        default=Path("output/diffraction/followup_20261021/angles_batch_grating_pm025"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage3_angles_merged"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists")
    pilot_rows = load_rows(args.pilot)
    zero_rows = [row for row in pilot_rows if row.get("component") == "zero"]
    rows = [row for row in pilot_rows if row.get("component") != "zero"]
    for run in (args.camera, args.grating):
        rows.extend(load_rows(run))
    geometries = sorted({(row["component"], row["axis"], float(row["angle_deg"]))
                         for row in rows})
    expected = {(component, axis, angle)
                for component in ("camera", "grating")
                for axis in ("x", "y")
                for angle in (-.25, .25)}
    if set(geometries) != expected:
        raise ValueError(f"angle geometry set mismatch: {geometries}")
    failures = [row for row in rows if row.get("status") != "ok"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    plot_angles(rows, zero_rows, args.output)
    summary = {
        "stage": "3_angles_merged",
        "status": "complete" if not failures else "complete_with_failures",
        "geometries": [list(item) for item in geometries],
        "row_count": len(rows),
        "zero_rows": zero_rows,
        "failed": failures,
        "sources": [str(args.pilot), str(args.camera), str(args.grating)],
        "notes": ["known uses tilted model + independent calibration",
                  "ignored uses nominal model + zero-angle calibration",
                  "fixed/auto centres use the corresponding geometry's fixed/auto calibration",
                  "direct uses zero initialisation, no truth initialisation",
                  "optimisation stop is not recovery; ignored geometries retain large errors"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage3-angle-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# 阶段 3 进度：有限角度对照", "",
             f"- 非零几何：{len(geometries)} 个（camera/grating × x/y × ±0.25°），行数 {len(rows)}，失败 {len(failures)}。",
             "- known：倾斜模型 + 对应几何 fixed/auto 独立标定；ignored：零角度模型 + 零角度 fixed/auto 标定；direct 均从零初值。",
             "- 优化停止不等于恢复成功；ignored 几何保留大的非零误差。",
             "", "| 部件 | 轴 | 角度 | 几何 | 方法 | 中心 | RMSE / waves |",
             "|---|---|---:|---|---|---|---:|"]
    for row in sorted(rows, key=lambda item: (item["component"], item["axis"], item["angle_deg"],
                                              item["geometry"], item["method"],
                                              (item.get("settings") or {}).get("center_mode", ""))):
        lines.append(f"| {row['component']} | {row['axis']} | {row['angle_deg']:g} | "
                     f"{row['geometry']} | {row['method']} | "
                     f"{(row.get('settings') or {}).get('center_mode', '—')} | {row['rmse_waves']:.6g} |")
    lines += ["", "图：`angle_known_ignored.png/svg`（log 纵轴，含零角度基线）。", ""]
    (docs / "stage3-angle-progress.md").write_text("\n".join(lines))
    print("geometries", len(geometries), "rows", len(rows), "failures", len(failures))


if __name__ == "__main__":
    main()
