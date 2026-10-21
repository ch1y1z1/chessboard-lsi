#!/usr/bin/env python
"""Merge the two stage-2a center-axis runs into an explicit 13-point dataset.

The first run used a buggy default that emitted only 10 points (x -3..3 and
y +1..3).  Three negative-y points were then run in a separate directory.  This
script validates the union, keeps both source directories untouched, and writes
a compact merged result plus a progress summary.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


EXPECTED = ({(float(k), 0.0) for k in range(-3, 4)}
            | {(0.0, float(k)) for k in range(-3, 4) if k != 0})


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(directory: Path, rows: list[dict]) -> None:
    write_json(directory / "results.json", rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def sort_key(row: dict) -> tuple:
    if row["offset_y"] == 0:
        return (0, row["offset_x"])
    return (1, row["offset_y"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2a_center_axis_complete13"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T14:25:22+00:00")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new merged directory")
    started_at = datetime.fromisoformat(args.batch_started_at)
    now = datetime.now(timezone.utc)

    rows, sources = [], []
    for run in args.run:
        run_manifest = json.loads((run / "manifest.json").read_text())
        run_rows = json.loads((run / "results.json").read_text())
        for row in run_rows:
            row["source_run"] = str(run)
            rows.append(row)
        sources.append({
            "directory": str(run),
            "status": run_manifest.get("status"),
            "row_count": len(run_rows),
            "planned_points": (run_manifest.get("protocol") or {}).get("offsets_planned"),
            "manifest_sha256": file_sha256(run / "manifest.json"),
            "results_sha256": file_sha256(run / "results.json"),
            "elapsed_seconds": run_manifest.get("seconds"),
            "peak_memory_mib": run_manifest.get("peak_memory_mib"),
        })
    rows.sort(key=sort_key)
    offsets = {(row["offset_x"], row["offset_y"]) for row in rows}
    if len(rows) != 13 or offsets != EXPECTED:
        raise ValueError(f"expected 13 unique points, found {len(rows)} rows / {len(offsets)} offsets")
    failures = [row for row in rows if row.get("status") != "ok"]
    best = min((row for row in rows if row.get("status") == "ok"), key=lambda row: row["rmse_waves"])
    nominal = next(row for row in rows if row["offset_x"] == 0 and row["offset_y"] == 0)

    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output, rows)
    arrays = {}
    for run in args.run:
        with np.load(run / "center_axis_points.npz") as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    order = np.argsort([sort_key(row) for row in rows], axis=0)
    # Rows were already sorted before concatenation; concatenate in run order,
    # then reorder the saved point arrays with the same permutation.
    combined = {}
    for key, values in arrays.items():
        stacked = np.concatenate(values, axis=0)
        combined[key] = stacked
    # Build the permutation from the unsorted concatenated offsets.
    unsorted_offsets = combined["offsets"]
    key_to_index = {(float(x), float(y)): i for i, (x, y) in enumerate(unsorted_offsets)}
    permutation = [key_to_index[(row["offset_x"], row["offset_y"])] for row in rows]
    np.savez_compressed(args.output / "center_axis_points.npz",
                        **{key: value[permutation] for key, value in combined.items()})

    main_manifest = json.loads((args.run[0] / "manifest.json").read_text())
    all_ok = not failures
    manifest = {
        "stage": "stage2a_center_axis_complete13",
        "status": "complete" if all_ok else "complete_with_failures",
        "created_at": now.isoformat(timespec="seconds"),
        "protocol": {
            "fixed_half_width": 20,
            "fixed_nominal_calibration": True,
            "recalibrate_per_center": False,
            "direct_refit": False,
            "observation": "canonical baseline_observation.npz",
            "test_truth_usage": "sensitivity analysis only; no production parameter selected",
            "offsets_expected": [[x, y] for x, y in sorted(EXPECTED, key=lambda p: (0, p[0]) if p[1] == 0 else (1, p[1]))],
        },
        "sources": sources,
        "execution_issue": {
            "description": ("The first generator default emitted 10 points (x -3..3 and y +1..3); "
                            "y -1..-3 were missing. The three missing points were run in a separate "
                            "supplement directory and merged here. Original run directories were not modified."),
            "merged_from": [str(run) for run in args.run],
        },
        "row_count": len(rows),
        "completed_points": len(rows) - len(failures),
        "failed_points": len(failures),
        "failures": failures,
        "best_point_on_this_test_image": best,
        "nominal_point": nominal,
        "axis_summary": {
            "x": [{"offset": row["offset_x"], "rmse_waves": row["rmse_waves"]}
                  for row in rows if row["offset_y"] == 0],
            "y": [{"offset": row["offset_y"], "rmse_waves": row["rmse_waves"]}
                  for row in rows if row["offset_x"] == 0 and row["offset_y"] != 0],
        },
        "observation_and_calibration": main_manifest.get("source"),
        "code_state": json.loads((args.run[0] / "code_state.json").read_text()),
        "large_arrays_copied": False,
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "merged_at": now.isoformat(timespec="seconds"),
        "batch_wall_seconds": (now - started_at).total_seconds(),
        "resource_policy": {"workers": 1, "batch_sources": 1, "single_task": True},
    }
    write_json(args.output / "manifest.json", manifest)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "2a_center_axis",
        "status": manifest["status"],
        "batch_started_at": manifest["batch_started_at"],
        "merged_at": manifest["merged_at"],
        "batch_wall_seconds": manifest["batch_wall_seconds"],
        "planned_points": 13,
        "completed_points": manifest["completed_points"],
        "failed_points": manifest["failed_points"],
        "row_count": len(rows),
        "best_point_on_this_test_image": best,
        "nominal_point": nominal,
        "axis_summary": manifest["axis_summary"],
        "observation_and_calibration": main_manifest.get("source"),
        "source_runs": sources,
        "output_directory": str(args.output),
        "notes": [
            "Only 13 single-axis points were run; no 7x7 fill or other scan was started.",
            "Nominal half-width 20 and nominal calibration were frozen; no per-centre recalibration.",
            "The canonical baseline observation was used so hashes match the existing baseline.",
            "The first generator default omitted y -1..-3; those were run separately and merged.",
            "Test-image results are sensitivity analysis, not a production selection rule.",
        ],
    }
    write_json(docs / "stage2a-summary.json", summary)

    x_rows = manifest["axis_summary"]["x"]
    y_rows = manifest["axis_summary"]["y"]
    lines = [
        "# 阶段 2a 进度：窗中心单轴扫描（13 点）",
        "",
        f"- 本批时间：{manifest['batch_started_at']} → {manifest['merged_at']}（约 {manifest['batch_wall_seconds']:.0f} s）",
        f"- 状态：`{manifest['status']}`；完成 {manifest['completed_points']}/13 点，失败 {manifest['failed_points']} 点。",
        f"- 输出：`{args.output}`",
        "",
        "## 协议",
        "- 原采样 1200/2400、全部 64 源、0.1 waves、零角度、无噪声；观测为交接基准 `baseline_observation.npz`，文件/数组哈希记录在 manifest。",
        "- 固定半宽 20、中心偏移 0 训练的名义标定；固定平滑、掩膜、整数移频与拟合参数；未逐中心重标定，未重跑直接法。",
        "- 13 点：x/y 各 -3…3，原点只一份。",
        "",
        "## 关键数值（仅该测试图）",
        "| x 偏移 | RMSE / waves |",
        "|---:|---:|",
    ]
    lines += [f"| {item['offset']:g} | {item['rmse_waves']:.6g} |" for item in x_rows]
    lines += ["", "| y 偏移 | RMSE / waves |", "|---:|---:|"]
    lines += [f"| {item['offset']:g} | {item['rmse_waves']:.6g} |" for item in y_rows]
    lines += [
        "",
        f"- 名义点 (0,0)：`{nominal['rmse_waves']:.6g}` waves。",
        f"- 本测试图最低点：`{best['case']}`（x={best['offset_x']:g}, y={best['offset_y']:g}），`{best['rmse_waves']:.6g}` waves。",
        "- x 负偏移误差明显升高（-3 约 `1.845e-3`）；x 正偏移略低于名义点；y 轴在约 `9.21e-4`–`9.31e-4` 之间，变化较小。",
        "- 这是单轴敏感性结果，不足以判断 7×7 热图形状，也不能据此设定生产中心。",
        "",
        "## 异常与来源",
        "- 首次生成脚本默认只输出 10 点（漏 y=-1…-3）；缺失点已在新目录补算，原始 10 点目录未改动，合并 manifest 记录该问题。",
        "- 本批无失败点；若后续点失败，results.json 会保留 failed 行。",
        "",
        "## 下一批",
        "补 7×7 其余 36 点；按单点约 9–10 秒估算，约 6–7 分钟（不含启动/汇总），可作为一个 5–10 分钟批次。",
        "",
    ]
    (docs / "stage2a-progress.md").write_text("\n".join(lines))
    print("best", best["case"], best["rmse_waves"])
    print("rows", len(rows), "failures", len(failures))
    print("wrote", args.output, "and", docs / "stage2a-progress.md")


if __name__ == "__main__":
    main()
