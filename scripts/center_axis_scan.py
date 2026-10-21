#!/usr/bin/env python
"""Stage 2a: single-axis FFT-window centre scan at fixed nominal calibration.

Thirteen unique points: (k, 0) and (0, k) for k = -3..3, origin once.  The
canonical baseline observation and the saved nominal calibration (half-width
20, centre offset 0) are reused; no 64-source propagation, no recalibration
and no direct-fit rerun.  Each point writes a compact row immediately.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import platform
import resource
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from lsi.config import DiffractionConfig
from lsi.fourier import FourierCalibration, FourierExtractor, FourierSettings
from lsi.metrics import wavefront_metrics
from lsi.propagation import DiffractionModel


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array) -> str:
    array = np.ascontiguousarray(array)
    return sha256_bytes(array.tobytes())


def peak_memory_mib() -> float:
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / (1024**2 if platform.system() == "Darwin" else 1024))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_rows(directory: Path, rows: list[dict]) -> None:
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "results.json")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    temporary.replace(directory / "results.csv")


def atomic_points(path: Path, rows: list[dict]) -> None:
    def vector(key, length, default=np.nan):
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, list) and len(value) == length:
                values.append(value)
            else:
                values.append([default] * length)
        return np.asarray(values, float)

    offsets = np.asarray([[row["offset_x"], row["offset_y"]] for row in rows], float)
    centers = []
    weights, enbw = [], []
    for row in rows:
        centers.append(row.get("centers") or [[np.nan, np.nan], [np.nan, np.nan]])
        metrics = row.get("window_metrics") or [{}, {}]
        weights.append([item.get("weight_sum", np.nan) for item in metrics])
        enbw.append([item.get("equivalent_noise_bins", np.nan) for item in metrics])
    np.savez_compressed(
        path,
        offsets=offsets,
        coeffs_waves=vector("coeffs_waves", 4),
        centers=np.asarray(centers, float),
        rmse_waves=np.asarray([row.get("rmse_waves", np.nan) for row in rows], float),
        pv_waves=np.asarray([row.get("pv_waves", np.nan) for row in rows], float),
        coverage=np.asarray([row.get("coverage", np.nan) for row in rows], float),
        phase_residual_rms=np.asarray([row.get("phase_residual_rms", np.nan) for row in rows], float),
        window_weight=np.asarray(weights, float),
        window_enbw=np.asarray(enbw, float),
        status=np.asarray([row["status"] for row in rows]),
    )


def git_output(arguments: list[str]) -> str:
    environment = dict(os.environ)
    environment.pop("GIT_CONFIG_COUNT", None)
    try:
        return subprocess.check_output(["git", *arguments], text=True, env=environment,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--frozen-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_frozen"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2a_center_axis"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--offsets", nargs=2, type=float, action="append", metavar=("X", "Y"),
                        help="run only the supplied centre offsets; repeat for multiple points")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new stage-2a directory")

    baseline = args.baseline
    manifest_source = json.loads((baseline / "manifest.json").read_text())
    calibration_meta = json.loads((baseline / "baseline_calibration.json").read_text())
    calibration_npz = np.load(baseline / "baseline_calibration.npz")
    observation_npz = np.load(baseline / "baseline_observation.npz")
    baseline_rows = json.loads((baseline / "results.json").read_text())
    direct_row = next(row for row in baseline_rows if row.get("method") == "direct")
    config = DiffractionConfig(**manifest_source["config"])
    indices = tuple(int(value) for value in manifest_source["indices"])
    truth = np.asarray(observation_npz["truth"], float)
    if not np.array_equal(truth, np.asarray(manifest_source["truth_coefficients_waves"], float)):
        raise ValueError("observation truth and manifest truth differ")
    settings = FourierSettings(**calibration_meta["settings"])
    if float(settings.half_width) != 20 or settings.offset_x != 0 or settings.offset_y != 0:
        raise ValueError("canonical calibration is not the nominal half-width-20 / zero-offset setting")
    if settings.center_mode != "fixed":
        raise ValueError("canonical calibration centre mode is not fixed")

    frozen_observation = np.load(args.frozen_run / "regions_observation.npz")
    frozen_difference = float(np.max(np.abs(np.asarray(frozen_observation["observed"], float)
                                             - np.asarray(observation_npz["observed"], float))))

    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot"
    (snapshot / "lsi").mkdir(parents=True)
    for source in Path("lsi").glob("*.py"):
        shutil.copy2(source, snapshot / "lsi" / source.name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    (snapshot / "tracked.diff").write_text(git_output(["diff", "--", "lsi/study.py"]) + "\n")
    code_state = {
        "branch": git_output(["branch", "--show-current"]),
        "head": git_output(["rev-parse", "HEAD"]),
        "status_short": git_output(["status", "--short"]),
        "source_sha256": {path.name: file_sha256(path) for path in Path("lsi").glob("*.py")},
        "tracked_diff": "lsi/study.py",
    }
    atomic_json(args.output / "code_state.json", code_state)

    if args.offsets:
        points = [(float(x), float(y)) for x, y in args.offsets]
    else:
        points = [(float(k), 0.0) for k in range(-3, 4)]
        points += [(0.0, float(k)) for k in range(-3, 4) if k != 0]
    if len(set(points)) != len(points):
        raise ValueError("centre offsets contain duplicates")

    manifest = {
        "stage": "stage2a_center_axis",
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "command": ("uv run --locked python scripts/center_axis_scan.py "
                    f"--baseline {baseline} --frozen-run {args.frozen_run} --output {args.output}"),
        "protocol": {
            "profile": "reference",
            "truth_rms_waves": 0.1,
            "angle_deg": 0.0,
            "noise": None,
            "sources": 64,
            "observation": "canonical baseline_observation.npz",
            "calibration": "saved nominal baseline_calibration.npz (half-width 20, offset 0)",
            "window_half_width": 20,
            "window_shape": "rect",
            "window_sigma": settings.window_sigma,
            "phase_sigma": settings.phase_sigma,
            "recenter_adjust": [settings.recenter_adjust_x, settings.recenter_adjust_y],
            "erosion": settings.erosion,
            "amplitude_threshold": settings.amplitude_threshold,
            "regression": settings.regression,
            "offsets_planned": len(points),
            "offset_source": ("custom --offsets" if args.offsets else
                              "default 13-point single-axis set"),
            "offsets": [[x, y] for x, y in points],
            "recalibrate_per_center": False,
            "direct_refit": False,
            "test_truth_usage": "evaluation only; no production parameter is selected from the test image",
        },
        "config": manifest_source["config"],
        "indices": list(indices),
        "truth_coefficients_waves": truth.tolist(),
        "source": {
            "baseline_directory": str(baseline),
            "baseline_observation": {
                "path": str(baseline / "baseline_observation.npz"),
                "file_sha256": file_sha256(baseline / "baseline_observation.npz"),
                "observed_array_sha256": array_sha256(observation_npz["observed"]),
                "raw_intensity_array_sha256": array_sha256(observation_npz["raw_intensity"]),
                "truth_array_sha256": array_sha256(observation_npz["truth"]),
            },
            "baseline_calibration": {
                "path": str(baseline / "baseline_calibration.npz"),
                "file_sha256": file_sha256(baseline / "baseline_calibration.npz"),
                "json_path": str(baseline / "baseline_calibration.json"),
                "json_sha256": file_sha256(baseline / "baseline_calibration.json"),
                "settings": calibration_meta["settings"],
                "condition": calibration_meta.get("condition"),
                "held_out_mixed_validation": calibration_meta.get("held_out_mixed_validation"),
            },
            "baseline_results": {
                "path": str(baseline / "results.json"),
                "file_sha256": file_sha256(baseline / "results.json"),
            },
            "frozen_run_comparison": {
                "path": str(args.frozen_run / "regions_observation.npz"),
                "max_abs_observed_difference": frozen_difference,
                "used_for_this_scan": False,
                "reason": "canonical baseline observation is used so its file/array hashes match the existing baseline exactly",
            },
            "direct_reference": {
                "reused": True,
                "source": str(baseline / "results.json"),
                "row": direct_row,
                "note": "same canonical observation; not refitted in this batch",
            },
            "large_arrays_copied": False,
        },
        "resource_policy": {"workers": 1, "batch_sources": 1, "single_task": True},
        "model_construction_seconds": None,
        "completed_points": 0,
        "failed_points": 0,
        "rows": 0,
        "peak_memory_mib": None,
        "seconds": None,
        "output_directory": str(args.output),
    }
    atomic_json(args.output / "manifest.json", manifest)

    model_started = perf_counter()
    model = DiffractionModel(config, indices)
    manifest["model_construction_seconds"] = perf_counter() - model_started
    calibration = FourierCalibration(
        baseline=calibration_npz["baseline"],
        response=calibration_npz["response"],
        amplitude=float(calibration_npz["amplitude"]),
        indices=tuple(int(value) for value in calibration_npz["indices"]),
        settings=calibration_meta["settings"],
        config=manifest_source["config"],
        reference_image=calibration_npz["reference_image"],
    )
    image = np.asarray(observation_npz["observed"], float)
    rows: list[dict] = []
    for offset_x, offset_y in points:
        case = (f"center_x_{offset_x:g}" if offset_y == 0 else f"center_y_{offset_y:g}")
        settings_point = FourierSettings(**{**calibration_meta["settings"],
                                            "offset_x": offset_x, "offset_y": offset_y})
        point_started = perf_counter()
        try:
            extractor = FourierExtractor(config, settings_point)
            result = calibration.reconstruct(extractor, image, details=False)
            metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
            row = {
                "case": case,
                "method": "fourier_calibrated",
                "status": "ok",
                "offset_x": offset_x,
                "offset_y": offset_y,
                "settings": settings_point.to_dict(),
                "calibration_policy": "frozen_nominal",
                "centers": result["centers"],
                "coverage": result["coverage"],
                "phase_residual_rms": result["phase_residual_rms"],
                "window_metrics": result["window_metrics"],
                "calibration_condition": result["calibration_condition"],
                "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(),
                **metrics,
            }
        except Exception as error:  # keep failures, do not discard them
            row = {
                "case": case,
                "method": "fourier_calibrated",
                "status": "failed",
                "offset_x": offset_x,
                "offset_y": offset_y,
                "settings": settings_point.to_dict(),
                "calibration_policy": "frozen_nominal",
                "reason": f"{type(error).__name__}: {error}",
            }
        row["seconds"] = perf_counter() - point_started
        rows.append(row)
        atomic_rows(args.output, rows)
        atomic_points(args.output / "center_axis_points.npz", rows)
        manifest.update(
            completed_points=sum(item["status"] == "ok" for item in rows),
            failed_points=sum(item["status"] != "ok" for item in rows),
            rows=len(rows),
            peak_memory_mib=peak_memory_mib(),
            seconds=perf_counter() - started,
        )
        atomic_json(args.output / "manifest.json", manifest)
        print(f"{case} status={row['status']} rmse={row.get('rmse_waves')} "
              f"coverage={row.get('coverage')} seconds={row['seconds']:.2f}", flush=True)

    status = "complete" if manifest["failed_points"] == 0 else "complete_with_failures"
    manifest.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    seconds=perf_counter() - started, peak_memory_mib=peak_memory_mib())
    atomic_json(args.output / "manifest.json", manifest)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if row["status"] == "ok"]
    best = min(successful, key=lambda row: row["rmse_waves"]) if successful else None
    summary = {
        "stage": "2a_center_axis",
        "status": status,
        "started_at": manifest["started_at"],
        "finished_at": manifest["finished_at"],
        "wall_seconds": manifest["seconds"],
        "peak_memory_mib": manifest["peak_memory_mib"],
        "planned_points": len(points),
        "completed_points": manifest["completed_points"],
        "failed_points": manifest["failed_points"],
        "row_count": len(rows),
        "best_point_on_this_test_image": best,
        "direct_reference": direct_row,
        "source": manifest["source"],
        "output_directory": str(args.output),
        "notes": [
            "Only 13 single-axis points were run; no 7x7 fill or other scan was started.",
            "Nominal half-width 20 and nominal calibration were frozen; offsets were not recalibrated.",
            "The canonical baseline observation was used so its hashes match the existing baseline.",
            "Test-image results are a sensitivity analysis, not a production selection rule.",
        ],
    }
    atomic_json(docs / "stage2a-summary.json", summary)
    progress = f"""# 阶段 2a 进度：窗中心单轴扫描

- 时间：{manifest['started_at']} → {manifest['finished_at']}（UTC）
- 状态：`{status}`；完成 {manifest['completed_points']}/{len(points)} 点，失败 {manifest['failed_points']} 点。
- 墙钟：{manifest['seconds']:.1f} s；峰值 RSS：{manifest['peak_memory_mib']:.0f} MiB；worker/batch=1。
- 输出：`{args.output}`

## 协议
- 原采样 1200/2400、全部 64 源、0.1 waves、零角度、无噪声；观测为交接基准 `baseline_observation.npz`（文件与数组哈希已记录）。
- 固定半宽 20、中心偏移 0 训练的名义标定；固定平滑/掩膜/整数移频/拟合参数。
- 13 个点：x/y 各 -3…3，原点只算一次；未逐中心重标定，未重跑直接法。

## 关键数值
- 本测试图最低点：{best['case'] if best else '无'}，RMSE {best['rmse_waves']:.6g} waves（offset x={best['offset_x']:g}, y={best['offset_y']:g}）。
- 直接法参考线（复用）：{direct_row['rmse_waves']:.6g} waves。
- 名义点（0,0）：{next((row['rmse_waves'] for row in successful if row['offset_x']==0 and row['offset_y']==0), None)} waves。

## 限制
- 只覆盖单轴 13 点，不足以判断二维热图整体形状。
- 结果仅用于该测试图的敏感性分析；不能据此设定生产窗中心。
- 冻结与逐窗重标定是不同协议，本批不比较重标定。
"""
    (docs / "stage2a-progress.md").write_text(progress)
    print("wrote", args.output, "and", docs / "stage2a-summary.json")


if __name__ == "__main__":
    main()
