#!/usr/bin/env python
"""Stage 2b: fill the missing 36 points of the 7x7 FFT-window-centre grid.

The existing 13-point single-axis dataset is verified and reused.  Only the
difference set (x != 0 and y != 0) is computed, then the union is asserted to
be the complete 49-point grid.  The canonical baseline observation and the
saved nominal half-width-20 calibration are reused; no propagation,
recalibration or direct refit is performed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

COORDINATES = range(-3, 4)


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def peak_memory_mib() -> float:
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / (1024**2 if platform.system() == "Darwin" else 1024))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(directory: Path, rows: list[dict], stem: str = "results") -> None:
    atomic_json(directory / f"{stem}.json", rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / f"{stem}.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    temporary.replace(directory / f"{stem}.csv")


def points_arrays(rows: list[dict]) -> dict:
    offsets = np.asarray([[row["offset_x"], row["offset_y"]] for row in rows], float)
    coeffs = np.asarray([row.get("coeffs_waves", [np.nan] * 4) for row in rows], float)
    centers = np.asarray([row.get("centers") or [[np.nan, np.nan], [np.nan, np.nan]]
                          for row in rows], float)
    weights, enbw = [], []
    for row in rows:
        metrics = row.get("window_metrics") or [{}, {}]
        weights.append([item.get("weight_sum", np.nan) for item in metrics])
        enbw.append([item.get("equivalent_noise_bins", np.nan) for item in metrics])
    return {
        "offsets": offsets, "coeffs_waves": coeffs, "centers": centers,
        "rmse_waves": np.asarray([row.get("rmse_waves", np.nan) for row in rows], float),
        "pv_waves": np.asarray([row.get("pv_waves", np.nan) for row in rows], float),
        "coverage": np.asarray([row.get("coverage", np.nan) for row in rows], float),
        "phase_residual_rms": np.asarray([row.get("phase_residual_rms", np.nan) for row in rows], float),
        "window_weight": np.asarray(weights, float),
        "window_enbw": np.asarray(enbw, float),
        "status": np.asarray([row["status"] for row in rows]),
    }


def save_points(path: Path, rows: list[dict]) -> None:
    np.savez_compressed(path, **points_arrays(rows))


def git_output(arguments: list[str]) -> str:
    environment = dict(os.environ)
    environment.pop("GIT_CONFIG_COUNT", None)
    try:
        return subprocess.check_output(["git", *arguments], text=True, env=environment,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def grid_matrix(rows: list[dict], field: str) -> np.ndarray:
    matrix = np.full((7, 7), np.nan)
    for row in rows:
        x, y = int(row["offset_x"]), int(row["offset_y"])
        matrix[y + 3, x + 3] = row.get(field, np.nan)
    return matrix


def grid_list(matrix: np.ndarray) -> list[list[float | None]]:
    return [[None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix]


def isolated_drop(rows: list[dict]) -> dict | None:
    """Find an isolated local improvement relative to its in-grid neighbours."""
    matrix = grid_matrix(rows, "rmse_waves")
    candidate = None
    for y in COORDINATES:
        for x in COORDINATES:
            value = matrix[y + 3, x + 3]
            if not np.isfinite(value):
                continue
            neighbours = [matrix[y + 3 + dy, x + 3 + dx]
                          for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if -3 <= x + dx <= 3 and -3 <= y + dy <= 3
                          and np.isfinite(matrix[y + 3 + dy, x + 3 + dx])]
            if len(neighbours) < 2:
                continue
            median = float(np.median(neighbours))
            delta = median - value
            if delta > 3e-4 and (candidate is None or delta > candidate["neighbour_median_minus_point_waves"]):
                candidate = {"offset_x": float(x), "offset_y": float(y),
                             "rmse_waves": float(value), "neighbour_median_waves": median,
                             "neighbour_median_minus_point_waves": float(delta)}
    return candidate


def plot_heatmap(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = grid_matrix(rows, "rmse_waves")
    successful = [row for row in rows if row["status"] == "ok"]
    best = min(successful, key=lambda row: row["rmse_waves"])
    nominal = next(row for row in successful
                   if row["offset_x"] == 0 and row["offset_y"] == 0)
    anomaly = isolated_drop(rows)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    image = ax.imshow(matrix, origin="lower", extent=(-3.5, 3.5, -3.5, 3.5),
                      cmap="viridis", aspect="equal")
    for y in COORDINATES:
        for x in COORDINATES:
            value = matrix[y + 3, x + 3]
            if np.isfinite(value):
                text_x, text_y = x, y
                if x == 0 and y == 0:
                    text_x, text_y = 0.40, -0.36
                elif anomaly and x == anomaly["offset_x"] and y == anomaly["offset_y"]:
                    text_x, text_y = x + 0.42, y - 0.34
                ax.text(text_x, text_y, f"{value * 1e4:.2f}", ha="center", va="center",
                        color="white" if value < np.nanmedian(matrix) else "black", fontsize=7)
    ax.plot(nominal["offset_x"], nominal["offset_y"], "o", mfc="none", mec="white",
            ms=13, mew=2, label="nominal (0,0)")
    ax.plot(best["offset_x"], best["offset_y"], "*", color="red", ms=15,
            label="post-hoc test minimum")
    if anomaly:
        ax.plot(anomaly["offset_x"], anomaly["offset_y"], "D", color="black", ms=8,
                mfc="none", mew=1.5, label="isolated local drop")
    ax.set_xticks(list(COORDINATES))
    ax.set_yticks(list(COORDINATES))
    ax.set_xlabel("window centre offset x / bins")
    ax.set_ylabel("window centre offset y / bins")
    ax.set_title("Fourier wavefront RMSE / 1e-4 waves\n(fixed nominal calibration, test image)")
    fig.colorbar(image, ax=ax, label="RMSE / waves")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 0.075))
    fig.text(0.5, 0.012,
             "Red star: post-hoc test-image minimum (not independently selected for production). "
             "Black diamond: isolated local drop retained for review.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.13, 1, 0.90))
    for suffix in ("png", "svg"):
        fig.savefig(path / f"center_grid_heatmap.{suffix}", dpi=170)
    plt.close(fig)


def plot_profiles(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_rows = sorted((row for row in rows if row["offset_y"] == 0),
                    key=lambda row: row["offset_x"])
    y_rows = sorted((row for row in rows if row["offset_x"] == 0 and row["offset_y"] != 0),
                    key=lambda row: row["offset_y"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([row["offset_x"] for row in x_rows], [row["rmse_waves"] for row in x_rows],
            "o-", label="x profile (y=0)")
    ax.plot([row["offset_y"] for row in y_rows], [row["rmse_waves"] for row in y_rows],
            "s--", label="y profile (x=0)")
    nominal = next(row for row in rows if row["offset_x"] == 0 and row["offset_y"] == 0)
    ax.axhline(nominal["rmse_waves"], color="k", ls=":", lw=1, alpha=.6,
               label="nominal (0,0)")
    ax.axvline(0, color="k", ls=":", lw=1, alpha=.6)
    ax.set_xlabel("single-axis centre offset / bins")
    ax.set_ylabel("Wavefront RMSE / waves")
    ax.set_title("Single-axis centre sensitivity (test image; frozen nominal calibration)")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"center_axis_profiles.{suffix}", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--existing-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2a_center_axis_complete13"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2b_center_grid_49"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T14:31:56+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    process_started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new stage-2b directory")

    baseline = args.baseline
    base_manifest = json.loads((baseline / "manifest.json").read_text())
    base_calibration = json.loads((baseline / "baseline_calibration.json").read_text())
    base_cal_npz = np.load(baseline / "baseline_calibration.npz")
    observation_npz = np.load(baseline / "baseline_observation.npz")
    baseline_rows = json.loads((baseline / "results.json").read_text())
    direct_row = next(row for row in baseline_rows if row.get("method") == "direct")
    base_settings = FourierSettings(**base_calibration["settings"])
    config = DiffractionConfig(**base_manifest["config"])
    indices = tuple(int(value) for value in base_manifest["indices"])
    truth = np.asarray(observation_npz["truth"], float)
    if not np.array_equal(truth, np.asarray(base_manifest["truth_coefficients_waves"], float)):
        raise ValueError("observation truth differs from canonical baseline truth")

    existing_manifest = json.loads((args.existing_run / "manifest.json").read_text())
    existing_rows = json.loads((args.existing_run / "results.json").read_text())
    existing_set = {(float(row["offset_x"]), float(row["offset_y"])) for row in existing_rows}
    expected = {(float(x), float(y)) for x in COORDINATES for y in COORDINATES}
    missing = expected - existing_set
    if len(expected) != 49 or len(existing_set) != 13 or len(missing) != 36:
        raise ValueError(f"grid set mismatch: expected 49, existing {len(existing_set)}, missing {len(missing)}")
    if not all(-3 <= x <= 3 and -3 <= y <= 3 for x, y in existing_set | missing):
        raise ValueError("coordinate outside -3..3")
    if not all(x != 0 and y != 0 for x, y in missing):
        raise ValueError("missing set is not exactly the off-axis 36 points")
    if any(row.get("status") != "ok" for row in existing_rows):
        raise ValueError("existing 13-point run contains a non-ok row")
    for row in existing_rows:
        settings = row.get("settings", {})
        for key in ("half_width", "aspect", "shape", "window_sigma", "phase_sigma", "regression",
                    "erosion", "amplitude_threshold", "center_mode"):
            if settings.get(key) != getattr(base_settings, key):
                raise ValueError(f"existing point {row['case']} differs in {key}")

    main_source_dir = Path(existing_manifest["sources"][0]["directory"])
    main_source_manifest = json.loads((main_source_dir / "manifest.json").read_text())
    if json.dumps(main_source_manifest["config"], sort_keys=True) != json.dumps(base_manifest["config"], sort_keys=True):
        raise ValueError("existing run config differs from canonical baseline config")
    if tuple(int(value) for value in main_source_manifest["indices"]) != indices:
        raise ValueError("existing run indices differ from canonical baseline indices")
    old_source = existing_manifest["observation_and_calibration"]
    checks = {
        "observation_file_sha256": (old_source["baseline_observation"]["file_sha256"],
                                    file_sha256(baseline / "baseline_observation.npz")),
        "observation_array_sha256": (old_source["baseline_observation"]["observed_array_sha256"],
                                     array_sha256(observation_npz["observed"])),
        "calibration_file_sha256": (old_source["baseline_calibration"]["file_sha256"],
                                    file_sha256(baseline / "baseline_calibration.npz")),
        "calibration_json_sha256": (old_source["baseline_calibration"]["json_sha256"],
                                    file_sha256(baseline / "baseline_calibration.json")),
        "baseline_results_sha256": (old_source["baseline_results"]["file_sha256"],
                                    file_sha256(baseline / "results.json")),
    }
    if any(left != right for left, right in checks.values()):
        raise ValueError(f"existing source hashes differ from canonical baseline: {checks}")
    if not np.array_equal(truth, np.asarray(base_manifest["truth_coefficients_waves"], float)):
        raise ValueError("observation truth differs from canonical baseline truth")

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

    manifest = {
        "stage": "stage2b_center_grid_49",
        "status": "running",
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "command": ("uv run --locked python scripts/center_grid_fill.py "
                    f"--baseline {baseline} --existing-run {args.existing_run} --output {args.output}"),
        "protocol": {
            "profile": "reference",
            "truth_rms_waves": 0.1,
            "angle_deg": 0.0,
            "noise": None,
            "sources": 64,
            "window_half_width": 20,
            "window_shape": "rect",
            "window_sigma": base_settings.window_sigma,
            "phase_sigma": base_settings.phase_sigma,
            "recenter_adjust": [base_settings.recenter_adjust_x, base_settings.recenter_adjust_y],
            "erosion": base_settings.erosion,
            "amplitude_threshold": base_settings.amplitude_threshold,
            "regression": base_settings.regression,
            "expected_coordinates": 49,
            "existing_coordinates": 13,
            "new_coordinates": 36,
            "recalibrate_per_center": False,
            "direct_refit": False,
            "test_truth_usage": "sensitivity/post-hoc only; no production centre selected",
        },
        "config": base_manifest["config"],
        "indices": list(indices),
        "truth_coefficients_waves": truth.tolist(),
        "source": {
            "baseline_directory": str(baseline),
            "existing_run": str(args.existing_run),
            "existing_manifest_sha256": file_sha256(args.existing_run / "manifest.json"),
            "existing_results_sha256": file_sha256(args.existing_run / "results.json"),
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
                "settings": base_calibration["settings"],
                "condition": base_calibration.get("condition"),
                "held_out_mixed_validation": base_calibration.get("held_out_mixed_validation"),
            },
            "baseline_results": {
                "path": str(baseline / "results.json"),
                "file_sha256": file_sha256(baseline / "results.json"),
            },
            "reused_direct": {
                "reused": True,
                "source": str(baseline / "results.json"),
                "row": direct_row,
                "note": "same canonical observation; not refitted in this batch",
            },
            "existing_hash_checks": {key: {"existing": left, "current": right}
                                     for key, (left, right) in checks.items()},
            "large_arrays_copied": False,
        },
        "assertions": {
            "expected_49": True,
            "existing_13_unique": len(existing_set) == 13,
            "missing_36": len(missing) == 36,
            "missing_all_both_nonzero": all(x != 0 and y != 0 for x, y in missing),
        },
        "resource_policy": {"workers": 1, "batch_sources": 1, "single_task": True},
        "model_construction_seconds": None,
        "completed_new_points": 0,
        "failed_new_points": 0,
        "peak_memory_mib": None,
        "seconds": None,
    }
    atomic_json(args.output / "manifest.json", manifest)

    model_started = perf_counter()
    model = DiffractionModel(config, indices)
    manifest["model_construction_seconds"] = perf_counter() - model_started
    calibration = FourierCalibration(
        baseline=base_cal_npz["baseline"], response=base_cal_npz["response"],
        amplitude=float(base_cal_npz["amplitude"]),
        indices=tuple(int(value) for value in base_cal_npz["indices"]),
        settings=base_calibration["settings"], config=base_manifest["config"],
        reference_image=base_cal_npz["reference_image"])
    image = np.asarray(observation_npz["observed"], float)
    new_rows: list[dict] = []
    for x, y in sorted(missing, key=lambda point: (point[0], point[1])):
        case = f"grid_x_{x:g}_y_{y:g}"
        settings_point = replace(base_settings, offset_x=x, offset_y=y)
        point_started = perf_counter()
        try:
            result = calibration.reconstruct(FourierExtractor(config, settings_point), image,
                                             details=False)
            metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
            row = {
                "case": case, "method": "fourier_calibrated", "status": "ok",
                "offset_x": x, "offset_y": y, "settings": settings_point.to_dict(),
                "calibration_policy": "frozen_nominal", "centers": result["centers"],
                "coverage": result["coverage"], "phase_residual_rms": result["phase_residual_rms"],
                "window_metrics": result["window_metrics"],
                "calibration_condition": result["calibration_condition"],
                "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(), **metrics,
            }
        except Exception as error:
            row = {
                "case": case, "method": "fourier_calibrated", "status": "failed",
                "offset_x": x, "offset_y": y, "settings": settings_point.to_dict(),
                "calibration_policy": "frozen_nominal",
                "reason": f"{type(error).__name__}: {error}",
            }
        row["seconds"] = perf_counter() - point_started
        new_rows.append(row)
        write_rows(args.output, new_rows)
        save_points(args.output / "center_grid_points_36.npz", new_rows)
        manifest.update(completed_new_points=sum(item["status"] == "ok" for item in new_rows),
                        failed_new_points=sum(item["status"] != "ok" for item in new_rows),
                        peak_memory_mib=peak_memory_mib(),
                        seconds=perf_counter() - process_started)
        atomic_json(args.output / "manifest.json", manifest)
        print(f"{case} status={row['status']} rmse={row.get('rmse_waves')} "
              f"seconds={row['seconds']:.2f}", flush=True)

    all_rows = existing_rows + new_rows
    all_set = {(float(row["offset_x"]), float(row["offset_y"])) for row in all_rows}
    if len(all_rows) != 49 or len(all_set) != 49 or all_set != expected:
        raise ValueError(f"merged grid is incomplete/duplicated: rows {len(all_rows)}, unique {len(all_set)}")
    all_rows.sort(key=lambda row: (int(row["offset_y"]), int(row["offset_x"])))
    write_rows(args.output, all_rows)
    write_rows(args.output, new_rows, stem="results_36_new")
    write_rows(args.output, all_rows, stem="results_49")
    save_points(args.output / "center_grid_points_49.npz", all_rows)
    plot_heatmap(all_rows, args.output)
    plot_profiles(all_rows, args.output)

    successful = [row for row in all_rows if row["status"] == "ok"]
    failed = [row for row in all_rows if row["status"] != "ok"]
    best = min(successful, key=lambda row: row["rmse_waves"])
    nominal = next(row for row in successful if row["offset_x"] == 0 and row["offset_y"] == 0)
    rmse_grid = grid_matrix(all_rows, "rmse_waves")
    pv_grid = grid_matrix(all_rows, "pv_waves")
    coverage_grid = grid_matrix(all_rows, "coverage")
    phase_grid = grid_matrix(all_rows, "phase_residual_rms")
    boundary = abs(int(best["offset_x"])) == 3 or abs(int(best["offset_y"])) == 3
    status = "complete" if not failed else "complete_with_failures"
    manifest.update(
        status=status,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seconds=perf_counter() - process_started,
        batch_wall_seconds=(datetime.now(timezone.utc) - started_at).total_seconds(),
        peak_memory_mib=peak_memory_mib(),
        total_points=len(all_rows), total_ok=len(successful), total_failed=len(failed),
        assertions={**manifest["assertions"], "merged_49_complete": True,
                    "merged_49_unique": len(all_set) == 49},
        best_point_on_this_test_image=best,
        nominal_point=nominal,
        boundary_minimum=boundary,
        grid_rmse_waves=grid_list(rmse_grid),
        grid_pv_waves=grid_list(pv_grid),
        grid_coverage=grid_list(coverage_grid),
        grid_phase_residual_rms=grid_list(phase_grid),
        metric_ranges={"coverage": [float(np.nanmin(coverage_grid)), float(np.nanmax(coverage_grid))],
                       "phase_residual_rms": [float(np.nanmin(phase_grid)), float(np.nanmax(phase_grid))],
                       "pv_waves": [float(np.nanmin(pv_grid)), float(np.nanmax(pv_grid))],
                       "rmse_waves": [float(np.nanmin(rmse_grid)), float(np.nanmax(rmse_grid))]},
    )
    atomic_json(args.output / "manifest.json", manifest)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "2b_center_grid_49",
        "status": status,
        "batch_started_at": manifest["batch_started_at"],
        "finished_at": manifest["finished_at"],
        "batch_wall_seconds": manifest["batch_wall_seconds"],
        "planned_new_points": 36,
        "completed_new_points": manifest["completed_new_points"],
        "failed_new_points": manifest["failed_new_points"],
        "total_points": len(all_rows),
        "total_ok": len(successful),
        "total_failed": len(failed),
        "assertions": manifest["assertions"],
        "nominal_point": nominal,
        "best_point_on_this_test_image": best,
        "boundary_minimum": boundary,
        "metric_ranges": manifest["metric_ranges"],
        "grid_rmse_waves": grid_list(rmse_grid),
        "source": manifest["source"],
        "per_new_point_seconds": {
            "mean": float(np.mean([row["seconds"] for row in new_rows])),
            "min": float(np.min([row["seconds"] for row in new_rows])),
            "max": float(np.max([row["seconds"] for row in new_rows])),
        },
        "peak_memory_mib": manifest["peak_memory_mib"],
        "output_directory": str(args.output),
        "notes": [
            "The existing 13 points were verified against the canonical baseline hashes and reused unchanged.",
            "Only the 36 off-axis points (x != 0 and y != 0) were computed in this batch.",
            "The nominal half-width-20 calibration and canonical baseline observation were frozen.",
            "The heatmap minimum is a post-hoc test-image sensitivity result, not an independently selected production centre.",
            "No 7x7 extrapolation beyond +/-3 bins is claimed.",
        ],
    }
    atomic_json(docs / "stage2b-summary.json", summary)
    lines = [
        "# 阶段 2b 进度：窗中心 7×7 网格（49 点）",
        "",
        f"- 本批时间：{manifest['batch_started_at']} → {manifest['finished_at']}（约 {manifest['batch_wall_seconds']:.0f} s）",
        f"- 状态：`{status}`；新增 {manifest['completed_new_points']}/36 点，失败 {manifest['failed_new_points']} 点；合计 {len(all_rows)}/49 点，失败 {len(failed)} 点。",
        f"- 每点（本批）：均值 {summary['per_new_point_seconds']['mean']:.2f} s，范围 {summary['per_new_point_seconds']['min']:.2f}–{summary['per_new_point_seconds']['max']:.2f} s。",
        f"- 峰值 RSS：{manifest['peak_memory_mib']:.0f} MiB；worker/batch=1。",
        f"- 输出：`{args.output}`",
        "",
        "## 集合校验",
        "- 期望 49 个唯一坐标 `(x,y)`，x/y 均取 -3…3；旧 13 点唯一且范围正确。",
        "- 差集 36 点全部满足 x≠0、y≠0；合并后 49 点无缺失、无重复。",
        "- 旧 13 点使用的观测/标定文件哈希与当前交接基准一致，配置和固定参数一致。",
        "",
        "## 关键数值（仅该测试图）",
        f"- 名义点 (0,0)：`{nominal['rmse_waves']:.6g}` waves。",
        f"- 本测试图事后最低点：`{best['case']}`（x={best['offset_x']:g}, y={best['offset_y']:g}），`{best['rmse_waves']:.6g}` waves。",
        f"- 最低点是否位于 ±3 边界：`{boundary}`；若为边界，只能说明有限范围内趋势，不能外推。",
        f"- coverage 范围：`{manifest['metric_ranges']['coverage'][0]:.6g}`–`{manifest['metric_ranges']['coverage'][1]:.6g}`。",
        f"- phase residual 范围：`{manifest['metric_ranges']['phase_residual_rms'][0]:.6g}`–`{manifest['metric_ranges']['phase_residual_rms'][1]:.6g}`。",
        f"- PV 范围：`{manifest['metric_ranges']['pv_waves'][0]:.6g}`–`{manifest['metric_ranges']['pv_waves'][1]:.6g}` waves。",
        "",
        "## 图",
        "- `center_grid_heatmap.png/svg`：7×7 RMSE，标出名义中心与事后最低点。",
        "- `center_axis_profiles.png/svg`：x/y 单轴曲线。",
        "",
        "## 限制",
        "- 固定名义半宽 20 与固定标定；本批不比较逐中心重标定。",
        "- 49 点仅覆盖 ±3 bin 的整数偏移；不能外推更大范围或连续最优。",
        "- 测试真值最低点只是事后敏感性，不是独立验证选出的生产中心。",
        "",
        "## 下一批建议",
        "形状/长宽比有限方案：固定半宽 20、固定标定，比较 rect（基线）、circle、ellipse 0.5/2、gaussian；预计 5–7 个解算，约 2–3 分钟（不含启动/汇总）。之后再做空间掩膜腐蚀/低幅剔除。",
        "",
    ]
    (docs / "stage2b-progress.md").write_text("\n".join(lines))
    print("best", best["case"], best["rmse_waves"], "boundary", boundary)
    print("rows", len(all_rows), "failures", len(failed))
    print("wrote", args.output, "and", docs / "stage2b-summary.json")


if __name__ == "__main__":
    main()
