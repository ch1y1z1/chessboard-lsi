#!/usr/bin/env python
"""Stage 2c: finite window-shape scan plus a two-point anomaly diagnosis.

Main scan: five unique windows at centre (0,0), nominal half-width 20, frozen
rect calibration.  The nominal rect row is reused, not recomputed.  The four
new windows are equal-area/equal-weight variants.

Diagnosis: recompute (-2,-3) and (-2,-2) with the exact saved settings and
compare reproduction, sideband wrapped phase, unwrapped phase and coefficients.
No other scans are started and no large arrays are written.
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
from scipy import ndimage
from skimage.restoration import unwrap_phase

from lsi.config import DiffractionConfig
from lsi.fourier import FourierCalibration, FourierExtractor, FourierSettings, _UNWRAP_RANDOM
from lsi.metrics import wavefront_metrics
from lsi.propagation import DiffractionModel


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def git_output(arguments: list[str]) -> str:
    environment = dict(os.environ)
    environment.pop("GIT_CONFIG_COUNT", None)
    try:
        return subprocess.check_output(["git", *arguments], text=True, env=environment,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def shape_area_definition(shape: str, aspect: float) -> str:
    if shape == "rect":
        return "rect: half-width 20 x 20; continuous area 1600 bins^2"
    if shape == "circle":
        return ("circle: half-width 20; radius 20*2/sqrt(pi)=22.568; "
                "continuous area 1600 bins^2, equal to rect")
    if shape == "ellipse":
        return (f"ellipse: aspect={aspect:g}; hx=20/sqrt({aspect:g}), hy=20*sqrt({aspect:g}); "
                "continuous area 1600 bins^2, equal to rect")
    if shape == "gaussian":
        return ("gaussian: sx=20*sqrt(2/pi), sy=20*sqrt(2/pi); continuous integrated weight "
                "1600 bins^2, equal to rect area; infinite support")
    raise ValueError(shape)


def plot_shapes(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["rect (reused)" if row.get("reused") else row["case"].replace("shape_", "") for row in rows]
    values = [row["rmse_waves"] for row in rows]
    enbw = [np.mean([m["equivalent_noise_bins"] for m in row["window_metrics"]]) for row in rows]
    dc = [max(m.get("dc_weight", 0) for m in row["window_metrics"]) for row in rows]
    colors = ["C0" if row.get("reused") else "C2" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(range(len(rows)), values, color=colors)
    axes[0].set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Wavefront RMSE / waves")
    axes[0].set_title("Five windows, fixed rect calibration")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value * 1e4:.2f}",
                     ha="center", va="bottom", fontsize=7)
    axes[0].grid(True, axis="y", alpha=.25)
    axes[1].bar(range(len(rows)), enbw, color="C0", alpha=.7, label="ENBW / bins")
    axes[1].set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    axes[1].set_ylabel("equivalent noise bandwidth / bins")
    twin = axes[1].twinx()
    twin.plot(range(len(rows)), dc, "o--", color="C3", label="max DC weight")
    twin.set_ylabel("window value at zero frequency")
    axes[1].set_title("Discrete window metrics")
    handles, labels_ = axes[1].get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    axes[1].legend(handles + h2, labels_ + l2, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"shape_comparison.{suffix}", dpi=170)
    plt.close(fig)


def phase_comparison(result_a: dict, result_b: dict, extractor, sigma: float) -> list[dict]:
    stats = []
    for k in range(2):
        mask = extractor.masks[k]
        wrapped_a = result_a["wrapped"][k]
        wrapped_b = result_b["wrapped"][k]
        amp_a = result_a["amplitudes"][k][mask]
        amp_b = result_b["amplitudes"][k][mask]
        phase_a = wrapped_a[mask]
        phase_b = wrapped_b[mask]
        wrapped_difference = phase_a - phase_b
        wrapped_difference -= np.median(wrapped_difference)
        unwrapped_a = ndimage.gaussian_filter(unwrap_phase(wrapped_a, **_UNWRAP_RANDOM), sigma)
        unwrapped_b = ndimage.gaussian_filter(unwrap_phase(wrapped_b, **_UNWRAP_RANDOM), sigma)
        unwrapped_difference = unwrapped_a[mask] - unwrapped_b[mask]
        unwrapped_difference -= np.median(unwrapped_difference)
        window_a, window_b = result_a["windows"][k], result_b["windows"][k]
        overlap = float(np.minimum(window_a, window_b).sum() / max(np.maximum(window_a, window_b).sum(), 1e-30))
        stats.append({
            "sideband": k,
            "wrapped_difference_rms_rad": float(np.sqrt(np.mean(wrapped_difference**2))),
            "wrapped_difference_max_abs_rad": float(np.max(np.abs(wrapped_difference))),
            "unwrapped_difference_rms_rad": float(np.sqrt(np.mean(unwrapped_difference**2))),
            "unwrapped_difference_max_abs_rad": float(np.max(np.abs(unwrapped_difference))),
            "unwrapped_difference_p99_abs_rad": float(np.percentile(np.abs(unwrapped_difference), 99)),
            "fraction_abs_wrapped_difference_gt_pi_over_2": float(np.mean(np.abs(wrapped_difference) > np.pi/2)),
            "fraction_abs_unwrapped_difference_gt_pi": float(np.mean(np.abs(unwrapped_difference) > np.pi)),
            "amplitude_ratio_median": float(np.median(amp_a / np.maximum(amp_b, 1e-300))),
            "window_overlap_fraction": overlap,
        })
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--stage2b-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2b_center_grid_49"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2c_shape_aspect"))
    parser.add_argument("--diagnostic-output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2c_anomaly_diag"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T14:43:27+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    started = perf_counter()
    for path in (args.output, args.diagnostic_output):
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose a new stage-2c directory")

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

    stage2b_manifest = json.loads((args.stage2b_run / "manifest.json").read_text())
    stage2b_rows = json.loads((args.stage2b_run / "results_49.json").read_text())
    stage2b_nominal = next(row for row in stage2b_rows
                           if row["offset_x"] == 0 and row["offset_y"] == 0)
    if stage2b_nominal["status"] != "ok":
        raise ValueError("stage2b nominal row is not ok")
    if (stage2b_nominal["settings"]["half_width"] != 20
            or stage2b_nominal["settings"]["shape"] != "rect"
            or stage2b_nominal["settings"]["offset_x"] != 0
            or stage2b_nominal["settings"]["offset_y"] != 0):
        raise ValueError("stage2b nominal row is not the rect half-width-20 centre row")
    source_checks = {
        "observation_file_sha256": (stage2b_manifest["source"]["baseline_observation"]["file_sha256"],
                                    file_sha256(baseline / "baseline_observation.npz")),
        "observation_array_sha256": (stage2b_manifest["source"]["baseline_observation"]["observed_array_sha256"],
                                     array_sha256(observation_npz["observed"])),
        "calibration_file_sha256": (stage2b_manifest["source"]["baseline_calibration"]["file_sha256"],
                                    file_sha256(baseline / "baseline_calibration.npz")),
        "calibration_json_sha256": (stage2b_manifest["source"]["baseline_calibration"]["json_sha256"],
                                    file_sha256(baseline / "baseline_calibration.json")),
        "baseline_results_sha256": (stage2b_manifest["source"]["baseline_results"]["file_sha256"],
                                    file_sha256(baseline / "results.json")),
    }
    if any(left != right for left, right in source_checks.values()):
        raise ValueError(f"stage2b source hashes differ from canonical baseline: {source_checks}")

    model = DiffractionModel(config, indices)
    calibration = FourierCalibration(
        baseline=base_cal_npz["baseline"], response=base_cal_npz["response"],
        amplitude=float(base_cal_npz["amplitude"]),
        indices=tuple(int(value) for value in base_cal_npz["indices"]),
        settings=base_calibration["settings"], config=base_manifest["config"],
        reference_image=base_cal_npz["reference_image"])
    image = np.asarray(observation_npz["observed"], float)

    # Main shape scan: five unique windows; rect is reused from stage 2b.
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot" / "lsi"
    snapshot.mkdir(parents=True)
    for source in Path("lsi").glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(Path(__file__), args.output / "source_snapshot" / Path(__file__).name)
    code_state = {
        "branch": git_output(["branch", "--show-current"]),
        "head": git_output(["rev-parse", "HEAD"]),
        "status_short": git_output(["status", "--short"]),
        "source_sha256": {path.name: file_sha256(path) for path in Path("lsi").glob("*.py")},
    }
    atomic_json(args.output / "code_state.json", code_state)
    settings_rows = [dict(stage2b_nominal, reused=True,
                          reused_from=str(args.stage2b_run / "results_49.json"),
                          reused_source_sha256=file_sha256(args.stage2b_run / "results_49.json"),
                          area_definition=shape_area_definition("rect", 1.0),
                          seconds=0.0)]
    specs = [("circle", 1.0), ("ellipse", .5), ("ellipse", 2.0), ("gaussian", 1.0)]
    rows = [settings_rows[0]]
    for shape, aspect in specs:
        settings_point = replace(base_settings, shape=shape, aspect=aspect,
                                 offset_x=0.0, offset_y=0.0)
        case = f"shape_{shape}" if shape != "ellipse" else f"shape_ellipse_{aspect:g}"
        point_started = perf_counter()
        try:
            result = calibration.reconstruct(FourierExtractor(config, settings_point), image,
                                             details=False)
            metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
            row = {
                "case": case, "method": "fourier_calibrated", "status": "ok",
                "offset_x": 0.0, "offset_y": 0.0, "shape": shape, "aspect": aspect,
                "settings": settings_point.to_dict(),
                "area_definition": shape_area_definition(shape, aspect),
                "calibration_policy": "frozen_nominal_rect",
                "centers": result["centers"], "coverage": result["coverage"],
                "phase_residual_rms": result["phase_residual_rms"],
                "window_metrics": result["window_metrics"],
                "calibration_condition": result["calibration_condition"],
                "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(), **metrics,
                "reused": False,
            }
        except Exception as error:
            row = {"case": case, "method": "fourier_calibrated", "status": "failed",
                   "offset_x": 0.0, "offset_y": 0.0, "shape": shape, "aspect": aspect,
                   "settings": settings_point.to_dict(),
                   "area_definition": shape_area_definition(shape, aspect),
                   "calibration_policy": "frozen_nominal_rect", "reused": False,
                   "reason": f"{type(error).__name__}: {error}"}
        row["seconds"] = perf_counter() - point_started
        rows.append(row)
        write_rows(args.output, rows)
        atomic_json(args.output / "manifest.json", {
            "stage": "stage2c_shape_aspect", "status": "running",
            "batch_started_at": started_at.isoformat(timespec="seconds"),
            "protocol": {"fixed_half_width": 20, "fixed_centre": [0, 0],
                         "fixed_nominal_rect_calibration": True, "recalibrate_per_shape": False,
                         "direct_refit": False, "test_truth_usage": "comparison only; no production shape selected"},
            "rows": rows, "peak_memory_mib": peak_memory_mib(),
            "seconds": perf_counter() - started,
            "config": base_manifest["config"], "indices": list(indices),
            "truth_coefficients_waves": truth.tolist(),
            "source": {"baseline_directory": str(baseline),
                       "stage2b_run": str(args.stage2b_run),
                       "stage2b_results_sha256": file_sha256(args.stage2b_run / "results_49.json"),
                       "baseline_observation": {"path": str(baseline / "baseline_observation.npz"),
                                                 "file_sha256": file_sha256(baseline / "baseline_observation.npz"),
                                                 "observed_array_sha256": array_sha256(observation_npz["observed"])},
                       "baseline_calibration": {"path": str(baseline / "baseline_calibration.npz"),
                                                 "file_sha256": file_sha256(baseline / "baseline_calibration.npz"),
                                                 "json_sha256": file_sha256(baseline / "baseline_calibration.json")},
                       "reused_direct": {"reused": True, "row": direct_row}},
            "source_hash_checks": {key: {"stage2b": left, "current": right}
                                   for key, (left, right) in source_checks.items()},
        })
        print(f"{case} status={row['status']} rmse={row.get('rmse_waves')} seconds={row['seconds']:.2f}", flush=True)
    plot_shapes(rows, args.output)
    failed = [row for row in rows if row["status"] != "ok"]
    shape_summary = {
        "stage": "2c_shape_aspect", "status": "complete" if not failed else "complete_with_failures",
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_new": sum(not row.get("reused") and row["status"] == "ok" for row in rows),
        "reused": sum(bool(row.get("reused")) for row in rows),
        "failed": len(failed), "row_count": len(rows),
        "rows": rows,
        "area_definitions": {row["case"]: row["area_definition"] for row in rows},
        "fixed_calibration_note": ("rect half-width-20 calibration is frozen; shape changes are "
                                   "not independently recalibrated"),
        "source": {"baseline_observation_sha256": file_sha256(baseline / "baseline_observation.npz"),
                   "baseline_calibration_sha256": file_sha256(baseline / "baseline_calibration.npz"),
                   "stage2b_results_sha256": file_sha256(args.stage2b_run / "results_49.json")},
        "peak_memory_mib": peak_memory_mib(),
    }
    atomic_json(args.output / "summary.json", shape_summary)

    # Anomaly diagnosis: two points, exact saved settings, no large arrays saved.
    args.diagnostic_output.mkdir(parents=True, exist_ok=False)
    diag_rows = {}
    for case in ("(-2,-3)", "(-2,-2)"):
        x, y = (-2.0, -3.0) if case == "(-2,-3)" else (-2.0, -2.0)
        stored = next(row for row in stage2b_rows
                      if row["offset_x"] == x and row["offset_y"] == y)
        settings_point = FourierSettings(**stored["settings"])
        result = calibration.reconstruct(FourierExtractor(config, settings_point), image, details=True)
        metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
        diag_rows[case] = {"stored": stored, "recomputed": result, "metrics": metrics}
    reproduction = {}
    for case, data in diag_rows.items():
        stored, recomputed = data["stored"], data["recomputed"]
        reproduction[case] = {
            "rmse_delta_waves": float(data["metrics"]["rmse_waves"] - stored["rmse_waves"]),
            "phase_residual_delta": float(recomputed["phase_residual_rms"] - stored["phase_residual_rms"]),
            "coverage_delta": float(recomputed["coverage"] - stored["coverage"]),
            "coeffs_max_abs_delta_waves": float(np.max(np.abs(np.asarray(recomputed["coeffs_waves"])
                                                               - np.asarray(stored["coeffs_waves"])))),
            "raw_coeffs_max_abs_delta": (float(np.max(np.abs(np.asarray(recomputed["raw_coeffs"])
                                                              - np.asarray(stored["raw_coeffs"]))))
                                          if "raw_coeffs" in stored and "raw_coeffs" in recomputed else None),
            "centers_equal": bool(np.array_equal(np.asarray(recomputed["centers"]),
                                                  np.asarray(stored["centers"]))),
            "settings_identical": bool(settings_point.to_dict() == stored["settings"]),
        }
    sidebands = phase_comparison(diag_rows["(-2,-3)"]["recomputed"],
                                 diag_rows["(-2,-2)"]["recomputed"],
                                 FourierExtractor(config, FourierSettings(**diag_rows["(-2,-3)"]["stored"]["settings"])),
                                 diag_rows["(-2,-3)"]["stored"]["settings"]["phase_sigma"])
    reproduction_deltas = []
    for item in reproduction.values():
        reproduction_deltas.extend([abs(item["rmse_delta_waves"]), abs(item["phase_residual_delta"]),
                                    item["coeffs_max_abs_delta_waves"]])
        if item["raw_coeffs_max_abs_delta"] is not None:
            reproduction_deltas.append(item["raw_coeffs_max_abs_delta"])
    max_reproduction_delta = max(reproduction_deltas)
    anomaly_row = diag_rows["(-2,-3)"]["stored"]
    neighbor = diag_rows["(-2,-2)"]["stored"]
    neighbour_median = float(np.median([stage2b_rows[i]["rmse_waves"] for i in range(len(stage2b_rows))
                                        if (stage2b_rows[i]["offset_x"], stage2b_rows[i]["offset_y"])
                                        in ((-3.0, -3.0), (-1.0, -3.0), (-2.0, -2.0))]))
    if max_reproduction_delta < 1e-12:
        reproduction_conclusion = "stable exact reproduction; storage/nondeterminism is not the cause"
    else:
        reproduction_conclusion = "small numerical differences only; review thresholds before claiming nondeterminism"
    max_wrapped = max(item["wrapped_difference_rms_rad"] for item in sidebands)
    max_unwrapped = max(item["unwrapped_difference_rms_rad"] for item in sidebands)
    if max_wrapped < 0.05 and max_unwrapped > 1.0:
        interpretation = ("wrapped phase is close between the two points, while the unwrapped phase "
                          "differs substantially; this is consistent with unwrap-branch sensitivity, "
                          "but it is not a proof of the physical cause")
    elif max_unwrapped > 1.0:
        interpretation = ("large phase differences appear both before and after unwrapping; "
                          "the isolated low value is not explained by a simple unwrap branch alone")
    else:
        interpretation = ("phase fields are similar after the common processing; the isolated low value "
                          "remains unexplained despite stable reproduction")
    diagnostic = {
        "stage": "2c_anomaly_diagnostic",
        "points": ["(-2,-3)", "(-2,-2)"],
        "stored_values": {
            "(-2,-3)": {"rmse_waves": anomaly_row["rmse_waves"], "pv_waves": anomaly_row["pv_waves"],
                        "phase_residual_rms": anomaly_row["phase_residual_rms"],
                        "coverage": anomaly_row["coverage"], "coeffs_waves": anomaly_row["coeffs_waves"]},
            "(-2,-2)": {"rmse_waves": neighbor["rmse_waves"], "pv_waves": neighbor["pv_waves"],
                        "phase_residual_rms": neighbor["phase_residual_rms"],
                        "coverage": neighbor["coverage"], "coeffs_waves": neighbor["coeffs_waves"]},
        },
        "reproduction": reproduction,
        "reproduction_conclusion": reproduction_conclusion,
        "neighbour_median_rmse_waves": neighbour_median,
        "neighbour_median_minus_anomaly_waves": neighbour_median - anomaly_row["rmse_waves"],
        "sideband_comparison": sidebands,
        "interpretation": interpretation,
        "settings_identical_to_stage2b": all(item["settings_identical"] for item in reproduction.values()),
        "unwrap_seed": _UNWRAP_RANDOM,
        "large_arrays_saved": False,
        "output": str(args.diagnostic_output),
    }
    atomic_json(args.diagnostic_output / "anomaly_diagnostic.json", diagnostic)
    # Compact diagnostic figure: two-point metrics and unwrapped-difference histograms.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["(-2,-3)", "(-2,-2)"]
    metric_values = [[diag_rows[case]["metrics"]["rmse_waves"] for case in labels],
                     [diag_rows[case]["metrics"]["pv_waves"] for case in labels],
                     [diag_rows[case]["metrics"]["phase_residual_rms"] for case in labels]]
    for values, label in zip(metric_values, ("RMSE / waves", "PV / waves", "phase residual / rad")):
        axes[0].plot(labels, values, "o-", label=label)
    axes[0].set_yscale("log")
    axes[0].set_title("Recomputed metrics")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=.25)
    for sideband in sidebands:
        k = sideband["sideband"]
        extractor = FourierExtractor(config, FourierSettings(**diag_rows["(-2,-3)"]["stored"]["settings"]))
        mask = extractor.masks[k]
        ua = ndimage.gaussian_filter(unwrap_phase(diag_rows["(-2,-3)"]["recomputed"]["wrapped"][k],
                                                  **_UNWRAP_RANDOM), diag_rows["(-2,-3)"]["stored"]["settings"]["phase_sigma"])
        ub = ndimage.gaussian_filter(unwrap_phase(diag_rows["(-2,-2)"]["recomputed"]["wrapped"][k],
                                                  **_UNWRAP_RANDOM), diag_rows["(-2,-2)"]["stored"]["settings"]["phase_sigma"])
        difference = ua[mask] - ub[mask]
        difference -= np.median(difference)
        axes[1].hist(difference, bins=80, alpha=.55, label=f"sideband {k}")
    axes[1].set_title("Unwrapped phase difference (-2,-3) minus (-2,-2)")
    axes[1].set_xlabel("radians after median removal")
    axes[1].set_ylabel("pixels")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.diagnostic_output / f"anomaly_diagnostic.{suffix}", dpi=170)
    plt.close(fig)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    combined = {
        "stage": "2c_shape_and_anomaly",
        "shape": shape_summary,
        "anomaly": diagnostic,
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch_wall_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        "peak_memory_mib": peak_memory_mib(),
    }
    atomic_json(docs / "stage2c-summary.json", combined)
    lines = [
        "# 阶段 2c 进度：形状/长宽比与异常诊断",
        "",
        f"- 本批时间：{combined['batch_started_at']} → {combined['finished_at']}（约 {combined['batch_wall_seconds']:.0f} s）",
        f"- 形状设置：5 个（1 个复用 rect + 4 个新算），失败 {shape_summary['failed']} 个。",
        f"- 异常诊断：(-2,-3) 与 (-2,-2) 两点；结论：{diagnostic['reproduction_conclusion']}。",
        f"- 峰值 RSS：{combined['peak_memory_mib']:.0f} MiB；worker/batch=1。",
        "",
        "## 五个窗口（固定 rect 名义标定）",
        "| 设置 | RMSE / waves | PV / waves | coverage | phase residual | weight_sum | ENBW | max DC weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("window_metrics") or [{}, {}]
        lines.append(f"| {row['case']} | {row.get('rmse_waves', float('nan')):.6g} | "
                     f"{row.get('pv_waves', float('nan')):.6g} | {row.get('coverage', float('nan')):.6g} | "
                     f"{row.get('phase_residual_rms', float('nan')):.6g} | "
                     f"{np.mean([m.get('weight_sum', np.nan) for m in metrics]):.6g} | "
                     f"{np.mean([m.get('equivalent_noise_bins', np.nan) for m in metrics]):.6g} | "
                     f"{max(m.get('dc_weight', 0) for m in metrics):.6g} |")
    lines += [
        "",
        "## 面积/权重定义",
        "- rect：40×40 方窗，连续面积 1600 bins²。",
        "- circle：半径 `20*2/sqrt(pi)=22.568`，连续面积 1600 bins²。",
        "- ellipse 0.5/2：`hx=20/sqrt(aspect)`、`hy=20*sqrt(aspect)`，连续面积 1600 bins²。",
        "- gaussian：`sx=sy=20*sqrt(2/pi)`，连续积分权重 1600 bins²；无限支撑，零频泄漏需看实际 DC weight。",
        "- 所有形状仍固定 rect 名义标定；形状变化与标定适配混在一起，不能声称每个形状各自的最佳精度。",
        "",
        "## 异常诊断",
        f"- 复现结论：{diagnostic['reproduction_conclusion']}。",
        f"- `(-2,-3)` 相对邻点中位数低约 `{diagnostic['neighbour_median_minus_anomaly_waves']:.6g}` waves。",
        f"- 解释：{diagnostic['interpretation']}。",
        "- 不修改默认流程，不把未验证推测写成结论；原异常行保留。",
        "",
        "## 限制",
        "- 名义“半宽 20”在不同形状下不是同一截止频率；形状/面积/权重/支撑差异共同影响结果。",
        "- 仅 5 个设置、单一测试图、固定 rect 标定；不能据此选择生产形状。",
        "- 异常诊断只含两点；如需定位具体分支，应在后续单独设计中验证。",
        "",
        "## 下一批建议",
        "空间掩膜有限方案：固定 rect/半宽 20/名义标定、中心 0，先做 erosion 0/2/4/8（4 点），再做低幅剔除 0/.1/.3（3 点）；预计 7 个解算约 1–2 分钟，加启动/汇总约 3–5 分钟。",
        "",
    ]
    (docs / "stage2c-progress.md").write_text("\n".join(lines))
    print("shape failures", len(failed), "max reproduction delta", max_reproduction_delta)
    print("anomaly", diagnostic["reproduction_conclusion"], diagnostic["interpretation"])
    print("wrote", args.output, args.diagnostic_output)


if __name__ == "__main__":
    main()
