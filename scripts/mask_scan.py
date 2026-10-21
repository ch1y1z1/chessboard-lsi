#!/usr/bin/env python
"""Stage 2d: finite spatial-mask scan at fixed nominal rect calibration.

Eight unique settings: baseline (erosion 0, threshold 0), erosion 2/4/8/12
with threshold 0, and per-sideband amplitude thresholds 0.1/0.3/0.5 with
erosion 0.  No cross combinations, no per-mask recalibration, no propagation
or direct-fit rerun.  Evaluation stays on the same fixed annular pupil.
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


def require_keys(mapping: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{context} is missing expected keys: {missing}")


def plot_masks(representatives: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, data) in zip(axes, representatives.items()):
        ax.imshow(data, origin="lower", cmap="gray")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Representative sideband-0 valid masks (white = valid)", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"mask_representatives.{suffix}", dpi=170)
    plt.close(fig)


def plot_error_coverage(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = next(row for row in rows if row.get("reused"))
    erosion = sorted((row for row in rows if row["case"].startswith("erosion")),
                     key=lambda row: row["settings"]["erosion"])
    threshold = sorted((row for row in rows if row["case"].startswith("amplitude")),
                       key=lambda row: row["settings"]["amplitude_threshold"])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ex = [baseline] + erosion
    tx = [baseline] + threshold
    ax.semilogy([row["coverage"] for row in ex], [row["rmse_waves"] for row in ex],
                "o-", color="C2", label="erosion 0,2,4,8,12")
    ax.semilogy([row["coverage"] for row in tx], [row["rmse_waves"] for row in tx],
                "D--", color="C3", label="threshold 0,0.1,0.3,0.5")
    ax.scatter([baseline["coverage"]], [baseline["rmse_waves"]], s=90, color="C0",
               zorder=5, label="baseline (reused)")
    for row in ex[1:]:
        ax.annotate(f"e{row['settings']['erosion']:g}", (row["coverage"], row["rmse_waves"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6, color="C2")
    for row in tx[1:]:
        ax.annotate(f"t{row['settings']['amplitude_threshold']:g}", (row["coverage"], row["rmse_waves"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6, color="C3")
    ax.set_xlabel("coverage (fixed 2*n*n denominator)")
    ax.set_ylabel("Wavefront RMSE / waves (log scale)")
    ax.set_title("Spatial-mask scan (fixed nominal rect calibration)")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(path / f"mask_error_coverage.{suffix}", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--stage2b-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2b_center_grid_49"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2d_mask_scan"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T14:51:00+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new stage-2d directory")

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
    baseline_row = next(row for row in stage2b_rows
                        if row["offset_x"] == 0 and row["offset_y"] == 0)
    require_keys(baseline_row, ("coverage", "phase_residual_rms", "settings",
                                "rmse_waves", "pv_waves", "evaluation_pixels", "window_metrics"),
                 "stage2b nominal row")
    if (baseline_row["settings"]["half_width"] != 20
            or baseline_row["settings"]["shape"] != "rect"
            or baseline_row["settings"]["erosion"] != 0
            or baseline_row["settings"]["amplitude_threshold"] != 0.0):
        raise ValueError("stage2b nominal row is not the rect / width-20 / no-mask baseline")
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

    baseline_masks = FourierExtractor(config, base_settings).masks
    baseline_valid = [int(mask.sum()) for mask in baseline_masks]
    baseline_total = sum(baseline_valid)
    if abs(baseline_total / (2 * config.n * config.n) - baseline_row["coverage"]) > 1e-9:
        raise ValueError("baseline coverage does not match mask pixel counts")
    rows = [dict(baseline_row, reused=True,
                 reused_from=str(args.stage2b_run / "results_49.json"),
                 reused_source_sha256=file_sha256(args.stage2b_run / "results_49.json"),
                 mask_pixels_per_sideband=[int(mask.sum()) for mask in baseline_masks],
                 valid_pixels_per_sideband=baseline_valid,
                 valid_pixels_total=baseline_total,
                 valid_fraction_per_sideband=[1.0, 1.0],
                 seconds=0.0)]
    reps: dict[str, np.ndarray] = {}
    specs = [("erosion", int(value), 0.0) for value in (2, 4, 8, 12)]
    specs += [("amplitude", 0, float(value)) for value in (.1, .3, .5)]
    for kind, value, threshold in specs:
        settings_point = replace(base_settings, erosion=value, amplitude_threshold=threshold,
                                 offset_x=0.0, offset_y=0.0, half_width=20, shape="rect")
        case = f"erosion_{value}" if kind == "erosion" else f"amplitude_{threshold:g}"
        point_started = perf_counter()
        try:
            extractor = FourierExtractor(config, settings_point)
            result = calibration.reconstruct(extractor, image, details=True)
            require_keys(result, ("coverage", "phase_residual_rms", "centers", "window_metrics",
                                  "coeffs_waves", "masks", "amplitudes"), f"{case} result")
            metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
            require_keys(metrics, ("rmse_waves", "pv_waves", "evaluation_pixels"),
                         f"{case} wavefront_metrics")
            mask_pixels, valid_pixels = [], []
            valid_mask = np.zeros((config.n, config.n), bool)
            for k, mask in enumerate(result["masks"]):
                amplitude = np.abs(result["amplitudes"][k])[mask]
                good = (amplitude >= threshold * amplitude.max()) if threshold else np.ones(mask.sum(), bool)
                mask_pixels.append(int(mask.sum()))
                valid_pixels.append(int(good.sum()))
                if k == 0:
                    valid_mask[mask] = good
            valid_total = sum(valid_pixels)
            if abs(valid_total / (2 * config.n * config.n) - result["coverage"]) > 1e-9:
                raise ValueError(f"{case} coverage does not match valid pixel counts")
            row = {
                "case": case, "method": "fourier_calibrated", "status": "ok",
                "offset_x": 0.0, "offset_y": 0.0, "settings": settings_point.to_dict(),
                "calibration_policy": "frozen_nominal_rect", "reused": False,
                "centers": result["centers"], "coverage": result["coverage"],
                "phase_residual_rms": result["phase_residual_rms"],
                "window_metrics": result["window_metrics"],
                "mask_pixels_per_sideband": mask_pixels,
                "valid_pixels_per_sideband": valid_pixels,
                "valid_pixels_total": valid_total,
                "valid_fraction_per_sideband": [valid / max(mask, 1) for valid, mask in zip(valid_pixels, mask_pixels)],
                "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(), **metrics,
            }
            if case in ("erosion_8", "amplitude_0.3"):
                reps[case] = valid_mask.copy()
        except Exception as error:
            row = {"case": case, "method": "fourier_calibrated", "status": "failed",
                   "offset_x": 0.0, "offset_y": 0.0, "settings": settings_point.to_dict(),
                   "calibration_policy": "frozen_nominal_rect", "reused": False,
                   "reason": f"{type(error).__name__}: {error}"}
        row["seconds"] = perf_counter() - point_started
        rows.append(row)
        write_rows(args.output, rows)
        atomic_json(args.output / "manifest.json", {
            "stage": "stage2d_mask_scan", "status": "running",
            "batch_started_at": started_at.isoformat(timespec="seconds"),
            "protocol": {"fixed_half_width": 20, "fixed_centre": [0, 0],
                         "fixed_nominal_rect_calibration": True, "recalibrate_per_mask": False,
                         "direct_refit": False, "cross_combinations": False,
                         "evaluation_pupil": "fixed annular pupil; same pixels for all rows",
                         "amplitude_threshold_definition": "per sideband: amplitude >= threshold * max amplitude inside that sideband's valid mask"},
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
        print(f"{case} status={row['status']} rmse={row.get('rmse_waves')} coverage={row.get('coverage')} seconds={row['seconds']:.2f}", flush=True)

    reps["baseline"] = np.asarray(baseline_masks[0], bool)
    for name in ("erosion_8", "amplitude_0.3"):
        if name not in reps:
            reps[name] = np.zeros((config.n, config.n), bool)
    reps = {"baseline": reps["baseline"], "erosion_8": reps["erosion_8"],
            "amplitude_0.3": reps["amplitude_0.3"]}
    plot_masks(reps, args.output)
    plot_error_coverage(rows, args.output)
    failed = [row for row in rows if row["status"] != "ok"]
    evaluation_pixels = sorted({row["evaluation_pixels"] for row in rows if row["status"] == "ok"})
    erosion_rows = sorted((row for row in rows if row["case"].startswith("erosion")),
                          key=lambda row: row["settings"]["erosion"])
    threshold_rows = sorted((row for row in rows if row["case"].startswith("amplitude")),
                            key=lambda row: row["settings"]["amplitude_threshold"])
    summary = {
        "stage": "2d_mask_scan",
        "status": "complete" if not failed else "complete_with_failures",
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_new": sum(not row.get("reused") and row["status"] == "ok" for row in rows),
        "reused": sum(bool(row.get("reused")) for row in rows),
        "failed": len(failed), "row_count": len(rows),
        "evaluation_pupil_pixels": evaluation_pixels,
        "evaluation_pupil_constant": len(evaluation_pixels) == 1,
        "erosion_rows": erosion_rows,
        "threshold_rows": threshold_rows,
        "all_rows": rows,
        "coverage_range": [min(row["coverage"] for row in rows if row["status"] == "ok"),
                           max(row["coverage"] for row in rows if row["status"] == "ok")],
        "rmse_range": [min(row["rmse_waves"] for row in rows if row["status"] == "ok"),
                       max(row["rmse_waves"] for row in rows if row["status"] == "ok")],
        "pv_range": [min(row["pv_waves"] for row in rows if row["status"] == "ok"),
                     max(row["pv_waves"] for row in rows if row["status"] == "ok")],
        "phase_residual_range": [min(row["phase_residual_rms"] for row in rows if row["status"] == "ok"),
                                 max(row["phase_residual_rms"] for row in rows if row["status"] == "ok")],
        "fixed_calibration_note": "rect half-width-20 calibration is frozen; mask changes include calibration mismatch",
        "source": {"baseline_observation_sha256": file_sha256(baseline / "baseline_observation.npz"),
                   "baseline_calibration_sha256": file_sha256(baseline / "baseline_calibration.npz"),
                   "stage2b_results_sha256": file_sha256(args.stage2b_run / "results_49.json")},
        "peak_memory_mib": peak_memory_mib(),
        "batch_wall_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
    }
    atomic_json(args.output / "summary.json", summary)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 阶段 2d 进度：空间有效区有限扫描",
        "",
        f"- 本批时间：{summary['batch_started_at']} → {summary['finished_at']}（约 {(datetime.fromisoformat(summary['finished_at']) - started_at).total_seconds():.0f} s）",
        f"- 设置：8 个（1 个复用 baseline + 7 个新算），失败 {len(failed)} 个。",
        f"- 评价光瞳像素：{evaluation_pixels}；是否恒定：{summary['evaluation_pupil_constant']}。",
        f"- 峰值 RSS：{summary['peak_memory_mib']:.0f} MiB；worker/batch=1。",
        "",
        "## 8 个设置",
        "| 设置 | erosion | threshold | RMSE / waves | PV / waves | coverage | valid pixels | per-sideband valid | phase residual |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['settings']['erosion']} | "
                     f"{row['settings']['amplitude_threshold']:g} | "
                     f"{row.get('rmse_waves', float('nan')):.6g} | {row.get('pv_waves', float('nan')):.6g} | "
                     f"{row.get('coverage', float('nan')):.6g} | {row.get('valid_pixels_total', '—')} | "
                     f"{row.get('valid_pixels_per_sideband', '—')} | {row.get('phase_residual_rms', float('nan')):.6g} |")
    lines += [
        "",
        "## 定义与限制",
        "- 幅值阈值定义：在每个边带的有效掩膜内，保留 `amplitude >= threshold * max(amplitude)` 的像素；0.1/0.3/0.5 均按各自边带归一化。",
        "- coverage 分母固定为 `2*n*n`；评价始终使用同一个固定物镜环形光瞳，未随解算掩膜缩小。",
        "- 固定 rect/半宽 20/名义标定；未逐掩膜重标定，不能把结果当作各掩膜各自的最佳精度。",
        "- 只做单变量扫描，不含 erosion × threshold 交叉组合；单测试图，不作为生产阈值选择依据。",
        "",
        "## 下一批建议",
        "角度试点：相机 x 轴 +0.25°，同一观测比较 known/ignored 几何下 direct 与 FT fixed/auto；known 用对应几何独立标定，ignored 用零角度标定；预计 4 个主要解算 + 2 次标定，需实测确认是否能在 5–10 分钟返回。",
        "",
    ]
    (docs / "stage2d-progress.md").write_text("\n".join(lines))
    atomic_json(docs / "stage2d-summary.json", summary)
    print("rows", len(rows), "failed", len(failed), "evaluation_pixels", evaluation_pixels)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
