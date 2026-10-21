#!/usr/bin/env python
"""Representative noise, mixed-RMS and direct-initialisation checks.

Noise: 40 dB Gaussian, three fixed seeds, FT fixed and direct zero start on the
same noisy image.  RMS: 0.05/0.2/0.4 waves, two independent random mode
directions, FT fixed and direct with zero + one independent random start.
Initialisation: one 0.2-wave case with five observed-residual-screened starts.
Truth is used only for evaluation and scaling, never for initialisation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lsi.config import DiffractionConfig
from lsi.direct import fit_direct
from lsi.fourier import FourierCalibration, FourierExtractor, FourierSettings
from lsi.metrics import wavefront_metrics
from lsi.propagation import DiffractionModel


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(directory: Path, rows: list[dict]) -> None:
    atomic_json(directory / "results.json", rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    temporary.replace(directory / "results.csv")


def peak_memory_mib() -> float:
    import resource
    import platform
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / (1024**2 if platform.system() == "Darwin" else 1024))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage3b_noise_init"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T15:20:00+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new directory")

    baseline = args.baseline
    manifest = json.loads((baseline / "manifest.json").read_text())
    calibration_meta = json.loads((baseline / "baseline_calibration.json").read_text())
    calibration_npz = np.load(baseline / "baseline_calibration.npz")
    observation_npz = np.load(baseline / "baseline_observation.npz")
    config = DiffractionConfig(**manifest["config"])
    indices = tuple(int(value) for value in manifest["indices"])
    truth_base = np.asarray(observation_npz["truth"], float)
    model = DiffractionModel(config, indices)
    settings = FourierSettings(**calibration_meta["settings"])
    calibration = FourierCalibration(
        baseline=calibration_npz["baseline"], response=calibration_npz["response"],
        amplitude=float(calibration_npz["amplitude"]),
        indices=tuple(int(value) for value in calibration_npz["indices"]),
        settings=calibration_meta["settings"], config=manifest["config"],
        reference_image=calibration_npz["reference_image"])
    clean = np.asarray(observation_npz["observed"], float)
    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []

    def solve_ft(image, case, **context):
        point_started = perf_counter()
        result = calibration.reconstruct(FourierExtractor(config, settings), image, details=False)
        metrics = wavefront_metrics(model, result["coeffs_waves"], truth_base if "truth" not in context else np.asarray(context["truth"], float))
        row = {"case": case, "method": "fourier_calibrated", "status": "ok",
               "settings": settings.to_dict(), "coverage": result["coverage"],
               "phase_residual_rms": result["phase_residual_rms"],
               "centers": result["centers"], "window_metrics": result["window_metrics"],
               "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(), **metrics,
               **{k: v for k, v in context.items() if k != "truth"}}
        row["seconds"] = perf_counter() - point_started
        rows.append(row)
        write_rows(args.output, rows)
        return row

    def solve_direct(image, case, truth, starts, screen_initials, seed, **context):
        point_started = perf_counter()
        result = fit_direct(model, image, starts=starts, seed=seed, samples=8192,
                            max_nfev=60, progress=True, screen_initials=screen_initials)
        np.save(args.output / f"{case}_sample_rows.npy", result.pop("sample_rows"))
        result["coeffs"] = np.asarray(result["coeffs"]).tolist()
        for trial in result["trials"]:
            trial["metrics"] = wavefront_metrics(model, trial["coeffs"], truth)
        atomic_json(args.output / f"{case}_direct_trials.json", result)
        best = result["trials"][result["best_trial"]]
        row = {"case": case, "method": "direct", "status": "ok",
               "starts": starts, "screen_initials": screen_initials,
               "coeffs_waves": result["coeffs"], "converged": best["converged"],
               "nfev": best["nfev"], "full_intensity_rms": best["full_intensity_rms"],
               "gain": best["gain"], "background": best["background"],
               **wavefront_metrics(model, result["coeffs"], truth),
               **{k: v for k, v in context.items() if k != "truth"}}
        row["seconds"] = perf_counter() - point_started
        rows.append(row)
        write_rows(args.output, rows)
        return row

    # Noise: same clean image and noise realisation for FT and direct.
    noise_images = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        noisy = clean + rng.normal(0, 10 ** (-40 / 20), clean.shape)
        noise_images.append(noisy)
        np.savez_compressed(args.output / f"noise_snr40_seed{seed}.npz",
                            observed=noisy, clean=clean, noise=noisy - clean,
                            truth=truth_base)
        solve_ft(noisy, f"noise_snr40_seed{seed}", snr_db=40.0, noise_seed=seed, kind="noise")
        solve_direct(noisy, f"noise_snr40_seed{seed}", truth_base, 1, 0, seed,
                     snr_db=40.0, noise_seed=seed, kind="noise")

    # Two independent mixed directions at three RMS levels.
    rms_rows = {}
    for level in (.05, .2, .4):
        for direction in (0, 1):
            rng = np.random.default_rng(9201 + direction)
            truth = rng.normal(size=len(indices))
            wavefront = model.wavefront(truth)
            truth *= level / np.std(wavefront[model.pupil])
            image = model.evaluate(truth)
            image /= image.max()
            case = f"rms_{level:g}_dir{direction}"
            rms_rows[case] = (truth, image)
            solve_ft(image, case, rms_waves=level, direction=direction, kind="rms", truth=truth)
            solve_direct(image, case, truth, 2, 0, 42, rms_waves=level,
                         direction=direction, kind="rms")
            np.savez_compressed(args.output / f"{case}_observation.npz",
                                observed=image, truth=truth)

    # One five-start stability check at 0.2 waves, direction 0.
    truth, image = rms_rows["rms_0.2_dir0"]
    solve_direct(image, "init_stability_0.2_dir0", truth, 5, 8, 42,
                 rms_waves=.2, direction=0, kind="initialisation")

    failed = [row for row in rows if row["status"] != "ok"]
    summary = {
        "stage": "3b_noise_init", "status": "complete" if not failed else "complete_with_failures",
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": len(rows), "failed": len(failed),
        "rows": rows,
        "source": {"baseline_observation_sha256": file_sha256(baseline / "baseline_observation.npz"),
                   "baseline_calibration_sha256": file_sha256(baseline / "baseline_calibration.npz")},
        "peak_memory_mib": peak_memory_mib(),
        "noise_definition": "SNR = 20 log10(clean peak / Gaussian noise sd); peak is 1 after normalisation",
        "direct_initialisation": {"noise": "single zero start", "rms": "zero + one independent random start",
                                  "stability": "zero + four candidates screened by observed intensity residual"},
        "truth_usage": "evaluation and amplitude scaling only; no initialisation from truth",
    }
    atomic_json(args.output / "summary.json", summary)
    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    atomic_json(docs / "stage3b-summary.json", summary)
    lines = ["# 阶段 3b 进度：噪声与初值稳定性", "",
             f"- 时间：{summary['batch_started_at']} → {summary['finished_at']}",
             f"- 行数：{len(rows)}，失败 {len(failed)}；峰值 RSS {summary['peak_memory_mib']:.0f} MiB。",
             "", "## 噪声 40 dB（同一噪声实现）", "| 种子 | FT RMSE | direct RMSE | direct nfev |",
             "|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("kind") == "noise" and row["method"] == "fourier_calibrated":
            direct = next(item for item in rows if item.get("kind") == "noise"
                          and item.get("case") == row["case"] and item["method"] == "direct")
            lines.append(f"| {row['seed']} | {row['rmse_waves']:.6g} | {direct['rmse_waves']:.6g} | {direct['nfev']} |")
    lines += ["", "## 混合 RMS / 两个方向", "| 设置 | FT RMSE | direct RMSE | direct converged |", "|---|---:|---:|---|"]
    for row in rows:
        if row.get("kind") == "rms" and row["method"] == "fourier_calibrated":
            direct = next(item for item in rows if item.get("kind") == "rms"
                          and item.get("case") == row["case"] and item["method"] == "direct")
            lines.append(f"| {row['case']} | {row['rmse_waves']:.6g} | {direct['rmse_waves']:.6g} | {direct['converged']} |")
    lines += ["", "## 初值稳定性（0.2 waves, dir0, 5 starts）",
              "- 直接法 trials 与逐次 metrics 已保存到 `init_stability_0.2_dir0_direct_trials.json`。",
              "- 初值筛选只使用观测强度残差，不使用测试真值。", ""]
    (docs / "stage3b-progress.md").write_text("\n".join(lines))
    print("rows", len(rows), "failed", len(failed), "peak", summary["peak_memory_mib"])


if __name__ == "__main__":
    main()
