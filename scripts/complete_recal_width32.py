#!/usr/bin/env python
"""Complete the interrupted recalibrated half-width-32 row without rerunning it.

The source run saved the per-window calibration and observation, then was
terminated while writing the large Fourier detail archive.  This script reuses
that calibration and writes only a compact result in a *new* directory.  It
never overwrites the interrupted run and never regenerates the independent
calibration images.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
import resource
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


def peak_memory_mib() -> float:
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / (1024**2 if platform.system() == "Darwin" else 1024))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_recal"))
    parser.add_argument("--frozen-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_frozen"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_recal_supplement_w32"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new supplement directory")

    source = args.source_run
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_rows = json.loads((source / "results.json").read_text())
    calibration_meta = json.loads((source / "width_32_calibration.json").read_text())
    if float(calibration_meta["settings"]["half_width"]) != 32:
        raise ValueError("source calibration is not half-width 32")
    observation = np.load(source / "regions_observation.npz")
    calibration_npz = np.load(source / "width_32_calibration.npz")
    config = DiffractionConfig(**source_manifest["config"])
    model = DiffractionModel(config, tuple(int(v) for v in source_manifest["indices"]))
    calibration = FourierCalibration(
        baseline=calibration_npz["baseline"],
        response=calibration_npz["response"],
        amplitude=float(calibration_npz["amplitude"]),
        indices=tuple(int(v) for v in calibration_npz["indices"]),
        settings=calibration_meta["settings"],
        config=source_manifest["config"],
        reference_image=calibration_npz["reference_image"],
    )
    settings = FourierSettings(**calibration_meta["settings"])
    truth = np.asarray(observation["truth"], float)
    image = np.asarray(observation["observed"], float)

    compute_started = perf_counter()
    result = calibration.reconstruct(FourierExtractor(config, settings), image, details=False)
    metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
    compute_seconds = perf_counter() - compute_started
    row = {
        "case": "width_32",
        "method": "fourier_calibrated",
        "status": "ok",
        "coverage": result["coverage"],
        "phase_residual_rms": result["phase_residual_rms"],
        "centers": result["centers"],
        "seconds": result["seconds"],
        "window_metrics": result["window_metrics"],
        "calibration_condition": result["calibration_condition"],
        "calibration_settings": result["calibration_settings"],
        "settings": result["settings"],
        "scan": "width",
        "calibration_policy": "per_window",
        "coeffs_waves": np.asarray(result["coeffs_waves"]).tolist(),
        **metrics,
        "source": f"supplement from {source}",
    }

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "width_32_fourier_compact.npz",
        raw_coeffs=result["raw_coeffs"],
        coeffs_waves=result["coeffs_waves"],
        centers=np.asarray(result["centers"], float),
        window_weight=np.asarray([m["weight_sum"] for m in result["window_metrics"]]),
        window_enbw=np.asarray([m["equivalent_noise_bins"] for m in result["window_metrics"]]),
        coverage=np.asarray(result["coverage"]),
        phase_residual_rms=np.asarray(result["phase_residual_rms"]),
    )
    write_json(args.output / "results.json", [row])
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})

    frozen_rows = json.loads((args.frozen_run / "results.json").read_text())
    frozen_widths = {}
    for item in frozen_rows:
        if item.get("method") == "fourier_calibrated" and item.get("case", "").startswith("width_"):
            frozen_widths[float(item["settings"]["half_width"])] = item["rmse_waves"]
    source_widths = {}
    for item in source_rows:
        if item.get("method") == "fourier_calibrated" and item.get("case", "").startswith("width_"):
            source_widths[float(item["settings"]["half_width"])] = item["rmse_waves"]
    combined_widths = [{"half_width": width,
                        "recalibrated_rmse_waves": (source_widths.get(width)
                                                    if width != 32 else row["rmse_waves"]),
                        "frozen_nominal_calibration_rmse_waves": frozen_widths.get(width),
                        "source": ("supplement_w32" if width == 32 else "interrupted_source_rows")}
                       for width in sorted(set(source_widths) | {32.0})]
    selection = json.loads((source / "width_selection.json").read_text())
    combined = {
        "source_run": str(source),
        "source_manifest_status": source_manifest.get("status"),
        "source_row_count": len(source_rows),
        "supplement_run": str(args.output),
        "supplement_row": row,
        "combined_widths": combined_widths,
        "width_selection": {
            "selected_width": selection.get("selected_width"),
            "validation_rms_waves": selection.get("validation_rms"),
            "repeats": selection.get("repeats"),
            "scores": selection.get("scores"),
            "test_truth_used": selection.get("test_truth_used"),
            "source": str(source / "width_selection.json"),
            "source_sha256": file_sha256(source / "width_selection.json"),
        },
    }
    write_json(args.output / "combined_widths.json", combined)

    manifest = {
        "stage": "stage1_recalibrated_width32_supplement",
        "status": "complete",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": str(source),
        "source_run_status": source_manifest.get("status"),
        "source_run_row_count": len(source_rows),
        "source_partial_width_32_npz": {
            "path": str(source / "width_32_fourier.npz"),
            "used": False,
            "reason": "decompression failed (invalid block type); treated as an interrupted partial write",
        },
        "reused_calibration": {
            "path": str(source / "width_32_calibration.npz"),
            "sha256": file_sha256(source / "width_32_calibration.npz"),
            "settings_json": str(source / "width_32_calibration.json"),
            "settings_json_sha256": file_sha256(source / "width_32_calibration.json"),
            "training_images_saved": calibration_meta.get("training_images_saved"),
            "training_image_sha256": calibration_meta.get("training_image_sha256"),
            "independent_calibration_rerun": False,
        },
        "reused_observation": {
            "path": str(source / "regions_observation.npz"),
            "sha256": file_sha256(source / "regions_observation.npz"),
        },
        "command": ("uv run --locked python scripts/complete_recal_width32.py "
                    f"--source-run {source} --output {args.output}"),
        "config": source_manifest["config"],
        "indices": source_manifest["indices"],
        "truth_coefficients_waves": truth.tolist(),
        "workers": 1,
        "batch_sources": 1,
        "compute_seconds": compute_seconds,
        "wall_seconds": perf_counter() - started,
        "peak_memory_mib": peak_memory_mib(),
        "row_count": 1,
        "code_commit": source_manifest.get("git_commit"),
        "source_sha256": {name: file_sha256(Path("lsi") / name)
                          for name in ("study.py", "fourier.py", "propagation.py", "metrics.py")},
        "notes": [
            "Only the missing half-width-32 reconstruction was computed.",
            "The independent calibration images were not regenerated.",
            "The interrupted source directory was not modified.",
            "Large intermediate arrays are not copied; the source observation and calibration are referenced by path and SHA256.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    stage_summary = {
        "stage": 1,
        "title": "recalfibrated width-32 supplement",
        "status": "complete",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": manifest["finished_at"],
        "wall_seconds": manifest["wall_seconds"],
        "completed": [
            "reconstructed recalibrated half-width 32 from saved calibration and observation",
            "merged widths 8,12,16,20,24,28,32 for the recalibrated scan",
            "summarized the already-complete independent validation selection",
        ],
        "not_done": [
            "center-offset scan",
            "shape/aspect scan",
            "spatial-mask scan",
            "angle scans",
            "noise and initialization studies",
            "formal Chinese report and PR",
        ],
        "key_results": {
            "recalibrated_width_32_rmse_waves": row["rmse_waves"],
            "recalibrated_width_24_rmse_waves": source_widths.get(24.0),
            "recalibrated_width_28_rmse_waves": source_widths.get(28.0),
            "recalibrated_width_20_rmse_waves": source_widths.get(20.0),
            "frozen_width_22_best_test_truth_rmse_waves": min(frozen_widths.values()),
            "frozen_width_32_rmse_waves": frozen_widths.get(32.0),
            "validation_selected_width": selection.get("selected_width"),
            "validation_rms_waves": selection.get("validation_rms"),
            "validation_repeats": selection.get("repeats"),
            "direct_reference_rmse_waves": next((item.get("rmse_waves") for item in source_rows
                                                  if item.get("method") == "direct"), None),
        },
        "resources": {
            "compute_seconds": manifest["compute_seconds"],
            "wall_seconds": manifest["wall_seconds"],
            "peak_memory_mib": manifest["peak_memory_mib"],
            "workers": 1,
            "batch_sources": 1,
        },
        "code": {
            "branch": "diffraction-window-tilt-followup",
            "head": manifest.get("code_commit"),
            "modified_tracked": ["lsi/study.py"],
            "untracked": ["scripts/", "work/"],
            "committed": False,
        },
        "outputs": {
            "supplement_dir": str(args.output),
            "combined_widths": str(args.output / "combined_widths.json"),
            "manifest": str(args.output / "manifest.json"),
            "progress": str(docs / "stage1-progress.md"),
        },
        "estimate": {
            "this_batch_compute_seconds": manifest["compute_seconds"],
            "remaining_research": ("Region center/shape/mask scans and angle scans are measured "
                                   "at roughly 10-15 s per Fourier solve and ~34 s per geometry-specific "
                                   "calibration; remaining compute is approximately 1.5-2.5 h, plus "
                                   "report/PR work. Angle, noise and initialization stages have not been "
                                   "trial-run, so their ETA remains uncertain."),
        },
        "limitations": [
            "The 0.05-wave three-sample validation selected width 32 and is suboptimal on this 0.1-wave test image; this does not establish amplitude as the cause because the wavefront mixture and small validation set also differ.",
            "The recalibrated width-32 row reuses a saved per-window calibration, not a fresh calibration run.",
            "The source run remains marked interrupted; the supplement is a separate derived result.",
        ],
    }
    write_json(docs / "stage1-summary.json", stage_summary)
    progress = f"""# 阶段 1 进度：补齐重标定半宽 32

- 时间：{manifest['started_at']} → {manifest['finished_at']}（UTC）
- 状态：完成；源目录保持 `interrupted`，未覆盖。
- 本批计算：{manifest['compute_seconds']:.1f} s；脚本墙钟：{manifest['wall_seconds']:.1f} s；峰值 RSS：{manifest['peak_memory_mib']:.0f} MiB。
- 输出目录：`{args.output}`

## 完成
- 复用 `regions_width_recal/width_32_calibration.npz` 与同一张 `regions_observation.npz`，只补算半宽 32 的傅里叶重建。
- 合并重标定窗宽 8,12,16,20,24,28,32；源目录中截断的 `width_32_fourier.npz` 已确认不可读，未使用。
- 独立验证选窗文件已完整（2 个 0.05 waves 样本，7 个半宽），本批只汇总，不重算。

## 关键数值
- 重标定半宽 32：**{row['rmse_waves']:.6g} waves**。
- 重标定 24 / 28 / 20：{source_widths.get(24.0):.6g} / {source_widths.get(28.0):.6g} / {source_widths.get(20.0):.6g} waves。
- 冻结标定测试真值最低：半宽 22，{min(frozen_widths.values()):.6g} waves；半宽 32 为 {frozen_widths.get(32.0):.6g} waves。
- 独立验证选中：半宽 32（{selection.get('validation_rms')} waves，{selection.get('repeats')} 个样本）。

## 限制
- 验证集只有 3 个 0.05 waves 混合波前（冻结实验 3 个；重标定实验 2 个），在本 0.1 waves 测试图上次优；不能仅据此因果归因为幅度差异，波前组合和小样本也有影响。
- 本批只补齐一个缺失行，不是完整重标定研究结论。

## 下一批建议
按原计划先做二维窗中心扫描（49 个偏移，冻结名义标定），预计约 10–15 分钟；随后再做形状/长宽比与空间掩膜。
"""
    (docs / "stage1-progress.md").write_text(progress)
    print("width_32 rmse", row["rmse_waves"])
    print("compute_seconds", compute_seconds, "peak_mib", manifest["peak_memory_mib"])
    print("wrote", args.output, "and", docs / "stage1-progress.md")


if __name__ == "__main__":
    main()
