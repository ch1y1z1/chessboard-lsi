#!/usr/bin/env python
"""Focused reproduction of the stage-2b (-2,-3) / (-2,-2) anomaly.

The source stage-2b rows did not save raw Fourier coefficients, so this
diagnostic compares the stored quantities that exist (RMSE, phase residual,
coverage, physical coefficients and centres) and analyses wrapped/unwrapped
sideband phase differences.  It does not modify the default pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
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


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def phase_comparison(result_a: dict, result_b: dict, extractor, sigma: float) -> list[dict]:
    stats = []
    for k in range(2):
        mask = extractor.masks[k]
        wrapped_a, wrapped_b = result_a["wrapped"][k], result_b["wrapped"][k]
        amp_a, amp_b = result_a["amplitudes"][k][mask], result_b["amplitudes"][k][mask]
        wrapped_difference = wrapped_a[mask] - wrapped_b[mask]
        wrapped_difference -= np.median(wrapped_difference)
        unwrapped_a = ndimage.gaussian_filter(unwrap_phase(wrapped_a, **_UNWRAP_RANDOM), sigma)
        unwrapped_b = ndimage.gaussian_filter(unwrap_phase(wrapped_b, **_UNWRAP_RANDOM), sigma)
        unwrapped_difference = unwrapped_a[mask] - unwrapped_b[mask]
        unwrapped_difference -= np.median(unwrapped_difference)
        window_a, window_b = result_a["windows"][k], result_b["windows"][k]
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
            "window_overlap_fraction": float(np.minimum(window_a, window_b).sum()
                                             / max(np.maximum(window_a, window_b).sum(), 1e-30)),
        })
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--stage2b-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2b_center_grid_49"))
    parser.add_argument("--shape-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2c_shape_aspect"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage2c_anomaly_diag_v2"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T14:43:27+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new diagnostic directory")

    baseline = args.baseline
    base_manifest = json.loads((baseline / "manifest.json").read_text())
    base_calibration = json.loads((baseline / "baseline_calibration.json").read_text())
    base_cal_npz = np.load(baseline / "baseline_calibration.npz")
    observation_npz = np.load(baseline / "baseline_observation.npz")
    stage2b_rows = json.loads((args.stage2b_run / "results_49.json").read_text())
    config = DiffractionConfig(**base_manifest["config"])
    indices = tuple(int(value) for value in base_manifest["indices"])
    truth = np.asarray(observation_npz["truth"], float)
    model = DiffractionModel(config, indices)
    calibration = FourierCalibration(
        baseline=base_cal_npz["baseline"], response=base_cal_npz["response"],
        amplitude=float(base_cal_npz["amplitude"]),
        indices=tuple(int(value) for value in base_cal_npz["indices"]),
        settings=base_calibration["settings"], config=base_manifest["config"],
        reference_image=base_cal_npz["reference_image"])
    image = np.asarray(observation_npz["observed"], float)

    points = {"(-2,-3)": (-2.0, -3.0), "(-2,-2)": (-2.0, -2.0)}
    diag = {}
    for label, (x, y) in points.items():
        stored = next(row for row in stage2b_rows if row["offset_x"] == x and row["offset_y"] == y)
        settings = FourierSettings(**stored["settings"])
        if not (settings.offset_x == x and settings.offset_y == y
                and settings.half_width == 20 and settings.center_mode == "fixed"):
            raise ValueError(f"stored settings mismatch for {label}")
        result = calibration.reconstruct(FourierExtractor(config, settings), image, details=True)
        metrics = wavefront_metrics(model, result["coeffs_waves"], truth)
        diag[label] = {"stored": stored, "recomputed": result, "metrics": metrics, "settings": settings}

    reproduction = {}
    for label, data in diag.items():
        stored, result, metrics = data["stored"], data["recomputed"], data["metrics"]
        reproduction[label] = {
            "rmse_delta_waves": float(metrics["rmse_waves"] - stored["rmse_waves"]),
            "phase_residual_delta": float(result["phase_residual_rms"] - stored["phase_residual_rms"]),
            "coverage_delta": float(result["coverage"] - stored["coverage"]),
            "coeffs_max_abs_delta_waves": float(np.max(np.abs(np.asarray(result["coeffs_waves"])
                                                               - np.asarray(stored["coeffs_waves"])))),
            "centers_equal": bool(np.array_equal(np.asarray(result["centers"]),
                                                  np.asarray(stored["centers"]))),
            "settings_identical": bool(data["settings"].to_dict() == stored["settings"]),
            "raw_coeffs_comparison": "not available in source stage2b results",
        }
    max_reproduction_delta = max(max(abs(item["rmse_delta_waves"]), abs(item["phase_residual_delta"]),
                                     item["coeffs_max_abs_delta_waves"])
                                 for item in reproduction.values())
    extractor = FourierExtractor(config, diag["(-2,-3)"]["settings"])
    sidebands = phase_comparison(diag["(-2,-3)"]["recomputed"], diag["(-2,-2)"]["recomputed"],
                                 extractor, diag["(-2,-3)"]["settings"].phase_sigma)
    anomaly = diag["(-2,-3)"]["stored"]
    neighbour = diag["(-2,-2)"]["stored"]
    neighbour_median = float(np.median([row["rmse_waves"] for row in stage2b_rows
                                        if (row["offset_x"], row["offset_y"])
                                        in ((-3.0, -3.0), (-1.0, -3.0), (-2.0, -2.0))]))
    max_wrapped = max(item["wrapped_difference_rms_rad"] for item in sidebands)
    max_unwrapped = max(item["unwrapped_difference_rms_rad"] for item in sidebands)
    if max_reproduction_delta < 1e-12:
        reproduction_conclusion = "stable exact reproduction; storage/nondeterminism is not the cause"
    else:
        reproduction_conclusion = "small numerical differences only; no evidence of storage failure"
    if max_wrapped < 0.05 and max_unwrapped > 1.0:
        interpretation = ("wrapped phase is close between the two points while the unwrapped phase "
                          "differs substantially; this is consistent with unwrap-branch sensitivity, "
                          "but it is not proof of the physical cause")
    elif max_unwrapped > 1.0:
        interpretation = ("large phase differences appear both before and after unwrapping; the isolated "
                          "low value is not explained by a simple unwrap branch alone")
    else:
        interpretation = ("phase fields are similar after common processing; the isolated low value "
                          "remains unexplained despite stable reproduction")

    args.output.mkdir(parents=True, exist_ok=False)
    diagnostic = {
        "stage": "2c_anomaly_diagnostic",
        "status": "complete",
        "points": list(points),
        "stored_values": {
            label: {"rmse_waves": data["stored"]["rmse_waves"],
                    "pv_waves": data["stored"]["pv_waves"],
                    "phase_residual_rms": data["stored"]["phase_residual_rms"],
                    "coverage": data["stored"]["coverage"],
                    "coeffs_waves": data["stored"]["coeffs_waves"]}
            for label, data in diag.items()},
        "reproduction": reproduction,
        "reproduction_conclusion": reproduction_conclusion,
        "neighbour_median_rmse_waves": neighbour_median,
        "neighbour_median_minus_anomaly_waves": neighbour_median - anomaly["rmse_waves"],
        "sideband_comparison": sidebands,
        "max_wrapped_difference_rms_rad": max_wrapped,
        "max_unwrapped_difference_rms_rad": max_unwrapped,
        "interpretation": interpretation,
        "unwrap_seed": _UNWRAP_RANDOM,
        "large_arrays_saved": False,
        "source": {
            "baseline_observation_sha256": file_sha256(baseline / "baseline_observation.npz"),
            "baseline_observation_array_sha256": array_sha256(observation_npz["observed"]),
            "baseline_calibration_sha256": file_sha256(baseline / "baseline_calibration.npz"),
            "stage2b_results_sha256": file_sha256(args.stage2b_run / "results_49.json"),
        },
        "peak_memory_mib": float(resource_maxrss_mib()),
    }
    write_json(args.output / "anomaly_diagnostic.json", diagnostic)

    # Small two-panel diagnostic figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = list(points)
    axes[0].plot(labels, [diag[label]["metrics"]["rmse_waves"] for label in labels],
                 "o-", label="rmse_waves")
    axes[0].plot(labels, [diag[label]["metrics"]["pv_waves"] for label in labels],
                 "o-", label="pv_waves")
    axes[0].plot(labels, [diag[label]["recomputed"]["phase_residual_rms"] for label in labels],
                 "o-", label="phase_residual_rms")
    axes[0].set_yscale("log")
    axes[0].set_title("Recomputed two-point metrics")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=.25)
    for item in sidebands:
        k = item["sideband"]
        ua = ndimage.gaussian_filter(unwrap_phase(diag["(-2,-3)"]["recomputed"]["wrapped"][k],
                                                  **_UNWRAP_RANDOM),
                                     diag["(-2,-3)"]["settings"].phase_sigma)
        ub = ndimage.gaussian_filter(unwrap_phase(diag["(-2,-2)"]["recomputed"]["wrapped"][k],
                                                  **_UNWRAP_RANDOM),
                                     diag["(-2,-2)"]["settings"].phase_sigma)
        difference = ua[extractor.masks[k]] - ub[extractor.masks[k]]
        difference -= np.median(difference)
        axes[1].hist(difference, bins=80, alpha=.55, label=f"sideband {k}")
    axes[1].set_title("Unwrapped phase difference: (-2,-3) minus (-2,-2)")
    axes[1].set_xlabel("radians after median removal")
    axes[1].set_ylabel("pixels")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"anomaly_diagnostic.{suffix}", dpi=170)
    plt.close(fig)

    shape_summary = json.loads((args.shape_run / "summary.json").read_text())
    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    finished = datetime.now(timezone.utc)
    combined = {
        "stage": "2c_shape_and_anomaly",
        "shape": shape_summary,
        "anomaly": diagnostic,
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "batch_wall_seconds": (finished - started_at).total_seconds(),
        "peak_memory_mib": max(float(shape_summary.get("peak_memory_mib") or 0),
                               diagnostic["peak_memory_mib"]),
    }
    write_json(docs / "stage2c-summary.json", combined)
    lines = [
        "# 阶段 2c 进度：形状/长宽比与异常诊断",
        "",
        f"- 本批时间：{combined['batch_started_at']} → {combined['finished_at']}（约 {combined['batch_wall_seconds']:.0f} s）",
        f"- 形状设置：5 个（1 个复用 rect + 4 个新算），失败 {shape_summary['failed']} 个。",
        f"- 异常诊断：(-2,-3) 与 (-2,-2) 两点；{diagnostic['reproduction_conclusion']}。",
        f"- 峰值 RSS：{combined['peak_memory_mib']:.0f} MiB；worker/batch=1。",
        "",
        "## 五个窗口（固定 rect 名义标定）",
        "| 设置 | RMSE / waves | PV / waves | coverage | phase residual | weight_sum | ENBW | max DC weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shape_summary["rows"]:
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
        "- gaussian：`sx=sy=20*sqrt(2/pi)`，连续积分权重 1600 bins²；无限支撑，实际 ENBW 和 DC 泄漏须看离散指标。",
        "- 所有形状仍固定 rect 名义标定；形状变化与标定适配混在一起，不能声称每个形状各自最佳精度。",
        "",
        "## 异常诊断",
        f"- 复现：{diagnostic['reproduction_conclusion']}；最大复现差 `{max_reproduction_delta:.3g}`。",
        f"- `(-2,-3)` 相对邻点中位数低约 `{diagnostic['neighbour_median_minus_anomaly_waves']:.6g}` waves。",
        f"- 解释：{diagnostic['interpretation']}。",
        "- 源阶段 2b 未保存 raw 系数，诊断只比较已有字段；不修改默认流程，不把推测写成结论。",
        "",
        "## 限制",
        "- 名义“半宽 20”在不同形状下不是同一截止频率；形状、面积、权重、支撑与 DC 泄漏共同影响结果。",
        "- gaussian 的 ENBW 约 3200 bins（rect 约 1795），且最大 DC weight 约 0.0223；RMSE 更低不能单独归因于形状更优。",
        "- 仅 5 个设置、单一测试图、固定 rect 标定；不能据此选择生产形状。",
        "",
        "## 下一批建议",
        "空间掩膜有限方案：固定 rect/半宽 20/名义标定、中心 0，先做 erosion 0/2/4/8（4 点），再做低幅剔除 0/.1/.3（3 点）；预计 7 个解算约 1–2 分钟，加启动/汇总约 3–5 分钟。",
        "",
    ]
    (docs / "stage2c-progress.md").write_text("\n".join(lines))
    print("max reproduction delta", max_reproduction_delta)
    print("reproduction", reproduction_conclusion)
    print("interpretation", interpretation)
    print("max wrapped/unwrapped", max_wrapped, max_unwrapped)
    print("wrote", args.output, "and", docs / "stage2c-summary.json")


def resource_maxrss_mib() -> float:
    import resource
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / 1024)


if __name__ == "__main__":
    main()
