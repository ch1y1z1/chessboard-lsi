#!/usr/bin/env python
"""Compare frozen and per-window calibration on one independent validation set.

The same three independent mixed wavefronts (0.05/0.1/0.2 waves) are used for
both strategies and the common measured width set 8..32.  Test-image truth is
only used afterwards to show selection regret, never to choose a width.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
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


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_calibration(directory: Path, width: int) -> FourierCalibration:
    meta = json.loads((directory / f"width_{width}_calibration.json").read_text())
    saved = np.load(directory / f"width_{width}_calibration.npz")
    return FourierCalibration(
        baseline=saved["baseline"], response=saved["response"],
        amplitude=float(saved["amplitude"]),
        indices=tuple(int(value) for value in saved["indices"]),
        settings=meta["settings"], config=meta["config"],
        reference_image=saved["reference_image"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("work/handoff-20261021/output/diffraction/reference_compare"))
    parser.add_argument("--frozen-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_frozen"))
    parser.add_argument("--recal-run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_recal"))
    parser.add_argument("--combined", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_recal_supplement_w32/combined_widths.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage3c_selection_strategy"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    parser.add_argument("--batch-started-at", default="2026-10-21T15:20:00+00:00")
    args = parser.parse_args()
    started_at = datetime.fromisoformat(args.batch_started_at)
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new directory")

    base = args.baseline
    manifest = json.loads((base / "manifest.json").read_text())
    calibration_meta = json.loads((base / "baseline_calibration.json").read_text())
    base_npz = np.load(base / "baseline_calibration.npz")
    config = DiffractionConfig(**manifest["config"])
    indices = tuple(int(value) for value in manifest["indices"])
    model = DiffractionModel(config, indices)
    base_settings = FourierSettings(**calibration_meta["settings"])
    frozen = FourierCalibration(
        baseline=base_npz["baseline"], response=base_npz["response"],
        amplitude=float(base_npz["amplitude"]),
        indices=tuple(int(value) for value in base_npz["indices"]),
        settings=calibration_meta["settings"], config=manifest["config"],
        reference_image=base_npz["reference_image"])
    widths = [8, 12, 16, 20, 24, 28, 32]
    recal = {width: load_calibration(args.recal_run, width) for width in widths}
    combined = json.loads(args.combined.read_text())
    test_frozen = {float(row["half_width"]): row["frozen_nominal_calibration_rmse_waves"]
                   for row in combined["combined_widths"]}
    test_recal = {float(row["half_width"]): row["recalibrated_rmse_waves"]
                  for row in combined["combined_widths"]}

    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    for level, seed in ((.05, 9501), (.1, 9502), (.2, 9503)):
        rng = np.random.default_rng(seed)
        truth = rng.normal(size=len(indices))
        wavefront = model.wavefront(truth)
        truth *= level / np.std(wavefront[model.pupil])
        image = model.evaluate(truth)
        image /= image.max()
        record = {"level_waves": level, "seed": seed, "truth": truth.tolist(),
                  "frozen": {}, "recalibrated": {}}
        for width in widths:
            settings = replace(base_settings, half_width=width)
            frozen_fit = frozen.reconstruct(FourierExtractor(config, settings), image)
            recal_fit = recal[width].reconstruct(FourierExtractor(config, settings), image)
            record["frozen"][width] = wavefront_metrics(model, frozen_fit["coeffs_waves"], truth)["rmse_waves"]
            record["recalibrated"][width] = wavefront_metrics(model, recal_fit["coeffs_waves"], truth)["rmse_waves"]
        samples.append(record)
        np.savez_compressed(args.output / f"validation_level_{level:g}.npz",
                            observed=image, truth=truth)
    scores = {}
    for strategy in ("frozen", "recalibrated"):
        scores[strategy] = [{"half_width": width,
                             "validation_rmse_waves": float(np.mean([sample[strategy][width] for sample in samples])),
                             "per_sample": [sample[strategy][width] for sample in samples]}
                            for width in widths]
    selected = {strategy: min(rows, key=lambda row: row["validation_rmse_waves"])["half_width"]
                for strategy, rows in scores.items()}
    test_best = {strategy: min(test_frozen if strategy == "frozen" else test_recal,
                               key=(test_frozen if strategy == "frozen" else test_recal).get)
                 for strategy in ("frozen", "recalibrated")}
    summary = {
        "stage": "3c_selection_strategy",
        "status": "complete",
        "batch_started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": samples,
        "scores": scores,
        "selected_width": selected,
        "test_best_width_analysis_only": test_best,
        "test_image_rmse": {"frozen": test_frozen, "recalibrated": test_recal},
        "test_truth_used_for_selection": False,
        "validation_levels_waves": [.05, .1, .2],
        "validation_sample_count": len(samples),
        "note": "same independent samples and same measured width set for both strategies; test truth is post-hoc only",
    }
    write_json(args.output / "selection_strategy.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(widths, [row["validation_rmse_waves"] for row in scores["frozen"]],
            "o-", color="C0", label="frozen nominal: validation mean")
    ax.plot(widths, [row["validation_rmse_waves"] for row in scores["recalibrated"]],
            "s-", color="C2", label="per-window recalibrated: validation mean")
    ax.plot(widths, [test_frozen[width] for width in widths], "o--", color="C0", alpha=.45,
            label="frozen: test image (analysis only)")
    ax.plot(widths, [test_recal[width] for width in widths], "s--", color="C2", alpha=.45,
            label="recalibrated: test image (analysis only)")
    for strategy, color in (("frozen", "C0"), ("recalibrated", "C2")):
        ax.axvline(selected[strategy], color=color, ls=":", lw=1, alpha=.7)
    ax.set_xlabel("FFT window half-width / bins")
    ax.set_ylabel("Wavefront RMSE / waves")
    ax.set_yscale("log")
    ax.set_title("Same independent validation set, two calibration protocols")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"selection_strategy.{suffix}", dpi=170)
    plt.close(fig)

    docs = args.docs_out
    docs.mkdir(parents=True, exist_ok=True)
    write_json(docs / "stage3c-summary.json", summary)
    lines = ["# 阶段 3c 进度：选窗策略比较", "",
             f"- 时间：{summary['batch_started_at']} → {summary['finished_at']}",
             f"- 共用独立验证样本：{len(samples)} 个（0.05/0.1/0.2 waves），共同宽度集合 8–32。",
             f"- 冻结标定验证选中：{selected['frozen']} bin；逐窗重标定选中：{selected['recalibrated']} bin。",
             f"- 测试真值事后最优：冻结 {test_best['frozen']} bin；重标定 {test_best['recalibrated']} bin（不参与选择）。",
             "", "| 宽度 | 冻结验证均值 | 重标定验证均值 | 冻结测试 | 重标定测试 |",
             "|---:|---:|---:|---:|---:|"]
    for width in widths:
        frow = next(row for row in scores["frozen"] if row["half_width"] == width)
        rrow = next(row for row in scores["recalibrated"] if row["half_width"] == width)
        lines.append(f"| {width} | {frow['validation_rmse_waves']:.6g} | {rrow['validation_rmse_waves']:.6g} | "
                     f"{test_frozen[width]:.6g} | {test_recal[width]:.6g} |")
    lines += ["", "## 限制",
              "- 只有 3 个独立混合验证样本；不能据此估计普遍选窗成功率。",
              "- 验证幅度与测试幅度、波前组合不同；不能把差异单独归因于幅度。",
              "- 测试真值最优只作事后下界，不用于选择；生产规则尚未在真实数据上冻结。", ""]
    (docs / "stage3c-progress.md").write_text("\n".join(lines))
    print("selected", selected, "test_best", test_best)


if __name__ == "__main__":
    main()
