#!/usr/bin/env python
"""Summarise the completed noise/RMS/initialisation run without recomputing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("output/diffraction/followup_20261021/stage3b_noise_init_v4"))
    parser.add_argument("--docs-out", type=Path, default=Path("docs/results/followup"))
    args = parser.parse_args()
    run = args.run
    rows = json.loads((run / "results.json").read_text())
    trials = json.loads((run / "init_stability_0.2_dir0_direct_trials.json").read_text())
    noise, rms = [], []
    for row in rows:
        if row.get("kind") == "noise" and row["method"] == "fourier_calibrated":
            direct = next(item for item in rows if item.get("kind") == "noise"
                          and item.get("case") == row["case"] and item["method"] == "direct")
            noise.append({"case": row["case"], "noise_seed": row.get("noise_seed", row.get("seed")),
                          "ft_rmse_waves": row["rmse_waves"],
                          "direct_rmse_waves": direct["rmse_waves"],
                          "direct_nfev": direct["nfev"], "direct_converged": direct["converged"]})
        if row.get("kind") == "rms" and row["method"] == "fourier_calibrated":
            direct = next(item for item in rows if item.get("kind") == "rms"
                          and item.get("case") == row["case"] and item["method"] == "direct")
            rms.append({"case": row["case"], "rms_waves": row.get("rms_waves"),
                        "direction": row.get("direction"),
                        "ft_rmse_waves": row["rmse_waves"],
                        "direct_rmse_waves": direct["rmse_waves"],
                        "direct_starts": direct.get("starts"),
                        "direct_nfev": direct["nfev"], "direct_converged": direct["converged"]})
    trial_rows = []
    for index, trial in enumerate(trials["trials"]):
        trial_rows.append({"trial": index, "converged": trial["converged"],
                           "nfev": trial["nfev"],
                           "rmse_waves": trial.get("metrics", {}).get("rmse_waves"),
                           "initial_vector_norm": float(np.linalg.norm(trial["initial"]))})
    summary = {
        "stage": "3b_noise_init",
        "status": "complete",
        "run": str(run),
        "row_count": len(rows),
        "noise": noise,
        "rms": rms,
        "initialisation": {"case": "init_stability_0.2_dir0", "starts": 5,
                           "screen_initials": 8, "trials": trial_rows,
                           "best_trial": trials["best_trial"],
                           "all_converged": all(row["converged"] for row in trial_rows)},
        "peak_memory_mib_observed": 1023.0,
        "batch_started_at_approx": "2026-10-21T15:54:30+00:00",
        "batch_finished_at": "2026-10-21T16:10:01+00:00",
        "batch_seconds_approx": 931,
        "noise_definition": "SNR = 20 log10(clean peak / Gaussian noise sd); peak=1 after normalisation",
        "noise_ft_calibration": "frozen zero-angle rect half-width-20 calibration",
        "direct_initialisation": {"noise": "single zero start", "rms": "zero + one independent random start",
                                  "stability": "zero + four candidates screened by observed intensity residual"},
        "truth_usage": "evaluation and amplitude scaling only; no truth initialisation",
        "limitations": [
            "40 dB uses three fixed noise seeds; this is not a general recovery-rate estimate.",
            "0.05/0.2/0.4 waves use two random mode directions; no exhaustive mode study.",
            "The 5-start stability case uses one 0.2-wave direction and observed-residual screening.",
        ],
    }
    args.docs_out.mkdir(parents=True, exist_ok=True)
    (args.docs_out / "stage3b-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# 阶段 3b 进度：噪声与初值稳定性", "",
             f"- 运行：`{run}`；19 行，0 失败；约 15.5 分钟；峰值 RSS 约 1.0 GiB（运行中观测）。",
             "", "## 40 dB 噪声（同一噪声实现）", "| seed | FT RMSE | direct RMSE | direct nfev |",
             "|---:|---:|---:|---:|"]
    for item in noise:
        lines.append(f"| {item['noise_seed']} | {item['ft_rmse_waves']:.6g} | "
                     f"{item['direct_rmse_waves']:.6g} | {item['direct_nfev']} |")
    lines += ["", "## 混合 RMS / 两方向", "| 设置 | FT RMSE | direct RMSE | direct starts | converged |",
              "|---|---:|---:|---:|---|"]
    for item in rms:
        lines.append(f"| {item['case']} | {item['ft_rmse_waves']:.6g} | "
                     f"{item['direct_rmse_waves']:.6g} | {item['direct_starts']} | {item['direct_converged']} |")
    lines += ["", "## 5 初值稳定性（0.2 waves, dir0）", "",
              "| trial | converged | nfev | RMSE | initial norm |", "|---:|---|---:|---:|---:|"]
    for item in trial_rows:
        lines.append(f"| {item['trial']} | {item['converged']} | {item['nfev']} | "
                     f"{item['rmse_waves']:.6g} | {item['initial_vector_norm']:.4g} |")
    lines += ["", "## 限制",
              "- 40 dB 三 seed 不是一般恢复率；0.05/0.2/0.4 只有两个方向；5 初值只覆盖一个 0.2-wave 方向。",
              "- FT 使用冻结零角度 rect 标定；0.2/0.4 waves 明显超出线性标定范围，不能把 FT 退化归因为单一因素。",
              "- direct 的极小误差只说明这些同模型无噪声/加噪算例成功，不外推仪器精度。", ""]
    (args.docs_out / "stage3b-progress.md").write_text("\n".join(lines))
    print("noise", len(noise), "rms", len(rms), "trials", len(trial_rows))


if __name__ == "__main__":
    main()
