#!/usr/bin/env python
"""Re-score saved nominal Fourier calibration on independent mixed wavefronts.

This answers a narrow question left open by the main window scan: does a
validation-selected window transfer when the validation aberration amplitude
is varied?  It reuses the already saved calibration and never looks at the
test-image truth.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lsi.config import DiffractionConfig
from lsi.fourier import FourierCalibration, FourierExtractor, FourierSettings
from lsi.metrics import wavefront_metrics
from lsi.propagation import DiffractionModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("output/diffraction/followup_20261021/regions_width_frozen"))
    parser.add_argument("--levels", type=float, nargs="+", default=[.05, .1, .2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--widths", type=float, nargs="+",
                        default=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    calibration_meta = json.loads((run / "nominal_calibration.json").read_text())
    config = DiffractionConfig(**manifest["config"])
    model = DiffractionModel(config, tuple(manifest["indices"]))
    saved = np.load(run / "nominal_calibration.npz")
    calibration = FourierCalibration(
        baseline=saved["baseline"], response=saved["response"],
        amplitude=float(saved["amplitude"]), indices=tuple(int(v) for v in saved["indices"]),
        settings=calibration_meta["settings"], config=manifest["config"],
        reference_image=saved["reference_image"])
    base_settings = FourierSettings(**calibration_meta["settings"])
    levels = {}
    images, truths = [], []
    for level in args.levels:
        scores = {float(width): [] for width in args.widths}
        for repeat in range(args.repeats):
            rng = np.random.default_rng(7000 + int(round(level * 1000)) * 10 + repeat)
            truth = rng.normal(size=len(model.indices))
            wavefront = model.wavefront(truth)
            truth *= level / np.std(wavefront[model.pupil])
            image = model.evaluate(truth)
            images.append(image)
            truths.append(truth)
            for width in args.widths:
                settings = replace(base_settings, half_width=width)
                fit = calibration.reconstruct(FourierExtractor(config, settings), image)
                score = wavefront_metrics(model, fit["coeffs_waves"], truth)
                scores[float(width)].append(score["rmse_waves"])
        rows = [{"half_width": width,
                 "validation_rmse_waves": float(np.mean(values)),
                 "validation_rmse_std_waves": float(np.std(values)),
                 "per_repeat": values}
                for width, values in scores.items()]
        selected = min(rows, key=lambda row: row["validation_rmse_waves"])["half_width"]
        levels[f"{level:g}"] = {"rms_waves": level, "repeats": args.repeats,
                                "scores": rows, "selected_width": selected}
    output = {"source_run": str(run), "calibration_policy": "frozen_nominal",
              "test_truth_used": False, "levels": levels}
    (run / "width_validation_levels.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(run / "width_validation_levels.npz",
                        observed=np.asarray(images), truth=np.asarray(truths))
    for level, item in levels.items():
        print(level, "selected", item["selected_width"],
              "scores", [round(row["validation_rmse_waves"], 7) for row in item["scores"]])


if __name__ == "__main__":
    main()
