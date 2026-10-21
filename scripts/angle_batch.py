#!/usr/bin/env python
"""Bounded batch of angle geometries, reusing the zero-angle pilot calibrations.

The pilot (camera x +0.25 deg) already saved the nominal fixed/auto
calibrations and zero-angle baseline.  This script runs only the requested
remaining geometries.  For each geometry:
  known:   independent fixed/auto calibrations on the tilted model
  ignored: zero-angle fixed/auto calibrations, nominal model
  direct:  one zero initialisation, matching or nominal model respectively
No truth initialisation and no unknown-angle joint fit are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the repository importable when this file is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lsi.config import DiffractionConfig
from lsi.fourier import FourierCalibration
from lsi.study import Study, settings_for


def load_calibration(directory: Path, stem: str) -> FourierCalibration:
    meta = json.loads((directory / f"{stem}_calibration.json").read_text())
    saved = np.load(directory / f"{stem}_calibration.npz")
    return FourierCalibration(
        baseline=saved["baseline"], response=saved["response"],
        amplitude=float(saved["amplitude"]),
        indices=tuple(int(value) for value in saved["indices"]),
        settings=meta["settings"], config=meta["config"],
        reference_image=saved["reference_image"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
                        default=Path("output/diffraction/followup_20261021/angles_pilot_camera_x_pos025"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", nargs="+", choices=("camera", "grating"), required=True)
    parser.add_argument("--axes", nargs="+", choices=("x", "y"), required=True)
    parser.add_argument("--angles", nargs="+", type=float, required=True)
    parser.add_argument("--truth-rms", type=float, default=.1)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--samples", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new directory")

    pilot_rows = json.loads((args.pilot / "results.json").read_text())
    done = {(row.get("component"), row.get("axis"), float(row.get("angle_deg", 0)))
            for row in pilot_rows if row.get("component") not in (None, "zero")}
    nominal = {center: load_calibration(args.pilot, f"nominal_{center}")
               for center in ("fixed", "auto")}
    zero_observed = np.load(args.pilot / "angle_zero_observation.npz")["observed"]

    study_args = argparse.Namespace(
        command="angles_batch", profile="reference", config=None,
        output=str(args.output), allow_large_run=True,
        modes="7,25,11,15", workers=1, batch_sources=1, backend="separable",
        starts=1, screen_initials=0, samples=args.samples, max_nfev=args.max_nfev,
        seed=42, truth_rms=args.truth_rms, calibration_amplitude=.02,
        calibration_cache=None, recovery_tolerance=.01, snr_db=None,
        widths=[8, 12, 16, 20, 24, 28, 32], offsets=[-3, -2, -1, 0, 1, 2, 3],
        shapes=["rect", "circle", "ellipse", "gaussian"], erosions=[0, 2, 4, 8, 12],
        aspects=[.5, 1, 2], thresholds=[0, .1, .3, .5],
        recalibrate_windows=False, selection_validation_rms=.05,
        selection_validation_repeats=1, cross_angles=[],
        components=args.components, axes=args.axes, angles=args.angles,
        geometry=["known", "ignored"], center_modes=["fixed", "auto"],
        reference_directory="", all_sources=False,
    )
    config = DiffractionConfig.profile("reference")
    study = Study(study_args, config)
    study.manifest["stage"] = "stage3_angle_batch"
    study.manifest["pilot_source"] = str(args.pilot)
    study.manifest["zero_baseline_source"] = str(args.pilot / "angle_zero_observation.npz")
    study.manifest["nominal_calibrations"] = {
        center: {"path": str(args.pilot / f"nominal_{center}_calibration.npz"),
                 "settings": nominal[center].settings}
        for center in nominal}
    planned = [(component, axis, angle)
               for component in args.components for axis in args.axes for angle in args.angles
               if (component, axis, angle) not in done]
    study.manifest["geometries_planned"] = [list(item) for item in planned]
    study.manifest["geometries_skipped_already_in_pilot"] = [
        list(item) for item in sorted(done, key=lambda item: (str(item[0]), str(item[1]), item[2]))]
    study.manifest["direct_initialisation"] = "single zero start; no truth initialisation"
    study.manifest["known_geometry"] = "tilted model and independent fixed/auto calibrations"
    study.manifest["ignored_geometry"] = "nominal model and zero-angle fixed/auto calibrations"
    study.initialize()
    study.rows = []
    settings_by_center = {center: study.settings.__class__(**{**study.settings.to_dict(),
                                                              "center_mode": center})
                          for center in ("fixed", "auto")}
    zero_norm = float(np.linalg.norm(zero_observed))
    try:
        for component, axis, angle in planned:
            model_config = study.config.with_angle(component, angle, axis)
            model = study.make_model(model_config)
            name = f"{component}_{axis}_{angle:g}"
            image = study.observe(model, name)
            image_change = float(np.linalg.norm(image - zero_observed) / zero_norm)
            context = {"component": component, "axis": axis, "angle_deg": float(angle),
                       "image_change_relative_l2": image_change}
            known_cals = {}
            for center, settings in settings_by_center.items():
                known_cals[center], _ = study.calibrate(
                    model, settings, f"{name}_known_{center}", save_training_images=False)
            study.solve_direct(model, image, name + "_known", geometry="known", **context)
            for center, settings in settings_by_center.items():
                study.solve_fourier(model_config, image, known_cals[center], settings,
                                    name + "_known_" + center, geometry="known", **context)
            study.solve_direct(study.model, image, name + "_ignored", geometry="ignored", **context)
            for center, settings in settings_by_center.items():
                study.solve_fourier(study.config, image, nominal[center], settings,
                                    name + "_ignored_" + center, geometry="ignored", **context)
        if study.rows:
            study.plot_sweeps("angles")
        study.finish()
    except (KeyboardInterrupt, Exception) as error:
        study.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     str(error) or type(error).__name__)
        raise
    print("wrote", args.output, "rows", len(study.rows))


if __name__ == "__main__":
    main()
