"""Small FT-window screen using archived single-source calibration spectra.

Only four new window configurations are evaluated. The original rectangular
and equal-area circular results are reused after input and metric checks.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "experiments/ft-region-20261111"),
    str(ROOT / "experiments/na-talbot-balanced-20261111"),
]
from region_study import RegionFeatures, RegionSettings
from study import config, geometry
from lsi.basis import basis
from lsi.execution import exclusive_study, interrupt_on_termination

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
NAMES = ["legacy_rect", "legacy_circle", *PROTOCOL["new_windows"]]


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Monitor:
    def __init__(self, output, run_id, attempt_id):
        self.output = output
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.index = -1
        self.step_times = []
        now = utc()
        titles = ["Prepare and verify inputs", *PROTOCOL["new_windows"], "Verify and export"]
        self.state = dict(
            schema_version="1.0", run_id=run_id, attempt_id=attempt_id, seq=0,
            timestamp=now, started_at=now, state="running", execution="sequential",
            tasks=[], metrics={}, error=None,
        )
        for i, title in enumerate(titles):
            self.state["tasks"].append(dict(
                id=f"stage-{i + 1}", title=title, state="pending",
                progress=dict(completed=0, total=72 if 1 <= i <= 4 else 1,
                              unit="feature" if 1 <= i <= 4 else "stage"),
                eta=dict(remaining_seconds=None, method="unknown", updated_at=now),
                dimensions={}, metrics={},
            ))
        self.emit()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def emit(self):
        with self.lock:
            self.state["seq"] += 1
            self.state["timestamp"] = utc()
            self.state["metrics"]["elapsed_seconds"] = time.monotonic() - self.started
            line = json.dumps(self.state, ensure_ascii=False, allow_nan=False)
            with (self.output / "monitor.jsonl").open("a") as f:
                f.write(line + "\n")
            save(self.output / "status.json", self.state)

    def heartbeat(self):
        while not self.stop.wait(PROTOCOL["heartbeat_seconds"]):
            self.emit()

    def begin(self, index):
        with self.lock:
            if index and self.state["tasks"][index - 1]["state"] != "succeeded":
                raise RuntimeError("previous stage has not succeeded")
            self.index = index
            self.step_times = []
            self.stage_started = time.monotonic()
            self.state["tasks"][index]["state"] = "running"
            self.emit()

    def advance(self, completed, seconds=None, **metrics):
        with self.lock:
            task = self.state["tasks"][self.index]
            task["progress"]["completed"] = completed
            task["metrics"].update(metrics)
            if seconds is not None:
                self.step_times.append(seconds)
                per_step = float(np.mean(self.step_times[-12:]))
                remaining = task["progress"]["total"] - completed
                now = utc()
                task["eta"] = dict(remaining_seconds=remaining * per_step,
                                   method="rate", updated_at=now)
                # Budgets for later windows remain separate stage estimates.
                for later in self.state["tasks"][self.index + 1:5]:
                    if later["state"] == "pending":
                        later["eta"] = dict(remaining_seconds=72 * per_step,
                                            method="historical", updated_at=now)
            self.state["last_progress_at"] = utc()
            self.emit()

    def finish_stage(self):
        with self.lock:
            task = self.state["tasks"][self.index]
            task["progress"]["completed"] = task["progress"]["total"]
            task["state"] = "succeeded"
            task["eta"] = dict(remaining_seconds=0, method="reported", updated_at=utc())
            task["metrics"]["seconds"] = time.monotonic() - self.stage_started
            self.emit()

    def close(self, error=None, cancelled=False):
        self.stop.set()
        self.thread.join()
        with self.lock:
            if error:
                self.state["state"] = "cancelled" if cancelled else "failed"
                self.state["error"] = dict(code="cancelled" if cancelled else "execution_error",
                                          message=str(error)[:1200])
                if self.index >= 0:
                    self.state["tasks"][self.index]["state"] = self.state["state"]
                for task in self.state["tasks"]:
                    if task["state"] != "succeeded":
                        task["eta"] = dict(remaining_seconds=None, method="unknown",
                                           updated_at=utc())
            else:
                assert all(t["state"] == "succeeded" for t in self.state["tasks"])
                self.state["state"] = "succeeded"
            self.emit()


def radial_tukey(rho, alpha):
    w = np.zeros_like(rho)
    w[rho <= 1 - alpha] = 1
    taper = (rho > 1 - alpha) & (rho <= 1)
    w[taper] = .5 * (1 + np.cos(np.pi * (rho[taper] - 1 + alpha) / alpha))
    return w


def window(n, center, name, half_width=112):
    """Sample continuous, centered weights without changing FFT dimensions."""
    cx, cy = center
    radius = 2 * half_width / np.sqrt(np.pi)
    extent = half_width if name == "hann_separable" else radius
    x0, x1 = max(0, int(np.floor(cx - extent))), min(n, int(np.ceil(cx + extent)) + 1)
    y0, y1 = max(0, int(np.floor(cy - extent))), min(n, int(np.ceil(cy + extent)) + 1)
    dx = np.arange(x0, x1)[None, :] - cx
    dy = np.arange(y0, y1)[:, None] - cy
    if name == "hann_separable":
        wx = np.where(abs(dx) <= half_width, .5 * (1 + np.cos(np.pi * dx / half_width)), 0)
        wy = np.where(abs(dy) <= half_width, .5 * (1 + np.cos(np.pi * dy / half_width)), 0)
        block = wx * wy
    else:
        rho = np.hypot(dx, dy) / radius
        if name == "hann_radial":
            block = np.where(rho <= 1, .5 * (1 + np.cos(np.pi * rho)), 0)
        elif name == "tukey_radial":
            block = radial_tukey(rho, PROTOCOL["tukey_alpha"])
        elif name == "gaussian_truncated":
            block = np.exp(-.5 * (rho / PROTOCOL["gaussian_sigma_over_radius"])**2)
            block *= radial_tukey(rho, PROTOCOL["gaussian_edge_alpha"])
        else:
            raise ValueError(name)
    result = np.zeros((n, n), dtype=np.float64)
    result[y0:y1, x0:x1] = block
    return result


def verify_windows():
    """A known isolated carrier must keep its phase along both FFT axes."""
    n, half, carrier, phase = 128, 10, 21, .7
    yy, xx = np.indices((n, n))
    for name in PROTOCOL["new_windows"]:
        for axis, coord in [(1, xx), (0, yy)]:
            image = 1 + .5 * np.cos(2 * np.pi * carrier * coord / n + phase)
            center = (n // 2 + (carrier if axis == 1 else 0),
                      n // 2 + (carrier if axis == 0 else 0))
            h = window(n, center, name, half)
            signal_field = fft.ifft2(fft.ifftshift(np.roll(
                fft.fftshift(fft.fft2(image)) * h, -carrier, axis)))
            assert h[center[1], center[0]] == 1
            assert np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1
            assert np.max(abs(np.angle(signal_field) - phase)) < 1e-12, (name, axis)
    rho = np.array([0, .25, .5, .75, 1, 1.1])
    assert np.array_equal(radial_tukey(rho, .5)[:3], np.ones(3))
    assert radial_tukey(rho, .5)[-1] == 0


class Evaluation:
    """The original native pupil and basis, without constructing propagation."""
    def __init__(self, c):
        self.c = c
        freq = fft.fftshift(fft.fftfreq(c.n, c.super_lattice_period / c.N_lattice))
        fx, fy = np.meshgrid(freq, freq)
        rho = c.wavelength * np.hypot(fx, fy) / c.NA_out
        theta = np.arctan2(fy, fx)
        self.mask = (rho >= c.epsilon) & (rho <= 1)
        assert self.mask.sum() == 52428
        self.rho = rho[self.mask]
        self.z = basis(tuple(range(1, 36)), self.rho, theta[self.mask], c.epsilon, c.Nz)
        iy, ix = np.nonzero(self.mask)
        self.box = (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))

    def values(self, coefficients):
        return np.asarray(coefficients) @ self.z

    def metrics(self, coefficients, truth):
        error = self.values(np.asarray(coefficients) - truth)
        return dict(
            raw_rms_waves=float(np.sqrt(np.mean(error**2))),
            raw_pv_waves=float(np.ptp(error)),
            relative_rms=float(np.sqrt(np.mean(error**2)) /
                               np.sqrt(np.mean(self.values(truth)**2))),
            coefficient_max_error_waves=float(np.max(abs(np.asarray(coefficients) - truth))),
            evaluation_pixels=int(self.mask.sum()),
        )

    def map(self, values):
        image = np.full(self.mask.shape, np.nan)
        image[self.mask] = values
        return image[self.box]


def install_window(features, name):
    features.windows = []
    features.references = []
    features.window_metrics = []
    features.rs = replace(features.rs, shape=name)
    for k, axis in enumerate((1, 0)):
        h = window(features.c.camera_n, features.centers[k], name, features.half_width)
        features.windows.append(h)
        features.references.append(fft.ifft2(fft.ifftshift(np.roll(
            features.zero_spec * h, -features.carrier, axis)))[features.roi].copy())
        s1, s2 = float(h.sum()), float(np.sum(h * h))
        features.window_metrics.append(dict(
            weight_sum=s1, weight_squared_sum=s2, N_eff=s1*s1/s2,
            center_weight=float(h[int(features.centers[k][1]), int(features.centers[k][0])]),
            dc_weight=float(h[h.shape[0]//2, h.shape[1]//2]),
            half_width_bins=features.half_width,
            radial_radius_bins=2*features.half_width/np.sqrt(np.pi),
        ))


def export(output, rows, evaluation, truth):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    baseline = rows[0]["raw_rms_waves"]
    fields = ["window", "origin", "raw_rms_waves", "ratio_to_default",
              "change_percent", "calibration_condition", "seconds"]
    for row in rows:
        row["ratio_to_default"] = row["raw_rms_waves"] / baseline
        row["change_percent"] = 100 * (row["ratio_to_default"] - 1)
    save(output / "results.json", rows)
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    maps = np.stack([evaluation.map(evaluation.values(np.array(r["coefficients"]) - truth))
                     for r in rows])
    np.savez_compressed(output / "error-maps.npz", names=np.array(NAMES),
                        error_waves=maps, truth_waves=evaluation.map(evaluation.values(truth)))
    labels = ["Soft rectangle", "Circle", "Hann (separable)", "Hann (radial)",
              "Tukey (radial)", "Truncated Gaussian"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout="constrained")
    values = [r["raw_rms_waves"] * 1e4 for r in rows]
    bars = ax.barh(labels, values, color=["#698294", "#698294", *["#1b7390"]*4])
    ax.invert_yaxis()
    ax.set_xlabel("Full-pupil wavefront RMS error (1e-4 waves)")
    ax.set_title("Initial window screen: single source, 40 dB, seed 42")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4)
    ax.set_xlim(0, max(values)*1.16)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    limit = float(np.nanmax(abs(maps)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, label, image in zip(axes.flat, labels, maps):
        im = ax.imshow(image * 1e4, cmap="RdBu_r", vmin=-limit*1e4, vmax=limit*1e4)
        ax.set_title(label)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="Wavefront error (1e-4 waves)", shrink=.8)
    fig.savefig(output / "error-maps.png", dpi=180)
    plt.close(fig)
    report = "# FT window initial screen\n\n"
    report += "Single source; original main wavefront; 40 dB noise, seed 42. "
    report += "Six concrete implementations at the archived window scale. "
    report += "Each new window is independently recalibrated. Bandwidth is not matched.\n\n"
    report += "| Window | RMS (waves) | Change vs default | Origin |\n|---|---:|---:|---|\n"
    for label, row in zip(labels, rows):
        report += f"| {label} | {row['raw_rms_waves']:.9g} | {row['change_percent']:+.2f}% | {row['origin']} |\n"
    report += "\n![RMS comparison](comparison.png)\n\n![Error maps](error-maps.png)\n\n"
    report += "This is one noise realization and one wavefront, without bandwidth matching. "
    report += "The ranking does not establish a general optimum or isolate geometric shape from weighting.\n"
    (output / "report.md").write_text(report)


def run(args, monitor):
    monitor.begin(0)
    verify_windows()
    c = config(PROTOCOL["na_out"], "single")
    cache = args.cache / "na0.020_single"
    observed_folder = args.run08 / "na0.020_single"
    saved_config = json.loads((observed_folder / "config.json").read_text())
    assert c.to_dict() == saved_config, "configuration differs from archived run-08"
    expected_keys = ["zero"] + [f"j{j:02d}{tag}" for j in range(1, 36) for tag in ("p", "m")]
    inputs = {}
    for key in expected_keys:
        path = cache / f"calib_spec_{key}.npy"
        a = np.load(path, mmap_mode="r")
        assert a.shape == (c.camera_n, c.camera_n) and a.dtype == np.complex128
        inputs[path.name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                 file_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        del a
    observation = np.load(observed_folder / "observation_40db_seed42.npy")
    observed_sha = array_hash(observation)
    baseline_json = json.loads((observed_folder / "ft_40db_seed42.json").read_text())
    assert observed_sha == baseline_json["observed_sha256"], "observation array hash mismatch"
    truth = np.array(json.loads((cache / "main_truth.json").read_text())["coefficients"])
    assert np.array_equal(truth, baseline_json["truth"])
    old_rows = json.loads((args.cache / "sweep_results.json").read_text())
    rows = []
    evaluation = Evaluation(c)
    for name, group, old_name in [
        ("legacy_rect", "E0_baseline", "baseline"),
        ("legacy_circle", "E3_shape", "circle_s0"),
    ]:
        hits = [r for r in old_rows if r.get("kind") == "single" and r.get("group") == group
                and r.get("name") == old_name and r.get("condition") == "40db_seed42"
                and "error" not in r]
        assert len(hits) == 1
        old = hits[0]
        assert np.array_equal(old["truth"], truth)
        check = evaluation.metrics(old["coefficients"], truth)
        assert np.isclose(check["raw_rms_waves"], old["raw_rms_waves"], rtol=1e-8, atol=1e-12)
        if name == "legacy_rect":
            assert np.max(abs(np.array(old["coefficients"]) - baseline_json["coefficients"])) < 1e-12
        rows.append(dict(window=name, origin="archived run-01", **check,
                         coefficients=old["coefficients"], truth=truth.tolist(),
                         calibration_condition=old["cal_condition"], seconds=0,
                         archived_settings=old["settings"], observed_sha256=observed_sha))
    save(args.output / "results.json", rows)
    zero = np.load(observed_folder / "zero.npy")
    features = RegionFeatures(c, zero, RegionSettings())
    assert features.half_width == 112 and features.shear == 54 and features.carrier == 282
    del zero
    observation_spec = features.spectrum(observation)
    del observation
    save(args.output / "inputs.json", dict(
        observed_sha256=observed_sha, config_sha256=file_hash(observed_folder / "config.json"),
        old_results_sha256=file_hash(args.cache / "sweep_results.json"),
        truth=truth.tolist(), calibration_spectra=inputs, geometry=geometry(c)))
    monitor.advance(1)
    monitor.finish_stage()
    for index, name in enumerate(PROTOCOL["new_windows"], 1):
        monitor.begin(index)
        started = time.monotonic()
        install_window(features, name)
        features_by_key = {}
        count = 0
        for key in expected_keys:
            t0 = time.monotonic()
            path = cache / f"calib_spec_{key}.npy"
            sp = np.load(path)
            if index == 1:
                inputs[path.name]["array_sha256"] = array_hash(sp)
            vector, diag = features.extract_from_spectrum(sp)
            del sp
            assert np.isfinite(vector).all()
            features_by_key[key] = vector
            count += 1
            monitor.advance(count, time.monotonic() - t0, window=name,
                            calibration_key=key, completed_windows=index-1)
        response = np.stack([
            (features_by_key[f"j{j:02d}p"] - features_by_key[f"j{j:02d}m"]) /
            (2 * PROTOCOL["calibration_amplitude_waves"]) for j in range(1, 36)
        ], axis=1)
        assert np.linalg.matrix_rank(response) == 35, "rank-deficient calibration"
        target_feature, diagnostics = features.extract_from_spectrum(observation_spec)
        coefficients = np.linalg.lstsq(response, target_feature - features_by_key["zero"], rcond=None)[0]
        row = dict(
            window=name, origin="new recalibration", **evaluation.metrics(coefficients, truth),
            coefficients=coefficients.tolist(), truth=truth.tolist(),
            calibration_condition=float(np.linalg.cond(response)),
            calibration_rank=35, calibration_singular_values=np.linalg.svd(response, compute_uv=False).tolist(),
            window_metrics=features.window_metrics, diagnostics=diagnostics,
            observed_sha256=observed_sha, seconds=time.monotonic()-started,
            code_commit=args.expected_commit, protocol_sha256=file_hash(HERE/"protocol.json"),
        )
        np.savez_compressed(args.output / f"calibration-{name}.npz", response=response,
                            zero_feature=features_by_key["zero"], target_feature=target_feature,
                            **features_by_key)
        rows.append(row)
        save(args.output / "results.json", rows)
        save(args.output / "calibration-inputs.json", inputs)
        monitor.advance(72, window=name, raw_rms_waves=row["raw_rms_waves"])
        monitor.finish_stage()
        print(json.dumps(dict(window=name, rms=row["raw_rms_waves"], seconds=row["seconds"])), flush=True)
        gc.collect()
    monitor.begin(5)
    assert len(rows) == 6 and [r["window"] for r in rows] == NAMES
    assert all(np.isfinite(r["raw_rms_waves"]) for r in rows)
    export(args.output, rows, evaluation, truth)
    save(args.output / "audit.json", dict(
        result_count=6, new_windows=4, reused_windows=2, native_pupil_pixels=int(evaluation.mask.sum()),
        observed_sha256=observed_sha, all_result_conditions="single / 40db_seed42",
        calibration_payloads_hashed=len(inputs),
        scope="one condition; archived metrics recomputed; no optical propagation",
    ))
    monitor.advance(1)
    monitor.finish_stage()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run08", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", default="ft-window-screen-20261116-run-01")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_large_run:
        parser.error("explicit --allow-large-run is required")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError("actual code commit differs from recorded commit")
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "version.json", dict(
        experiment=PROTOCOL["experiment"], commit=commit, command=sys.argv,
        start_utc=utc(), hostname=platform.node(), python=sys.version,
        numpy=np.__version__, protocol=PROTOCOL,
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in [
            Path(__file__), HERE/"protocol.json",
            ROOT/"experiments/ft-region-20261111/region_study.py",
            ROOT/"experiments/na-talbot-balanced-20261111/study.py",
            ROOT/"lsi/basis.py", ROOT/"lsi/fourier.py",
        ]},
    ))
    monitor = Monitor(args.output, args.run_id, args.attempt_id)
    try:
        with exclusive_study(), interrupt_on_termination(), \
             threadpool_limits(limits=PROTOCOL["threads"]), fft.set_workers(PROTOCOL["threads"]):
            run(args, monitor)
        monitor.close()
    except BaseException as error:
        traceback.print_exc()
        monitor.close(error, isinstance(error, KeyboardInterrupt))
        raise


if __name__ == "__main__":
    main()
