"""Validate the CUDA composed operator against CPU at both extreme NA values."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits
import torch

from gpu_backend import GPUCameraModel
from study import (HERE, REPO, INDICES, PROTOCOL, ExtendedCameraModel, Monitor,
                   config, digest, exclusive_study, geometry,
                   interrupt_on_termination, save, truth_for, utc)


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert head == args.expected_commit
    assert not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                       "--untracked-files=no"], text=True)
    assert args.allow_large_run and os.uname().nodename == "sjtu-gxyserver"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    checks = []
    gpu_name = torch.cuda.get_device_name(0)
    with exclusive_study(), interrupt_on_termination(), threadpool_limits(
            limits=PROTOCOL["blas_threads"]), fft.set_workers(PROTOCOL["fft_threads"]):
        with Monitor(args.output, "backend_validation") as monitor:
            for na in (PROTOCOL["na_values"][0], PROTOCOL["na_values"][-1]):
                monitor.update(na_out=na, stage="backend_setup")
                cpu = ExtendedCameraModel(config(na), INDICES, batch_sources=1, workers=1)
                gpu = GPUCameraModel(cpu)
                truth = truth_for(cpu)
                for label, coeff in (("zero", np.zeros(35)), ("truth", truth)):
                    monitor.update(stage="backend_full", field=label)
                    started = time.perf_counter()
                    reference = cpu.evaluate(coeff)
                    cpu_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    actual = gpu.evaluate(coeff)
                    gpu_seconds = time.perf_counter()-started
                    error = relative(actual, reference)
                    peak_error = float(np.max(abs(actual-reference))/reference.max())
                    assert error < 1e-9 and peak_error < 1e-9
                    row = dict(na_out=na, field=label, source_count=64,
                        intensity_relative_l2=error, peak_normalized_maximum_error=peak_error,
                        cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
                        speedup=cpu_seconds/gpu_seconds,
                        cpu_sha256=digest(reference), gpu_sha256=digest(actual))
                    rows.append(row)
                    save(args.output/"full_comparisons.json", rows)
                    monitor.event("backend_full_comparison", **row)
                    del reference, actual
                radius = geometry(cpu.config)["camera_pupil_radius_m"]
                rho = np.hypot(cpu.x_cam, cpu.y_cam)/radius
                candidates = np.flatnonzero(((rho > .4) & (rho < .9)).ravel())
                selected = np.sort(np.random.default_rng(9211).choice(candidates, 128, replace=False))
                monitor.update(stage="backend_jacobian")
                ref, ref_jac = cpu.evaluate(truth, jacobian=True, rows=selected)
                actual, actual_jac = gpu.evaluate(truth, jacobian=True, rows=selected)
                jac_errors = [relative(actual_jac[j], ref_jac[j]) for j in range(35)]
                assert relative(actual, ref) < 1e-9 and max(jac_errors) < 1e-7
                finite = []
                for j in (0, 6, 10, 24):
                    delta = np.zeros(35)
                    delta[j] = 1e-5
                    numerical = (gpu.evaluate(truth+delta, rows=selected)-
                                 gpu.evaluate(truth-delta, rows=selected))/(2e-5)
                    error = relative(actual_jac[j], numerical)
                    assert error < 1e-5
                    finite.append(dict(mode=j+1, relative_error=error))
                checks.append(dict(na_out=na, jacobian_column_relative_errors=jac_errors,
                    finite_difference_checks=finite, sample_count=128,
                    sample_sha256=digest(selected)))
                save(args.output/"jacobian_comparisons.json", checks)
                del cpu, gpu, rho, ref, ref_jac, actual, actual_jac
                gc.collect()
                torch.cuda.empty_cache()
            speedup = statistics.median(r["speedup"] for r in rows)
            assert speedup > 2, "Keep CPU production if the GPU does not materially accelerate it"
            save(args.output/"validation.json", dict(passed=True, commit=head,
                timestamp_utc=utc(), gpu_name=gpu_name,
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                torch=torch.__version__, cuda=torch.version.cuda,
                dtype="complex128 matrices and float64 intensities/Jacobians",
                tf32=False, deterministic_algorithms=True,
                gpu_backend_sha256=hashlib.sha256((HERE/"gpu_backend.py").read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest(),
                median_full_frame_speedup=speedup,
                full_comparisons=rows, jacobian_checks=checks,
                scope="Same discrete operator on CPU/CUDA; not an independent optical or grid-convergence validation."))


if __name__ == "__main__":
    main()
