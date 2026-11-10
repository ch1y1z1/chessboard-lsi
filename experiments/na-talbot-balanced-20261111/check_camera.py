"""Tiny independent checks of enlarged camera coordinates and sampled Jacobian."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits

from study import BalancedConfig, DiffractionModel, ExtendedCameraModel


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = BalancedConfig(N_lattice=20, N_super_lattice=4,
        dx_grating=67e-9, up_sample_grating=2, dx_cam=100e-6,
        grating_period=670e-9, source_kind="single", camera_multiplier=3)
    indices = (7, 25, 11, 15)
    truth = np.array([.001, .003, -.001, .004])
    reference = DiffractionModel(replace(cfg, camera_multiplier=1), indices, batch_sources=1)
    extended = ExtendedCameraModel(cfg, indices, batch_sources=1)
    small = reference.evaluate(truth)
    large = extended.evaluate(truth)
    start = (extended.n-cfg.n)//2
    central = relative(large[start:start+cfg.n, start:start+cfg.n], small)
    assert central < 1e-9, central
    source = extended.source_field(cfg.sources[0])
    spectrum = fft.fftshift(fft.fft2(fft.ifftshift(source)))*extended.pupil*np.exp(
        2j*np.pi*extended.wavefront(truth))
    image = fft.fftshift(fft.ifft2(fft.ifftshift(spectrum)))
    exact = abs(extended.propagate_image(image))**2
    czt = relative(large, exact)
    assert czt < 1e-9, czt
    rows = np.sort(np.random.default_rng(3511).choice(large.size, 64, replace=False))
    sampled, jac = extended.evaluate(truth, jacobian=True, rows=rows)
    subset = relative(sampled, large.ravel()[rows])
    assert subset < 1e-10
    errors = []
    for j in range(len(indices)):
        delta = np.zeros(len(indices))
        delta[j] = 1e-6
        numerical = (extended.evaluate(truth+delta, rows=rows)-
                     extended.evaluate(truth-delta, rows=rows))/(2e-6)
        error = relative(jac[j], numerical)
        assert error < 1e-5, error
        errors.append(error)
    result = dict(passed=True, object_n=cfg.n, camera_n=extended.n,
        central_reference_relative_l2=central, whole_czt_relative_l2=czt,
        subset_relative_l2=subset, jacobian_relative_errors=errors,
        description="Tiny actual reference/separable/CZT/Jacobian checks; not a production precision result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(limits=4), fft.set_workers(2):
        main()
