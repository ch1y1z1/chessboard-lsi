"""Compare padded compact features with the full-grid implementation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits

from study import BalancedConfig, ExtendedCameraModel, INDICES, Q1Features, rms, truth_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = BalancedConfig(N_lattice=20, N_super_lattice=4,
        dx_grating=67e-9, up_sample_grating=2, dx_cam=100e-6,
        grating_period=670e-9, source_kind="single", camera_multiplier=4.5)
    model = ExtendedCameraModel(cfg, INDICES, batch_sources=1)
    zero = model.evaluate(np.zeros(35))
    full = Q1Features(cfg, zero, compact=False)
    compact = Q1Features(cfg, zero, compact=True)
    assert full.basis_rank == compact.basis_rank
    inverse_error = float(np.max(abs(full.inverse-compact.inverse)))
    assert inverse_error < 1e-12
    differences = []
    for seed in (1807, 1808, 1809):
        observed = model.evaluate(truth_for(model, seed))
        for snr in (None, 30):
            image = observed.copy()
            if snr:
                image += 10**(-snr/20)*rms(image)*np.random.default_rng(seed).normal(size=image.shape)
            a, diag_a = full(image)
            b, diag_b = compact(image)
            error = float(np.max(abs(a-b)))
            assert error < 1e-12, error
            assert diag_a == diag_b
            differences.append(dict(seed=seed, snr_db=snr, maximum_feature_difference=error))
    result = dict(passed=True, full_grid_shape=list(zero.shape),
        compact_grid_shape=list(compact.masks[0].shape),
        maximum_inverse_difference=inverse_error, comparisons=differences,
        scope="Feature-operator parity only. Full observation FFT and sideband inverse FFT are unchanged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    with threadpool_limits(limits=4), fft.set_workers(2):
        main()
