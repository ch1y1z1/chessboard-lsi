"""Local parity check: RegionFeatures (baseline) vs study.Q1Features on a tiny grid."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "na-talbot-balanced-20261111"))
import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits
from study import (BalancedConfig, ExtendedCameraModel, INDICES, Q1Features,
                   truth_for, rms)
from region_study import RegionFeatures, RegionSettings


def main():
    cfg = BalancedConfig(N_lattice=20, N_super_lattice=4, dx_grating=67e-9,
        up_sample_grating=2, dx_cam=100e-6, grating_period=670e-9,
        source_kind="single", camera_multiplier=4.5)
    model = ExtendedCameraModel(cfg, INDICES, batch_sources=1)
    zero = model.evaluate(np.zeros(35))
    # run-08 Q1Features (compact ROI).
    q1 = Q1Features(cfg, zero, compact=True)
    # Mine at baseline settings (defaults reproduce run-08).
    rs = RegionSettings().resolved(
        q1.geometry["fundamental_bins"], q1.geometry["shear_pixels"])
    mine = RegionFeatures(cfg, zero, rs)
    print("Q1 carrier/shear/hw", q1.carrier, q1.shear, q1.settings.half_width,
          "mine", mine.carrier, mine.shear, mine.half_width)
    assert q1.carrier == mine.carrier and q1.shear == mine.shear
    assert q1.settings.half_width == mine.half_width
    assert abs(q1.settings.phase_sigma - mine.rs.phase_sigma) < 1e-9
    assert q1.basis_rank == mine.basis_rank
    assert abs(q1.basis_condition - mine.basis_condition) < 1e-9
    inv_err = float(np.max(abs(q1.inverse - mine.A_pinv)))
    print("inverse max abs diff", inv_err)
    assert inv_err < 1e-12
    diffs = []
    for seed in (1807, 1808, 1809):
        image = model.evaluate(truth_for(model, seed))
        a, da = q1(image)
        b, db = mine(image)
        e = float(np.max(abs(a - b)))
        print("seed", seed, "feature max diff", e, "small_phase", da["small_phase_valid"], db["small_phase_valid"])
        assert e < 1e-12, e
        diffs.append(e)
    # Noisy comparison.
    image = model.evaluate(truth_for(model, 1807))
    image += 10**(-30/20) * rms(image) * np.random.default_rng(42).normal(size=image.shape)
    a, _ = q1(image)
    b, _ = mine(image)
    e = float(np.max(abs(a - b)))
    print("noisy feature max diff", e)
    assert e < 1e-12
    print("PARITY OK")


if __name__ == "__main__":
    with threadpool_limits(limits=4), fft.set_workers(2):
        main()
