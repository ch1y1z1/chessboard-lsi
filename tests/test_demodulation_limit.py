"""Where the phase-shift / demodulation route stops being exact.

The linear route (frames -> demodulated phase -> unwrap -> Zernike fit) is exact
to machine precision for moderate aberrations, but loses whole waves once the
differential phase across the shear gets large.  These tests pin down the
mechanism so the degradation is documented rather than mistaken for a bug:

  * the unwrapper is *not* the limit -- handed the true wrapped phase, with the
    very same mask, it recovers the differential exactly;
  * harmonic-1 demodulation collects both (1,0).(0,0)* and (0,1).(0,0)*, so the
    orthogonal shear leaks in.  The leaked lobe drives the demodulated
    modulation |Z| through a null inside the usable (overlap) region;
  * at a null the wrapped phase carries vortices (residues), and unwrapping then
    loses whole waves.  Suppressing the (0,+-1) orders removes the null and the
    route stays exact at any amplitude.

Measured for a single Z7 with the shear of the paper (s = 0.0731): exact up to
~3 waves, null crossed between 3.00 and 3.25 waves (4 residues), then 1 wave
lost at 3.5 and 2 waves lost at >= 4 -- which is why the 6-wave case in
scripts/04 recovers Z7 = 5.674 where the LM route returns 6.00000 exactly.
See README section 4.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi import unwrap as U
from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.phaseshift import lsq_phase_shift, shear_region_masks
from lsi.pipeline import demodulate_phase_shift

GRID = Grid(n=96, extent=1.10)

#: beams (0,0), (+-1,0) only -> no orthogonal shear to leak in
X_ONLY_ORDERS = [(0, 0), (1, 0), (-1, 0)]


def _model(orders=None):
    cfg = SystemConfig(grid=GRID)
    return ForwardModel(cfg) if orders is None else ForwardModel(cfg, orders)


def _demodulate(fm, amp):
    """Return (mask, wrapped harmonic-1 phase, modulation |Z|, true dW)."""
    x, y = GRID.coords()
    wf = ZernikeWavefront(np.array([amp]), np.array([7]))
    fx = fm.phase_shift_frames(wf, "x", 8)
    if (0, 1) in fm.indices and (0, -1) in fm.indices:
        fy = fm.phase_shift_frames(wf, "y", 8)
        diff = demodulate_phase_shift(fm, fx, fy)
        mask = diff.mask["x"]
        phase = U.wrap(diff.phase["x"] - fm.demodulation_offset())
    else:
        # This is deliberately an x-only optical experiment used to isolate the
        # orthogonal-order crosstalk.  It is not a valid full 2-D public
        # pipeline input now that that API enforces symmetric pairs in both
        # directions, so demodulate its one available direction directly.
        result = lsq_phase_shift(fx)
        mask = shear_region_masks(GRID, fm.s)["region_x"]
        phase = U.wrap(result.phase - fm.demodulation_offset())

    n = fx.shape[0]
    z = (2.0 / n) * np.sum(
        fx * np.exp(-2j * np.pi * np.arange(n)[:, None, None] / n), axis=0
    )
    dw_true = amp * wf.terms(x + fm.s, y)[0] - amp * wf.terms(x - fm.s, y)[0]
    return mask, phase, np.abs(z), dw_true


def _residue_count(phase: np.ndarray, mask: np.ndarray) -> int:
    """Number of 2x2 loops whose wrapped curl is a non-zero multiple of 2*pi."""
    d1 = U.wrap(phase[1:, :] - phase[:-1, :])
    d2 = U.wrap(phase[1:, 1:] - phase[1:, :-1])
    d3 = U.wrap(phase[:-1, 1:] - phase[:-1, :-1])
    d4 = U.wrap(phase[1:, 1:] - phase[:-1, 1:])
    curl = (d1[:, :-1] + d2 - d3 - d4) / (2 * np.pi)
    curl = np.where(np.isfinite(curl), curl, 0.0)
    loops = np.round(curl).astype(int)
    inner = mask[1:, 1:] & mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
    return int(np.count_nonzero(loops[inner]))


def _dW_error(fm, mask, phase, dw_true):
    got = U.unwrap_seed_growth(phase, mask=mask)
    err = (got / np.pi - dw_true)[mask]
    return float(np.max(np.abs(err - np.median(err))))


def test_unwrap_is_not_the_limit_on_clean_phase():
    """Same mask, same gradient, but the *true* wrapped phase: exact recovery.

    At 6 waves of Z7 the differential spans ~12.5 waves, yet the per-pixel step
    stays below pi everywhere, so unwrapping the clean phase is well posed.
    """
    fm = _model()
    mask, _phase, _z, dw_true = _demodulate(fm, 6.0)
    phase_true = 2 * np.pi * dw_true

    step = np.abs(np.diff(phase_true, axis=1))
    inner = mask[:, :-1] & mask[:, 1:]
    assert np.max(step[inner]) < np.pi  # no aliasing anywhere in the mask
    assert _residue_count(U.wrap(phase_true), mask) == 0

    got = U.unwrap_seed_growth(U.wrap(phase_true), mask=mask)
    err = got[mask] - phase_true[mask]
    err -= np.median(err)
    assert np.max(np.abs(err)) < 1e-9


@pytest.mark.parametrize("amp", [0.5, 1.0, 2.0, 3.0])
def test_route_is_exact_below_the_null(amp):
    """Below the null the whole linear route is exact, leakage notwithstanding."""
    fm = _model()
    mask, phase, z, dw_true = _demodulate(fm, amp)
    assert _residue_count(phase, mask) == 0
    assert _dW_error(fm, mask, phase, dw_true) < 1e-9


def test_modulation_null_crossed_between_3_and_3_25_waves():
    """The 5-beam modulation reaches zero just past 3 waves of Z7."""
    mask_a, phase_a, z_a, _ = _demodulate(_model(), 3.0)
    mask_b, phase_b, z_b, _ = _demodulate(_model(), 3.25)

    assert np.min(z_a[mask_a]) > 1e-3  # no null yet
    assert np.min(z_b[mask_b]) < 1e-4  # null crossed
    assert _residue_count(phase_a, mask_a) == 0
    assert _residue_count(phase_b, mask_b) > 0


def test_orthogonal_shear_is_what_creates_the_null():
    """Drop the (0,+-1) orders and the null, residues and wave loss all vanish."""
    for amp in (3.25, 4.0):
        mask5, phase5, z5, dw5 = _demodulate(_model(), amp)
        mask3, phase3, z3, dw3 = _demodulate(_model(X_ONLY_ORDERS), amp)

        # 5-beam: null -> vortices -> unwrap loses whole waves
        assert np.min(z5[mask5]) < 1e-3
        assert _residue_count(phase5, mask5) > 0
        assert _dW_error(_model(), mask5, phase5, dw5) >= 0.5

        # 3-beam: same amplitude, no orthogonal shear -> still exact
        assert np.min(z3[mask3]) > 1e-2
        assert _residue_count(phase3, mask3) == 0
        assert _dW_error(_model(X_ONLY_ORDERS), mask3, phase3, dw3) < 1e-9