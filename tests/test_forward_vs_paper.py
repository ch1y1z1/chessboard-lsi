"""The order-superposition forward model vs the dissertation's explicit
4-beam / 5-beam region formulas, eqs. (2-12) ... (2-15).

The dissertation writes the intensity inside each overlap region as an
explicit sum of cosine terms (with a common first-order amplitude
``A_{+1,+1}`` for all four first orders).  Expanding the superposition model

    I = | sum_ab A_ab exp(i[2 pi W(x+as, y+bs) + delta_ab]) |^2

over the same set of beams must reproduce those expressions *exactly* -- this
validates the beam-to-order mapping, the phase-shift law and the region
bookkeeping in one shot.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, paper_region_intensity
from lsi.grating import OrderSet
from lsi.zernike import zernike_value

A1 = 2.0 / np.pi**2          # 表2-3 first-order amplitude
A0 = 0.5                     # 表2-3 zero-order amplitude


class _W:
    """Smooth, non-separable test wavefront (waves)."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def w(self, x, y):
        W = 0.40 * zernike_value(7, x, y) - 0.25 * zernike_value(4, x, y)
        W = W + 0.18 * zernike_value(11, x, y) + 0.12 * zernike_value(5, x, y)
        W = W + 0.20 * np.exp(-3.0 * (x**2 + y**2)) * np.cos(4.0 * x)
        return self.scale * W


def _set(orders):
    ab = np.array(orders, dtype=int)
    amp = np.array([A0 if tuple(o) == (0, 0) else A1 for o in orders], dtype=complex)
    return OrderSet(ab, amp)


def _order_disc(grid, a, b, s):
    """Support of order (a, b): the unit disk translated by ``(-a s, -b s)``."""
    x, y = grid.coords()
    return (x + a * s) ** 2 + (y + b * s) ** 2 <= 1.0


REGIONS = {
    # region name -> (orders, dissertation region tag)
    "x1": ([(0, 0), (1, 0), (0, 1), (-1, 0)], "x1"),
    "x2": ([(0, 0), (1, 0), (-1, 0), (0, -1)], "x2"),
    "x5": ([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], "x5"),
    "y1": ([(0, 0), (1, 0), (0, 1), (0, -1)], "y1"),
    "y2": ([(0, 0), (-1, 0), (0, 1), (0, -1)], "y2"),
    "y5": ([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], "y5"),
}


@pytest.mark.parametrize("name", sorted(REGIONS))
@pytest.mark.parametrize("t", [0.0, 0.125, 0.3])
def test_superposition_equals_paper_region_formula(name, t):
    orders, tag = REGIONS[name]
    cfg = SystemConfig(grid=Grid(n=96, extent=1.10))
    fm = ForwardModel(cfg, _set(orders))
    w = _W()

    direction = "x" if name.startswith("x") else "y"
    deltas = fm.phase_shift_deltas(t, 0.0) if direction == "x" else fm.phase_shift_deltas(0.0, t)
    delta = 2.0 * np.pi * t
    I_model = fm.intensity(w, deltas=deltas)
    x, y = cfg.grid.coords()
    mask = np.ones(cfg.grid.shape, dtype=bool)
    for a, b in orders:
        mask &= _order_disc(cfg.grid, a, b, cfg.s)
    assert mask.sum() > 200

    I_paper = paper_region_intensity(tag, w.w, x, y, cfg.s, delta, A0=A0, A1=A1)
    diff = np.abs(I_model[mask] - I_paper[mask])
    scale = max(1e-12, np.abs(I_paper[mask]).max())
    assert diff.max() / scale < 1e-12


def test_phase_shift_law_matches_section_2_3_1():
    """Moving the grating by p/sqrt(2) (t=1) changes (+1,+1) by 2 pi and
    leaves (+1,-1) untouched; the pattern repeats for t=1."""
    cfg = SystemConfig(grid=Grid(n=64, extent=1.10))
    fm = ForwardModel(cfg, _set([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]))
    d = fm.phase_shift_deltas(1.0, 0.0)
    per_order = dict(zip(fm.indices, d))
    assert np.isclose(per_order[(1, 0)] % (2 * np.pi), 0.0)
    assert np.isclose(per_order[(-1, 0)] % (2 * np.pi), 0.0)
    assert per_order[(0, 1)] == 0.0 and per_order[(0, -1)] == 0.0
    w = _W()
    I0 = fm.intensity(w, deltas=fm.phase_shift_deltas(0.0, 0.0))
    I1 = fm.intensity(w, deltas=d)
    assert np.allclose(I0, I1)


def test_y_shift_only_moves_y_orders():
    cfg = SystemConfig(grid=Grid(n=64, extent=1.10))
    fm = ForwardModel(cfg, _set([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]))
    d = fm.phase_shift_deltas(0.0, 0.25)
    per_order = dict(zip(fm.indices, d))
    assert per_order[(1, 0)] == 0.0 and per_order[(-1, 0)] == 0.0
    assert np.isclose(per_order[(0, 1)], np.pi / 2)
    assert np.isclose(per_order[(0, -1)], -np.pi / 2)


def test_frequency_one_phase_is_half_the_two_sided_difference():
    """The core identity behind the whole pipeline: inside the +-1 shear
    region the demodulated (frequency-1) phase equals
    ``pi [W(x+s) - W(x-s)]`` (modulo an unobservable constant)."""
    from lsi.phaseshift import lsq_phase_shift, shear_region_masks
    from lsi.unwrap import wrap

    cfg = SystemConfig(grid=Grid(n=128, extent=1.10))
    fm = ForwardModel(cfg, _set([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]))
    w = _W(scale=0.6)
    frames = fm.phase_shift_frames(w, "x", 8)
    res = lsq_phase_shift(frames)
    region = shear_region_masks(cfg.grid, cfg.s)["region_x"]
    x, y = cfg.grid.coords()
    pred = np.pi * (w.w(x + cfg.s, y) - w.w(x - cfg.s, y))
    d = wrap(res.phase - pred)[region]
    assert np.abs(np.angle(np.mean(np.exp(1j * d)))).max() < 1e-9  # constant offset
    assert np.sqrt(np.mean(d**2)) < 1e-9