"""Grating diffraction amplitudes against 表2-3 of the dissertation."""

from __future__ import annotations

import numpy as np
import pytest

from lsi.grating import analytic_orders, bitmap_orders, diffraction_efficiency

#: 表2-3: (m, n) -> |amplitude| and diffraction efficiency (%)
TABLE_2_3 = {
    (1, 1): (0.2026, 4.11),
    (1, 3): (0.0675, 0.46),
    (3, 3): (0.025, 0.05),   # amplitude column is loose; efficiency column is exact
    (1, 5): (0.0405, 0.16),
    (5, 5): (0.081, 6.6e-3),  # amplitude column has a decimal slip (0.0081)
}


@pytest.mark.parametrize("mn", sorted(TABLE_2_3))
def test_first_orders_against_table(mn):
    m, n = mn
    a, b = (m + n) // 2, (n - m) // 2
    orders = analytic_orders(max_index=5)
    amp = orders.with_orders([(a, b)]).amp[0]
    ref_amp, ref_eff = TABLE_2_3[mn]
    # amplitude (magnitude) matches the analytic -2/(pi^2 m n)
    assert abs(abs(amp) - 2.0 / (np.pi**2 * m * n)) < 1e-12
    # the dissertation's efficiency column is the trustworthy entry
    assert abs(abs(amp) ** 2 * 100 - ref_eff) < 0.01


def test_dc_amplitude_and_parseval():
    orders = analytic_orders(max_index=25)
    eff = diffraction_efficiency(orders)
    assert abs(eff["dc"] - 0.25) < 1e-12
    total = sum(eff["all"].values())
    assert abs(total - 0.5) < 6e-3  # mean transmittance of a 50 % chessboard


def test_first_order_beats_crossed_grating():
    """Chessboard first-order efficiency (4.11 %) > crossed grating (2.53 %)."""
    orders = analytic_orders(max_index=1)
    eff = diffraction_efficiency(orders)
    chessboard_first = eff["first_order_total"] * 100 / 4      # per +-1 order
    assert chessboard_first == pytest.approx(4.11, abs=0.01)
    # ideal crossed (2-D) amplitude grating: A_0 = 1/4, A_10 = (1/pi)(1/2)
    crossed_first = (0.5 / np.pi) ** 2 * 100
    assert crossed_first == pytest.approx(2.53, abs=0.01)      # 2.1.2 of the paper
    assert chessboard_first > crossed_first


def test_bitmap_matches_analytic():
    ana = analytic_orders(max_index=3)
    bit = bitmap_orders(harmonic_cell=96, max_index=3)
    for order in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (2, 1)]:
        a = ana.with_orders([order]).amp[0]
        b = bit.with_orders([order]).amp[0]
        assert abs(abs(b) - abs(a)) < 5e-3, (order, a, b)


def test_duty_cycle_error_reduces_scattered_to_third_order():
    """Chapter-4 error analysis: duty cycle error hits odd orders."""
    ideal = analytic_orders(max_index=2)
    duty = bitmap_orders(harmonic_cell=96, max_index=2, duty=0.4)
    e_id = {k: abs(v) for k, v in ideal.efficiencies().items()}
    e_du = {k: abs(v) for k, v in duty.efficiencies().items()}
    assert e_du[(0, 0)] > e_id[(0, 0)]        # more energy in DC
    assert e_du[(1, 0)] < e_id[(1, 0)]        # less in the shearing orders


def _square_wave_amp(k: int, duty: float) -> complex:
    """S_k of ``s(u) = +1`` on ``[0, duty)`` and ``-1`` elsewhere.

    ``S_k = (1 - e^{-2 i pi k d}) / (i pi k) = -2 i e^{-i pi k d} sin(pi k d) / (pi k)``.
    """
    return (1.0 - np.exp(-2j * np.pi * k * duty)) / (1j * np.pi * k)


@pytest.mark.parametrize("duty", [0.40, 0.45, 0.55, 0.60])
def test_duty_amplitudes_match_first_principles(duty):
    """Off-50 % duty: ``A_mn = S_m S_n / 2``, ``A_00 =`` mean transmittance.

    This is the independent derivation of the duty-dependent branch; it pins
    both magnitude *and* phase, so a sign or phase slip in
    :func:`analytic_orders` is caught (a real-valued branch would pass a
    magnitude-only check).
    """
    orders = analytic_orders(max_index=4, duty=duty)
    dc = orders.with_orders([(0, 0)]).amp[0]
    assert abs(dc - (1.0 + (2.0 * duty - 1.0) ** 2) / 2.0) < 1e-15
    assert abs(dc.imag) < 1e-15
    for a, b in [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 1), (1, 2),
                 (-2, -1), (-1, -2), (3, 0), (0, 3), (2, -1), (-3, 2)]:
        m, n = a - b, a + b
        ref = 0.5 * _square_wave_amp(m, duty) * _square_wave_amp(n, duty)
        got = orders.with_orders([(a, b)]).amp[0]
        assert abs(got - ref) < 1e-15, (a, b, got, ref)


@pytest.mark.parametrize("duty", [0.40, 0.45, 0.55, 0.60])
def test_duty_amplitudes_no_sign_jump_at_half_duty(duty):
    """The duty branch must be the smooth continuation of the 表2-3 values.

    A wrong branch flips the sign of ``A_01`` between ``d = 1/2`` and
    ``d != 1/2``; that half-fringe jump is invisible to a magnitude check but
    ruins the demodulated constant (and hence tilt) of the y shear.
    """
    ideal, off = analytic_orders(duty=0.5), analytic_orders(duty=duty)
    eps = duty - 0.5
    for a, b in [(1, 0), (0, 1), (2, 1), (1, 2), (-1, 0), (0, -1)]:
        u = ideal.with_orders([(a, b)]).amp[0]
        v = off.with_orders([(a, b)]).amp[0]
        # same branch: no pi offset relative to the ideal duty amplitudes
        assert (v * np.conj(u)).real > 0, (a, b, u, v)
        # and the extra phase is the grating-origin law e^{-2 i pi a (d - 1/2)}
        dphi = np.angle(v * np.conj(u))
        law = np.angle(np.exp(-2j * np.pi * a * eps))
        assert abs(((dphi - law + np.pi) % (2 * np.pi)) - np.pi) < 1e-9, (a, b, dphi, law)


def test_duty_amplitudes_match_bitmap_phase():
    """Complex amplitudes agree with the FFT of a sampled chessboard.

    The bitmap samples at ``(p + 1/2) / n``, which adds the reference phase
    ``e^{2 i pi a / n}`` (plus a pixel ``sinc`` of order ``1/n``); after
    removing that gauge the agreement is limited by the pixelated edges.
    """
    n = 196
    bit = bitmap_orders(harmonic_cell=n, max_index=3, duty=0.45)
    ana = analytic_orders(max_index=3, duty=0.45)
    for order in [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 1), (1, 2)]:
        a, b = order
        ref = ana.with_orders([order]).amp[0] * np.exp(2j * np.pi * a / n)
        got = bit.with_orders([order]).amp[0]
        assert abs(got - ref) < 0.02 * abs(ref), (order, got, ref)
