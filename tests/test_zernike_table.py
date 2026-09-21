"""Validation of the Zernike basis against 表2-5 of the dissertation.

The dissertation tabulates the *differential* Zernike polynomials
``dZx = Z(x+s,y) - Z(x-s,y)`` and ``dZy = Z(x,y+s) - Z(x,y-s)`` for the first
16 indices.  Those analytic expressions pin down (a) the index ordering
(n, m), (b) the cos/sin pairing, (c) the radial normalization and (d) the
sign convention -- i.e. everything about the basis we use for wavefront
reconstruction.

Notes on the transcription
--------------------------
* The dissertation's row 11 lost its index digit during printing/OCR; it is
  the sin counterpart of row 10.
* Row 14's ``dZx`` prints ``20s3x2`` and ``10sx4`` where the analytic value is
  ``200 s^3 x^2`` and ``100 s x^4`` (two dropped zeros); the other seven terms
  of that row match term by term.
* Row 16's two long expressions are too corrupted to serve as a reference and
  are checked only for internal symmetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi.zernike import differential_zernike, radial_polynomial

S = 0.0731


def _d(j, x, y, direction):
    return differential_zernike(j, x, y, S, direction)


#: (j, dZx, dZy) exactly as printed in 表2-5 (only indices 1..13 are free of
#: OCR damage; see module docstring).
TABLE = {
    1: ("0", "0"),
    2: ("2*s", "0"),
    3: ("0", "2*s"),
    4: ("8*s*x", "8*s*y"),
    5: ("4*s*x", "-4*s*y"),
    6: ("4*s*y", "4*s*x"),
    7: ("-4*s+6*s**3+18*s*x**2+6*s*y**2", "12*s*x*y"),
    8: ("12*s*x*y", "-4*s+6*s**3+6*s*x**2+18*s*y**2"),
    9: ("-24*s*x+48*s**3*x+48*s*x**3+48*s*x*y**2",
        "-24*s*y+48*s**3*y+48*s*y**3+48*s*x**2*y"),
    10: ("2*s**3+6*s*x**2-6*s*y**2", "-12*s*x*y"),
    11: ("12*s*x*y", "-2*s**3+6*s*x**2-6*s*y**2"),
    12: ("-12*s*x+32*s**3*x+32*s*x**3", "12*s*y-32*s**3*y-32*s*y**3"),
    13: ("-12*s*y+16*s**3*y+48*s*x**2*y+16*s*y**3",
         "-12*s*x+16*s**3*x+16*s*x**3+48*s*x*y**2"),
    # row 14: dZx corrected for the two OCR digit drops, dZy verbatim
    14: ("20*s**5+200*s**3*x**2+40*s**3*y**2-24*s**3+100*s*x**4"
         "+120*s*x**2*y**2-72*s*x**2+20*s*y**4-24*s*y**2+6*s",
         "80*s**3*x*y+80*s*x**3*y+80*s*x*y**3-48*s*x*y"),
    15: ("80*s**3*x*y+80*s*x**3*y+80*s*x*y**3-48*s*x*y",
         "20*s**5+200*s**3*y**2+40*s**3*x**2-24*s**3+100*s*y**4"
         "+120*s*x**2*y**2-72*s*y**2+20*s*x**4-24*s*x**2+6*s"),
}


@pytest.mark.parametrize("j", sorted(TABLE))
def test_differential_zernike_matches_table(j):
    rng = np.random.default_rng(1234 + j)
    x = rng.uniform(-0.9, 0.9, 400)
    y = rng.uniform(-0.9, 0.9, 400)
    env = {"s": S, "x": x, "y": y, "np": np}
    exp_x = np.asarray(eval(TABLE[j][0], {"__builtins__": {}}, env), dtype=float)  # noqa: S307
    exp_y = np.asarray(eval(TABLE[j][1], {"__builtins__": {}}, env), dtype=float)  # noqa: S307
    got_x = _d(j, x, y, "x")
    got_y = _d(j, x, y, "y")
    scale = max(1.0, np.abs(exp_x).max(), np.abs(exp_y).max())
    assert np.max(np.abs(got_x - exp_x)) / scale < 1e-12
    assert np.max(np.abs(got_y - exp_y)) / scale < 1e-12


def test_piston_and_index_table():
    """j=1 is piston (zero differential); j=4 is defocus (8 s x)."""
    x = np.linspace(-0.5, 0.5, 21)
    assert np.allclose(_d(1, x, x, "x"), 0.0)
    assert np.allclose(_d(4, x, x, "x"), 8 * S * x)
    assert np.allclose(_d(4, x, x, "y"), 8 * S * x)


def test_radial_polynomials():
    rho = np.linspace(0.0, 1.0, 101)
    assert np.allclose(radial_polynomial(2, 0, rho), 2 * rho**2 - 1)
    assert np.allclose(radial_polynomial(3, 1, rho), 3 * rho**3 - 2 * rho)
    assert np.allclose(radial_polynomial(4, 0, rho), 6 * rho**4 - 6 * rho**2 + 1)
    assert np.allclose(radial_polynomial(6, 0, rho),
                       20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1)


def test_row16_symmetry():
    """dZx(j=16) must be dZy(j=16) with x <-> y swapped."""
    j = 16
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.8, 0.8, 200)
    y = rng.uniform(-0.8, 0.8, 200)
    assert np.allclose(_d(j, x, y, "x"), _d(j, y, x, "y"))


def test_difference_models_consistency():
    """one_sided and two_sided coincide to O(s^3) for smooth low-order terms."""
    x = np.linspace(-0.6, 0.6, 41)
    x2, y2 = np.meshgrid(x, x, indexing="xy")
    two = differential_zernike(2, x2, y2, S, "x", "two_sided")
    one = differential_zernike(2, x2, y2, S, "x", "one_sided_doubled")
    assert np.allclose(two, one)  # tilt: exact
    two = differential_zernike(7, x2, y2, S, "x", "two_sided")
    one = differential_zernike(7, x2, y2, S, "x", "one_sided_doubled")
    # coma: the two differ by s^2 * W_xx = 0.0577 wave at s = 0.0731, i.e. the
    # paper's one-sided reading of a carrier lobe is only an O(s^2) model.
    assert 0.05 < np.max(np.abs(two - one)) < 0.07
    tilt_d = differential_zernike(2, x2, y2, S, "x", "two_sided")
    tilt_o = differential_zernike(2, x2, y2, S, "x", "one_sided")
    assert np.allclose(tilt_o, tilt_d / 2)  # tilt: exact to first order