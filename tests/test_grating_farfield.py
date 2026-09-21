"""Far field of the 1-D, crossed and chessboard gratings.

Regression tests for the two former silent degeneracies:
``far_field_amplitude`` used to evaluate a 1-D pattern for every grating name
(the crossed and chessboard spectra were wrong), and its frequency axis was
``fftfreq(m, d=1.0)``, which made the ``orders`` crop a no-op.
"""

import numpy as np
import pytest

from lsi.grating import bitmap_orders, far_field_amplitude


def _orders(fx, fy, A, tol=1e-9):
    """Map ``(fx, fy) -> amplitude`` for every spectral sample above ``tol``."""
    out = {}
    for i, j in np.argwhere(A > tol):
        out[(round(float(fx[i, j]), 9), round(float(fy[i, j]), 9))] = float(A[i, j])
    return out


def test_first_order_efficiencies_match_a_square_wave():
    # duty = 0.5 square wave: first order = 2/pi of DC.  The chessboard is the
    # product of two such square waves, so its diagonal orders carry 4/pi^2.
    # A *sampled* hard edge sits a relative (pi/N)^2/6 above the ideal value
    # (1.3e-5 at N = 512 for the chessboard), hence the tolerance.
    rtol = 2e-4
    fx, fy, A = far_field_amplitude("grating1d")
    o = _orders(fx, fy, A)
    assert np.isclose(o[(1.0, 0.0)], 2 / np.pi, rtol=rtol)
    assert (0.0, 1.0) not in o

    fx, fy, A = far_field_amplitude("crossed")
    o = _orders(fx, fy, A)
    assert np.isclose(o[(1.0, 0.0)], 2 / np.pi, rtol=rtol)
    assert np.isclose(o[(1.0, 1.0)], 4 / np.pi**2, rtol=rtol)

    fx, fy, A = far_field_amplitude("chessboard")
    o = _orders(fx, fy, A)
    assert np.isclose(o[(1.0, 1.0)], 4 / np.pi**2, rtol=rtol)
    assert (1.0, 0.0) not in o
    assert (0.0, 1.0) not in o


def test_chessboard_keeps_only_diagonal_odd_orders():
    fx, fy, A = far_field_amplitude("chessboard")
    for i, j in np.argwhere(A > 1e-9):
        a, b = float(fx[i, j]), float(fy[i, j])
        assert a == int(a) and b == int(b)
        assert (int(a) % 2 == 1 and int(b) % 2 == 1) or (a == 0.0 and b == 0.0)


def test_crossed_and_chessboard_are_not_degenerate():
    # a 1-D pattern would leave the crossed and chessboard spectra identical
    f1, _, a1 = far_field_amplitude("grating1d")
    f2, _, a2 = far_field_amplitude("crossed")
    f3, _, a3 = far_field_amplitude("chessboard")
    assert not np.allclose(a1, a2)
    assert not np.allclose(a2, a3)
    # the crossed grating keeps the axial orders the chessboard suppresses
    assert (0.0, 1.0) in _orders(f2, far_field_amplitude("crossed")[1], a2)
    assert (0.0, 1.0) not in _orders(f3, far_field_amplitude("chessboard")[1], a3)
    del f1


def test_orders_actually_crops():
    for orders in (1, 3, 6, 12):
        fx, fy, A = far_field_amplitude("chessboard", orders=orders)
        assert A.shape == (2 * orders + 1, 2 * orders + 1)
        assert np.isclose(fx[0, 0], -orders)
        assert np.isclose(fy[0, 0], -orders)
    small = far_field_amplitude("chessboard", orders=3)[2]
    large = far_field_amplitude("chessboard", orders=6)[2]
    assert np.allclose(small, large[3:10, 3:10])


def test_nyquist_limit_warns_instead_of_silently_ignoring_orders():
    with pytest.warns(UserWarning, match="Nyquist"):
        far_field_amplitude("chessboard", orders=300, extent=1.0)
    with pytest.warns(UserWarning, match="Nyquist"):
        far_field_amplitude("chessboard", orders=12, extent=200.0)
    # a window that can resolve the request stays quiet
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        far_field_amplitude("chessboard", orders=12, extent=8.0)


def test_rotation_moves_the_first_orders_onto_the_diagonal_axis():
    # a 45-degree rotated chessboard is periodic along the detector axes only
    # over 2 periods, so it needs a wider window than the one-period default
    fx, fy, A = far_field_amplitude("chessboard", orders=6, rotation_deg=45.0, extent=25.0)
    B = A.copy()
    B[np.unravel_index(int(np.argmax(B)), B.shape)] = 0.0   # drop DC
    i, j = np.unravel_index(int(np.argmax(B)), B.shape)
    assert min(abs(fx[i, j]), abs(fy[i, j])) < 0.05          # on an axis
    assert np.isclose(max(abs(fx[i, j]), abs(fy[i, j])), np.sqrt(2), atol=0.05)


def test_rotation_changes_the_spectrum():
    flat = far_field_amplitude("chessboard", orders=4, extent=25.0)[2]
    rot = far_field_amplitude("chessboard", orders=4, extent=25.0, rotation_deg=30.0)[2]
    assert not np.allclose(flat, rot)
    # 360 degrees is the identity
    full = far_field_amplitude("chessboard", orders=4, extent=25.0, rotation_deg=360.0)[2]
    assert np.allclose(flat, full)


def test_far_field_input_validation():
    with pytest.raises(ValueError):
        far_field_amplitude("bogus")
    with pytest.raises(ValueError):
        far_field_amplitude("crossed", n=1)
    with pytest.raises(ValueError):
        far_field_amplitude("crossed", orders=0)
    with pytest.raises(ValueError):
        far_field_amplitude("crossed", duty=1.0)
    with pytest.raises(ValueError):
        far_field_amplitude("crossed", extent=0.0)


def test_bitmap_rotation_refuses_angles_the_integer_orders_cannot_hold():
    # The unit-cell FFT only *is* the detector-frame order set while the
    # sampled pattern stays commensurate with the lattice, i.e. for multiples
    # of 90 degrees.  At any other angle the energy scatters off the integer
    # bins and the ``m, n`` both-odd selection would keep a minority of it
    # (18 % of the non-DC energy at 45 degrees, where the strongest orders are
    # the axial bins that get dropped), so the amplitudes would not describe
    # the rotated grating.  Refuse instead of answering.
    for bad in (45.0, 20.0, -45.0, 1.0):
        with pytest.raises(NotImplementedError):
            bitmap_orders(rotation_deg=bad, max_index=2)
    with pytest.raises(ValueError):
        bitmap_orders(rotation_deg=float("nan"), max_index=2)


def test_bitmap_rotation_quarter_turns_are_chessboard_symmetries():
    def amps_by_order(o):
        return {tuple(int(v) for v in ab): a for ab, a in zip(o.ab, o.amp)}

    ref = amps_by_order(bitmap_orders(rotation_deg=0.0, max_index=2))
    # a half turn about a lattice origin leaves the chessboard unchanged
    for same in (180.0, 360.0):
        o = amps_by_order(bitmap_orders(rotation_deg=same, max_index=2))
        assert o.keys() == ref.keys()
        for k in ref:
            assert np.isclose(ref[k], o[k])
    # A quarter turn maps the pattern onto itself: the two orientations differ
    # by a half-cell origin shift, which cannot change any |A| -- that is the
    # physical content here.  Under this module's cell-origin convention the
    # shift also flips the phase of every odd order, so the sign is pinned as
    # well; it is a convention, not a law.
    for quarter in (90.0, 270.0, -90.0):
        o = amps_by_order(bitmap_orders(rotation_deg=quarter, max_index=2))
        assert o.keys() == ref.keys()
        for k in ref:
            if k == (0, 0):
                assert np.isclose(ref[k], o[k])          # DC: unchanged
            else:
                assert np.isclose(np.abs(ref[k]), np.abs(o[k]))
                assert np.isclose(ref[k], -o[k])         # odd orders: negated


def test_bitmap_orders_input_validation():
    with pytest.raises(ValueError):
        bitmap_orders(edge="abc")
    with pytest.raises(ValueError):
        bitmap_orders(harmonic_cell=1)
    with pytest.raises(ValueError):
        bitmap_orders(duty=1.5)
