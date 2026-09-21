"""Regressions for the forward model pupil and the differential Zernike fit.

Two former silent bugs are covered here:

* ``ForwardModel(pupil=...)`` stored the array but ``order_geometry`` always
  used the unit disk, so a custom aperture was silently ignored;
* ``fit_differential_zernike(fit_offsets=True)`` appended a single shared
  offset column, so two directions could not carry two offsets (and the
  returned ``offsets`` dict only ever had an ``"x"`` key).

``pupil`` also accepts a callable ``f(x, y) -> mask``.  An array pupil is
resampled by whole-pixel shifts, so a non-integer shear misplaces its edge by
up to half a pixel; the callable path evaluates the aperture at the true
shifted coordinates and is exact, and the tests below pin that down.
"""

import warnings

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.phaseshift import shear_region_masks
from lsi.reconstruct import differential_zernike_matrix, fit_differential_zernike


def _cfg(n=64, extent=1.0):
    return SystemConfig(grid=Grid(n=n, extent=extent))


def test_custom_pupil_is_honoured():
    cfg = _cfg()
    disc = ForwardModel(cfg)
    p = np.zeros((64, 64), dtype=bool)
    p[22:42, 22:42] = True                       # 20 x 20 square aperture
    # an array pupil is resampled by whole-pixel shifts, and this config's
    # shear is not a whole number of pixels; the model says so once
    with pytest.warns(UserWarning, match="not an integer"):
        square = ForwardModel(cfg, pupil=p)
    wf = ZernikeWavefront([0.1, 0.05], [2, 3])

    assert [int(m.sum()) for *_, m in square.order_geometry()] == [400] * 5
    assert int(disc.order_geometry()[0][4].sum()) > 3000
    assert not np.allclose(disc.intensity(wf), square.intensity(wf))

    with pytest.warns(UserWarning, match="not an integer"):
        dark = ForwardModel(cfg, pupil=np.zeros((64, 64), dtype=bool))
    assert np.allclose(dark.intensity(wf), 0.0)


def test_pupil_shape_is_validated():
    cfg = _cfg()
    with pytest.raises(ValueError, match="shape"):
        ForwardModel(cfg, pupil=np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        ForwardModel(cfg, pupil=np.ones((64, 64, 2), dtype=bool))


def test_float_pupil_is_thresholded():
    cfg = _cfg()
    soft = np.zeros((64, 64))
    soft[22:42, 22:42] = 1.0
    with pytest.warns(UserWarning, match="not an integer"):
        fm = ForwardModel(cfg, pupil=soft)
    with pytest.warns(UserWarning, match="not an integer"):
        ref = ForwardModel(cfg, pupil=soft > 0.5)
    assert np.array_equal(
        fm._pupil, ref._pupil
    )


def _synthetic_differential(offset_x=0.2, offset_y=-0.4):
    """A differential phase with *independent* per-direction offsets."""
    cfg = _cfg()
    fm = ForwardModel(cfg)
    x, y = cfg.grid.coords()
    indices = (4, 5, 6, 7)
    coeffs = np.array([0.12, -0.04, 0.03, 0.02])
    zx = differential_zernike_matrix(indices, x.ravel(), y.ravel(), fm.s, "x")
    zy = differential_zernike_matrix(indices, x.ravel(), y.ravel(), fm.s, "y")
    d_wx = (zx @ coeffs + offset_x).reshape(x.shape)
    d_wy = (zy @ coeffs + offset_y).reshape(y.shape)
    return cfg, fm, x, y, indices, coeffs, d_wx, d_wy


def test_offsets_are_estimated_per_direction():
    cfg, fm, x, y, indices, coeffs, d_wx, d_wy = _synthetic_differential()
    mask = np.ones_like(x, dtype=bool)
    fit = fit_differential_zernike(
        d_wx, d_wy, mask, mask, fm.s, x, y, indices=indices, fit_offsets=True
    )
    assert set(fit.offsets) == {"x", "y"}
    assert np.isclose(fit.offsets["x"], 0.2, atol=1e-9)
    assert np.isclose(fit.offsets["y"], -0.4, atol=1e-9)
    assert np.allclose(fit.coeffs, coeffs, atol=1e-9)
    assert fit.rms_residual < 1e-9


def test_per_direction_offsets_beat_a_shared_one():
    cfg, fm, x, y, indices, coeffs, d_wx, d_wy = _synthetic_differential()
    mask = np.ones_like(x, dtype=bool)
    two = fit_differential_zernike(
        d_wx, d_wy, mask, mask, fm.s, x, y, indices=indices, fit_offsets=True
    )
    none = fit_differential_zernike(
        d_wx, d_wy, mask, mask, fm.s, x, y, indices=indices, fit_offsets=False
    )
    assert two.rms_residual < none.rms_residual
    assert two.rms_residual < 1e-9


def test_offset_and_tilt_are_only_determined_up_to_a_gauge():
    # the tilt term has a constant difference (dZ/dx = 2 * shear), so it is
    # collinear with a per-direction constant: warn instead of silently
    # returning a min-norm answer
    cfg, fm, x, y, _, _, _, _ = _synthetic_differential()
    indices = (2, 3, 4)
    coeffs = np.array([0.1, -0.05, 0.03])
    zx = differential_zernike_matrix(indices, x.ravel(), y.ravel(), fm.s, "x")
    zy = differential_zernike_matrix(indices, x.ravel(), y.ravel(), fm.s, "y")
    d_wx = (zx @ coeffs + 0.2).reshape(x.shape)
    d_wy = (zy @ coeffs - 0.4).reshape(y.shape)
    mask = np.ones_like(x, dtype=bool)
    with pytest.warns(UserWarning, match="rank deficient"):
        fit = fit_differential_zernike(
            d_wx, d_wy, mask, mask, fm.s, x, y, indices=indices, fit_offsets=True
        )
    # the two offsets are still reported separately
    assert set(fit.offsets) == {"x", "y"}
    # the data is reproduced exactly by *some* gauge
    assert fit.rms_residual < 1e-9


def test_direction_argument_is_validated():
    cfg, fm, x, y, *_ = _synthetic_differential()
    image = np.zeros((64, 64))
    from lsi.ftmode import demodulate_lobe
    from lsi.pipeline import demodulate_fourier

    with pytest.raises(ValueError, match="direction"):
        fm.demodulation_offset("abc")
    with pytest.raises(ValueError, match="direction"):
        fm.phase_shift_delta_table(direction="abc")
    with pytest.raises(ValueError, match="direction"):
        fm.phase_shift_frames(ZernikeWavefront([0.0], [2]), "abc")
    with pytest.raises(ValueError, match="direction"):
        demodulate_lobe(image, cfg.grid, direction="abc")
    with pytest.raises(ValueError, match="direction"):
        demodulate_fourier(fm, image, direction="abc")
    assert np.isfinite(fm.demodulation_offset("x"))
    assert np.isfinite(fm.demodulation_offset("y"))


def test_config_validation():
    with pytest.raises(ValueError):
        Grid(n=0)
    with pytest.raises(ValueError):
        Grid(n=2.5)
    with pytest.raises(ValueError):
        Grid(extent=-1.0)
    with pytest.raises(ValueError):
        SystemConfig(wavelength_nm=-500)
    with pytest.raises(ValueError):
        SystemConfig(na=0.0)
    with pytest.raises(ValueError):
        SystemConfig(na=1.5)
    with pytest.raises(ValueError):
        SystemConfig(period_um=0.0)
    with pytest.raises(ValueError):
        SystemConfig(phase_steps=0)
    with pytest.raises(ValueError):
        SystemConfig(talbot_number=0)
    with pytest.raises(ValueError):
        SystemConfig(shear_ratio=1.5)
    # the defaults stay valid
    assert SystemConfig().grid.n == 256


# --------------------------------------------------------------------------- #
# Callable pupils: the exact order geometry for a non-circular aperture
# --------------------------------------------------------------------------- #
def _pupil_frame():
    """Return ``(cfg, X, Y)`` on a non-integer-shear model grid."""
    cfg = _cfg(extent=1.10)
    n = cfg.grid.n
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    dx = 2 * cfg.grid.extent / n
    return cfg, (xx - c) * dx, (yy - c) * dx


def test_callable_pupil_reproduces_the_analytic_unit_disk():
    """The default pupil must be the special case ``f = x*x + y*y <= 1``."""
    cfg = _cfg(extent=1.10)
    default = ForwardModel(cfg)
    callable_disc = ForwardModel(cfg, pupil=lambda x, y: x * x + y * y <= 1.0)

    assert callable_disc.has_custom_pupil
    assert np.array_equal(default.aperture(), callable_disc.aperture())
    for ab in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        assert np.array_equal(
            default.order_support(*ab), callable_disc.order_support(*ab)
        ), f"order {ab} differs from the analytic unit disk"


def test_callable_pupil_keeps_the_order_geometry_exact():
    """A callable is evaluated at the true shifted coordinates, so nothing snaps.

    An *array* pupil can only be resampled by whole pixels, which misplaces the
    aperture edge by up to half a pixel; a callable has no such error, which is
    why it is the recommended path for a non-circular aperture.
    """
    cfg, X, Y = _pupil_frame()
    half = 0.55
    square = lambda x, y: (np.abs(x) <= half) & (np.abs(y) <= half)
    fm = ForwardModel(cfg, pupil=square)

    px = fm.s / cfg.grid.dx
    assert not np.isclose(px, np.rint(px)), "the test needs a non-integer shear"
    for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        expected = (np.abs(X + a * fm.s) <= half) & (np.abs(Y + b * fm.s) <= half)
        assert np.array_equal(fm.order_support(a, b), expected), f"order {a,b}"
    # a 2h x 2h square, unshifted
    assert int(fm.order_support(0, 0).sum()) == int(
        ((np.abs(X) <= half) & (np.abs(Y) <= half)).sum()
    )


def test_aperture_matches_the_zero_order_support_for_every_pupil_kind():
    """``aperture()`` is what the demodulation stages use as the region mask."""
    cfg, X, Y = _pupil_frame()
    kinds = [
        None,
        lambda x, y: x * x + y * y <= 1.0,
        (X * X + Y * Y) <= 1.0,
    ]
    for kind in kinds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm = ForwardModel(cfg, pupil=kind)
        assert np.array_equal(fm.aperture(), fm.order_support(0, 0))


def test_array_pupil_snapping_is_reported_once_and_points_at_the_callable():
    cfg = _cfg(extent=1.10)
    n = cfg.grid.n
    with pytest.warns(UserWarning, match="not an integer") as record:
        ForwardModel(cfg, pupil=np.ones((n, n), dtype=bool))
    assert len(record) == 1, "the notice must not repeat per order_geometry call"
    assert "callable" in str(record[0].message)


def test_callable_pupil_output_shape_is_validated():
    cfg = _cfg(extent=1.10)
    fm = ForwardModel(cfg, pupil=lambda x, y: np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="shape of"):
        fm.order_support(1, 0)


def test_callable_pupil_float_output_is_thresholded():
    cfg, X, Y = _pupil_frame()
    half = 0.5
    soft = lambda x, y: (
        (np.abs(x) <= half) & (np.abs(y) <= half)
    ).astype(float)
    fm = ForwardModel(cfg, pupil=soft)
    hard = ForwardModel(
        cfg, pupil=lambda x, y: (np.abs(x) <= half) & (np.abs(y) <= half)
    )
    for ab in ((0, 0), (1, 0), (0, 1)):
        assert np.array_equal(fm.order_support(*ab), hard.order_support(*ab))


def _region_cfg():
    return SystemConfig(
        grid=Grid(n=128, extent=1.10),
        wavelength_nm=532.0,
        period_um=2.0,
        na=0.6,
        shear_ratio=0.35,
    )


def _square_pupil(half):
    return lambda x, y: (np.abs(x) <= half) & (np.abs(y) <= half)


def _analytic_region_x(cfg, half):
    """The x shear region of a square aperture, evaluated on pixel centres."""
    n = cfg.grid.n
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    dx = cfg.grid.dx
    X = (xx - c) * dx
    Y = (yy - c) * dx
    return (
        (np.abs(X) <= half)
        & (np.abs(X - cfg.s) <= half)
        & (np.abs(X + cfg.s) <= half)
        & (np.abs(Y) <= half)
    )


def test_callable_pupil_makes_the_shear_regions_exact():
    """Route A must get the same sub-pixel exactness as ``order_support``.

    ``shear_region_masks`` shifts the *sampled* aperture by whole pixels when it
    is handed an array, so an array pupil over-includes whenever the region edge
    does not fall on a grid point.  Passing the callable through (which is what
    ``demodulate_phase_shift`` does via ``pupil_definition``) removes that.
    """
    cfg = _region_cfg()
    fm = ForwardModel(cfg)
    assert not np.isclose(fm.s / cfg.grid.dx, np.rint(fm.s / cfg.grid.dx))

    # the definition is forwarded, not the sampled mask
    callable_model = ForwardModel(cfg, pupil=_square_pupil(0.70))
    assert callable(callable_model.pupil_definition)
    assert ForwardModel(cfg).pupil_definition is None

    for half in (0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        expected = _analytic_region_x(cfg, half)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_callable = shear_region_masks(
                cfg.grid, fm.s, aperture=_square_pupil(half)
            )["region_x"]
        assert np.array_equal(from_callable, expected), f"half={half}"

    # ... whereas the array path snaps, which is why the callable exists
    yy, xx = np.mgrid[0:cfg.grid.n, 0:cfg.grid.n]
    c = (cfg.grid.n - 1) / 2.0
    X = (xx - c) * cfg.grid.dx
    Y = (yy - c) * cfg.grid.dx
    sampled = _square_pupil(0.70)(X, Y)
    with pytest.warns(UserWarning, match="not an integer"):
        from_array = shear_region_masks(cfg.grid, fm.s, aperture=sampled)["region_x"]
    assert from_array.sum() > _analytic_region_x(cfg, 0.70).sum()


def test_callable_aperture_shape_is_validated():
    cfg = _region_cfg()
    fm = ForwardModel(cfg)
    with pytest.raises(ValueError, match="shape"):
        shear_region_masks(cfg.grid, fm.s, aperture=lambda x, y: np.zeros(4))


def test_callable_aperture_cannot_be_combined_with_a_fitted_circle():
    cfg = _region_cfg()
    fm = ForwardModel(cfg)
    with pytest.raises(ValueError, match="fitted-circle"):
        shear_region_masks(
            cfg.grid, fm.s, center=(0.1, 0.0), radius=0.5,
            aperture=_square_pupil(0.5),
        )


def test_shear_region_aperture_uses_the_same_threshold_convention_as_the_model():
    """A float aperture means the same thing in both places.

    ``ForwardModel._prepare_pupil`` takes a boolean array as given and
    threshold-samples anything else at ``> 0.5``; a float value of 0.3 is
    therefore *outside* the aperture.  Casting with ``dtype=bool`` here instead
    would call it inside, so the two conventions are pinned together.
    """
    cfg = _region_cfg()
    fm = ForwardModel(cfg)
    n = cfg.grid.n
    soft = np.full((n, n), 0.3)          # below the threshold everywhere
    hard = np.full((n, n), 0.7)          # above it everywhere

    # the array path also snaps (not what this test is about)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert not shear_region_masks(cfg.grid, fm.s, aperture=soft)["zero"].any()
        assert shear_region_masks(cfg.grid, fm.s, aperture=hard)["zero"].all()
    # and the model agrees about the same array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert not ForwardModel(cfg, pupil=soft).aperture().any()
        assert ForwardModel(cfg, pupil=hard).aperture().all()
