"""Regression tests for the second review round.

Each test here pins a defect that produced a *silently wrong* result (or a
confusing late failure) rather than an exception:

* ``unwrap_masked_poisson(method="jacobi")`` solved the normal equations with
  the opposite sign and treated the image border as periodic, so it returned
  ``-truth`` (and the wrap corrupted even that) while the sparse branch agreed
  with the ground truth.
* ``OrderTermsCache`` mixed the original order index with the index into its
  compressed per-order lists, so a legal ``OrderSet`` with an interior
  zero-amplitude order raised ``IndexError`` from ``model_and_jacobian``.
* The demodulation support was a hardcoded unit circle while
  ``ForwardModel(pupil=...)`` really was honoured, so a custom-pupil model came
  back with an empty (or wrongly shaped) region and an all-zero wavefront.
* ``bitmap_orders(rotation_deg=...)`` was a no-op; making it rotate also made
  it advertise a spectrum the integer-order model cannot represent (see
  ``test_grating_farfield.py``).
* Enum-valued arguments fell through to a default on a typo, and ``SystemConfig``
  accepted ``64.0`` for a count.
"""

import pathlib
import warnings

import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import lsi
from lsi.config import (
    Grid,
    SystemConfig,
    check_difference_model,
    check_offset_mode,
    check_region_mode,
    check_unwrap_method,
)
from lsi.forward import ForwardModel, ZernikeWavefront
from lsi.grating import analytic_orders
from lsi.phaseshift import shear_region_masks
from lsi.pipeline import (
    DiffPhase,
    demodulate_fourier,
    demodulate_phase_shift,
    fourier_to_wavefront,
    offset_in_dW,
    reconstruct,
)
from lsi.unwrap import unwrap_masked_poisson

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ramp(n=48, ax=0.35, ay=0.22):
    """A wrapped ramp whose local pitch is well below pi, so the true
    unwrapped phase is unambiguous."""
    yy, xx = np.mgrid[0:n, 0:n]
    truth = ax * xx + ay * yy
    return np.angle(np.exp(1j * truth)), truth


def _rms(u, ref, mask):
    d = (u - ref)[mask]
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def _square_aperture(n=64, half=0.6):
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - (n - 1) / 2) / ((n - 1) / 2)
    y = (yy - (n - 1) / 2) / ((n - 1) / 2)
    return x, y, (np.abs(x) <= half) & (np.abs(y) <= half)


def _wf(coeffs=(0.0, 0.05, -0.03)):
    return ZernikeWavefront(np.array(coeffs), np.arange(1, len(coeffs) + 1))


def _diff_phase(fm, image, **kwargs):
    """A :class:`DiffPhase` from the single-frame carrier route."""
    dW, mask, phase, confidence = {}, {}, {}, {}
    for direction in ("x", "y"):
        d_w, m, lobe = demodulate_fourier(fm, image, direction=direction, **kwargs)
        dW[direction], mask[direction], phase[direction] = d_w, m, lobe.phase
        confidence[direction] = lobe.amplitude
    return DiffPhase(
        dW=dW,
        phase=phase,
        mask=mask,
        difference_model=kwargs.get("difference_model", "one_sided"),
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# unwrap_masked_poisson(method="jacobi")
# --------------------------------------------------------------------------- #
def test_jacobi_solver_has_the_same_sign_as_the_sparse_solver():
    # The jacobi update used ``+rho`` where the sparse branch builds
    # ``A psi = -rho``: it converged to -truth + const.
    phi, truth = _ramp()
    yy, xx = np.mgrid[0:48, 0:48]
    mask = (xx - 23.5) ** 2 + (yy - 23.5) ** 2 <= 15.0 ** 2

    sp = unwrap_masked_poisson(phi, mask, method="sparse")
    ja = unwrap_masked_poisson(phi, mask, method="jacobi")

    assert _rms(sp, truth, mask) < 1e-6
    assert _rms(ja, truth, mask) < 1e-4
    assert _rms(ja, sp, mask) < 1e-4


def test_jacobi_solver_does_not_wrap_across_the_image_border():
    # ``np.roll`` made row 0 a neighbour of row n-1 (and column 0 of column
    # n-1) whenever the mask touched the border.  The sparse branch bounds-checks
    # instead, so the two disagreed on a ramp by ~7.8 rad rms.
    phi, truth = _ramp()

    band = np.zeros(phi.shape, dtype=bool)
    band[20:28, :] = True          # one connected mask spanning both borders
    for mask, label in ((np.ones(phi.shape, dtype=bool), "full frame"),
                        (band, "left/right border band")):
        sp = unwrap_masked_poisson(phi, mask, method="sparse")
        ja = unwrap_masked_poisson(phi, mask, method="jacobi")
        assert _rms(sp, truth, mask) < 1e-6, label
        assert _rms(ja, truth, mask) < 1e-3, label
        assert _rms(ja, sp, mask) < 1e-3, label


def test_jacobi_and_sparse_agree_on_a_disconnected_mask():
    from scipy import ndimage

    phi, truth = _ramp(32)
    mask = np.zeros_like(phi, dtype=bool)
    mask[4:14, 4:14] = True
    mask[18:28, 18:28] = True

    # the relative offset of the two components is not recoverable, which the
    # solver reports rather than silently pretending otherwise
    with pytest.warns(UserWarning, match="disconnected"):
        sp = unwrap_masked_poisson(phi, mask, method="sparse")
    with pytest.warns(UserWarning, match="disconnected"):
        ja = unwrap_masked_poisson(phi, mask, method="jacobi")

    # each component carries its own undetermined constant, so compare the
    # de-meaned (i.e. geometry-carrying) part component by component
    labels, n_comp = ndimage.label(mask)
    assert n_comp == 2
    for lab in range(1, n_comp + 1):
        comp = labels == lab
        for field in (sp, ja):
            vals = field[comp]
            t = truth[comp]
            assert np.allclose(vals - vals.mean(), t - t.mean(), atol=1e-4)
        d = ja[comp] - sp[comp]
        assert np.allclose(d, d.mean(), atol=1e-4)


# --------------------------------------------------------------------------- #
# OrderTermsCache / ForwardModel.model_and_jacobian
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "order_list",
    [
        [(0, 0), (99, 99), (1, 0)],              # interior zero
        [(0, 0), (99, 99), (1, 0), (-1, 0)],     # interior zero, longer set
        [(0, 0), (1, 0), (99, 99)],              # trailing zero (used to work)
    ],
)
def test_zero_amplitude_orders_do_not_break_the_jacobian(order_list):
    cfg = SystemConfig(grid=Grid(n=16))
    orders = analytic_orders(max_index=1).with_orders(order_list)
    fm = ForwardModel(cfg, orders=orders)
    coeffs = np.array([0.0, 0.02, -0.01])
    wf = _wf(coeffs)
    rows = np.arange(16 * 16)
    deltas = [np.zeros(len(orders.ab))]

    I, J = fm.model_and_jacobian(coeffs, wf, deltas, rows=rows)

    assert I.shape == (16 * 16,)
    assert J.shape == (16 * 16, 3)
    assert np.all(np.isfinite(I)) and np.all(np.isfinite(J))

    # the cached model must reproduce the plain forward model on the same rows
    assert np.allclose(I, fm.intensity(wf).reshape(-1)[rows])

    # ... and its Jacobian must be the derivative of that intensity
    eps = 1e-6
    for j in range(len(coeffs)):
        up, dn = coeffs.copy(), coeffs.copy()
        up[j] += eps
        dn[j] -= eps
        Iu, _ = fm.model_and_jacobian(up, wf, deltas, rows=rows)
        Id, _ = fm.model_and_jacobian(dn, wf, deltas, rows=rows)
        fd = (Iu - Id) / (2 * eps)
        assert np.allclose(J[:, j], fd, rtol=1e-5, atol=1e-7)


def test_zero_amplitude_order_keeps_the_phase_shift_indices_aligned():
    # ``factor`` indexes ``deltas`` with the *original* order index; the cache
    # index must therefore be the loop variable, not the stored one.
    cfg = SystemConfig(grid=Grid(n=16))
    orders = analytic_orders(max_index=1).with_orders([(0, 0), (99, 99), (1, 0)])
    fm = ForwardModel(cfg, orders=orders)
    wf = _wf()
    rows = np.arange(16 * 16)

    d0 = np.zeros(len(orders.ab))
    d1 = np.zeros(len(orders.ab))
    d1[2] = 0.7                       # the (1, 0) order
    I0, _ = fm.model_and_jacobian(np.zeros(3), wf, [d0], rows=rows)
    I1, _ = fm.model_and_jacobian(np.zeros(3), wf, [d1], rows=rows)
    assert not np.allclose(I0, I1)


# --------------------------------------------------------------------------- #
# custom pupil: the demodulation support must follow the model aperture
# --------------------------------------------------------------------------- #
def test_shear_regions_follow_a_custom_aperture():
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))

    default = shear_region_masks(cfg.grid, cfg.s)
    assert default["region_x"].sum() > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        custom = shear_region_masks(cfg.grid, cfg.s, aperture=square)

    for key in ("zero", "region_x", "region_y", "center5"):
        assert custom[key].sum() > 0 or key == "center5"
        assert np.all(square[custom[key]]), key
    # a square pupil is not a circle, so the region must really change
    assert custom["region_x"].sum() != default["region_x"].sum()


def test_shear_regions_match_the_model_order_supports():
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)
        masks = shear_region_masks(cfg.grid, cfg.s, aperture=fm.aperture())

    expected_x = (
        fm.order_support(0, 0) & fm.order_support(1, 0) & fm.order_support(-1, 0)
    )
    expected_y = (
        fm.order_support(0, 0) & fm.order_support(0, 1) & fm.order_support(0, -1)
    )
    assert np.array_equal(masks["region_x"], expected_x)
    assert np.array_equal(masks["region_y"], expected_y)


def test_fourier_route_masks_stay_inside_a_custom_aperture():
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)
        image = fm.intensity(_wf())

        for direction in ("x", "y"):
            # erode_px=0: the default 4 iterations eat the narrow square-pupil
            # overlap completely (see the empty-mask test below)
            dW, mask, lobe = demodulate_fourier(
                fm, image, direction=direction, erode_px=0
            )
            assert mask.sum() > 0, direction
            assert np.all(square[mask]), direction
            assert np.all(np.isfinite(dW[mask]))


def test_empty_demodulation_mask_raises_instead_of_returning_zeros():
    # With the unit-circle support the x mask of a square-pupil model came out
    # as 0 pixels and the pipeline returned an all-zero wavefront silently.
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)
        image = fm.intensity(_wf())
        # the default erode_px=4 shrinks the small square overlap away
        with pytest.raises(ValueError, match="empty"):
            demodulate_fourier(fm, image, direction="x")


def test_custom_pupil_region_is_never_silently_empty():
    # a hardcoded unit-circle support produced an EMPTY mask for the x
    # direction of a square-pupil model, and ``fourier_to_wavefront`` then
    # returned an all-zero wavefront without complaining.
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)
        image = fm.intensity(_wf())
        frames_x = fm.phase_shift_frames(_wf(), "x")
        frames_y = fm.phase_shift_frames(_wf(), "y")

        diff = demodulate_phase_shift(fm, frames_x, frames_y)
        assert diff.mask["x"].sum() > 0
        assert diff.mask["y"].sum() > 0
        assert np.all(square[diff.mask["x"]])
        assert np.all(square[diff.mask["y"]])


def test_modulation_region_mode_rejects_a_non_circular_aperture():
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)
        frames_x = fm.phase_shift_frames(_wf(), "x")
        frames_y = fm.phase_shift_frames(_wf(), "y")

        with pytest.raises(ValueError, match="circle"):
            demodulate_phase_shift(
                fm, frames_x, frames_y, region_mode="modulation"
            )


def test_custom_pupil_shear_snapping_is_reported_once():
    _, _, square = _square_aperture()
    cfg = SystemConfig(grid=Grid(n=64))
    with pytest.warns(UserWarning, match="not an integer"):
        ForwardModel(cfg, orders=analytic_orders(max_index=1), pupil=square)


def test_custom_pupil_shape_is_validated():
    cfg = SystemConfig(grid=Grid(n=32))
    with pytest.raises(ValueError, match="shape"):
        ForwardModel(cfg, orders=analytic_orders(max_index=1),
                     pupil=np.ones((8, 8), dtype=bool))


# --------------------------------------------------------------------------- #
# enum-valued arguments
# --------------------------------------------------------------------------- #
def test_enum_validators_are_the_single_source_of_truth():
    assert check_region_mode("analytic") == "analytic"
    assert check_region_mode("modulation") == "modulation"
    assert check_unwrap_method("seed") == "seed"
    assert check_offset_mode("estimate") == "estimate"

    for bad, fn in (
        ("modle", check_offset_mode),
        ("xyz", check_offset_mode),
        ("", check_offset_mode),
        ("analyticc", check_region_mode),
        ("poison", check_unwrap_method),
        ("on_sided", check_difference_model),
        ("ONE_SIDED", check_difference_model),
    ):
        with pytest.raises(ValueError):
            fn(bad)


def test_reconstruct_rejects_a_typo_in_offset_mode():
    # "modle" used to fall through to "none" and silently return the
    # un-offset-corrected coefficients.
    cfg = SystemConfig(grid=Grid(n=64))
    fm = ForwardModel(cfg, orders=analytic_orders(max_index=1))
    image = fm.ft_mode_frame(_wf())
    diff = _diff_phase(fm, image)

    ref = reconstruct(fm, diff, indices=[2], offset_mode="none")
    assert np.all(np.isfinite(ref.coeffs))
    with pytest.raises(ValueError):
        reconstruct(fm, diff, indices=[2], offset_mode="modle")
    with pytest.raises(ValueError):
        reconstruct(fm, diff, indices=[2], offset_mode="xyz")


def test_reconstruct_rejects_a_typo_in_difference_model():
    cfg = SystemConfig(grid=Grid(n=64))
    fm = ForwardModel(cfg, orders=analytic_orders(max_index=1))
    image = fm.ft_mode_frame(_wf())
    diff = _diff_phase(fm, image)

    with pytest.raises(ValueError):
        reconstruct(fm, diff, indices=[2], difference_model="on_sided")


def test_offset_in_dW_and_demodulate_fourier_reject_typos():
    # both read ``... if difference_model == "one_sided" else ...``, so every
    # unknown string silently became the two-sided model.
    cfg = SystemConfig(grid=Grid(n=64))
    fm = ForwardModel(cfg, orders=analytic_orders(max_index=1))
    image = fm.ft_mode_frame(_wf())

    with pytest.raises(ValueError):
        offset_in_dW(fm, "x", "junk")
    with pytest.raises(ValueError):
        demodulate_fourier(fm, image, direction="x", difference_model="junk")

    one = demodulate_fourier(fm, image, direction="x",
                             difference_model="one_sided")[0]
    with pytest.warns(UserWarning, match=r"O\(s\^2\).*approximation"):
        two = demodulate_fourier(fm, image, direction="x",
                                 difference_model="two_sided")[0]
    assert not np.allclose(one, two)


def test_demodulate_phase_shift_rejects_typos():
    cfg = SystemConfig(grid=Grid(n=32))
    fm = ForwardModel(cfg, orders=analytic_orders(max_index=1))
    fx = fm.phase_shift_frames(_wf(), "x")
    fy = fm.phase_shift_frames(_wf(), "y")

    with pytest.raises(ValueError):
        demodulate_phase_shift(fm, fx, fy, region_mode="analyticc")
    with pytest.raises(ValueError):
        demodulate_phase_shift(fm, fx, fy, unwrap="poison")


# --------------------------------------------------------------------------- #
# SystemConfig / Grid validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 64.0},
        {"n": 64.5},
        {"n": 0},
        {"n": -8},
        {"n": True},
        {"extent": 0.0},
        {"extent": -1.0},
        {"extent": float("nan")},
    ],
)
def test_grid_rejects_non_integer_or_non_positive_parameters(kwargs):
    with pytest.raises(ValueError):
        Grid(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_steps": 8.0},
        {"phase_steps": 2},
        {"phase_steps": 0},
        {"phase_steps": -3},
        {"phase_steps": True},
        {"talbot_number": 1.5},
        {"talbot_number": 0},
        {"shear_ratio": 0.0},
        {"shear_ratio": 1.0},
        {"shear_ratio": 2.0},
        {"shear_ratio": True},
        {"na": 0.0},
        {"na": 1.5},
        {"wavelength_nm": 0.0},
        {"wavelength_nm": -1.0},
        {"period_um": 0.0},
        {"period_um": float("inf")},
    ],
)
def test_system_config_rejects_out_of_range_parameters(kwargs):
    with pytest.raises(ValueError):
        SystemConfig(**kwargs)


def test_valid_parameters_are_still_accepted():
    cfg = SystemConfig(phase_steps=3, talbot_number=2, shear_ratio=0.5)
    assert cfg.phase_steps == 3 and isinstance(cfg.phase_steps, int)
    assert cfg.talbot_number == 2
    assert np.isclose(cfg.carrier_f0, 2.0 / (2.0 * 0.5))

    grid = Grid(n=32)
    assert isinstance(grid.n, int)
    assert grid.shape == (32, 32)
    assert np.zeros(grid.shape).shape == (32, 32)


def test_default_configuration_still_selects_the_documented_shear():
    # the dissertation presets must survive the validation tightening
    assert np.isclose(SystemConfig().s, 0.0731139, rtol=1e-6)
    assert np.isclose(SystemConfig(period_um=30.0).s, 0.0438684, rtol=1e-6)


# --------------------------------------------------------------------------- #
# normalize_orders
# --------------------------------------------------------------------------- #
def test_normalize_orders_rejects_an_all_zero_order_set():
    # ``scale = 0`` divided silently into every amplitude and produced NaN.
    cfg = SystemConfig(grid=Grid(n=16))
    orders = analytic_orders(max_index=1).with_orders([(99, 99), (98, 98)])
    assert not np.any(orders.amp)

    with pytest.raises(ValueError):
        ForwardModel(cfg, orders=orders, normalize_orders=True)


def test_normalize_orders_still_normalizes_a_real_set():
    cfg = SystemConfig(grid=Grid(n=16))
    orders = analytic_orders(max_index=1)
    fm = ForwardModel(cfg, orders=orders, normalize_orders=True)
    amps = fm.orders.amp
    assert np.all(np.isfinite(amps))
    assert np.isclose(np.sum(np.abs(amps) ** 2), 1.0)


# --------------------------------------------------------------------------- #
# packaging
# --------------------------------------------------------------------------- #
def test_version_has_a_single_source_of_truth():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]

    # the version must be dynamic (read from lsi/__init__.py), not repeated
    assert "version" not in project
    assert "version" in project.get("dynamic", [])
    assert isinstance(lsi.__version__, str) and lsi.__version__.count(".") == 2


def test_declared_license_file_exists():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]

    assert project["license"] == "MIT"          # PEP 639 SPDX expression
    files = project.get("license-files") or []
    assert files, "a license-files entry is needed for the SPDX string"
    for name in files:
        assert (ROOT / name).exists(), name


def test_the_one_sided_doubled_model_stays_consistent_with_the_demodulator():
    """``"one_sided_doubled"`` must keep working: it is not a typo.

    The demodulator converts phase to a wavefront difference with
    ``dW = phase / factor``, where ``factor`` is ``2 pi`` only for
    ``"one_sided"`` and ``pi`` otherwise.  The doubled Zernike model expects
    ``2 d`` while ``phase / pi`` supplies exactly that, so the demodulator and
    the doubled fit agree and recover the same coefficients as the one-sided
    pair.  Validating the argument must therefore accept all three names
    instead of rejecting the third one as unknown.
    """
    cfg = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
    fm = ForwardModel(cfg)
    indices = np.array([2, 3, 4, 5, 6, 7])
    coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6])
    frames = fm.ft_mode_frame(ZernikeWavefront(coeffs, indices))

    one, mask, _ = demodulate_fourier(
        fm, frames, direction="x", difference_model="one_sided"
    )
    doubled, _, _ = demodulate_fourier(
        fm, frames, direction="x", difference_model="one_sided_doubled"
    )
    # the only difference between the two names is this exact factor of two
    assert np.allclose(doubled[mask] / one[mask], 2.0)

    def recovered(model):
        fit, _ = fourier_to_wavefront(fm, frames, difference_model=model)
        return np.array([fit.coeffs[list(fit.indices).index(int(j))] for j in indices])

    assert np.allclose(recovered("one_sided_doubled"), recovered("one_sided"))
    assert np.isclose(recovered("one_sided_doubled")[-1], coeffs[-1], atol=0.05)
    for name in ("one_sided", "two_sided", "one_sided_doubled"):
        assert check_difference_model(name) == name
    with pytest.raises(ValueError, match="one_sided"):
        check_difference_model("one_sided_double")
