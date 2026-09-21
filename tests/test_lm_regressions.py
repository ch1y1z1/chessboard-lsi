"""Regressions for the Levenberg-Marquardt solver.

Covered former defects:

* ``LMResult.cond`` was the condition number of the *normal matrix* at the
  *starting point*: the wrong matrix (``cond(J^T J)`` is the square of the
  interpretable ``cond(J)``, and it disagrees with the convention used by
  ``lsi.reconstruct``) and the wrong point (it said nothing about the returned
  fit);
* the variable-projection step used ``a * J_raw`` as the reduced Jacobian,
  which is only a first-order approximation: it ignores both the projection
  onto the nuisance basis ``span{m, 1}`` and the derivative of the fitted
  ``(a, b)``.
"""

import numpy as np

from lsi.lm import LMConfig, levenberg_marquardt, reduced_scale_background


def _decay_problem(truth=(2.0, 0.3, 5.0), n=200):
    t = np.linspace(0.0, 1.0, n)
    truth = np.asarray(truth, dtype=float)

    def model(x):
        return x[0] * np.exp(-x[1] * t) + x[2] * t

    def res_jac(x):
        e = np.exp(-x[1] * t)
        jac = np.column_stack([e, -x[0] * t * e, t])
        return model(x) - model(truth), jac

    return t, truth, model, res_jac


def test_cond_is_the_jacobian_at_the_solution_not_the_normal_matrix_at_the_start():
    _, truth, _, res_jac = _decay_problem()
    x0 = np.array([5.0, 3.0, 0.05])          # deliberately far from the solution
    out = levenberg_marquardt(res_jac, x0)
    assert np.allclose(out.x, truth, rtol=1e-6)
    _, jac_solution = res_jac(out.x)
    _, jac_start = res_jac(x0)
    cond_solution = np.linalg.cond(jac_solution)
    cond_start = np.linalg.cond(jac_start.T @ jac_start)
    assert not np.isclose(cond_solution, cond_start, rtol=1e-2)
    assert np.isclose(out.cond, cond_solution, rtol=1e-9)
    # and it must not be the squared (normal-matrix) value
    assert not np.isclose(out.cond, cond_solution ** 2, rtol=1e-6)


def test_reduced_residual_is_the_optimal_affine_fit():
    rng = np.random.default_rng(11)
    model = rng.normal(size=40) + 3.0
    meas = rng.normal(size=40)
    f, _, a, b = reduced_scale_background(model - meas, np.zeros((40, 1)), meas)
    design = np.column_stack([model, np.ones_like(model)])
    sol, *_ = np.linalg.lstsq(design, meas, rcond=None)
    assert np.isclose(a, sol[0])
    assert np.isclose(b, sol[1])
    assert np.allclose(f, design @ sol - meas)
    # the reduced residual is orthogonal to the nuisance basis by construction
    assert abs(float(np.dot(f, model))) < 1e-9
    assert abs(float(f.sum())) < 1e-9


def test_reduced_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 120)
    meas = rng.normal(size=t.size)

    def model(c):
        return 1.0 + 0.5 * np.sin(3.0 * c * t) + 0.2 * c**2

    def jac(c):
        return (1.5 * t * np.cos(3.0 * c * t) + 0.4 * c)[:, None]

    c0 = 0.7
    _, j_proj, a, _ = reduced_scale_background(model(c0) - meas, jac(c0), meas)
    h = 1e-6
    finite_diff = (
        reduced_scale_background(model(c0 + h) - meas, jac(c0 + h), meas)[0]
        - reduced_scale_background(model(c0 - h) - meas, jac(c0 - h), meas)[0]
    ) / (2 * h)

    assert np.allclose(j_proj[:, 0], finite_diff, atol=1e-6)
    # the former approximation was wrong by an order of magnitude
    error_new = float(np.max(np.abs(j_proj[:, 0] - finite_diff)))
    error_old = float(np.max(np.abs(a * jac(c0)[:, 0] - finite_diff)))
    assert error_new < 1e-6
    assert error_old > 10 * error_new


def test_variable_projection_handles_a_rank_deficient_nuisance_basis():
    # a constant model makes the nuisance basis [m, 1] rank deficient: this
    # must not raise, and the constant offset must still be removed
    meas = np.linspace(0.0, 1.0, 32)
    constant = np.full(32, 2.0)
    f, j, a, b = reduced_scale_background(constant - meas, np.zeros((32, 1)), meas)
    # a * 2 + b can only reproduce a constant, so the best affine fit is the
    # mean of the measurement itself
    assert np.allclose(f, meas.mean() - meas)
    assert np.allclose(j, 0.0)
    assert np.isfinite([a, b]).all()


def test_lm_reports_solution_invariants():
    _, truth, _, res_jac = _decay_problem()
    out = levenberg_marquardt(res_jac, [2.5, 0.35, 4.5])
    assert np.allclose(out.x, truth, rtol=1e-4)
    assert out.converged
    assert out.n_iter >= 1
    assert out.cond > 0
    assert np.isfinite(out.rms_residual)
    assert out.rms_residual < 1e-9


def test_scale_background_config_is_honoured():
    _, truth, _, res_jac = _decay_problem()
    out = levenberg_marquardt(res_jac, [2.5, 0.35, 4.5], LMConfig(fit_scale_background=False))
    assert out.scale == 1.0
    assert out.background == 0.0
    assert np.allclose(out.x, truth, rtol=1e-4)
