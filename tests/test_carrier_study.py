"""carrier_study 的正确性测试（研究计划阶段 0.1 / 0.2 / 0.4）。"""

from __future__ import annotations

import numpy as np
import pytest

from carrier_study import config
from carrier_study.fourier import FourierChain, unwrap_fast
from carrier_study.generator import MATCHED, Generator, Noise, Scene, add_noise, grating_orders
from carrier_study.metrics import Whitener, fim
from carrier_study.nominal import NominalModel
from carrier_study.problem import Budget, LSProblem
from carrier_study.solvers import ALL_METHODS, FIRST_ORDER, run
from carrier_study.truths import sample_q
from carrier_study.zern import basis, fringe_nm, rms_factors, to_c
from lsi.invert import fourier_to_wavefront, unwrap_poisson, wrap
from lsi.model import DEFAULT_ORDERS, ForwardModel, Grid, chessboard_orders, zernike, zernike_wavefront

CFG64 = config.system(n=64)
IDX = tuple(range(2, 14))


def _rng(k=0):
    return np.random.default_rng(k)


# --------------------------------------------------------------------------- #
def test_zernike_matches_lsi_and_rms():
    rng = _rng(1)
    x, y = rng.uniform(-0.7, 0.7, (2, 50))
    for j in range(1, 37):
        np.testing.assert_allclose(basis([j], x, y)[:, 0], zernike(j, x, y), atol=1e-12)
    # RMS 因子：极坐标数值积分
    r = np.linspace(0, 1, 801)
    t = np.linspace(0, 2 * np.pi, 801, endpoint=False)
    R, T = np.meshgrid(r, t)
    X, Y = R * np.cos(T), R * np.sin(T)
    for j in (2, 4, 7, 9, 22, 36):
        Z = basis([j], X, Y)[..., 0]
        ms = np.trapezoid((Z**2 * R).mean(axis=0), r) * 2.0      # (1/pi) ∫∫ Z^2 r dr dθ
        assert np.sqrt(ms) == pytest.approx(rms_factors([j])[0], rel=2e-3)


def test_nominal_matches_lsi_forward():
    model = NominalModel(CFG64, IDX)
    q = sample_q("flat", 0.4, IDX, _rng(2))
    fm = ForwardModel(CFG64)
    img = fm.carrier_frame(zernike_wavefront(to_c(q, IDX), IDX))
    np.testing.assert_allclose(model.intensity(q), model.take(img), atol=1e-12)
    # 像素集外没有光
    outside = np.setdiff1d(np.arange(64 * 64), model.pix)
    assert np.abs(img.ravel()[outside]).max() == 0.0


def test_nominal_derivatives():
    model = NominalModel(CFG64, IDX)
    q = sample_q("pow1", 0.5, IDX, _rng(3))
    I, st = model.forward(q)
    J = model.jacobian(st)
    h = 1e-6
    for j in (0, 3, 11):
        e = np.zeros(model.p)
        e[j] = h
        fd = (model.intensity(q + e) - model.intensity(q - e)) / (2 * h)
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-5, atol=1e-7)
    u = _rng(4).normal(size=model.M)
    np.testing.assert_allclose(model.jtv(st, u), J.T @ u, rtol=1e-11, atol=1e-10)
    # Hessian 项：sum u_i d2I_i 的方向导数 = d/dt [J(q + t v)^T u]
    H = model.hessian_terms(st, u)
    v = _rng(5).normal(size=model.p)
    Jp = model.jacobian(model.forward(q + h * v)[1])
    Jm = model.jacobian(model.forward(q - h * v)[1])
    np.testing.assert_allclose(H @ v, (Jp - Jm).T @ u / (2 * h), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(H, H.T, atol=1e-9)


def test_grating_orders_reduce_to_ideal():
    g = grating_orders(0.5, max_index=2)
    ideal = chessboard_orders(max_index=2)
    assert set(g) == set(ideal)
    for k, v in ideal.items():
        assert g[k][0] == pytest.approx(v, abs=1e-14)
    for k, v in DEFAULT_ORDERS.items():
        assert g[k][1] == "main" and g[k][0] == pytest.approx(v, abs=1e-14)
    g45 = grating_orders(0.45, max_index=2)
    assert any(kind == "duty" for _, kind in g45.values())


def test_generator_matched_equals_nominal_and_supersample_converges():
    model = NominalModel(CFG64, IDX)
    q = sample_q("pow2", 0.3, IDX, _rng(6))
    img = Generator(CFG64, IDX, MATCHED).clean(q)
    np.testing.assert_allclose(model.take(img), model.intensity(q), atol=1e-12)
    # 光瞳边缘以内的像素：被积函数光滑，中点法误差 ~ 1/S^2
    x, y = CFG64.grid.coords()
    s, dx = CFG64.s, CFG64.grid.dx
    inner = np.ones(x.shape, bool)
    for a_, b_ in DEFAULT_ORDERS:
        inner &= np.hypot(x + a_ * s, y + b_ * s) < 1.0 - 2 * dx
    a = Generator(CFG64, IDX, Scene(supersample=4)).clean(q)
    b = Generator(CFG64, IDX, Scene(supersample=8)).clean(q)
    assert np.abs(a - b)[inner].max() < 0.1 * np.abs(a - img)[inner].max()


def test_noise_statistics():
    I = np.full((64, 64), 2.0)
    y, var, _ = add_noise(I, Noise("gauss", 40.0), _rng(7))
    assert np.std(y - I) == pytest.approx(np.sqrt(var[0, 0]), rel=0.05)
    y, var, var_est = add_noise(I, Noise("poisson", 1e4), _rng(8))
    assert np.var(y - I) == pytest.approx(var[0, 0], rel=0.08)
    assert var_est.mean() == pytest.approx(var[0, 0], rel=0.05)


def test_fourier_chain_matches_lsi():
    cfg = config.system(n=128)
    idx = tuple(range(2, 37))
    q = sample_q("pow2", 0.3, idx, _rng(9))
    img = Generator(cfg, idx).clean(q)
    ch = FourierChain(cfg, idx)
    q_est, _ = ch.estimate(img)
    fit, _ = fourier_to_wavefront(ForwardModel(cfg), img, indices=idx)
    np.testing.assert_allclose(q_est, fit.coeffs * rms_factors(idx), atol=1e-9)


def test_unwrap_fast_matches_lsi():
    rng = _rng(10)
    yy, xx = np.mgrid[0:40, 0:40]
    phi = 0.02 * (xx - 20) ** 2 / 3 + 0.3 * yy
    mask = (xx - 20) ** 2 + (yy - 21) ** 2 < 17**2
    w = wrap(phi + 0.01 * rng.normal(size=phi.shape))
    a = unwrap_fast(w, mask)
    b = unwrap_poisson(w, mask)
    np.testing.assert_allclose(a[mask], b[mask], atol=1e-9)


def test_fourier_linearization_matches_finite_difference():
    cfg = config.system(n=128)
    q = sample_q("pow2", 0.2, IDX, _rng(11))
    img = Generator(cfg, IDX).clean(q)
    ch = FourierChain(cfg, IDX)
    lin = ch.linearize(img)
    d = _rng(12).normal(size=img.shape)
    eps = 1e-6
    fd = (ch.estimate(img + eps * d)[0] - ch.estimate(img - eps * d)[0]) / (2 * eps)
    np.testing.assert_allclose(lin.apply(d)[0], fd, rtol=1e-4, atol=1e-6)
    # 批量与单列一致
    cols = np.zeros((3,) + img.shape)
    cols[0].flat[5000] = cols[1].flat[8200] = cols[2].flat[100] = 1.0
    np.testing.assert_allclose(lin.apply(cols), np.stack([lin.apply(c)[0] for c in cols]), atol=1e-12)


def test_problem_caching_and_budget():
    model = NominalModel(CFG64, IDX)
    q = sample_q("flat", 0.2, IDX, _rng(13))
    P = LSProblem(model, model.intensity(q), budget=10.0)
    x = np.zeros(model.p)
    P.loss(x); P.loss(x); P.grad(x); P.grad(x)
    assert P.counts["F"] == 1 and P.counts["G"] == 1
    P.jac(x); P.jac(x)
    assert P.counts["J"] == 1
    with pytest.raises(Budget):
        for k in range(10):
            P.loss(x + k + 1.0)
    # 梯度 = J^T r
    P2 = LSProblem(model, model.intensity(q))
    g = P2.grad(x)
    np.testing.assert_allclose(g, P2.jac(x).T @ P2.residual(x), rtol=1e-10, atol=1e-14)


class _Quadratic:
    """线性残差 r = A q - b（良态），检验求解器本身。"""

    def __init__(self, p=6, seed=0):
        rng = _rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(40, p)))
        self.A = Q * np.linspace(1.0, 3.0, p)
        self.q_star = rng.normal(size=p)
        self.M, self.p = 40, p

    def forward(self, q):
        return self.A @ q, None

    def jacobian(self, state):
        return self.A

    def jtv(self, state, u):
        return self.A.T @ u

    def y(self):
        return self.A @ self.q_star


@pytest.mark.parametrize("method", ALL_METHODS)
def test_solvers_on_quadratic(method):
    m = _Quadratic()
    # Hessian = A^T A / M 的特征值在 [1/40, 9/40]，固定步长取 4 < 2/L
    params = {"gd_fixed": {"eta": 4.0}, "nesterov_fixed": {"eta": 4.0}, "adam": {"lr": 3e-2}}.get(method, {})
    P = LSProblem(m, m.y(), budget=20000.0)
    out = run(method, P, np.zeros(m.p), **params)
    tol = 1e-3 if method == "adam" else 1e-7
    assert np.max(np.abs(out["q"] - m.q_star)) < tol, out["status"]


@pytest.mark.parametrize("method", ("gn", "lm_main", "lm_minpack", "trf", "bfgs", "lbfgs", "gd_bb", "nesterov"))
def test_solvers_recover_carrier_truth(method):
    model = NominalModel(CFG64, IDX)
    q = sample_q("pow1", 0.3, IDX, _rng(14))
    P = LSProblem(model, model.intensity(q), budget=4000.0)
    out = run(method, P, q + 0.05 * _rng(15).normal(size=model.p) / np.sqrt(model.p))
    tol = 1e-8 if method in ("gn", "lm_main", "lm_minpack", "trf") else 1e-4
    assert np.linalg.norm(out["q"] - q) < tol, out["status"]


def test_first_order_requires_budget():
    model = NominalModel(CFG64, IDX)
    P = LSProblem(model, model.intensity(np.zeros(model.p)))
    for method in FIRST_ORDER:
        with pytest.raises(ValueError):
            run(method, P, np.zeros(model.p))


def test_whitener():
    model = NominalModel(CFG64, IDX)
    q = sample_q("flat", 0.2, IDX, _rng(16))
    W = Whitener(fim(model, q, 1e-4))
    dq = _rng(17).multivariate_normal(np.zeros(model.p), W.crb, size=4000)
    assert np.mean(W.d_rel(dq) ** 2) == pytest.approx(1.0, rel=0.05)
