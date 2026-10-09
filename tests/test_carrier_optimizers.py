"""九法求解器的正确性验证：解析 toy 问题、缓存/预算不变量、native 梯度契约。

这些是数值正确性测试，不做方法间性能排名；载频 fixture 只检查
同初值同目标下各方法正确执行并返回观测损失 incumbent。
"""
import numpy as np
import pytest

from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront
from lsi.optimizers import (
    METHODS, PRIMARY_METHODS, LEGACY_METHODS, SolverOptions, _Evaluation, solve,
)
from lsi.problem import IntensityProblem, StackedIntensityProblem


# --------------------------------------------------------------------- #
# 解析 toy 问题（独立于物理模型，验证数值正确性与接口契约）
# --------------------------------------------------------------------- #
class ToyLeastSquares:
    """最小 problem 接口：evaluate / to_parameters / to_coefficients / indices。

    residual(q) 与解析 J 由调用者给定；scale=1 使 q 与系数坐标一致。
    """

    def __init__(self, residual, jacobian, n_params):
        self._residual, self._jacobian = residual, jacobian
        self.indices = np.arange(2, 2 + n_params)
        self.scale = np.ones(n_params)
        self.precompute_seconds = 0.0
        self.residual_calls = 0
        self.jacobian_calls = 0

    def to_parameters(self, c):
        return np.asarray(c, dtype=float)

    def to_coefficients(self, q):
        return np.asarray(q, dtype=float)

    def evaluate(self, q, jacobian=True):
        q = np.asarray(q, dtype=float)
        if jacobian:
            self.jacobian_calls += 1
        else:
            self.residual_calls += 1
        return (self._residual(q), self._jacobian(q) if jacobian else None)


class NativeToyLeastSquares(ToyLeastSquares):
    """T1/T3 契约 fixture：``value_gradient(q) -> (loss, g_q)`` 融合通道。"""

    supports_native_gradient = True
    gradient_backend = "toy_native_jtr"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_calls = 0

    def value_gradient(self, q):
        self.native_calls += 1
        q = np.asarray(q, dtype=float)
        r, j = self._residual(q), self._jacobian(q)
        return float(r @ r / 2), j.T @ r


class NativeOnlyToyLeastSquares(NativeToyLeastSquares):
    """只暴露融合通道的问题：无 evaluate，检验残差请求的诚实处理。"""

    evaluate = None


def quadratic_problem(native=False, n=4, m=6, seed=7):
    """r(q) = A q - b 的良态二次目标，极小点 q* = pinv(A) b。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, n))
    a = a @ np.diag(1.0 / np.linalg.svd(a, compute_uv=False))  # 近正交列
    q_star = rng.normal(size=n)
    b = a @ q_star
    cls = NativeToyLeastSquares if native else ToyLeastSquares
    return cls(lambda q: a @ q - b, lambda q: a, n), q_star


def rosenbrock_problem(native=False):
    """r = [10(q1-q0^2), 1-q0]；非线性最小二乘，极小点 (1,1)。"""
    def residual(q):
        return np.array([10.0 * (q[1] - q[0] ** 2), 1.0 - q[0]])

    def jacobian(q):
        return np.array([[-20.0 * q[0], 10.0], [-1.0, 0.0]])

    cls = NativeToyLeastSquares if native else ToyLeastSquares
    return cls(residual, jacobian, 2), np.array([1.0, 1.0])


@pytest.fixture
def carrier():
    """小型单帧载频问题：各法共享同一观测、坐标与初值。"""
    fm = ForwardModel(SystemConfig(grid=Grid(n=24), period_um=30.0))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    image = fm.carrier_frame(zernike_wavefront(truth, indices))
    problem = StackedIntensityProblem(
        fm, indices, [image], [fm.carrier_phases(fm.config.carrier_f0)])
    return problem, truth


@pytest.fixture
def unmodulated():
    """零初值退化仅存在于无调制单帧问题。"""
    fm = ForwardModel(SystemConfig(grid=Grid(n=24)))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    image = fm.intensity(zernike_wavefront(truth, indices))
    return IntensityProblem(fm, indices, image), truth


# --------------------------------------------------------------------- #
# 方法注册与参数验证
# --------------------------------------------------------------------- #
def test_method_registry():
    assert PRIMARY_METHODS == (
        "gd", "nesterov", "adam", "nlcg", "bfgs", "lbfgs", "gn", "lm", "trf")
    assert LEGACY_METHODS == ("momentum",)
    assert set(METHODS) == set(PRIMARY_METHODS) | set(LEGACY_METHODS)


@pytest.mark.parametrize("kwargs", [
    {"patience": 0}, {"patience": -2}, {"patience": 1.5},
    {"gd_step": "bb2"}, {"nesterov_restart": "bogus"},
    {"lbfgs_memory": 0}, {"step_grow": 0.5}, {"step_grow": np.nan},
    {"learning_rate": 0}, {"armijo": 1.0}, {"momentum": 1.0},
])
def test_option_validation(kwargs):
    with pytest.raises(ValueError):
        SolverOptions(**kwargs)


# --------------------------------------------------------------------- #
# 解析 toy 上的数值正确性
# --------------------------------------------------------------------- #
SECOND_ORDER = ("bfgs", "lbfgs", "gn", "lm", "trf")


@pytest.mark.parametrize("method", SECOND_ORDER + ("nlcg",))
def test_quadratic_exact_recovery(method):
    """良态二次：二阶/共轭类在有限预算内到达极小点邻域。

    bfgs/gn/lm/trf 到机器精度；nlcg（非精确线搜索）与 lbfgs 受自身
    ftol/线搜索精度限制，量级略松但仍收敛到同一极小点。
    """
    problem, q_star = quadratic_problem()
    result = solve(problem, np.zeros(4), method,
                   SolverOptions(max_forward=400, max_jacobian=400,
                                 max_iter=500, gtol=1e-11))
    np.testing.assert_allclose(result.coeffs, q_star, atol=1e-6)
    assert result.loss < 1e-13


@pytest.mark.parametrize("method", ["gd", "nesterov", "adam", "momentum"])
def test_quadratic_first_order_decreases_and_tracks(method):
    """一阶法在二次目标上必须实质下降且 incumbent 为观测最优。"""
    problem, q_star = quadratic_problem()
    options = SolverOptions(max_forward=2000, max_jacobian=2000,
                            max_iter=1500, learning_rate=0.2)
    result = solve(problem, np.zeros(4), method, options)
    assert result.loss < 0.05 * result.history[0]["loss"]
    assert result.loss == min(ev["loss"] for ev in result.history)


@pytest.mark.parametrize("method", METHODS)
def test_rosenbrock_all_methods_run(method):
    """非线性 toy：所有方法从 (0,0) 起步损失有限且不差于初值。"""
    problem, q_star = rosenbrock_problem()
    options = SolverOptions(max_forward=400, max_jacobian=400, max_iter=600,
                            learning_rate=0.05)
    result = solve(problem, np.array([0.0, 0.0]), method, options)
    assert np.isfinite(result.loss)
    assert result.loss <= result.history[0]["loss"]
    if method in SECOND_ORDER:
        np.testing.assert_allclose(result.coeffs, q_star, atol=1e-5)


# --------------------------------------------------------------------- #
# 预算、缓存与 incumbent 不变量（全部十法，含 legacy momentum）
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_budget_and_incumbent_invariants(carrier, method):
    problem, truth = carrier
    c0 = truth * 0.9
    result = solve(problem, c0, method,
                   SolverOptions(max_forward=30, max_jacobian=30))
    assert result.n_forward <= 30
    assert result.n_jacobian <= result.n_forward
    assert len(result.history) == result.n_forward
    assert result.loss == min(ev["loss"] for ev in result.history)
    assert result.loss <= result.history[0]["loss"]
    for event in result.history:
        assert event["request"] in ("function", "gradient", "jacobian")
        assert event["kind"] in ("residual", "jacobian", "native_gradient")
        assert np.isfinite(event["loss"])


@pytest.mark.parametrize("method", METHODS)
def test_zero_initial_is_not_recovery(unmodulated, method):
    """零初值命中退化不动点：容差停止≠恢复成功。"""
    problem, truth = unmodulated
    result = solve(problem, np.zeros_like(truth), method)
    assert result.stopped_by_tolerance
    assert result.loss > 1e-5
    np.testing.assert_array_equal(result.coeffs, 0)


def test_request_and_compute_counters(carrier):
    problem, truth = carrier
    q = problem.to_parameters(truth * 0.9)
    e = _Evaluation(problem, q, SolverOptions())
    e.loss(q)                # function request -> residual eval
    e.value_gradient(q)      # gradient request -> explicit J + J^T r
    e.residual(q)            # 缓存命中
    e.jacobian(q)            # 缓存命中（J 已在）
    assert (e.n_function_requests, e.n_gradient_requests,
            e.n_jacobian_requests) == (2, 1, 1)
    assert (e.n_forward, e.n_jacobian, e.n_gradient) == (2, 1, 1)
    assert len(e.history) == 2


def test_cache_promotion_counts_honestly(carrier):
    """残差缓存升级为 J 时必须重新计算并如实计数；反之不冒充。"""
    problem, truth = carrier
    q = problem.to_parameters(truth * 0.9)
    e = _Evaluation(problem, q, SolverOptions())
    e.residual(q)                            # fwd=1, jac=0
    assert e.j is None
    e.jacobian(q)                            # 升级：完整 J 重算
    assert (e.n_forward, e.n_jacobian) == (2, 1)
    e.value_gradient(q)                      # 已付费 J：matvec 即梯度
    assert (e.n_forward, e.n_jacobian, e.n_gradient) == (2, 1, 1)
    e.loss(q)
    e.residual(q)
    assert len(e.history) == 2


def test_directional_derivative(carrier):
    """value_gradient 交付的 g 与中心差分方向导数一致。"""
    problem, truth = carrier
    q = problem.to_parameters(truth * 0.8)
    e = _Evaluation(problem, q, SolverOptions())
    _, g = e.value_gradient(q)
    d = np.random.default_rng(3).normal(size=q.size)
    eps = 1e-7
    fd = (e.loss(q + eps * d) - e.loss(q - eps * d)) / (2 * eps)
    assert g @ d == pytest.approx(fd, rel=1e-5, abs=1e-9)


# --------------------------------------------------------------------- #
# native 梯度契约（T1/T3 接口的独立 fixture 验证）
# --------------------------------------------------------------------- #
FIRST_ORDER_AND_QN = ("gd", "nesterov", "adam", "nlcg", "bfgs", "lbfgs")


@pytest.mark.parametrize("method", FIRST_ORDER_AND_QN)
def test_native_backend_used_for_gradient_path(method):
    """native 问题：一阶/准牛顿走 value_gradient 融合通道并如实计数。"""
    problem, q_star = quadratic_problem(native=True)
    result = solve(problem, np.zeros(4), method,
                   SolverOptions(max_forward=400, max_jacobian=400,
                                 max_iter=400, learning_rate=0.2))
    assert problem.native_calls > 0
    assert result.gradient_backend == "toy_native_jtr"
    assert result.n_fused_gradient == problem.native_calls
    # 一阶/准牛顿在融合通道下不生成显式 J；n_jacobian 只计 J 级通道。
    assert result.n_jacobian == result.n_fused_gradient
    assert problem.jacobian_calls == 0
    assert result.loss < result.history[0]["loss"]


@pytest.mark.parametrize("method", ("gn", "lm", "trf"))
def test_explicit_jacobian_methods_ignore_native(method):
    """GN/LM/TRF 即使面对 native 问题也用显式 J（契约要求）。"""
    problem, q_star = quadratic_problem(native=True)
    result = solve(problem, np.zeros(4), method,
                   SolverOptions(max_forward=400, max_jacobian=400))
    assert problem.native_calls == 0
    assert problem.jacobian_calls > 0
    np.testing.assert_allclose(result.coeffs, q_star, atol=1e-7)


def test_native_cache_does_not_fake_residual(carrier):
    """融合通道只给出 loss/g：残差请求必须做额外显式计算并计数。"""
    problem, truth = carrier

    class NativeCarrier(StackedIntensityProblem):
        supports_native_gradient = True
        gradient_backend = "wrapped_native"

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.native_calls = 0

        def value_gradient(self, q):
            self.native_calls += 1
            r, j = super().evaluate(q, jacobian=True)
            return float(r @ r / 2), j.T @ r

    native = NativeCarrier(
        problem.forward, problem.indices,
        [problem.observed.reshape(problem.forward.shape)],
        problem.mods, scaling="rms")
    q = problem.to_parameters(truth * 0.9)
    e = _Evaluation(native, q, SolverOptions())
    loss, g = e.value_gradient(q)
    assert e.r is None and e.j is None        # 不冒充残差缓存
    r = e.residual(q)                          # 残差请求触发额外显式计算
    assert (e.n_forward, e.n_jacobian, e.n_gradient) == (2, 1, 1)
    np.testing.assert_allclose(
        r, problem.evaluate(q, jacobian=False)[0], atol=1e-15)
    # g 与显式 J^T r 一致（融合通道数值等价性）
    _, j = e.evaluate(q, jacobian=True)
    np.testing.assert_allclose(g, j.T @ r, atol=1e-12)


def test_native_only_problem_fallback():
    """无 evaluate 的纯 native 问题：一阶可解，显式通道请求诚实失败。"""
    problem, q_star = quadratic_problem()
    native_only = NativeOnlyToyLeastSquares(
        problem._residual, problem._jacobian, 4)
    result = solve(native_only, np.zeros(4), "gd",
                   SolverOptions(max_forward=300, max_jacobian=300,
                                 learning_rate=0.2))
    assert result.n_fused_gradient > 0
    # 线搜索试探也经融合通道（函数请求升级为 J 级计算，如实计数）
    assert result.history[0]["request"] == "gradient"
    trf_result = solve(native_only, np.zeros(4), "trf",
                       SolverOptions(max_forward=10))
    assert trf_result.status == "unsupported_problem"


# --------------------------------------------------------------------- #
# 载频 fixture：九法同初值同目标的正确性冒烟（非性能排名）
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method", PRIMARY_METHODS)
def test_carrier_nine_methods_same_problem(carrier, method):
    problem, truth = carrier
    c0 = truth * 0.95
    result = solve(problem, c0, method,
                   SolverOptions(max_forward=300, max_jacobian=300,
                                 max_iter=400, learning_rate=0.05))
    assert np.isfinite(result.loss)
    assert result.loss <= result.history[0]["loss"]
    r, _ = problem.evaluate(problem.to_parameters(result.coeffs))
    assert result.loss == pytest.approx(float(r @ r / 2), abs=1e-15)


@pytest.mark.parametrize("method", SECOND_ORDER)
def test_carrier_local_recovery(carrier, method):
    """载频打破符号退化后，二阶法从近真值恢复系数。"""
    problem, truth = carrier
    result = solve(problem, truth * 0.98, method,
                   SolverOptions(max_forward=300, max_jacobian=300,
                                 gtol=1e-12))
    np.testing.assert_allclose(result.coeffs, truth, atol=1e-6)
    assert result.rms_residual < 1e-6
