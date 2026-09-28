"""同一光强目标上的七种局部优化器及统一实际计算预算。

所有方法返回已评价过的最低目标点（incumbent），轨迹记录每次真实光强计算，
包含线搜索/拒绝步，不冒充已接受迭代。预算耗尽仍返回该点。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, minimize

from .lm import _solve_damped
from .problem import IntensityProblem

METHODS = ("gd", "momentum", "adam", "bfgs", "gn", "lm", "trf")


@dataclass(frozen=True)
class SolverOptions:
    max_iter: int = 2000
    max_forward: int = 500
    max_jacobian: int = 500
    max_seconds: float | None = None
    gtol: float = 1e-10
    ftol: float = 1e-12
    xtol: float = 1e-10
    learning_rate: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lambda0: float = 1e-3
    armijo: float = 1e-4
    max_backtracks: int = 30

    def __post_init__(self):
        for name in ("max_iter", "max_forward", "max_jacobian", "max_backtracks"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("gtol", "ftol", "xtol", "learning_rate", "epsilon", "lambda0"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须有限且大于零")
        if min(self.gtol, self.ftol, self.xtol) <= np.finfo(float).eps:
            raise ValueError("容差必须大于机器精度")
        for name in ("momentum", "beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} 必须在 [0,1) 内")
        if not 0 < self.armijo < 1:
            raise ValueError("armijo 必须在 (0,1) 内")
        if self.max_seconds is not None and (
            not np.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds 必须有限且大于零")


@dataclass
class OptimizationResult:
    method: str
    indices: np.ndarray
    coeffs: np.ndarray
    loss: float
    rms_residual: float
    status: str
    stopped_by_tolerance: bool
    n_iter: int | None
    n_forward: int
    n_jacobian: int
    n_gradient: int
    elapsed_seconds: float
    precompute_seconds: float
    options: dict
    history: list[dict]

    def as_dict(self):
        return dict(zip(map(int, self.indices), map(float, self.coeffs)))


class _Stop(Exception):
    pass


class _Evaluation:
    def __init__(self, problem, q0, options):
        self.problem, self.options = problem, options
        self.started = perf_counter()
        self.n_forward = self.n_jacobian = self.n_gradient = 0
        self.history = []
        self.q = self.r = self.j = self.g = None
        self.best_q, self.best_loss = q0.copy(), np.inf
        self.iteration = None

    def check_time(self):
        # 第一点评价总会执行，使极小时间预算也有合法结果；记录实际超时。
        if (self.n_forward and self.options.max_seconds is not None
                and perf_counter() - self.started >= self.options.max_seconds):
            raise _Stop("time_budget")

    def evaluate(self, q, jacobian=False):
        self.check_time()
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all():
            raise _Stop("nonfinite_parameters")
        cached = self.q is not None and np.array_equal(q, self.q)
        if cached and (not jacobian or self.j is not None):
            return self.r, self.j
        if self.n_forward >= self.options.max_forward:
            raise _Stop("forward_budget")
        if jacobian and self.n_jacobian >= self.options.max_jacobian:
            raise _Stop("jacobian_budget")
        self.n_forward += 1
        self.n_jacobian += int(jacobian)
        self.r, self.j = self.problem.evaluate(q, jacobian=jacobian)
        self.q, self.g = q.copy(), None
        loss = float(self.r @ self.r / 2)
        if not np.isfinite(loss) or (self.j is not None and not np.isfinite(self.j).all()):
            raise _Stop("nonfinite_evaluation")
        if loss < self.best_loss:
            self.best_q, self.best_loss = q.copy(), loss
        self.history.append({
            "event": "evaluation", "iteration": self.iteration,
            "elapsed_seconds": perf_counter() - self.started,
            "n_forward": self.n_forward, "n_jacobian": self.n_jacobian,
            "loss": loss, "best_loss": self.best_loss,
            "gradient_inf": None, "q": q.tolist(),
        })
        return self.r, self.j

    def value_gradient(self, q):
        r, j = self.evaluate(q, True)
        if self.g is None:
            self.g = j.T @ r
            self.n_gradient += 1
            self.history[-1]["gradient_inf"] = float(np.linalg.norm(self.g, ord=np.inf))
        return float(r @ r / 2), self.g

    def residual(self, q):
        # TRF requests a Jacobian at each accepted point; trials only need I.
        return self.evaluate(q)[0]

    def jacobian(self, q):
        return self.evaluate(q, True)[1]


def _local(e, q, method, o):
    velocity = np.zeros_like(q)
    first, second = np.zeros_like(q), np.zeros_like(q)
    lam, nu = o.lambda0, 2.0
    status = "iteration_budget"
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        loss, g = e.value_gradient(q)
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        r, j = e.r, e.j
        if method == "lm":
            step, _, diag = _solve_damped(j, r, lam)
            trial = q + step
            r_new, _ = e.evaluate(trial)
            new_loss = float(r_new @ r_new / 2)
            predicted = float(step @ (lam * diag * step - g) / 2)
            rho = (loss - new_loss) / predicted if predicted > 0 else -np.inf
            if rho <= 0:
                lam = min(lam * nu, 1e10)
                nu *= 2
                if lam >= 1e10:
                    return "damping_limit", iteration
                continue
            lam = float(np.clip(lam * max(1/3, 1 - (2*rho - 1)**3), 1e-12, 1e10))
            nu = 2.0
        elif method in ("gd", "gn"):
            direction = (-g if method == "gd" else
                         np.linalg.lstsq(j, -r, rcond=None)[0])
            slope = float(g @ direction)
            if slope >= 0:
                return "non_descent_direction", iteration
            alpha = 1.0
            for _ in range(o.max_backtracks):
                step = alpha * direction
                trial = q + step
                r_new, _ = e.evaluate(trial)
                new_loss = float(r_new @ r_new / 2)
                if new_loss <= loss + o.armijo * alpha * slope:
                    break
                alpha *= 0.5
            else:
                return "line_search_failed", iteration
        else:
            if method == "momentum":
                velocity = o.momentum * velocity - o.learning_rate * g
                step = velocity
            else:
                first = o.beta1 * first + (1 - o.beta1) * g
                second = o.beta2 * second + (1 - o.beta2) * g*g
                step = -o.learning_rate * (first / (1-o.beta1**iteration)) / (
                    np.sqrt(second / (1-o.beta2**iteration)) + o.epsilon)
            trial = q + step
            r_new, _ = e.evaluate(trial)
            new_loss = float(r_new @ r_new / 2)
        old_q = q
        q = trial
        if np.linalg.norm(step) <= o.xtol * (o.xtol + np.linalg.norm(old_q)):
            return "step_tolerance", iteration
        # Momentum/Adam 的单步小变化可能是振荡，不用单步 ftol 提前停机。
        if method not in ("momentum", "adam") and (
            0 <= loss-new_loss <= o.ftol * max(loss, np.finfo(float).tiny)
        ):
            return "loss_tolerance", iteration
    return status, o.max_iter


def solve(problem: IntensityProblem, c0, method="lm", options=None):
    """c0/输出均为 Fringe 系数 waves；内部统一使用 problem 的 q 坐标。

    TRF 在 SciPy>=1.10 上可用；不依赖新版 callback，其迭代数不报告，
    以实际前向/J 调用及时间限额控制。max_iter 仅适用于其余方法。
    """
    if method not in METHODS:
        raise ValueError(f"未知优化器 {method!r}; 可选 {METHODS}")
    o = options or SolverOptions()
    q = problem.to_parameters(c0)
    e = _Evaluation(problem, q, o)
    n_iter = None
    try:
        # 统一评价初值，为小预算和异常退出保留可比较的候选解。
        e.value_gradient(q)
        if method == "bfgs":
            result = minimize(e.value_gradient, q, jac=True, method="BFGS",
                              options={"gtol": o.gtol, "maxiter": o.max_iter})
            n_iter = int(result.nit)
            status = {0: "gradient_tolerance", 1: "iteration_budget",
                      2: "precision_loss", 3: "nonfinite_evaluation"}.get(
                          int(result.status), "scipy_failure")
        elif method == "trf":
            result = least_squares(
                e.residual, q, jac=e.jacobian, method="trf", x_scale=1.0,
                gtol=o.gtol, ftol=o.ftol, xtol=o.xtol, max_nfev=o.max_forward,
            )
            status = {0: "forward_budget", 1: "gradient_tolerance",
                      2: "loss_tolerance", 3: "step_tolerance",
                      4: "loss_and_step_tolerance"}.get(int(result.status), "scipy_failure")
        else:
            status, n_iter = _local(e, q, method, o)
    except _Stop as stopped:
        status = str(stopped)
        n_iter = e.iteration
    elapsed = perf_counter() - e.started
    # 最后一次不可中断的线性代数/模型计算也计入实际超时。
    if o.max_seconds is not None and elapsed > o.max_seconds:
        status = "time_budget"
    return OptimizationResult(
        method, problem.indices.copy(), problem.to_coefficients(e.best_q),
        e.best_loss, float(np.sqrt(2*e.best_loss)), status,
        status.endswith("tolerance"), n_iter, e.n_forward, e.n_jacobian,
        e.n_gradient, elapsed, problem.precompute_seconds, asdict(o), e.history,
    )
