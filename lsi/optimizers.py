"""同一光强目标上的局部优化器及统一实际计算预算。

主比较九法：gd / nesterov / adam / nlcg / bfgs / lbfgs / gn / lm / trf；
``momentum`` 仅为旧配置兼容保留（diagnostic-only），见 ``LEGACY_METHODS``。

所有方法返回已评价过的最低观测损失点（incumbent），轨迹记录每次真实
光强/导数计算，包含线搜索试探与拒绝步，不冒充已接受迭代。
预算耗尽仍返回该点。

后端契约（与 problem 层的约定）：问题对象声明
``supports_native_gradient=True`` 且实现
``value_gradient(q) -> (loss, g_q)`` 时，一阶与准牛顿方法走融合梯度通道；
GN/LM/TRF 一律使用显式雅可比 ``problem.evaluate(q, jacobian=True)``。
旧问题对象没有该能力标识时保持显式 J 兼容路径。
融合通道只返回 (loss, g)，不产出残差向量：残差/雅可比请求不会命中其
缓存，必须做额外显式计算并如实计数；反向地，一次融合 Jᵀr 与一次显式
雅可比是同量级导数计算，二者都计入 n_jacobian（见 _Evaluation 注释）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, minimize

from .lm import _solve_damped
from .problem import IntensityProblem

METHODS = (
    "gd", "nesterov", "adam", "nlcg", "bfgs", "lbfgs", "gn", "lm", "trf",
    "momentum",
)
#: 主比较九法；momentum 保留为旧配置兼容 / 诊断用途，不进主比较。
PRIMARY_METHODS = METHODS[:9]
LEGACY_METHODS = METHODS[9:]

#: BB/回溯估计的试探步长裁剪范围，只为排除数值病态，不构成算法超参。
_ALPHA_MIN, _ALPHA_MAX = 1e-16, 1e12


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
    # --- 九法扩展参数；详见 docs/carrier/solver-spec.md ---
    patience: int | None = 100          # 平台期早停窗口（迭代数），None 禁用
    step_grow: float = 2.0              # 接受步后下一步试探步长的放大因子
    gd_step: str = "bb1"                # GD 步长策略：bb1 / constant
    nesterov_restart: str = "function"  # Nesterov 重启：function / gradient / none
    lbfgs_memory: int = 10              # L-BFGS-B maxcor

    def __post_init__(self):
        for name in ("max_iter", "max_forward", "max_jacobian", "max_backtracks",
                     "lbfgs_memory"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.patience is not None and (
            not isinstance(self.patience, int) or self.patience <= 0
        ):
            raise ValueError("patience 必须是正整数或 None")
        for name in ("gtol", "ftol", "xtol", "learning_rate", "epsilon", "lambda0"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须有限且大于零")
        if min(self.gtol, self.ftol, self.xtol) <= np.finfo(float).eps:
            raise ValueError("容差必须大于机器精度")
        if not np.isfinite(self.step_grow) or self.step_grow < 1:
            raise ValueError("step_grow 必须有限且不小于 1")
        for name in ("momentum", "beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} 必须在 [0,1) 内")
        if not 0 < self.armijo < 1:
            raise ValueError("armijo 必须在 (0,1) 内")
        if self.gd_step not in ("bb1", "constant"):
            raise ValueError("gd_step 必须是 bb1 或 constant")
        if self.nesterov_restart not in ("function", "gradient", "none"):
            raise ValueError("nesterov_restart 必须是 function/gradient/none")
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
    n_function_requests: int = 0
    n_gradient_requests: int = 0
    n_jacobian_requests: int = 0
    n_fused_gradient: int = 0
    gradient_backend: str = "explicit_jacobian"

    def as_dict(self):
        return dict(zip(map(int, self.indices), map(float, self.coeffs)))


class _Stop(Exception):
    pass


class _Evaluation:
    """唯一模型访问点：统一预算、缓存、计数、incumbent 与评价轨迹。

    实际计算计数（缓存命中不计数、不进轨迹）：

    * ``n_forward``   —— 每次真实光强计算（显式 evaluate 或 native 融合通道）
    * ``n_jacobian``  —— 每次 Jacobian 级导数计算：显式 J 生成，或 native
      融合 Jᵀr 通道（内部做同一量级的 dE 累积）；事件 ``kind`` 区分两者
    * ``n_gradient``  —— 每次交付的梯度向量（显式 Jᵀr 乘积或融合 Jᵀr）
    * ``n_fused_gradient`` —— 其中经 native 融合通道交付的次数

    ``n_function/n_gradient/n_jacobian_request`` 记录求解器请求数，
    与实际计算分列。``max_forward``/``max_jacobian`` 作用于实际计算。
    """

    def __init__(self, problem, q0, options):
        self.problem, self.options = problem, options
        self.started = perf_counter()
        self.n_forward = self.n_jacobian = self.n_gradient = 0
        self.n_fused_gradient = 0
        self.n_function_requests = 0
        self.n_gradient_requests = 0
        self.n_jacobian_requests = 0
        self.history = []
        # 缓存按量分层：r/j/loss/g 各自独立，缺失的量触发如实补算。
        self.q = self.r = self.j = self.g = None
        self.loss_value = None
        self.best_q, self.best_loss = np.asarray(q0, dtype=float).copy(), np.inf
        self.iteration = None
        self.has_evaluate = callable(getattr(problem, "evaluate", None))
        self.native_gradient = bool(
            getattr(problem, "supports_native_gradient", False)
            and callable(getattr(problem, "value_gradient", None))
        )
        self.gradient_backend = (
            str(getattr(problem, "gradient_backend", "value_gradient"))
            if self.native_gradient else "explicit_jacobian"
        )

    # ------------------------------------------------------------- 内部
    def check_time(self):
        # 第一点评价总会执行，使极小时间预算也有合法结果；记录实际超时。
        if (self.n_forward and self.options.max_seconds is not None
                and perf_counter() - self.started >= self.options.max_seconds):
            raise _Stop("time_budget")

    def _same(self, q):
        return self.q is not None and np.array_equal(q, self.q)

    def _check_point(self, q):
        self.check_time()
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all():
            raise _Stop("nonfinite_parameters")
        return q

    def _check_budget(self, derivative):
        if self.n_forward >= self.options.max_forward:
            raise _Stop("forward_budget")
        if derivative and self.n_jacobian >= self.options.max_jacobian:
            raise _Stop("jacobian_budget")

    def _commit(self, q, request, kind):
        if self.loss_value < self.best_loss:
            self.best_q, self.best_loss = q.copy(), self.loss_value
        self.history.append({
            "event": "evaluation", "iteration": self.iteration,
            "request": request, "kind": kind,
            "backend": self.gradient_backend,
            "elapsed_seconds": perf_counter() - self.started,
            "n_forward": self.n_forward, "n_jacobian": self.n_jacobian,
            "n_gradient": self.n_gradient,
            "loss": self.loss_value, "best_loss": self.best_loss,
            "gradient_inf": (float(np.linalg.norm(self.g, ord=np.inf))
                             if self.g is not None else None),
            "q": q.tolist(),
        })

    def annotate(self, **fields):
        """把求解器侧信息（alpha/lam/rho/accepted/restart）并入最近事件。"""
        if self.history:
            self.history[-1].update(
                {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                 for k, v in fields.items()})

    def _compute_explicit(self, q, request):
        """problem.evaluate：request=="function" 仅残差，否则含显式 J。"""
        derivative = request != "function"
        self._check_budget(derivative)
        if not self._same(q):
            self.q, self.r, self.j, self.g, self.loss_value = (
                q.copy(), None, None, None, None)
        self.n_forward += 1
        self.n_jacobian += int(derivative)
        r, j = self.problem.evaluate(q, jacobian=derivative)
        if (not np.isfinite(r).all()
                or (j is not None and not np.isfinite(j).all())):
            raise _Stop("nonfinite_evaluation")
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(r @ r / 2)
        if not np.isfinite(loss):
            raise _Stop("nonfinite_evaluation")
        self.r, self.j, self.loss_value = r, j, loss
        if request == "gradient":
            self.g = j.T @ r
            self.n_gradient += 1
        self._commit(q, request, "jacobian" if derivative else "residual")

    def _compute_native(self, q, request):
        """problem.value_gradient 融合通道：一次光强 + Jᵀr（J 级工作量）。"""
        self._check_budget(derivative=True)
        if not self._same(q):
            self.q, self.r, self.j, self.g, self.loss_value = (
                q.copy(), None, None, None, None)
        self.n_forward += 1
        self.n_jacobian += 1        # 融合 Jᵀr 内部是 Jacobian 级导数计算
        loss, g = self.problem.value_gradient(q)
        loss, g = float(loss), np.asarray(g, dtype=float)
        if not np.isfinite(loss) or not np.isfinite(g).all():
            raise _Stop("nonfinite_evaluation")
        # r/j 保持不可用：融合结果不冒充残差/雅可比缓存。
        self.loss_value, self.g = loss, g
        self.n_gradient += 1
        self.n_fused_gradient += 1
        self._commit(q, request, "native_gradient")

    # ------------------------------------------------------------- 对外
    def loss(self, q):
        """观测损失 F(q)=||r||²/2：线搜索试探点的最低成本入口。"""
        self.n_function_requests += 1
        q = self._check_point(q)
        if self._same(q) and self.loss_value is not None:
            return self.loss_value
        if self.has_evaluate:
            self._compute_explicit(q, "function")
        elif self.native_gradient:
            # 无显式接口的问题：残差请求只能升级为融合梯度评价，如实计 J 级成本。
            self._compute_native(q, "function")
        else:
            raise _Stop("unsupported_problem")
        return self.loss_value

    def value_gradient(self, q):
        """(loss, g)：native 问题走融合通道，否则显式 J + Jᵀr。"""
        self.n_gradient_requests += 1
        q = self._check_point(q)
        if self._same(q):
            if self.g is not None:
                return self.loss_value, self.g
            if self.j is not None:
                # 已付过代价的显式 J：matvec 即梯度，不另起 J 级计算。
                self.g = self.j.T @ self.r
                self.n_gradient += 1
                self.history[-1]["gradient_inf"] = float(
                    np.linalg.norm(self.g, ord=np.inf))
                return self.loss_value, self.g
        if self.native_gradient:
            self._compute_native(q, "gradient")
        else:
            self._compute_explicit(q, "gradient")
        return self.loss_value, self.g

    def evaluate(self, q, jacobian=False):
        """显式残差（及可选 J）：GN/LM/TRF 的通道。"""
        if jacobian:
            self.n_jacobian_requests += 1
        else:
            self.n_function_requests += 1
        q = self._check_point(q)
        if self._same(q) and self.r is not None \
                and (not jacobian or self.j is not None):
            return self.r, self.j
        if not self.has_evaluate:
            raise _Stop("unsupported_problem")
        self._compute_explicit(q, "jacobian" if jacobian else "function")
        return self.r, self.j

    def residual(self, q):
        # TRF requests a Jacobian at each accepted point; trials only need I.
        return self.evaluate(q)[0]

    def jacobian(self, q):
        return self.evaluate(q, True)[1]

    def gradient(self):
        """当前缓存点的 (loss, Jᵀr)；要求该点已有显式 J（GN/LM 用）。"""
        if self.j is None or self.r is None:
            raise _Stop("unsupported_problem")
        if self.g is None:
            self.g = self.j.T @ self.r
            self.n_gradient += 1
            self.history[-1]["gradient_inf"] = float(
                np.linalg.norm(self.g, ord=np.inf))
        return self.loss_value, self.g


class _Stall:
    """可审计平台期判据：best_loss 连续 patience 次迭代无 ftol 相对改进则停。"""

    def __init__(self, e, o):
        self.options = o
        self.ref = e.best_loss
        self.count = 0

    def update(self, e):
        o = self.options
        if o.patience is None:
            return False
        if (not np.isfinite(self.ref)
                or e.best_loss < self.ref - o.ftol * max(
                    abs(self.ref), np.finfo(float).tiny)):
            self.ref, self.count = e.best_loss, 0
        else:
            self.count += 1
        return self.count >= o.patience


def _armijo(e, q, loss, g, direction, alpha0, o):
    """Armijo 回溯；返回 (alpha, trial, new_loss) 或 None。

    试探点都是真实评价（记入轨迹与 incumbent）；非有限步直接缩短。
    """
    slope = float(g @ direction)
    if not np.isfinite(slope) or slope >= 0:
        return None
    alpha = float(np.clip(alpha0, _ALPHA_MIN, _ALPHA_MAX))
    for _ in range(o.max_backtracks):
        trial = q + alpha * direction
        if np.isfinite(trial).all():
            new_loss = e.loss(trial)
            accepted = new_loss <= loss + o.armijo * alpha * slope
            e.annotate(alpha=alpha, armijo_accepted=accepted)
            if accepted:
                return alpha, trial, new_loss
        alpha *= 0.5
    return None


def _step_too_small(step, q_old, o):
    return np.linalg.norm(step) <= o.xtol * (o.xtol + np.linalg.norm(q_old))


def _loss_stalled(loss, new_loss, o):
    return 0 <= loss - new_loss <= o.ftol * max(
        loss, np.finfo(float).tiny)


def _gd(e, q, o):
    """GD + Armijo：首步 learning_rate，其后 BB1 谱步长（sᵀy>0 保障正值，
    非凸失效时退回上一接受步长）；gd_step="constant" 时固定 learning_rate。"""
    alpha_prev = None
    q_prev = g_prev = None
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        loss, g = e.value_gradient(q)
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        if o.gd_step == "bb1" and q_prev is not None:
            y = g - g_prev
            denom = float(y @ y)
            alpha0 = float((q - q_prev) @ y) / denom if denom > 0 else alpha_prev
            if not np.isfinite(alpha0) or alpha0 <= 0:
                alpha0 = alpha_prev      # 正值保障：退回已接受的正步长
        else:
            alpha0 = o.learning_rate
        accepted = _armijo(e, q, loss, g, -g, alpha0, o)
        if accepted is None:
            return "line_search_failed", iteration
        alpha, trial, new_loss = accepted
        q_prev, g_prev = q, g
        step = trial - q
        q = trial
        alpha_prev = alpha
        if _step_too_small(step, q_prev, o):
            return "step_tolerance", iteration
        if _loss_stalled(loss, new_loss, o):
            return "loss_tolerance", iteration
    return "iteration_budget", o.max_iter


def _nesterov(e, x, o):
    """FISTA 式 Nesterov：每步对前瞻点做 Armijo 保障的负梯度步。

    非凸保障：函数重启（O'Donoghue–Candès，f(x_new)>f(x) 时重置 t=1）
    或梯度重启（g·(x_new-x)>0）；Armijo 保证相对前瞻点充分下降。
    """
    y, t = x.copy(), 1.0
    alpha_prev, loss_x = None, None
    stall = _Stall(e, o)
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        loss_y, g = e.value_gradient(y)
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        if loss_x is None:
            loss_x = loss_y            # 初始 y==x：f(x_0)
        alpha0 = (o.learning_rate if alpha_prev is None
                  else alpha_prev * o.step_grow)
        accepted = _armijo(e, y, loss_y, g, -g, alpha0, o)
        if accepted is None:
            return "line_search_failed", iteration
        alpha_prev, x_new, loss_new = accepted
        step = x_new - x
        if o.nesterov_restart == "function" and loss_new > loss_x:
            restart = True
        elif o.nesterov_restart == "gradient" and float(g @ step) > 0:
            restart = True
        else:
            restart = False
        if restart:
            t, y = 1.0, x_new.copy()
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * step
            t = t_new
        e.annotate(restart=restart)
        x_old, x, loss_x = x, x_new, loss_new
        if _step_too_small(step, x_old, o):
            return "step_tolerance", iteration
        if stall.update(e):
            return "plateau_tolerance", iteration
    return "iteration_budget", o.max_iter


def _fixed_step(e, q, method, o):
    """Adam / legacy momentum 固定步长类：无单步下降保障，用平台期早停。"""
    velocity = np.zeros_like(q)
    first, second = np.zeros_like(q), np.zeros_like(q)
    stall = _Stall(e, o)
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        loss, g = e.value_gradient(q)
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        if method == "momentum":
            velocity = o.momentum * velocity - o.learning_rate * g
            step = velocity
        else:
            first = o.beta1 * first + (1 - o.beta1) * g
            second = o.beta2 * second + (1 - o.beta2) * g * g
            step = -o.learning_rate * (first / (1 - o.beta1**iteration)) / (
                np.sqrt(second / (1 - o.beta2**iteration)) + o.epsilon)
        trial = q + step
        e.loss(trial)
        e.annotate(alpha=o.learning_rate)
        old_q = q
        q = trial
        if _step_too_small(step, old_q, o):
            return "step_tolerance", iteration
        if stall.update(e):
            return "plateau_tolerance", iteration
    return "iteration_budget", o.max_iter


def _nlcg(e, q, o):
    """非线性共轭梯度：PR+ 组合系数、非下降显式重启、Armijo 线搜索。

    线搜索失败后以 -g 兜底重试一次；beta=0 即自动回到最速下降，
    故不做固定周期重启。注意此处为 Armijo 而非强 Wolfe 条件。
    """
    g_prev = d = alpha_prev = None
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        loss, g = e.value_gradient(q)
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        if d is None:
            d = -g
            steepest = True
        else:
            beta = max(float(g @ (g - g_prev)) / float(g_prev @ g_prev), 0.0)
            d = -g + beta * d
            steepest = beta == 0.0
            if not np.isfinite(d).all() or float(g @ d) >= 0:
                d, steepest = -g, True     # 非下降方向：显式重启
        alpha0 = (o.learning_rate if alpha_prev is None
                  else alpha_prev * o.step_grow)
        accepted = _armijo(e, q, loss, g, d, alpha0, o)
        if accepted is None and not steepest:
            d = -g
            accepted = _armijo(e, q, loss, g, d, o.learning_rate, o)
            if accepted is not None:
                e.annotate(restart=True)
        if accepted is None:
            return "line_search_failed", iteration
        alpha, trial, new_loss = accepted
        q_prev, g_prev = q, g
        step = trial - q
        q = trial
        alpha_prev = alpha
        if _step_too_small(step, q_prev, o):
            return "step_tolerance", iteration
        if _loss_stalled(loss, new_loss, o):
            return "loss_tolerance", iteration
    return "iteration_budget", o.max_iter


def _gn(e, q, o):
    """阻尼 GN：全步长先试，Armijo 回溯；显式 J 通道。"""
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        r, j = e.evaluate(q, jacobian=True)
        loss, g = e.gradient()
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        direction = np.linalg.lstsq(j, -r, rcond=None)[0]
        if float(g @ direction) >= 0:
            return "non_descent_direction", iteration
        accepted = _armijo(e, q, loss, g, direction, 1.0, o)
        if accepted is None:
            return "line_search_failed", iteration
        _, trial, new_loss = accepted
        step = trial - q
        q_old, q = q, trial
        if _step_too_small(step, q_old, o):
            return "step_tolerance", iteration
        if _loss_stalled(loss, new_loss, o):
            return "loss_tolerance", iteration
    return "iteration_budget", o.max_iter


def _lm(e, q, o):
    """Marquardt 阻尼 + Nielsen 更新的 LM；数学目标与旧实现一致。"""
    lam, nu = o.lambda0, 2.0
    for iteration in range(1, o.max_iter + 1):
        e.iteration = iteration
        r, j = e.evaluate(q, jacobian=True)
        loss, g = e.gradient()
        if np.linalg.norm(g, ord=np.inf) <= o.gtol:
            return "gradient_tolerance", iteration
        step, _, diag = _solve_damped(j, r, lam)
        trial = q + step
        new_loss = e.loss(trial)
        predicted = float(step @ (lam * diag * step - g) / 2)
        rho = (loss - new_loss) / predicted if predicted > 0 else -np.inf
        e.annotate(lam=lam, rho=rho, accepted=bool(rho > 0))
        if rho <= 0:
            lam = min(lam * nu, 1e10)
            nu *= 2
            if lam >= 1e10:
                return "damping_limit", iteration
            continue
        lam = float(np.clip(lam * max(1/3, 1 - (2*rho - 1)**3), 1e-12, 1e10))
        nu = 2.0
        q_old, q = q, trial
        if _step_too_small(step, q_old, o):
            return "step_tolerance", iteration
        if _loss_stalled(loss, new_loss, o):
            return "loss_tolerance", iteration
    return "iteration_budget", o.max_iter


_LOCAL = {
    "gd": _gd, "nesterov": _nesterov, "nlcg": _nlcg,
    "gn": _gn, "lm": _lm,
}

_LBFGS_STATUS = (
    ("EVALUATIONS EXCEEDS LIMIT", "forward_budget"),
    ("ITERATIONS REACHED LIMIT", "iteration_budget"),
    ("ABNORMAL_TERMINATION_IN_LNSRCH", "line_search_failed"),
    ("PROJECTED GRADIENT", "gradient_tolerance"),
    ("REDUCTION OF F", "loss_tolerance"),
)


def solve(problem: IntensityProblem, c0, method="lm", options=None):
    """c0/输出均为 Fringe 系数 waves；内部统一使用 problem 的 q 坐标。

    TRF/L-BFGS-B 经 SciPy，其迭代数按库返回报告，预算仍由统一评价器
    强制；max_iter 对 TRF 无效（库内不区分），其余方法均为自身迭代数。
    一阶与准牛顿方法在 ``supports_native_gradient`` 问题上走
    ``problem.value_gradient`` 融合通道，GN/LM/TRF 始终显式 J。
    """
    if method not in METHODS:
        raise ValueError(f"未知优化器 {method!r}; 可选 {METHODS}")
    o = options or SolverOptions()
    q = problem.to_parameters(c0)
    e = _Evaluation(problem, q, o)
    n_iter = None
    try:
        if method == "bfgs":
            result = minimize(e.value_gradient, q, jac=True, method="BFGS",
                              options={"gtol": o.gtol, "maxiter": o.max_iter})
            n_iter = int(result.nit)
            status = {0: "gradient_tolerance", 1: "iteration_budget",
                      2: "precision_loss", 3: "nonfinite_evaluation"}.get(
                          int(result.status), "scipy_failure")
        elif method == "lbfgs":
            # 无 bounds 的 L-BFGS-B：与主比较同一无约束问题，不引入边界差异。
            result = minimize(
                e.value_gradient, q, jac=True, method="L-BFGS-B", bounds=None,
                options={"maxcor": o.lbfgs_memory, "ftol": o.ftol,
                         "gtol": o.gtol, "maxiter": o.max_iter,
                         "maxfun": o.max_forward})
            n_iter = int(result.nit)
            message = str(result.message)
            status = next((s for key, s in _LBFGS_STATUS if key in message),
                          "scipy_failure")
        elif method == "trf":
            result = least_squares(
                e.residual, q, jac=e.jacobian, method="trf", x_scale=1.0,
                gtol=o.gtol, ftol=o.ftol, xtol=o.xtol, max_nfev=o.max_forward,
            )
            status = {0: "forward_budget", 1: "gradient_tolerance",
                      2: "loss_tolerance", 3: "step_tolerance",
                      4: "loss_and_step_tolerance"}.get(int(result.status), "scipy_failure")
        elif method in ("adam", "momentum"):
            status, n_iter = _fixed_step(e, q, method, o)
        else:
            status, n_iter = _LOCAL[method](e, q, o)
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
        n_function_requests=e.n_function_requests,
        n_gradient_requests=e.n_gradient_requests,
        n_jacobian_requests=e.n_jacobian_requests,
        n_fused_gradient=e.n_fused_gradient,
        gradient_backend=e.gradient_backend,
    )
