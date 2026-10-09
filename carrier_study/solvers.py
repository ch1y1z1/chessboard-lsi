"""求解器套件。全部求解同一个 ``LSProblem``，按观测损失保留最佳点（incumbent）。

最小二乘类（用雅可比）：
    gn            Gauss-Newton（法方程 + Cholesky）+ Armijo 回溯（本模块实现）
    lm_main       lsi.lm.levenberg_marquardt（main 分支实现，Marquardt+Nielsen）
    lm_minpack    SciPy least_squares(method="lm")，MINPACK lmder
    trf           SciPy least_squares(method="trf", tr_solver="exact")
准牛顿 / 共轭梯度（只用梯度，SciPy）：
    bfgs, lbfgs, cg
一阶（只用梯度，本模块实现）：
    gd            最速下降 + Armijo 回溯（步长自适应放大）
    gd_fixed      固定步长梯度下降（步长 eta 需调参）
    gd_bb         Barzilai-Borwein 步长 + 单调 Armijo 保障
    nesterov      FISTA 式加速梯度 + Lipschitz 回溯 + 函数值重启
    nesterov_fixed 固定步长加速梯度 + 梯度重启（eta 需调参）
    adam          Adam（lr 需调参）

所有一阶方法必须在有限预算下运行；停止由预算或数值停滞决定。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize

from lsi.lm import levenberg_marquardt

from .problem import Budget, EarlyStop, LSProblem, Stalled

LSQ_METHODS = ("gn", "lm_main", "lm_minpack", "trf")
QN_METHODS = ("bfgs", "lbfgs", "cg")
FIRST_ORDER = ("gd", "gd_fixed", "gd_bb", "nesterov", "nesterov_fixed", "adam")
ALL_METHODS = LSQ_METHODS + QN_METHODS + FIRST_ORDER

#: 需要调参的方法及其默认值与网格
TUNABLE = {
    "gd_fixed": ("eta", 1.0, (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)),
    "nesterov_fixed": ("eta", 1.0, (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)),
    "adam": ("lr", 1e-3, (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)),
}

_MAX_ITER = 10**7


# --------------------------------------------------------------------------- #
def gn(P: LSProblem, q0, armijo: float = 1e-4, max_iter: int = 1000):
    q = np.array(q0, dtype=float)
    for _ in range(max_iter):
        r = P.residual(q)
        J = P.jac(q)
        F0 = 0.5 * float(r @ r)
        g = J.T @ r
        P.charge("NE")
        try:
            delta = -cho_solve(cho_factor(J.T @ J), g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        slope = float(g @ delta)
        if not slope < 0.0:
            return "stationary"
        a = 1.0
        for _ in range(60):
            qn = q + a * delta
            Fn = P.loss(qn)
            if Fn <= F0 + armijo * a * slope:
                break
            a *= 0.5
        else:
            return "line_search"
        q = qn
        if F0 - Fn <= 1e-15 * F0 or a * np.linalg.norm(delta) <= 1e-15 * (1.0 + np.linalg.norm(q)):
            return "converged"
    return "max_iter"


def lm_main(P: LSProblem, q0):
    def rj(q):
        r = P.residual(q)
        J = P.jac(q)
        P.charge("LS")
        return r, J

    *_, converged = levenberg_marquardt(rj, q0)
    return "converged" if converged else "max_iter"


def _lsq_scipy(P: LSProblem, q0, method: str, event: str):
    def jac(q):
        J = P.jac(q)
        P.charge(event)
        return J

    kw = dict(ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=10**6)
    if method == "trf":
        kw["tr_solver"] = "exact"
    res = least_squares(P.residual, np.array(q0, float), jac=jac, method=method, **kw)
    return f"scipy:{res.status}"


def lm_minpack(P, q0):
    return _lsq_scipy(P, q0, "lm", "QR")


def trf(P, q0):
    return _lsq_scipy(P, q0, "trf", "SVD")


def _minimize(P: LSProblem, q0, method: str):
    opts = {
        "bfgs": dict(gtol=1e-30, maxiter=_MAX_ITER),
        "lbfgs": dict(maxcor=10, ftol=0.0, gtol=0.0, maxiter=_MAX_ITER, maxfun=_MAX_ITER, maxls=50),
        "cg": dict(gtol=1e-30, maxiter=_MAX_ITER),
    }[method]
    name = {"bfgs": "BFGS", "lbfgs": "L-BFGS-B", "cg": "CG"}[method]
    res = minimize(P.loss_grad, np.array(q0, float), jac=True, method=name, options=opts)
    return f"scipy:{res.status}"


def bfgs(P, q0):
    return _minimize(P, q0, "bfgs")


def lbfgs(P, q0):
    return _minimize(P, q0, "lbfgs")


def cg(P, q0):
    return _minimize(P, q0, "cg")


# --------------------------------------------------------------------------- #
def gd(P: LSProblem, q0, a0: float = 1.0, c1: float = 1e-4):
    q = np.array(q0, dtype=float)
    F, g = P.loss_grad(q)
    a = a0
    for _ in range(_MAX_ITER):
        gg = float(g @ g)
        if gg == 0.0:
            return "stationary"
        while True:
            qn = q - a * g
            if np.array_equal(qn, q):
                return "stalled"
            if P.loss(qn) <= F - c1 * a * gg:
                break
            a *= 0.5
        q = qn
        F, g = P.loss_grad(q)
        a *= 2.0
    return "max_iter"


def gd_fixed(P: LSProblem, q0, eta: float = 1.0):
    q = np.array(q0, dtype=float)
    for _ in range(_MAX_ITER):
        F, g = P.loss_grad(q)
        if not np.isfinite(F):
            return "diverged"
        qn = q - eta * g
        if np.array_equal(qn, q):
            return "stalled"
        q = qn
    return "max_iter"


def gd_bb(P: LSProblem, q0, a0: float = 1.0, c1: float = 1e-4):
    q = np.array(q0, dtype=float)
    F, g = P.loss_grad(q)
    a = a0
    for _ in range(_MAX_ITER):
        gg = float(g @ g)
        if gg == 0.0:
            return "stationary"
        while True:
            qn = q - a * g
            if np.array_equal(qn, q):
                return "stalled"
            if P.loss(qn) <= F - c1 * a * gg:
                break
            a *= 0.5
        Fn, gn_ = P.loss_grad(qn)
        s, yv = qn - q, gn_ - g
        sy = float(s @ yv)
        a = float(s @ s) / sy if sy > 0 else 2.0 * a
        q, F, g = qn, Fn, gn_
    return "max_iter"


def nesterov(P: LSProblem, q0, L0: float = 1.0):
    x = np.array(q0, dtype=float)
    Fx = P.loss(x)
    y, t, L = x.copy(), 1.0, L0
    for _ in range(_MAX_ITER):
        Fy, gy = P.loss_grad(y)
        gg = float(gy @ gy)
        if gg == 0.0:
            return "stationary"
        while True:
            xn = y - gy / L
            if np.array_equal(xn, y):
                return "stalled"
            Fxn = P.loss(xn)
            if Fxn <= Fy - 0.5 * gg / L:
                break
            L *= 2.0
            if L > 1e300:
                return "line_search"
        if Fxn > Fx:                       # 函数值重启
            y, t = x.copy(), 1.0
            continue
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, Fx, t = xn, Fxn, tn
        L *= 0.9
    return "max_iter"


def nesterov_fixed(P: LSProblem, q0, eta: float = 1.0):
    x = np.array(q0, dtype=float)
    y, t = x.copy(), 1.0
    for _ in range(_MAX_ITER):
        Fy, gy = P.loss_grad(y)
        if not np.isfinite(Fy):
            return "diverged"
        xn = y - eta * gy
        if np.array_equal(xn, x) and np.array_equal(y, x):
            return "stalled"
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        if float(gy @ (xn - x)) > 0.0:     # 梯度重启
            y, t = xn.copy(), 1.0
        else:
            y = xn + ((t - 1.0) / tn) * (xn - x)
            t = tn
        x = xn
    return "max_iter"


def adam(P: LSProblem, q0, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    q = np.array(q0, dtype=float)
    m = np.zeros_like(q)
    v = np.zeros_like(q)
    for k in range(1, _MAX_ITER):
        F, g = P.loss_grad(q)
        if not np.isfinite(F):
            return "diverged"
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        qn = q - lr * (m / (1.0 - b1**k)) / (np.sqrt(v / (1.0 - b2**k)) + eps)
        if np.array_equal(qn, q):
            return "stalled"
        q = qn
    return "max_iter"


_IMPL = {
    "gn": gn, "lm_main": lm_main, "lm_minpack": lm_minpack, "trf": trf,
    "bfgs": bfgs, "lbfgs": lbfgs, "cg": cg,
    "gd": gd, "gd_fixed": gd_fixed, "gd_bb": gd_bb,
    "nesterov": nesterov, "nesterov_fixed": nesterov_fixed, "adam": adam,
}


def run(method: str, P: LSProblem, q0, **params) -> dict:
    """运行一个求解器，捕获预算/早停/数值异常。"""
    if method in FIRST_ORDER and not np.isfinite(P.budget):
        raise ValueError(f"{method} 必须在有限预算下运行")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            status = _IMPL[method](P, q0, **params)
    except Budget:
        status = "budget"
    except EarlyStop:
        status = "early_stop"
    except Stalled:
        status = "stalled"
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        status = f"error:{type(exc).__name__}"
    return {
        "status": status,
        "q": None if P.best_q is None else P.best_q.copy(),
        "loss": P.best_loss,
        "cost": P.cost,
        "counts": dict(P.counts),
        "n_points": P.n_points,
    }


def method_label(method: str, params: dict | None = None) -> str:
    if not params:
        return method
    return method + "[" + ",".join(f"{k}={v:g}" for k, v in params.items()) + "]"
