"""统一最小二乘目标、成本计量与预算。

    F(q) = 1/(2M) * sum_i w_i (I_i(q) - y_i)^2 = 1/2 ||r||^2,   r_i = sqrt(w_i/M) (I_i - y_i)

所有求解器只通过本类接触模型。每种原语按事件计数，再乘以串行标定的
单位成本（前向当量，F = 1）累计成本：

    F    新点的前向光强
    G    同一点上 J^T r（不构造 J）
    J    同一点上完整雅可比
    NE   一次法方程 J^T J 构造 + Cholesky 求解（GN 步）
    LS   一次 M x p 最小二乘（main 的 LM 阻尼步，增广 lstsq）
    QR   MINPACK lm 每个雅可比一次 QR
    SVD  SciPy trf(exact) 每个雅可比一次 SVD

同一点的重复请求命中缓存，不计数。超出预算时抛出 ``Budget``。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

EVENTS = ("F", "G", "J", "NE", "LS", "QR", "SVD")
DEFAULT_UNITS = {"F": 1.0, "G": 1.0, "J": 3.0, "NE": 1.0, "LS": 4.0, "QR": 4.0, "SVD": 7.0}


class Budget(Exception):
    """成本预算用尽。"""


class EarlyStop(Exception):
    """观察者要求停止（离线实验加速用，不影响已发生的轨迹）。"""


class Stalled(Exception):
    """连续大量请求都落在同一个点上：求解器已无法移动参数。"""


_MAX_CACHE_HITS = 100_000


class LSProblem:
    def __init__(
        self,
        model,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        units: dict | None = None,
        budget: float = np.inf,
        observer: Callable | None = None,
    ) -> None:
        self.model = model
        self.y = np.asarray(y, dtype=float)
        M = self.y.size
        w = np.ones(M) if weights is None else np.asarray(weights, dtype=float)
        self.sw = np.sqrt(w / M)
        self.units = dict(DEFAULT_UNITS if units is None else units)
        self.budget = float(budget)
        self.observer = observer
        self.cost = 0.0
        self.counts: dict[str, int] = defaultdict(int)
        self.n_points = 0
        self.best_loss = np.inf
        self.best_q: np.ndarray | None = None
        self._key = None
        self._hits = 0

    # ---------------------------------------------------------------- 计量
    def charge(self, event: str, k: int = 1) -> None:
        c = self.units[event] * k
        if self.cost + c > self.budget:
            raise Budget
        self.cost += c
        self.counts[event] += k

    def _at(self, q) -> None:
        q = np.asarray(q, dtype=float)
        key = q.tobytes()
        if key == self._key:
            self._hits += 1
            if self._hits > _MAX_CACHE_HITS:
                raise Stalled
            return
        self._hits = 0
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("非有限参数")
        self.charge("F")
        I, state = self.model.forward(q)
        r = self.sw * (I - self.y)
        self._key, self._q = key, q.copy()
        self._state, self._r = state, r
        self._loss = 0.5 * float(r @ r)
        self._grad = None
        self._J = None
        self.n_points += 1
        improved = self._loss < self.best_loss
        if improved:
            self.best_loss, self.best_q = self._loss, q.copy()
        if self.observer is not None:
            self.observer(self._q, self._loss, self.cost, improved)

    # ---------------------------------------------------------------- 接口
    def residual(self, q) -> np.ndarray:
        self._at(q)
        return self._r.copy()

    def loss(self, q) -> float:
        self._at(q)
        return self._loss

    def grad(self, q) -> np.ndarray:
        self._at(q)
        if self._grad is None:
            if self._J is not None:
                self._grad = self._J.T @ self._r
            else:
                self.charge("G")
                self._grad = self.model.jtv(self._state, self.sw * self._r)
        return self._grad.copy()

    def loss_grad(self, q):
        g = self.grad(q)
        return self._loss, g

    def jac(self, q) -> np.ndarray:
        self._at(q)
        if self._J is None:
            self.charge("J")
            self._J = self.sw[:, None] * self.model.jacobian(self._state)
        return self._J

    def hessian(self, q) -> np.ndarray:
        """精确 Hessian（诊断用，不计成本）。"""
        self._at(q)
        J = self.sw[:, None] * self.model.jacobian(self._state)
        return J.T @ J + self.model.hessian_terms(self._state, self.sw * self._r)
