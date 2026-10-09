"""受控 instrumentation：在不复制实现的前提下捕获 LM 的 λ/ρ 轨迹。

做法：monkeypatch 求解阻尼子问题所用的模块级函数 _solve_damped，
记录每次调用的 (lambda, delta, g, diag)；ρ 由评价历史按调用序重建。
所有 patch 在 finally 中恢复；本文件不产生被修改的 LM 副本。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LMSpy:
    """_solve_damped 调用记录；与评价历史配对后重建 ρ。"""

    steps: list = field(default_factory=list)

    def wrap(self, module):
        original = module._solve_damped
        steps = self.steps

        def spy(J, f, lam):
            delta, g, diag = original(J, f, lam)
            steps.append({"lam": float(lam), "delta": delta.copy(),
                          "g": g.copy(), "diag": diag.copy()})
            return delta, g, diag

        return original, spy


def run_benchmark_lm(problem, c0, options):
    """真实 lsi.optimizers.solve(method="lm")，返回 (result, 迭代记录)。

    迭代记录：每次 _solve_damped 调用对应一次 trial 评价；按 history
    顺序配对（缓存命中不产生新事件），重建 ρ、接受标志与 q 轨迹。
    """
    import lsi.optimizers as opt

    spy = LMSpy()
    original, wrapped = spy.wrap(opt)
    opt._solve_damped = wrapped
    try:
        result = opt.solve(problem, c0, "lm", options)
    finally:
        opt._solve_damped = original

    # 配对：每次 spy 调用后先跳过 q_current 处的重评事件（接受步后补算
    # 雅可比、或拒绝步后回到旧点重评，均缓存未命中而记入 history），
    # 其后的下一个评价事件即 trial。delta==0 时 trial 命中缓存无事件。
    trials = []
    events = result.history
    cursor = 1
    q_current = np.asarray(events[0]["q"], dtype=float)
    loss_current = float(events[0]["loss"])
    for step in spy.steps:
        trial_q = q_current + step["delta"]
        if np.linalg.norm(step["delta"]) == 0.0:
            loss_new = loss_current
        else:
            while (cursor < len(events)
                   and np.array_equal(np.asarray(events[cursor]["q"]),
                                      q_current)):
                cursor += 1
            assert cursor < len(events), "spy/history 配对失败：缺 trial 事件"
            ev = events[cursor]
            assert np.array_equal(np.asarray(ev["q"]), trial_q), (
                "spy/history 配对失败")
            cursor += 1
            loss_new = float(ev["loss"])
        predicted = float(step["delta"] @ (step["lam"] * step["diag"]
                                           * step["delta"] - step["g"]) / 2)
        rho = ((loss_current - loss_new) / predicted
               if predicted > 0 else None)
        accepted = bool(rho is not None and rho > 0)
        trials.append({"lam": step["lam"], "rho": rho,
                       "accepted": accepted,
                       "loss_trial": loss_new, "loss_current": loss_current,
                       "step_norm": float(np.linalg.norm(step["delta"])),
                       "g_inf": float(np.linalg.norm(step["g"], ord=np.inf)),
                       "q_trial": trial_q.tolist()})
        if accepted:
            q_current = trial_q
            loss_current = loss_new
    return result, trials


def run_mainline_lm(fm, indices, image, c0=None, samples=4096):
    """真实 lsi.lm 单帧载频 LM，返回 (LMResult, 迭代记录, residual快照)。

    patch lsi.lm._solve_damped；residual_and_jac 包装记录每次完整
    (f,J) 评价的 cost，用于重建 ρ 与接受标志。不改变算法常数。
    """
    import lsi.lm as lm_mod

    spy = LMSpy()
    original, wrapped = spy.wrap(lm_mod)
    lm_mod._solve_damped = wrapped
    cost_log = []
    try:
        # 与 fit_wavefront_from_carrier_frame 相同装配，但加 cost 记录；
        # 不复制 LM 迭代体，评价包装只读观测。
        frames = [np.asarray(image, dtype=float)]
        n_pix = fm.shape[0] * fm.shape[1]
        if samples is not None and samples < n_pix:
            rows = np.sort(
                (np.arange(samples) * 0.6180339887498949 % 1.0
                 * n_pix).astype(int))
        else:
            rows = np.arange(n_pix)
        meas = np.concatenate([fr.ravel()[rows] for fr in frames])
        mods = [fm.carrier_phases(fm.config.carrier_f0)]
        mods = [m.reshape(len(fm.order_list), -1)[:, rows] for m in mods]
        cache = fm.zernike_samples(indices, rows)

        def residual_and_jac(c):
            f_parts, j_parts = [], []
            for mod in mods:
                I, J = lm_mod._frame_and_jacobian(cache, c, mod)
                f_parts.append(I - meas)
                j_parts.append(J)
            f, J = np.concatenate(f_parts), np.vstack(j_parts)
            cost_log.append({"c": np.asarray(c, dtype=float).copy(),
                             "cost": float(f @ f)})
            return f, J

        x0 = np.zeros(len(indices)) if c0 is None else np.asarray(c0, float)
        x, cost, n_iter, rms, converged = lm_mod.levenberg_marquardt(
            residual_and_jac, x0)
    finally:
        lm_mod._solve_damped = original

    # 配对：cost_log[0] 为初值评价；此后每次 spy 调用紧跟一个 trial 评价。
    # 例外：命中 gtol/xtol 的末次 spy 调用之后直接 break，无 trial 评价。
    trials = []
    loss_current = cost_log[0]["cost"]
    c_current = cost_log[0]["c"]
    cursor = 1
    for step in spy.steps:
        c_trial = c_current + step["delta"]
        if cursor < len(cost_log):
            ev = cost_log[cursor]
            assert np.allclose(ev["c"], c_trial, atol=1e-9), (
                "mainline spy/cost 配对失败")
            cursor += 1
            cost_new = ev["cost"]
        else:
            # 末次阻尼步命中 g/x 容差预检，未评价 trial
            cost_new = None
        predicted = float(step["delta"] @ (step["lam"] * step["diag"]
                                           * step["delta"] - step["g"]))
        rho = (((loss_current - cost_new) / predicted)
               if (cost_new is not None and predicted > 0) else None)
        accepted = bool(rho is not None and rho > 0)
        trials.append({"lam": step["lam"], "rho": rho,
                       "accepted": accepted, "cost_trial": cost_new,
                       "cost_current": loss_current,
                       "step_norm": float(np.linalg.norm(step["delta"])),
                       "g_norm": float(np.linalg.norm(step["g"])),
                       "c_trial": c_trial.tolist()})
        if accepted:
            c_current = c_trial
            loss_current = cost_new
    from lsi.lm import LMResult
    result = LMResult(indices=np.asarray(indices), coeffs=x, cost=cost,
                      n_iter=n_iter, rms_residual=rms, converged=converged)
    return result, trials, rows, cost_log


def run_scipy_lm(residual, jacobian, x0, *, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                 max_nfev=None):
    """SciPy least_squares(method='lm')（MINPACK lmder 封装）。

    MINPACK 内部 λ 不可经公开 API 读取；记录每次 fun 评价序列与
    终止状态。其余参数原生默认。
    """
    from scipy.optimize import least_squares

    calls = []

    def fun(x):
        f = residual(x)
        calls.append({"x": np.asarray(x, dtype=float).copy(),
                      "cost": float(f @ f)})
        return f

    result = least_squares(fun, np.asarray(x0, dtype=float), jac=jacobian,
                           method="lm", ftol=ftol, xtol=xtol, gtol=gtol,
                           max_nfev=max_nfev)
    return result, calls
