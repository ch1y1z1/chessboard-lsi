"""端点/轨迹点曲率诊断：g、J 谱、精确 Hessian 谱、二阶项规模、局部扰动。

口径：q = scale ⊙ c（RMS 坐标），r = (I - I_obs)/sqrt(M)（与
StackedIntensityProblem 一致），loss = ||r||^2/2。H = J^T J + sum r_i ∇²r_i。
"""
from __future__ import annotations

import numpy as np

from .hessian import problem_curvature


def point_diagnostics(problem, q):
    """单点的梯度/J谱/精确Hessian谱/二阶项度量。"""
    cur = problem_curvature(problem, q)
    r, J, g = cur["r"], cur["J"], cur["g"]
    gn, second, hess = cur["gn"], cur["second"], cur["hess"]
    sv = np.linalg.svd(J, compute_uv=False)
    eig_h = np.linalg.eigvalsh(hess)
    eig_gn = np.linalg.eigvalsh(gn)
    return {
        "loss": float(r @ r / 2),
        "g_inf": float(np.linalg.norm(g, ord=np.inf)),
        "g_2": float(np.linalg.norm(g)),
        "j_sv": sv.tolist(),
        "j_cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
        "gn_eig": eig_gn.tolist(),
        "h_eig": eig_h.tolist(),
        "h_min_eig": float(eig_h[0]),
        "h_max_eig": float(eig_h[-1]),
        "h_neg_count": int(np.sum(eig_h < -1e-12 * max(eig_h[-1], 1e-30))),
        "second_norm_fro": float(np.linalg.norm(second)),
        "gn_norm_fro": float(np.linalg.norm(gn)),
        "second_over_gn": float(np.linalg.norm(second)
                                / max(np.linalg.norm(gn), 1e-300)),
        "residual_rms": float(np.sqrt(2 * (r @ r / 2))),
    }


def perturbation_probe(problem, q, rng, n_dirs=24, radii=(1e-4, 1e-3, 1e-2)):
    """端点局部扰动：随机方向 + -g 与最小特征向量方向的定向检查。

    返回 loss 下降命中数、最小探测 loss、以及定向检查下降量。
    """
    q = np.asarray(q, dtype=float)
    cur = problem_curvature(problem, q)
    base = float(cur["r"] @ cur["r"] / 2)
    n = q.size
    hits, min_loss = 0, base
    for radius in radii:
        dirs = rng.normal(size=(n_dirs, n))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for d in dirs:
            loss = _loss(problem, q + radius * (1 + np.linalg.norm(q)) * d)
            min_loss = min(min_loss, loss)
            hits += int(loss < base)
    g = cur["g"]
    eigvals, eigvecs = np.linalg.eigh(cur["hess"])
    directed = {}
    for name, d in (("neg_g", -g), ("min_eigvec", eigvecs[:, 0]),
                    ("min_eigvec_neg", -eigvecs[:, 0])):
        nrm = np.linalg.norm(d)
        if nrm == 0:
            directed[name] = None
            continue
        d = d / nrm
        drops = [base - _loss(problem, q + radius * (1 + np.linalg.norm(q)) * d)
                 for radius in radii]
        directed[name] = drops
    return {"base_loss": base, "random_hits": hits,
            "random_trials": n_dirs * len(radii), "min_loss": min_loss,
            "directed_drops": directed}


def _loss(problem, q):
    r, _ = problem.evaluate(np.asarray(q, dtype=float), jacobian=False)
    return float(r @ r / 2)


def classify_endpoint(diag, probe, gtol=1e-10, hess_tol_ratio=1e-8):
    """端点证据标签（保守，多标签可叠加）。

    - stopped_gradient: ||g||_inf <= gtol（数值梯度意义驻点）
    - descent_found: 有限扰动/定向探测发现更小 loss（停机不足/鞍点证据）
    - indefinite_hessian: 精确 H 有超阈负特征值
    - local_min_candidate: 驻点 + H 半正定 + 无下降证据（疑似局部点，
      不能仅凭此断言；truth 对齐在报告层另列）
    - ill_conditioned: cond(J) 或 |lambda_max/lambda_min| 大（病态记录）
    """
    labels = []
    if diag["g_inf"] <= gtol:
        labels.append("stopped_gradient")
    if probe is not None and (probe["random_hits"] > 0 or any(
            d and max(d) > 0 for d in probe["directed_drops"].values())):
        labels.append("descent_found")
    scale = max(abs(diag["h_max_eig"]), 1e-300)
    if diag["h_min_eig"] < -hess_tol_ratio * scale:
        labels.append("indefinite_hessian")
    if ("stopped_gradient" in labels and "descent_found" not in labels
            and "indefinite_hessian" not in labels):
        labels.append("local_min_candidate")
    if diag["j_cond"] is not None and diag["j_cond"] > 1e6:
        labels.append("ill_conditioned_j")
    return labels


def trust_exact(problem, q0, *, max_iter=50, radius0=1.0, gtol=1e-12,
                eta=0.1, seed=0):
    """精确 Hessian 信赖域诊断求解器（非生产代码）。

    子问题用特征分解解 (H + mu I) p = -g, ||p|| <= radius；
    硬情形以 mu 扫描近似。返回轨迹列表。仅用于检验精确二阶信息
    在给定端点/初值附近的行为，不做超参搜索。
    """
    q = np.asarray(q0, dtype=float)
    radius = float(radius0)
    trace = []
    for it in range(max_iter + 1):
        cur = problem_curvature(problem, q)
        g = cur["g"]
        loss = float(cur["r"] @ cur["r"] / 2)
        eigvals, eigvecs = np.linalg.eigh(cur["hess"])
        trace.append({"iter": it, "loss": loss, "q": q.copy(),
                      "g_inf": float(np.linalg.norm(g, ord=np.inf)),
                      "min_eig": float(eigvals[0]), "radius": radius})
        if np.linalg.norm(g, ord=np.inf) <= gtol and eigvals[0] >= -1e-12:
            break
        # 解信赖域子问题：找 mu >= max(0,-lmin) 使 ||p(mu)||<=radius
        p, mu = _tr_step(eigvals, eigvecs, g, radius)
        if p is None:
            break
        q_new = q + p
        loss_new = _loss(problem, q_new)
        predicted = -(g @ p + 0.5 * p @ (cur["hess"] @ p))
        rho = (loss - loss_new) / predicted if predicted > 0 else -np.inf
        trace[-1].update({"mu": mu, "rho": float(rho),
                          "step_norm": float(np.linalg.norm(p))})
        if rho > eta:
            q = q_new
            radius = min(2 * radius, 100.0) if rho > 0.75 else radius
        else:
            radius *= 0.25
            if radius < 1e-14:
                break
    return trace


def _tr_step(eigvals, eigvecs, g, radius):
    """特征分解解精确 Hessian TR 子问题；返回 (p, mu)。"""
    ge = eigvecs.T @ g
    lmin = eigvals[0]

    def step(mu):
        denom = eigvals + mu
        if np.any(denom <= 0):
            return None
        return -eigvecs @ (ge / denom)

    if lmin > 0:
        p = step(0.0)
        if np.linalg.norm(p) <= radius:
            return p, 0.0
        lo, hi = 0.0, max(1.0, -lmin + np.linalg.norm(g) / radius)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            p = step(mid)
            if np.linalg.norm(p) <= radius:
                hi = mid
            else:
                lo = mid
        return step(hi), hi
    # 不定情形：mu > -lmin；扫描 + 二分
    lo = -lmin + 1e-12 * max(1.0, abs(lmin))
    hi = lo
    p = step(hi)
    while p is not None and np.linalg.norm(p) > radius:
        lo = hi
        hi *= 2.0
        if hi > 1e14:
            return None, None
        p = step(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = step(mid)
        if p is None or np.linalg.norm(p) > radius:
            lo = mid
        else:
            hi = mid
    return step(hi), hi
