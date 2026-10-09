"""光强残差对 Zernike 系数的解析二阶导数（离线诊断用）。

E = sum_k e_k, e_k = A_k exp(i phi_k) * inside_k
dE/dc_j      = i 2pi sum_k e_k Z_{k,j}
d2E/dc_jdc_l = -(2pi)^2 sum_k e_k Z_{k,j} Z_{k,l}
d2I/dc_jdc_l = 2 Re( conj(E) d2E/dc_jdc_l + dE/dc_j conj(dE/dc_l) )

坐标约定与 StackedIntensityProblem 一致：r_tilde = (I - I_obs)/sqrt(M)，
q_j = scale_j c_j，故 Hessian(r_tilde_i, q) = H2_c[i] / (sqrt(M) s_j s_l)。
"""
from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def frame_hessian(cache, coeffs, modulation=None):
    """一帧的 I、dI/dc、d2I/dc2（原始未归一化 coeff 坐标）。

    返回 (I (n_pix,), J (n_pix, n_terms), H2 (n_pix, n_terms, n_terms))。
    与 lsi.lm._frame_and_jacobian 相同的级次叠加结构，只读其 cache 格式。
    """
    n_rows = cache[0][2].size
    coeffs = np.asarray(coeffs, dtype=float)
    E = np.zeros(n_rows, dtype=complex)
    dE = np.zeros((coeffs.size, n_rows), dtype=complex)
    d2E = np.zeros((coeffs.size, coeffs.size, n_rows), dtype=complex)
    for k, amp, inside, Z in cache:
        phase = _TWO_PI * (coeffs @ Z)
        if modulation is not None:
            phase = phase + modulation[k]
        e = np.where(inside, amp * np.exp(1j * phase), 0.0)
        E += e
        dE += (2j * np.pi) * e[None, :] * Z
        d2E += -_TWO_PI**2 * np.einsum("i,ji,li->jli", e, Z, Z)
    intensity = np.abs(E) ** 2
    jac = 2.0 * np.real(np.conj(E)[None, :] * dE).T          # (n_pix, n_terms)
    h2 = 2.0 * np.real(
        np.conj(E)[None, None, :] * d2E
        + dE[:, None, :] * np.conj(dE[None, :, :])
    )                                                        # (t, t, n_pix)
    return intensity, jac, np.moveaxis(h2, -1, 0)            # (n_pix, t, t)


def problem_curvature(problem, q):
    """统一口径（q 坐标、归一化残差 r_tilde）下的曲率量。

    返回 dict：
      r, J    与 problem.evaluate(q) 一致的归一化残差/雅可比
      g       J^T r，即 loss = ||r||^2/2 的梯度
      gn      J^T J（Gauss-Newton 曲率）
      second  sum_i r_i d2r_i（残差加权的二阶项，逐帧堆叠累计）
      hess    gn + second（loss 的精确 Hessian，已对称化）
    交叉校验：内部重算的 I/J 与 problem.evaluate 输出一致。
    """
    q = np.asarray(q, dtype=float)
    c = problem.to_coefficients(q)
    r, J = problem.evaluate(q)
    mods = getattr(problem, "mods", [None])
    blocks = getattr(problem, "blocks", problem.observed[None, :])
    n_pix_frame = blocks.shape[1]
    outer = np.outer(problem.scale, problem.scale)
    gn = J.T @ J
    second = np.zeros_like(gn)
    for i_frame, mod in enumerate(mods):
        intensity, jac_raw, h2 = frame_hessian(problem.cache, c, mod)
        r_raw = intensity - blocks[i_frame]
        # 解析路径与 evaluate 的一致性（断言级校验，防坐标/归一化错配）
        seg = slice(i_frame * n_pix_frame, (i_frame + 1) * n_pix_frame)
        np.testing.assert_allclose(
            r_raw / problem.normalizer, r[seg], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            jac_raw / (problem.normalizer * problem.scale[None, :]),
            J[seg], rtol=1e-10, atol=1e-12)
        second += np.einsum("i,ijl->jl", r_raw, h2)
    second /= problem.normalizer**2 * outer
    hess = gn + second
    hess = 0.5 * (hess + hess.T)
    return {"r": r, "J": J, "g": J.T @ r, "gn": gn,
            "second": second, "hess": hess}


def fd_hessian_check(problem, q, directions, h=1e-5):
    """方向有限差分校验精确 Hessian。

    对每个单位方向 v：∇²r_i·v ≈ (J(q+hv) - J(q-hv))_i / (2h)，
    H·v = J^T(J v) + sum_i r_i (∇²r_i v)。
    返回每个方向解析/差分 Hessian-vector 的相对误差。
    """
    q = np.asarray(q, dtype=float)
    cur = problem_curvature(problem, q)
    errors = []
    for v in directions:
        v = np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)
        _, jp = problem.evaluate(q + h * v)
        _, jm = problem.evaluate(q - h * v)
        d_r = (jp - jm) / (2.0 * h)                    # ∇²r_i · v，逐像素
        hv_fd = cur["J"].T @ (cur["J"] @ v) + d_r.T @ cur["r"]
        hv_ex = cur["hess"] @ v
        errors.append(float(
            np.linalg.norm(hv_fd - hv_ex) / max(np.linalg.norm(hv_ex), 1e-30)))
    return errors
