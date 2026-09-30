"""原点驻点定性：无调制单帧 c=0 处 J(0)=0，Hessian 完全由数据决定。

解析公式（E = sum_k A_k e^{i phi_k}，phi_k = 2 pi Z_k c + mod_k）：

    dE/dc_j        = sum_k A_k e^{i phi_k} (i 2pi) Z_j^{(k)}
    d^2E/dc_j dc_k = sum_k A_k e^{i phi_k} (i 2pi)^2 Z_j^{(k)} Z_k^{(k)}
    d^2I/dc_j dc_k = 2 Re[ (dE/dc_k)^* dE/dc_j + E^* d^2E/dc_j dc_k ]

代价 F = sum_frames ||I - I_meas||^2：

    grad F = 2 J^T f， Hess F = 2 (J^T J + sum_i f_i d^2I_i)

无调制单帧 J(0) = 0 精确成立（dE 纯虚、E 实），故 Hess F(0) = 2 sum_i
f_i d^2I_i(0) 完全由测量数据决定。

脚本依次做：
    1. 解析 H 对解析 J 的中心差分校验（含 grad/Hess 整体的 FD 校验）
    2. 多个真值波前 x 多个帧设计在 c=0 的 F / grad / 特征谱
    3. 原点是局部极小时沿特征/随机方向做一维剖面扫描估计假盆地尺度
    4. 鞍点情形用 LM 微扰逃逸做实证

输出：output/origin_hessian_{summary,spectrum,basin}.csv，
      output/origin_hessian.png

运行：  uv run python analysis/origin_hessian.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lsi.lm import fit_wavefront_from_frames  # noqa: E402
from lsi.model import ForwardModel, zernike_wavefront  # noqa: E402

from common import (  # noqa: E402
    CFG_CARRIER,
    CFG_SHIFT,
    INDICES,
    SAMPLES,
    TRUTH_C,
    TRUTH_IDX,
    sample_rows,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. 解析导数
# --------------------------------------------------------------------------- #
def _order_fields(cache, coeffs, modulation):
    """逐衍射级迭代出 (e_k, Z_k)：e_k = inside_k · A_k e^{i phi_k}。"""
    for k, amp, inside, Z in cache:
        phase = 2.0 * np.pi * (coeffs @ Z)
        if modulation is not None:
            phase = phase + modulation[k]
        yield np.where(inside, amp * np.exp(1j * phase), 0.0), Z


def frame_derivatives(cache, coeffs, modulation):
    """一帧在采样像素上的 I (n_rows,)、J (n_rows,n_terms)、H (n_rows,t,t)。

    H[i, j, k] = d^2 I_i / dc_j dc_k，由 docstring 中的解析式计算。
    """
    n_rows = cache[0][2].size
    n_terms = coeffs.size
    E = np.zeros(n_rows, dtype=complex)
    dE = np.zeros((n_terms, n_rows), dtype=complex)
    ddE = np.zeros((n_terms, n_terms, n_rows), dtype=complex)
    for e, Z in _order_fields(cache, coeffs, modulation):
        E += e
        dE += (2j * np.pi) * e[None, :] * Z
        ddE += (2j * np.pi) ** 2 * e[None, None, :] * Z[:, None, :] * Z[None, :, :]
    I = np.abs(E) ** 2
    J = 2.0 * np.real(np.conj(E)[None, :] * dE).T
    # term1[i, j, k] = conj(dE[k, i]) * dE[j, i]
    # term2[i, j, k] = conj(E[i]) * ddE[j, k, i]
    H = 2.0 * np.real(
        np.conj(dE).T[:, None, :] * dE.T[:, :, None]
        + np.conj(E)[:, None, None] * ddE.transpose(2, 0, 1)
    )
    return I, J, H


def frame_intensity_only(cache, coeffs, modulation):
    """只要光强（盆地扫描用，不算 J/H）。"""
    E = sum(e for e, _ in _order_fields(cache, coeffs, modulation))
    return np.abs(E) ** 2


def cost_and_derivs(fm, indices, frames, modulations, coeffs, samples):
    """多帧堆叠的 F, grad F = 2 J^T f, Hess F = 2(J^T J + sum f_i H_i)。

    ``modulations`` 每项为 (n_orders,) 或展平到采样行的 (n_orders, n_rows)。
    """
    rows = sample_rows(fm, samples)
    cache = fm.zernike_samples(indices, rows)
    F = grad = hess = None
    for fr, mod in zip(frames, modulations):
        if mod is not None:
            mod = np.asarray(mod, dtype=float)
            if mod.ndim == 3:
                mod = mod.reshape(len(cache), -1)[:, rows]
        I, J, H = frame_derivatives(cache, coeffs, mod)
        f = I - fr.ravel()[rows]
        F_i = float(f @ f)
        g_i = 2.0 * (J.T @ f)
        h_i = 2.0 * (J.T @ J + np.einsum("i,ijk->jk", f, H))
        F = F_i if F is None else F + F_i
        grad = g_i if grad is None else grad + g_i
        hess = h_i if hess is None else hess + h_i
    return F, grad, hess


def verify_derivatives(fm, truth_wf, rng) -> None:
    """解析 H / grad / Hess 对中心差分的校验。

    覆盖三种调制路径：无调制、(n_orders,) 标量相移、
    (n_orders, n_rows) 逐像素载频相位（与生产路径同形）。
    """
    rows = sample_rows(fm, 1024)
    cache = fm.zernike_samples(INDICES, rows)
    meas = fm.intensity(truth_wf).ravel()[rows]
    c = rng.normal(scale=0.15, size=len(INDICES))  # 非零点，覆盖 J != 0
    carrier_rows = fm.carrier_phases(fm.config.carrier_f0).reshape(
        len(cache), -1)[:, rows]

    for tag, m in (
        ("unmodulated", None),
        ("dx_1/8", fm.phase_shift_deltas(1 / 8, 0)),
        ("carrier", carrier_rows),
    ):
        I0, J0, H0 = frame_derivatives(cache, c, m)
        f0 = I0 - meas

        # (a) J 对 I 的 FD（抽查两列）
        h = 1e-7
        for j in (0, 5):
            cp, cm = c.copy(), c.copy()
            cp[j] += h
            cm[j] -= h
            Ip = frame_derivatives(cache, cp, m)[0]
            Im = frame_derivatives(cache, cm, m)[0]
            np.testing.assert_allclose(J0[:, j], (Ip - Im) / (2 * h),
                                       rtol=1e-5, atol=1e-8)
        # (b) H 对 J 的 FD：H[:, j, k] = dJ[:, j]/dc_k
        h = 1e-6
        for k in range(len(INDICES)):
            cp, cm = c.copy(), c.copy()
            cp[k] += h
            cm[k] -= h
            Jp = frame_derivatives(cache, cp, m)[1]
            Jm = frame_derivatives(cache, cm, m)[1]
            np.testing.assert_allclose(H0[:, :, k], (Jp - Jm) / (2 * h),
                                       rtol=2e-4, atol=2e-6)
        # (c) grad/Hess 对 F、grad 的 FD
        def F_of(cc):
            I, _, _ = frame_derivatives(cache, cc, m)
            f = I - meas
            return float(f @ f)

        def grad_of(cc):
            _, J, _ = frame_derivatives(cache, cc, m)
            f = _frame_meas(cache, cc, m) - meas
            return 2.0 * (J.T @ f)

        g = 2.0 * (J0.T @ f0)
        Hf = 2.0 * (J0.T @ J0 + np.einsum("i,ijk->jk", f0, H0))
        h = 1e-6
        g_fd = np.empty(len(INDICES))
        for j in range(len(INDICES)):
            cp, cm = c.copy(), c.copy()
            cp[j] += h
            cm[j] -= h
            g_fd[j] = (F_of(cp) - F_of(cm)) / (2 * h)
        np.testing.assert_allclose(g, g_fd, rtol=1e-5, atol=1e-10)
        H_fd = np.empty_like(Hf)
        for j in range(len(INDICES)):
            cp, cm = c.copy(), c.copy()
            cp[j] += h
            cm[j] -= h
            H_fd[:, j] = (grad_of(cp) - grad_of(cm)) / (2 * h)
        np.testing.assert_allclose(Hf, 0.5 * (H_fd + H_fd.T), rtol=5e-4, atol=5e-7)
        print(f"    derivative check [{tag}]: J/H/grad/Hess all match FD "
              f"(|grad|={np.linalg.norm(g):.3e})")


def _frame_meas(cache, coeffs, mod):
    return frame_derivatives(cache, coeffs, mod)[0]


# --------------------------------------------------------------------------- #
# 2. c=0 定性
# --------------------------------------------------------------------------- #
def build_meas_for_truth(fm, truth_wf):
    """truth -> {label: (frames, mods)}，结构同 build_designs（6/9 设计）。"""
    dx = lambda t: fm.phase_shift_deltas(t, 0.0)
    dy = lambda t: fm.phase_shift_deltas(0.0, t)
    out = {
        "raw1": ([fm.intensity(truth_wf)], [None]),
        "dx_1/8": ([fm.intensity(truth_wf, deltas=dx(1 / 8))], [dx(1 / 8)]),
        "dy_1/8": ([fm.intensity(truth_wf, deltas=dy(1 / 8))], [dy(1 / 8)]),
        "dx+dy_1/8": (
            [fm.intensity(truth_wf, deltas=dx(1 / 8)),
             fm.intensity(truth_wf, deltas=dy(1 / 8))],
            [dx(1 / 8), dy(1 / 8)],
        ),
        "dx8+dy8": (
            [fm.intensity(truth_wf, deltas=dx(i / 8)) for i in range(8)]
            + [fm.intensity(truth_wf, deltas=dy(i / 8)) for i in range(8)],
            [dx(i / 8) for i in range(8)] + [dy(i / 8) for i in range(8)],
        ),
    }
    fmc = ForwardModel(CFG_CARRIER)
    out["carrier_p30"] = (
        [fmc.carrier_frame(truth_wf)],
        [fmc.carrier_phases(fmc.config.carrier_f0)],
    )
    return out


TRUTHS = {
    "nominal": (TRUTH_IDX, TRUTH_C),
    "neg_nominal": (TRUTH_IDX, -TRUTH_C),
    "flat_W0": (TRUTH_IDX, np.zeros(5)),
    "defocus_Z4": (np.array([4]), np.array([0.5])),
    "astig_Z5": (np.array([5]), np.array([-0.4])),
    "coma_Z7": (np.array([7]), np.array([0.4])),
    "spherical_Z9": (np.array([9]), np.array([0.3])),
    "big_2x": (TRUTH_IDX, 2.0 * TRUTH_C),
    "small_0.1x": (TRUTH_IDX, 0.1 * TRUTH_C),
}


def basin_scan(F_of, direction, t_max=1.0, n=800):
    """沿方向 u 的剖面 F(t u)：返回首个局部极大对应的 t（盆地边界估计）。

    单调上升至 t_max 记 inf；起点即下降（负曲率方向）记 0。
    """
    ts = np.linspace(0.0, t_max, n)
    vals = np.array([F_of(t * direction) for t in ts])
    if vals[1] < vals[0]:
        return 0.0, vals
    below = np.nonzero(np.diff(vals) < 0)[0]
    return (float(ts[below[0]]) if below.size else np.inf), vals


def main() -> None:
    rng = np.random.default_rng(7)
    fm = ForwardModel(CFG_SHIFT)
    fm_c = ForwardModel(CFG_CARRIER)
    zero = np.zeros(len(INDICES))

    print("== 解析导数校验 ==")
    truth_nom = zernike_wavefront(TRUTHS["nominal"][1], TRUTHS["nominal"][0])
    verify_derivatives(fm, truth_nom, rng)

    # ------------------------------------------------------------------ #
    # 2a. 谱系：truths x designs
    # ------------------------------------------------------------------ #
    print("\n== c=0 定性：真值 x 帧设计 ==")
    print(f"  {'truth':>13s} {'design':>10s} {'F(0)':>9s} {'|grad|':>9s} "
          f"{'cos(-g,c*)':>10s} {'lam_min':>10s} {'lam_max':>10s} "
          f"{'n<0':>3s} {'n>0':>3s}  class")

    summary_rows, spectra, basin_rows = [], {}, []
    designs_probe = ["raw1", "dx_1/8", "dy_1/8", "dx+dy_1/8", "dx8+dy8",
                     "carrier_p30"]

    for t_name, (t_idx, t_c) in TRUTHS.items():
        wf = zernike_wavefront(t_c, t_idx)
        meas_map = build_meas_for_truth(fm, wf)
        # 当前真值在 INDICES 顺序下的系数方向（flat 真值为 0，cos 记 nan）
        t_map = dict(zip(np.asarray(t_idx).tolist(), np.asarray(t_c).tolist()))
        c_t = np.array([t_map.get(int(j), 0.0) for j in INDICES])
        c_hat_t = c_t / np.linalg.norm(c_t) if np.linalg.norm(c_t) else None
        for label in designs_probe:
            frames, mods = meas_map[label]
            fm_d = fm_c if label == "carrier_p30" else fm
            F, g, H = cost_and_derivs(fm_d, INDICES, frames, mods,
                                      zero, SAMPLES)
            w, V = np.linalg.eigh(H)
            n_neg = int(np.count_nonzero(w < -1e-8 * max(abs(w[0]), abs(w[-1]), 1)))
            n_pos = int(np.count_nonzero(w > 1e-8 * max(abs(w[0]), abs(w[-1]), 1)))
            gn = float(np.linalg.norm(g))
            stationary = gn < 1e-9 * (1 + F)
            if not stationary:
                cls = "non-stationary"
            elif n_neg and n_pos:
                cls = "saddle"
            elif n_pos and not n_neg:
                cls = "local-min"
            elif n_neg and not n_pos:
                cls = "local-max"
            else:
                cls = "degenerate"
            cos_g = (
                float(-(g @ c_hat_t) / gn)
                if gn > 0 and c_hat_t is not None else np.nan
            )
            spectra[(t_name, label)] = (w, V)
            summary_rows.append(
                dict(truth=t_name, design=label, F0=F, grad_norm=gn,
                     cos_neg_grad_cstar=cos_g, lam_min=w[0], lam_max=w[-1],
                     n_neg=n_neg, n_pos=n_pos, stationary=stationary,
                     cls=cls)
            )
            print(f"  {t_name:>13s} {label:>10s} {F:9.3e} {gn:9.3e} "
                  f"{cos_g:10.3f} {w[0]:10.3e} {w[-1]:10.3e} "
                  f"{n_neg:3d} {n_pos:3d}  {cls}", flush=True)

    # ------------------------------------------------------------------ #
    # 2b. raw1 驻点的盆地尺度 / 鞍点逃逸
    # ------------------------------------------------------------------ #
    print("\n== raw1 原点盆地剖面扫描与微扰逃逸 ==")
    rand_dirs = rng.normal(size=(12, len(INDICES)))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    for t_name in TRUTHS:
        w, V = spectra[(t_name, "raw1")]
        t_idx, t_c = TRUTHS[t_name]
        wf = zernike_wavefront(t_c, t_idx)
        frame = fm.intensity(wf)
        rows = sample_rows(fm, SAMPLES)
        cache = fm.zernike_samples(INDICES, rows)
        meas = frame.ravel()[rows]

        def F_of(c):
            f = frame_intensity_only(cache, c, None) - meas
            return float(f @ f)

        # 只扫最小 6 个特征方向（鞍点/盆地边界由弱曲率方向决定）
        n_scan = min(6, len(INDICES))
        scans = []
        for i in range(n_scan):
            u = V[:, i]
            t_b, _ = basin_scan(F_of, u)
            t_bm, _ = basin_scan(F_of, -u)
            scans.append((i, w[i], t_b, t_bm))
            basin_rows.append(
                dict(truth=t_name, kind="eig", index=i, lam=w[i],
                     t_plus=t_b, t_minus=t_bm)
            )
        rand_bounds = []
        for u in rand_dirs:
            t_b, _ = basin_scan(F_of, u)
            rand_bounds.append(t_b)
            basin_rows.append(dict(truth=t_name, kind="rand", index=-1,
                                   lam=np.nan, t_plus=t_b, t_minus=np.nan))
        finite = [t for t in rand_bounds if np.isfinite(t)]
        med = float(np.median(finite)) if finite else np.inf
        eig_str = " ".join(f"{i}:{min(a,b):.3f}"
                           for i, _, a, b in scans[:4])
        print(f"  {t_name:>13s}: 前4特征向量的首个局部极大距离 {eig_str} | "
              f"随机方向中位 {med:.3f} "
              f"(有限 {len(finite)}/{len(rand_bounds)})", flush=True)

        # 微扰逃逸实证：沿最小特征值方向 ±eps 起步跑 LM（按当前真值分类）
        u_min = V[:, 0]
        t_idx, t_c = TRUTHS[t_name]
        t_map = dict(zip(np.asarray(t_idx).tolist(), np.asarray(t_c).tolist()))
        c_fit = np.array([t_map.get(int(j), 0.0) for j in INDICES])
        for eps in (1e-4, 1e-3):
            for sgn in (+1.0, -1.0):
                res = fit_wavefront_from_frames(
                    fm, INDICES, [frame], samples=SAMPLES, x0=sgn * eps * u_min
                )
                err_p = np.max(np.abs(res.coeffs - c_fit))
                err_m = np.max(np.abs(res.coeffs + c_fit))
                tag = "+W" if err_p < 1e-6 else ("-W" if err_m < 1e-6 else "other")
                print(f"    eps={sgn * eps:+.0e}: it={res.n_iter:3d} "
                      f"r={res.rms_residual:.2e} -> {tag} "
                      f"(|c_fin|={np.linalg.norm(res.coeffs):.3f})", flush=True)

    # ------------------------------------------------------------------ #
    # CSV / PNG
    # ------------------------------------------------------------------ #
    with open(OUT / "origin_hessian_summary.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        w_.writeheader()
        w_.writerows(summary_rows)
    with open(OUT / "origin_hessian_spectrum.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["truth", "design", "k", "eig"])
        for (t_name, label), (w, _) in spectra.items():
            for k, lam in enumerate(w):
                w_.writerow([t_name, label, k, f"{lam:.8e}"])
    with open(OUT / "origin_hessian_basin.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(basin_rows[0]))
        w_.writeheader()
        w_.writerows(basin_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for t_name in TRUTHS:
        w, _ = spectra[(t_name, "raw1")]
        ax.semilogy(np.arange(len(w)), np.abs(w), "o-", ms=4, label=t_name)
        n_neg = np.count_nonzero(w < 0)
        ax.plot([], [], " ", label=f"  ({n_neg} neg)")
    ax.set_title("eig Hess F(0), raw1 (|lam|, sorted)")
    ax.set_xlabel("index")
    ax.legend(fontsize=7)
    ax = axes[1]
    for t_name in TRUTHS:
        row = [r for r in summary_rows if r["truth"] == t_name
               and r["design"] == "raw1"][0]
        ax.barh(t_name, np.sign(row["lam_min"]) * abs(row["lam_min"]),
                color="#2ca02c" if row["cls"] == "local-min" else "#d62728")
    ax.set_title("raw1 origin: lambda_min sign by truth")
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "origin_hessian.png", dpi=150)
    print(f"\nwrote {OUT}/origin_hessian_summary.csv, "
          f"origin_hessian_spectrum.csv, origin_hessian_basin.csv, "
          f"origin_hessian.png")


if __name__ == "__main__":
    main()
