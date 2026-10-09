"""J^T r 直算接口的数值等价性 / 内存 / 诊断计时检查（T3 验证脚本）。

对照 ``lm._frame_and_jacobian`` 的显式雅可比路径，在
n∈{128,256}、参数数∈{12,20,35}、采样∈{全图,固定坏点,黄金比子采样}、
调制∈{无,每级标量,空间载频}、系数/残差多档组合下报告：

* intensity 与 jtr=J^T(I-target) 的最大绝对误差与相对误差
  （相对分母为 max(1, max|jtr_ref|)，大尺度情形按 eps*scale 解释）；
* 内存：理论峰值字节估算 + tracemalloc 实测（注意：tracemalloc 不追踪
  NumPy 缓冲区 malloc，仅作 Python 层下限；另记录 ru_maxrss 增量作参考）；
* ``--timing``：同机诊断计时（非独占机器，结果不进入性能结论）。

用法::

    python3 analysis/carrier_gradient_check.py [--output DIR] [--timing N]

``--output`` 写 verification.json 与 T1 接入用 fixture.npz。
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tracemalloc
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsi.gradient import frame_intensity_jtr            # noqa: E402
from lsi.lm import _frame_and_jacobian                  # noqa: E402
from lsi.model import ForwardModel, Grid, SystemConfig  # noqa: E402

PARAM_SETS = {12: list(range(2, 14)), 20: list(range(2, 22)),
              35: list(range(2, 37))}


def _rows(fm, kind):
    n_pix = fm.shape[0] * fm.shape[1]
    if kind == "full":
        return np.arange(n_pix)
    if kind == "bad_pixels":
        mask = np.ones(n_pix, dtype=bool)
        mask[np.arange(0, n_pix, 7919)] = False
        mask[12345:12360] = False
        return np.flatnonzero(mask)
    if kind == "subsample":
        return np.sort(
            (np.arange(4096) * 0.6180339887498949 % 1.0 * n_pix).astype(int))
    raise ValueError(kind)


def _modulation(fm, rows, kind):
    if kind == "none":
        return None
    if kind == "scalar":
        return fm.phase_shift_deltas(0.37, -0.11)
    if kind == "spatial":
        return fm.carrier_phases(fm.config.carrier_f0).reshape(
            len(fm.order_list), -1)[:, rows]
    raise ValueError(kind)


def _coeffs(rng, n_terms, rms):
    c = rng.normal(size=n_terms)
    return c * (rms / np.linalg.norm(c) * np.sqrt(n_terms))


def estimated_peak_bytes(n_orders, n_rows, n_terms):
    """直算路径的持久峰值字节估算（不含调用方持有的 target/cache）。

    E(16M) + e_k 列表(16K*M) + I(8M) + r(8M) + acc/jtr(O(p)) +
    逐光束临时：第一遍 phase 链约 48M 字节，第二遍 w_imag 链约 24M。
    对照显式路径额外物化 dE(16pM) 与 J(8pM)。
    """
    m, p, k = n_rows, n_terms, n_orders
    direct = (64 + 16 * k) * m + 24 * p
    explicit_extra = 24 * p * m
    return {"direct_persistent": direct, "explicit_extra_jacobian": explicit_extra}


def _measure(fn):
    tracemalloc.start()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result, peak, max(0, rss1 - rss0) * 1024


def run_case(n, n_params, row_kind, mod_kind, coeff_rms, residual_kind,
             seed=20261009):
    fm = ForwardModel(SystemConfig(grid=Grid(n=n), period_um=30.0))
    indices = PARAM_SETS[n_params]
    rows = _rows(fm, row_kind)
    cache = fm.zernike_samples(indices, rows)
    modulation = _modulation(fm, rows, mod_kind)
    rng = np.random.default_rng(seed + n + n_params)
    coeffs = _coeffs(rng, len(indices), coeff_rms)
    i_ref, j_ref = _frame_and_jacobian(cache, coeffs, modulation)
    if residual_kind == "model":
        target = i_ref.copy()
        target[::7] += 1e-6
    else:
        other = _coeffs(rng, len(indices), coeff_rms)
        target, _ = _frame_and_jacobian(cache, other, modulation,
                                        compute_jacobian=False)
        target += 0.05
    jtr_ref = j_ref.T @ (i_ref - target)

    (intensity, jtr), tm_peak, rss_delta = _measure(
        lambda: frame_intensity_jtr(cache, coeffs, target, modulation))

    i_diff = float(np.abs(intensity - i_ref).max())
    g_diff = float(np.abs(jtr - jtr_ref).max())
    g_scale = max(1.0, float(np.abs(jtr_ref).max()))
    mem = estimated_peak_bytes(len(cache), rows.size, len(indices))
    return {
        "n": n, "n_params": n_params, "n_rows": int(rows.size),
        "rows": row_kind, "modulation": mod_kind, "coeff_rms": coeff_rms,
        "residual": residual_kind,
        "intensity_max_abs_diff": i_diff,
        "jtr_max_abs_diff": g_diff,
        "jtr_max_rel_diff": g_diff / g_scale,
        "jtr_ref_scale": g_scale,
        "tracemalloc_peak_bytes": int(tm_peak),
        "ru_maxrss_delta_bytes": int(rss_delta),
        **{f"est_{k}": v for k, v in mem.items()},
    }


CASES = [
    (128, 12, "full", "none", 0.1, "model"),
    (128, 12, "subsample", "spatial", 0.1, "model"),
    (128, 12, "bad_pixels", "spatial", 0.1, "large"),
    (128, 20, "bad_pixels", "scalar", 0.1, "model"),
    (128, 20, "subsample", "none", 3.0, "large"),
    (128, 35, "full", "spatial", 0.1, "model"),
    (128, 35, "bad_pixels", "spatial", 3.0, "large"),
    (128, 35, "subsample", "scalar", 1.0, "model"),
    (256, 12, "subsample", "spatial", 0.1, "model"),
    (256, 20, "full", "scalar", 0.1, "model"),
    (256, 35, "bad_pixels", "spatial", 0.3, "large"),
    (64, 20, "full", "none", 3.0, "large"),
]


def run_matrix():
    rows = [run_case(*case) for case in CASES]
    header = ("n", "p", "M", "rows", "mod", "c_rms", "res",
              "dI_max", "dg_max", "dg_rel", "tm_MB", "rss_MB")
    print(("{:>3} {:>3} {:>6} {:<10} {:<7} {:>5} {:<6} {:>9} {:>9} {:>9}"
           " {:>7} {:>7}").format(*header))
    for r in rows:
        print(("{n:>3} {n_params:>3} {n_rows:>6} {rows:<10} {modulation:<7}"
               " {coeff_rms:>5} {residual:<6} {intensity_max_abs_diff:>9.2e}"
               " {jtr_max_abs_diff:>9.2e} {jtr_max_rel_diff:>9.2e}"
               " {trm:>7.2f} {rss:>7.2f}").format(
            trm=r["tracemalloc_peak_bytes"] / 2**20,
            rss=r["ru_maxrss_delta_bytes"] / 2**20, **r))
    worst = max(r["jtr_max_rel_diff"] for r in rows)
    print(f"\nworst jtr rel diff: {worst:.3e}  (float64 预期 ~1e-12 内)")
    return rows


def run_timing(repeats):
    """同机诊断计时：非独占条件，只用于 sanity，不作性能证据。"""
    from time import perf_counter

    fm = ForwardModel(SystemConfig(grid=Grid(n=256), period_um=30.0))
    indices = PARAM_SETS[35]
    rows = _rows(fm, "full")
    cache = fm.zernike_samples(indices, rows)
    modulation = _modulation(fm, rows, "spatial")
    rng = np.random.default_rng(1)
    coeffs = _coeffs(rng, len(indices), 0.3)
    target, _ = _frame_and_jacobian(cache, coeffs, modulation,
                                    compute_jacobian=False)
    out = {}
    for name, fn in {
        "explicit_J+JTr": lambda: (
            lambda ij: ij[1].T @ (ij[0] - target))(
            _frame_and_jacobian(cache, coeffs, modulation)),
        "direct_jtr": lambda: frame_intensity_jtr(
            cache, coeffs, target, modulation),
    }.items():
        times = []
        for _ in range(repeats):
            t0 = perf_counter()
            fn()
            times.append(perf_counter() - t0)
        out[name] = {"min": min(times), "median": float(np.median(times)),
                     "times": times}
        print(f"{name:>16}: min={min(times)*1e3:.2f} ms "
              f"median={np.median(times)*1e3:.2f} ms  ({repeats} 次)")
    print("注意：共享机器上的诊断计时，不能作为性能结论。")
    return out


def write_fixture(out, rows):
    """T1 接入用最小 fixture：n=64 载频、12 参数、固定坏点子采样。"""
    fm = ForwardModel(SystemConfig(grid=Grid(n=64), period_um=30.0))
    indices = PARAM_SETS[12]
    sel = _rows(fm, "bad_pixels")
    cache = fm.zernike_samples(indices, sel)
    modulation = _modulation(fm, sel, "spatial")
    rng = np.random.default_rng(4242)
    coeffs = _coeffs(rng, len(indices), 0.2)
    i_ref, j_ref = _frame_and_jacobian(cache, coeffs, modulation)
    target = i_ref.copy()
    target[::5] += 1e-4
    intensity, jtr = frame_intensity_jtr(cache, coeffs, target, modulation)
    np.savez(
        out / "fixture.npz",
        indices=np.asarray(indices), rows=sel, coeffs=coeffs,
        target=target, modulation=modulation,
        order_list=np.asarray(fm.order_list),
        amplitudes=fm.amplitudes,
        inside=np.stack([np.asarray(c[2]) for c in cache]),
        basis=np.stack([np.asarray(c[3]) for c in cache]),
        expected_intensity=intensity, expected_jtr=jtr,
        explicit_jacobian=j_ref,
        expected_jtr_explicit=j_ref.T @ (i_ref - target),
    )
    meta = {"verification": rows}
    (out / "verification.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"fixture + verification.json -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="写 verification.json / fixture.npz 的目录")
    parser.add_argument("--timing", type=int, default=0, metavar="N",
                        help="每路径重复 N 次的同机诊断计时（非性能证据）")
    args = parser.parse_args()
    rows = run_matrix()
    if args.timing:
        timing = run_timing(args.timing)
        if args.output:
            (args.output / "timing-diagnostic.json").write_text(
                json.dumps(timing, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_fixture(args.output, rows)


if __name__ == "__main__":
    main()
