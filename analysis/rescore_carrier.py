"""载频 medium-carrier 旧结果复评（T4）：固定符号口径 + 配对簇 bootstrap。

只读取旧产物（dataset.npz 真值、runs.csv 的 fitted_Z 列、multistart.csv、
metadata.json），重建光瞳 Zernike 基后重新计分；**不运行前向反演、不重新
求解**。真值仅用于事后评价，不替换原观测选解（多初值仍按 loss 最小复核）。

口径（相对旧指标的变化）
------------------------
旧 ``lsi.metrics.recovery_metrics`` 允许整体 ± 变号取最优；本脚本主口径为
**固定符号 +1** 的波前 RMS 误差（std 去 piston、保留 tilt），符号对齐结果
降级为诊断列。倾斜周期等价（tilt alias）同样仅作诊断，不判成功。

成功率分层：原工程阈值 1e-2 waves + 研究工程阈值 1e-3 waves（另报
1e-5/1e-7 连续通过率曲线），严格口径为固定符号 max_coeff_error<1e-6 且
intensity_rms<1e-10。以上阈值均为研究口径，不是应用性能承诺。

统计单位是 20 个独立 truth；同一 truth 内 9 个非零初值为组内重复，
配对差值/比值 CI 用 truth 簇 bootstrap（默认 20000 次重采样）。
8 进程并行下 elapsed_seconds 受 CPU 竞争污染，只记录不排名。

用法::

    python analysis/rescore_carrier.py \
        --input  <medium-carrier 目录> \
        --output <输出目录> [--bootstrap 20000] [--seed 20261009]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# 允许从仓库根外调用：把仓库根（analysis/ 的父目录）加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lsi.model import Grid, SystemConfig, zernike_matrix  # noqa: E402

METHODS = ["gd", "momentum", "adam", "bfgs", "gn", "lm", "trf"]
FIT_COLS = [f"fitted_Z{j}" for j in range(2, 14)]
# 研究口径阈值曲线（waves）：1e-2 为原工程口径，1e-3 为本研究工程口径。
WAVEFRONT_TOLS = [1e-2, 1e-3, 1e-5, 1e-7]
COEFF_TOL = 1e-6
INTENSITY_TOL = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def pupil_basis(config: dict, indices: np.ndarray) -> np.ndarray:
    """按 metadata.config 重建评价用光瞳基（与 lsi.problem 一致）。"""
    grid = Grid(n=int(config["n"]), extent=float(config["extent"]))
    x, y = grid.coords()
    pupil = grid.pupil()
    return zernike_matrix(indices, x[pupil], y[pupil])


def nominal_rms(rms_values, nominal):
    """truth_rms 列为逐真值 std 计算值（含 ~1e-17 浮点尾差），归并到配置档。"""
    nominal = np.asarray(nominal, dtype=float)
    return [float(nominal[np.argmin(np.abs(nominal - v))]) for v in rms_values]


def rescore_rows(rows, truths, basis, shear, rms_bins):
    """逐行复算固定符号误差 + 符号/别名诊断，并交叉核对原始列。"""
    out, checks = [], {"wavefront": 0.0, "coeff": 0.0, "sign": 0, "alias": 0}
    bins = dict(zip(range(len(truths)),
                    nominal_rms([float(np.std(basis @ t)) for t in truths],
                                rms_bins)))
    for r in rows:
        c = np.array([float(r[k]) for k in FIT_COLS])
        truth = truths[int(r["truth_id"])]
        # 固定符号 +1：去 piston（std）保留 tilt 的波前 RMS 误差。
        err_fixed = float(np.std(basis @ (c - truth)))
        coeff_fixed = float(np.max(np.abs(c - truth)))
        # 诊断：旧口径 ± 对齐与倾斜周期代表（复算以核对原列）。
        errs = [float(np.std(basis @ (c - s * truth))) for s in (1, -1)]
        sign = (1, -1)[int(np.argmin(errs))]
        delta = c - sign * truth
        tilt = np.isin(np.arange(len(c)), [0, 1])  # indices Z2,Z3
        periods = np.zeros(len(c), dtype=int)
        periods[tilt] = np.rint(delta[tilt] * shear).astype(int)
        alias = bool(np.any(periods))
        checks["wavefront"] = max(
            checks["wavefront"], abs(min(errs) - float(r["wavefront_rms_error"])))
        checks["coeff"] = max(
            checks["coeff"], abs(float(np.max(np.abs(delta)))
                                 - float(r["max_coeff_error"])))
        checks["sign"] += int(sign != int(r["truth_sign"]))
        checks["alias"] += int(alias != (r["tilt_alias"] == "True"))
        intensity_rms = float(r["intensity_rms"])
        rec = {
            "case_id": r["case_id"], "truth_id": r["truth_id"],
            "truth_rms": bins[int(r["truth_id"])],
            "truth_rms_measured": r["truth_rms"], "method": r["method"],
            "initial_id": r["initial_id"], "initial_rms": r["initial_rms"],
            "direction": r["direction"], "loss": r["loss"],
            "intensity_rms": r["intensity_rms"],
            "wavefront_rms_fixed": err_fixed,
            "max_coeff_error_fixed": coeff_fixed,
            "aligned_sign": sign,
            "wavefront_rms_aligned": min(errs),
            "tilt_alias": alias,
            "strict_success_fixed": bool(
                coeff_fixed < COEFF_TOL and intensity_rms < INTENSITY_TOL),
            "status": r["status"], "n_iter": r["n_iter"],
            "n_forward": r["n_forward"], "n_jacobian": r["n_jacobian"],
            "elapsed_seconds_contaminated": r["elapsed_seconds"],
        }
        for tol in WAVEFRONT_TOLS:
            rec[f"eng_{tol:g}_fixed"] = bool(err_fixed < tol)
        out.append(rec)
    return out, checks


def _rate(flags):
    return sum(flags) / len(flags) if flags else float("nan")


def method_overall(rescored):
    """每方法（非零初值）主表：固定符号成功率、连续误差、调用计数。"""
    table = []
    for m in METHODS:
        runs = [r for r in rescored if r["method"] == m
                and float(r["initial_rms"]) > 0]
        errs = np.array([r["wavefront_rms_fixed"] for r in runs])
        fwd = np.array([int(r["n_forward"]) for r in runs])
        jac = np.array([int(r["n_jacobian"]) for r in runs])
        row = {
            "method": m, "runs": len(runs),
            "strict_fixed": sum(r["strict_success_fixed"] for r in runs),
            "sign_matched": sum(r["aligned_sign"] == 1 for r in runs),
            "tilt_alias": sum(r["tilt_alias"] for r in runs),
            "err_median": float(np.median(errs)),
            "err_p90": float(np.quantile(errs, 0.9)),
            "err_max": float(errs.max()),
            "fwd_median": float(np.median(fwd)),
            "fwd_mean": float(np.mean(fwd)),
            "fwd_min": int(fwd.min()), "fwd_max": int(fwd.max()),
            "jac_median": float(np.median(jac)),
        }
        for tol in WAVEFRONT_TOLS:
            row[f"eng_{tol:g}_fixed"] = sum(r[f"eng_{tol:g}_fixed"] for r in runs)
        table.append(row)
    return table


def stratified(rescored, key, values, nonzero_only=True):
    """方法 × 分层变量的 固定符号严格/工程(1e-3) 通过率表。"""
    table = []
    for m in METHODS:
        for v in values:
            runs = [r for r in rescored if r["method"] == m
                    and np.isclose(float(r[key]), v)
                    and (not nonzero_only or float(r["initial_rms"]) > 0)]
            if not runs:
                continue
            errs = [r["wavefront_rms_fixed"] for r in runs]
            row = {
                "method": m, key: v, "runs": len(runs),
                "strict_fixed": sum(r["strict_success_fixed"] for r in runs),
                "eng_1e-2_fixed": sum(r["eng_0.01_fixed"] for r in runs),
                "eng_1e-3_fixed": sum(r["eng_0.001_fixed"] for r in runs),
                "err_median": float(np.median(errs)),
            }
            table.append(row)
    return table


def zero_initial(rescored):
    table = []
    for m in METHODS:
        runs = [r for r in rescored if r["method"] == m
                and float(r["initial_rms"]) == 0]
        row = {"method": m, "runs": len(runs),
               "strict_fixed": sum(r["strict_success_fixed"] for r in runs),
               "sign_matched": sum(r["aligned_sign"] == 1 for r in runs),
               "err_median": float(np.median(
                   [r["wavefront_rms_fixed"] for r in runs]))}
        for tol in WAVEFRONT_TOLS:
            row[f"eng_{tol:g}_fixed"] = sum(r[f"eng_{tol:g}_fixed"] for r in runs)
        table.append(row)
    return table


def multistart(rescored, ms_rows):
    """复核多初值选解：池应恰为 9 个非零初值，按观测 loss 最小。"""
    by_group = {}
    for r in rescored:
        if float(r["initial_rms"]) > 0:
            by_group.setdefault((int(r["case_id"]), r["method"]), []).append(r)
    pool_sizes = sorted({len(g) for g in by_group.values()})
    table, mismatches = [], 0
    ms_lookup = {(int(r["case_id"]), r["method"]): r for r in ms_rows}
    for (case_id, method), group in sorted(by_group.items()):
        best = min(group, key=lambda r: float(r["loss"]))
        orig = ms_lookup[(case_id, method)]
        mismatches += int(int(orig["initial_id"]) != int(best["initial_id"]))
        table.append({
            "case_id": case_id, "truth_id": best["truth_id"],
            "truth_rms": best["truth_rms"], "method": method,
            "chosen_initial_id": best["initial_id"],
            "chosen_initial_rms": best["initial_rms"],
            "loss": best["loss"],
            "wavefront_rms_fixed": best["wavefront_rms_fixed"],
            "max_coeff_error_fixed": best["max_coeff_error_fixed"],
            "aligned_sign": best["aligned_sign"],
            "strict_success_fixed": best["strict_success_fixed"],
            "eng_1e-2_fixed": best["eng_0.01_fixed"],
            "eng_1e-3_fixed": best["eng_0.001_fixed"],
            "n_forward": best["n_forward"],
        })
    summary = []
    for m in METHODS:
        rows = [r for r in table if r["method"] == m]
        errs = [float(r["wavefront_rms_fixed"]) for r in rows]
        summary.append({
            "method": m, "cases": len(rows),
            "strict_fixed": sum(bool(r["strict_success_fixed"]) for r in rows),
            "eng_1e-2_fixed": sum(bool(r["eng_1e-2_fixed"]) for r in rows),
            "eng_1e-3_fixed": sum(bool(r["eng_1e-3_fixed"]) for r in rows),
            "worst_err": max(errs), "median_err": float(np.median(errs)),
            "sign_matched": sum(r["aligned_sign"] == 1 for r in rows),
        })
    return table, summary, {"pool_sizes": pool_sizes, "selection_mismatches": mismatches}


def paired_bootstrap(rescored, ms_table, n_boot, seed):
    """以 truth 为簇的配对 bootstrap：逐簇统计量的差值/比值 CI。

    单初值口径取每簇 9 个非零初值的中位数；多初值取观测选解行。
    报告：log10 误差比中位数、n_forward 差中位数、1e-3 通过率差均值。
    """
    truths = sorted({int(r["truth_id"]) for r in rescored})
    n_clusters = len(truths)
    err_med, fwd_med, eng_frac = {}, {}, {}
    for m in METHODS:
        for t in truths:
            runs = [r for r in rescored if r["method"] == m
                    and int(r["truth_id"]) == t and float(r["initial_rms"]) > 0]
            err_med[m, t] = float(np.median(
                [r["wavefront_rms_fixed"] for r in runs]))
            fwd_med[m, t] = float(np.median([int(r["n_forward"]) for r in runs]))
            eng_frac[m, t] = _rate([r["eng_0.001_fixed"] for r in runs])
    ms_err = {(r["method"], int(r["truth_id"])):
              float(r["wavefront_rms_fixed"]) for r in ms_table}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    q = np.array([0.025, 0.5, 0.975])
    table = []
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
    for a, b in pairs:
        for label, d in (
            ("single_log10_err_ratio",
             np.log10([err_med[a, t] / err_med[b, t] for t in truths])),
            ("ms_log10_err_ratio",
             np.log10([ms_err[a, t] / ms_err[b, t] for t in truths])),
            ("single_fwd_diff",
             np.array([fwd_med[a, t] - fwd_med[b, t] for t in truths])),
            ("single_eng1e-3_frac_diff",
             np.array([eng_frac[a, t] - eng_frac[b, t] for t in truths])),
        ):
            d = np.asarray(d, dtype=float)
            boot = np.median(d[idx], axis=1) if "frac" not in label else \
                np.mean(d[idx], axis=1)
            lo, mid, hi = np.quantile(boot, q)
            table.append({
                "statistic": label, "method_a": a, "method_b": b,
                "point": float(np.median(d) if "frac" not in label
                               else np.mean(d)),
                "ci_lo": float(lo), "ci_med": float(mid), "ci_hi": float(hi),
            })
    return table


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x):
    return f"{x:.3e}" if isinstance(x, float) else str(x)


def write_report(path, overall, ms_summary, boot, checks, ms_checks,
                 input_dir, config):
    lines = [
        "# medium-carrier 固定符号复评（T4）", "",
        f"输入：`{input_dir}`（旧产物只读；SHA256 见同目录 MANIFEST.sha256）。",
        "口径：固定符号 +1 波前 RMS（std 去 piston、保留 tilt）；严格 = "
        "max_coeff_error<1e-6 且 intensity_rms<1e-10；工程阈值列 1e-2（原）/"
        "1e-3（研究）。阈值均为研究口径，非应用性能承诺。",
        "符号对齐与倾斜周期别名仅作诊断；多初值池 = 9 个非零初值，"
        "按观测 loss 复核选解。计时列受 8 进程并行污染，不排名。",
        "统计单位为 20 个独立 truth；CI 为 truth 簇配对 bootstrap。20 个场景"
        "不支持 99% 级可靠性断言。", "",
        "## 每方法总表（非零初值，180/方法）", "",
        "| 方法 | 严格(固定) | 1e-2 | 1e-3 | 1e-5 | 1e-7 | 误差中位 | 最差 | "
        "符号+1 | 前向中位(均值/min-max) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in overall:
        lines.append(
            f"| {r['method']} | {r['strict_fixed']}/{r['runs']} | "
            f"{r['eng_0.01_fixed']} | {r['eng_0.001_fixed']} | "
            f"{r['eng_1e-05_fixed']} | {r['eng_1e-07_fixed']} | "
            f"{r['err_median']:.3e} | {r['err_max']:.3e} | "
            f"{r['sign_matched']} | {r['fwd_median']:.0f}"
            f"({r['fwd_mean']:.1f}/{r['fwd_min']}-{r['fwd_max']}) |")
    lines += ["", "## 多初值（9 非零初值按观测 loss 选解，20 场景/方法）", "",
              "| 方法 | 严格(固定) | 1e-2 | 1e-3 | 最差误差 | 误差中位 | 符号+1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in ms_summary:
        lines.append(
            f"| {r['method']} | {r['strict_fixed']}/{r['cases']} | "
            f"{r['eng_1e-2_fixed']} | {r['eng_1e-3_fixed']} | "
            f"{r['worst_err']:.3e} | {r['median_err']:.3e} | {r['sign_matched']} |")
    lines += ["", "## 复核自检", "",
              f"- 复算 ± 对齐波前误差与原列最大绝对差：{checks['wavefront']:.3e}",
              f"- 复算对齐系数误差与原列最大绝对差：{checks['coeff']:.3e}",
              f"- 复算符号与原列不一致行数：{checks['sign']}",
              f"- 复算 tilt_alias 与原列不一致行数：{checks['alias']}",
              f"- multistart 初值池规模集合：{ms_checks['pool_sizes']}"
              "（应为 [9]，即排除零初值）",
              f"- multistart 选解与原 multistart.csv 不一致数："
              f"{ms_checks['selection_mismatches']}", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    src, out = args.input, args.output
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((src / "metadata.json").read_text())
    config = metadata["config"]
    dataset = np.load(src / "dataset.npz")
    truths, indices = dataset["truths"], dataset["indices"]
    assert list(indices) == list(range(2, 14)), "indices 与 fitted_Z 列失配"
    basis = pupil_basis(config, indices)
    shear = SystemConfig(
        wavelength_nm=config["wavelength_nm"], na=config["na"],
        period_um=config["period_um"]).s

    rows = load_runs(src / "runs.csv")
    ms_rows = load_runs(src / "multistart.csv")
    assert len(rows) == 1400, f"runs.csv 行数 {len(rows)} != 1400"
    assert len(ms_rows) == 140, f"multistart.csv 行数 {len(ms_rows)} != 140"
    assert len(truths) == 20

    rescored, checks = rescore_rows(
        rows, truths, basis, shear, config["truth_rms"])
    overall = method_overall(rescored)
    by_rms = stratified(rescored, "truth_rms",
                        sorted({float(r["truth_rms"]) for r in rescored}))
    by_init = stratified(rescored, "initial_rms",
                         sorted({float(r["initial_rms"]) for r in rescored
                                 if float(r["initial_rms"]) > 0}))
    zero = zero_initial(rescored)
    ms_table, ms_summary, ms_checks = multistart(rescored, ms_rows)
    boot = paired_bootstrap(rescored, ms_table, args.bootstrap, args.seed)

    _write_csv(out / "runs_rescored.csv", [
        {k: (f"{v:.17g}" if isinstance(v, float) else v) for k, v in r.items()}
        for r in rescored])
    for name, table in (
        ("table_method_overall.csv", overall),
        ("table_method_truth_rms.csv", by_rms),
        ("table_method_initial_rms.csv", by_init),
        ("table_zero_initial.csv", zero),
        ("table_multistart_selected.csv", ms_table),
        ("table_multistart_summary.csv", ms_summary),
        ("table_paired_bootstrap_ci.csv", boot),
    ):
        _write_csv(out / name, table)
    write_report(out / "RESCORE_REPORT.md", overall, ms_summary, boot,
                 checks, ms_checks, src, config)
    provenance = {
        "input_dir": str(src),
        "input_sha256": {p.name: sha256(p) for p in sorted(src.glob("*"))
                         if p.is_file()},
        "config": {k: config[k] for k in
                   ("acquisition", "n", "extent", "indices", "period_um",
                    "scaling", "truth_seed", "initial_seed")},
        "bootstrap": {"n_boot": args.bootstrap, "seed": args.seed,
                      "cluster": "truth_id", "n_clusters": 20},
        "thresholds": {"coeff_tol": COEFF_TOL, "intensity_tol": INTENSITY_TOL,
                       "wavefront_tols": WAVEFRONT_TOLS},
        "crosscheck": checks, "multistart_check": ms_checks,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"wrote {out}")
    print("crosscheck:", checks)
    print("multistart:", ms_checks)


if __name__ == "__main__":
    main()
