"""单帧无调制优化器比较。运行方式及配置见 README.md。

所有观测先生成并保存；求解器只接收观测和固定初值，报告层独立访问真值。
默认重现已有真值及31初值。--config 支持随机波前、噪声、模态数及预算。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# 默认单线程，调用者可在启动进程前覆盖；实际环境写入 metadata。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from lsi.metrics import recovery_metrics
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_matrix, zernike_wavefront
from lsi.optimizers import METHODS, SolverOptions, solve
from lsi.problem import IntensityProblem


@dataclass
class BenchmarkConfig:
    n: int = 128
    extent: float = 1.10
    indices: list[int] = field(default_factory=lambda: list(range(2, 14)))
    wavelength_nm: float = 632.8
    na: float = 0.34
    period_um: float = 18.0
    scaling: str = "rms"
    truth_kind: str = "demo"  # demo / dense / sparse / single
    truth_rms: list[float] = field(default_factory=lambda: [0.03, 0.1, 0.3])
    truths_per_rms: int = 10
    truth_seed: int = 20260928
    initial_seed: int = 42
    directions: int = 10
    initial_rms: list[float] = field(default_factory=lambda: [0.03, 0.1, 0.3])
    initial_coordinates: str = "coefficients"  # demo 复现；统计实验用 rms
    include_zero: bool = True
    noise_snr_db: list[float | None] = field(default_factory=lambda: [None])
    noise_repeats: int = 1
    noise_seed: int = 20260929
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    options: dict = field(default_factory=dict)
    method_options: dict = field(default_factory=dict)
    coeff_tolerance: float = 1e-6
    intensity_tolerance: float = 1e-10
    wavefront_tolerance: float = 1e-2

    def __post_init__(self):
        if self.n < 16 or not 0 < self.na <= 1 or min(
            self.extent, self.wavelength_nm, self.period_um
        ) <= 0:
            raise ValueError("网格至少16，光学参数必须有效")
        if self.truth_kind not in ("demo", "dense", "sparse", "single"):
            raise ValueError("未知 truth_kind")
        if min(self.truths_per_rms, self.directions, self.noise_repeats) < 1:
            raise ValueError("样本数/方向数必须为正")
        for values in (self.truth_rms, self.initial_rms):
            if not values or not all(np.isfinite(v) and v > 0 for v in values):
                raise ValueError("RMS 序列必须非空且为正有限值")
        if not self.noise_snr_db or any(v is not None and not np.isfinite(v)
                                        for v in self.noise_snr_db):
            raise ValueError("SNR 应为有限数，null 表示无噪声")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods 必须非空且不重复")
        if set(self.methods) - set(METHODS) or set(self.method_options) - set(METHODS):
            raise ValueError("未知优化器")
        if self.initial_coordinates not in ("coefficients", "rms"):
            raise ValueError("initial_coordinates 必须是 coefficients 或 rms")
        if self.scaling not in ("rms", "none"):
            raise ValueError("scaling 必须是 rms 或 none")
        for name in ("coeff_tolerance", "intensity_tolerance", "wavefront_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        for method in self.methods:
            SolverOptions(**(self.options | self.method_options.get(method, {})))


def _normalize(c, rms, basis):
    return c * rms / np.std(basis @ c)


def make_dataset(config):
    fm = ForwardModel(SystemConfig(
        wavelength_nm=config.wavelength_nm, na=config.na, period_um=config.period_um,
        grid=Grid(n=config.n, extent=config.extent),
    ))
    # 复用输入校验，避免 demo 缺失某个非零真值项时静默截断。
    check = IntensityProblem(fm, config.indices, np.zeros(fm.shape), scaling=config.scaling)
    basis = check.pupil_basis
    n_terms = len(config.indices)
    if config.truth_kind == "demo":
        demo = {4: .31, 5: -.12, 6: .07, 7: .42, 8: .05}
        if not set(demo).issubset(config.indices):
            raise ValueError("demo 必须包含 Z4…Z8；请选 dense/sparse/single")
        truths = np.array([[demo.get(j, 0.) for j in config.indices]])
    else:
        rng = np.random.default_rng(config.truth_seed)
        truths = []
        for rms in config.truth_rms:
            count = n_terms if config.truth_kind == "single" else config.truths_per_rms
            for k in range(count):
                c = rng.normal(size=n_terms) / basis.std(axis=0)
                if config.truth_kind == "single":
                    c = np.eye(n_terms)[k]
                elif config.truth_kind == "sparse":
                    keep = rng.choice(n_terms, min(3, n_terms), replace=False)
                    c[~np.isin(np.arange(n_terms), keep)] = 0
                truths.append(_normalize(c, rms, basis))
        truths = np.array(truths)
    rng = np.random.default_rng(config.initial_seed)
    directions = rng.normal(size=(config.directions, n_terms))
    if config.initial_coordinates == "rms":
        directions /= basis.std(axis=0)
    initials, initial_info = [], []
    if config.include_zero:
        initials.append(np.zeros(n_terms))
        initial_info.append({"rms": 0., "direction": -1})
    for rms in config.initial_rms:
        for direction, c in enumerate(directions):
            initials.append(_normalize(c, rms, basis))
            initial_info.append({"rms": rms, "direction": direction})
    cases, images = [], []
    for truth_id, truth in enumerate(truths):
        clean = fm.intensity(zernike_wavefront(truth, config.indices))
        for snr_id, snr in enumerate(config.noise_snr_db):
            for repeat in range(1 if snr is None else config.noise_repeats):
                seed = [config.noise_seed, truth_id, snr_id, repeat]
                sigma = 0. if snr is None else float(clean.max() / 10**(snr/20))
                image = clean + np.random.default_rng(seed).normal(0, sigma, clean.shape)
                images.append(image)
                cases.append({"case_id": len(cases), "truth_id": truth_id,
                              "truth_rms": float(np.std(basis @ truth)),
                              "snr_db": snr, "noise_repeat": repeat,
                              "noise_sigma": sigma, "noise_seed": seed})
    return fm, truths, np.array(initials), initial_info, np.array(images), cases


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, methods):
    summaries, selected = [], []
    for method in methods:
        # 零初值是单独的退化诊断，不计入主要恢复率。
        runs = [r for r in rows if r["method"] == method and r["initial_rms"] > 0]
        groups = {}
        for run in runs:
            groups.setdefault(run["case_id"], []).append(run)
        chosen = []
        for case_id, group in groups.items():
            best = min(group, key=lambda row: row["loss"])
            selected.append(best | {"total_seconds": sum(r["elapsed_seconds"] for r in group),
                                    "total_forward": sum(r["n_forward"] for r in group)})
            chosen.append(best)
        summaries.append({
            "method": method, "runs": len(runs),
            "strict_trials": sum(r["strict_success"] is not None for r in runs),
            "strict_cases": sum(r["strict_success"] is not None for r in chosen),
            "strict_successes": sum(bool(r["strict_success"]) for r in runs),
            "engineering_successes": sum(r["engineering_success"] for r in runs),
            "cases": len(chosen),
            "multistart_strict_successes": sum(bool(r["strict_success"]) for r in chosen),
            "multistart_engineering_successes": sum(r["engineering_success"] for r in chosen),
            "median_wavefront_error": float(np.median([r["wavefront_rms_error"] for r in runs])),
            "median_seconds": float(np.median([r["elapsed_seconds"] for r in runs])),
            "total_seconds": sum(r["elapsed_seconds"] for r in runs),
        })
    return summaries, selected


def write_report(out, rows, config):
    summary, selected = summarize(rows, config.methods)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "multistart.csv", selected)
    lines = ["# 单帧无调制优化器实验", "",
             "所有方法共享单帧观测、非零初值池及 RMS/原始系数坐标配置。",
             "真值只用于事后评价；多初值最终解按观测损失最小选择。",
             "本表为配置指定的有限预算结果，超参数未经独立验证集调优，不能视为算法总体排名。", "",
             "| 方法 | 严格恢复/非零运行 | 工程恢复/非零运行 | 多初值严格恢复/场景 | 波前误差中位数 | 总秒数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['method']} | {r['strict_successes']}/{r['strict_trials']} | "
                     f"{r['engineering_successes']}/{r['runs']} | "
                     f"{r['multistart_strict_successes']}/{r['strict_cases']} | "
                     f"{r['median_wavefront_error']:.3e} | {r['total_seconds']:.2f} |")
    lines += ["", f"严格恢复要求无噪声、整体符号对齐后最大系数误差 < {config.coeff_tolerance:g} waves，"
              f"且光强 RMS 残差 < {config.intensity_tolerance:g}。",
              f"工程恢复要求整体符号对齐后波前 RMS 误差 < {config.wavefront_tolerance:g} waves。",
              "倾斜周期等价误差另存于 runs.csv；不以此掩盖原始系数越出范围。",
              "含噪声运行的 strict_success 留空；请用工程指标及连续误差评价。",
              "零初值不计入上表，详见 runs.csv。停止标记不等于恢复成功。",
              "", "时间包括求解器的轨迹记录，不含共享预计算、事后评价、绘图及文件写入。",
              "前向/J 计数是真实模型计算，包含拒绝步及仅有残差缓存时补算 J 的前向计算。",
              "梯度计数为适配层的 Jᵀr 计算次数，不含 SciPy/线性求解内部计算。",
              "TRF 的迭代数留空，使用前向/J/时间预算；max_iter 仅适用于其余方法。",
              "轨迹为评价轨迹，包含未接受候选点；每种方法均返回已评价的最低损失点。",
              "固定初值池汇总的实际耗时不同；下图给出按串行搜索顺序累计的观测选解表现。",
              "尚未统计峰值内存、配对置信区间或独立验证集调参结果。", "",
              "![非零初值结果](comparison.png)", "",
              "![多初值累计预算](budget_curves.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = [[max(r["wavefront_rms_error"], 1e-16) for r in rows
             if r["method"] == m and r["initial_rms"] > 0] for m in config.methods]
    axes[0].boxplot(data)
    axes[0].set_xticks(range(1, len(config.methods)+1), config.methods)
    axes[0].set(yscale="log", ylabel="Wavefront RMS error (waves)")
    axes[1].bar(config.methods, [r["engineering_successes"]/r["runs"] for r in summary], label="engineering")
    axes[1].set(ylabel="Engineering recovery fraction", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return summary


def budget_curves(out, config, cases, truths, initials, fm):
    """按观测损失选择 incumbent，事后计算误差；无真值选解/早停。

    横轴含前一初值完整耗时，曲线以整个固定初值池结束时间为止。
    每个场景独立计时；不把初始化次数当成独立真值样本。
    """
    x, y = fm.grid.coords()
    pupil = fm.grid.pupil()
    basis = zernike_matrix(config.indices, x[pupil], y[pupil])
    scale = basis.std(axis=0) if config.scaling == "rms" else np.ones(len(config.indices))
    curves = {}
    with gzip.open(out / "trajectories.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if not np.any(initials[record["initial_id"]]):
                continue
            key = record["method"], record["case_id"]
            curve = curves.setdefault(key, {"time": [0.], "error": [float("inf")],
                                           "cost": float("inf"), "offset": 0.})
            truth = truths[cases[record["case_id"]]["truth_id"]]
            for event in record["history"]:
                if event["loss"] < curve["cost"]:
                    curve["cost"] = event["loss"]
                    c = np.asarray(event["q"]) / scale
                    error = min(np.std(basis @ (c-truth)), np.std(basis @ (c+truth)))
                    curve["time"].append(curve["offset"] + event["elapsed_seconds"])
                    curve["error"].append(float(error))
            curve["offset"] += record["elapsed_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    points = []
    # 只在所有方法/场景都有实际运行覆盖的公共时间区间比较。
    horizon = min(curve["offset"] for curve in curves.values())
    times = np.linspace(horizon / 100, horizon, 100)
    for method in config.methods:
        errors = []
        for case in cases:
            curve = curves[method, case["case_id"]]
            positions = np.searchsorted(curve["time"], times, side="right") - 1
            errors.append(np.asarray(curve["error"])[positions])
        errors = np.array(errors)
        rates = np.mean(errors < config.wavefront_tolerance, axis=0)
        # 缺少任何一次评价时，误差为无穷；绘图用 NaN，不伪装有限误差。
        median = np.median(errors, axis=0)
        median[~np.isfinite(median)] = np.nan
        axes[0].plot(times, rates, label=method)
        axes[1].plot(times, np.maximum(median, 1e-16), label=method)
        for t, rate, err in zip(times, rates, median):
            points.append({"method": method, "seconds": t, "engineering_success_rate": rate,
                           "median_wavefront_error": err if np.isfinite(err) else None})
    axes[0].set(xlabel="Cumulative solver seconds", ylabel="Engineering recovery fraction", ylim=(0, 1.05))
    axes[1].set(xlabel="Cumulative solver seconds", ylabel="Median wavefront RMS error", yscale="log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "budget_curves.png", dpi=150)
    plt.close(fig)
    _write_csv(out / "budget_curves.csv", points)


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fm, truths, initials, initial_info, images, cases = make_dataset(config)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, "unknown"
    source_dir = out / "source"
    source_dir.mkdir()
    source_root = Path(__file__).resolve().parent
    sources = [source_root / "benchmark_optimizers.py", *sorted((source_root / "lsi").glob("*.py")),
               source_root / "pyproject.toml", source_root / "uv.lock"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(source_root)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"config": asdict(config), "revision": revision, "working_tree_status": dirty,
                "state": "running", "source_sha256": source_hashes,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "platform": platform.platform(), "cases": cases, "initials": initial_info,
                "threads": {name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    np.savez_compressed(out / "dataset.npz", truths=truths, initials=initials,
                        images=images, indices=config.indices)
    # 同一非零初值的少量预热，不使用真值；不计入测量轨迹。
    warm = IntensityProblem(fm, config.indices, images[0], scaling=config.scaling)
    for method in config.methods:
        opts = config.options | config.method_options.get(method, {})
        opts.update(max_iter=2, max_forward=5, max_jacobian=5, max_seconds=None)
        solve(warm, initials[int(config.include_zero)], method, SolverOptions(**opts))
    rows = []
    with gzip.open(out / "trajectories.jsonl.gz", "wt") as traces, (out / "runs.csv").open("w", newline="") as stream:
        writer = None
        for case, image in zip(cases, images):
            problem = IntensityProblem(fm, config.indices, image, scaling=config.scaling)
            truth = truths[case["truth_id"]]
            _, truth_j = problem.evaluate(problem.to_parameters(truth))
            sv = np.linalg.svd(truth_j, compute_uv=False)
            for initial_id, c0 in enumerate(initials):
                # 循环轮换执行顺序，减小固定方法顺序带来的时间偏差。
                rotation = initial_id % len(config.methods)
                methods = config.methods[rotation:] + config.methods[:rotation]
                for method in methods:
                    options = SolverOptions(**(config.options | config.method_options.get(method, {})))
                    result = solve(problem, c0, method, options)
                    metrics = recovery_metrics(result.coeffs, truth, config.indices, problem.pupil_basis, fm.s)
                    strict = (metrics["max_coeff_error"] < config.coeff_tolerance
                              and result.rms_residual < config.intensity_tolerance)
                    row = {"case_id": case["case_id"], "truth_id": case["truth_id"],
                           "truth_rms": case["truth_rms"], "snr_db": case["snr_db"],
                           "noise_repeat": case["noise_repeat"], "method": method,
                           "initial_id": initial_id, "initial_rms": initial_info[initial_id]["rms"],
                           "direction": initial_info[initial_id]["direction"],
                           "loss": result.loss, "intensity_rms": result.rms_residual,
                           **metrics,
                           "strict_success": bool(strict) if case["snr_db"] is None else None,
                           "engineering_success": bool(metrics["wavefront_rms_error"] < config.wavefront_tolerance),
                           "status": result.status, "stopped_by_tolerance": result.stopped_by_tolerance,
                           "n_iter": result.n_iter, "n_forward": result.n_forward,
                           "n_jacobian": result.n_jacobian, "n_gradient": result.n_gradient,
                           "elapsed_seconds": result.elapsed_seconds,
                           "precompute_seconds": result.precompute_seconds,
                           "standalone_seconds": result.elapsed_seconds + result.precompute_seconds,
                           "truth_jacobian_rank": int(np.linalg.matrix_rank(truth_j)),
                           "truth_jacobian_condition": float(sv[0]/sv[-1]) if sv[-1] > 0 else None}
                    row.update({f"fitted_Z{j}": float(c) for j, c in zip(config.indices, result.coeffs)})
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    traces.write(json.dumps({"case_id": case["case_id"], "initial_id": initial_id,
                                             "method": method, "elapsed_seconds": result.elapsed_seconds,
                                             "history": result.history}, allow_nan=False) + "\n")
                    print(f"case={case['case_id']} init={initial_id} {method:8} "
                          f"loss={result.loss:.3e} Werr={metrics['wavefront_rms_error']:.3e} "
                          f"{result.status}", flush=True)
    summary = write_report(out, rows, config)
    budget_curves(out, config, cases, truths, initials, fm)
    metadata["state"] = "complete"
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nReport: {out / 'report.md'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON 配置；未提供时复现现有31初值案例")
    parser.add_argument("--output", type=Path, default=None, help="新输出目录（拒绝覆盖）")
    args = parser.parse_args()
    config = BenchmarkConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    out = args.output or Path("output") / datetime.now(timezone.utc).strftime("optimizers-%Y%m%dT%H%M%S%fZ")
    run(config, out)


if __name__ == "__main__":
    main()
