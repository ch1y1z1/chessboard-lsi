"""相移/载频采集模式下的优化器比较：对调制光强堆叠直接拟合 Zernike 系数。

观测为采集协议的原始帧——相移模式为 N 步 x + N 步 y 的干涉图序列，
载频模式为单帧载频干涉图；各帧的附加相位（光栅相移 delta 或空间
载频 ramp）作为已知调制进入前向模型，不经解调/解包裹。求解器、预算、
RMS 参数尺度、初值池及评价口径与 benchmark_optimizers.py 的单帧
无调制实验一致；真值只用于事后评价，多初值最终解按观测损失最小选择。

输出格式（dataset.npz / runs.csv / trajectories / summary / report）
与单帧基准相同。一次调用计数对应"完整采集堆叠的一次模型评价"，
不能与单帧实验的逐帧成本直接等比。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy

from benchmark_optimizers import _normalize, budget_curves, write_report
from lsi.metrics import recovery_metrics
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront
from lsi.optimizers import METHODS, SolverOptions, solve
from lsi.problem import IntensityProblem, StackedIntensityProblem


@dataclass
class ModulatedConfig:
    acquisition: str = "phase_shift"    # phase_shift / carrier
    n: int = 128
    extent: float = 1.10
    indices: list[int] = field(default_factory=lambda: list(range(2, 14)))
    wavelength_nm: float = 632.8
    na: float = 0.34
    period_um: float = 18.0             # 载频模式应设 30（论文傅里叶模式）
    n_steps: int = 8                    # 相移步数/方向；帧数 = 2*n_steps
    scaling: str = "rms"
    truth_kind: str = "dense"           # demo / dense / sparse / single
    truth_rms: list[float] = field(default_factory=lambda: [0.03, 0.1, 0.3])
    truths_per_rms: int = 10
    truth_seed: int = 20260928
    initial_seed: int = 42
    directions: int = 10
    initial_rms: list[float] = field(default_factory=lambda: [0.03, 0.1, 0.3])
    initial_coordinates: str = "rms"
    include_zero: bool = True
    workers: int = 1                  # >1 时并行求解，逐次计时被 CPU 竞争污染
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    options: dict = field(default_factory=dict)
    method_options: dict = field(default_factory=dict)
    coeff_tolerance: float = 1e-6
    intensity_tolerance: float = 1e-10
    wavefront_tolerance: float = 1e-2

    def __post_init__(self):
        if self.acquisition not in ("phase_shift", "carrier"):
            raise ValueError("acquisition 必须是 phase_shift 或 carrier")
        if self.n < 16 or not 0 < self.na <= 1 or min(
            self.extent, self.wavelength_nm, self.period_um
        ) <= 0:
            raise ValueError("网格至少16，光学参数必须有效")
        if self.acquisition == "phase_shift" and self.n_steps < 3:
            raise ValueError("相移至少需要3步")
        if self.truth_kind not in ("demo", "dense", "sparse", "single"):
            raise ValueError("未知 truth_kind")
        if min(self.truths_per_rms, self.directions) < 1:
            raise ValueError("样本数/方向数必须为正")
        if not self.truth_rms or not all(np.isfinite(v) and v > 0 for v in self.truth_rms):
            raise ValueError("RMS 序列必须非空且为正有限值")
        if not self.initial_rms or not all(np.isfinite(v) and v > 0 for v in self.initial_rms):
            raise ValueError("初值 RMS 序列必须非空且为正有限值")
        if not isinstance(self.workers, int) or not 1 <= self.workers <= (os.cpu_count() or 1):
            raise ValueError("workers 必须是 1..cpu_count 的整数")
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


def make_truths(config, basis):
    """与 benchmark_optimizers 相同的真值/方向生成，保证两实验共用真值池。"""
    n_terms = len(config.indices)
    if config.truth_kind == "demo":
        demo = {4: .31, 5: -.12, 6: .07, 7: .42, 8: .05}
        if not set(demo).issubset(config.indices):
            raise ValueError("demo 必须包含 Z4…Z8；请选 dense/sparse/single")
        return np.array([[demo.get(j, 0.) for j in config.indices]])
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
    return np.array(truths)


def acquire(fm, wf, config):
    """按采集协议生成 (frames, modulations)。"""
    if config.acquisition == "phase_shift":
        frames, mods = [], []
        for direction in ("x", "y"):
            frames.extend(fm.phase_shift_frames(wf, direction, config.n_steps))
            mods.extend(
                fm.phase_shift_deltas(i / config.n_steps, 0.0)
                if direction == "x" else
                fm.phase_shift_deltas(0.0, i / config.n_steps)
                for i in range(config.n_steps)
            )
        return frames, mods
    return [fm.carrier_frame(wf)], [fm.carrier_phases(fm.config.carrier_f0)]


_WORKER_PROBLEMS: dict[int, StackedIntensityProblem] = {}
_WORKER_OPTIONS: dict[str, dict] = {}


def _solve_task(task):
    """并行 worker：fork 后共享父进程已建好的问题对象。"""
    case_id, initial_id, c0, method = task
    options = SolverOptions(**_WORKER_OPTIONS[method])
    result = solve(_WORKER_PROBLEMS[case_id], c0, method, options)
    return case_id, initial_id, method, result


def _execute(config, problems, tasks):
    """按任务序产出 (case_id, initial_id, method, result)。

    workers>1 时经 fork 进程池并行：问题对象由子进程共享内存继承，
    逐次 elapsed_seconds 受 CPU 竞争影响，仅作记录不作时间比较依据。
    """
    opt_by_method = {
        m: config.options | config.method_options.get(m, {}) for m in config.methods
    }
    if config.workers <= 1:
        for case_id, initial_id, c0, method in tasks:
            yield case_id, initial_id, method, solve(
                problems[case_id], c0, method, SolverOptions(**opt_by_method[method]))
        return
    _WORKER_PROBLEMS.update(enumerate(problems))
    _WORKER_OPTIONS.update(opt_by_method)
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx) as pool:
        yield from pool.map(_solve_task, tasks, chunksize=1)


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fm = ForwardModel(SystemConfig(
        wavelength_nm=config.wavelength_nm, na=config.na, period_um=config.period_um,
        grid=Grid(n=config.n, extent=config.extent),
    ))
    check = IntensityProblem(fm, config.indices, np.zeros(fm.shape), scaling=config.scaling)
    basis = check.pupil_basis
    truths = make_truths(config, basis)
    rng = np.random.default_rng(config.initial_seed)
    directions = rng.normal(size=(config.directions, len(config.indices)))
    if config.initial_coordinates == "rms":
        directions /= basis.std(axis=0)
    initials, initial_info = [], []
    if config.include_zero:
        initials.append(np.zeros(len(config.indices)))
        initial_info.append({"rms": 0., "direction": -1})
    for rms in config.initial_rms:
        for direction, c in enumerate(directions):
            initials.append(_normalize(c, rms, basis))
            initial_info.append({"rms": rms, "direction": direction})
    # 每个真值一次采集：帧与调制只生成一次，供全部初值/方法共用。
    cases, stacks = [], []
    for truth_id, truth in enumerate(truths):
        frames, mods = acquire(fm, zernike_wavefront(truth, config.indices), config)
        stacks.append({"frames": np.stack(frames), "mods": mods})
        cases.append({"case_id": len(cases), "truth_id": truth_id,
                      "truth_rms": float(np.std(basis @ truth)),
                      "n_frames": len(frames), "snr_db": None, "noise_repeat": 0})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, "unknown"
    source_dir = out / "source"
    source_dir.mkdir()
    source_root = Path(__file__).resolve().parent
    sources = [source_root / "benchmark_modulated.py", *sorted((source_root / "lsi").glob("*.py")),
               source_root / "pyproject.toml", source_root / "uv.lock"]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(source_root)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"config": asdict(config), "revision": revision,
                "working_tree_status": dirty, "state": "running",
                "source_sha256": source_hashes,
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "platform": platform.platform(),
                "cases": cases, "initials": initial_info,
                "parallel_workers": config.workers,
                "timing_contaminated": config.workers > 1,
                "threads": {name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    np.savez_compressed(out / "dataset.npz", truths=truths, initials=np.array(initials),
                        indices=config.indices,
                        frames=np.stack([s["frames"] for s in stacks]),
                        modulations=np.stack(
                            [np.stack([np.asarray(m) for m in s["mods"]])
                             for s in stacks]))
    # 全部问题对象在派发前建好：串行路径与并行 worker（fork 继承）共用。
    problems, diagnostics = [], []
    for case, stack in zip(cases, stacks):
        problem = StackedIntensityProblem(
            fm, config.indices, stack["frames"], stack["mods"],
            scaling=config.scaling)
        truth = truths[case["truth_id"]]
        _, truth_j = problem.evaluate(problem.to_parameters(truth))
        sv = np.linalg.svd(truth_j, compute_uv=False)
        _, zero_j = problem.evaluate(np.zeros(len(config.indices)))
        problems.append(problem)
        diagnostics.append((sv, zero_j))
    tasks = []
    for case in cases:
        for initial_id, c0 in enumerate(initials):
            rotation = initial_id % len(config.methods)
            methods = config.methods[rotation:] + config.methods[:rotation]
            for method in methods:
                tasks.append((case["case_id"], initial_id, c0, method))
    rows = []
    with gzip.open(out / "trajectories.jsonl.gz", "wt") as traces, \
            (out / "runs.csv").open("w", newline="") as stream:
        writer = None
        for case_id, initial_id, method, result in _execute(config, problems, tasks):
            case = cases[case_id]
            truth = truths[case["truth_id"]]
            sv, zero_j = diagnostics[case_id]
            metrics = recovery_metrics(
                result.coeffs, truth, config.indices,
                problems[case_id].pupil_basis, fm.s)
            strict = (metrics["max_coeff_error"] < config.coeff_tolerance
                      and result.rms_residual < config.intensity_tolerance)
            row = {"case_id": case["case_id"], "truth_id": case["truth_id"],
                   "truth_rms": case["truth_rms"], "n_frames": case["n_frames"],
                   "snr_db": case["snr_db"], "noise_repeat": case["noise_repeat"],
                   "method": method, "initial_id": initial_id,
                   "initial_rms": initial_info[initial_id]["rms"],
                   "direction": initial_info[initial_id]["direction"],
                   "loss": result.loss, "intensity_rms": result.rms_residual,
                   **metrics,
                   "strict_success": bool(strict),
                   "engineering_success": bool(
                       metrics["wavefront_rms_error"] < config.wavefront_tolerance),
                   "status": result.status,
                   "stopped_by_tolerance": result.stopped_by_tolerance,
                   "n_iter": result.n_iter, "n_forward": result.n_forward,
                   "n_jacobian": result.n_jacobian, "n_gradient": result.n_gradient,
                   "elapsed_seconds": result.elapsed_seconds,
                   "precompute_seconds": result.precompute_seconds,
                   "standalone_seconds": result.elapsed_seconds
                   + result.precompute_seconds,
                   "truth_jacobian_rank": int(np.linalg.matrix_rank(truth_j)),
                   "truth_jacobian_condition": float(sv[0] / sv[-1])
                   if sv[-1] > 0 else None,
                   "zero_jacobian_max": float(np.abs(zero_j).max())}
            row.update({f"fitted_Z{j}": float(c)
                        for j, c in zip(config.indices, result.coeffs)})
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            traces.write(json.dumps(
                {"case_id": case["case_id"], "initial_id": initial_id,
                 "method": method, "elapsed_seconds": result.elapsed_seconds,
                 "history": result.history}, allow_nan=False) + "\n")
            print(f"case={case['case_id']} init={initial_id} {method:8} "
                  f"loss={result.loss:.3e} Werr={metrics['wavefront_rms_error']:.3e} "
                  f"{result.status}", flush=True)
    summary = write_report(out, rows, config)
    budget_curves(out, config, cases, truths, np.array(initials), fm)
    metadata["state"] = "complete"
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nReport: {out / 'report.md'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = ModulatedConfig(**json.loads(args.config.read_text()))
    out = args.output or Path("output") / datetime.now(timezone.utc).strftime(
        "modulated-%Y%m%dT%H%M%S%fZ")
    run(config, out)


if __name__ == "__main__":
    main()
