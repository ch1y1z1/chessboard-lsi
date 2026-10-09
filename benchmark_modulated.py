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
from lsi.invert import fourier_to_wavefront
from lsi.metrics import recovery_metrics
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront
from lsi.optimizers import METHODS, SolverOptions, solve
from lsi.problem import IntensityProblem, StackedIntensityProblem

EVALUATION_VERSION = "carrier-protocol-v2"
INITIAL_CLASSES = ("zero", "random", "fourier")


@dataclass
class NoiseConfig:
    """观测噪声链配置；物理语义详见 docs/carrier/protocol-v2.md。

    model:
      ``none``     干净帧直通（bad_pixel_fraction>0 时仍生成掩膜）；
      ``gaussian`` I_obs = I + N(0, sigma)，sigma 由 ``snr_db``
                   （20 log10(干净峰值/σ)，振幅约定，与旧基准一致）或
                   ``sigma``（强度单位）二者之一指定；
      ``camera``   电子域链：Poisson(λ = I · peak_signal_e / I_peak)
                   + 读出 N(0, readout_sigma_e)，可选满阱截断
                   ``full_well_e`` 与 ADU 量化（``gain_e_per_adu`` +
                   ``bit_depth`` 须同时给出），最后按同一标度折算回
                   强度单位。不内置任何实测相机参数。
    ``negative``   负强度策略："clip" 截到 0（物理、引入偏置）或
                   "keep"（保留，零均值但允许非物理负值）。
    ``bad_pixel_fraction`` 固定坏点比例，经 mask 从拟合中排除，
                   使用独立派生种子流，与像素噪声互不重叠。
    """
    model: str = "none"
    snr_db: float | None = None
    sigma: float | None = None
    peak_signal_e: float | None = None
    readout_sigma_e: float = 0.0
    full_well_e: float | None = None
    gain_e_per_adu: float | None = None
    bit_depth: int | None = None
    negative: str = "clip"
    bad_pixel_fraction: float = 0.0

    def __post_init__(self):
        if self.model not in ("none", "gaussian", "camera"):
            raise ValueError("noise.model 必须是 none/gaussian/camera")
        gaussian_params = (self.snr_db is not None, self.sigma is not None)
        camera_params = (self.peak_signal_e is not None,
                         self.readout_sigma_e != 0.0,
                         self.full_well_e is not None,
                         self.gain_e_per_adu is not None,
                         self.bit_depth is not None)
        if self.model == "gaussian":
            if any(camera_params):
                raise ValueError("gaussian 模型不接受相机链参数")
            if self.snr_db is None and self.sigma is None:
                raise ValueError("gaussian 需要 snr_db 或 sigma 之一")
            if self.snr_db is not None and self.sigma is not None:
                raise ValueError("snr_db 与 sigma 只能指定一个")
            if self.snr_db is not None and not np.isfinite(self.snr_db):
                raise ValueError("snr_db 必须为有限数")
            if self.sigma is not None and (not np.isfinite(self.sigma)
                                           or self.sigma <= 0):
                raise ValueError("sigma 必须为正有限数（强度单位）")
        elif self.model == "camera":
            if any(gaussian_params):
                raise ValueError("camera 模型不接受 snr_db/sigma")
            if (self.peak_signal_e is None or not np.isfinite(self.peak_signal_e)
                    or self.peak_signal_e <= 0):
                raise ValueError("camera 需要正的有限 peak_signal_e"
                                 "（干净峰值光强对应的期望电子数；无默认实测值）")
            if not np.isfinite(self.readout_sigma_e) or self.readout_sigma_e < 0:
                raise ValueError("readout_sigma_e 必须为非负有限数（电子）")
            if self.full_well_e is not None and (
                    not np.isfinite(self.full_well_e) or self.full_well_e <= 0):
                raise ValueError("full_well_e 必须为正有限数（电子）")
            if (self.gain_e_per_adu is None) != (self.bit_depth is None):
                raise ValueError("量化须同时给出 gain_e_per_adu 与 bit_depth")
            if self.gain_e_per_adu is not None and (
                    not np.isfinite(self.gain_e_per_adu) or self.gain_e_per_adu <= 0):
                raise ValueError("gain_e_per_adu 必须为正有限数（e-/ADU）")
            if self.bit_depth is not None and (
                    not isinstance(self.bit_depth, int)
                    or not 1 <= self.bit_depth <= 32):
                raise ValueError("bit_depth 必须是 1..32 的整数")
        elif any(gaussian_params) or any(camera_params):
            raise ValueError("model=none 不接受噪声参数")
        if self.negative not in ("clip", "keep"):
            raise ValueError("negative 必须是 clip 或 keep")
        if (not np.isfinite(self.bad_pixel_fraction)
                or not 0 <= self.bad_pixel_fraction < 1):
            raise ValueError("bad_pixel_fraction 必须在 [0,1) 内")


def _realize_frame(clean, cfg: NoiseConfig, rng):
    """单帧噪声实现，返回 (观测帧, 帧级信息)。输出保持强度单位。"""
    info = {"model": cfg.model}
    if cfg.model == "none":
        return clean.copy(), info
    if cfg.model == "gaussian":
        sigma = (float(cfg.sigma) if cfg.sigma is not None
                 else float(clean.max()) * 10.0 ** (-cfg.snr_db / 20.0))
        info.update(sigma=sigma, snr_db=cfg.snr_db,
                    reference="peak_intensity_amplitude")
        noisy = clean + rng.normal(0.0, sigma, clean.shape)
    else:
        scale = cfg.peak_signal_e / float(clean.max())  # e- / 强度单位
        lam = np.clip(clean, 0.0, None) * scale
        electrons = rng.poisson(lam).astype(float)
        if cfg.readout_sigma_e:
            electrons = electrons + rng.normal(0.0, cfg.readout_sigma_e, clean.shape)
        if cfg.full_well_e is not None:
            saturated = electrons > cfg.full_well_e
            info["saturated_pixels"] = int(saturated.sum())
            electrons = np.minimum(electrons, cfg.full_well_e)
        if cfg.gain_e_per_adu is not None:
            adu = np.floor(np.clip(electrons, 0.0, None) / cfg.gain_e_per_adu)
            electrons = np.clip(adu, 0.0, 2.0 ** cfg.bit_depth - 1.0) * cfg.gain_e_per_adu
        noisy = electrons / scale
        info.update(peak_signal_e=cfg.peak_signal_e,
                    readout_sigma_e=cfg.readout_sigma_e,
                    electrons_per_unit=float(scale))
    if cfg.negative == "clip":
        info["clipped_pixels"] = int((noisy < 0).sum())
        noisy = np.clip(noisy, 0.0, None)
    else:
        info["negative_pixels_kept"] = int((noisy < 0).sum())
    return noisy, info


def realize_observation(clean_frames, cfg: NoiseConfig, seed):
    """对一个 case 的全部帧实现噪声与坏点，返回 (frames, mask, info)。

    ``seed`` 为可序列化熵源（int 列表）；像素噪声与坏点各占独立派生
    通道（追加 1 / 2），同一 case 的观测被所有方法与初值共享。
    """
    rng_pix = np.random.default_rng([*seed, 1])
    rng_bad = np.random.default_rng([*seed, 2])
    frames, infos = [], []
    for clean in clean_frames:
        noisy, info = _realize_frame(np.asarray(clean, dtype=float), cfg, rng_pix)
        frames.append(noisy)
        infos.append(info)
    mask = None
    bad = 0
    if cfg.bad_pixel_fraction > 0:
        n_pix = clean_frames[0].size
        bad = int(round(cfg.bad_pixel_fraction * n_pix))
        mask = np.ones(clean_frames[0].shape, dtype=bool)
        mask.ravel()[rng_bad.choice(n_pix, bad, replace=False)] = False
    return frames, mask, {"frames": infos, "bad_pixels": bad}


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
    # 协议 v2：非 None 时以初值类 ["zero","random","fourier"] 取代旧初值池；
    # random 类为独立于 truth 的共享随机方向（initial_random_rms 单初值），
    # fourier 类由同帧观测经载频解调生成（逐 case，不读 truth）。
    initial_classes: list[str] | None = None
    initial_random_rms: float = 0.1
    sign_policy: str = "auto"         # auto: carrier→fixed，其余→free
    noise: dict | None = None         # NoiseConfig 参数
    noise_seed: int = 20260929
    noise_repeats: int = 1
    report_tolerances: list[float] = field(
        default_factory=lambda: [1e-2, 1e-3, 1e-5, 1e-7])
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
        if self.sign_policy not in ("auto", "fixed", "free"):
            raise ValueError("sign_policy 必须是 auto/fixed/free")
        try:
            NoiseConfig(**(self.noise or {}))
        except TypeError as exc:
            raise ValueError(f"noise 含未知参数: {exc}") from exc
        if not isinstance(self.noise_repeats, int) or self.noise_repeats < 1:
            raise ValueError("noise_repeats 必须是正整数")
        if not isinstance(self.noise_seed, (int, np.integer)):
            raise ValueError("noise_seed 必须是整数")
        if self.initial_classes is not None:
            if (not self.initial_classes
                    or len(set(self.initial_classes)) != len(self.initial_classes)):
                raise ValueError("initial_classes 必须非空且不重复")
            unknown = set(self.initial_classes) - set(INITIAL_CLASSES)
            if unknown:
                raise ValueError(f"未知初值类 {sorted(unknown)}，可选 {INITIAL_CLASSES}")
            if "fourier" in self.initial_classes and self.acquisition != "carrier":
                raise ValueError("fourier 初值类仅适用于 carrier 采集")
        if not np.isfinite(self.initial_random_rms) or self.initial_random_rms <= 0:
            raise ValueError("initial_random_rms 必须为正有限数")
        if (not self.report_tolerances or not all(
                np.isfinite(t) and t > 0 for t in self.report_tolerances)):
            raise ValueError("report_tolerances 必须非空且为正有限数")
        for name in ("coeff_tolerance", "intensity_tolerance", "wavefront_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        for method in self.methods:
            SolverOptions(**(self.options | self.method_options.get(method, {})))

    def resolved_sign_policy(self):
        """auto：载频固定符号（观测打破 ± 简并）；其余协议保留自由变号。"""
        if self.sign_policy != "auto":
            return self.sign_policy
        return "fixed" if self.acquisition == "carrier" else "free"


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


def _initial_specs(config, basis):
    """初值规格列表，与 initial_info 一一对应。

    旧路径（initial_classes=None）保留原 include_zero + directions×rms 池；
    新路径按类生成：zero 零初值 / random 独立共享随机单初值 /
    fourier 占位（逐 case 由观测帧解调，见 _resolve_initials）。
    所有初值生成均不读取真值。
    """
    rng = np.random.default_rng(config.initial_seed)
    n_terms = len(config.indices)
    specs, info = [], []
    if config.initial_classes is not None:
        for cls in config.initial_classes:
            if cls == "zero":
                specs.append(np.zeros(n_terms))
                info.append({"class": "zero", "rms": 0.0, "direction": -1})
            elif cls == "random":
                direction = rng.normal(size=n_terms)
                if config.initial_coordinates == "rms":
                    direction = direction / basis.std(axis=0)
                specs.append(_normalize(direction, config.initial_random_rms, basis))
                info.append({"class": "random", "rms": config.initial_random_rms,
                             "direction": 0})
            else:
                specs.append(None)  # 逐 case 解析
                info.append({"class": "fourier", "rms": None, "direction": -1})
        return specs, info
    directions = rng.normal(size=(config.directions, n_terms))
    if config.initial_coordinates == "rms":
        directions /= basis.std(axis=0)
    if config.include_zero:
        specs.append(np.zeros(n_terms))
        info.append({"class": "zero", "rms": 0.0, "direction": -1})
    for rms in config.initial_rms:
        for direction, c in enumerate(directions):
            specs.append(_normalize(c, rms, basis))
            info.append({"class": "random", "rms": rms, "direction": direction})
    return specs, info


def _resolve_initials(config, fm, spec, frame):
    """逐 case 初值解析：fourier 类只读观测帧（载频解调），失败时退化为
    零初值并在 provenance 中显式记录，绝不读取 truth。"""
    if spec is None:
        try:
            fit, _ = fourier_to_wavefront(fm, frame, indices=list(config.indices))
            c0 = np.asarray(fit.coeffs, dtype=float)
            if c0.shape != (len(config.indices),) or not np.isfinite(c0).all():
                raise ValueError("fourier 初值非有限")
            return c0, "fourier"
        except Exception as exc:  # 不静默：行级 provenance 记录失败
            return np.zeros(len(config.indices)), f"fourier_failed:{type(exc).__name__}"
    return spec, ("zero" if not np.any(spec) else "random")


_WORKER_PROBLEMS: dict[int, StackedIntensityProblem] = {}
_WORKER_OPTIONS: dict[str, dict] = {}


def _solve_task(task):
    """并行 worker：fork 后共享父进程已建好的问题对象。

    求解器异常不中断实验：返回错误描述，由主进程落盘为
    status="solver_exception" 的运行行（见 run() 行构造）。
    """
    case_id, initial_id, c0, method = task
    options = SolverOptions(**_WORKER_OPTIONS[method])
    try:
        result = solve(_WORKER_PROBLEMS[case_id], c0, method, options)
        return case_id, initial_id, method, result, None
    except Exception as exc:
        return case_id, initial_id, method, None, f"{type(exc).__name__}: {exc}"


def _execute(config, problems, tasks):
    """按任务序产出 (case_id, initial_id, method, result, error)。

    workers>1 时经 fork 进程池并行：问题对象由子进程共享内存继承，
    逐次 elapsed_seconds 受 CPU 竞争影响，仅作记录不作时间比较依据。
    求解器异常以 error 字符串返回（串行路径同），不中断实验。
    """
    opt_by_method = {
        m: config.options | config.method_options.get(m, {}) for m in config.methods
    }
    if config.workers <= 1:
        for case_id, initial_id, c0, method in tasks:
            try:
                result = solve(problems[case_id], c0, method,
                               SolverOptions(**opt_by_method[method]))
                yield case_id, initial_id, method, result, None
            except Exception as exc:
                yield (case_id, initial_id, method, None,
                       f"{type(exc).__name__}: {exc}")
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
    noise_cfg = NoiseConfig(**(config.noise or {}))
    sign_policy = config.resolved_sign_policy()
    specs, initial_info = _initial_specs(config, basis)
    # 每个真值一次采集：帧与调制只生成一次；噪声逐 repeat 实现；
    # 观测/mask 由全部初值与方法共享。
    cases, stacks = [], []
    for truth_id, truth in enumerate(truths):
        clean_frames, mods = acquire(fm, zernike_wavefront(truth, config.indices), config)
        noisy = noise_cfg.model != "none" or noise_cfg.bad_pixel_fraction > 0
        for repeat in range(config.noise_repeats if noisy else 1):
            seed = [config.noise_seed, truth_id, repeat]
            frames, mask, noise_info = realize_observation(clean_frames, noise_cfg,
                                                           seed=seed)
            stacks.append({"frames": np.stack(frames), "mods": mods, "mask": mask})
            cases.append({"case_id": len(cases), "truth_id": truth_id,
                          "truth_rms": float(np.std(basis @ truth)),
                          "n_frames": len(frames),
                          "snr_db": noise_cfg.snr_db if noisy else None,
                          "noise_repeat": repeat, "noise_model": noise_cfg.model,
                          "noise": noise_info, "noise_seed": seed,
                          "n_valid_pixels": int(mask.sum()) if mask is not None
                          else int(np.prod(fm.shape))})
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
                "evaluation_version": EVALUATION_VERSION,
                "sign_policy": sign_policy,
                "noise": asdict(noise_cfg),
                "gradient_backend": check.gradient_backend,
                "initials_read_truth": False,
                "seed_streams": {"truth": config.truth_seed,
                                 "initial": config.initial_seed,
                                 "noise": config.noise_seed,
                                 "noise_channels": {"pixels": 1,
                                                    "bad_pixels": 2}},
                "schema": {
                    "version": EVALUATION_VERSION,
                    "phases": ["precompute", "init", "solve", "restart"],
                    "trajectory_events": ["evaluation"],
                    "cache_states": ["cold"],
                    "run_status": ["gradient_tolerance", "step_tolerance",
                                   "loss_tolerance", "loss_and_step_tolerance",
                                   "iteration_budget", "forward_budget",
                                   "jacobian_budget", "time_budget",
                                   "damping_limit", "non_descent_direction",
                                   "line_search_failed", "precision_loss",
                                   "nonfinite_parameters",
                                   "nonfinite_evaluation", "scipy_failure",
                                   "solver_exception"],
                    "failure_policy": "求解器异常记 status=solver_exception 行，"
                                      "指标列为 NaN；超时/预算状态见 status 列",
                },
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "platform": platform.platform(),
                "cases": cases, "initials": initial_info,
                "parallel_workers": config.workers,
                "timing_contaminated": config.workers > 1,
                "threads": {name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    # 全部问题对象在派发前建好：串行路径与并行 worker（fork 继承）共用。
    problems, diagnostics = [], []
    for case, stack in zip(cases, stacks):
        problem = StackedIntensityProblem(
            fm, config.indices, stack["frames"], stack["mods"],
            mask=stack["mask"], scaling=config.scaling)
        truth = truths[case["truth_id"]]
        _, truth_j = problem.evaluate(problem.to_parameters(truth))
        sv = np.linalg.svd(truth_j, compute_uv=False)
        _, zero_j = problem.evaluate(np.zeros(len(config.indices)))
        problems.append(problem)
        diagnostics.append((sv, zero_j))
    # 逐 case 解析初值（fourier 类只读观测帧），再派发任务。
    case_initials, case_provenance = [], []
    for case, stack in zip(cases, stacks):
        vectors, provenance = [], []
        for spec in specs:
            c0, src = _resolve_initials(config, fm, spec, stack["frames"][0])
            vectors.append(c0)
            provenance.append(src)
        case_initials.append(np.stack(vectors))
        case_provenance.append(provenance)
    tasks = []
    for case in cases:
        for initial_id, c0 in enumerate(case_initials[case["case_id"]]):
            rotation = initial_id % len(config.methods)
            methods = config.methods[rotation:] + config.methods[:rotation]
            for method in methods:
                tasks.append((case["case_id"], initial_id, c0, method))
    np.savez_compressed(
        out / "dataset.npz", truths=truths, indices=config.indices,
        case_initials=np.stack(case_initials),
        masks=np.stack([s["mask"] if s["mask"] is not None
                        else np.ones(fm.shape, dtype=bool) for s in stacks]),
        frames=np.stack([s["frames"] for s in stacks]),
        modulations=np.stack(
            [np.stack([np.asarray(m) for m in s["mods"]]) for s in stacks]))
    initial_nonzero = [info["class"] != "zero" for info in initial_info]
    nan = float("nan")
    metric_keys = ("wavefront_rms_error", "max_coeff_error", "piston_removed",
                   "equivalent_wavefront_rms_error", "equivalent_max_coeff_error",
                   "historical_coeff_error_pm")
    rows = []
    with gzip.open(out / "trajectories.jsonl.gz", "wt") as traces, \
            (out / "runs.csv").open("w", newline="") as stream:
        writer = None
        for case_id, initial_id, method, result, error in _execute(
                config, problems, tasks):
            case = cases[case_id]
            truth = truths[case["truth_id"]]
            sv, zero_j = diagnostics[case_id]
            info = initial_info[initial_id]
            # fourier 类逐 case 生成：记录该初值的实际波前 RMS（仅由观测决定）。
            init_rms = info["rms"]
            if info["class"] == "fourier":
                init_rms = float(np.std(
                    basis @ case_initials[case_id][initial_id]))
            base = {"case_id": case["case_id"], "truth_id": case["truth_id"],
                    "truth_rms": case["truth_rms"], "n_frames": case["n_frames"],
                    "snr_db": case["snr_db"], "noise_repeat": case["noise_repeat"],
                    "noise_model": case["noise_model"],
                    "method": method, "initial_id": initial_id,
                    "initial_class": info["class"],
                    "initial_rms": init_rms,
                    "direction": info["direction"],
                    "initial_provenance": case_provenance[case_id][initial_id],
                    "sign_policy": sign_policy,
                    "evaluation_version": EVALUATION_VERSION,
                    "cache_state": "cold"}
            trace = {"case_id": case["case_id"], "initial_id": initial_id,
                     "method": method, "initial_class": info["class"],
                     "evaluation_version": EVALUATION_VERSION}
            if result is None:
                row = base | {"loss": float("inf"), "intensity_rms": nan,
                              "status": "solver_exception", "error": error,
                              "stopped_by_tolerance": False, "n_iter": None,
                              "n_forward": 0, "n_jacobian": 0, "n_gradient": 0,
                              "elapsed_seconds": 0.0, "precompute_seconds":
                              problems[case_id].precompute_seconds,
                              "standalone_seconds":
                              problems[case_id].precompute_seconds,
                              "strict_success": False,
                              "engineering_success": False,
                              "equivalent_engineering_success": False,
                              "tilt_alias": False, "truth_sign": None,
                              "equivalent_truth_sign": None,
                              "sign_policy": sign_policy,
                              "truth_jacobian_rank": int(np.linalg.matrix_rank(truth_j)),
                              "truth_jacobian_condition": float(sv[0] / sv[-1])
                              if sv[-1] > 0 else None,
                              "zero_jacobian_max": float(np.abs(zero_j).max())}
                row.update({k: nan for k in metric_keys})
                for t in config.report_tolerances:
                    row[f"success_wf_{t:.0e}"] = False
                row.update({f"fitted_Z{j}": nan for j in config.indices})
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                traces.write(json.dumps(
                    trace | {"status": "solver_exception", "error": error,
                             "elapsed_seconds": 0.0, "history": []},
                    allow_nan=False) + "\n")
                print(f"case={case['case_id']} init={initial_id} {method:8} "
                      f"solver_exception {error}", flush=True)
                continue
            metrics = recovery_metrics(
                result.coeffs, truth, config.indices,
                problems[case_id].pupil_basis, fm.s, sign=sign_policy)
            strict = (metrics["max_coeff_error"] < config.coeff_tolerance
                      and result.rms_residual < config.intensity_tolerance)
            row = base | {
                "loss": result.loss, "intensity_rms": result.rms_residual,
                **metrics,
                "strict_success": (bool(strict) if case["noise_model"] == "none"
                                   else None),
                "engineering_success": bool(
                    metrics["wavefront_rms_error"] < config.wavefront_tolerance),
                "equivalent_engineering_success": bool(
                    metrics["equivalent_wavefront_rms_error"]
                    < config.wavefront_tolerance),
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
                "zero_jacobian_max": float(np.abs(zero_j).max()),
                "error": None}
            for t in config.report_tolerances:
                row[f"success_wf_{t:.0e}"] = bool(
                    metrics["wavefront_rms_error"] < t)
            row.update({f"fitted_Z{j}": float(c)
                        for j, c in zip(config.indices, result.coeffs)})
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            traces.write(json.dumps(
                trace | {"elapsed_seconds": result.elapsed_seconds,
                         "status": result.status, "history": result.history},
                allow_nan=False) + "\n")
            print(f"case={case['case_id']} init={initial_id} {method:8} "
                  f"loss={result.loss:.3e} Werr={metrics['wavefront_rms_error']:.3e} "
                  f"{result.status}", flush=True)
    summary = write_report(out, rows, config, sign_policy=sign_policy)
    budget_curves(out, config, cases, truths, case_initials[0], fm,
                  sign_policy=sign_policy, initial_nonzero=initial_nonzero)
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
