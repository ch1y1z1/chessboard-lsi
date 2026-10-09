"""carrier-protocol-v2 协议层回归：符号策略、倾斜域、归一化梯度、
噪声语义、种子流与载频基准兼容性。"""
import csv
import importlib.util
import json

import numpy as np
import pytest

from lsi.metrics import recovery_metrics
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront
from lsi.problem import IntensityProblem, StackedIntensityProblem


@pytest.fixture
def carrier_scene():
    fm = ForwardModel(SystemConfig(grid=Grid(n=24), period_um=30.0))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    wf = zernike_wavefront(truth, indices)
    frame = fm.carrier_frame(wf)
    mods = [fm.carrier_phases(fm.config.carrier_f0)]
    problem = StackedIntensityProblem(fm, indices, [frame], mods)
    return problem, truth, frame


@pytest.fixture
def unmodulated_scene():
    fm = ForwardModel(SystemConfig(grid=Grid(n=24)))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    image = fm.intensity(zernike_wavefront(truth, indices))
    return IntensityProblem(fm, indices, image), truth


# --------------------------------------------------------------------------- #
# 符号策略与倾斜周期域
# --------------------------------------------------------------------------- #
def test_fixed_vs_free_sign_metrics(carrier_scene):
    problem, truth, _ = carrier_scene
    basis, s = problem.pupil_basis, problem.forward.s
    flipped = recovery_metrics(-truth, truth, problem.indices, basis, s,
                               sign="free")
    assert flipped["truth_sign"] == -1
    assert flipped["wavefront_rms_error"] < 1e-12
    fixed = recovery_metrics(-truth, truth, problem.indices, basis, s,
                             sign="fixed")
    assert fixed["truth_sign"] == 1 and fixed["sign_policy"] == "fixed"
    assert fixed["wavefront_rms_error"] > 0.1  # 符号翻转在载频协议下是物理错误
    assert fixed["wavefront_rms_error"] == pytest.approx(
        2.0 * np.std(basis @ truth), rel=1e-12)
    exact = recovery_metrics(truth, truth, problem.indices, basis, s,
                             sign="fixed")
    assert exact["wavefront_rms_error"] == pytest.approx(0.0, abs=1e-15)
    assert exact["piston_removed"] == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        recovery_metrics(truth, truth, problem.indices, basis, s, sign="±")


def test_tilt_alias_domains_not_silent_success(carrier_scene):
    problem, truth, _ = carrier_scene
    basis, s = problem.pupil_basis, problem.forward.s
    alias = truth.copy()
    alias[0] += 1.0 / s  # Z2 平移一个载频周期
    m = recovery_metrics(alias, truth, problem.indices, basis, s, sign="fixed")
    assert m["tilt_alias"]
    assert m["wavefront_rms_error"] > 1.0           # raw 域不被别名掩盖
    assert m["equivalent_wavefront_rms_error"] < 1e-12
    assert m["equivalent_truth_sign"] == 1          # 别名域内符号仍固定
    # 自由符号协议下的旧行为不变
    legacy = recovery_metrics(alias, truth, problem.indices, basis, s)
    assert legacy["sign_policy"] == "free" and legacy["tilt_alias"]


def test_piston_only_removed_tilt_kept(carrier_scene):
    problem, truth, _ = carrier_scene
    basis = problem.pupil_basis
    tilt_only = np.zeros_like(truth)
    tilt_only[0] = 0.05  # 纯 tilt 误差必须完整保留在主指标中
    m = recovery_metrics(truth + tilt_only, truth, problem.indices, basis,
                         problem.forward.s, sign="fixed")
    assert m["wavefront_rms_error"] == pytest.approx(
        float(np.std(basis @ tilt_only)), rel=1e-12)
    assert not m["tilt_alias"]


# --------------------------------------------------------------------------- #
# 归一化梯度 value_gradient
# --------------------------------------------------------------------------- #
def test_value_gradient_directional_and_consistent(unmodulated_scene):
    problem, truth = unmodulated_scene
    q = problem.to_parameters(truth * 0.8)
    loss, g = problem.value_gradient(q)
    r, j = problem.evaluate(q)
    assert loss == pytest.approx(float(r @ r) / 2.0, rel=1e-14)
    np.testing.assert_allclose(g, j.T @ r, rtol=1e-12, atol=1e-18)
    rng = np.random.default_rng(7)
    for _ in range(3):
        direction = rng.normal(size=q.size)
        direction /= np.linalg.norm(direction)
        h = 1e-7
        fd = (problem.value_gradient(q + h * direction)[0]
              - problem.value_gradient(q - h * direction)[0]) / (2 * h)
        assert g @ direction == pytest.approx(fd, abs=1e-8)


def test_stacked_value_gradient_directional(carrier_scene):
    problem, truth, _ = carrier_scene
    q = problem.to_parameters(truth * 1.05)
    loss, g = problem.value_gradient(q)
    r, j = problem.evaluate(q)
    assert loss == pytest.approx(float(r @ r) / 2.0, rel=1e-14)
    np.testing.assert_allclose(g, j.T @ r, rtol=1e-12, atol=1e-18)
    direction = np.zeros_like(q)
    direction[2] = 1.0
    h = 1e-7
    fd = (problem.value_gradient(q + h * direction)[0]
          - problem.value_gradient(q - h * direction)[0]) / (2 * h)
    assert g @ direction == pytest.approx(fd, abs=1e-8)


def test_gradient_backend_honest(unmodulated_scene):
    problem, _ = unmodulated_scene
    native_available = importlib.util.find_spec("lsi.gradient") is not None
    assert problem.supports_native_gradient == native_available
    if native_available:
        assert problem.gradient_backend["name"] == "native"
    else:
        # 兼容路径如实标记：一次梯度附带完整显式 J，不冒充 native。
        assert problem.gradient_backend["name"] == "explicit_jacobian_fallback"
        assert problem.gradient_backend["explicit_jacobian"] is True


# --------------------------------------------------------------------------- #
# 噪声链语义
# --------------------------------------------------------------------------- #
def test_gaussian_noise_deterministic_and_units():
    from benchmark_modulated import NoiseConfig, realize_observation
    rng_clean = np.random.default_rng(0)
    clean = np.abs(rng_clean.normal(size=(24, 24))) + 0.5
    cfg = NoiseConfig(model="gaussian", snr_db=30.0)
    f1, mask1, info1 = realize_observation([clean], cfg, seed=[1, 0, 0])
    f2, _, _ = realize_observation([clean], cfg, seed=[1, 0, 0])
    f3, _, _ = realize_observation([clean], cfg, seed=[1, 0, 1])
    np.testing.assert_array_equal(f1[0], f2[0])
    assert not np.array_equal(f1[0], f3[0])  # 独立 repeat
    assert mask1 is None
    expected_sigma = clean.max() * 10.0 ** (-30.0 / 20.0)
    assert info1["frames"][0]["sigma"] == pytest.approx(expected_sigma)
    # keep 策略保留零均值非物理负值；clip 截断并计数
    loud = NoiseConfig(model="gaussian", sigma=10.0, negative="keep")
    fk, _, ik = realize_observation([clean], loud, seed=[2, 0, 0])
    assert ik["frames"][0]["negative_pixels_kept"] > 0
    loud_clip = NoiseConfig(model="gaussian", sigma=10.0, negative="clip")
    fc, _, ic = realize_observation([clean], loud_clip, seed=[2, 0, 0])
    assert fc[0].min() >= 0.0 and ic["frames"][0]["clipped_pixels"] > 0
    np.testing.assert_allclose(np.maximum(fk[0], 0.0), fc[0])


def test_camera_chain_semantics():
    from benchmark_modulated import NoiseConfig, realize_observation
    clean = np.full((16, 16), 0.8) + np.linspace(0, 0.4, 256).reshape(16, 16)
    # 高电子数下统计涨落相对缩小：均值应接近原强度（同一标度）
    cfg = NoiseConfig(model="camera", peak_signal_e=2.0e5, readout_sigma_e=3.0)
    frames, _, info = realize_observation([clean], cfg, seed=[9, 0, 0])
    assert info["frames"][0]["electrons_per_unit"] == pytest.approx(
        2.0e5 / clean.max())
    np.testing.assert_allclose(frames[0].mean(), clean.mean(), rtol=0.02)
    # 饱和：满阱低于中位电子数 → 输出被截断且计数
    sat = NoiseConfig(model="camera", peak_signal_e=1.0e4,
                      full_well_e=5.0e3)
    fs, _, isat = realize_observation([clean], sat, seed=[9, 0, 0])
    scale = 1.0e4 / clean.max()
    assert fs[0].max() <= 5.0e3 / scale
    assert isat["frames"][0]["saturated_pixels"] > 0
    # 量化：bit_depth 限制输出离散电平数
    quant = NoiseConfig(model="camera", peak_signal_e=1.0e4,
                        readout_sigma_e=0.0, gain_e_per_adu=10.0,
                        bit_depth=4)
    fq, _, _ = realize_observation([clean], quant, seed=[9, 0, 0])
    levels = np.unique(fq[0] * scale / 10.0)
    assert len(levels) <= 16
    np.testing.assert_allclose(levels, np.floor(levels + 1e-9))


def test_bad_pixels_excluded_via_mask():
    from benchmark_modulated import NoiseConfig, realize_observation
    clean = np.ones((20, 20))
    cfg = NoiseConfig(model="none", bad_pixel_fraction=0.02)
    frames, mask, info = realize_observation([clean], cfg, seed=[3, 0, 0])
    assert info["bad_pixels"] == 8 and mask.sum() == 400 - 8
    frames2, mask2, _ = realize_observation([clean], cfg, seed=[3, 0, 0])
    np.testing.assert_array_equal(mask, mask2)  # 固定坏点确定性


@pytest.mark.parametrize("kwargs", [
    {"model": "unknown"},
    {"model": "gaussian"},                          # 缺 sigma/snr
    {"model": "gaussian", "snr_db": 30, "sigma": 1},
    {"model": "gaussian", "sigma": -1.0},
    {"model": "gaussian", "snr_db": 30, "peak_signal_e": 1e3},
    {"model": "camera"},                            # 缺 peak_signal_e
    {"model": "camera", "peak_signal_e": -1.0},
    {"model": "camera", "peak_signal_e": 1e3, "readout_sigma_e": -1},
    {"model": "camera", "peak_signal_e": 1e3, "gain_e_per_adu": 2.0},
    {"model": "camera", "peak_signal_e": 1e3, "bit_depth": 12},
    {"model": "camera", "peak_signal_e": 1e3, "bit_depth": 0,
     "gain_e_per_adu": 1.0},
    {"model": "none", "snr_db": 30.0},
    {"model": "gaussian", "sigma": 1.0, "negative": "zero"},
    {"model": "none", "bad_pixel_fraction": 1.0},
])
def test_invalid_noise_rejected(kwargs):
    from benchmark_modulated import ModulatedConfig, NoiseConfig
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)
    with pytest.raises((ValueError, TypeError)):
        ModulatedConfig(acquisition="carrier", n=16, noise=kwargs)


def test_noise_unknown_key_rejected():
    from benchmark_modulated import ModulatedConfig
    with pytest.raises(ValueError):
        ModulatedConfig(acquisition="carrier", n=16,
                        noise={"model": "none", "mystery": 1})


# --------------------------------------------------------------------------- #
# 基准层：种子流、共享观测、固定符号评价、旧路径兼容
# --------------------------------------------------------------------------- #
def _tiny_config(**extra):
    from benchmark_modulated import ModulatedConfig
    base = dict(acquisition="carrier", n=16, period_um=30.0,
                truth_kind="dense", truth_rms=[0.1], truths_per_rms=1,
                truth_seed=101, initial_seed=202,
                methods=["lm", "gd"],
                options={"max_forward": 6, "max_jacobian": 6})
    return ModulatedConfig(**(base | extra))


def test_carrier_run_fixed_sign_and_schema(tmp_path):
    from benchmark_modulated import EVALUATION_VERSION, run
    out = tmp_path / "carrier"
    summary = run(_tiny_config(initial_classes=["zero", "random"], include_zero=False), out)
    assert len(summary) == 2
    rows = list(csv.DictReader((out / "runs.csv").open()))
    assert len(rows) == 4  # 1 case x 2 initials x 2 methods
    assert {r["sign_policy"] for r in rows} == {"fixed"}
    assert {r["evaluation_version"] for r in rows} == {EVALUATION_VERSION}
    assert {r["cache_state"] for r in rows} == {"cold"}
    assert {r["initial_class"] for r in rows} == {"zero", "random"}
    for name in ("success_wf_1e-02", "success_wf_1e-03", "success_wf_1e-05",
                 "success_wf_1e-07", "equivalent_engineering_success",
                 "initial_provenance", "error"):
        assert name in rows[0]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["state"] == "complete"
    assert meta["sign_policy"] == "fixed"
    assert meta["evaluation_version"] == EVALUATION_VERSION
    assert meta["initials_read_truth"] is False
    assert meta["schema"]["run_status"]
    assert meta["gradient_backend"]["name"] in (
        "native", "explicit_jacobian_fallback")
    data = np.load(out / "dataset.npz")
    assert data["case_initials"].shape == (1, 2, len(data["indices"]))


def test_fourier_initial_reads_only_observation(tmp_path):
    """同帧 Fourier 初值：只依赖观测帧，不读 truth；同一观测初值确定。"""
    from benchmark_modulated import _initial_specs, _resolve_initials
    # 载频可分辨需要足够网格（2 f0 < Nyquist）：与 pilot 配置一致用 n=128。
    config = _tiny_config(n=128, initial_classes=["fourier"])
    fm = ForwardModel(SystemConfig(
        wavelength_nm=config.wavelength_nm, na=config.na,
        period_um=config.period_um, grid=Grid(n=config.n, extent=config.extent)))
    truth = np.array([.02, -.01, .05, -.02, .01, .03, .01] + [0.] * 5)
    frame = fm.carrier_frame(zernike_wavefront(truth, config.indices))
    specs, info = _initial_specs(config, np.ones((10, len(config.indices))))
    assert specs == [None] and info[0]["class"] == "fourier"
    c0a, srca = _resolve_initials(config, fm, None, frame)
    c0b, _ = _resolve_initials(config, fm, None, frame.copy())
    np.testing.assert_array_equal(c0a, c0b)
    assert srca == "fourier" and np.isfinite(c0a).all()
    np.testing.assert_allclose(c0a, truth, atol=2e-3)
    # fourier 初值不得用于非载频采集
    from benchmark_modulated import ModulatedConfig
    with pytest.raises(ValueError):
        ModulatedConfig(acquisition="phase_shift", n=16,
                        initial_classes=["fourier"])


def test_seed_streams_independent(tmp_path):
    """truth/initial/noise 种子互不重叠；观测帧只随 noise_seed 变。"""
    from benchmark_modulated import run
    out_a = tmp_path / "a"
    run(_tiny_config(noise={"model": "gaussian", "snr_db": 40.0},
                     noise_seed=11, initial_classes=["zero", "random"]), out_a)
    out_b = tmp_path / "b"
    run(_tiny_config(noise={"model": "gaussian", "snr_db": 40.0},
                     noise_seed=11, initial_classes=["zero", "random"],
                     initial_seed=999), out_b)
    out_c = tmp_path / "c"
    run(_tiny_config(noise={"model": "gaussian", "snr_db": 40.0},
                     noise_seed=12, initial_classes=["zero", "random"]), out_c)
    fa = np.load(out_a / "dataset.npz")["frames"]
    fb = np.load(out_b / "dataset.npz")["frames"]
    fc = np.load(out_c / "dataset.npz")["frames"]
    np.testing.assert_array_equal(fa, fb)   # initial_seed 不影响观测
    assert not np.array_equal(fa, fc)       # noise_seed 改变观测
    rows_a = list(csv.DictReader((out_a / "runs.csv").open()))
    # 含噪运行的 strict_success 留空（与旧无调制基准一致口径）
    assert all(r["strict_success"] == "" for r in rows_a)
    # 全部方法/初值共享同一观测
    assert len({r["case_id"] for r in rows_a}) == 1


def test_legacy_noiseless_carrier_compat(tmp_path):
    """旧式配置（无新增字段）仍兼容：默认 auto→fixed，噪声 none。"""
    from benchmark_modulated import run
    out = tmp_path / "legacy"
    run(_tiny_config(), out)  # 缺省 include_zero + initial_rms 池
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["noise"]["model"] == "none"
    assert meta["sign_policy"] == "fixed"
    rows = list(csv.DictReader((out / "runs.csv").open())
                )
    assert all(r["noise_model"] == "none" and r["noise_repeat"] == "0"
               for r in rows)
    assert all(r["strict_success"] in ("True", "False") for r in rows)


def test_phase_shift_free_sign_preserved(tmp_path):
    """无载频（相移）旧行为：auto→free。"""
    from benchmark_modulated import ModulatedConfig
    config = ModulatedConfig(acquisition="phase_shift", n=16, n_steps=3,
                             truth_rms=[0.1], truths_per_rms=1,
                             methods=["lm"], options={"max_forward": 3})
    assert config.resolved_sign_policy() == "free"
    with pytest.raises(ValueError):
        ModulatedConfig(acquisition="carrier", n=16, sign_policy="bogus")
    with pytest.raises(ValueError):
        ModulatedConfig(acquisition="carrier", n=16,
                        report_tolerances=[-1.0])
    with pytest.raises(ValueError):
        ModulatedConfig(acquisition="carrier", n=16,
                        initial_classes=["random", "random"])
