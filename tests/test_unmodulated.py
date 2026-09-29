"""共享物理、导数、预算、等价类及基准输出的风险回归。"""
import json

import numpy as np
import pytest

from lsi.metrics import recovery_metrics
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront
from lsi.optimizers import METHODS, SolverOptions, _Evaluation, solve
from lsi.problem import IntensityProblem, StackedIntensityProblem


@pytest.fixture
def scene():
    fm = ForwardModel(SystemConfig(grid=Grid(n=24)))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    image = fm.intensity(zernike_wavefront(truth, indices))
    return IntensityProblem(fm, indices, image), truth


@pytest.fixture
def phase_shift_scene():
    fm = ForwardModel(SystemConfig(grid=Grid(n=24)))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    wf = zernike_wavefront(truth, indices)
    frames, mods = [], []
    for direction in ("x", "y"):
        frames.extend(fm.phase_shift_frames(wf, direction, 4))
        mods.extend(
            fm.phase_shift_deltas(i / 4, 0.0) if direction == "x"
            else fm.phase_shift_deltas(0.0, i / 4)
            for i in range(4)
        )
    return StackedIntensityProblem(fm, indices, frames, mods), truth


def test_unmodulated_derivative_scaling_and_physics(scene):
    problem, truth = scene
    q = problem.to_parameters(truth * .8)
    r, j = problem.evaluate(q)
    predicted = problem.forward.intensity(zernike_wavefront(truth*.8, problem.indices))
    np.testing.assert_allclose(r, (predicted.ravel()-problem.observed)/problem.normalizer, atol=1e-15)
    for k in range(q.size):
        perturb = np.eye(q.size)[k]*1e-7
        finite_difference = (problem.evaluate(q+perturb, jacobian=False)[0]
                             - problem.evaluate(q-perturb, jacobian=False)[0]) / 2e-7
        np.testing.assert_allclose(j[:, k], finite_difference, atol=2e-9, rtol=1e-5)
    original = IntensityProblem(problem.forward, problem.indices,
                                problem.observed.reshape(problem.forward.shape), scaling="none")
    np.testing.assert_allclose(original.evaluate(truth*.8)[0], r, atol=1e-15)
    np.testing.assert_allclose(original.evaluate(truth*.8)[1], j*problem.scale, atol=1e-14)


def test_sign_zero_and_tilt_alias(scene):
    problem, truth = scene
    np.testing.assert_allclose(problem.evaluate(problem.to_parameters(-truth))[0], 0, atol=1e-15)
    _, zero_j = problem.evaluate(np.zeros_like(truth))
    np.testing.assert_array_equal(zero_j, 0)
    alias = truth.copy()
    alias[0] += 1 / problem.forward.s
    np.testing.assert_allclose(problem.evaluate(problem.to_parameters(alias))[0], 0, atol=1e-14)
    m = recovery_metrics(alias, truth, problem.indices, problem.pupil_basis, problem.forward.s)
    assert m["tilt_alias"]
    assert m["equivalent_max_coeff_error"] < 1e-13
    assert m["max_coeff_error"] > 1


@pytest.mark.parametrize("method", METHODS)
def test_zero_stop_is_not_recovery(scene, method):
    problem, truth = scene
    result = solve(problem, np.zeros_like(truth), method)
    assert result.stopped_by_tolerance
    assert result.loss > 1e-5
    np.testing.assert_array_equal(result.coeffs, 0)


@pytest.mark.parametrize("method", METHODS)
def test_budget_and_best_observed_point(scene, method):
    problem, truth = scene
    c0 = truth * .95
    result = solve(problem, c0, method, SolverOptions(max_forward=12, max_jacobian=12))
    assert result.n_forward <= 12 and result.n_jacobian <= 12
    assert result.n_jacobian <= result.n_forward
    assert result.loss <= result.history[0]["loss"]
    assert result.loss == min(row["loss"] for row in result.history)
    r, _ = problem.evaluate(problem.to_parameters(result.coeffs))
    assert result.loss == pytest.approx(r@r/2, abs=1e-15)
    assert len(result.history) == result.n_forward


@pytest.mark.parametrize("method", ["gn", "lm", "bfgs", "trf"])
def test_local_recovery(scene, method):
    problem, truth = scene
    result = solve(problem, truth*.98, method,
                   SolverOptions(max_forward=300, max_jacobian=300, gtol=1e-12))
    np.testing.assert_allclose(result.coeffs, truth, atol=1e-6)
    assert result.rms_residual < 1e-8


def test_cache_and_limits(scene):
    problem, truth = scene
    q = problem.to_parameters(truth*.9)
    e = _Evaluation(problem, q, SolverOptions(max_forward=1))
    e.value_gradient(q)
    e.value_gradient(q.copy())
    e.residual(q)
    e.jacobian(q)
    assert (e.n_forward, e.n_jacobian, e.n_gradient) == (1, 1, 1)
    result = solve(problem, truth*.9, "lm", SolverOptions(max_forward=1))
    assert result.status == "forward_budget"
    assert np.isfinite(result.loss)
    result = solve(problem, truth*.9, "adam", SolverOptions(max_seconds=1e-12))
    assert result.status == "time_budget" and result.n_forward == 1
    result = solve(problem, truth*.9, "lm", SolverOptions(max_jacobian=1))
    assert result.status == "jacobian_budget" and result.n_jacobian == 1


@pytest.mark.parametrize("method", ["momentum", "adam"])
def test_first_order_step_uses_shared_scaled_gradient(scene, method):
    problem, truth = scene
    c0 = truth*.95
    q0 = problem.to_parameters(c0)
    r, j = problem.evaluate(q0)
    gradient = j.T @ r
    options = SolverOptions(max_iter=1, learning_rate=1e-6)
    expected_step = -options.learning_rate * gradient
    if method == "adam":
        expected_step /= np.abs(gradient) + options.epsilon
    result = solve(problem, c0, method, options)
    np.testing.assert_allclose(problem.to_parameters(result.coeffs), q0+expected_step,
                               rtol=1e-12, atol=1e-15)
    assert result.loss < result.history[0]["loss"]


def test_phase_shift_stack_derivatives_and_identifiability(phase_shift_scene):
    problem, truth = phase_shift_scene
    r, j = problem.evaluate(problem.to_parameters(truth))
    np.testing.assert_allclose(r, 0, atol=1e-15)
    assert r.size == 8 * problem.rows.size
    q = problem.to_parameters(truth * .8)
    r, j = problem.evaluate(q)
    for k in range(q.size):
        perturb = np.eye(q.size)[k] * 1e-7
        fd = (problem.evaluate(q + perturb, jacobian=False)[0]
              - problem.evaluate(q - perturb, jacobian=False)[0]) / 2e-7
        np.testing.assert_allclose(j[:, k], fd, atol=2e-9, rtol=1e-5)
    # 已知相移打破无调制下的两个退化：整体符号与 J(0)=0。
    r_minus, _ = problem.evaluate(problem.to_parameters(-truth))
    assert r_minus @ r_minus > 1e-6
    _, zero_j = problem.evaluate(np.zeros_like(truth))
    assert np.abs(zero_j).max() > 1e-3
    # 相移堆叠下 LM 能从零初值直接恢复真值。
    result = solve(problem, np.zeros_like(truth), "lm",
                   SolverOptions(max_forward=300, max_jacobian=300, gtol=1e-12))
    np.testing.assert_allclose(result.coeffs, truth, atol=1e-6)


def test_carrier_stack_modulation_shape():
    fm = ForwardModel(SystemConfig(grid=Grid(n=24), period_um=30.0))
    indices = [2, 3, 4, 5, 6, 7, 8]
    truth = np.array([.02, -.01, .31, -.12, .07, .42, .05])
    wf = zernike_wavefront(truth, indices)
    mods = [fm.carrier_phases(fm.config.carrier_f0)]
    problem = StackedIntensityProblem(fm, indices, [fm.carrier_frame(wf)], mods)
    r, _ = problem.evaluate(problem.to_parameters(truth))
    np.testing.assert_allclose(r, 0, atol=1e-15)
    # 载频同样打破整体符号等价。
    r_minus, _ = problem.evaluate(problem.to_parameters(-truth))
    assert r_minus @ r_minus > 1e-6
    with pytest.raises(ValueError):
        StackedIntensityProblem(fm, indices, [fm.carrier_frame(wf)],
                                [np.zeros(3)])
    with pytest.raises(ValueError):
        StackedIntensityProblem(fm, indices, [fm.carrier_frame(wf)], [])
    with pytest.raises(ValueError):
        StackedIntensityProblem(fm, indices, [], [])


def test_modulated_benchmark_parallel_matches_serial(tmp_path):
    from benchmark_modulated import ModulatedConfig, run
    common = dict(acquisition="phase_shift", n=16, n_steps=3,
                  truth_kind="dense", truth_rms=[.1], truths_per_rms=1,
                  directions=1, initial_rms=[.1], include_zero=True,
                  options={"max_forward": 30, "max_jacobian": 30})
    serial = run(ModulatedConfig(**common, workers=1), tmp_path / "s")
    parallel = run(ModulatedConfig(**common, workers=2), tmp_path / "p")
    assert len(serial) == len(parallel) == len(METHODS)
    for s, p in zip(serial, parallel):
        assert (s["method"], s["runs"], s["strict_successes"]) == \
               (p["method"], p["runs"], p["strict_successes"])
    meta = json.loads((tmp_path / "p" / "metadata.json").read_text())
    assert meta["state"] == "complete" and meta["parallel_workers"] == 2
    assert meta["timing_contaminated"] is True


def test_multistart_selects_observed_loss_not_truth():
    from benchmark_optimizers import summarize
    common = dict(method="lm", case_id=0, initial_rms=.1, n_forward=10,
                  elapsed_seconds=.1, strict_success=False, engineering_success=False)
    misleading = common | dict(loss=.01, wavefront_rms_error=.5)
    accurate = common | dict(loss=.02, wavefront_rms_error=.001,
                             engineering_success=True)
    _, selected = summarize([accurate, misleading], ["lm"])
    assert selected[0]["wavefront_rms_error"] == .5
    assert selected[0]["total_forward"] == 20


def test_bad_inputs(scene):
    problem, _ = scene
    for indices in ([1, 4], [4, 4], [], [2.2]):
        with pytest.raises(ValueError):
            IntensityProblem(problem.forward, indices, np.zeros(problem.forward.shape))
    with pytest.raises(ValueError):
        solve(problem, np.zeros(1))
    with pytest.raises(ValueError):
        SolverOptions(max_forward=0)
    with pytest.raises(ValueError):
        IntensityProblem(problem.forward, [4], np.zeros((2, 24, 24)))


def test_benchmark_reproducible_noise_and_outputs(tmp_path):
    from benchmark_optimizers import BenchmarkConfig, make_dataset, run
    config = BenchmarkConfig(n=16, directions=1, initial_rms=[.1],
                             noise_snr_db=[None, 30.], options={"max_forward": 3})
    first, second = make_dataset(config), make_dataset(config)
    for i in (1, 2, 4):
        np.testing.assert_array_equal(first[i], second[i])
    output = tmp_path / "benchmark"
    summary = run(config, output)
    assert len(summary) == len(METHODS)
    assert all(row["runs"] == 2 and row["strict_trials"] == 1 for row in summary)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["config"]["noise_snr_db"] == [None, 30.]
    assert metadata["state"] == "complete"
    assert (output / "source" / "lsi" / "optimizers.py").is_file()
    for name in ("dataset.npz", "runs.csv", "report.md", "budget_curves.csv",
                 "trajectories.jsonl.gz", "comparison.png", "multistart.csv"):
        assert (output / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        run(config, output)
