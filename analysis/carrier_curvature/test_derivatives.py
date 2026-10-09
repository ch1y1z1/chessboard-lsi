"""T5 自有导数验证：解析 J/H 对有限差分与既有 evaluate 路径的校验。

运行：python -m pytest analysis/carrier_curvature/test_derivatives.py
   或 python analysis/carrier_curvature/test_derivatives.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.carrier_curvature import common
from analysis.carrier_curvature.hessian import fd_hessian_check, problem_curvature


def _problem(case_id=0, scaling="rms"):
    fm = common.make_forward()
    dataset = common.load_dataset()
    return common.carrier_problem(fm, dataset, case_id, scaling=scaling)


def test_jacobian_against_fd():
    """problem.evaluate 的解析 J 对强度差分（间接验证 H 的 J^TJ 项）。"""
    problem = _problem()
    rng = np.random.default_rng(0)
    q = rng.normal(size=len(common.INDICES)) * 0.05
    _, J = problem.evaluate(q)
    c = problem.to_coefficients(q)
    h = 1e-6
    for j in rng.choice(len(q), 4, replace=False):
        dc = np.zeros_like(c)
        dc[j] = h
        from lsi.lm import _frame_and_jacobian
        ip, _ = _frame_and_jacobian(problem.cache, c + dc,
                                    problem.mods[0], compute_jacobian=False)
        im, _ = _frame_and_jacobian(problem.cache, c - dc,
                                    problem.mods[0], compute_jacobian=False)
        fd = (ip - im) / (2 * h * problem.normalizer * problem.scale[j])
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-4, atol=1e-6)


def test_hessian_directional_fd():
    """精确 Hessian 的方向有限差分校验（含残差二阶项）。"""
    problem = _problem()
    rng = np.random.default_rng(1)
    q = rng.normal(size=len(common.INDICES)) * 0.05
    dirs = [rng.normal(size=len(q)) for _ in range(6)]
    errors = fd_hessian_check(problem, q, dirs, h=1e-5)
    assert max(errors) < 1e-5, f"FD 校验失败: {errors}"


def test_hessian_at_matched_points():
    """在收敛解/非驻点/大残差点三处校验；problem_curvature 内部
    已对 evaluate 做 allclose 断言。"""
    problem = _problem()
    dataset = common.load_dataset()
    truth_q = problem.to_parameters(dataset["truths"][0])
    rng = np.random.default_rng(2)
    for q in (truth_q, np.zeros(len(truth_q)),
              rng.normal(size=len(truth_q)) * 0.3):
        errors = fd_hessian_check(problem, q, [rng.normal(size=len(q))],
                                  h=1e-5)
        assert errors[0] < 1e-5


def test_scaling_none_consistent():
    """scaling='none' 下 q=c，H 应与 rms 口径链式变换一致。"""
    p_rms = _problem(scaling="rms")
    p_none = _problem(scaling="none")
    rng = np.random.default_rng(3)
    c = rng.normal(size=len(common.INDICES)) * 0.05
    q = c * p_rms.scale
    cur_rms = problem_curvature(p_rms, q)
    cur_none = problem_curvature(p_none, c)
    s = np.outer(p_rms.scale, p_rms.scale)
    np.testing.assert_allclose(cur_rms["hess"] * s, cur_none["hess"],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cur_rms["g"] * p_rms.scale, cur_none["g"],
                               rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    test_jacobian_against_fd()
    test_hessian_directional_fd()
    test_hessian_at_matched_points()
    test_scaling_none_consistent()
    print("all derivative checks passed")
