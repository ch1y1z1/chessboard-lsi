from dataclasses import replace
import numpy as np

from lsi.config import SystemConfig
from lsi.forward import DiffractionModel


def test_fast_operator_and_sampled_derivative():
    system = SystemConfig.profile('test').with_angle('grating', .37, 'diagonal').with_angle('camera', -.2, 'y')
    system = replace(system, shift_x_grating=2 * system.dx_grating,
                     shift_y_grating=-system.dx_grating, shift_z_grating=1e-6)
    fast = DiffractionModel(system)
    oracle = DiffractionModel(system, backend='czt')
    coefficients = np.array([.04, -.025, .03, .035])
    expected, jacobian = oracle.evaluate(coefficients, jacobian=True)
    actual, derivative = fast.evaluate(coefficients, jacobian=True)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-10
    assert np.linalg.norm(derivative - jacobian) / np.linalg.norm(jacobian) < 1e-10
    rows = np.random.default_rng(2).choice(system.n**2, 500, replace=False)
    sampled, dsampled = fast.evaluate(coefficients, jacobian=True, rows=rows)
    np.testing.assert_allclose(sampled, expected.ravel()[rows], rtol=1e-9, atol=1e-24)
    np.testing.assert_allclose(dsampled, jacobian.reshape(4, -1)[:, rows], rtol=1e-8, atol=1e-23)
    for j in range(4):
        step = np.zeros(4); step[j] = 1e-6
        numerical = (fast.evaluate(coefficients + step) - fast.evaluate(coefficients - step)) / 2e-6
        assert np.linalg.norm(numerical - derivative[j]) / np.linalg.norm(numerical) < 1e-7
