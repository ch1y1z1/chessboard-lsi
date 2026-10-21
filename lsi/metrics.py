"""Common, fixed-pupil evaluation; never accept a sign flip as recovery."""
import numpy as np
from .annular import basis


def wavefront_metrics(model, fitted, truth, recovery_tolerance=0.01):
    fitted, truth = np.asarray(fitted), np.asarray(truth)
    if fitted.shape != truth.shape or fitted.shape != (len(model.indices),):
        raise ValueError("coefficient shapes do not match comparison basis")
    difference = model.wavefront(fitted) - model.wavefront(truth)
    values = difference[model.pupil]
    full = values - values.mean()
    low = basis((0, 1, 2, 3), model.rho, model.theta, model.config.epsilon, model.config.Nz)
    A = low[:, model.pupil].T
    adjusted = values - A @ np.linalg.lstsq(A, values, rcond=None)[0]
    rmse = float(np.sqrt(np.mean(adjusted**2)))
    return {"rmse_waves": rmse,
            "rmse_waves_with_defocus_tilt": float(np.sqrt(np.mean(full**2))),
            "pv_waves": float(np.ptp(adjusted)),
            "max_coefficient_error_waves": float(np.max(abs(fitted - truth))),
            "coefficient_errors_waves": (fitted - truth).tolist(),
            "recovered": bool(rmse < recovery_tolerance),
            "recovery_tolerance_waves": recovery_tolerance,
            "evaluation_pixels": int(model.pupil.sum())}
