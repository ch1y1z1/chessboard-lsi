"""Least-squares fitting to the complete reference diffraction propagation."""
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares


def fit_direct(model, observed, starts=2, seed=42, samples=8192, max_nfev=60,
               initial_rms=0.05, coefficient_bound=2.0, progress=False, screen_initials=0):
    """Independent starts; gain/background fitted jointly, no truth initialization.

    Predictions share the *zero-wavefront* scale throughout optimization. An
    observed image may have any positive exposure scale and constant offset.
    """
    observed = np.asarray(observed, float)
    start = perf_counter()
    forward_before, jacobian_before = model.n_forward, model.n_jacobian
    if observed.shape != (model.n, model.n) or not np.isfinite(observed).all():
        raise ValueError("invalid observed image")
    if starts < 1 or max_nfev < 1 or screen_initials < 0 or samples is not None and samples < 16:
        raise ValueError("invalid solver budget")
    count = len(model.indices)
    total = observed.size
    rng = np.random.default_rng(seed)
    rows = (np.arange(total) if samples is None or samples >= total else
            np.sort(rng.choice(total, size=samples, replace=False)))
    image0 = model.evaluate(np.zeros(count))
    scale = float(image0.max())
    if scale <= 0 or observed.max() <= observed.min():
        raise ValueError("no intensity contrast")
    y = observed.ravel()[rows]
    exposure = max(float(observed.max() - observed.min()), np.finfo(float).tiny)
    y = y / exposure
    pred0 = image0.ravel()[rows] / scale
    gain0, background0 = np.linalg.lstsq(np.column_stack((pred0, np.ones(len(rows)))), y, rcond=None)[0]
    gain0 = max(gain0, 1e-4)
    initials = [np.zeros(count)]
    screened = []
    for i in range(max(starts-1, screen_initials)):
        direction = rng.normal(size=count)
        rms = np.std(model.wavefront(direction)[model.pupil] - model.fixed_wavefront[model.pupil])
        amplitude = (.03, .1, .3)[i % 3] if screen_initials else initial_rms
        candidate = direction * amplitude / max(rms, 1e-12)
        candidate = np.clip(candidate, -coefficient_bound*.99, coefficient_bound*.99)
        if screen_initials:
            pred = model.evaluate(candidate, rows=rows) / scale
            X = np.column_stack((pred, np.ones(len(rows))))
            ab = np.linalg.lstsq(X, y, rcond=None)[0]
            if ab[0] <= 0:
                ab = np.array([0.0, y.mean()])
            residual = X @ ab - y
            screened.append({'coeffs': candidate.tolist(), 'cost': float(residual @ residual),
                             'rms_waves': amplitude})
            if progress and (i+1) % 8 == 0:
                print(f'screened {i+1}/{max(starts-1, screen_initials)} initial candidates', flush=True)
        else:
            initials.append(candidate)
    if screen_initials:
        screened.sort(key=lambda row: row['cost'])
        initials.extend(np.array(row['coeffs']) for row in screened[:starts-1])
    trials = []
    for trial, initial in enumerate(initials):
        if progress:
            print(f'direct trial {trial+1}/{starts}', flush=True)
        cached_x, cached = None, None
        history = []

        def evaluate(x):
            nonlocal cached_x, cached
            if cached_x is None or not np.array_equal(x, cached_x):
                image, dI = model.evaluate(x[:count], jacobian=True, rows=rows)
                a = float(np.exp(x[count]))
                pred = image / scale
                residual = a * pred + x[count + 1] - y
                J = np.column_stack((a * dI.T / scale,
                                     a * pred, np.ones(len(rows))))
                cached_x, cached = x.copy(), (residual, J)
                history.append({"cost": float(residual @ residual), "coeffs": x[:count].tolist()})
                if progress and len(history) % 5 == 0:
                    print(f'  evaluation {len(history)} cost={residual @ residual:.6g}', flush=True)
            return cached

        x0 = np.r_[np.clip(initial, -coefficient_bound * .99, coefficient_bound * .99),
                    np.log(gain0), background0]
        sol = least_squares(lambda x: evaluate(x)[0], x0, jac=lambda x: evaluate(x)[1],
                            bounds=(np.r_[np.full(count, -coefficient_bound), -20, -np.inf],
                                    np.r_[np.full(count, coefficient_bound), 20, np.inf]),
                            method="trf", x_scale="jac", max_nfev=max_nfev,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        image = model.evaluate(sol.x[:count])
        prediction = exposure * (np.exp(sol.x[count]) * image / scale + sol.x[count+1])
        full_rms = float(np.sqrt(np.mean((prediction - observed)**2)))
        singular = np.linalg.svd(sol.jac, compute_uv=False)
        trials.append({"coeffs": sol.x[:count].tolist(), "initial": initial.tolist(),
                       "gain": float(exposure * np.exp(sol.x[count]) / scale),
                       "background": float(exposure * sol.x[count+1]),
                       "converged": bool(sol.success), "status": int(sol.status),
                       "message": sol.message, "nfev": sol.nfev, "njev": sol.njev,
                       "cost": float(2 * sol.cost), "optimality": float(sol.optimality),
                       "full_intensity_rms": full_rms,
                       "jacobian_singular_values": singular.tolist(), "history": history})
    best = min(range(len(trials)), key=lambda j: trials[j]["full_intensity_rms"])
    return {"coeffs": np.array(trials[best]["coeffs"]), "trials": trials,
            "best_trial": best, "seconds": perf_counter() - start,
            "sample_rows": rows, "normalization_zero_peak": scale, "indices": model.indices,
            "screened_initials": screened,
            "n_forward_fit": model.n_forward-forward_before,
            "n_jacobian_fit": model.n_jacobian-jacobian_before,
            "n_forward_total": model.n_forward, "n_jacobian_total": model.n_jacobian}
