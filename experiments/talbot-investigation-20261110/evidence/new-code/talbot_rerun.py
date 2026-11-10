"""Checkpointed, matched-physics FT and LM experiments on the new baseline."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from lsi.execution import exclusive_study
from lsi.forward import DiffractionModel
from lsi.fourier import FourierExtractor
from talbot_common import (ROOT, RESULTS, REFERENCE_COMMIT, Monitor, config,
                           digest, event, save, truth_for, utc)
from talbot_ft import OriginalSingleFT, settings_for, spectrum_diagnostics


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def metrics(model, coefficients, truth):
    delta = coefficients - truth
    wave = model.wavefront(truth)
    reconstructed = model.wavefront(coefficients)
    error = reconstructed - wave
    values = error[model.pupil]
    edges = np.linspace(model.config.epsilon, 1, 11)
    radial = []
    for number, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = model.pupil & (model.rho >= lo)
        mask &= model.rho <= hi if number == 9 else model.rho < hi
        radial.append(dict(low=float(lo), high=float(hi), pixels=int(mask.sum()),
                           rms_waves=rms(error[mask])))
    return dict(raw_rms_waves=rms(values),
                raw_rms_nm=rms(values)*model.config.wavelength*1e9,
                raw_pv_waves=float(np.ptp(values)),
                relative_rms=rms(values)/rms(wave[model.pupil]),
                truth_raw_rms_waves=rms(wave[model.pupil]),
                error_mean_waves=float(values.mean()),
                coefficient_rms_waves=rms(delta),
                coefficient_max_abs_waves=float(abs(delta).max()),
                coefficient_error_waves_36=[0., *delta.tolist()],
                coefficients_waves_36=[0., *coefficients.tolist()],
                truth_coefficients_waves_36=[0., *truth.tolist()],
                outer10_rms_waves=rms(error[model.pupil & (model.rho >= .9)]),
                evaluation_pixels=int(model.pupil.sum()), radial=radial,
                metric="sqrt(mean((W_fit-W_true)^2)) on full discrete objective pupil; no fitted removal")


def save_wavefront(path, model, coefficients, truth):
    yy, xx = np.nonzero(model.pupil)
    bounds = [int(yy.min())-3, int(yy.max())+4,
              int(xx.min())-3, int(xx.max())+4]
    crop = np.s_[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    true = model.wavefront(truth)
    fitted = model.wavefront(coefficients)
    np.savez_compressed(path, truth=true[crop], reconstruction=fitted[crop],
                        error=(fitted-true)[crop], pupil=model.pupil[crop],
                        rho=model.rho[crop], crop_bounds=bounds, original_n=model.n,
                        truth_coefficients=truth, coefficients=coefficients)


def calibrate(model, zero, extractor, folder, tag, amplitude, rcond, monitor):
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / f"{tag}_features.json"
    checkpoint = json.loads(path.read_text()) if path.exists() else {}
    feature = (lambda image: extractor.extract(image)["raw_coeffs"]
               if tag == "original" else
               extractor.extract(image, reference_image=zero)["raw_coeffs"])
    jobs = [("zero", np.zeros(35))]
    for j in range(35):
        for sign in (1, -1):
            coefficients = np.zeros(35)
            coefficients[j] = sign*amplitude
            jobs.append((f"j{j+1:02d}{'p' if sign > 0 else 'm'}", coefficients))
    for key, coefficients in jobs:
        if key in checkpoint:
            continue
        monitor.update(stage="calibration", pipeline=tag, active_key=key,
                       calibration_completed=len(checkpoint), calibration_total=71)
        started = time.monotonic()
        image = zero if key == "zero" else model.evaluate(coefficients)
        raw = feature(image)
        checkpoint[key] = dict(raw_coefficients=raw.tolist(),
                               image_sha256=digest(image),
                               seconds=time.monotonic()-started)
        save(path, checkpoint)
        durations = [row["seconds"] for k, row in checkpoint.items() if k != "zero"]
        eta = (71-len(checkpoint))*float(np.mean(durations)) if durations else None
        monitor.update(calibration_completed=len(checkpoint),
                       eta_remaining_seconds=eta,
                       eta_scope="remaining current calibration only")
        event("calibration_feature", source_kind=model.config.source_kind, pipeline=tag,
              key=key, completed=len(checkpoint), total=71,
              seconds=checkpoint[key]["seconds"])
    baseline = np.array(checkpoint["zero"]["raw_coefficients"])
    response = np.stack([
        (np.array(checkpoint[f"j{j:02d}p"]["raw_coefficients"]) -
         np.array(checkpoint[f"j{j:02d}m"]["raw_coefficients"]))/(2*amplitude)
        for j in range(1, 36)], axis=1)
    singular = np.linalg.svd(response, compute_uv=False)
    cutoff = rcond if rcond is not None else max(response.shape)*np.finfo(float).eps
    rank = int((singular > cutoff*singular[0]).sum()) if singular[0] else 0
    dZ_singular = np.linalg.svd(extractor.A, compute_uv=False)
    train_error = []
    for key, coefficients in jobs[1:]:
        raw = np.array(checkpoint[key]["raw_coefficients"])
        estimate = np.linalg.lstsq(response, raw-baseline, rcond=rcond)[0]
        train_error.append(float(np.max(abs(estimate-coefficients))))
    diagnostic = dict(
        amplitude_waves=amplitude, rcond=cutoff, effective_rank=rank,
        singular_values=singular.tolist(),
        condition_2=float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        full_rank_condition_2=float(singular[0]/singular[-1]) if rank == 35 else None,
        retained_condition_2=float(singular[0]/singular[rank-1]) if rank else None,
        dZ_rank=int((dZ_singular > dZ_singular[0]*max(extractor.A.shape)*np.finfo(float).eps).sum()),
        dZ_condition_2=float(dZ_singular[0]/dZ_singular[-1]),
        dZ_shape=list(extractor.A.shape), shear_integer_pixels=extractor.shear,
        train_max_coefficient_error_waves=max(train_error),
        settings=extractor.settings.to_dict(),
        phase_reference_subtracted=tag != "original",
        raw_regression="Lasso alpha=.02" if tag == "original" else "lstsq",
        calibration_wall_seconds=sum(row["seconds"] for row in checkpoint.values()),
    )
    save(folder / f"{tag}_calibration.json", diagnostic)
    np.savez_compressed(folder / f"{tag}_calibration.npz",
                        baseline=baseline, response=response, singular_values=singular,
                        dZ_singular_values=dZ_singular)
    # Hold-out mixtures never enter calibration, window selection, or fitting.
    validations = []
    random = np.random.default_rng(1807)
    for target_rms in (.005, .02, .05):
        coefficients = random.normal(size=35)
        coefficients *= target_rms/rms(model.wavefront(coefficients)[model.pupil])
        raw = feature(model.evaluate(coefficients))
        estimate = np.linalg.lstsq(response, raw-baseline, rcond=rcond)[0]
        validations.append(dict(target_rms_waves=target_rms,
                                **metrics(model, estimate, coefficients)))
    save(folder / f"{tag}_holdout_validation.json", validations)
    return baseline, response, feature, diagnostic


def fit_lm(model, zero, observed, folder, condition, truth, monitor):
    rows = np.sort(np.random.default_rng(42).choice(observed.size, 2048, replace=False))
    np.save(folder / "lm_sampled_pixels.npy", rows)
    scale = float(zero.max())
    target = observed.ravel()[rows]
    initial_prediction = zero.ravel()[rows]/scale
    gain0, background0 = np.linalg.lstsq(
        np.column_stack((initial_prediction, np.ones(len(rows)))), target, rcond=None)[0]
    initial = np.r_[np.zeros(35), np.log(max(gain0, 1e-8)), background0]
    cached_x, cached = None, None
    count = 0
    started = time.monotonic()
    trace = folder / f"lm_{condition}_trajectory.jsonl"
    if trace.exists():
        raise RuntimeError(f"Unfinished LM trace exists: {trace}; inspect before resuming")

    def evaluate(x):
        nonlocal cached_x, cached, count
        if cached_x is None or not np.array_equal(cached_x, x):
            image, derivative = model.evaluate(x[:35], jacobian=True, rows=rows)
            gain = np.exp(x[35])
            prediction = image/scale
            residual = gain*prediction+x[36]-target
            jacobian = np.column_stack((gain*derivative.T/scale,
                                        gain*prediction, np.ones(len(rows))))
            cached_x, cached = x.copy(), (residual, jacobian)
            count += 1
            row = dict(timestamp_utc=utc(), joint_evaluation=count,
                       elapsed_seconds=time.monotonic()-started, cost=float(residual@residual),
                       intensity_rms=rms(residual),
                       coefficients_waves_36=[0., *x[:35].tolist()],
                       gain=float(gain), background=float(x[36]))
            with trace.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            monitor.update(stage="lm", condition=condition, lm_joint_evaluations=count,
                           latest_cost=row["cost"],
                           eta_remaining_seconds=None,
                           eta_scope="LM iteration count unknown; elapsed and evaluations recorded")
            event("lm_evaluation", source_kind=model.config.source_kind,
                  condition=condition, joint_evaluation=count, cost=row["cost"],
                  elapsed_seconds=row["elapsed_seconds"])
        return cached

    chunks = []
    for budget in (80, 160):
        solution = least_squares(lambda x: evaluate(x)[0], initial,
                                 jac=lambda x: evaluate(x)[1], method="lm",
                                 x_scale="jac", max_nfev=budget,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        chunks.append(dict(max_nfev=budget, nfev=solution.nfev,
                           njev=solution.njev, success=bool(solution.success),
                           message=str(solution.message)))
        if solution.success:
            break
        initial = solution.x
    residual, jacobian = evaluate(solution.x)
    nuisance = jacobian[:, 35:]
    projected = jacobian[:, :35] - nuisance@np.linalg.lstsq(
        nuisance, jacobian[:, :35], rcond=None)[0]
    singular = np.linalg.svd(projected, compute_uv=False)
    rank = int((singular > singular[0]*1e-12).sum())
    predicted = np.exp(solution.x[35])*model.evaluate(solution.x[:35])/scale+solution.x[36]
    np.save(folder / f"lm_{condition}_predicted.npy", predicted)
    save_wavefront(folder / f"lm_{condition}_wavefront.npz", model, solution.x[:35], truth)
    result = dict(
        method="single_lm" if model.config.source_kind == "single" else "multi64_lm",
        solver="scipy.optimize.least_squares(method='lm', analytic Jacobian)",
        condition=condition, source_count=len(model.config.sources),
        success=bool(solution.success), status=int(solution.status),
        message=str(solution.message), nfev=sum(row["nfev"] for row in chunks),
        njev=sum(row["njev"] for row in chunks), chunks=chunks,
        joint_evaluations=count, sample_count=len(rows), sample_seed=42,
        initial_coefficients_waves_36=[0.]*36, piston_fixed_to_zero=True,
        free_wavefront_parameters=35, fitted_gain=float(np.exp(solution.x[35])),
        fitted_background=float(solution.x[36]), normalization_zero_peak=scale,
        full_intensity_rmse=rms(predicted-observed),
        full_intensity_nrmse=rms(predicted-observed)/rms(observed),
        sampled_intensity_rmse=rms(residual), optimality=float(solution.optimality),
        projected_wavefront_singular_values=singular.tolist(),
        projected_wavefront_rank=rank,
        projected_wavefront_condition_2=float(singular[0]/singular[-1]),
        seconds=time.monotonic()-started,
        observed_sha256=digest(observed), **metrics(model, solution.x[:35], truth))
    save(folder / f"lm_{condition}.json", result)
    return result


def worker(kind):
    output = RESULTS / kind
    output.mkdir(parents=True, exist_ok=True)
    with Monitor("setup", kind) as monitor:
        cfg = config(kind)
        model = DiffractionModel(cfg, tuple(range(1, 36)), batch_sources=1, workers=1)
        truth = truth_for(model)
        preflight = np.load(RESULTS / f"preflight_{kind}.npz")
        zero = preflight["zero"]
        raw_truth = preflight["truth"]
        assert np.allclose(model.evaluate(truth), raw_truth, rtol=1e-10, atol=0.)
        peak = float(raw_truth.max())
        clean = raw_truth/peak
        sigma = .01*rms(clean)
        noisy = clean+sigma*np.random.default_rng(42).standard_normal(clean.shape)
        np.save(output / "clean.npy", clean)
        np.save(output / "observed_40db.npy", noisy)
        np.save(output / "zero_raw.npy", zero)
        save(output / "observation.json", dict(
            system=cfg.to_dict(), reference_commit=REFERENCE_COMMIT,
            sources=[list(source) for source in cfg.sources],
            source_count=len(cfg.sources), seed=42, noise_sigma=sigma,
            noise="sigma=.01*RMS(clean/clean.max()); iid Gaussian; seed42",
            peak_raw_truth=peak, clean_sha256=digest(clean),
            observed_40db_sha256=digest(noisy),
            truth_coefficients_waves_36=[0., *truth.tolist()],
            truth_raw_rms_waves=rms(model.wavefront(truth)[model.pupil])))
        settings, geometry = settings_for(cfg)
        geometry, spectrum, profiles = spectrum_diagnostics(zero, cfg, settings, geometry)
        save(output / "sideband_geometry.json", geometry)
        np.savez_compressed(output / "sideband_diagnostics.npz",
                            spectrum_crop=spectrum[model.n//2-100:model.n//2+101,
                                                   model.n//2-100:model.n//2+101],
                            x_profile=profiles[0], y_profile=profiles[1],
                            settings=json.dumps(settings.to_dict()))
        del spectrum, profiles
        # Keep the two historical FT pipelines explicit. A matched single-source
        # OLS pipeline is added to separate source-count and regression effects.
        pipelines = (("original", OriginalSingleFT, .005, 1e-10),
                     ("matched", FourierExtractor, .02, None)) if kind == "single" else (
                         ("matched", FourierExtractor, .02, None),)
        for tag, cls, amplitude, rcond in pipelines:
            if (output / f"ft_{tag}_40db.json").exists():
                continue
            extractor = cls(cfg, settings)
            baseline, response, feature, diagnostic = calibrate(
                model, zero, extractor, output, tag, amplitude, rcond, monitor)
            for condition, observation in (("clean", clean), ("40db", noisy)):
                started = time.monotonic()
                raw_feature = feature(observation)
                estimate = np.linalg.lstsq(response, raw_feature-baseline, rcond=rcond)[0]
                method = ("single_ft" if tag == "original" else "single_ft_matched") if kind == "single" else "multi64_ft"
                predicted = model.evaluate(estimate)/peak
                np.save(output / f"ft_{tag}_{condition}_predicted.npy", predicted)
                save_wavefront(output / f"ft_{tag}_{condition}_wavefront.npz",
                               model, estimate, truth)
                result = dict(method=method, condition=condition, pipeline=tag,
                              source_count=len(cfg.sources),
                              calibration_rank=diagnostic["effective_rank"],
                              calibration_condition_2=diagnostic["full_rank_condition_2"],
                              raw_feature_coefficients=raw_feature.tolist(),
                              observed_sha256=digest(observation),
                              full_intensity_rmse=rms(predicted-observation),
                              full_intensity_nrmse=rms(predicted-observation)/rms(observation),
                              intensity_prediction="fixed clean scale; no fitted exposure or background",
                              seconds=time.monotonic()-started,
                              **metrics(model, estimate, truth))
                save(output / f"ft_{tag}_{condition}.json", result)
                event("result_complete", method=method, condition=condition,
                      raw_rms_waves=result["raw_rms_waves"])
            del extractor, feature, baseline, response
            gc.collect()
        for condition, observation in (("clean", clean), ("40db", noisy)):
            if (output / f"lm_{condition}.json").exists():
                continue
            result = fit_lm(model, zero, observation, output, condition, truth, monitor)
            event("result_complete", method=result["method"], condition=condition,
                  raw_rms_waves=result["raw_rms_waves"], success=result["success"])
        save(output / "completed.json", dict(status="complete", timestamp_utc=utc(),
             peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("single", "multi64"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (20*1024**3, 20*1024**3))
    preflight = json.loads((RESULTS / "preflight.json").read_text())
    if not preflight.get("passed"):
        raise RuntimeError("Full-grid preflight has not passed")
    order_check = json.loads((RESULTS / "diffraction_order_validation.json").read_text())
    if not order_check.get("passed"):
        raise RuntimeError("Isolated diffraction-order check has not passed")
    with exclusive_study(), threadpool_limits(limits=args.threads):
        worker(args.kind)


if __name__ == "__main__":
    main()
