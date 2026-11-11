"""FT region-selection study on the 0.670 um first-Talbot baseline (NA_out=0.020).

Reuses the run-08 forward model, cached observations and calibration protocol
from experiments/na-talbot-balanced-20261111. Only the FT spectral window,
center, shape, spatial mask, smoothing and center-estimation choices vary.
Every window change re-calibrates from cached calibration images; the main
line is recalibrate-per-region. Observations (clean + 40 dB seeds 41..45 +
30 dB seed 42) are reused verbatim from run-08/na0.020_{single,multi64}.
"""
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from scipy import fft, ndimage
from skimage.restoration import unwrap_phase
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "na-talbot-balanced-20261111"))
from lsi.basis import basis
from lsi.fourier import spectral_window
from study import (BalancedConfig, ExtendedCameraModel, INDICES, PROTOCOL,
                   Monitor, config, digest, geometry, metrics, rms, truth_for,
                   utc, save, WRITE_LOCK)

HERE = Path(__file__).resolve().parent
NA = 0.020
KINDS = ("single", "multi64")
NOISE_40 = PROTOCOL["noise_40db_seeds"]
NOISE_30 = PROTOCOL["noise_30db_seeds"]
HOLDOUT_SEEDS = (1807, 1808, 1809)
NEW_TRUTH_SEEDS = tuple(range(2001, 2011))  # 10 independent random truths
CALIB_AMP = PROTOCOL["calibration_amplitude_waves"]
BASE_PHASE_SIGMA = 10 * 10e-6 / PROTOCOL["base_camera_dx_m"]


@dataclass(frozen=True)
class RegionSettings:
    """All selectable FT-region parameters; run-08 baseline is the default."""
    half_width: float = 0.0          # in bins; 0 => int(0.4*fundamental)
    offset_x: float = 0.0            # window center offset in bins (x/column)
    offset_y: float = 0.0            # window center offset in bins (y/row)
    shape: str = "rect"
    window_sigma: float = 2.0
    phase_sigma: float = BASE_PHASE_SIGMA
    shear_px: int = 0                # 0 => round(shear_pixels)
    erosion: int = 0
    amplitude_threshold: float = 0.0
    mask_shift_x: int = 0            # spatial pupil-mask integer shift
    mask_shift_y: int = 0
    aspect: float = 1.0
    subpixel: bool = False           # fractional center via complex shift
    center_mode: str = "geometric"   # geometric | peak | subpixel
    search_radius: int = 8
    carrier_override: int = 0        # 0 => round(fundamental_bins); E5c uses 2x
    regression: str = "lstsq"
    alpha: float = 0.0
    sideband_label: str = "fundamental"

    def resolved(self, fundamental, shear_pixels):
        return self.__class__(**{**asdict(self),
            "half_width": self.half_width or int(0.4 * fundamental),
            "shear_px": self.shear_px or int(round(shear_pixels)),
            "carrier_override": self.carrier_override or int(round(fundamental))})


class RegionFeatures:
    """Generalized Q1 extractor; reproduces study.Q1Features at the baseline."""
    def __init__(self, c, zero, settings, compact=True):
        self.c = c
        self.settings = s = settings
        n = c.camera_n
        g = geometry(c)
        self.geometry = g
        fund = g["fundamental_bins"]
        shp = g["shear_pixels"]
        rs = s.resolved(fund, shp)
        self.carrier = rs.carrier_override
        self.shear = rs.shear_px
        self.half_width = rs.half_width
        # Compact ROI mirroring Q1Features.
        radius = int(np.ceil(g["camera_pupil_radius_m"] / c.dx_cam)) + abs(self.shear)
        radius += int(np.ceil(4 * rs.phase_sigma)) + 2
        start, stop = max(0, n // 2 - radius), min(n, n // 2 + radius + 1)
        self.roi = (slice(start, stop), slice(start, stop)) if compact else (slice(None), slice(None))
        x = (np.arange(n) - n // 2) * c.dx_cam
        x = x[self.roi[0]]
        xx, yy = np.meshgrid(x, x)
        rho = np.hypot(xx, yy) / ((g["paraxial_radius_m"] + c.z_cam) * c.NA_out * c.f1 / c.f2)
        theta = np.arctan2(yy, xx)
        pupil = (rho >= c.epsilon) & (rho <= 1)
        if rs.mask_shift_x or rs.mask_shift_y:
            pupil = np.roll(pupil, (rs.mask_shift_y, rs.mask_shift_x), axis=(0, 1))
        self.masks = []
        for axis in (1, 0):
            m = pupil & np.roll(pupil, self.shear, axis)
            if rs.erosion:
                m = ndimage.binary_erosion(m, iterations=rs.erosion)
            if not m.any():
                raise ValueError("spatial mask is empty for this region")
            self.masks.append(m)
        rows = [[], []]
        for j in INDICES:
            z = basis((j,), rho, theta, c.epsilon, c.Nz)[0]
            for k, axis in enumerate((1, 0)):
                rows[k].append((np.roll(z, self.shear, axis) - z)[self.masks[k]])
        A = np.vstack([np.stack(r, axis=1) for r in rows])
        self.A_pinv = np.linalg.pinv(A)
        self.basis_rank = int(np.linalg.matrix_rank(A))
        self.basis_condition = float(np.linalg.cond(A))
        del rows, A
        # Build spectral windows at the per-axis centers (with offsets).
        self.windows = []
        self.centers = []
        for k, axis in enumerate((1, 0)):
            off = rs.offset_x if axis == 1 else rs.offset_y
            cx = n // 2 + (self.carrier if axis == 1 else 0) + off
            cy = n // 2 + (self.carrier if axis == 0 else 0)
            center = (cx, cy)
            self.centers.append(center)
            win = spectral_window(n, center, rs)
            self.windows.append(win)
        self.rs = rs
        self.zero_spec = self.spectrum(zero)
        self.references = []
        for k, axis in enumerate((1, 0)):
            self.references.append(fft.ifft2(fft.ifftshift(
                np.roll(self.zero_spec * self.windows[k], -self.carrier, axis)))[self.roi].copy())
        # Precompute diagnostics for the chosen windows.
        self.window_metrics = []
        for k, win in enumerate(self.windows):
            w = float(win.sum())
            self.window_metrics.append(dict(weight_sum=w,
                equivalent_noise_bins=float(w * w / max(float(np.sum(win * win)), 1e-300)),
                dc_weight=float(win[n // 2, n // 2]),
                center=self.centers[k]))

    @staticmethod
    def spectrum(image):
        image = image - image.min()
        return fft.fftshift(fft.fft2(image / image.max()))

    def _locate_center(self, spec, axis):
        # The window offset applies only along its own axis.
        n = self.c.camera_n
        s = self.rs
        if axis == 1:
            cx = n // 2 + self.carrier + s.offset_x
            cy = n // 2
        else:
            cx = n // 2
            cy = n // 2 + self.carrier + s.offset_y
        if s.center_mode == "geometric":
            return cx, cy
        r = s.search_radius
        x0, x1 = max(1, int(cx) - r), min(n - 1, int(cx) + r + 1)
        y0, y1 = max(1, int(cy) - r), min(n - 1, int(cy) + r + 1)
        iy, ix = np.unravel_index(np.argmax(abs(spec[y0:y1, x0:x1])), (y1 - y0, x1 - x0))
        cx, cy = x0 + ix, y0 + iy
        if s.center_mode == "subpixel" and abs(spec[cy, cx]) > 0:
            g = abs(spec)
            denom = (g[cy - 1, cx] + g[cy + 1, cx] - 2 * g[cy, cx])
            dy = 0.0 if denom == 0 else 0.5 * (g[cy - 1, cx] - g[cy + 1, cx]) / denom
            denom = (g[cy, cx - 1] + g[cy, cx + 1] - 2 * g[cy, cx])
            dx = 0.0 if denom == 0 else 0.5 * (g[cy, cx - 1] - g[cy, cx + 1]) / denom
            cx += dx
            cy += dy
        return cx, cy

    def extract_from_spectrum(self, spec):
        """Core feature extraction given a precomputed fftshift spectrum."""
        s = self.rs
        values, maxima, branch = [], [], []
        for k, axis in enumerate((1, 0)):
            cx, cy = self._locate_center(spec, axis)
            ix = int(round(cx - self.centers[k][0]))
            iy = int(round(cy - self.centers[k][1]))
            rolled = np.roll(spec * self.windows[k], -self.carrier - ix, axis)
            if iy:
                rolled = np.roll(rolled, -iy, axis=(1 - axis))
            signal_field = fft.ifft2(fft.ifftshift(rolled))[self.roi].copy()
            if s.subpixel:
                n = self.c.camera_n
                yy, xx = np.indices(rolled.shape)
                sx = (cx - self.centers[k][0]) - ix
                sy = (cy - self.centers[k][1]) - iy
                signal_field *= np.exp(-2j * np.pi * (sx * xx + sy * yy) / n)
            phase = np.where(self.masks[k], np.angle(
                signal_field * np.conj(self.references[k])), 0.0)
            maxima.append(float(abs(phase[self.masks[k]]).max()))
            branch.append(float(np.mean(abs(phase[self.masks[k]]) > 2.8)))
            if s.phase_sigma:
                phase = ndimage.gaussian_filter(phase, s.phase_sigma)
            good = np.ones(self.masks[k].sum(), bool)
            if s.amplitude_threshold:
                amp = abs(signal_field)[self.masks[k]]
                good = amp >= s.amplitude_threshold * amp.max()
            values.append(phase[self.masks[k]] if good.all() else phase[self.masks[k]][good])
        return self.A_pinv @ np.concatenate(values), dict(
            max_absolute_phase_rad=maxima, near_branch_fraction=branch,
            small_phase_valid=bool(max(maxima) < 2.8),
            window_metrics=self.window_metrics, basis_rank=self.basis_rank,
            basis_condition=self.basis_condition)

    def __call__(self, image):
        return self.extract_from_spectrum(self.spectrum(image))


def load_observations(run08, kind):
    """Reuse run-08 clean + noise observations verbatim; return (clean, {name: obs})."""
    folder = run08 / f"na{NA:.3f}_{kind}"
    clean = np.load(folder / "clean.npy")
    obs = {"clean": clean}
    for seed in NOISE_40:
        obs[f"40db_seed{seed}"] = np.load(folder / f"observation_40db_seed{seed}.npy")
    obs["30db_seed42"] = np.load(folder / "observation_30db_seed42.npy")
    return clean, obs


def prep(model, kind, cache, run08, monitor):
    """Generate and cache calibration image spectra + truth images for one kind."""
    folder = cache / f"na{NA:.3f}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / "config.json", model.config.to_dict())
    save(folder / "geometry.json", geometry(model.config))
    # Zero and main truth (truth_for(model) reproduces run-08 PROTOCOL truth).
    zero = np.load(run08 / f"na{NA:.3f}_{kind}" / "zero.npy")
    spectra_path = folder / "calib_spectra.npy"
    spec = np.memmap(spectra_path, dtype=complex128, mode="w+",
                     shape=(71, model.n, model.n)) if not spectra_path.exists() else None
    calib_meta = []
    if spec is not None:
        # index 0 = zero; 1..70 = +/-0.02 waves single-mode calibrations.
        monitor.update(stage="prep_spectra", kind=kind, image=0, total=71)
        spec[0] = fft.fftshift(fft.fft2((zero - zero.min()) / zero.max()))
        calib_meta.append(dict(key="zero", image_sha256=digest(zero)))
        idx = 1
        for j in range(35):
            for sign in (1, -1):
                coeff = np.zeros(35)
                coeff[j] = sign * CALIB_AMP
                image = model.evaluate(coeff)
                spec[idx] = fft.fftshift(fft.fft2((image - image.min()) / image.max()))
                calib_meta.append(dict(key=f"j{j+1:02d}{'p' if sign > 0 else 'm'}",
                    image_sha256=digest(image), coefficient=coeff.tolist()))
                idx += 1
                monitor.update(image=idx, eta_remaining_seconds=(71 - idx) * 11.0)
                if idx % 10 == 0:
                    monitor.event("prep_spectrum", kind=kind, completed=idx)
        del image
        save(folder / "calib_meta.json", calib_meta)
        spec.flush()
    # Independent truth images: run-08 holdouts + new random truths.
    truths = {}
    for seed in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS:
        path = folder / f"truth_{seed}.npy"
        if path.exists():
            truths[seed] = np.load(path)
        else:
            t = truth_for(model, seed)
            truths[seed] = model.evaluate(t) / model.evaluate(t).max()
            np.save(path, truths[seed])
        monitor.update(stage="prep_truths", kind=kind, completed=len(truths))
    save(folder / "truths.json", {int(k): digest(v) for k, v in truths.items()})
    # Main truth coeffs (run-08 PROTOCOL truth) for metrics.
    save(folder / "main_truth.json", dict(
        coefficients=truth_for(model).tolist(),
        indices=list(PROTOCOL["truth_indices"]),
        weights=PROTOCOL["truth_weights"]))
    del zero
    gc.collect()
    return folder, truths


def calibrate(features, calib_spec, meta_path):
    """Build response matrix and baseline from cached calibration spectra."""
    meta = json.loads(Path(meta_path).read_text())
    idx_of = {m["key"]: i for i, m in enumerate(meta)}
    cols = []
    for j in range(35):
        fplus = None
        for sign, tag in ((1, "p"), (-1, "m")):
            key = f"j{j+1:02d}{tag}"
            f, _ = features.extract_from_spectrum(np.asarray(calib_spec[idx_of[key]]))
            if sign == 1:
                fplus = f
            else:
                cols.append((fplus - f) / (2 * CALIB_AMP))
    R = np.stack(cols, axis=1)
    baseline, _ = features.extract_from_spectrum(np.asarray(calib_spec[0]))
    sv = np.linalg.svd(R, compute_uv=False)
    errs = []
    for j in range(35):
        for sign, tag in ((1, "p"), (-1, "m")):
            key = f"j{j+1:02d}{tag}"
            f, _ = features.extract_from_spectrum(np.asarray(calib_spec[idx_of[key]]))
            est = np.linalg.lstsq(R, f - baseline, rcond=None)[0]
            target = np.zeros(35)
            target[j] = sign * CALIB_AMP
            errs.append(float(abs(est - target).max()))
    return dict(response=R, baseline=baseline,
        rank=int(np.linalg.matrix_rank(R)), condition=float(np.linalg.cond(R)),
        singular_values=sv.tolist(), training_max_coefficient_error=max(errs))


def apply_calibration(cal, features, image_or_spec, from_spectrum=False):
    if from_spectrum:
        f, diag = features.extract_from_spectrum(image_or_spec)
    else:
        f, diag = features(image_or_spec)
    est = np.linalg.lstsq(cal["response"], f - cal["baseline"], rcond=None)[0]
    return est, diag


def config_list(fundamental, shear_pixels):
    """All sweep configurations; baseline (E0) first."""
    base = RegionSettings()
    hw0 = int(0.4 * fundamental)
    sh0 = int(round(shear_pixels))
    configs = [dict(group="E0_baseline", name="baseline", settings=base)]
    # E1: window width.
    for hw_frac in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50):
        configs.append(dict(group="E1_width", name=f"hw{hw_frac:.2f}",
            settings=RegionSettings(half_width=int(hw_frac * fundamental))))
    # E2: 2D window center offset (bins), integer + subpixel.
    for ox in (-3, -2, -1, 0, 1, 2, 3):
        configs.append(dict(group="E2_center_x", name=f"ox{ox:+d}",
            settings=RegionSettings(offset_x=float(ox))))
    for oy in (-3, -2, -1, 1, 2, 3):
        configs.append(dict(group="E2_center_y", name=f"oy{oy:+d}",
            settings=RegionSettings(offset_y=float(oy))))
    for ox in (-1.5, -0.5, 0.5, 1.5):
        configs.append(dict(group="E2_center_subpx", name=f"sx{ox:+.1f}",
            settings=RegionSettings(offset_x=ox, subpixel=True)))
    # E3: window shape / transition, ENBW reported alongside.
    for shape, sigma in (("rect", 0), ("rect", 5), ("circle", 0),
                        ("gaussian", 0), ("ellipse", 0)):
        configs.append(dict(group="E3_shape", name=f"{shape}_s{sigma}",
            settings=RegionSettings(shape=shape, window_sigma=float(sigma),
                                    aspect=(0.5 if shape == "ellipse" else 1.0))))
    # E4: spatial mask / smoothing.
    for er in (2, 4, 8, 12):
        configs.append(dict(group="E4_erosion", name=f"er{er}",
            settings=RegionSettings(erosion=er)))
    for thr in (0.1, 0.3, 0.5):
        configs.append(dict(group="E4_amplitude", name=f"thr{thr}",
            settings=RegionSettings(amplitude_threshold=thr)))
    for ps in (0.0, 5.0, 20.0):
        configs.append(dict(group="E4_phase_sigma", name=f"ps{ps:.0f}",
            settings=RegionSettings(phase_sigma=ps)))
    # E5a: center estimation method (evaluated across conditions, not rebuilt per image here;
    #       the sweep sets the mode; per-image peak search happens in extract_from_spectrum).
    for mode in ("peak", "subpixel"):
        configs.append(dict(group="E5a_center_mode", name=f"cm_{mode}",
            settings=RegionSettings(center_mode=mode)))
    # E5c: shear integer, mask shift, sideband (2f).
    for sh in (sh0 - 1, sh0 + 1):
        configs.append(dict(group="E5c_shear", name=f"sh{sh}",
            settings=RegionSettings(shear_px=sh)))
    for ms in (-2, -1, 1, 2):
        configs.append(dict(group="E5c_mask_shift", name=f"msx{ms:+d}",
            settings=RegionSettings(mask_shift_x=ms)))
    configs.append(dict(group="E5c_sideband", name="sideband_2f",
        settings=RegionSettings(carrier_override=int(round(2 * fundamental)),
                                 sideband_label="2f")))
    return configs


def sweep_one(model, kind, cache, run08, monitor):
    """Run the full configuration sweep for one (NA, source_kind)."""
    folder = cache / f"na{NA:.3f}_{kind}"
    g = geometry(model.config)
    fund = g["fundamental_bins"]
    shp = g["shear_pixels"]
    calib_spec = np.memmap(folder / "calib_spectra.npy", dtype=complex128,
                           mode="r", shape=(71, model.n, model.n))
    meta_path = folder / "calib_meta.json"
    clean, obs = load_observations(run08, kind)
    truths = {int(s): np.load(folder / f"truth_{s}.npy")
              for s in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS}
    main_truth = np.array(json.loads((folder / "main_truth.json").read_text())["coefficients"])
    zero_image = np.load(run08 / f"na{NA:.3f}_{kind}" / "zero.npy")
    obs_spec = {}
    for name, im in obs.items():
        im = im - im.min()
        obs_spec[name] = fft.fftshift(fft.fft2(im / im.max()))
    truth_spec = {}
    for s, im in truths.items():
        im = im - im.min()
        truth_spec[s] = fft.fftshift(fft.fft2(im / im.max()))
    configs = config_list(fund, shp)
    rows = []
    total = len(configs)
    for ci, cfg in enumerate(configs):
        monitor.update(stage="sweep", kind=kind, config=f"{ci+1}/{total}",
                        group=cfg["group"], name=cfg["name"],
                        eta_remaining_seconds=(total - ci) * 140.0)
        rs = cfg["settings"].resolved(fund, shp)
        try:
            feat = RegionFeatures(model.config, zero_image, rs)
            cal = calibrate(feat, calib_spec, meta_path)
        except Exception as exc:
            rows.append(dict(group=cfg["group"], name=cfg["name"], kind=kind,
                error=repr(exc), settings=asdict(rs)))
            save(cache / "sweep_results.json", rows)
            continue
        for name in ("clean",) + tuple(f"40db_seed{s}" for s in NOISE_40) + ("30db_seed42",):
            est, diag = apply_calibration(cal, feat, obs_spec[name], from_spectrum=True)
            row = dict(group=cfg["group"], name=cfg["name"], kind=kind, condition=name,
                settings=asdict(rs), cal_rank=cal["rank"], cal_condition=cal["condition"],
                cal_training_error=cal["training_max_coefficient_error"], **diag,
                **metrics(model, est, main_truth))
            rows.append(row)
        for s in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS:
            t = truth_for(model, s)
            est, diag = apply_calibration(cal, feat, truth_spec[s], from_spectrum=True)
            row = dict(group=cfg["group"], name=cfg["name"], kind=kind,
                condition=f"truth{int(s)}", truth_seed=int(s),
                settings=asdict(rs), cal_rank=cal["rank"], cal_condition=cal["condition"],
                cal_training_error=cal["training_max_coefficient_error"], **diag,
                **metrics(model, est, t))
            rows.append(row)
        save(cache / "sweep_results.json", rows)
        monitor.event("sweep_config", kind=kind, group=cfg["group"], name=cfg["name"],
            completed=ci + 1, total=total)
        del feat
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prep", "sweep", "all"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run08", type=Path,
        default=Path("/data/yk/shear-reference-lm/na-talbot-balanced-20261111/run-08"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert head == args.expected_commit
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    assert not dirty, dirty
    assert args.allow_large_run and os.uname().nodename == "sjtu-gxyserver"
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "version.json", dict(commit=head, stage=args.stage,
        command=sys.argv, start_utc=utc(), hostname=os.uname().nodename,
        python=sys.version, numpy=np.__version__, na_out=NA, kinds=list(KINDS),
        source_hashes={str(Path(__file__).relative_to(REPO)):
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    from lsi.execution import exclusive_study, interrupt_on_termination
    with exclusive_study(), interrupt_on_termination(), threadpool_limits(
            limits=16), fft.set_workers(8):
        with Monitor(args.output, args.stage) as monitor:
            for kind in KINDS:
                c = config(NA, kind)
                model = ExtendedCameraModel(c, INDICES, batch_sources=1, workers=1)
                if args.stage in ("prep", "all"):
                    prep(model, kind, args.output, args.run08, monitor)
                if args.stage in ("sweep", "all"):
                    sweep_one(model, kind, args.output, args.run08, monitor)
                del model
                gc.collect()


if __name__ == "__main__":
    main()
