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
        values, good_masks, maxima, branch = [], [], [], []
        for k, axis in enumerate((1, 0)):
            cx, cy = self._locate_center(spec, axis)
            ix = int(round(cx - self.centers[k][0]))
            iy = int(round(cy - self.centers[k][1]))
            rolled = np.roll(spec * self.windows[k], -self.carrier - ix, axis)
            if iy:
                rolled = np.roll(rolled, -iy, axis=(1 - axis))
            signal_field = fft.ifft2(fft.ifftshift(rolled))[self.roi].copy()
            if s.subpixel:
                # Use ROI-sized indices, not full-grid indices.
                yy, xx = np.indices(signal_field.shape)
                sx = (cx - self.centers[k][0]) - ix
                sy = (cy - self.centers[k][1]) - iy
                signal_field *= np.exp(-2j * np.pi * (sx * xx + sy * yy) / self.c.camera_n)
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
            values.append(phase[self.masks[k]])
            good_masks.append(good)
        b = np.concatenate(values)
        valid = np.concatenate(good_masks)
        A = self.A_pinv[:, valid] if not valid.all() else self.A_pinv
        # A_pinv is shape (35, total_pixels); we need (35, valid_pixels) = pinv slice.
        # Actually A_pinv @ b[valid] where A_pinv maps from pixel-space to coeff-space.
        # A_pinv columns correspond to pixels; selecting valid columns applies the mask.
        result = (A @ b[valid]) if not valid.all() else (self.A_pinv @ b)
        return result, dict(
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
    """Generate and cache calibration spectra one file at a time (no memmap)."""
    folder = cache / f"na{NA:.3f}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / "config.json", model.config.to_dict())
    save(folder / "geometry.json", geometry(model.config))
    zero = np.load(run08 / f"na{NA:.3f}_{kind}" / "zero.npy")
    calib_meta_path = folder / "calib_meta.json"
    # Build calib_meta incrementally; skip already-computed keys.
    calib_meta = json.loads(calib_meta_path.read_text()) if calib_meta_path.exists() else {}
    # key -> dict(image_sha256, coefficient)
    total = 71  # zero + 35*2
    done = len(calib_meta)
    if "zero" not in calib_meta:
        sp = fft.fftshift(fft.fft2((zero - zero.min()) / zero.max()))
        np.save(folder / "calib_spec_zero.npy", sp)
        del sp
        calib_meta["zero"] = dict(image_sha256=digest(zero))
        save(calib_meta_path, calib_meta)
        done = len(calib_meta)
    monitor.update(stage="prep_spectra", kind=kind, image=done, total=total)
    sec_per_image = []
    for j in range(35):
        for sign, tag in ((1, "p"), (-1, "m")):
            key = f"j{j+1:02d}{tag}"
            if key in calib_meta:
                continue
            coeff = np.zeros(35); coeff[j] = sign * CALIB_AMP
            t0 = time.monotonic()
            image = model.evaluate(coeff)
            sp = fft.fftshift(fft.fft2((image - image.min()) / image.max()))
            np.save(folder / f"calib_spec_{key}.npy", sp)
            del sp, image; gc.collect()
            elapsed = time.monotonic() - t0
            sec_per_image.append(elapsed)
            calib_meta[key] = dict(image_sha256="", coefficient=coeff.tolist())
            save(calib_meta_path, calib_meta)
            done = len(calib_meta)
            avg = np.mean(sec_per_image)
            monitor.update(image=done, eta_remaining_seconds=(total - done) * avg)
            if done % 10 == 0:
                monitor.event("prep_spectrum", kind=kind, completed=done)
    # Independent truth images.
    truths_meta = {}
    for seed in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS:
        path = folder / f"truth_{seed}.npy"
        if not path.exists():
            t = truth_for(model, seed)
            img = model.evaluate(t); img = img / img.max()
            np.save(path, img); del img; gc.collect()
        truths_meta[int(seed)] = dict(truth=truth_for(model, seed).tolist())
        monitor.update(stage="prep_truths", kind=kind, completed=len(truths_meta))
    save(folder / "truths_meta.json", truths_meta)
    save(folder / "main_truth.json", dict(
        coefficients=truth_for(model).tolist(),
        indices=list(PROTOCOL["truth_indices"]),
        weights=PROTOCOL["truth_weights"]))
    del zero; gc.collect()


def load_calib_spec(folder, key):
    """Load a single precomputed spectrum by key (zero / j01p / j01m …)."""
    return np.load(folder / f"calib_spec_{key}.npy")


def calibrate(features, folder):
    """Build response matrix from individual per-key spectrum files in folder."""
    cols = []
    for j in range(35):
        fplus = None
        for sign, tag in ((1, "p"), (-1, "m")):
            key = f"j{j+1:02d}{tag}"
            sp = load_calib_spec(folder, key)
            f, _ = features.extract_from_spectrum(sp)
            del sp
            if sign == 1:
                fplus = f
            else:
                cols.append((fplus - f) / (2 * CALIB_AMP))
    R = np.stack(cols, axis=1)
    sp = load_calib_spec(folder, "zero")
    baseline, _ = features.extract_from_spectrum(sp)
    del sp
    sv = np.linalg.svd(R, compute_uv=False)
    # Spot-check 4 diagonal modes instead of all 70 to keep memory low.
    errs = []
    for j in (0, 8, 17, 33):
        for sign, tag in ((1, "p"), (-1, "m")):
            key = f"j{j+1:02d}{tag}"
            sp = load_calib_spec(folder, key)
            f, _ = features.extract_from_spectrum(sp)
            del sp
            est = np.linalg.lstsq(R, f - baseline, rcond=None)[0]
            target = np.zeros(35); target[j] = sign * CALIB_AMP
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
    clean, obs = load_observations(run08, kind)
    zero_image = np.load(run08 / f"na{NA:.3f}_{kind}" / "zero.npy")
    main_truth = np.array(json.loads((folder / "main_truth.json").read_text())["coefficients"])
    obs_spec = {}
    for name, im in obs.items():
        im2 = im - im.min()
        obs_spec[name] = fft.fftshift(fft.fft2(im2 / im2.max()))
    del obs, clean
    truth_specs = {}
    for s in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS:
        im = np.load(folder / f"truth_{s}.npy")
        im2 = im - im.min()
        truth_specs[int(s)] = fft.fftshift(fft.fft2(im2 / im2.max()))
        del im, im2
    truth_coeffs = {s: np.array(v["truth"])
                    for s, v in json.loads((folder / "truths_meta.json").read_text()).items()}
    configs = config_list(fund, shp)
    results_path = cache / "sweep_results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    done_names = {(r["kind"], r["group"], r["name"], r.get("condition")) for r in rows
                  if "error" not in r}
    total = len(configs)
    times = []
    for ci, cfg in enumerate(configs):
        monitor.update(stage="sweep", kind=kind, config=f"{ci+1}/{total}",
                       group=cfg["group"], name=cfg["name"],
                       eta_remaining_seconds=(total - ci) * (np.mean(times) if times else 90.0))
        rs = cfg["settings"].resolved(fund, shp)
        t0 = time.monotonic()
        try:
            feat = RegionFeatures(model.config, zero_image, rs)
            cal = calibrate(feat, folder)
        except Exception as exc:
            rows.append(dict(group=cfg["group"], name=cfg["name"], kind=kind,
                             error=repr(exc), settings=asdict(rs)))
            save(results_path, rows)
            continue
        obs_conditions = (("clean",) +
                          tuple(f"40db_seed{s}" for s in NOISE_40) +
                          ("30db_seed42",))
        for cond in obs_conditions:
            if (kind, cfg["group"], cfg["name"], cond) in done_names:
                continue
            est, diag = apply_calibration(cal, feat, obs_spec[cond], from_spectrum=True)
            row = dict(group=cfg["group"], name=cfg["name"], kind=kind, condition=cond,
                       settings=asdict(rs), cal_rank=cal["rank"],
                       cal_condition=cal["condition"],
                       cal_training_error=cal["training_max_coefficient_error"],
                       **diag, **metrics(model, est, main_truth))
            rows.append(row)
        for s in HOLDOUT_SEEDS + NEW_TRUTH_SEEDS:
            cond = f"truth{int(s)}"
            if (kind, cfg["group"], cfg["name"], cond) in done_names:
                continue
            t = np.array(truth_coeffs[str(s)])
            est, diag = apply_calibration(cal, feat, truth_specs[int(s)], from_spectrum=True)
            row = dict(group=cfg["group"], name=cfg["name"], kind=kind,
                       condition=cond, truth_seed=int(s),
                       settings=asdict(rs), cal_rank=cal["rank"],
                       cal_condition=cal["condition"],
                       cal_training_error=cal["training_max_coefficient_error"],
                       **diag, **metrics(model, est, t))
            rows.append(row)
        save(results_path, rows)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        monitor.event("sweep_config", kind=kind, group=cfg["group"], config_name=cfg["name"],
                      completed=ci + 1, total=total, seconds=elapsed)
        del feat, cal
        gc.collect()
    return rows
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
