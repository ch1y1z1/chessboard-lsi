"""Retrieval0806 sideband extraction and explicit pupil-plane calibration.

Raw differential coefficients are in the reference's arbitrary phase-fit
units. Only an independently generated calibration maps these to pupil waves.
"""
from dataclasses import asdict, dataclass
from inspect import signature
from time import perf_counter
import numpy as np
from scipy import ndimage, linalg
from skimage.restoration import unwrap_phase
from sklearn.linear_model import Lasso
from .annular import basis, mode_labels

_UNWRAP_RANDOM = ({'rng': 0} if 'rng' in signature(unwrap_phase).parameters else {'seed': 0})


@dataclass(frozen=True)
class FourierSettings:
    half_width: float = 20
    aspect: float = 1.0
    offset_x: float = 0
    offset_y: float = 0
    shape: str = "rect"
    window_sigma: float = 2.0
    carrier_x: float = 44
    carrier_y: float = 44
    recenter_adjust_x: float = -1
    recenter_adjust_y: float = -1
    phase_sigma: float = 10.0
    shear_adjust: float = 1.0
    erosion: int = 0
    amplitude_threshold: float = 0.0
    center_mode: str = "fixed"
    search_radius: int = 8
    subpixel: bool = False
    regression: str = "Lasso"
    alpha: float = 0.02
    lasso_max_iter: int = 1000
    lasso_tol: float = 1e-4

    def __post_init__(self):
        if self.half_width <= 0 or self.aspect <= 0 or self.phase_sigma < 0 or self.window_sigma < 0:
            raise ValueError("invalid window or smoothing width")
        if self.shape not in ("rect", "circle", "ellipse", "gaussian"):
            raise ValueError("unknown window shape")
        if self.regression not in ("Lasso", "Ridge", "lstsq") or self.alpha < 0:
            raise ValueError("invalid regression")
        if self.center_mode not in ("fixed", "auto"):
            raise ValueError("unknown center mode")
        if self.erosion < 0 or not 0 <= self.amplitude_threshold < 1:
            raise ValueError("invalid spatial selection")

    def to_dict(self):
        return asdict(self)


def spectral_window(n, center, settings):
    yy, xx = np.indices((n, n))
    dx, dy = xx - center[0], yy - center[1]
    hx = settings.half_width / np.sqrt(settings.aspect)
    hy = settings.half_width * np.sqrt(settings.aspect)
    if settings.shape == "rect":
        w = ((dx >= -hx) & (dx < hx) & (dy >= -hy) & (dy < hy)).astype(float)
    elif settings.shape in ("circle", "ellipse"):
        # Equal continuous area to the reference rectangle.
        if settings.shape == "circle":
            hx = hy = np.sqrt(hx * hy)
        w = ((dx / (hx * 2 / np.sqrt(np.pi)))**2 +
             (dy / (hy * 2 / np.sqrt(np.pi)))**2 <= 1).astype(float)
    else:
        # Equal integrated continuous window weight; also report discrete ENBW.
        sx, sy = hx * np.sqrt(2 / np.pi), hy * np.sqrt(2 / np.pi)
        return np.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2))
    return ndimage.gaussian_filter(w, settings.window_sigma) if settings.window_sigma else w


class FourierExtractor:
    def __init__(self, config, settings=None):
        self.config = c = config
        self.settings = s = settings or FourierSettings()
        self.n = c.n
        v = (np.arange(c.n) - c.n // 2) * c.dx_cam
        x, y = np.meshgrid(v, v)
        radius = (c.z_grating + c.z_cam) * c.NA_out * c.f1 / c.f2
        rho, theta = np.hypot(x, y) / radius, np.arctan2(y, x)
        self.pupil = (rho >= c.epsilon) & (rho <= 1)
        self.shear = int(c.wavelength / c.grating_period *
                         (c.z_grating + c.z_cam) / c.dx_cam * s.shear_adjust)
        if self.shear < 1:
            raise ValueError("integer shear is zero on this grid")
        self.masks = []
        for axis in (1, 0):
            mask = np.roll(self.pupil, self.shear, axis) & np.roll(self.pupil, -self.shear, axis)
            if s.erosion:
                mask = ndimage.binary_erosion(mask, iterations=s.erosion)
            if not mask.any():
                raise ValueError("spatial mask is empty")
            self.masks.append(mask)
        # Each row uses exactly the reference integer, periodic np.roll stencil.
        # Construct one mode at a time to avoid storing 3 full basis stacks.
        self.basis_indices = tuple(range(1, len(mode_labels(c.Nz))))
        rows_x, rows_y = [], []
        for j in self.basis_indices:
            z = basis((j,), rho, theta, c.epsilon, c.Nz)[0]
            rows_x.append((np.roll(z, -self.shear, 1) - np.roll(z, self.shear, 1))[self.masks[0]])
            rows_y.append((np.roll(z, -self.shear, 0) - np.roll(z, self.shear, 0))[self.masks[1]])
        self.A = np.vstack((np.stack(rows_x, axis=1), np.stack(rows_y, axis=1)))
        self._linear_inverse = None

    def extract(self, image, details=False, reference_image=None):
        start = perf_counter()
        s, n = self.settings, self.n
        image = np.asarray(image, float)
        if image.shape != (n, n) or not np.isfinite(image).all():
            raise ValueError("invalid Fourier input image")
        # Faithful preprocessing from wf_recon_sim0924.py.
        image = image - image.min()
        maximum = image.max()
        if maximum <= 0:
            raise ValueError("constant image has no shear signal")
        spectrum = np.fft.fftshift(np.fft.fft2(image / maximum))
        reference_spectrum = None
        if reference_image is not None:
            reference_image = np.asarray(reference_image, float)
            reference_image = reference_image - reference_image.min()
            if reference_image.shape != image.shape or reference_image.max() <= 0:
                raise ValueError("invalid independent reference image")
            reference_spectrum = np.fft.fftshift(np.fft.fft2(reference_image / reference_image.max()))
        wrapped, phases, amps, windows, centers, valid_rows, window_metrics = [], [], [], [], [], [], []
        for k, axis in enumerate((1, 0)):
            carrier = s.carrier_x if k == 0 else s.carrier_y
            cx = n // 2 + (carrier if k == 0 else 0) + s.offset_x
            cy = n // 2 + (carrier if k == 1 else 0) + s.offset_y
            if s.center_mode == "auto":
                r = s.search_radius
                x0, x1 = max(0, int(cx) - r), min(n, int(cx) + r + 1)
                y0, y1 = max(0, int(cy) - r), min(n, int(cy) + r + 1)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("sideband search is outside FFT grid")
                iy, ix = np.unravel_index(np.argmax(abs(spectrum[y0:y1, x0:x1])), (y1-y0, x1-x0))
                cx, cy = x0 + ix, y0 + iy
            if not 0 <= cx < n or not 0 <= cy < n:
                raise ValueError("sideband center outside FFT grid")
            win = spectral_window(n, (cx, cy), s)
            weight = float(win.sum())
            window_metrics.append({'weight_sum': weight,
                                   'equivalent_noise_bins': float(weight*weight / max(float(np.sum(win*win)), 1e-300)),
                                   'dc_weight': float(win[n//2, n//2])})
            adjust = s.recenter_adjust_x if k == 0 else s.recenter_adjust_y
            sx, sy = cx - n // 2, cy - n // 2
            if k == 0:
                sx += adjust
            else:
                sy += adjust
            # Integer mode reproduces np.roll. Fractional shifting is a separate option.
            if s.subpixel:
                sig = np.fft.ifft2(np.fft.ifftshift(spectrum * win))
                yy, xx = np.indices((n, n))
                sig *= np.exp(-2j * np.pi * (sx * xx + sy * yy) / n)
            else:
                shifted = np.roll(spectrum * win, (-int(sy), -int(sx)), axis=(0, 1))
                sig = np.fft.ifft2(np.fft.ifftshift(shifted))
            mask = self.masks[k]
            phase_signal = sig
            if reference_spectrum is not None:
                # Subtract the *independent zero-aberration reference* in complex
                # form before unwrapping. This prevents unrelated whole-cycle
                # branches from entering the pupil coefficient calibration.
                refsig = np.fft.ifft2(np.fft.ifftshift(reference_spectrum * win))
                if s.subpixel:
                    refsig *= np.exp(-2j * np.pi * (sx * xx + sy * yy) / n)
                else:
                    refsig = np.fft.ifft2(np.fft.ifftshift(np.roll(
                        reference_spectrum * win, (-int(sy), -int(sx)), axis=(0, 1))))
                phase_signal = sig * np.conj(refsig)
            phase_wrapped = np.where(mask, np.angle(phase_signal), 0.0)
            phase = unwrap_phase(phase_wrapped, **_UNWRAP_RANDOM)
            if s.phase_sigma:
                phase = ndimage.gaussian_filter(phase, s.phase_sigma)
            good = np.ones(mask.sum(), bool)
            if s.amplitude_threshold:
                amp = abs(sig)[mask]
                good = amp >= s.amplitude_threshold * amp.max()
            valid_rows.append(good)
            phases.append(phase[mask])
            centers.append((float(cx), float(cy)))
            if details:
                wrapped.append(phase_wrapped); amps.append(abs(sig)); windows.append(win)
        b, good = np.concatenate(phases), np.concatenate(valid_rows)
        A = self.A if good.all() else self.A[good]
        b = b[good]
        if len(b) <= A.shape[1]:
            raise ValueError("insufficient valid phase pixels")
        intercept = 0.0
        if s.regression == "Lasso":
            clf = Lasso(alpha=s.alpha, max_iter=s.lasso_max_iter, tol=s.lasso_tol)
            clf.fit(A, b)
            coef, intercept = clf.coef_, float(clf.intercept_)
        elif s.regression == "Ridge" and s.alpha:
            coef = linalg.lstsq(np.vstack((A, np.sqrt(s.alpha) * np.eye(A.shape[1]))),
                                np.r_[b, np.zeros(A.shape[1])], lapack_driver="gelsy")[0]
        else:
            if good.all():
                if self._linear_inverse is None:
                    self._linear_inverse = linalg.pinv(A)
                coef = self._linear_inverse @ b
            else:
                coef = linalg.lstsq(A, b, lapack_driver="gelsy")[0]
        result = {"raw_coeffs": coef, "units": "reference_phase_fit_au",
                  "reference_subtracted": reference_spectrum is not None,
                  "phase_residual_rms": float(np.sqrt(np.mean((A @ coef + intercept - b)**2))),
                  "coverage": float(good.sum() / (2 * n * n)), "centers": centers,
                  "window_metrics": window_metrics,
                  "seconds": perf_counter() - start,
                  "settings": s.to_dict(), "intercept": intercept}
        if details:
            result.update(wrapped=np.array(wrapped), amplitudes=np.array(amps),
                          windows=np.array(windows), masks=np.array(self.masks),
                          spectrum_log=np.log10(abs(spectrum) + 1e-12))
        return result


@dataclass
class FourierCalibration:
    baseline: np.ndarray
    response: np.ndarray
    amplitude: float
    indices: tuple
    settings: dict
    config: dict
    reference_image: np.ndarray

    @classmethod
    def fit(cls, extractor, images, indices, amplitude):
        """images = zero, then plus/minus for each *independent* calibration mode."""
        if len(images) != 1 + 2 * len(indices) or amplitude <= 0:
            raise ValueError("invalid calibration data")
        features = [extractor.extract(im, reference_image=images[0])["raw_coeffs"] for im in images]
        R = np.stack([(features[1+2*j] - features[2+2*j]) / (2 * amplitude)
                      for j in range(len(indices))], axis=1)
        if np.linalg.matrix_rank(R) < len(indices):
            raise ValueError("Fourier calibration is rank deficient; inspect window and modes")
        return cls(features[0], R, float(amplitude), tuple(indices),
                   extractor.settings.to_dict(), extractor.config.to_dict(), np.array(images[0]))

    def apply(self, raw_coeffs):
        return np.linalg.lstsq(self.response, np.asarray(raw_coeffs) - self.baseline, rcond=None)[0]

    def reconstruct(self, extractor, image, details=False):
        features = extractor.extract(image, details=details, reference_image=self.reference_image)
        features["coeffs_waves"] = self.apply(features["raw_coeffs"])
        features["calibration_settings"] = self.settings
        features["calibration_condition"] = float(np.linalg.cond(self.response))
        return features


def calibration_images(model, amplitude=0.02):
    if amplitude <= 0:
        raise ValueError("calibration amplitude must be positive")
    zero = np.zeros(len(model.indices))
    images = [model.evaluate(zero)]
    for j in range(len(zero)):
        p = zero.copy(); p[j] = amplitude
        images.extend((model.evaluate(p), model.evaluate(-p)))
    return images
