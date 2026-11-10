"""Original single-source Lasso extraction and new-geometry diagnostics."""
from inspect import signature
from dataclasses import replace
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase
from sklearn.linear_model import Lasso

from lsi.fourier import FourierExtractor, FourierSettings, spectral_window

RANDOM = {"rng": 0} if "rng" in signature(unwrap_phase).parameters else {"seed": 0}


def settings_for(cfg):
    focus_shift = (cfg.f2 / cfg.f1)**2 * cfg.shift_z if cfg.error_mode == "ON" else 0.
    distance = cfg.z_grating - cfg.shift_z_grating - focus_shift + cfg.z_cam
    actual_R = distance-cfg.z_cam
    fundamental = cfg.n*cfg.dx_cam*actual_R/(cfg.grating_period*distance)
    # The existing +/- shear stencil corresponds to the +1/-1 cross term.
    nominal = 2*fundamental
    carrier = int(np.rint(nominal))
    # Separate this band from the adjacent 0/+1 and +1/-2 harmonics.
    half_width = max(2, int(np.floor(.4 * fundamental)))
    settings = FourierSettings(
        half_width=half_width, carrier_x=carrier, carrier_y=carrier,
        phase_sigma=10., recenter_adjust_x=-1, recenter_adjust_y=-1,
        regression="lstsq", alpha=0., window_sigma=2.,
    )
    return settings, dict(
        rule="opposite-order carrier=round(2*N*dx_cam*R/(p*(R+z_cam))); half_width=floor(0.4*fundamental)",
        focus_shift_m=focus_shift, actual_focus_to_grating_paraxial_m=distance-cfg.z_cam,
        nominal_carrier_bins=nominal, selected_carrier_bins=carrier,
        half_width_bins=half_width,
        shadow_fundamental_bins=fundamental,
        selected_diffraction_orders=[-1,1],
        validation="Directly isolated original CZT fields: 0/+1 peak=63; -1/+1 peak=125",
        selection_uses_truth=False,
        interpretation="Paraxial geometric anchor; zero-wavefront spectra recorded independently.",
    )


class OriginalSingleFT(FourierExtractor):
    """Preserve mask -> unwrap -> smooth -> Lasso, without reference subtraction."""

    def extract(self, image, details=False, reference_image=None):
        s, n = self.settings, self.n
        image = np.asarray(image, float)
        image = image - image.min()
        spectrum = np.fft.fftshift(np.fft.fft2(image / image.max()))
        phases = []
        for k, axis in enumerate((1, 0)):
            carrier = s.carrier_x if k == 0 else s.carrier_y
            center = (n // 2 + (carrier if k == 0 else 0),
                      n // 2 + (carrier if k == 1 else 0))
            window = spectral_window(n, center, s)
            adjust = s.recenter_adjust_x if k == 0 else s.recenter_adjust_y
            shifted = np.roll(spectrum * window, -int(carrier + adjust), axis=axis)
            phase = np.angle(np.fft.ifft2(np.fft.ifftshift(shifted)))
            phase[~self.masks[k]] = 0.
            phase = gaussian_filter(unwrap_phase(phase, **RANDOM), s.phase_sigma)
            phases.append(phase[self.masks[k]])
        target = np.concatenate(phases)
        solver = Lasso(alpha=.02, max_iter=1000, tol=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver.fit(self.A, target)
        return dict(raw_coeffs=solver.coef_, intercept=float(solver.intercept_),
                    regression="Lasso", alpha=.02,
                    warning_messages=[str(w.message) for w in caught],
                    phase_residual_rms=float(np.sqrt(np.mean(
                        (self.A @ solver.coef_ + solver.intercept_ - target)**2))))


def spectrum_diagnostics(zero, cfg, settings, geometry):
    from scipy.signal import find_peaks
    image = zero - zero.min()
    spectrum = abs(np.fft.fftshift(np.fft.fft2(image / image.max())))
    middle = cfg.n // 2
    profiles = [spectrum[middle, middle:], spectrum[middle:, middle]]
    diagnostics = []
    for profile in profiles:
        peaks, _ = find_peaks(profile)
        peaks = peaks[(peaks >= 5) & (peaks < min(middle, 200))]
        peaks = peaks[np.argsort(profile[peaks])[::-1]][:15]
        lo = max(1, int(np.floor(geometry["nominal_carrier_bins"] - 3)))
        hi = int(np.ceil(geometry["nominal_carrier_bins"] + 3)) + 1
        anchored = lo + int(np.argmax(profile[lo:hi]))
        diagnostics.append(dict(
            strongest_peaks_bins=peaks.tolist(),
            strongest_peaks_amplitude_relative_dc=(profile[peaks]/profile[0]).tolist(),
            maximum_within_geometry_plusminus_3_bins=anchored,
            selected_bin_amplitude_relative_dc=float(profile[int(settings.carrier_x)]/profile[0]),
        ))
    windows = [
        spectral_window(cfg.n, (middle + settings.carrier_x, middle), settings),
        spectral_window(cfg.n, (middle, middle + settings.carrier_y), settings),
    ]
    geometry = {**geometry, "axes": diagnostics,
                "dc_window_weights": [float(w[middle, middle]) for w in windows],
                "x_y_window_overlap_fraction": float(
                    np.minimum(*windows).sum()/windows[0].sum())}
    return geometry, spectrum, profiles
    def __init__(self, cfg, settings):
        super().__init__(cfg, replace(settings, regression="Lasso", alpha=.02))
