"""Reference BUV0725 propagation, with cached batched centered CZTs.

The phase-ramp tilt convention deliberately matches the reference. It is not
a rotated-detector coordinate transform. All fields use complex128.
"""
from time import perf_counter
import numpy as np
from scipy import fft
from scipy.signal import CZT
from .config import DiffractionConfig
from .annular import basis, configured_coeffs


class CenteredCZT:
    def __init__(self, n, m, dx, f0, df):
        self.plan = CZT(n, m, np.exp(-2j * np.pi * df * dx), np.exp(2j * np.pi * f0 * dx))
        self.factor = np.exp(2j * np.pi * (f0 + np.arange(m) * df) * (n / 2) * dx) * dx

    def __call__(self, field):
        a = self.plan(field, axis=-1) * self.factor
        a = self.plan(a, axis=-2)
        return a * self.factor[:, None]


def grid(n, dx):
    v = (np.arange(n) - n // 2) * dx
    return np.meshgrid(v, v)


def tilt_phase(x, y, degrees, ax, ay, k):
    norm = np.hypot(ax, ay)
    if norm == 0:
        raise ValueError("tilt axis cannot be zero")
    return np.exp(-1j * k * np.tan(np.deg2rad(degrees)) * (-ay * x + ax * y) / norm)


class DiffractionModel:
    def __init__(self, config=None, indices=None, batch_sources=2, workers=1, backend='separable'):
        self.config = c = config or DiffractionConfig()
        self.indices = tuple(indices if indices is not None else (7, 25, 11, 15))
        if len(set(self.indices)) != len(self.indices) or 0 in self.indices:
            raise ValueError("fit distinct non-piston modes")
        self.batch_sources = max(1, int(batch_sources))
        self.workers = int(workers)
        self.n = n = c.n
        self.ng = ng = n * c.up_sample_grating
        self.k = k = 2 * np.pi / c.wavelength
        self.dx_obj = c.super_lattice_period / c.N_lattice
        self.dx_img = self.dx_obj * c.f2 / c.f1
        self.x_obj, self.y_obj = grid(n, self.dx_obj)
        self.x_img, self.y_img = grid(n, self.dx_img)
        self.x_g, self.y_g = grid(ng, c.dx_grating)
        self.x_cam, self.y_cam = grid(n, c.dx_cam)
        freq = fft.fftshift(fft.fftfreq(n, self.dx_obj))
        fx, fy = np.meshgrid(freq, freq)
        self.rho = c.wavelength * np.hypot(fx, fy) / c.NA_out
        self.theta = np.arctan2(fy, fx)
        self.pupil = (self.rho >= c.epsilon) & (self.rho <= 1)
        self.Z = basis(self.indices, self.rho, self.theta, c.epsilon, c.Nz)
        # Never leak configured test-truth coefficients into the inverse model.
        # A scene containing further nonzero terms must expose those as unknowns.
        all_c = configured_coeffs(c)
        fixed = [j for j, v in enumerate(all_c) if v and j not in self.indices]
        if fixed:
            raise ValueError(f"configured nonzero modes {fixed} are absent from fit indices; expand indices")
        self.fixed_wavefront = np.zeros((n, n))
        z = c.z_grating - c.shift_z_grating
        self.pre_g = np.exp(1j * np.pi * (self.x_img**2 + self.y_img**2) / (c.wavelength * z))
        self.pre_g *= tilt_phase(self.x_img, self.y_img, c.tilted_angle_grating,
                                 c.tilt_axis_x_grating, c.tilt_axis_y_grating, k)
        self.post_g = np.exp(1j * k * z) / (1j * c.wavelength * z) * np.exp(
            1j * np.pi * (self.x_g**2 + self.y_g**2) / (c.wavelength * z))
        self.cg = CenteredCZT(n, ng, self.dx_img, self.x_g[0, 0] / (c.wavelength * z),
                              c.dx_grating / (c.wavelength * z))
        self.pre_cam = np.exp(1j * np.pi * (self.x_g**2 + self.y_g**2) / (c.wavelength * c.z_cam))
        self.pre_cam *= tilt_phase(self.x_g, self.y_g, c.tilted_angle_cam,
                                   c.tilt_axis_x_cam, c.tilt_axis_y_cam, k)
        self.post_cam = np.exp(1j * k * c.z_cam) / (1j * c.wavelength * c.z_cam) * np.exp(
            1j * np.pi * (self.x_cam**2 + self.y_cam**2) / (c.wavelength * c.z_cam))
        self.cc = CenteredCZT(ng, n, c.dx_grating, self.x_cam[0, 0] / (c.wavelength * c.z_cam),
                              c.dx_cam / (c.wavelength * c.z_cam))
        period = int(c.grating_period / c.dx_grating)
        size = int(c.grating_period * c.duty_cycle / 100 / c.dx_grating)
        xx, yy = np.meshgrid(np.arange(ng), np.arange(ng))
        if c.grating_type == "Square":
            start = (period - size) // 2
            self.grating = ((xx % period >= start) & (xx % period < start + size)
                            & (yy % period >= start) & (yy % period < start + size)).astype(float)
        else:
            self.grating = ((abs(xx % period - period // 2) + abs(yy % period - period // 2)) <= period // 2).astype(float)
        self.grating = np.roll(self.grating, (int(c.shift_x_grating / c.dx_grating),
                                              int(c.shift_y_grating / c.dx_grating)), axis=(1, 0))
        if backend not in ('separable', 'czt'):
            raise ValueError('unknown propagation backend')
        self.backend = backend if c.grating_type == 'Square' else 'czt'
        if self.backend == 'separable':
            self._prepare_separable()
        self.source_spectra = None
        self.n_forward = self.n_jacobian = 0
        self.seconds = 0.0

    def _prepare_separable(self):
        """Exact composition of separable Fresnel kernels and square mask.

        Columns are the reference propagation of unit pupil-frequency modes.
        This changes evaluation order, not the sampled propagation operator.
        """
        c = self.config
        ys, xs = np.nonzero(self.pupil)
        self.fy_indices = np.arange(ys.min(), ys.max()+1)
        self.fx_indices = np.arange(xs.min(), xs.max()+1)
        self.pupil_box = self.pupil[np.ix_(self.fy_indices, self.fx_indices)]
        z = c.z_grating - c.shift_z_grating
        period = int(c.grating_period / c.dx_grating)
        size = int(c.grating_period * c.duty_cycle / 100 / c.dx_grating)
        start = (period-size)//2
        index = np.arange(self.ng)
        aperture = ((index % period >= start) & (index % period < start+size)).astype(float)
        matrices = []
        for axis, frequencies in enumerate((self.fx_indices, self.fy_indices)):
            xi = self.x_img[0] if axis == 0 else self.y_img[:, 0]
            xg = self.x_g[0] if axis == 0 else self.y_g[:, 0]
            xc = self.x_cam[0] if axis == 0 else self.y_cam[:, 0]
            gnorm = np.hypot(c.tilt_axis_x_grating, c.tilt_axis_y_grating)
            cnorm = np.hypot(c.tilt_axis_x_cam, c.tilt_axis_y_cam)
            gp = (-c.tilt_axis_y_grating if axis == 0 else c.tilt_axis_x_grating)/gnorm
            cp = (-c.tilt_axis_y_cam if axis == 0 else c.tilt_axis_x_cam)/cnorm
            pre = np.exp(1j*np.pi*xi**2/(c.wavelength*z)
                         -1j*self.k*np.tan(np.deg2rad(c.tilted_angle_grating))*gp*xi)
            middle = np.exp(1j*np.pi*xg**2/(c.wavelength*z) +
                            1j*np.pi*xg**2/(c.wavelength*c.z_cam)
                            -1j*self.k*np.tan(np.deg2rad(c.tilted_angle_cam))*cp*xg)
            shift = int((c.shift_x_grating if axis == 0 else c.shift_y_grating)/c.dx_grating)
            middle *= np.roll(aperture, shift)
            seeds = np.zeros((len(frequencies), self.n), complex)
            seeds[np.arange(len(frequencies)), frequencies] = 1
            image = fft.fftshift(fft.ifft(fft.ifftshift(seeds, axes=-1), axis=-1), axes=-1)
            g = self.cg.plan(image*pre, axis=-1)*self.cg.factor
            cam = self.cc.plan(g*middle, axis=-1)*self.cc.factor
            cam *= np.exp(1j*np.pi*xc**2/(c.wavelength*c.z_cam))
            matrices.append(np.ascontiguousarray(cam.T))
        self.Bx, self.By = matrices
        self.separable_prefactor = (np.exp(1j*self.k*z)/(1j*c.wavelength*z) *
                                   np.exp(1j*self.k*c.z_cam)/(1j*c.wavelength*c.z_cam))

    def _transfer_spectrum(self, spectrum, rows=None):
        if rows is None:
            return (self.By @ spectrum @ self.Bx.T) * self.separable_prefactor
        yy, xx = np.divmod(rows, self.n)
        unique_y, inverse = np.unique(yy, return_inverse=True)
        temporary = self.By[unique_y] @ spectrum
        return np.sum(temporary[..., inverse, :] * self.Bx[xx], axis=-1)*self.separable_prefactor

    def source_field(self, source):
        c = self.config
        field = np.zeros((self.n, self.n), complex)
        field[c.N_lattice // 2, c.N_lattice // 2] = 1
        field = np.roll(field, (source[0] * c.N_lattice, source[1] * c.N_lattice), axis=(1, 0))
        if c.error_mode == "OFF":
            return field
        frequency = 2 * np.pi * fft.fftfreq(self.n, self.dx_obj)
        kx, ky = np.meshgrid(frequency, frequency)
        valid = kx**2 + ky**2 <= self.k**2
        kz = np.sqrt(np.maximum(self.k**2 - kx**2 - ky**2, 0))
        phase = tilt_phase(self.x_obj, self.y_obj, c.tilted_angle, c.tilt_axis_x, c.tilt_axis_y, self.k)
        u_ref = fft.ifft2(fft.fft2(field * phase) * np.where(valid, np.exp(1j * kz * (1e-3 - c.shift_z)), 0))
        return fft.ifft2(fft.fft2(u_ref) * np.where(valid, np.exp(-1j * kz * 1e-3), 0))

    def prepare_sources(self):
        if self.source_spectra is None:
            self.source_spectra = np.stack([
                fft.fftshift(fft.fft2(fft.ifftshift(self.source_field(s))))[self.pupil]
                for s in self.config.sources
            ])

    def propagate_image(self, field, plane="camera"):
        if plane not in ('camera', 'grating'):
            raise ValueError("plane must be camera or grating")
        grating_field = self.cg(field * self.pre_g) * self.post_g
        if plane == "grating":
            return grating_field
        return self.cc(grating_field * self.grating * self.pre_cam) * self.post_cam

    def evaluate(self, coeffs, jacobian=False, rows=None):
        coeffs = np.asarray(coeffs, float)
        if coeffs.shape != (len(self.indices),) or not np.isfinite(coeffs).all():
            raise ValueError("invalid coefficients")
        start = perf_counter()
        self.prepare_sources()
        H = self.pupil * np.exp(2j * np.pi * (np.tensordot(coeffs, self.Z, axes=1) + self.fixed_wavefront))
        if rows is not None:
            rows = np.asarray(rows, int)
            if rows.ndim != 1 or np.any((rows < 0) | (rows >= self.n*self.n)):
                raise ValueError('invalid camera sample rows')
        if self.backend == 'separable':
            shape = (self.n, self.n) if rows is None else (len(rows),)
            image = np.zeros(shape)
            derivative = np.zeros((len(coeffs), *shape)) if jacobian else None
            Hb = H[np.ix_(self.fy_indices, self.fx_indices)]
            for b in range(0, len(self.config.sources), self.batch_sources):
                spectra = self.source_spectra[b:b+self.batch_sources]
                spec = np.zeros((len(spectra), *self.pupil_box.shape), complex)
                spec[:, self.pupil_box] = spectra
                spec *= Hb
                u = self._transfer_spectrum(spec, rows)
                image += np.sum(abs(u)**2, axis=0)
                if jacobian:
                    for j, z in enumerate(self.Z):
                        zb = z[np.ix_(self.fy_indices, self.fx_indices)]
                        du = self._transfer_spectrum(spec*(2j*np.pi*zb), rows)
                        derivative[j] += 2*np.real(np.sum(np.conj(u)*du, axis=0))
            self.n_forward += 1
            self.n_jacobian += int(jacobian)
            self.seconds += perf_counter()-start
            return (image, derivative) if jacobian else image
        image = np.zeros((self.n, self.n))
        derivative = np.zeros((len(coeffs), self.n, self.n)) if jacobian else None
        with fft.set_workers(self.workers):
            for b in range(0, len(self.config.sources), self.batch_sources):
                spectra = self.source_spectra[b:b + self.batch_sources]
                spec = np.zeros((len(spectra), self.n, self.n), complex)
                spec[:, self.pupil] = spectra * H[self.pupil]
                u_img = fft.fftshift(fft.ifft2(fft.ifftshift(spec, axes=(-2, -1))), axes=(-2, -1))
                u = self.propagate_image(u_img)
                image += np.sum(abs(u)**2, axis=0)
                if jacobian:
                    # One mode at a time bounds memory independently of mode count.
                    for j, z in enumerate(self.Z):
                        du_img = fft.fftshift(fft.ifft2(fft.ifftshift(spec * (2j * np.pi * z), axes=(-2, -1))), axes=(-2, -1))
                        du = self.propagate_image(du_img)
                        derivative[j] += 2 * np.real(np.sum(np.conj(u) * du, axis=0))
        self.n_forward += 1
        self.n_jacobian += int(jacobian)
        self.seconds += perf_counter() - start
        if rows is not None:
            image = image.ravel()[rows]
            if jacobian:
                derivative = derivative.reshape(len(coeffs), -1)[:, rows]
        return (image, derivative) if jacobian else image

    def wavefront(self, coeffs):
        return np.tensordot(coeffs, self.Z, axes=1) + self.fixed_wavefront
