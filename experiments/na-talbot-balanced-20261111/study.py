"""Balanced NA sweep, original sampled scalar operator and independent fitting.

Run only on SJTU in the private Conda environment. Every numerical invocation
requires an immutable, remotely verified commit and an explicit large-run flag.
"""
import argparse
from dataclasses import dataclass, fields
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
from types import SimpleNamespace

import numpy as np
from scipy import fft, ndimage
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lsi.basis import basis
from lsi.config import SystemConfig
from lsi.execution import exclusive_study, interrupt_on_termination
from lsi.forward import CenteredCZT, DiffractionModel, grid, tilt_phase
from lsi.fourier import FourierSettings, spectral_window

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
INDICES = tuple(range(1, 36))
WRITE_LOCK = threading.RLock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n")
        tmp.replace(path)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array)**2)))


@dataclass(frozen=True)
class BalancedConfig(SystemConfig):
    source_kind: str = "multi64"
    camera_multiplier: float = 4.5

    @property
    def camera_n(self):
        pixels = self.n*self.camera_multiplier
        assert pixels == round(pixels) and round(pixels) % 2 == 0
        return round(pixels)

    @property
    def sources(self):
        if self.source_kind == "single":
            return ((self.N_super_lattice//2, self.N_super_lattice//2),)
        return tuple((i, j) for i in range(3, 11) for j in range(3, 11))


class ExtendedCameraModel(DiffractionModel):
    """Same discrete propagation with independent image and camera dimensions."""
    def __init__(self, c, indices=INDICES, **kwargs):
        super().__init__(c, indices, **kwargs)
        self.object_n = c.n
        self.prepare_sources()
        self.n = c.camera_n
        self.x_cam, self.y_cam = grid(self.n, c.dx_cam)
        self.post_cam = np.exp(1j*self.k*c.z_cam)/(1j*c.wavelength*c.z_cam)*np.exp(
            1j*np.pi*(self.x_cam**2+self.y_cam**2)/(c.wavelength*c.z_cam))
        self.cc = CenteredCZT(self.ng, self.n, c.dx_grating,
            self.x_cam[0, 0]/(c.wavelength*c.z_cam), c.dx_cam/(c.wavelength*c.z_cam))

    def source_field(self, source):
        c = self.config
        field = np.zeros((c.n, c.n), complex)
        field[c.N_lattice//2, c.N_lattice//2] = 1
        field = np.roll(field, (source[0]*c.N_lattice, source[1]*c.N_lattice), axis=(1, 0))
        if c.error_mode == "OFF":
            return field
        frequency = 2*np.pi*fft.fftfreq(c.n, self.dx_obj)
        kx, ky = np.meshgrid(frequency, frequency)
        valid = kx*kx+ky*ky <= self.k*self.k
        kz = np.sqrt(np.maximum(self.k*self.k-kx*kx-ky*ky, 0))
        phase = tilt_phase(self.x_obj, self.y_obj, c.tilted_angle,
                            c.tilt_axis_x, c.tilt_axis_y, self.k)
        field = fft.ifft2(fft.fft2(field*phase)*np.where(
            valid, np.exp(1j*kz*(1e-3-c.shift_z)), 0))
        return fft.ifft2(fft.fft2(field)*np.where(valid, np.exp(-1j*kz*1e-3), 0))

    def _prepare_separable(self):
        c = self.config
        ys, xs = np.nonzero(self.pupil)
        self.fy_indices = np.arange(ys.min(), ys.max()+1)
        self.fx_indices = np.arange(xs.min(), xs.max()+1)
        self.pupil_box = self.pupil[np.ix_(self.fy_indices, self.fx_indices)]
        z = c.z_grating-c.shift_z_grating
        period = int(c.grating_period/c.dx_grating)
        size = int(c.grating_period*c.duty_cycle/100/c.dx_grating)
        start = (period-size)//2
        ix = np.arange(self.ng)
        aperture = ((ix % period >= start) & (ix % period < start+size)).astype(float)
        xc = (np.arange(c.camera_n)-c.camera_n//2)*c.dx_cam
        camera_czt = CenteredCZT(self.ng, c.camera_n, c.dx_grating,
                                xc[0]/(c.wavelength*c.z_cam), c.dx_cam/(c.wavelength*c.z_cam))
        matrices = []
        for axis, frequencies in enumerate((self.fx_indices, self.fy_indices)):
            xi = self.x_img[0] if axis == 0 else self.y_img[:, 0]
            xg = self.x_g[0] if axis == 0 else self.y_g[:, 0]
            gp = (-c.tilt_axis_y_grating if axis == 0 else c.tilt_axis_x_grating)/np.hypot(
                c.tilt_axis_x_grating, c.tilt_axis_y_grating)
            cp = (-c.tilt_axis_y_cam if axis == 0 else c.tilt_axis_x_cam)/np.hypot(
                c.tilt_axis_x_cam, c.tilt_axis_y_cam)
            pre = np.exp(1j*np.pi*xi**2/(c.wavelength*z)-
                1j*self.k*np.tan(np.deg2rad(c.tilted_angle_grating))*gp*xi)
            middle = np.exp(1j*np.pi*xg**2/(c.wavelength*z)+
                1j*np.pi*xg**2/(c.wavelength*c.z_cam)-
                1j*self.k*np.tan(np.deg2rad(c.tilted_angle_cam))*cp*xg)
            shift = int((c.shift_x_grating if axis == 0 else c.shift_y_grating)/c.dx_grating)
            middle *= np.roll(aperture, shift)
            seeds = np.zeros((len(frequencies), c.n), complex)
            seeds[np.arange(len(frequencies)), frequencies] = 1
            image = fft.fftshift(fft.ifft(fft.ifftshift(seeds, axes=-1), axis=-1), axes=-1)
            grating = self.cg.plan(image*pre, axis=-1)*self.cg.factor
            camera = camera_czt.plan(grating*middle, axis=-1)*camera_czt.factor
            camera *= np.exp(1j*np.pi*xc**2/(c.wavelength*c.z_cam))
            matrices.append(np.ascontiguousarray(camera.T))
        self.Bx, self.By = matrices
        self.separable_prefactor = np.exp(1j*self.k*z)/(1j*c.wavelength*z)*np.exp(
            1j*self.k*c.z_cam)/(1j*c.wavelength*c.z_cam)


def config(na, kind="multi64", refine=False):
    p = PROTOCOL
    q = na / p["base_na"]
    period = p["base_period_m"] / q
    distance = p["base_camera_distance_m"] / q**2
    talbot = 2 * period**2 / p["wavelength_m"]
    radius = talbot * distance / (distance-talbot)
    shift = p["base_shift_z_m"] / q**2
    return BalancedConfig(
        wavelength=p["wavelength_m"], error_mode="ON",
        N_lattice=p["n_lattice"], N_super_lattice=p["n_super_lattice"],
        super_lattice_period=p["base_source_pitch_m"]/q,
        NA_in=p["na_in"], NA_out=na, shift_z=shift,
        grating_period=period, z_grating=radius+.04*shift,
        z_cam=distance, dx_grating=float(np.nextafter(
            p["base_grating_dx_m"]/q/(2 if refine else 1), 0)),
        up_sample_grating=p["up_sample_grating"]*(2 if refine else 1),
        dx_cam=p["base_camera_dx_m"]/q, source_kind=kind,
        camera_multiplier=p["camera_multiplier"])


def geometry(c):
    m = c.f2/c.f1
    radius = c.z_grating-m*m*c.shift_z
    effective = radius*c.z_cam/(radius+c.z_cam)
    pixels = int(c.grating_period/c.dx_grating)
    source_max = 3.5*c.super_lattice_period*m
    image_half = c.n*c.super_lattice_period/c.N_lattice*m/2
    grating_half = c.n*c.up_sample_grating*c.dx_grating/2
    camera_half = c.camera_n*c.dx_cam/2
    na_image = c.NA_out/m
    angular_radius = na_image/np.sqrt(1-na_image**2)
    center_camera = source_max*(radius+c.z_cam)/radius
    pupil_camera = (radius+c.z_cam)*na_image
    shear = c.wavelength*(radius+c.z_cam)/c.grating_period
    return dict(
        scale_q=c.NA_out/.02, epsilon=c.epsilon, image_na_sine=na_image,
        paraxial_radius_m=radius, effective_distance_m=effective,
        talbot_length_m=2*c.grating_period**2/c.wavelength,
        talbot_ratio=effective/(2*c.grating_period**2/c.wavelength),
        grating_period_pixels=pixels, aperture_pixels=int(
            c.grating_period*.5/c.dx_grating),
        source_pitch_object_m=c.super_lattice_period,
        image_full_width_m=2*image_half, grating_full_width_m=2*grating_half,
        camera_full_width_m=2*camera_half,
        camera_grid_n=c.camera_n, image_grid_n=c.n,
        camera_source_center_axis_m=center_camera,
        camera_pupil_radius_m=pupil_camera, camera_shear_m=shear,
        image_source_margin_m=image_half-source_max,
        grating_paraxial_margin_m=grating_half-source_max-radius*na_image,
        grating_sine_angle_margin_m=grating_half-source_max-radius*angular_radius,
        camera_paraxial_first_order_margin_m=camera_half-center_camera-pupil_camera-shear,
        camera_sine_angle_first_order_margin_m=(
            camera_half-center_camera-(radius+c.z_cam)*angular_radius-shear),
        image_sampling_na_limit=c.wavelength/(2*c.super_lattice_period/c.N_lattice),
        grating_camera_alias_period_m=c.wavelength*c.z_cam/c.dx_grating,
        camera_alias_margin_m=c.wavelength*c.z_cam/(2*c.dx_grating)-camera_half-grating_half,
        fundamental_bins=c.camera_n*c.dx_cam*radius/(c.grating_period*(radius+c.z_cam)),
        shear_pixels=shear/c.dx_cam,
        grating_quadratic_phase_nyquist_ratio=(
            c.dx_grating/c.wavelength*(2*grating_half*(1/c.z_grating+1/c.z_cam)
                                       +2*camera_half/c.z_cam)))


def truth_for(model, random_seed=None):
    if random_seed is None:
        c = np.zeros(35)
        c[np.array(PROTOCOL["truth_indices"])-1] = PROTOCOL["truth_weights"]
    else:
        c = np.random.default_rng(random_seed).normal(size=35)
    return c * PROTOCOL["truth_rms_waves"]/rms(model.wavefront(c)[model.pupil])


def metrics(model, c, truth):
    error = model.wavefront(c-truth)
    values = error[model.pupil]
    radial = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = model.pupil & (model.rho >= lo) & (model.rho <= hi)
        radial.append(dict(low=float(lo), high=float(hi), pixels=int(mask.sum()),
                           rms_waves=rms(error[mask]) if mask.any() else None))
    common = model.pupil & (model.rho*model.config.NA_out <= .02+1e-14)
    return dict(raw_rms_waves=rms(values), raw_pv_waves=float(np.ptp(values)),
                raw_rms_nm=rms(values)*model.config.wavelength*1e9,
                relative_rms=rms(values)/rms(model.wavefront(truth)[model.pupil]),
                outer10_rms_waves=rms(error[model.pupil & (model.rho >= .9)]),
                common_na002_rms_waves=rms(error[common]),
                coefficient_max_error_waves=float(abs(c-truth).max()),
                coefficients=c.tolist(), truth=truth.tolist(),
                evaluation_pixels=int(model.pupil.sum()), radial=radial)


class Monitor:
    def __init__(self, output, stage):
        self.output = Path(output)
        self.started = time.monotonic()
        self.state = dict(status="running", stage=stage)
        self.stop = threading.Event()
        self.high_temperature_samples = 0

    def event(self, name, **values):
        row = dict(timestamp_utc=utc(), event=name, **values)
        with WRITE_LOCK:
            with (self.output/"events.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row), flush=True)

    def update(self, **values):
        import psutil
        with WRITE_LOCK:
            self.state.update(values, timestamp_utc=utc(),
                              elapsed_seconds=time.monotonic()-self.started,
                              rss_gib=psutil.Process().memory_info().rss/1024**3,
                              peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                self.state.update(gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3,
                    gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3,
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            save(self.output/"status.json", self.state)

    def loop(self):
        while not self.stop.wait(30):
            self.update()
            with WRITE_LOCK:
                with (self.output/"eta.jsonl").open("a") as f:
                    f.write(json.dumps(self.state, allow_nan=False)+"\n")
            if self.state["rss_gib"] > PROTOCOL["limits"]["rss_gib"]:
                self.event("memory_limit", rss_gib=self.state["rss_gib"])
                os.kill(os.getpid(), signal.SIGTERM)
            if self.state.get("gpu_reserved_gib", 0) > 16:
                self.event("gpu_memory_limit", gpu_reserved_gib=self.state["gpu_reserved_gib"])
                os.kill(os.getpid(), signal.SIGTERM)
            if "gpu_reserved_gib" in self.state:
                device = os.environ["CUDA_VISIBLE_DEVICES"]
                try:
                    temperature = int(subprocess.check_output(["nvidia-smi",
                        "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
                        "--id="+device], text=True, timeout=5).strip())
                except (subprocess.SubprocessError, ValueError) as error:
                    self.event("gpu_temperature_read_failed", error=repr(error))
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                self.update(gpu_temperature_c=temperature)
                self.high_temperature_samples = self.high_temperature_samples+1 if temperature >= 80 else 0
                if self.high_temperature_samples >= 3:
                    self.event("sustained_gpu_temperature", temperature_c=temperature)
                    os.kill(os.getpid(), signal.SIGTERM)
            if time.monotonic()-self.started > PROTOCOL["limits"]["max_runtime_hours"]*3600:
                self.event("runtime_limit")
                os.kill(os.getpid(), signal.SIGTERM)

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.update()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, typ, value, tb):
        self.stop.set()
        self.update(status="failed" if typ else "complete", error=repr(value) if typ else None)


def edge_energy(field):
    intensity = abs(field)**2
    b = max(1, round(.05*field.shape[-1]))
    return float(1-intensity[b:-b, b:-b].sum()/intensity.sum())


def pilot(out, monitor):
    """Check the preceding worst sources before another all-source preflight."""
    results = []
    for na in (PROTOCOL["na_values"][0], PROTOCOL["na_values"][-1]):
        c = config(na)
        g = geometry(c)
        assert g["grating_quadratic_phase_nyquist_ratio"] < 1
        assert g["camera_alias_margin_m"] > 0
        monitor.update(stage="pilot", na_out=na)
        model = ExtendedCameraModel(c, INDICES, batch_sources=1, workers=1)
        truth = truth_for(model)
        for source in ((3, 3), (4, 4), (7, 7)):
            number = c.sources.index(source)
            for label, coeff in (("zero", np.zeros(35)), ("truth", truth)):
                spectrum = np.zeros(model.pupil_box.shape, complex)
                spectrum[model.pupil_box] = model.source_spectra[number]*np.exp(
                    2j*np.pi*model.wavefront(coeff)[model.pupil])
                camera = model._transfer_spectrum(spectrum)
                value = edge_energy(camera)
                row = dict(na_out=na, source=list(source), field=label,
                           camera_outer5_energy=value, geometry=g)
                results.append(row)
                save(out/"pilot.json", dict(rows=results, passed=False))
                monitor.event("pilot_camera_edge", **row)
                assert value < PROTOCOL["limits"]["camera_outer5_energy"]
                del camera, spectrum
        del model
        gc.collect()
    save(out/"pilot.json", dict(rows=results, passed=True,
        scope="Selected sources at extreme NA values; complete preflight still required."))


def preflight(out, reference, monitor):
    """Every source, zero/truth fields; exact sampled reference parity."""
    sys.path.insert(0, str(reference/"BUV0725"))
    from point2plate import Stage1
    from obj_sys_ft import Stage2_ab
    from grating2cam import Stage3
    raw = json.loads((reference/"BUV0725/config_file.json").read_text())
    rows = []
    for na in PROTOCOL["na_values"]:
        folder = out/f"na{na:.3f}"
        folder.mkdir(exist_ok=True)
        c = config(na)
        g = geometry(c)
        assert abs(g["talbot_ratio"]-1) < 1e-11
        assert g["grating_period_pixels"] == round(
            PROTOCOL["base_period_m"]/PROTOCOL["base_grating_dx_m"])
        assert abs(g["grating_period_pixels"]*c.dx_grating/c.grating_period-1) < 1e-12
        for key in ("grating_sine_angle_margin_m",
                    "camera_sine_angle_first_order_margin_m", "camera_alias_margin_m"):
            assert g[key] > 0, (na, key, g[key])
        assert g["image_sampling_na_limit"] > c.NA_out
        assert g["grating_quadratic_phase_nyquist_ratio"] < 1
        save(folder/"config.json", c.to_dict())
        save(folder/"geometry.json", g)
        monitor.update(stage="preflight", na_out=na, source=0)
        model = ExtendedCameraModel(c, INDICES, batch_sources=1, workers=1)
        truth = truth_for(model)
        model.prepare_sources()
        # The unmasked incident intensity is band limited by the finite image
        # support. Its diagnostic grid need not resolve the aperture edges.
        probe_n = PROTOCOL["grating_intensity_probe_n"]
        probe_dx = model.ng*c.dx_grating/probe_n
        z = c.z_grating-c.shift_z_grating
        intensity_nyquist_dx = c.wavelength*z/(2*model.object_n*model.dx_img)
        assert probe_dx < intensity_nyquist_dx
        probe_x = (np.arange(probe_n)-probe_n//2)*probe_dx
        probe = CenteredCZT(model.object_n, probe_n, model.dx_img,
            probe_x[0]/(c.wavelength*z), probe_dx/(c.wavelength*z))
        px, py = np.meshgrid(probe_x, probe_x)
        probe_post = np.exp(1j*model.k*z)/(1j*c.wavelength*z)*np.exp(
            1j*np.pi*(px*px+py*py)/(c.wavelength*z))
        del px, py
        convergence = []
        errors = []
        for number, source in enumerate(c.sources):
            for label, coeff in (("zero", np.zeros(35)), ("truth", truth)):
                spectrum = np.zeros((model.object_n, model.object_n), complex)
                spectrum[model.pupil] = (model.source_spectra[number] *
                    np.exp(2j*np.pi*model.wavefront(coeff)[model.pupil]))
                image = fft.fftshift(fft.ifft2(fft.ifftshift(spectrum)))
                grating = probe(image*model.pre_g)*probe_post
                box = np.zeros(model.pupil_box.shape, complex)
                box[model.pupil_box] = spectrum[model.pupil]
                camera = model._transfer_spectrum(box)
                if number == 0:
                    dense = model.propagate_image(image, plane="grating")
                    difference = abs(edge_energy(dense)-edge_energy(grating))
                    convergence.append(dict(field=label, dense_n=model.ng,
                        probe_n=probe_n, probe_dx_m=probe_dx,
                        intensity_nyquist_dx_m=intensity_nyquist_dx,
                        edge_energy_absolute_difference=difference))
                    assert difference < PROTOCOL["limits"]["grating_edge_probe_absolute_error"]
                    del dense
                errors.append(dict(source=list(source), field=label,
                    image_outer5_energy=edge_energy(image),
                    grating_outer5_energy=edge_energy(grating),
                    camera_outer5_energy=edge_energy(camera),
                    finite=bool(np.isfinite(camera).all()),
                    camera_energy=float(np.sum(abs(camera)**2))))
                if number == 0 and label == "zero":
                    start = (model.n-c.n)//2
                    first_zero = abs(camera[start:start+c.n, start:start+c.n])**2
                del spectrum, image, grating, camera
            monitor.update(source=number+1, source_total=64)
            if (number+1) % 8 == 0:
                save(folder/"source_edges.json", errors)
                monitor.event("source_edges", na_out=na, completed=number+1)
        maxima = {k: max(r[k] for r in errors) for k in (
            "image_outer5_energy", "grating_outer5_energy", "camera_outer5_energy")}
        save(folder/"source_edges.json", errors)
        save(folder/"edge_maxima.json", maxima)
        save(folder/"grating_probe_convergence.json", convergence)
        for key, value in maxima.items():
            assert value < PROTOCOL["limits"][key], (na, key, value)
        period = g["grating_period_pixels"]
        cell = model.grating[:period, :period].copy()
        f = fft.fftfreq(period, c.dx_grating)
        fx, fy = np.meshgrid(f, f)
        replica = fft.ifft2(fft.fft2(cell)*np.exp(
            -1j*np.pi*c.wavelength*g["effective_distance_m"]*(fx*fx+fy*fy)))
        talbot_error = float(np.linalg.norm(replica-cell)/np.linalg.norm(cell))
        assert talbot_error < 1e-8
        del model, probe, probe_post, box
        gc.collect()
        cfg = SimpleNamespace(**{**raw, **c.to_dict()})
        s1 = Stage1(cfg)
        s2 = Stage2_ab(cfg, mode="reference")
        s3 = Stage3(cfg, mode="camera")
        u = s3.forward(s2.forward(s1.forward(nx=3, ny=3)))
        parity = float(np.linalg.norm(abs(u)**2-first_zero)/np.linalg.norm(first_zero))
        assert parity < PROTOCOL["limits"]["reference_intensity_relative_l2"], parity
        result = dict(na_out=na, geometry=g, maxima=maxima,
                      reference_intensity_relative_l2=parity,
                      periodic_cell_field_error=talbot_error,
                      grating_intensity_probe_n=probe_n,
                      grating_probe_convergence=convergence,
                      source_fields_checked=len(errors))
        rows.append(result)
        save(out/"preflight.json", dict(passed=True, completed_na=len(rows), rows=rows))
        monitor.event("preflight_na_complete", na_out=na, **maxima, parity=parity)
        del s1, s2, s3, u, first_zero
        gc.collect()


class Q1Features:
    """Adjacent-order reference-subtracted small-phase Fourier features."""
    def __init__(self, c, zero, compact=True):
        self.c = c
        n = c.camera_n
        self.geometry = g = geometry(c)
        self.carrier = round(g["fundamental_bins"])
        self.shear = round(g["shear_pixels"])
        s = self.settings = FourierSettings(
            carrier_x=self.carrier, carrier_y=self.carrier,
            half_width=int(.4*g["fundamental_bins"]), window_sigma=2,
            phase_sigma=10*10e-6/PROTOCOL["base_camera_dx_m"], regression="lstsq", alpha=0)
        x = (np.arange(n)-n//2)*c.dx_cam
        radius = int(np.ceil(g["camera_pupil_radius_m"]/c.dx_cam))+abs(self.shear)
        radius += int(np.ceil(4*s.phase_sigma))+2
        start, stop = max(0, n//2-radius), min(n, n//2+radius+1)
        self.roi = (slice(start, stop), slice(start, stop)) if compact else (slice(None), slice(None))
        x = x[self.roi[0]]
        xx, yy = np.meshgrid(x, x)
        rho = np.hypot(xx, yy)/((g["paraxial_radius_m"]+c.z_cam)*c.NA_out*c.f1/c.f2)
        theta = np.arctan2(yy, xx)
        pupil = (rho >= c.epsilon) & (rho <= 1)
        self.masks = [pupil & np.roll(pupil, self.shear, axis) for axis in (1, 0)]
        rows = [[], []]
        for j in INDICES:
            z = basis((j,), rho, theta, c.epsilon, c.Nz)[0]
            for k, axis in enumerate((1, 0)):
                rows[k].append((np.roll(z, self.shear, axis)-z)[self.masks[k]])
        A = np.vstack([np.stack(row, axis=1) for row in rows])
        self.inverse = np.linalg.pinv(A)
        self.basis_rank = int(np.linalg.matrix_rank(A))
        self.basis_condition = float(np.linalg.cond(A))
        del rows, A
        self.windows, self.references = [], []
        spec = self.spectrum(zero)
        for k, axis in enumerate((1, 0)):
            center = (n//2+(self.carrier if axis == 1 else 0),
                      n//2+(self.carrier if axis == 0 else 0))
            win = spectral_window(n, center, s)
            self.windows.append(win)
            self.references.append(fft.ifft2(fft.ifftshift(
                np.roll(spec*win, -self.carrier, axis)))[self.roi].copy())

    @staticmethod
    def spectrum(image):
        image = image-image.min()
        return fft.fftshift(fft.fft2(image/image.max()))

    def __call__(self, image):
        spec = self.spectrum(image)
        values, maxima, branch = [], [], []
        for k, axis in enumerate((1, 0)):
            signal_field = fft.ifft2(fft.ifftshift(
                np.roll(spec*self.windows[k], -self.carrier, axis)))[self.roi].copy()
            phase = np.where(self.masks[k], np.angle(
                signal_field*np.conj(self.references[k])), 0)
            maxima.append(float(abs(phase[self.masks[k]]).max()))
            branch.append(float(np.mean(abs(phase[self.masks[k]]) > 2.8)))
            values.append(ndimage.gaussian_filter(
                phase, self.settings.phase_sigma)[self.masks[k]])
        return self.inverse@np.concatenate(values), dict(
            max_absolute_phase_rad=maxima, near_branch_fraction=branch,
            small_phase_valid=bool(max(maxima) < 2.8))


def calibrate(model, feature, folder, monitor):
    path = folder/"calibration_features.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    amplitude = PROTOCOL["calibration_amplitude_waves"]
    for j in range(35):
        for sign in (1, -1):
            key = f"j{j+1:02d}{'p' if sign > 0 else 'm'}"
            if key in data:
                continue
            coeff = np.zeros(35)
            coeff[j] = sign*amplitude
            started = time.monotonic()
            image = model.evaluate(coeff)
            f, diagnostics = feature(image)
            data[key] = dict(features=f.tolist(), diagnostics=diagnostics,
                             image_sha256=digest(image),
                             seconds=time.monotonic()-started)
            save(path, data)
            monitor.update(stage="calibration", completed=len(data), total=70,
                eta_remaining_seconds=(70-len(data))*np.mean([v["seconds"] for v in data.values()]))
    response = np.stack([(np.array(data[f"j{j:02d}p"]["features"])-
        np.array(data[f"j{j:02d}m"]["features"]))/(2*amplitude) for j in range(1, 36)], axis=1)
    singular = np.linalg.svd(response, compute_uv=False)
    errors = []
    for j in range(35):
        for sign in (1, -1):
            key = f"j{j+1:02d}{'p' if sign > 0 else 'm'}"
            target = np.zeros(35)
            target[j] = sign*amplitude
            errors.append(float(abs(np.linalg.lstsq(
                response, data[key]["features"], rcond=None)[0]-target).max()))
    summary = dict(rank=int(np.linalg.matrix_rank(response)),
        condition=float(np.linalg.cond(response)), singular_values=singular.tolist(),
        training_max_coefficient_error=max(errors),
        all_small_phase_valid=all(v["diagnostics"]["small_phase_valid"] for v in data.values()),
        seconds=sum(v["seconds"] for v in data.values()),
        dZ_rank=feature.basis_rank, dZ_condition=feature.basis_condition,
        settings=feature.settings.to_dict(), shear_pixels=feature.shear,
        feature_grid_shape=list(feature.masks[0].shape), camera_grid_shape=[model.n, model.n])
    np.savez_compressed(folder/"calibration.npz", response=response, singular_values=singular)
    save(folder/"calibration.json", summary)
    return response, summary


def fit_lm(model, zero, observed, folder, name, monitor):
    g = geometry(model.config)
    sampling_radius = (g["camera_pupil_radius_m"]+np.sqrt(2)*g["camera_source_center_axis_m"]+
                       2*g["camera_shear_m"])
    candidates = np.flatnonzero(
        np.hypot(model.x_cam, model.y_cam).ravel() <= sampling_radius)
    rows = np.sort(np.random.default_rng(PROTOCOL["lm_sample_seed"]).choice(
        candidates, PROTOCOL["lm_samples"], replace=False))
    np.save(folder/"lm_sample_rows.npy", rows)
    scale = float(zero.max())
    target = observed.ravel()[rows]
    a, b = np.linalg.lstsq(np.column_stack((zero.ravel()[rows]/scale,
                                          np.ones(len(rows)))), target, rcond=None)[0]
    x0 = np.r_[np.zeros(35), np.log(max(a, 1e-8)), b]
    cached_x, cached = None, None
    trajectory = []
    started = time.monotonic()

    def evaluate(x):
        nonlocal cached_x, cached
        if cached_x is None or not np.array_equal(x, cached_x):
            image, derivative = model.evaluate(x[:35], jacobian=True, rows=rows)
            gain = np.exp(x[35])
            prediction = image/scale
            residual = gain*prediction+x[36]-target
            jacobian = np.column_stack((gain*derivative.T/scale,
                                       gain*prediction, np.ones(len(rows))))
            cached_x, cached = x.copy(), (residual, jacobian)
            trajectory.append(dict(cost=float(residual@residual), coefficients=x[:35].tolist(),
                gain=float(gain), background=float(x[36]), timestamp_utc=utc()))
            save(folder/f"lm_{name}_trajectory.json", trajectory)
            monitor.update(stage="lm", condition=name, evaluations=len(trajectory),
                           current_cost=trajectory[-1]["cost"], eta_remaining_seconds=None)
        return cached

    solution = least_squares(lambda x: evaluate(x)[0], x0,
        jac=lambda x: evaluate(x)[1], method="lm", x_scale="jac",
        max_nfev=PROTOCOL["lm_max_nfev"], ftol=1e-10, xtol=1e-10, gtol=1e-10)
    residual, jac = evaluate(solution.x)
    nuisance = jac[:, 35:]
    projected = jac[:, :35]-nuisance@np.linalg.lstsq(nuisance, jac[:, :35], rcond=None)[0]
    singular = np.linalg.svd(projected, compute_uv=False)
    prediction = np.exp(solution.x[35])*model.evaluate(solution.x[:35])/scale+solution.x[36]
    result = dict(success=bool(solution.success), status=int(solution.status),
        message=str(solution.message), nfev=solution.nfev, njev=solution.njev,
        full_intensity_nrmse=rms(prediction-observed)/rms(observed),
        sampled_intensity_rmse=rms(residual), fitted_gain=float(np.exp(solution.x[35])),
        fitted_background=float(solution.x[36]), seconds=time.monotonic()-started,
        projected_rank=int(np.sum(singular > singular[0]*1e-12)),
        projected_condition=float(singular[0]/singular[-1]),
        projected_singular_values=singular.tolist(), sample_count=len(rows),
        sampling_radius_m=sampling_radius, sampling_candidate_pixels=len(candidates),
        sample_sha256=digest(rows), initial_coefficients=[0.]*35)
    return solution.x[:35], result


def run(out, preflight_dir, monitor, device="cpu"):
    checked = json.loads((preflight_dir/"preflight.json").read_text())
    assert checked["passed"] and checked["completed_na"] == 4
    assert json.loads((preflight_dir/"version.json").read_text())["protocol"] == PROTOCOL
    results = json.loads((out/"summary.json").read_text())["results"] if (out/"summary.json").exists() else []
    for na in PROTOCOL["na_values"]:
        for kind in ("single", "multi64"):
            c = config(na, kind)
            folder = out/f"na{na:.3f}_{kind}"
            folder.mkdir(exist_ok=True)
            monitor.update(na_out=na, source_kind=kind, stage="setup")
            model = ExtendedCameraModel(c, INDICES, batch_sources=1, workers=1)
            if device == "cuda":
                from gpu_backend import GPUCameraModel
                model = GPUCameraModel(model)
            truth = truth_for(model)
            zero = model.evaluate(np.zeros(35))
            raw = model.evaluate(truth)
            clean = raw/raw.max()
            np.save(folder/"zero.npy", zero)
            np.save(folder/"clean.npy", clean)
            save(folder/"config.json", c.to_dict())
            save(folder/"geometry.json", geometry(c))
            feature = Q1Features(c, zero)
            response, calibration = calibrate(model, feature, folder, monitor)
            validation = []
            for seed in (1807, 1808, 1809):
                path = folder/f"holdout_{seed}.json"
                if path.exists():
                    validation.append(json.loads(path.read_text()))
                    continue
                holdout = truth_for(model, seed)
                f, diag = feature(model.evaluate(holdout))
                estimate = np.linalg.lstsq(response, f, rcond=None)[0]
                row = dict(seed=seed, diagnostics=diag, **metrics(model, estimate, holdout))
                save(path, row)
                validation.append(row)
            save(folder/"holdout.json", validation)
            conditions = [("clean", None, None)]
            conditions += [(f"40db_seed{seed}", 40, seed) for seed in PROTOCOL["noise_40db_seeds"]]
            conditions += [(f"30db_seed{seed}", 30, seed) for seed in PROTOCOL["noise_30db_seeds"]]
            for name, db, seed in conditions:
                sigma = 0 if db is None else 10**(-db/20)*rms(clean)
                observed = clean if db is None else clean+sigma*np.random.default_rng(seed).normal(size=clean.shape)
                np.save(folder/f"observation_{name}.npy", observed)
                observed_hash = digest(observed)
                for method in ("ft", "lm"):
                    path = folder/f"{method}_{name}.json"
                    if path.exists():
                        continue
                    started = time.monotonic()
                    if method == "ft":
                        f, diagnostics = feature(observed)
                        estimate = np.linalg.lstsq(response, f, rcond=None)[0]
                        prediction = model.evaluate(estimate)/raw.max()
                        diagnostics.update(calibration_rank=calibration["rank"],
                            calibration_condition=calibration["condition"],
                            full_intensity_nrmse=rms(prediction-observed)/rms(observed),
                            seconds=time.monotonic()-started)
                    else:
                        estimate, diagnostics = fit_lm(model, zero, observed, folder, name, monitor)
                    row = dict(na_out=na, source_kind=kind, source_count=len(c.sources),
                        method=method, condition=name, snr_db=db, seed=seed, noise_sigma=sigma,
                        observed_sha256=observed_hash, diagnostics=diagnostics,
                        **metrics(model, estimate, truth))
                    save(path, row)
                    results.append(row)
                    save(out/"summary.json", dict(results=results, result_count=len(results),
                         expected_result_count=112, timestamp_utc=utc()))
                    monitor.event("result", na_out=na, source_kind=kind, method=method,
                        condition=name, rms_waves=row["raw_rms_waves"], completed=len(results), total=112)
            del model, feature, zero, clean, raw, observed
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("design", "pilot", "preflight", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-check", type=Path)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert head == args.expected_commit
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                    "--untracked-files=no"], text=True)
    assert not dirty, dirty
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/"version.json", dict(commit=head, protocol=PROTOCOL,
        command=sys.argv, hostname=os.uname().nodename, start_utc=utc(),
        python=sys.version, numpy=np.__version__,
        source_hashes={str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in [Path(__file__), HERE/"protocol.json",
                                HERE/"gpu_backend.py", *sorted((REPO/"lsi").glob("*.py"))]},
        device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    if args.stage == "design":
        save(args.output/"design.json", [
            dict(na_out=na, config=config(na).to_dict(), geometry=geometry(config(na)))
            for na in PROTOCOL["na_values"]])
        return
    assert args.allow_large_run
    assert os.uname().nodename == "sjtu-gxyserver", "Production computation must run on SJTU"
    with exclusive_study(), interrupt_on_termination(), threadpool_limits(
            limits=PROTOCOL["blas_threads"]), fft.set_workers(PROTOCOL["fft_threads"]):
        with Monitor(args.output, args.stage) as monitor:
            if args.stage == "pilot":
                pilot(args.output, monitor)
            elif args.stage == "preflight":
                assert args.reference
                preflight(args.output, args.reference, monitor)
            else:
                assert args.preflight
                if args.device == "cuda":
                    assert args.gpu_check
                    check = json.loads((args.gpu_check/"validation.json").read_text())
                    assert check["passed"] and check["commit"] == head
                    assert check["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES")
                    assert check["protocol_sha256"] == hashlib.sha256(
                        (HERE/"protocol.json").read_bytes()).hexdigest()
                    assert check["gpu_backend_sha256"] == hashlib.sha256(
                        (HERE/"gpu_backend.py").read_bytes()).hexdigest()
                    import torch
                    assert check["torch"] == torch.__version__ and check["cuda"] == torch.version.cuda
                    save(args.output/"gpu-version.json", dict(
                        validation_sha256=hashlib.sha256(
                            (args.gpu_check/"validation.json").read_bytes()).hexdigest(),
                        validation_directory=str(args.gpu_check),
                        torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=check["gpu_name"], cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                        matrix_dtype="complex128", real_dtype="float64",
                        gpu_reserved_limit_gib=16, temperature_stop=">=80C for three 30s samples"))
                run(args.output, args.preflight, monitor, args.device)


if __name__ == "__main__":
    main()
