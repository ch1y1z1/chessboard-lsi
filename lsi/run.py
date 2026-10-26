"""Three experiment workflows. Numerical methods live in forward/fourier/direct."""
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
import shutil

import numpy as np

from .config import FitConfig, Scene, SystemConfig
from .direct import fit_direct
from .evaluate import wavefront_metrics
from .fourier import FourierCalibration, FourierExtractor, FourierSettings, calibration_images
from .forward import DiffractionModel
from .io import peak_memory_mib, provenance, save_arrays, write_json, write_rows


def settings_for(system):
    """Reference defaults scaled to the explicitly selected camera grid."""
    ratio = system.n * system.dx_cam / .012
    return FourierSettings(carrier_x=round(44 * ratio), carrier_y=round(44 * ratio),
                           half_width=max(4, round(20 * ratio)),
                           phase_sigma=10 * system.n / 1200, regression='lstsq', alpha=0)


class Experiment:
    def __init__(self, output, system, scene=Scene(), fit=FitConfig(), fourier=None,
                 calibration_amplitude=.02):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f'{output} is not empty; choose a new output directory')
        self.output.mkdir(parents=True, exist_ok=True)
        self.system, self.scene, self.fit = system, scene, fit
        self.settings = fourier or settings_for(system)
        self.amplitude = calibration_amplitude
        self.rows = []
        self.started = perf_counter()
        self.manifest = {'system': asdict(system), 'scene': asdict(scene), 'fit': asdict(fit),
                         'fourier': asdict(self.settings), 'calibration_amplitude': self.amplitude,
                         'source_count': len(system.sources), 'status': 'running', **provenance()}
        write_json(self.output / 'manifest.json', self.manifest)

    def initialize(self):
        self.model = self.make_model(self.system)
        self.truth = self.scene.coefficients_for(self.fit.indices, self.system.Nz)
        if self.scene.rms is not None:
            rms = np.std(self.model.wavefront(self.truth)[self.model.pupil])
            if rms <= 0 or self.scene.rms <= 0:
                raise ValueError('positive scene RMS requires a nonzero wavefront')
            self.truth *= self.scene.rms / rms
        self.manifest['truth_coefficients_waves'] = self.truth.tolist()
        write_json(self.output / 'manifest.json', self.manifest)
        # A snapshot allows reproduction even if the working tree is uncommitted.
        root = Path(__file__).parent.parent
        for relative in self.manifest['source_sha256']:
            target = self.output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)

    def make_model(self, system):
        return DiffractionModel(system, self.fit.indices, batch_sources=1, workers=1)

    def observe(self, model, name, seed=None):
        raw = model.evaluate(self.truth)
        image = raw / raw.max()
        seed = self.scene.seed if seed is None else seed
        if self.scene.snr_db is not None:
            image += np.random.default_rng(seed).normal(0, 10**(-self.scene.snr_db/20), image.shape)
        self.observation = f'{name}_observation.npz'
        save_arrays(self.output / self.observation, raw_intensity=raw, observed=image,
                            truth=self.truth, pupil=model.pupil,
                            truth_wavefront=model.wavefront(self.truth))
        self.manifest.setdefault('observations', {})[name] = {
            'file': self.observation, 'system': asdict(model.config), 'noise_seed': seed}
        return image

    def calibrate(self, model, settings, name, images=None):
        started = perf_counter()
        print(f'calibrate {name}: {len(self.fit.indices)} modes', flush=True)
        images = calibration_images(model, self.amplitude) if images is None else images
        extractor = FourierExtractor(model.config, settings)
        cal = FourierCalibration.fit(extractor, images, model.indices, self.amplitude)
        # Independent mixture: not used to fit the response or initialize the direct method.
        truth = np.random.default_rng(9721).normal(size=len(model.indices))
        truth *= self.amplitude / (2 * np.linalg.norm(truth))
        observed = model.evaluate(truth)
        fitted = cal.reconstruct(extractor, observed)['coeffs_waves']
        save_arrays(self.output / f'{name}_calibration.npz', baseline=cal.baseline,
                            response=cal.response, reference_image=cal.reference_image,
                            indices=cal.indices, amplitude=cal.amplitude)
        save_arrays(self.output / f'{name}_validation.npz', observed=observed,
                            truth=truth, fitted=fitted)
        digest = sha256()
        for image in images:
            digest.update(np.ascontiguousarray(image).tobytes())
        write_json(self.output / f'{name}_calibration.json', {
            'system': asdict(model.config), 'settings': asdict(settings),
            'units': 'object-pupil waves', 'training': 'zero and independent symmetric single modes',
            'amplitude': self.amplitude, 'training_image_sha256': digest.hexdigest(),
            'seconds': perf_counter() - started,
            'condition': float(np.linalg.cond(cal.response)),
            'validation': wavefront_metrics(model, fitted, truth)})
        return cal, images

    def record(self, case, method, coeffs=None, **details):
        row = {'case': case, 'method': method, 'status': 'ok',
               'observation': self.observation, **details}
        if coeffs is not None:
            row.update(coeffs_waves=np.asarray(coeffs).tolist(),
                       **wavefront_metrics(self.model, coeffs, self.truth))
        self.rows.append(row)
        write_rows(self.output, self.rows)
        print(case, method, row.get('rmse_waves', row['status']), flush=True)
        return row

    def solve_direct(self, model, image, case, **context):
        fit = self.fit
        result = fit_direct(model, image, starts=fit.starts, seed=fit.seed, samples=fit.samples,
                            max_nfev=fit.max_nfev, screen_initials=fit.screen_initials, progress=True)
        np.save(self.output / f'{case}_sample_rows.npy', result.pop('sample_rows'))
        for trial in result['trials']:
            trial['metrics'] = wavefront_metrics(self.model, trial['coeffs'], self.truth)
        write_json(self.output / f'{case}_direct_trials.json', result)
        best = result['trials'][result['best_trial']]
        return self.record(case, 'direct', result['coeffs'], converged=best['converged'],
                           nfev=best['nfev'], seconds=result['seconds'],
                           full_intensity_rms=best['full_intensity_rms'], gain=best['gain'],
                           background=best['background'], **context)

    def solve_fourier(self, model, image, cal, settings, case, **context):
        try:
            result = cal.reconstruct(FourierExtractor(model.config, settings), image)
            return self.record(case, 'fourier_calibrated', result['coeffs_waves'],
                               settings=asdict(settings), calibration_settings=cal.settings,
                               calibration_condition=result['calibration_condition'],
                               phase_residual_rms=result['phase_residual_rms'],
                               centers=result['centers'], coverage=result['coverage'],
                               window_metrics=result['window_metrics'], seconds=result['seconds'], **context)
        except (ValueError, np.linalg.LinAlgError) as error:
            return self.record(case, 'fourier_calibrated', status='failed', reason=str(error),
                               settings=asdict(settings), **context)

    def compare(self, options):
        cal, _ = self.calibrate(self.model, self.settings, 'baseline')
        for seed in options.get('noise_seeds', [self.scene.seed]):
            case = 'baseline' if 'noise_seeds' not in options else f'noise_seed{seed}'
            image = self.observe(self.model, case, seed)
            self.solve_fourier(self.model, image, cal, self.settings, case, noise_seed=seed)
            self.solve_direct(self.model, image, case, noise_seed=seed)

    def regions(self, options):
        image = self.observe(self.model, 'regions')
        cal, images = self.calibrate(self.model, self.settings, 'nominal')
        self.solve_direct(self.model, image, 'regions_direct_reference')
        cases = [('nominal', 'nominal', self.settings)]
        widths = options.get('widths', [8, 12, 16, 20, 24, 28, 32])
        cases += [(f'width_{w:g}', 'width', replace(self.settings, half_width=w)) for w in widths]
        cases += [(f'center_{x:g}_{y:g}', 'center', replace(self.settings, offset_x=x, offset_y=y))
                  for x, y in product(options.get('offsets', [-3, -2, -1, 0, 1, 2, 3]), repeat=2)]
        cases += [(f'shape_{s}', 'shape', replace(self.settings, shape=s))
                  for s in options.get('shapes', ['rect', 'circle', 'gaussian'])]
        cases += [(f'aspect_{a:g}', 'aspect', replace(self.settings, shape='ellipse', aspect=a))
                  for a in options.get('aspects', [.5, 2])]
        cases += [(f'erosion_{e}', 'erosion', replace(self.settings, erosion=e))
                  for e in options.get('erosions', [2, 4, 8, 12])]
        cases += [(f'amplitude_{t:g}', 'amplitude', replace(self.settings, amplitude_threshold=t))
                  for t in options.get('thresholds', [.1, .3, .5])]
        if options.get('subpixel', False):
            cases.append(('subpixel', 'subpixel', replace(self.settings, subpixel=True)))
        recalibrate = options.get('recalibrate', False)
        for case, scan, settings in cases:
            selected = cal
            if recalibrate and settings != self.settings:
                selected, _ = self.calibrate(self.model, settings, case, images)
            self.solve_fourier(self.model, image, selected, settings, case, scan=scan,
                               calibration_policy='per_window' if recalibrate else 'frozen_nominal')

    def angles(self, options):
        if set(options.get('geometry', ['known', 'ignored'])) - {'known', 'ignored'}:
            raise ValueError('geometry must contain known and/or ignored')
        settings = {mode: replace(self.settings, center_mode=mode)
                    for mode in options.get('center_modes', ['fixed', 'auto'])}
        nominal = {mode: self.calibrate(self.model, setting, f'nominal_{mode}')[0]
                   for mode, setting in settings.items()}
        zero = self.observe(self.model, 'angle_zero')
        self.solve_direct(self.model, zero, 'angle_zero', geometry='known_and_ignored',
                          component='zero', axis='none', angle_deg=0)
        for mode, setting in settings.items():
            self.solve_fourier(self.model, zero, nominal[mode], setting, f'angle_zero_{mode}',
                               geometry='known_and_ignored', component='zero', axis='none', angle_deg=0)
        for component, axis, angle in product(options.get('components', ['camera', 'grating']),
                                               options.get('axes', ['x', 'y']),
                                               options.get('angles', [-.25, .25])):
            if angle == 0:
                continue
            model = self.make_model(self.system.with_angle(component, angle, axis))
            name = f'{component}_{axis}_{angle:g}'
            image = self.observe(model, name)
            context = {'component': component, 'axis': axis, 'angle_deg': angle,
                       'image_change_relative_l2': float(np.linalg.norm(image-zero)/np.linalg.norm(zero))}
            for geometry in options.get('geometry', ['known', 'ignored']):
                inverse = model if geometry == 'known' else self.model
                self.solve_direct(inverse, image, f'{name}_{geometry}', geometry=geometry, **context)
                for mode, setting in settings.items():
                    # Each center protocol has its own response: a full-protocol comparison.
                    cal = self.calibrate(model, setting, f'{name}_known_{mode}')[0] if geometry == 'known' else nominal[mode]
                    self.solve_fourier(inverse, image, cal, setting, f'{name}_{geometry}_{mode}',
                                       geometry=geometry, **context)

    def finish(self, status='complete', reason=None):
        if status == 'complete' and any(r['status'] == 'failed' for r in self.rows):
            status = 'complete_with_failures'
        self.manifest.update(status=status, row_count=len(self.rows), seconds=perf_counter()-self.started,
                             peak_memory_mib=peak_memory_mib())
        if reason is not None:
            self.manifest['stop_reason'] = reason
        write_json(self.output / 'manifest.json', self.manifest)
        write_rows(self.output, self.rows)
