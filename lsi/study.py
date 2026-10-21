"""Reproducible diffraction comparisons: python -m lsi.study --help."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import resource
import subprocess
from time import perf_counter

import numpy as np

from .annular import configured_coeffs, mode_labels
from .config import DiffractionConfig
from .direct import fit_direct
from .execution import exclusive_study, interrupt_on_termination, requires_large_run
from .fourier import (FourierCalibration, FourierExtractor, FourierSettings, calibration_images)
from .metrics import wavefront_metrics
from .propagation import DiffractionModel


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=serializable, indent=2, allow_nan=False) + '\n')


def write_rows(directory, rows):
    if not rows:
        return
    write_json(directory/'results.json', rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (directory/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, tuple, dict)) else v
                         for k, v in row.items()} for row in rows)


def peak_memory_mib():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(size / (1024**2 if platform.system() == 'Darwin' else 1024))


def provenance(args, config, indices):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    package_versions = {p: importlib.metadata.version(p) for p in
                        ('numpy', 'scipy', 'scikit-image', 'scikit-learn', 'matplotlib')}
    root = Path(__file__).parent
    return {'arguments': vars(args), 'config': config.to_dict(), 'indices': indices,
            'mode_labels': [mode_labels(config.Nz)[i] for i in indices],
            'git_commit': commit, 'python': platform.python_version(), 'platform': platform.platform(),
            'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
            'dependencies': package_versions, 'source_count': len(config.sources),
            'tilt_model': 'reference small-angle input-field phase ramp, not rotated-plane propagation',
            'profile_note': ('custom configuration; see config' if args.config else
                             'reference sampling' if args.profile in ('reference', 'retrieval') else
                             'coarser numerical sampling; not a substitute for reference-grid results')}


def settings_for(config, regression='lstsq'):
    # Same physical camera FOV in reference and quick profiles. The test scene
    # has a smaller FOV and hence a different number of carrier FFT bins.
    ratio = config.n * config.dx_cam / .012
    return FourierSettings(carrier_x=round(44*ratio), carrier_y=round(44*ratio),
                           half_width=max(4, round(20*ratio)),
                           phase_sigma=10*config.n/1200,
                           regression=regression, alpha=.02 if regression == 'Lasso' else 0)


class Study:
    def __init__(self, args, config=None):
        self.args = args
        self.output = Path(args.output or f'output/diffraction/{args.command}')
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output/'manifest.json').exists():
            raise FileExistsError(f'{self.output} already contains a run; choose a new --output directory')
        self.config = config if config is not None else (
            DiffractionConfig.from_json(args.config) if args.config else
            DiffractionConfig.profile(args.profile))
        self.indices = (tuple(range(1, (self.config.Nz+1)**2)) if args.modes == 'all' else
                        tuple(int(v) for v in args.modes.split(',')))
        self.rows = []
        self.started = perf_counter()
        self.manifest = provenance(args, self.config, self.indices)
        self.manifest['status'] = 'running'
        self.manifest['execution_policy'] = {'exclusive_process': True,
                                             'large_run_opt_in': args.allow_large_run}
        write_json(self.output/'manifest.json', self.manifest)

    def initialize(self):
        args = self.args
        self.model = self.make_model(self.config)
        self.truth = configured_coeffs(self.config)[list(self.indices)]
        if args.truth_rms is not None:
            wf = self.model.wavefront(self.truth) - self.model.fixed_wavefront
            rms = np.std(wf[self.model.pupil])
            if rms <= 0 or args.truth_rms <= 0:
                raise ValueError('truth RMS requires a nonzero configured wavefront and positive RMS')
            self.truth *= args.truth_rms / rms
        self.settings = settings_for(self.config)
        self.cal_images = None
        self.calibration = None
        self.validation_cache = {}
        self.manifest['truth_coefficients_waves'] = self.truth.tolist()
        self.manifest['fourier_settings'] = self.settings.to_dict()
        self.manifest['status'] = 'running'
        write_json(self.output/'manifest.json', self.manifest)
        snapshot = self.output/'source_snapshot'/'lsi'
        snapshot.mkdir(parents=True, exist_ok=True)
        for source in Path(__file__).parent.glob('*.py'):
            (snapshot/source.name).write_bytes(source.read_bytes())
        for name in ('pyproject.toml', 'uv.lock', 'experiment.py'):
            path = Path(__file__).parent.parent/name
            if path.exists():
                (snapshot.parent/name).write_bytes(path.read_bytes())

    def make_model(self, config):
        return DiffractionModel(config, self.indices, self.args.batch_sources, self.args.workers,
                                backend=self.args.backend)

    def observe(self, model, name, noise_id=0):
        raw = model.evaluate(self.truth)
        image = raw / raw.max()
        if self.args.snr_db is not None:
            rng = np.random.default_rng(self.args.seed + noise_id)
            image += rng.normal(0, 10**(-self.args.snr_db/20), image.shape)
        np.savez_compressed(self.output/f'{name}_observation.npz', raw_intensity=raw,
                            observed=image, truth=self.truth, pupil=model.pupil,
                            truth_wavefront=model.wavefront(self.truth))
        return image

    def calibrate(self, model, settings, name, images=None):
        print(f'calibrate {name}: {len(self.indices)} modes, {len(model.config.sources)} sources', flush=True)
        started = perf_counter()
        cache_used = None
        if images is None and self.args.calibration_cache:
            path = Path(self.args.calibration_cache)
            meta = json.loads(path.with_suffix('.json').read_text())
            cache_config = json.dumps(meta['config'], sort_keys=True)
            current_config = json.dumps(model.config.to_dict(), sort_keys=True)
            if cache_config == current_config:
                with np.load(path) as saved:
                    if tuple(saved['indices']) != self.indices or float(saved['amplitude']) != self.args.calibration_amplitude:
                        raise ValueError('calibration cache has different modes or amplitude')
                    images = [im.copy() for im in saved['training_images']]
                cache_used = {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}
                print(f'reused independent calibration images from {path}', flush=True)
            elif name in ('nominal','baseline'):
                raise ValueError('calibration cache physical config differs from nominal model')
        images = calibration_images(model, self.args.calibration_amplitude) if images is None else images
        cal = FourierCalibration.fit(FourierExtractor(model.config, settings), images,
                                      self.indices, self.args.calibration_amplitude)
        # A mixed wavefront is held out from the single-mode calibration.
        rng = np.random.default_rng(9721)
        validation_truth = rng.normal(size=len(self.indices))
        validation_truth *= self.args.calibration_amplitude / (2 * np.linalg.norm(validation_truth))
        validation_key = json.dumps(model.config.to_dict(), sort_keys=True)
        if validation_key not in self.validation_cache:
            self.validation_cache[validation_key] = model.evaluate(validation_truth)
        validation_image = self.validation_cache[validation_key]
        validation = cal.reconstruct(FourierExtractor(model.config, settings), validation_image)
        validation_error = wavefront_metrics(model, validation['coeffs_waves'], validation_truth,
                                             self.args.recovery_tolerance)
        np.savez_compressed(self.output/f'{name}_calibration_validation.npz',
                            observed=validation_image, truth=validation_truth,
                            fitted=validation['coeffs_waves'])
        np.savez_compressed(self.output/f'{name}_calibration.npz', baseline=cal.baseline,
                            response=cal.response, reference_image=cal.reference_image,
                            training_images=np.asarray(images),
                            indices=cal.indices, amplitude=cal.amplitude)
        write_json(self.output/f'{name}_calibration.json', {
            'config': cal.config, 'settings': cal.settings, 'amplitude_waves': cal.amplitude,
            'units': 'object-pupil waves', 'phase_reference': 'independent zero-aberration image',
            'condition': float(np.linalg.cond(cal.response)), 'seconds': perf_counter()-started,
            'training_image_cache': cache_used,
            'held_out_mixed_validation': validation_error,
            'training_cases': 'zero and symmetric single modes; no mixed test wavefronts'})
        return cal, images

    def append(self, name, method, coeffs=None, **extra):
        row = {'case': name, 'method': method, 'status': 'ok', **extra}
        if coeffs is not None:
            row.update(coeffs_waves=np.asarray(coeffs).tolist(),
                       **wavefront_metrics(self.model, coeffs, self.truth, self.args.recovery_tolerance))
        self.rows.append(row)
        write_rows(self.output, self.rows)
        print(name, method, 'rmse', row.get('rmse_waves'), 'status', row.get('status', 'ok'), flush=True)
        return row

    def solve_direct(self, infer_model, image, name, **context):
        result = fit_direct(infer_model, image, starts=self.args.starts, seed=self.args.seed,
                             samples=self.args.samples, max_nfev=self.args.max_nfev, progress=True,
                             screen_initials=self.args.screen_initials)
        np.save(self.output/f'{name}_sample_rows.npy', result.pop('sample_rows'))
        write_json(self.output/f'{name}_direct_trials.json', result)
        best = result['trials'][result['best_trial']]
        row = self.append(name, 'direct', result['coeffs'], seconds=result['seconds'],
                          converged=best['converged'], full_intensity_rms=best['full_intensity_rms'],
                          starts=self.args.starts, nfev=best['nfev'], gain=best['gain'],
                          background=best['background'], **context)
        return result, row

    def solve_fourier(self, infer_config, image, cal, settings, name, **context):
        started = perf_counter()
        try:
            extractor = FourierExtractor(infer_config, settings)
            result = cal.reconstruct(extractor, image, details=True)
            np.savez_compressed(self.output/f'{name}_fourier.npz', **{
                key: result[key] for key in ('wrapped', 'amplitudes', 'windows', 'masks', 'spectrum_log',
                                            'raw_coeffs', 'coeffs_waves')})
            row = self.append(name, 'fourier_calibrated', result['coeffs_waves'],
                              coverage=result['coverage'], phase_residual_rms=result['phase_residual_rms'],
                              centers=result['centers'], seconds=perf_counter()-started,
                              window_metrics=result['window_metrics'],
                              calibration_condition=result['calibration_condition'],
                              calibration_settings=cal.settings,
                              settings=settings.to_dict(), **context)
            return result, row
        except (ValueError, np.linalg.LinAlgError) as error:
            self.append(name, 'fourier_calibrated', status='failed', reason=str(error),
                        settings=settings.to_dict(), **context)
            return None, None

    def compare(self):
        image = self.observe(self.model, 'baseline')
        # The original reference method is preserved and labelled in raw units.
        raw = FourierExtractor(self.config, replace(self.settings, regression='Lasso', alpha=.02)).extract(image)
        write_json(self.output/'baseline_fourier_original.json', raw)
        self.append('baseline', 'fourier_original', units=raw['units'],
                    phase_residual_rms=raw['phase_residual_rms'], coverage=raw['coverage'],
                    seconds=raw['seconds'], status='raw_units_not_pupil_waves')
        self.calibration, self.cal_images = self.calibrate(self.model, self.settings, 'baseline')
        ft, _ = self.solve_fourier(self.config, image, self.calibration, self.settings, 'baseline')
        direct, _ = self.solve_direct(self.model, image, 'baseline')
        self.plot_comparison(image, ft, direct)
        self.finish()

    def regions(self):
        image = self.observe(self.model, 'regions')
        cal, images = self.calibrate(self.model, self.settings, 'nominal')
        self.solve_direct(self.model, image, 'regions_direct_reference')
        cases = [('nominal', self.settings)]
        cases += [(f'width_{w:g}', replace(self.settings, half_width=w)) for w in self.args.widths]
        cases += [(f'center_{x:g}_{y:g}', replace(self.settings, offset_x=x, offset_y=y))
                  for x, y in itertools.product(self.args.offsets, repeat=2)]
        cases += [(f'shape_{shape}', replace(self.settings, shape=shape)) for shape in self.args.shapes]
        cases += [(f'ellipse_aspect_{a:g}', replace(self.settings, shape='ellipse', aspect=a))
                  for a in self.args.aspects]
        cases += [(f'erosion_{e}', replace(self.settings, erosion=e)) for e in self.args.erosions]
        cases += [(f'amplitude_{a:g}', replace(self.settings, amplitude_threshold=a)) for a in self.args.thresholds]
        cases += [('subpixel', replace(self.settings, subpixel=True))]
        # Choose an operational width on an independent mixed validation scene;
        # the test image and its truth never enter this choice.
        rng = np.random.default_rng(5891)
        selection_truth = rng.normal(size=len(self.indices))
        wf = self.model.wavefront(selection_truth)
        selection_truth *= self.args.selection_validation_rms / np.std(wf[self.model.pupil])
        selection_image = self.model.evaluate(selection_truth)
        scores = []
        for width in self.args.widths:
            settings = replace(self.settings, half_width=width)
            selection_cal = cal
            if self.args.recalibrate_windows:
                selection_cal, _ = self.calibrate(self.model, settings, f'validation_width_{width:g}', images=images)
            fit = selection_cal.reconstruct(FourierExtractor(self.config, settings), selection_image)
            score = wavefront_metrics(self.model, fit['coeffs_waves'], selection_truth)
            scores.append({'half_width': width, 'validation_rmse_waves': score['rmse_waves']})
        selected = min(scores, key=lambda row: row['validation_rmse_waves'])['half_width']
        write_json(self.output/'width_selection.json', {'validation_truth': selection_truth,
                   'validation_seed': 5891, 'validation_rms': self.args.selection_validation_rms,
                   'scores': scores, 'selected_width': selected, 'test_truth_used': False})
        np.savez_compressed(self.output/'width_selection_validation.npz', observed=selection_image,
                            truth=selection_truth)
        cases.append(('selected_width', replace(self.settings, half_width=selected)))
        if self.args.cross_angles:
            # A separate cross sweep records angle × half-width. Nominal (zero
            # angle) calibration stays fixed, so this is an ignored-angle study.
            for angle in self.args.cross_angles:
                config = self.config.with_angle('camera', angle)
                angled_image = self.observe(self.make_model(config), f'cross_{angle:g}')
                for w in self.args.widths:
                    self.solve_fourier(self.config, angled_image, cal,
                                       replace(self.settings, half_width=w),
                                       f'cross_{angle:g}_width_{w:g}', angle_deg=angle, scan='angle_width')
        for name, settings in cases:
            selected_cal = cal
            try:
                if self.args.recalibrate_windows:
                    selected_cal, _ = self.calibrate(self.model, settings, name, images=images)
            except (ValueError, np.linalg.LinAlgError) as error:
                self.append(name, 'fourier_calibrated', status='failed', stage='calibration', reason=str(error))
                continue
            self.solve_fourier(self.config, image, selected_cal, settings, name,
                               scan=('width' if name.startswith('width') else
                                     'center' if name.startswith('center') else name.split('_')[0]),
                               calibration_policy=('per_window' if self.args.recalibrate_windows else 'frozen_nominal'))
        self.plot_sweeps('regions')
        self.finish()

    def angles(self):
        nominal_cal, _ = self.calibrate(self.model, self.settings, 'nominal')
        zero_image = self.observe(self.model, 'angle_zero')
        zero_direct, _ = self.solve_direct(self.model, zero_image, 'angle_zero', component='zero', angle_deg=0,
                                            axis='none', geometry='known_and_ignored')
        self.solve_fourier(self.config, zero_image, nominal_cal, self.settings, 'angle_zero',
                           component='zero', angle_deg=0, axis='none', geometry='known_and_ignored')
        for component, axis, angle in itertools.product(self.args.components, self.args.axes, self.args.angles):
            if angle == 0:
                continue
            config = self.config.with_angle(component, angle, axis)
            model = self.make_model(config)
            name = f'{component}_{axis}_{angle:g}'
            image = self.observe(model, name)
            image_change = float(np.linalg.norm(image-zero_image)/np.linalg.norm(zero_image))
            context = {'component': component, 'axis': axis, 'angle_deg': angle,
                       'image_change_relative_l2': image_change}
            if component == 'object' and self.args.snr_db is None and image_change < 1e-9:
                # A verified null control of the supplied single-pixel source
                # model, not an approximation replacing the source sum.
                write_json(self.output/f'{name}_null_control.json', context)
                self.append(name, 'direct', zero_direct['coeffs'], status='verified_null_control_reuse',
                             geometry='known_and_ignored', **context)
                self.solve_fourier(config, image, nominal_cal, self.settings, name,
                                   geometry='known_and_ignored', **context)
                continue
            if 'known' in self.args.geometry:
                cal, _ = self.calibrate(model, self.settings, name)
                self.solve_direct(model, image, name+'_known', geometry='known', **context)
                for center_mode in self.args.center_modes:
                    self.solve_fourier(config, image, cal, replace(self.settings, center_mode=center_mode),
                                       name+'_known_'+center_mode, geometry='known', **context)
            if 'ignored' in self.args.geometry:
                self.solve_direct(self.model, image, name+'_ignored', geometry='ignored', **context)
                for center_mode in self.args.center_modes:
                    self.solve_fourier(self.config, image, nominal_cal,
                                       replace(self.settings, center_mode=center_mode),
                                       name+'_ignored_'+center_mode, geometry='ignored', **context)
        self.plot_sweeps('angles')
        self.finish()

    def plot_comparison(self, image, ft, direct):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        truth = self.model.wavefront(self.truth)
        direct_w = self.model.wavefront(direct['coeffs'])
        ft_w = self.model.wavefront(ft['coeffs_waves']) if ft else np.full_like(truth, np.nan)
        vmin, vmax = np.min(truth[self.model.pupil]), np.max(truth[self.model.pupil])
        finite_errors = np.r_[(ft_w-truth)[self.model.pupil], (direct_w-truth)[self.model.pupil]]
        finite_errors = finite_errors[np.isfinite(finite_errors)]
        error_limit = max(float(np.max(abs(finite_errors))), 1e-12)
        yy, xx = np.nonzero(self.model.pupil)
        margin = max(2, int(.05 * max(np.ptp(xx), np.ptp(yy))))
        crop = np.s_[max(0, yy.min()-margin):yy.max()+margin+1,
                     max(0, xx.min()-margin):xx.max()+margin+1]
        fig, ax = plt.subplots(2, 3, figsize=(13, 8))
        entries = [(image, 'Observed diffraction intensity', None, None, False),
                   (truth, 'True pupil wavefront / waves', vmin, vmax, True),
                   (ft_w, 'Calibrated Fourier / waves', vmin, vmax, True),
                   (direct_w, 'Direct least squares / waves', vmin, vmax, True),
                   (ft_w-truth, 'Fourier error / waves', -error_limit, error_limit, True),
                   (direct_w-truth, 'Direct error / waves', -error_limit, error_limit, True)]
        for a, (data, title, lo, hi, masked) in zip(ax.flat, entries):
            data = np.where(self.model.pupil, data, np.nan) if masked else data
            if masked:
                data = data[crop]
            im = a.imshow(data, origin='lower', cmap='RdBu_r' if masked else 'gray', vmin=lo, vmax=hi)
            a.set_title(title); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, shrink=.8)
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(self.output/f'comparison.{suffix}', dpi=150)
        plt.close(fig)

    def plot_sweeps(self, kind):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        valid = [r for r in self.rows if 'rmse_waves' in r]
        if kind == 'angles':
            parts = sorted({r.get('component') for r in valid if r.get('component') not in ('object','zero',None)})
            directions = sorted({r.get('axis') for r in valid if r.get('axis') not in ('none',None)})
            if not parts:
                parts = ['object']
            fig, axes = plt.subplots(len(parts), max(len(directions),1), squeeze=False,
                                      figsize=(6*max(len(directions),1), 4*len(parts)+1))
            handles = {}
            for i, part in enumerate(parts):
                for j, axis in enumerate(directions or ['x']):
                    ax = axes[i,j]
                    selected = [r for r in valid if r.get('component') == part and r.get('axis') == axis]
                    groups = {}
                    for row in selected:
                        key = (row['method'], row.get('geometry'), row.get('settings',{}).get('center_mode','fixed'))
                        groups.setdefault(key,[]).append(row)
                    for (method, geometry, center), rows in groups.items():
                        rows.sort(key=lambda row: row['angle_deg'])
                        label = ('Direct' if method == 'direct' else 'Fourier '+center)+' / '+str(geometry)
                        color = 'C1' if method == 'direct' else ('C0' if center == 'fixed' else 'C2')
                        line, = ax.plot([r['angle_deg'] for r in rows], [r['rmse_waves'] for r in rows],
                                        'o--' if geometry == 'ignored' else 'o-', color=color, label=label)
                        handles[label] = line
                    ax.set_title(part+' / '+axis+' axis')
                    ax.set_xlabel('Reference phase-ramp tilt / degrees')
                    ax.set_ylabel('Wavefront RMSE / waves')
            fig.legend(list(handles.values()), list(handles), loc='lower center', ncol=3, fontsize=8)
            fig.tight_layout(rect=(0,.1,1,1))
            for suffix in ('png','svg'):
                fig.savefig(self.output/f'{kind}.{suffix}',dpi=150)
            plt.close(fig)
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if kind == 'regions':
            widths = [r for r in valid if r.get('scan') == 'width']
            axes[0].plot([r['settings']['half_width'] for r in widths],
                         [r['rmse_waves'] for r in widths], 'o-', label='Fourier')
            dr = [r for r in valid if r['method'] == 'direct']
            if dr:
                axes[0].axhline(dr[0]['rmse_waves'], color='C1', label='Direct (fixed image)')
            axes[0].set_xlabel('FFT window half-width / bins')
            centers = [r for r in valid if r.get('scan') == 'center']
            if centers:
                scat = axes[1].scatter([r['settings']['offset_x'] for r in centers],
                                       [r['settings']['offset_y'] for r in centers],
                                       c=[r['rmse_waves'] for r in centers], marker='s', s=160, cmap='viridis')
                fig.colorbar(scat, ax=axes[1], label='Wavefront RMSE / waves')
                axes[1].set_xlabel('Window center offset x / bins')
                axes[1].set_ylabel('Window center offset y / bins')
        axes[0].set_ylabel('Wavefront RMSE / waves')
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(self.output/f'{kind}.{suffix}', dpi=150)
        plt.close(fig)

    def finish(self, status='complete', reason=None):
        self.manifest.update(seconds=perf_counter()-self.started, peak_memory_mib=peak_memory_mib(),
                             row_count=len(self.rows), status=status)
        if reason is not None:
            self.manifest['stop_reason'] = reason
        write_json(self.output/'manifest.json', self.manifest)
        write_rows(self.output, self.rows)
        print('wrote', self.output, 'seconds', round(self.manifest['seconds'], 2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compare', 'regions', 'angles', 'verify-reference'))
    parser.add_argument('--profile', choices=('reference', 'retrieval', 'quick', 'test'),
                        help='required unless --config is supplied; no implicit large experiment')
    parser.add_argument('--config', help='BUV-compatible JSON; overrides the profile')
    parser.add_argument('--allow-large-run', action='store_true',
                        help='explicitly enable grids/source counts larger than the tiny test scene')
    parser.add_argument('--output')
    parser.add_argument('--modes', default='7,25,11,15', help='zero-based reference modes, or all')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-sources', type=int, default=1)
    parser.add_argument('--backend', choices=('separable','czt'), default='separable',
                        help='exact separable square-grating composition, or original sampled CZTs')
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--screen-initials', type=int, default=0,
                        help='screen independent .03/.1/.3-wave starts by observed intensity only')
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--max-nfev', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--truth-rms', type=float)
    parser.add_argument('--calibration-amplitude', type=float, default=.02)
    parser.add_argument('--calibration-cache', help='reuse independent training images in an existing *_calibration.npz')
    parser.add_argument('--recovery-tolerance', type=float, default=.01)
    parser.add_argument('--snr-db', type=float)
    parser.add_argument('--widths', type=float, nargs='+', default=[8,12,16,20,24,28,32])
    parser.add_argument('--offsets', type=float, nargs='+', default=[-3,-2,-1,0,1,2,3])
    parser.add_argument('--shapes', nargs='+', choices=('rect','circle','ellipse','gaussian'),
                        default=['rect','circle','ellipse','gaussian'])
    parser.add_argument('--erosions', type=int, nargs='+', default=[0,2,4,8,12])
    parser.add_argument('--aspects', type=float, nargs='+', default=[.5,1,2])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0,.1,.3,.5])
    parser.add_argument('--recalibrate-windows', action='store_true', help='diagnostic; default freezes nominal calibration')
    parser.add_argument('--selection-validation-rms', type=float, default=.05)
    parser.add_argument('--cross-angles', type=float, nargs='*', default=[])
    parser.add_argument('--components', nargs='+', choices=('object','grating','camera'), default=['object','grating','camera'])
    parser.add_argument('--axes', nargs='+', choices=('x','y','diagonal'), default=['x','y'])
    parser.add_argument('--angles', type=float, nargs='+', default=[-1,-.5,-.25,0,.25,.5,1])
    parser.add_argument('--geometry', nargs='+', choices=('known','ignored'), default=['known','ignored'])
    parser.add_argument('--center-modes', nargs='+', choices=('fixed','auto'), default=['fixed','auto'])
    parser.add_argument('--reference-directory', default='/Users/chiyizi/sjtu/二维剪切干涉/BUV0725')
    parser.add_argument('--all-sources', action='store_true')
    args = parser.parse_args(argv)
    if args.profile is None and args.config is None:
        parser.error('choose --profile test for a tiny check, or explicitly supply --profile/--config')
    config = (DiffractionConfig.from_json(args.config) if args.config else
              DiffractionConfig.profile(args.profile))
    if requires_large_run(config) and not args.allow_large_run:
        parser.error('this grid/source count requires --allow-large-run; '
                     'do not run large experiments on this memory-limited machine')
    if args.workers < 1 or args.batch_sources < 1:
        parser.error('--workers and --batch-sources must be positive')
    try:
        with exclusive_study(), interrupt_on_termination():
            study = Study(args, config)
            try:
                study.initialize()
                if args.command == 'verify-reference':
                    from .reference_check import check_reference
                    report = check_reference(study.model, args.reference_directory, args.all_sources)
                    write_json(study.output/'reference_parity.json', report)
                    if not report['passed']:
                        raise RuntimeError('reference parity failed')
                    print(report['errors'], flush=True)
                    study.finish()
                else:
                    {'compare': study.compare, 'regions': study.regions, 'angles': study.angles}[args.command]()
            except (KeyboardInterrupt, Exception) as error:
                if hasattr(study, 'manifest'):
                    study.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                 str(error) or type(error).__name__)
                raise
    except (RuntimeError, FileExistsError, KeyboardInterrupt) as error:
        raise SystemExit(str(error) or 'study interrupted; partial results retained') from error


if __name__ == '__main__':
    main()
