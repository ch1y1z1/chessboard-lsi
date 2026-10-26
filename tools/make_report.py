"""Plot saved observations and results; never runs diffraction or optimization.

    uv run --group report python -m tools.make_report output/my-run
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lsi.basis import basis
from lsi.config import SystemConfig


def make_report(directory, case=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = json.loads((directory / 'results.json').read_text())
    valid = [r for r in rows if r.get('rmse_waves') is not None]
    if not valid:
        raise ValueError('no wavefront results to plot')
    fig, ax = plt.subplots(figsize=(max(7, len(valid) * .3), 4))
    ax.bar(np.arange(len(valid)), [r['rmse_waves'] for r in valid],
           color=['C1' if r['method'] == 'direct' else 'C0' for r in valid])
    ax.set_xticks(np.arange(len(valid)), [r['case'] + '\n' + r['method'] for r in valid],
                  rotation=70, ha='right', fontsize=7)
    ax.set_ylabel('Wavefront RMSE / waves')
    fig.tight_layout()
    fig.savefig(directory / 'errors.png', dpi=160)
    plt.close(fig)

    selected = [r for r in valid if r['case'] == (case or valid[0]['case'])]
    if not selected:
        raise ValueError(f'no results for case {case}')
    if manifest.get('command') == 'apertures':
        selected = [r for r in valid if r['observation'] == selected[0]['observation']]
    with np.load(directory / selected[0]['observation']) as saved:
        image, truth, pupil = saved['observed'], saved['truth_wavefront'], saved['pupil']
    # Reconstruct the pupil wavefront from saved coefficients only.
    system = SystemConfig(**manifest['system'])
    if 'NA_in' in selected[0]:
        system = replace(system, NA_in=selected[0]['NA_in'], NA_out=selected[0]['NA_out'])
    frequency = np.fft.fftshift(np.fft.fftfreq(system.n, system.super_lattice_period / system.N_lattice))
    fx, fy = np.meshgrid(frequency, frequency)
    rho, theta = system.wavelength * np.hypot(fx, fy) / system.NA_out, np.arctan2(fy, fx)
    Z = basis(manifest['fit']['indices'], rho[pupil], theta[pupil], system.epsilon, system.Nz)
    entries = [(image, 'Observed intensity', 'intensity'),
               (np.where(pupil, truth, np.nan), 'True wavefront / waves', 'wavefront')]
    for row in selected:
        fitted = np.full(truth.shape, np.nan)
        fitted[pupil] = np.asarray(row['coeffs_waves']) @ Z
        method = 'Direct' if row['method'] == 'direct' else 'Fourier'
        if row.get('center_mode'):
            method += f" ({row['center_mode']})"
        entries.extend([(fitted, method + ' / waves', 'wavefront'),
                        (fitted - truth, method + ' full error / waves', 'error')])
    error_limit = max(max(float(np.nanmax(abs(v))) for v, _, kind in entries if kind == 'error'), 1e-12)
    fig, axes = plt.subplots(2, (len(entries) + 1) // 2, figsize=(13, 7), squeeze=False)
    for ax, (values, title, kind) in zip(axes.flat, entries):
        limits = ({'vmin': float(truth[pupil].min()), 'vmax': float(truth[pupil].max())}
                  if kind == 'wavefront' else {'vmin': -error_limit, 'vmax': error_limit}
                  if kind == 'error' else {})
        plot = ax.imshow(values, origin='lower', cmap='RdBu_r' if kind == 'error' else 'viridis', **limits)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(plot, ax=ax, shrink=.8)
    for ax in list(axes.flat)[len(entries):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--case')
    args = parser.parse_args()
    make_report(args.directory, args.case)
