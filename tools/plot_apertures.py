"""Plot the committed local NA table; no propagation or fitting is run.

    uv run --group report python -m tools.plot_apertures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path('docs/results/apertures/local-preliminary')
    rows = json.loads((directory / 'results.json').read_text())
    pairs = [(.00769, .02), (.012, .02), (.00769, .016)]

    def values(n, mode, snr, selected=pairs):
        found = []
        for inner, outer in selected:
            matches = [r for r in rows if (r['camera_n'], r['NA_in'], r['NA_out'],
                       r.get('center_mode', 'direct'), r['snr_db']) == (n, inner, outer, mode, snr)]
            if len(matches) != 1:
                raise ValueError(f'expected one row: {n}, {inner}, {outer}, {mode}, {snr}')
            found.append(matches[0]['rmse_waves'])
        return np.array(found)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(3)
    labels = ['Reference\n.00769 / .020', 'Higher inner NA\n.012 / .020',
              'Lower outer NA\n.00769 / .016']
    for ax, mode in zip(axes.flat, ['fixed', 'auto']):
        ax.plot(x, values(480, mode, None)*1000, 'o-', label='Clean')
        ax.plot(x, values(480, mode, 40)*1000, 's--', label='40 dB, seed 42')
        ax.set_title(f'Fourier {mode}, 480 x 480')
        ax.set_ylabel('RMSE / 0.001 waves')
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.bar(x, values(480, 'direct', 40)*1000, color='C2')
    ax.set_title('Direct, 40 dB, 480 x 480')
    ax.set_ylabel('RMSE / 0.001 waves')
    ax.set_xticks(x, labels, fontsize=8)
    ax.text(.02, .96, 'Clean direct: < 2.0e-17 waves (same model)',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 1]
    for mode, marker in [('fixed', 'o'), ('auto', 's')]:
        for snr, style in [(None, '-'), (40, '--')]:
            y = [values(n, mode, snr, pairs[:1])[0] for n in (240, 480)]
            ax.plot([240, 480], y, marker=marker, linestyle=style,
                    label=f'{mode}, {"clean" if snr is None else "40 dB"}')
    ax.set_yscale('log')
    ax.set_xticks([240, 480])
    ax.set_xlabel('Camera samples per side (same physical FOV)')
    ax.set_ylabel('Fourier RMSE / waves')
    ax.set_title('Reference NA: strong sampling sensitivity')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle('Local four-source NA pilot: 0.1 waves RMS, known aperture\n'
                 'One wavefront and one noise seed; sampling convergence not established', fontsize=11)
    fig.tight_layout()
    fig.savefig(directory / 'na-comparison.png', dpi=160)
    fig.savefig(directory / 'na-comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
