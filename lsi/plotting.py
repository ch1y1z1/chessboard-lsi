"""绘图辅助（Agg 后端，图片写入 ``output/``）。"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT = Path(os.environ.get("LSI_OUTPUT", Path(__file__).resolve().parent.parent / "output"))
OUTPUT.mkdir(parents=True, exist_ok=True)

__all__ = ["OUTPUT", "savefig", "imshow", "plot_wavefront", "new_fig"]


def new_fig(nrows=1, ncols=1, *, figsize=None, **kwargs):
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def savefig(fig, name: str, *, dpi: int = 150) -> Path:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def imshow(ax, data, grid=None, *, title="", cmap="viridis", mask_nan=True,
           colorbar=True, vmin=None, vmax=None, nan_color="#f0f0f0"):
    arr = np.asarray(data, dtype=float).copy()
    if mask_nan and np.any(~np.isfinite(arr)):
        arr = np.ma.masked_invalid(arr)
    extent = None
    if grid is not None:
        extent = [-grid.extent, grid.extent, -grid.extent, grid.extent]
    im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if np.ma.isMaskedArray(arr):
        im.cmap.set_bad(nan_color)
    ax.set_title(title, fontsize=9)
    if grid is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def plot_wavefront(ax, grid, W, *, title="", pupil=None, cmap="jet", **kwargs):
    arr = np.asarray(W, dtype=float)
    if pupil is not None:
        arr = np.where(pupil, arr, np.nan)
    return imshow(ax, arr, grid, title=title, cmap=cmap, **kwargs)