"""chessboard_lsi -- 45-degree chessboard-grating shearing interferometry.

Two wavefront-sensing routes from measured intensities to the wavefront:

* dissertation route:
  ``I --(phase-shift / Fourier transform)--> differential wavefront
     --(differential Zernike least squares)--> W``
* direct non-linear route:
  ``I --(Levenberg-Marquardt on the non-linear forward model)--> W``
"""

from __future__ import annotations

from .config import (
    PRESET_FOURIER,
    PRESET_PHASE_SHIFT,
    Grid,
    SystemConfig,
    preset_fourier,
    preset_phase_shift,
)
from .forward import (
    DEFAULT_ORDERS_5,
    DEFAULT_ORDERS_9,
    ForwardModel,
    ZernikeWavefront,
    add_noise,
    paper_region_intensity,
)
from .grating import OrderSet, analytic_orders, bitmap_orders

__version__ = "1.0.0"

__all__ = [
    "Grid",
    "SystemConfig",
    "PRESET_PHASE_SHIFT",
    "PRESET_FOURIER",
    "preset_phase_shift",
    "preset_fourier",
    "ForwardModel",
    "ZernikeWavefront",
    "add_noise",
    "paper_region_intensity",
    "OrderSet",
    "analytic_orders",
    "bitmap_orders",
    "DEFAULT_ORDERS_5",
    "DEFAULT_ORDERS_9",
    "__version__",
]