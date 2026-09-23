"""lsi -- 45° 棋盘光栅剪切干涉：前向模型 + 两条波前反演路线。

* 论文路线:
  ``I --(相移 / 傅里叶变换)--> 差分波前 --(差分 Zernike 最小二乘)--> W``
* 直接非线性路线:
  ``I --(非线性前向模型上的 Levenberg-Marquardt)--> W``
"""

from __future__ import annotations

from .config import Grid, SystemConfig, PRESET_FOURIER, PRESET_PHASE_SHIFT
from .forward import (
    DEFAULT_ORDERS,
    DEFAULT_ORDERS_9,
    ForwardModel,
    ZernikeWavefront,
    add_noise,
    paper_region_intensity,
)
from .grating import chessboard_orders, diffraction_efficiency
from .ftmode import LobeResult, demodulate_lobe, spectrum
from .lm import (
    LMConfig,
    LMResult,
    fit_wavefront_from_carrier_frame,
    fit_wavefront_from_frames,
    levenberg_marquardt,
    multistart_fit,
)
from .metrics import (
    coefficient_error_metrics,
    coefficient_errors,
    pv,
    rms,
    wavefront_error,
)
from .phaseshift import (
    PhaseShiftResult,
    circle_fit,
    find_pupil_circle,
    lsq_phase_shift,
    shear_regions,
)
from .pipeline import (
    DiffPhase,
    demodulate_fourier,
    demodulate_phase_shift,
    fourier_to_wavefront,
    phase_shift_to_wavefront,
    reconstruct,
)
from .reconstruct import ZernikeFit, fit_differential_zernike, wavefront_on_grid
from .unwrap import unwrap_poisson, wrap
from .zernike import (
    differential_zernike,
    differential_zernike_matrix,
    fringe_index,
    wavefront,
    zernike,
    zernike_matrix,
)

__version__ = "1.0.0"
