"""Fourier-transform (single frame, spatial carrier) demodulation.

Dissertation section 2.4.1: a defocus introduces a spatial carrier ``f0``
(``f0 = m / 2s`` with Talbot number ``m``), so the ``+-1`` interference terms
move to ``(+-f0, 0)`` and ``(0, +-f0)`` in the spectrum.  The pipeline is

    I  --carrier demodulation-->  c(x,y) = A e^{i psi}
       --arg-->  wrapped differential phase  --unwrap-->  dW

Three demodulators are provided:

``"lowpass"`` (default)
    remove the known carrier at each pixel and isolate the baseband lobe with
    a band-limited Fourier filter.  This keeps a physical displacement of the
    measured lobe relative to ``f0`` as differential-wavefront phase.

``"local"``
    windowed Fourier transform with the phase reference taken *at each
    pixel*::

        c(x, y) = [ I(x, y) e^{-i 2 pi f0 x} ] * G_sigma (x, y)

    This isolates the ``+f0`` lobe (a Gaussian of ``sigma = 0.06`` pupil
    units suppresses the zero-order lobe, which sits ``f0 = 11.4`` cycles/unit
    away) and keeps the phase reference at each pixel, so ``arg c`` is
    directly the differential phase up to the constant discussed below.
    The residual bias of the Gaussian smoothing is O(sigma^2) and is ~1 % of
    the recovered coefficient for ``sigma <= 0.06``.

``"spectrum"``
    window the ``+f0`` Fourier lobe, roll it to the origin and inverse
    transform, correcting only the known carrier's sub-pixel FFT-bin residual
    with a ramp.  Kept
    for reference: its phase reference differs from the local one by a
    constant tied to the FFT origin, which a constant-offset model cannot
    absorb without giving up the tilt.

Content of the ``+f0`` lobe: it is the beat of the zero order with the
``+(1,0)`` order, whose coefficient is ``A_10 conj(A_00)`` -- *negative* for a
real chessboard.  Hence

    psi(x, y) = 2 pi [ W(x+s, y) - W(x, y) ] + pi     (one-sided difference)

The two-sided reading of eq. (2-42),

    psi(x, y) = pi [ W(x+s, y) - W(x-s, y) ] + pi .

is retained by :mod:`lsi.pipeline` as the dissertation's approximation; it
differs from the physical isolated-lobe model by ``O(s^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import check_direction

__all__ = [
    "LobeResult",
    "spectrum",
    "find_carrier_peak",
    "demodulate_lobe",
    "ft_differential_phase",
]


def spectrum(image: np.ndarray) -> np.ndarray:
    """Shifted 2-D spectrum of a real image."""
    return np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=float)))


@dataclass
class LobeResult:
    phase: np.ndarray          # wrapped phase after phase_offset removal (rad)
    amplitude: np.ndarray      # |c(x, y)|
    complex_field: np.ndarray  # c(x, y)
    peak_index: tuple[int, int]
    peak_freq: tuple[float, float]
    direction: str
    method: str = "local"
    unwrapped_phase: np.ndarray | None = None
    phase_offset: float = 0.0


def _refine_peak_subpixel(mag: np.ndarray, i0: int, j0: int) -> tuple[float, float]:
    """Parabolic sub-pixel refinement of a spectrum peak (in pixels)."""
    n = mag.shape[0]

    def _delta(a_m, a_0, a_p):
        denom = a_m - 2.0 * a_0 + a_p
        if denom == 0.0:
            return 0.0
        return float(np.clip(0.5 * (a_m - a_p) / denom, -0.5, 0.5))

    di = _delta(mag[i0 - 1, j0], mag[i0, j0], mag[i0 + 1, j0]) if 0 < i0 < n - 1 else 0.0
    dj = _delta(mag[i0, j0 - 1], mag[i0, j0], mag[i0, j0 + 1]) if 0 < j0 < n - 1 else 0.0
    return di, dj


def find_carrier_peak(
    spec: np.ndarray,
    grid,
    *,
    expected_freq: tuple[float, float] | None = None,
    search_px: int = 8,
) -> tuple[int, int]:
    """Locate a first-order spectrum peak (integer grid index, ``(row, col)``).

    With ``expected_freq`` the search is restricted to a small window around
    the expected location, which is what selects the ``+(f0, 0)`` /
    ``(0, +f0)`` lobe rather than its conjugate.  Without a prior the
    strongest peak outside the zero-order lobe is returned.
    """
    spec = np.asarray(spec)
    if spec.shape != grid.shape:
        raise ValueError(
            f"spec must have shape {grid.shape} to match the grid, got {spec.shape}"
        )
    if isinstance(search_px, bool) or not isinstance(search_px, (int, np.integer)):
        raise ValueError(f"search_px must be a non-negative integer, got {search_px!r}")
    if search_px < 0:
        raise ValueError(f"search_px must be non-negative, got {search_px!r}")
    mag = np.abs(spec)
    n = mag.shape[0]
    cy, cx = n // 2, n // 2
    L = 2.0 * grid.extent
    if expected_freq is not None:
        fx, fy = (float(expected_freq[0]), float(expected_freq[1]))
        nyquist = 1.0 / (2.0 * grid.dx)
        if not np.isfinite(fx) or not np.isfinite(fy):
            raise ValueError(f"expected_freq must be finite, got {expected_freq!r}")
        if abs(fx) >= nyquist or abs(fy) >= nyquist:
            raise ValueError(
                f"expected carrier {expected_freq!r} is outside the sampled "
                f"frequency range (-{nyquist:g}, {nyquist:g}); increase grid.n "
                "or reduce the carrier frequency"
            )
        ix = int(round(fx * L)) + cx
        iy = int(round(fy * L)) + cy
        allowed = np.zeros_like(mag, dtype=bool)
        i0, i1 = max(0, iy - search_px), min(n, iy + search_px + 1)
        j0, j1 = max(0, ix - search_px), min(n, ix + search_px + 1)
        allowed[i0:i1, j0:j1] = True
    else:
        exclude = max(2.0, 2.5 * L)   # zero-order lobe radius in pixels
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        allowed = np.hypot(xx - cx, yy - cy) > exclude
    if not np.any(allowed):
        raise ValueError("carrier-peak search region contains no frequency bins")
    masked = np.where(allowed, mag, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), mag.shape)
    return int(idx[0]), int(idx[1])


def demodulate_lobe(
    image: np.ndarray,
    grid,
    direction: str = "x",
    *,
    f0: float | None = None,
    method: str = "lowpass",
    sigma_units: float | None = None,
    window_radius: float | None = None,
    window: str = "butterworth",
    phase_offset: float = 0.0,
) -> LobeResult:
    """Demodulate the ``+1`` carrier lobe of one interferogram.

    Parameters
    ----------
    direction:
        ``"x"`` extracts the ``(+f0, 0)`` lobe, ``"y"`` the ``(0, +f0)`` one.
    f0:
        Carrier frequency in cycles per pupil unit (defaults are supplied by
        :mod:`lsi.pipeline` from the system configuration).
    method:
        ``"local"`` (Gaussian windowed FT), ``"lowpass"`` (same phase
        reference, band-limited filter -- the most accurate) or
        ``"spectrum"`` (Fourier lobe extraction; reference implementation).
    sigma_units:
        Gaussian half-width in pupil units for the local demodulator.
    phase_offset:
        Constant phase of the lobe (``0`` or ``pi``, see
        ``ForwardModel.demodulation_offset``).  It is removed *before* taking
        the argument: for a real chessboard the offset is exactly ``pi``, so
        the raw phase would sit right on the ``+-pi`` branch cut and the
        wrapped map would be corrupted by ``+-pi`` jitter.
    """
    check_direction(direction)
    if method not in ("local", "lowpass", "spectrum"):
        raise ValueError("method must be 'local', 'lowpass' or 'spectrum'")
    if sigma_units is not None:
        if isinstance(sigma_units, bool):
            raise ValueError(
                f"sigma_units must be a positive finite number, got {sigma_units!r}"
            )
        sigma_units = float(sigma_units)
        if not np.isfinite(sigma_units) or sigma_units <= 0.0:
            raise ValueError(
                f"sigma_units must be a positive finite number, got {sigma_units!r}"
            )
    if window_radius is not None:
        if isinstance(window_radius, bool):
            raise ValueError(
                f"window_radius must be a positive finite number, got {window_radius!r}"
            )
        window_radius = float(window_radius)
        if not np.isfinite(window_radius) or window_radius <= 0.0:
            raise ValueError(
                f"window_radius must be a positive finite number, got {window_radius!r}"
            )
    if isinstance(phase_offset, bool):
        raise ValueError(f"phase_offset must be finite, got {phase_offset!r}")
    phase_offset = float(phase_offset)
    if not np.isfinite(phase_offset):
        raise ValueError(f"phase_offset must be finite, got {phase_offset!r}")
    I = np.asarray(image, dtype=float)
    if I.shape != grid.shape:
        raise ValueError(
            f"image must have shape {grid.shape} to match the grid, got {I.shape}"
        )
    if not np.all(np.isfinite(I)):
        raise ValueError("image must contain only finite values")
    if f0 is not None:
        f0 = float(f0)
        nyquist = 1.0 / (2.0 * grid.dx)
        if not np.isfinite(f0) or f0 <= 0.0:
            raise ValueError(f"f0 must be a positive finite frequency, got {f0!r}")
        if f0 >= nyquist:
            raise ValueError(
                f"f0={f0:g} is not below the grid Nyquist frequency "
                f"{nyquist:g}; increase grid.n or reduce f0"
            )
    x, y = grid.coords()
    n = I.shape[0]
    L = 2.0 * grid.extent
    expected = None
    if f0 is not None:
        expected = (f0, 0.0) if direction == "x" else (0.0, f0)

    spec = spectrum(I)
    i0, j0 = find_carrier_peak(spec, grid, expected_freq=expected)
    di, dj = _refine_peak_subpixel(np.abs(spec), i0, j0)
    fx = (j0 - n // 2 + dj) / L
    fy = (i0 - n // 2 + di) / L

    if method == "local":
        from scipy.ndimage import gaussian_filter

        f_use = f0 if f0 is not None else (fx if direction == "x" else fy)
        ramp = x if direction == "x" else y
        base = I * np.exp(-2j * np.pi * f_use * ramp)
        sigma = 0.06 if sigma_units is None else sigma_units
        sigma_px = sigma / (L / n)
        c = gaussian_filter(base, sigma_px)
        # When f0 is known, the residual phase after this multiplication is the
        # signal: a linear differential-wavefront phase legitimately moves the
        # observed spectral peak away from f0.  Estimating (fx-f0) from this same
        # science frame and removing it would therefore erase defocus and other
        # low-order content.  Peak refinement is diagnostic only in this branch.
    elif method == "spectrum":
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        rr = np.hypot(xx - j0, yy - i0)
        rad = window_radius if window_radius is not None else 2.5 * L
        if window == "gaussian":
            win = np.exp(-0.5 * (rr / rad) ** 2)
        elif window == "butterworth":
            win = 1.0 / (1.0 + (rr / rad) ** 4)
        elif window == "ideal":
            win = (rr <= rad).astype(float)
        else:
            raise ValueError(f"unknown window {window!r}")
        lobe = spec * win
        if f0 is None:
            shift_i, shift_j = i0, j0
            ref_fx, ref_fy = fx, fy
        else:
            # Centre the field on the *known* carrier, not on the measured lobe
            # maximum: the difference between them can be physical wavefront
            # slope and must remain in the recovered phase.
            shift_j = n // 2 + (int(round(f0 * L)) if direction == "x" else 0)
            shift_i = n // 2 + (int(round(f0 * L)) if direction == "y" else 0)
            ref_fx = f0 if direction == "x" else 0.0
            ref_fy = f0 if direction == "y" else 0.0
        lobe = np.roll(
            lobe, (-shift_i + n // 2, -shift_j + n // 2), axis=(0, 1)
        )
        c = np.fft.ifft2(np.fft.ifftshift(lobe))
        # the integer roll already removed the integer part of the carrier;
        # only the sub-pixel residual needs a ramp
        fx_int = (shift_j - n // 2) / L
        fy_int = (shift_i - n // 2) / L
        c = c * np.exp(
            -2j * np.pi * ((ref_fx - fx_int) * x + (ref_fy - fy_int) * y)
        )
    elif method == "lowpass":
        # same phase reference as "local" (the carrier is removed at each
        # pixel) but the lobe is isolated with a band-limited filter, which
        # avoids the O(sigma^2) smoothing bias of the Gaussian kernel
        f_use = f0 if f0 is not None else (fx if direction == "x" else fy)
        ramp = x if direction == "x" else y
        base = I * np.exp(-2j * np.pi * f_use * ramp)
        # As in the local branch, do not remove the observed (peak-f0)
        # displacement: it contains real linear differential phase.
        bs = np.fft.fftshift(np.fft.fft2(base))
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        rr = np.hypot(xx - n // 2, yy - n // 2)
        rad = window_radius if window_radius is not None else 4.0 * L
        if window == "gaussian":
            win = np.exp(-0.5 * (rr / rad) ** 2)
        elif window == "butterworth":
            win = 1.0 / (1.0 + (rr / rad) ** 4)
        elif window == "ideal":
            win = (rr <= rad).astype(float)
        else:
            raise ValueError(f"unknown window {window!r}")
        c = np.fft.ifft2(np.fft.ifftshift(bs * win))
    if phase_offset:
        c = c * np.exp(-1j * phase_offset)

    return LobeResult(
        phase=np.angle(c),
        amplitude=np.abs(c),
        complex_field=c,
        peak_index=(i0, j0),
        peak_freq=(float(fx), float(fy)),
        direction=direction,
        method=method,
        phase_offset=phase_offset,
    )


def ft_differential_phase(image, grid, direction="x", *, f0=None, **kwargs):
    """Convenience wrapper (identical to :func:`demodulate_lobe`)."""
    return demodulate_lobe(image, grid, direction=direction, f0=f0, **kwargs)
