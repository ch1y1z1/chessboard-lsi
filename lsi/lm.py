"""Levenberg-Marquardt inversion of the non-linear forward model.

The forward model ``I = |sum_ab A_ab exp(i 2 pi W(x+as, y+bs) ...)|^2`` is a
non-linear function of the Zernike coefficients ``c`` of the wavefront.
Instead of linearising the *interferogram* (phase-shift / Fourier route), we
minimise the intensity residual directly:

    min_c  || I_meas - I_model(c) ||^2

with the Levenberg-Marquardt iteration

    (J^T J + lambda diag(J^T J)) delta = -J^T f,      c <- c + delta
    rho = (F(c) - F(c+delta)) / delta^T (lambda diag(J^T J) delta - J^T f)

The damped step is computed as an augmented least-squares problem rather than
by explicitly solving these normal equations, avoiding the numerical
``cond(J)^2`` penalty.

and Nielsen's damping update.  The Jacobian is analytic:

    dI/dc_j = 2 Re{ E* dE/dc_j },
    dE/dc_j = sum_ab A_ab exp(i phi_ab) * i 2 pi * Z_j(x+as, y+bs)

so each iteration costs one order-superposition pass, no finite differences.

Advantages over the dissertation pipeline (see ``scripts/03_lm_inverse.py``):
no unwrapping, no shear-region detection, no modulation-sign problem, works
with a single carrier frame, and the affine nuisance parameters (intensity
scale and background) can be estimated jointly by variable projection.  The
phase steps and the shear are *not* fitted here: the parameter vector holds
the Zernike coefficients only, and both are taken from the forward model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .config import _as_float, _as_int
from .forward import ForwardModel, ZernikeWavefront

__all__ = [
    "LMConfig",
    "LMResult",
    "levenberg_marquardt",
    "fit_wavefront_from_frames",
    "fit_wavefront_from_carrier_frame",
    "multistart_fit",
]


@dataclass
class LMConfig:
    max_iter: int = 120
    lambda0: float = 1e-3
    lambda_min: float = 1e-12
    lambda_max: float = 1e10
    nu0: float = 2.0
    ftol: float = 1e-14
    xtol: float = 1e-12
    gtol: float = 1e-12
    verbose: bool = False
    #: estimate an affine intensity model ``alpha * I_model + beta`` in the
    #: loop (variable projection) -- protects against unknown exposure/offset
    fit_scale_background: bool = False

    def __post_init__(self) -> None:
        self.max_iter = _as_int(self.max_iter, "max_iter", minimum=1)
        self.lambda_min = _as_float(self.lambda_min, "lambda_min", low=0.0)
        self.lambda_max = _as_float(self.lambda_max, "lambda_max", low=0.0)
        if self.lambda_min > self.lambda_max:
            raise ValueError("lambda_min must not exceed lambda_max")
        self.lambda0 = _as_float(
            self.lambda0,
            "lambda0",
            low=self.lambda_min,
            high=self.lambda_max,
            inclusive_low=True,
        )
        self.nu0 = _as_float(self.nu0, "nu0", low=1.0)
        for name in ("ftol", "xtol", "gtol"):
            setattr(
                self,
                name,
                _as_float(
                    getattr(self, name),
                    name,
                    low=0.0,
                    inclusive_low=True,
                ),
            )
        if not isinstance(self.verbose, bool):
            raise ValueError(f"verbose must be a bool, got {self.verbose!r}")
        if not isinstance(self.fit_scale_background, bool):
            raise ValueError(
                "fit_scale_background must be a bool, got "
                f"{self.fit_scale_background!r}"
            )


@dataclass
class LMResult:
    x: np.ndarray
    cost: float
    history: dict = field(default_factory=dict)
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    scale: float = 1.0
    background: float = 0.0
    #: 2-norm condition number of the Jacobian itself, evaluated at the returned
    #: ``x``: the variable-projection Jacobian when a scale/background is being
    #: fitted, otherwise the raw one.  Same convention as
    #: ``reconstruct.fit_differential_zernike``, which reports ``cond`` of its
    #: design matrix -- reporting ``cond(J^T J)`` instead would square the
    #: number and make a well-understood ``1e6`` look like total loss of
    #: identifiability.
    cond: float = np.nan
    #: Numerical rank of that same final Jacobian.
    rank: int = 0
    rms_residual: float = 0.0
    n_residual: int = 0
    n_parameters: int = 0
    lambda_final: float = float("nan")


def _solve_damped(J: np.ndarray, f: np.ndarray, lam: float):
    """Solve the damped step without explicitly forming normal equations.

    The augmented least-squares system has the same minimizer as
    ``(J.T J + lam D) delta = -J.T f`` but avoids squaring ``cond(J)``::

        [ J             ] delta ~= [ -f ]
        [ sqrt(lam D)   ]          [  0 ]
    """
    JtJ = J.T @ J
    g = J.T @ f
    diag = np.clip(np.diag(JtJ), 1e-30, None)
    if lam > 0.0:
        damping = np.diag(np.sqrt(lam * diag))
        A = np.vstack([J, damping])
        rhs = np.concatenate([-f, np.zeros(J.shape[1])])
    else:
        A = J
        rhs = -f
    delta = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return delta, g, JtJ


def levenberg_marquardt(
    residual_and_jac: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0: Sequence[float],
    config: LMConfig | None = None,
    *,
    scale_background: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float, float]
    ]
    | None = None,
) -> LMResult:
    """LM minimisation of ``||f(x)||^2`` with an analytic Jacobian.

    ``residual_and_jac(x)`` must return ``(f, J)`` with ``f`` the residual
    vector and ``J`` its Jacobian.  If ``scale_background`` is given, it maps
    ``(f_raw, J_raw)`` onto ``(f, J, alpha, beta)`` for the affine intensity
    model (variable projection).
    """
    cfg = config or LMConfig()
    if cfg.fit_scale_background and scale_background is None:
        raise ValueError(
            "LMConfig(fit_scale_background=True) requires a "
            "scale_background callback"
        )
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x0 must be a non-empty one-dimensional parameter vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain only finite values")
    lam = cfg.lambda0
    nu = cfg.nu0
    history = {"cost": [], "lambda": [], "grad_norm": [], "step_norm": [], "rho": []}

    def evaluate(point):
        f_eval, J_eval = residual_and_jac(point)
        f_eval = np.asarray(f_eval, dtype=float)
        J_eval = np.asarray(J_eval, dtype=float)
        if f_eval.ndim != 1:
            raise ValueError(f"residual must be one-dimensional, got {f_eval.shape}")
        if J_eval.shape != (f_eval.size, point.size):
            raise ValueError(
                "Jacobian shape must be (n_residual, n_parameters), got "
                f"{J_eval.shape} for {f_eval.size} residuals and {point.size} parameters"
            )
        if f_eval.size == 0:
            raise ValueError("cannot run LM with zero residual observations")
        if not np.all(np.isfinite(f_eval)) or not np.all(np.isfinite(J_eval)):
            raise ValueError("residual and Jacobian must contain only finite values")
        a_eval, b_eval = 1.0, 0.0
        if cfg.fit_scale_background and scale_background is not None:
            f_eval, J_eval, a_eval, b_eval = scale_background(f_eval, J_eval)
            f_eval = np.asarray(f_eval, dtype=float)
            J_eval = np.asarray(J_eval, dtype=float)
            if J_eval.shape != (f_eval.size, point.size) or f_eval.size == 0:
                raise ValueError("reduced residual/Jacobian have inconsistent shapes")
            if not np.all(np.isfinite(f_eval)) or not np.all(np.isfinite(J_eval)):
                raise ValueError("reduced residual and Jacobian must be finite")
        return f_eval, J_eval, float(a_eval), float(b_eval)

    f, J, alpha, beta = evaluate(x)
    cost = float(f @ f)
    conv = False
    msg = "max_iter"
    n_iter = 0

    for it in range(cfg.max_iter):
        n_iter = it + 1
        delta, g, JtJ = _solve_damped(J, f, lam)
        gnorm = float(np.linalg.norm(g))
        step = float(np.linalg.norm(delta))
        history["cost"].append(cost)
        history["lambda"].append(lam)
        history["grad_norm"].append(gnorm)
        history["step_norm"].append(step)

        if gnorm <= cfg.gtol:
            conv, msg = True, "gradient tolerance"
            break
        if step <= cfg.xtol * (float(np.linalg.norm(x)) + cfg.xtol):
            conv, msg = True, "step tolerance"
            break

        x_new = x + delta
        f_new, J_new, a_new, b_new = evaluate(x_new)
        cost_new = float(f_new @ f_new)

        diag = np.clip(np.diag(JtJ), 1e-30, None)
        predicted = float(delta @ (lam * diag * delta - g))
        rho = (cost - cost_new) / predicted if predicted > 0 else -1.0
        history["rho"].append(float(rho))

        if rho > 0.0:
            x, f, J, cost = x_new, f_new, J_new, cost_new
            alpha, beta = a_new, b_new
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = cfg.nu0
            lam = float(np.clip(lam, cfg.lambda_min, cfg.lambda_max))
            if abs(cost - history["cost"][-1]) <= cfg.ftol * max(cost, 1e-30):
                conv, msg = True, "cost tolerance"
                break
        else:
            lam *= nu
            nu *= 2.0
            lam = float(np.clip(lam, cfg.lambda_min, cfg.lambda_max))
            if lam >= cfg.lambda_max:
                conv, msg = False, "lambda overflow"
                break

        if cfg.verbose:
            print(f"  iter {n_iter:3d}  cost={cost:.6e}  lambda={lam:.2e}  "
                  f"|g|={gnorm:.3e}  rho={rho:+.3f}")

    n_rows = f.size
    # Condition number of the Jacobian at the *solution*, not at the starting
    # point: it says whether the returned fit is identifiable.  When a
    # scale/background is fitted, J is the variable-projection Jacobian, so the
    # nuisances are already concentrated out and do not inflate this number.
    # Taking the normal matrix instead would report the square of this value.
    cond = float(np.linalg.cond(J)) if J.size else np.nan
    rank = int(np.linalg.matrix_rank(J)) if J.size else 0
    return LMResult(
        x=x,
        cost=cost,
        history=history,
        n_iter=n_iter,
        converged=conv,
        message=msg,
        scale=float(alpha),
        background=float(beta),
        n_residual=int(n_rows),
        n_parameters=int(x.size),
        lambda_final=float(lam),
        cond=cond,
        rank=rank,
        rms_residual=float(np.sqrt(cost / max(n_rows, 1))),
    )


def fit_wavefront_from_carrier_frame(
    forward: ForwardModel,
    wf_proto: ZernikeWavefront,
    image: np.ndarray,
    *,
    f0: float | None = None,
    **kwargs,
) -> LMResult:
    """LM fit from a *single* carrier-mode interferogram (no phase shifting).

    Builds the per-order carrier phases ``2 pi f0 (a x + b y)`` internally and
    forwards everything else to :func:`fit_wavefront_from_frames`.
    """
    f0 = forward.config.carrier_f0 if f0 is None else f0
    x, y = forward.grid.coords()
    carriers = [forward.carrier_phases(x, y, f0)]
    return fit_wavefront_from_frames(
        forward, wf_proto, [image], [None], carriers, **kwargs
    )


def multistart_fit(
    forward: ForwardModel,
    wf_proto: ZernikeWavefront,
    frames,
    deltas=None,
    carriers=None,
    *,
    term: int = 0,
    values: Sequence[float] = (),
    coarse_iter: int = 25,
    config: LMConfig | None = None,
    **kwargs,
) -> LMResult:
    """LM with a coarse search over one coefficient (default: the first term).

    Non-linear least squares on an interferogram is only locally convergent:
    for strongly aberrated wavefronts (many fringes) the cost landscape has
    several minima.  This helper scans ``term`` over ``values``, keeps the
    best starting point and refines it with LM.  For a mildly aberrated
    wavefront a plain :func:`fit_wavefront_from_frames` with a zero start is
    enough.
    """
    cfg = config or LMConfig()
    best_x0, best_cost = np.zeros(wf_proto.n_terms), np.inf
    coarse = replace(cfg, max_iter=coarse_iter)
    for v in values:
        x0 = np.zeros(wf_proto.n_terms)
        x0[term] = float(v)
        res = fit_wavefront_from_frames(
            forward, wf_proto, frames, deltas, carriers,
            x0=x0, config=coarse, **kwargs,
        )
        if res.cost < best_cost:
            best_cost, best_x0 = res.cost, np.array(res.x, dtype=float)
    return fit_wavefront_from_frames(
        forward, wf_proto, frames, deltas, carriers, x0=best_x0, config=cfg, **kwargs
    )


# --------------------------------------------------------------------------- #
# convenience wrapper: fit Zernike coefficients from measured frames
# --------------------------------------------------------------------------- #

def reduced_scale_background(f_raw, J_raw, meas):
    """Concentrate an affine intensity model out of the residual.

    Removes the nuisance parameters ``(a, b)`` of ``a * m + b`` from the fit by
    variable projection, where ``m = f_raw + meas`` is the raw model intensity
    and ``meas`` the measurement.  Returns ``(f, J, a, b)``: the residual of
    the optimal affine model, the Golub-Pereyra reduced Jacobian, and the
    fitted scale and background.

    ``a * J_raw`` alone is *not* the reduced Jacobian: it still contains the
    component of ``d(a * m) / dc`` that the re-optimised ``(a, b)`` absorb.
    The correct reduced Jacobian projects the model columns onto the
    orthogonal complement of the nuisance basis ``span{m, 1}``::

        d r~/dc_j = (I - Q Q^T) (a * d m/dc_j),   Q = orth([m, 1])

    which is exact because ``(a, b)`` are re-fitted at every step, so the
    reduced residual is orthogonal to ``m`` and ``1``.
    """
    m = np.asarray(f_raw) + meas
    A = np.stack([m, np.ones_like(m)], axis=1)
    G = A.T @ A
    Ginv = np.linalg.pinv(G)
    u = Ginv @ (A.T @ meas)              # [a, b]
    a, b = float(u[0]), float(u[1])
    f = A @ u - meas
    if not J_raw.size:
        return f, J_raw, a, b
    # Exact derivative of f(c) = P(c) meas - meas, where the projector is
    # P = A (A^T A)^-1 A^T and A(c) = [m(c), 1].  Differentiating the projector
    # gives, for each column j with dA_j = dA/dc_j and dG_j = d(A^T A)/dc_j,
    #     df/dc_j = dA_j u + A G^-1 (dA_j^T meas - dG_j u)
    # which is the Golub-Pereyra reduced Jacobian.  A plain ``a * J_raw``
    # misses both the projection and the derivative of the fitted (a, b).
    J = np.empty_like(J_raw, dtype=float)
    for j in range(J_raw.shape[1]):
        dA = np.zeros_like(A)
        dA[:, 0] = J_raw[:, j]
        dG = dA.T @ A + A.T @ dA
        J[:, j] = dA @ u + A @ (Ginv @ (dA.T @ meas - dG @ u))
    return f, J, a, b


def fit_wavefront_from_frames(
    forward: ForwardModel,
    wf_proto: ZernikeWavefront,
    frames: np.ndarray | Sequence[np.ndarray],
    deltas: Sequence[np.ndarray | None] | None = None,
    carriers: Sequence[np.ndarray | None] | None = None,
    *,
    x0: Sequence[float] | None = None,
    config: LMConfig | None = None,
    samples: int | None = 4096,
    seed: int = 0,
    pixel_mask: np.ndarray | None = None,
) -> LMResult:
    """Fit the Zernike coefficients of a wavefront from intensity frames.

    ``frames`` is a sequence of measured intensity arrays (e.g. the N
    phase-shift frames, or a single carrier frame).  ``deltas``/``carriers``
    describe the known per-frame per-order phase modulations; pass ``None``
    entries for the carrier-mode frames.  Piston (Z1) is rejected because a
    common field phase cancels exactly from every intensity observation.
    """
    if not isinstance(wf_proto, ZernikeWavefront):
        raise ValueError("wf_proto must be a ZernikeWavefront")
    if 1 in wf_proto.indices:
        raise ValueError(
            "piston Z1 is unobservable from intensity-only LSI data; "
            "remove it from wf_proto.indices and fix the piston gauge"
        )
    if isinstance(frames, np.ndarray) and frames.ndim == 2:
        frames = [frames]
    else:
        frames = list(frames)
    frames = [np.asarray(f, dtype=float) for f in frames]
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("frames must contain at least one measured image")
    for i, frame in enumerate(frames):
        if frame.shape != forward.shape:
            raise ValueError(
                f"frame {i} must have shape {forward.shape}, got {frame.shape}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError(f"frame {i} must contain only finite values")
    if deltas is None:
        deltas = [None] * n_frames
    else:
        deltas = list(deltas)
    if carriers is None:
        carriers = [None] * n_frames
    else:
        carriers = list(carriers)
    if len(deltas) != n_frames or len(carriers) != n_frames:
        raise ValueError("deltas/carriers must have one entry per frame")
    n_orders = len(forward.orders.ab)
    for i, delta in enumerate(deltas):
        if delta is not None:
            delta = np.asarray(delta, dtype=float)
            if delta.shape != (n_orders,) or not np.all(np.isfinite(delta)):
                raise ValueError(
                    f"deltas[{i}] must be a finite array with shape ({n_orders},)"
                )
            deltas[i] = delta
    for i, carrier in enumerate(carriers):
        if carrier is not None:
            carrier = np.asarray(carrier, dtype=float)
            expected = (n_orders, *forward.shape)
            if carrier.shape != expected or not np.all(np.isfinite(carrier)):
                raise ValueError(
                    f"carriers[{i}] must be a finite array with shape {expected}"
                )
            carriers[i] = carrier

    if samples is not None:
        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
            raise ValueError(f"samples must be a positive integer or None, got {samples!r}")
        samples = int(samples)
        if samples < 1:
            raise ValueError(f"samples must be positive, got {samples!r}")

    n_tot = forward.shape[0] * forward.shape[1]
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != forward.shape:
            raise ValueError(
                f"pixel_mask must have shape {forward.shape}, got {pixel_mask.shape}"
            )
        rows = np.nonzero(pixel_mask.ravel())[0]
        if samples is not None and 0 < samples < len(rows):
            rng = np.random.default_rng(seed)
            rows = np.sort(rng.choice(rows, size=samples, replace=False))
    else:
        rng = np.random.default_rng(seed)
        if samples is None or samples >= n_tot:
            rows = np.arange(n_tot)
        else:
            rows = np.sort(rng.choice(n_tot, size=samples, replace=False))

    if rows.size == 0:
        raise ValueError("pixel selection contains no observations")

    meas = np.concatenate([f.reshape(-1)[rows] for f in frames])
    # carrier maps arrive as 2-D arrays; the model works on the sampled pixels
    carriers = [
        None if c is None else np.asarray(c).reshape(np.shape(c)[0], -1)[:, rows]
        for c in carriers
    ]
    n_terms = wf_proto.n_terms
    coeff0 = (
        np.asarray(x0, dtype=float)
        if x0 is not None
        else np.zeros(n_terms, dtype=float)
    )
    if coeff0.shape != (n_terms,):
        raise ValueError(f"x0 must have shape ({n_terms},), got {coeff0.shape}")
    if not np.all(np.isfinite(coeff0)):
        raise ValueError("x0 must contain only finite values")
    if n_frames * rows.size < n_terms:
        raise ValueError(
            f"only {n_frames * rows.size} residuals are available for "
            f"{n_terms} fitted coefficients"
        )
    cache = forward.terms_cache(wf_proto, rows)

    def residual_and_jac(c):
        I_model, J = forward.model_and_jacobian(
            c, wf_proto, deltas, carriers, rows=rows, cache=cache
        )
        return I_model - meas, J

    scale_fn = None
    if (config or LMConfig()).fit_scale_background:
        def scale_fn(f_raw, J_raw):
            return reduced_scale_background(f_raw, J_raw, meas)

    return levenberg_marquardt(
        residual_and_jac,
        coeff0,
        config,
        scale_background=scale_fn,
    )
