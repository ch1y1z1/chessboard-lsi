"""Phase unwrapping utilities.

* :func:`unwrap_dct` -- Ghiglia-Romero least-squares unwrapping (DCT,
  Neumann boundary).  Fast, intended for *full-frame* maps; the outermost row
  and column carry the usual boundary approximation.
* :func:`unwrap_masked_poisson` -- exact least-squares unwrapping restricted
  to an arbitrary mask (the shear regions here have a lens shape), solved as
  a sparse linear system.  Noise tolerant.
* :func:`unwrap_seed_growth` -- the classical path-following (seed growing)
  unwrapping used by the dissertation pipeline; exact for noise-free data.
"""

from __future__ import annotations

import numpy as np
import warnings

__all__ = ["wrap", "unwrap_dct", "unwrap_masked_poisson", "unwrap_seed_growth"]


def wrap(x: np.ndarray) -> np.ndarray:
    """Wrap to ``(-pi, pi]``."""
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


def _wrap_diff(phi: np.ndarray, axis: int) -> np.ndarray:
    """Wrapped first difference along ``axis`` (forward differences)."""
    d = np.diff(phi, axis=axis)
    return wrap(d)


def unwrap_dct(phi: np.ndarray) -> np.ndarray:
    """Least-squares phase unwrapping with a DCT Poisson solver (full frame)."""
    from scipy.fft import dctn, idctn

    phi = np.asarray(phi, dtype=float)
    m, n = phi.shape
    dx = _wrap_diff(phi, axis=1)
    dy = _wrap_diff(phi, axis=0)

    rho = np.zeros_like(phi)
    rho[:, 1:] += dx
    rho[:, :-1] -= dx
    rho[1:, :] += dy
    rho[:-1, :] -= dy

    ii = np.arange(m)[:, None]
    jj = np.arange(n)[None, :]
    denom = 2.0 * (2.0 - np.cos(np.pi * ii / m) - np.cos(np.pi * jj / n))
    denom[0, 0] = 1.0
    spec = dctn(rho, type=2, norm="ortho") / denom
    spec[0, 0] = 0.0
    psi = idctn(spec, type=2, norm="ortho")
    return psi


def _masked_divergence(phi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divergence of the wrapped gradients, using only fully in-mask edges.

    This is the right-hand side of the least-squares normal equations of the
    masked unwrapping problem: each edge contributes its wrapped gradient
    ``g`` with ``+g`` at its start pixel and ``-g`` at its end pixel.
    """
    rho = np.zeros_like(phi)
    for di, dj in ((0, 1), (1, 0)):
        nbr = np.roll(phi, (-di, -dj), axis=(0, 1))
        valid = mask & np.roll(mask, (-di, -dj), axis=(0, 1))
        if di:
            valid[-1, :] = False
        if dj:
            valid[:, -1] = False
        g = np.where(valid, wrap(nbr - phi), 0.0)
        rho = rho + g
        rho = rho - np.roll(np.where(valid, g, 0.0), (di, dj), axis=(0, 1))
    return rho


def _neighbour_terms(
    a: np.ndarray, m: np.ndarray, di: int, dj: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shift ``a``/``m`` by ``(di, dj)`` with the periodic wrap removed.

    ``np.roll`` makes row ``0`` the neighbour of row ``-1`` and column ``0``
    the neighbour of column ``-1``.  On the *detector* those pixels are not
    neighbours, so the wrapped entries are zeroed out; otherwise the iteration
    would impose a periodic topology on the border and couple phantom edges
    that the sparse branch ignores.
    """
    rolled = np.roll(a, (di, dj), axis=(0, 1))
    rolled_m = np.roll(m, (di, dj), axis=(0, 1))
    if di > 0:
        rolled[0, :] = 0.0
        rolled_m[0, :] = 0.0
    elif di < 0:
        rolled[-1, :] = 0.0
        rolled_m[-1, :] = 0.0
    if dj > 0:
        rolled[:, 0] = 0.0
        rolled_m[:, 0] = 0.0
    elif dj < 0:
        rolled[:, -1] = 0.0
        rolled_m[:, -1] = 0.0
    return rolled, rolled_m


def unwrap_masked_poisson(
    phi: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    method: str = "sparse",
) -> np.ndarray:
    """Least-squares (Poisson) unwrapping restricted to ``mask``.

    Solves the Ghiglia-Romero normal equations

        ``sum_{in-mask neighbours}(psi_i - psi_j) = -rho_i``

    with ``rho`` the divergence of the wrapped gradient field returned by
    :func:`_masked_divergence` (only edges whose two endpoints are in the mask
    and inside the grid contribute).  The system is singular by a constant; the
    returned solution has zero mean over the mask.  Pixels outside the mask are
    NaN.

    ``method="sparse"`` assembles and solves the system directly;
    ``method="jacobi"`` runs the same system as a fixed-point iteration
    (slower, kept as an independent implementation and cross-checked in the
    tests).  Both must return the same map up to the constant.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be a 2-D array")
    if mask is None:
        mask = np.ones_like(phi, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != phi.shape:
        raise ValueError(f"mask must have shape {phi.shape}, got {mask.shape}")
    if method not in ("sparse", "jacobi"):
        raise ValueError(f"method must be 'sparse' or 'jacobi', got {method!r}")
    if not np.any(mask):
        return np.full(phi.shape, np.nan)

    rho = _masked_divergence(phi, mask)

    from scipy import ndimage

    # Every 4-connected component carries its own additive constant, and the
    # wrapped gradients of the mask never tie them together.  Both methods are
    # limited in exactly the same way, so the limitation is reported once here
    # instead of only on the branch that happens to notice it while pinning.
    labels, n_comp = ndimage.label(mask)
    if n_comp > 1:
        warnings.warn(
            f"mask has {n_comp} disconnected regions: their relative "
            "offsets are not determined by the wrapped gradients, so each "
            "region keeps an independent constant",
            stacklevel=2,
        )

    if method == "jacobi":
        psi = np.where(mask, phi, 0.0).copy()
        m_float = mask.astype(float)
        for _ in range(20000):
            neigh = np.zeros_like(psi)
            cnt = np.zeros_like(psi)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rolled, rolled_m = _neighbour_terms(
                    np.where(mask, psi, 0.0), m_float, di, dj
                )
                neigh = neigh + rolled * rolled_m
                cnt = cnt + rolled_m
            # A missing neighbour (outside the mask or outside the grid) falls
            # back on the centre value, and those terms cancel out of
            # ``cnt * psi - neigh``, which is exactly the equation the sparse
            # branch assembles.  ``rho`` enters with the sign of ``rhs = -rho``
            # there; using ``+rho`` solved the adjoint system and returned
            # ``-psi`` (a global sign, hence not a gauge).
            psi_new = np.where(
                mask, (neigh + (4.0 - cnt) * psi - rho) / 4.0, 0.0
            )
            step = float(np.max(np.abs(psi_new - psi)[mask]))
            psi = psi_new
            if step < 1e-10:
                break
        return psi - psi[mask].mean()

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    idx = -np.ones(phi.shape, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    idx[ys, xs] = np.arange(len(ys))
    n_pix = len(ys)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(zip(ys, xs)):
        n_nb = 0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < phi.shape[0] and 0 <= jj < phi.shape[1] and mask[ii, jj]:
                rows.append(k)
                cols.append(int(idx[ii, jj]))
                vals.append(-1.0)
                n_nb += 1
        rows.append(k)
        cols.append(k)
        vals.append(float(n_nb))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pix, n_pix))
    rhs = -rho[ys, xs]
    if n_pix > 1:
        # Every 4-connected component has its own constant null vector, so one
        # pinned pixel *per component* is needed to make the system
        # non-singular; a single global pin leaves it rank deficient for a
        # disconnected mask.  The relative offsets of the components are not
        # recoverable from the wrapped gradients (reported above), so each is
        # left at its pinned value and the result is re-centred globally below.
        pins = [int(np.argmax(labels[ys, xs] == lab)) for lab in range(1, n_comp + 1)]
        A = A.tolil()
        rhs = rhs.copy()
        for k in pins:
            A[k, :] = 0.0
            A[k, k] = 1.0
            rhs[k] = 0.0
        A = A.tocsr()
    sol = spsolve(A, rhs)
    if not np.all(np.isfinite(sol)):
        # Nearly unreachable: every connected component is pinned above, so the
        # matrix is non-singular.  Fall back to least squares rather than
        # raising, but say so -- an unexplained NaN would be worse than an
        # algorithm switch the caller can see.  This changes the algorithm (and
        # with it the residual), so it must not happen silently.
        warnings.warn(
            "the pinned masked-Poisson system was not solvable by the sparse "
            "LU factorisation; falling back to lsqr, which solves the same "
            "system in the least-squares sense and may differ in the residual",
            stacklevel=2,
        )
        from scipy.sparse.linalg import lsqr

        sol = lsqr(A, rhs, atol=1e-13, btol=1e-13, iter_lim=20000)[0]
    out = np.full(phi.shape, np.nan)
    out[ys, xs] = sol - sol.mean()
    return out


def unwrap_seed_growth(
    phi: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    seed: tuple[int, int] | None = None,
) -> np.ndarray:
    """Path-following unwrapping (breadth-first) over the valid mask.

    The result is anchored at ``seed`` (the default is the mask pixel closest
    to the grid centre).  Any 4-connected component that the seed cannot reach
    is anchored independently at its own centre-most pixel --- the relative
    offsets of disconnected regions are not determined by the wrapped
    gradients --- and a warning is issued.  An empty mask returns all NaN, as
    :func:`unwrap_masked_poisson` does.
    """
    from collections import deque

    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be a 2-D array")
    if mask is None:
        mask = np.ones_like(phi, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != phi.shape:
        raise ValueError(f"mask must have shape {phi.shape}, got {mask.shape}")
    out = np.full(phi.shape, np.nan)
    m, n = phi.shape
    if not np.any(mask):
        return out

    def grow(start: tuple[int, int]) -> None:
        """Unwrap everything connected to ``start``, which must be seeded."""
        q = deque([start])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if (
                    0 <= ii < m
                    and 0 <= jj < n
                    and mask[ii, jj]
                    and np.isnan(out[ii, jj])
                ):
                    out[ii, jj] = out[i, j] + wrap(phi[ii, jj] - phi[i, j])
                    q.append((ii, jj))

    if seed is None:
        ys, xs = np.nonzero(mask)
        cy, cx = m / 2.0, n / 2.0
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        seed = (int(ys[k]), int(xs[k]))
    elif not (0 <= seed[0] < m and 0 <= seed[1] < n):
        raise ValueError(f"seed {tuple(seed)} is outside the {phi.shape} grid")
    elif not mask[seed]:
        raise ValueError("seed is outside the mask")

    out[seed] = phi[seed]
    grow(seed)
    n_comp = 1
    todo = mask & np.isnan(out)
    while todo.any():
        ys, xs = np.nonzero(todo)
        cy, cx = m / 2.0, n / 2.0
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        start = (int(ys[k]), int(xs[k]))
        out[start] = phi[start]
        grow(start)
        n_comp += 1
        todo = mask & np.isnan(out)
    if n_comp > 1:
        warnings.warn(
            f"mask has {n_comp} disconnected regions: only differences *within* "
            "a region are unwrapped, each region is anchored at its own seed",
            stacklevel=2,
        )
    return out
