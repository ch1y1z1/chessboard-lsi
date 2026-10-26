"""RMS-normalized annular Zernike basis in the references' zero-based order."""
from functools import lru_cache
import numpy as np


def mode_labels(Nz=5):
    """j=0..35: iterate nn=0..5, l=-nn..nn, n=2*nn-|l|."""
    return tuple((2 * nn - abs(l), l) for nn in range(Nz + 1) for l in range(-nn, nn + 1))


@lru_cache(maxsize=128)
def radial_coefficients(n, l, epsilon):
    m = abs(l)
    if n < m or (n - m) % 2 or not 0 <= epsilon < 1:
        raise ValueError("invalid annular Zernike mode or obscuration")
    q = (n - m) // 2
    powers = m + 2 * np.arange(q + 1)
    p_sum = powers[:, None] + powers[None, :]
    gram = 2 * (1 - epsilon ** (p_sum + 2)) / ((1 - epsilon**2) * (p_sum + 2))
    previous = []
    for k in range(q + 1):
        c = np.zeros(q + 1)
        c[k] = 1.0
        for _ in range(2):
            for v in previous:
                c -= (v @ gram @ c) * v
        norm = c @ gram @ c
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("annular orthogonalization failed")
        c /= np.sqrt(norm)
        if c[k] < 0:
            c *= -1
        previous.append(c)
    return powers, previous[-1]


def annular_zernike(n, l, epsilon, rho, theta, outside_value=0.0):
    rho, theta = np.asarray(rho), np.asarray(theta)
    if rho.shape != theta.shape:
        raise ValueError("rho and theta must have the same shape")
    powers, coeffs = radial_coefficients(int(n), int(l), float(epsilon))
    radial = sum(c * rho**p for c, p in zip(coeffs, powers))
    angular = (np.sqrt(2) * np.cos(l * theta) if l > 0 else
               np.sqrt(2) * np.sin(abs(l) * theta) if l < 0 else 1.0)
    inside = (rho >= epsilon) & (rho <= 1)
    return np.where(inside, radial * angular, outside_value)


def basis(indices, rho, theta, epsilon, Nz=5):
    labels = mode_labels(Nz)
    if any(i < 0 or i >= len(labels) for i in indices):
        raise ValueError("mode index outside the configured basis")
    return np.stack([annular_zernike(*labels[i], epsilon, rho, theta) for i in indices])
