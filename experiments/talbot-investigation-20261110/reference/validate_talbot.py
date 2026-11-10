"""Validate the paraxial Talbot baseline and propagate its 64 physical sources.

Run from any directory with NumPy, SciPy and PyTorch installed:
    python BUV0725/validate_talbot.py --output output/talbot-validation.json

Edge energy is a truncation diagnostic, not a bound on reconstruction error.
The ON-mode distance uses the paraxial M**2 axial-focus correction.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grating2cam import Stage3
from obj_sys_ft import Stage2_ab
from point2plate import Stage1
from source_grid import source_indices


def edge_energy(field):
    intensity = np.abs(field) ** 2
    if not np.isfinite(intensity).all() or intensity.sum() <= 0:
        raise ValueError("Propagation produced a nonfinite or empty field")
    border = max(1, round(0.05 * intensity.shape[0]))
    return float(1 - intensity[border:-border, border:-border].sum() / intensity.sum())


def validate(config):
    indices = source_indices(config)
    magnification = config.f2 / config.f1
    focus_shift = magnification**2 * config.shift_z if config.error_mode == "ON" else 0.0
    radius = config.z_grating - config.shift_z_grating - focus_shift
    effective_distance = radius * config.z_cam / (radius + config.z_cam)
    pixels = int(config.grating_period / config.dx_grating)
    actual_period = pixels * config.dx_grating
    talbot = 2 * actual_period**2 / config.wavelength
    ratio = effective_distance / talbot
    assert np.isclose(actual_period, config.grating_period, rtol=1e-12, atol=0)
    assert np.isclose(ratio, 1.0, rtol=1e-10, atol=0), ratio

    stage1 = Stage1(config)
    stage2 = Stage2_ab(config, mode="aberration")
    stage3 = Stage3(config, mode="camera")

    # Test the actual rasterized aperture over one periodic unit cell.
    cell = stage3.grating[:pixels, :pixels]
    frequency = np.fft.fftfreq(pixels, d=config.dx_grating)
    fx, fy = np.meshgrid(frequency, frequency)
    transfer = np.exp(-1j * np.pi * config.wavelength * effective_distance * (fx**2 + fy**2))
    replica = np.fft.ifft2(np.fft.fft2(cell) * transfer)
    replica_error = float(np.linalg.norm(replica - cell) / np.linalg.norm(cell))
    assert replica_error < 1e-8, replica_error

    # Preserve original source coordinates, including the half-cell origin.
    n = config.N_lattice * config.N_super_lattice
    centers = [(i + 0.5 - config.N_super_lattice / 2) * config.super_lattice_period
               for i in indices]
    assert np.allclose(centers, (np.arange(8) - 3.5) * config.super_lattice_period,
                       rtol=1e-12, atol=0)
    image_edges, grating_edges, camera_edges = [], [], []
    summed_intensity = np.zeros((n, n))
    for ii in indices:
        for jj in indices:
            image = stage2.forward(stage1.forward(nx=ii, ny=jj))
            image_edges.append(edge_energy(image))
            grating = stage3._img2grating(image)
            grating_edges.append(edge_energy(grating))
            camera = stage3._grating2cam(grating * stage3.grating)
            camera_edges.append(edge_energy(camera))
            summed_intensity += np.abs(camera) ** 2
        print(f"Completed source row {ii}: {len(image_edges)}/64", flush=True)

    # Acceptance limits chosen for this baseline: 0.2% at the padded image
    # boundary and 0.1% at the grating boundary (outer 5% on every side).
    assert len(image_edges) == 64
    assert max(image_edges) < 0.002, max(image_edges)
    assert max(grating_edges) < 0.001, max(grating_edges)
    assert np.isfinite(summed_intensity).all() and summed_intensity.max() > 0
    half_grating = n * config.up_sample_grating * config.dx_grating / 2
    outer_radius = max(abs(c) for c in centers) * abs(magnification) + config.NA_out / abs(magnification) * radius
    assert half_grating > outer_radius
    result = {
        "config": vars(config),
        "source_count": len(image_edges),
        "source_indices": list(indices),
        "period_pixels": pixels,
        "actual_period_m": actual_period,
        "paraxial_focus_to_grating_m": radius,
        "effective_distance_m": effective_distance,
        "full_talbot_length_m": talbot,
        "talbot_ratio": ratio,
        "periodic_cell_relative_field_error": replica_error,
        "image_window_m": n * stage3.dx_img,
        "grating_window_m": 2 * half_grating,
        "geometric_grating_margin_m": half_grating - outer_radius,
        "image_outer_5pct_energy_max": max(image_edges),
        "grating_outer_5pct_energy_max": max(grating_edges),
        "camera_outer_5pct_energy_max": max(camera_edges),
        "all_camera_fields_finite": True,
        "limits": "Paraxial distance and periodic-cell self-imaging; finite-field edge diagnostics; no inverse reconstruction claim.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SimpleNamespace(**json.loads(Path(__file__).with_name("config_file.json").read_text()))
    result = validate(config)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
