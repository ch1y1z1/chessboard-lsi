"""BUV0725 physical parameters. Lengths are metres, angles are degrees."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class DiffractionConfig:
    wavelength: float = 6.7e-9
    distance_stage1: float = 2.0
    error_mode: str = "ON"
    N_lattice: int = 120
    N_super_lattice: int = 10
    super_lattice_period: float = 3.35e-6
    hole_diameter: float = 50e-9
    lattice_distance: float = 150e-9
    N_hole_rows: int = 1
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_z: float = -100e-6
    tilted_angle: float = 0.0
    tilt_axis_x: float = 1.0
    tilt_axis_y: float = 0.0
    f1: float = 1.2
    f2: float = 0.24
    NA_in: float = 0.00769
    NA_out: float = 0.02
    Nz: int = 5
    basis_factor: float = 0.01
    c_Zernike: tuple[float, ...] = (0.25, 0.15, 0.05, 0.2, 0.0)
    n_radial: tuple[int, ...] = (2, 5, 3, 3, 1)
    l_azimuthal: tuple[int, ...] = (1, -5, -1, 3, 0)
    grating_type: str = "Square"
    grating_period: float = 1.34e-6
    duty_cycle: float = 50.0
    z_grating: float = 134e-6
    dx_grating: float = 16.75e-9
    up_sample_grating: int = 2
    shift_x_grating: float = 0.0
    shift_y_grating: float = 0.0
    shift_z_grating: float = 0.0
    tilted_angle_grating: float = 0.0
    tilt_axis_x_grating: float = 1.0
    tilt_axis_y_grating: float = 0.0
    z_cam: float = 53.6e-3
    dx_cam: float = 10e-6
    tilted_angle_cam: float = 0.0
    tilt_axis_x_cam: float = 1.0
    tilt_axis_y_cam: float = 0.0

    def __post_init__(self):
        for name in ("c_Zernike", "n_radial", "l_azimuthal"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len({len(self.c_Zernike), len(self.n_radial), len(self.l_azimuthal)}) != 1:
            raise ValueError("coefficient and mode lists must have equal lengths")
        if self.N_lattice < 2 or self.N_super_lattice < 3 or self.up_sample_grating < 1:
            raise ValueError("invalid grid dimensions")
        if not 0 <= self.NA_in < self.NA_out or self.NA_out <= 0:
            raise ValueError("require 0 <= NA_in < NA_out")
        for key in ("wavelength", "f1", "f2", "grating_period", "dx_grating", "z_cam", "dx_cam"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.z_grating - self.shift_z_grating <= 0 or not 0 < self.duty_cycle < 100:
            raise ValueError("invalid grating distance or duty cycle")
        if self.grating_type not in ("Square", "Diamond") or self.error_mode not in ("ON", "OFF"):
            raise ValueError("unsupported reference grating or error mode")
        if int(self.grating_period * self.duty_cycle / 100 / self.dx_grating) < 1:
            raise ValueError("grating aperture is not resolved")
        for suffix in ("", "_grating", "_cam"):
            if getattr(self, "tilt_axis_x" + suffix) == getattr(self, "tilt_axis_y" + suffix) == 0:
                raise ValueError("tilt axis cannot be zero")

    @property
    def n(self):
        return self.N_lattice * self.N_super_lattice

    @property
    def sources(self):
        return tuple((i, j) for i in range(1, self.N_super_lattice - 1)
                     for j in range(1, self.N_super_lattice - 1))

    @property
    def epsilon(self):
        return self.NA_in / self.NA_out

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def profile(cls, name="reference"):
        if name == "reference":
            return cls()
        if name == "retrieval":
            return cls(error_mode="OFF", shift_z=0.0,
                       c_Zernike=(0.25, 0.15, 0.05, 0.2),
                       n_radial=(2, 5, 3, 3), l_azimuthal=(1, -5, -1, 3))
        if name == "quick":
            # Same physical FOVs and all 64 sources, coarser numerical sampling.
            return replace(cls(), N_lattice=30, dx_cam=40e-6, dx_grating=67e-9)
        if name == "test":
            # A distinct four-source scene, solely for tests and CLI smoke checks.
            return replace(cls(), N_lattice=20, N_super_lattice=4,
                           dx_cam=100e-6, dx_grating=67e-9)
        raise ValueError(f"unknown profile: {name}")

    def with_angle(self, component, angle, axis="x"):
        suffix = {"object": "", "grating": "_grating", "camera": "_cam"}[component]
        ax, ay = {"x": (1, 0), "y": (0, 1), "diagonal": (1, 1)}[axis]
        return replace(self, **{"tilted_angle" + suffix: float(angle),
                                "tilt_axis_x" + suffix: ax,
                                "tilt_axis_y" + suffix: ay})
