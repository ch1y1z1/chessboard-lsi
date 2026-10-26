"""BUV0725 physical parameters. Lengths are metres, angles are degrees."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class SystemConfig:
    wavelength: float = 6.7e-9
    error_mode: str = "ON"
    N_lattice: int = 120
    N_super_lattice: int = 10
    super_lattice_period: float = 3.35e-6
    shift_z: float = -100e-6
    tilted_angle: float = 0.0
    tilt_axis_x: float = 1.0
    tilt_axis_y: float = 0.0
    f1: float = 1.2
    f2: float = 0.24
    NA_in: float = 0.00769
    NA_out: float = 0.02
    Nz: int = 5
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def profile(cls, name="reference"):
        if name == "reference":
            return cls()
        if name == "retrieval":
            return cls(error_mode="OFF", shift_z=0.0)
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


@dataclass(frozen=True)
class Scene:
    """Simulation truth and measurement noise; never passed to either solver."""

    indices: tuple[int, ...] = (7, 25, 11, 15)
    coefficients: tuple[float, ...] = (.25, .15, .05, .2)
    rms: float | None = .1
    snr_db: float | None = None
    seed: int = 42

    def coefficients_for(self, indices, Nz=5):
        import numpy as np
        if len(self.indices) != len(self.coefficients) or len(set(self.indices)) != len(self.indices):
            raise ValueError("scene modes and coefficients must match and be distinct")
        if any(j < 1 or j >= (Nz + 1)**2 for j in self.indices):
            raise ValueError("scene mode outside the configured non-piston basis")
        truth = dict(zip(self.indices, self.coefficients))
        missing = {j for j, c in truth.items() if c and j not in indices}
        if missing:
            raise ValueError(f"nonzero scene modes {sorted(missing)} are absent from fit indices")
        return np.array([truth.get(j, 0.) for j in indices], dtype=float)


@dataclass(frozen=True)
class FitConfig:
    """Unknown modes and the finite optimization budget."""

    indices: tuple[int, ...] = (7, 25, 11, 15)
    starts: int = 2
    samples: int | None = 8192
    max_nfev: int = 60
    screen_initials: int = 0
    seed: int = 42
