import numpy as np
# import matplotlib.pyplot as plt
import torch
# from scipy.ndimage import zoom

class Stage1(torch.nn.Module):
    # Attention:
    # Please make sure the input plane for simulation is large enough for anti-aliasing
    # Ref. Fast numerical scalar diffraction simulation for optical neural networks, A Master Thesis of Christian Eder at Aalen University (2021)
    def __init__(self, config):
        super(Stage1, self).__init__()
        # Physical parameters
        self.lam = config.wavelength # wavelength [m]
        self.k = 2 * np.pi / self.lam

        # Parameters of the optical system - from the point source to the mask
        self.z = config.distance_stage1  # propagation distance [m]
        self.error_mode = config.error_mode # OFF mode, neglect the misalignment of hole array and of grating

        # Parameters of the hole mask
        self.N_lattice = config.N_lattice
        self.super_lattice_period = config.super_lattice_period
        self.dx_mask = self.super_lattice_period / self.N_lattice
        self.small_circle_radius = int(config.hole_diameter / 2 / self.dx_mask)
        self.lattice_spacing = int((config.lattice_distance + config.hole_diameter / 2) / self.dx_mask)
        self.N_hole_rows = config.N_hole_rows
        self.N_super_lattice = config.N_super_lattice

        # print("Stage 1: the simulation parameters for the point source diffraction")
        # print("dx_mask = ", "{:.4g}".format(self.dx_mask * 1e9), "nm")
        # print("L_mask = ", self.N_lattice * self.dx_mask * self.N_super_lattice * 1e6, "um")

        # spatial shift of the hole mask, unit of [m] in config.
        self.shift_x = int(config.shift_x / self.dx_mask) # unit of pixel
        self.shift_y = int(config.shift_y / self.dx_mask)  # unit of pixel

        # The distance from the object plane to reference plane
        # To analyze the shift along z, the field first propagates from the mask to the reference plane,
        # and then backwards to the object plane
        self.z_ref = 1e-3  # unit of [m]

        # shift_z is the distance from the object plane to the mask, unit of [m].
        # Limited to the range +0.5mm (close to objective) ~ -0.5mm (close to source).
        self.z_B = self.z_ref - config.shift_z
        
        # tilted hole mask
        self.Na = self.N_lattice * self.N_super_lattice
        xa = (np.arange(self.Na) - self.Na // 2) * self.dx_mask
        self.U, self.V = np.meshgrid(xa, xa, indexing="xy")

        kxy = 2.0 * np.pi * np.fft.fftfreq(self.Na, d=self.dx_mask)
        KX, KY = np.meshgrid(kxy, kxy, indexing="xy")
        kxy2 = KX**2 + KY**2

        # Propagating waves only
        kz = np.zeros_like(KX)
        valid = kxy2 <= self.k**2
        kz[valid] = np.sqrt(self.k**2 - kxy2[valid])

        # Propagator to the reference plane
        self.H_mask2ref = np.zeros_like(KX, dtype=complex)
        self.H_mask2ref[valid] = np.exp(1j * kz[valid] * self.z_B)

        # Propagator back to the object plane
        self.H_ref2obj = np.zeros_like(KX, dtype=complex)
        self.H_ref2obj[valid] = np.exp(-1j * kz[valid] * self.z_ref)

        # Axis of rotation in the xy plane.
        # Examples:
        #   (1, 0): plane A rotated about x axis
        #   (0, 1): plane A rotated about y axis
        #   (1, 1): plane A rotated about diagonal axis
        # can't be (0,0)
        self.tilt_axis_x = config.tilt_axis_x
        self.tilt_axis_y = config.tilt_axis_y

        self.tilt_angle = config.tilted_angle * np.pi / 180  # unit of degree

        # self.mask = self._generate_hex_rows()

        # point source
        self.mask = np.zeros((self.Na, self.Na), dtype=complex)
        self.mask[self.N_lattice//2, self.N_lattice//2] = 1

    def _small_tilt_propagate(self, U_A):
        """
        Efficient propagation from a slightly tilted plane A to plane B.
        This assumes:
            1. Tilt is small, e.g. < 2 degrees.
            2. Evanescent waves are negligible.
            3. The field is paraxial or moderately paraxial.
            4. U_A is sampled approximately on the same transverse x,y grid.

        Parameters
        ----------
        U_A : 2D complex array
            Field measured or defined on tilted plane A.

        Returns
        -------
        U_B : 2D complex array
            Field on plane B.

        """
        # Normalize the rotation axis in the xy plane
        axis = np.array([self.tilt_axis_x, self.tilt_axis_y], dtype=float)
        axis = axis / np.linalg.norm(axis)

        ax, ay = axis

        # This vector is perpendicular to the rotation axis.
        px = -ay
        py = ax

        # Height of tilted plane A.
        # For small tilt:
        #     z_A(x,y) = tan(theta) * distance_perpendicular_to_axis
        z_A = np.tan(self.tilt_angle) * (px * self.U + py * self.V)

        # Local phase correction:
        # Field on tilted plane A: U_A(x,y) approximately equals U_0(x,y) * exp(i k0 z_A)
        # Therefore, U_0(x,y) approximately equals U_A(x,y) * exp(-i k0 z_A)
        U_0 = U_A * np.exp(-1j * self.k * z_A)

        # Now use normal angular spectrum method from z = 0 to z = z_B
        U_B = self._angular_spectrum_propagate(U0=U_0, H=self.H_mask2ref)

        return U_B

    def _angular_spectrum_propagate(self, U0, H):
        """
        Standard angular spectrum method.

        U0 : 2D complex array
            Field on input xy plane.

        dx, dy : float
            Sampling intervals.

        wavelength : float
            Wavelength.

        z : float
            Propagation distance.

        Returns
        -------
        Uz : 2D complex array
            Field after propagation distance z.
        """
        A0 = np.fft.fft2(U0)
        Az = A0 * H
        Uz = np.fft.ifft2(Az)

        return Uz

    def _generate_hex_rows(self):
        # find the center
        row_center = int(self.N_lattice / 2)
        col_center = row_center

        # create the 2d image
        pattern = np.zeros((self.N_lattice, self.N_lattice), dtype=np.uint8)

        # Pixel coordinate grid
        rows, cols = np.indices((self.N_lattice, self.N_lattice))

        # Hexagonal lattice geometry
        dx = self.lattice_spacing
        dy = int(np.sqrt(3) / 2 * self.lattice_spacing)
        col_offset = 0

        for row in np.arange(self.N_hole_rows):
            c0 = col_center + col_offset - dx * (self.N_hole_rows - row - 1) # two line shift one
            num_col = 2 * self.N_hole_rows - row - 1

            r = row_center + row * dy
            for col in np.arange(num_col):
                c = c0 + col * dx
                circle_mask = (rows - r) ** 2 + (cols - c) ** 2 <= self.small_circle_radius ** 2
                pattern[circle_mask] = 1

            r = row_center - row * dy
            for col in np.arange(num_col):
                c = c0 + col * dx
                circle_mask = (rows - r) ** 2 + (cols - c) ** 2 <= self.small_circle_radius ** 2
                pattern[circle_mask] = 1

            col_offset = col_offset - int(dx / 2)

        # mask = np.tile(pattern, (self.N_super_lattice, self.N_super_lattice))
        mask = np.pad(
            pattern,
    ((0, self.N_lattice*(self.N_super_lattice - 1)), (0, self.N_lattice*(self.N_super_lattice - 1))),
        'constant'
        )

        return mask

    def _point_source_far_field(self, offset_x, offset_y, offset_z):
        """
        Calculate the far-field optical field from an on-axis point source
        on a distant flat screen.

        The phase variation becomes obvious only if the offset is on the scale of [um]
        """
        screen_size_x = self.Na * self.dx_mask
        screen_size_y = screen_size_x
        z = self.z + offset_z
        
        x = np.linspace(
            -screen_size_x / 2 + offset_x,
            screen_size_x / 2 + offset_x,
            self.Na,
            endpoint=False,
            dtype=np.float32,
        )

        y = np.linspace(
            -screen_size_y / 2 + offset_y,
            screen_size_y / 2 + offset_y,
            self.Na,
            endpoint=False,
            dtype=np.float32,
        )

        # Far-field phase:
        # phase = k*z + k*(x^2 + y^2)/(2z)
        #
        # Use broadcasting:
        # y[:, None] has shape (Ny, 1)
        # x[None, :] has shape (1, Nx)
        phase = self.k * z + self.k / (2.0 * z) * (y[:, None] ** 2 + x[None, :] ** 2)

        U_in = (1 / z) * np.exp(1j * phase)

        mask = self._generate_hex_rows() # 1st plan, 20260518
        mask = np.roll(mask, (self.shift_x, self.shift_y), axis=(1, 0))

        U_out = U_in * mask

        return U_out

    def forward(self, nx = 0, ny = 0):
        if (self.error_mode == "ON"):
            # sx = np.random.randint(-1, high=2)
            # sy = np.random.randint(-1, high=2)
            # U_in = np.roll(self.mask, (nx*self.N_lattice+sx, ny*self.N_lattice+sy), axis=(1, 0))

            U_in = np.roll(self.mask, (nx * self.N_lattice, ny * self.N_lattice), axis=(1, 0))
            U_ref = self._small_tilt_propagate(U_in)
            U_out = self._angular_spectrum_propagate(U0=U_ref, H=self.H_ref2obj)
        else:
            U_out = np.roll(self.mask, (nx*self.N_lattice, ny*self.N_lattice), axis=(1, 0))

        return U_out