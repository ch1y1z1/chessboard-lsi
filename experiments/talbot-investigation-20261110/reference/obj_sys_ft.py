import numpy as np
# import matplotlib.pyplot as plt
import torch

class Stage2_ab(torch.nn.Module):
    def __init__(self, config, mode = "basis"):
        super(Stage2_ab, self).__init__()
        # ============================================================
        # 1. Physical parameters
        # ============================================================
        self.lam = config.wavelength  # wavelength [m]
        self.k = 2 * np.pi / self.lam

        # Parameters of the objective as a 4f model
        self.f1 = config.f1  # first lens focal length [m]
        self.f2 = config.f2  # second lens focal length [m]
        self.M = -self.f2 / self.f1  # 4f magnification

        # print()
        # print("Stage 2: the simulation parameters for the objective")
        # print("Magnification M =", np.abs(self.M))

        # ============================================================
        # 2. Fourier-plane grid
        # ============================================================
        # Object-plane grid
        self.N = config.N_lattice * config.N_super_lattice
        self.dx_in = config.super_lattice_period / config.N_lattice

        # Spatial-frequency grid
        fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx_in))
        FX, FY = np.meshgrid(fx, fx)
        self.dfx = fx[1] - fx[0]
        self.dfy = self.dfx

        # Fourier-plane physical coordinates
        XF = self.lam * self.f1 * FX
        YF = self.lam * self.f1 * FY

        # dXF = self.lam * self.f1 * self.dfx
        # print("Fourier-plane dx =", "{:.4g}".format(dXF * 1e6), "um")
        
        R = np.sqrt(XF ** 2 + YF ** 2)
        self.theta = np.arctan2(YF, XF)

        # radius of the pupil at the Fourier plane
        Rp = config.f1 * config.NA_out  # NA should be smaller than 0.2
        # Rc = config.f1 * config.NA_in
        self.epsilon = config.NA_in / config.NA_out
        # print("Pupil outer radius =", "{:.4g}".format(Rp * 1e3), "mm")
        # print("Pupil inner radius =", "{:.4g}".format(Rc * 1e3), "mm")

        self.rho = R / Rp
        self.pupil = (self.rho >= self.epsilon) & (self.rho <= 1.0)

        # ============================================================
        # Fourier-plane mask: pupil + aberration phase
        # ============================================================
        self.basis_factor = config.basis_factor
        self.mode = mode

        # Reference wavefront
        if self.mode == "reference":
            self.H = self.pupil

        if self.mode == "aberration":
            # Aberration coefficients in waves.
            c_Zernike = config.c_Zernike
            n_radial = config.n_radial
            l_azimuthal = config.l_azimuthal

            #  The dimension of c, n, and l must be the same.
            num_poly = len(c_Zernike)

            # OPD
            W = np.zeros((self.N, self.N), dtype=float)

            for idx in range(num_poly):
                # An index framework used in
                # https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/Zernike_Polynomials_For_The_Web.pdf
                m = np.abs(l_azimuthal[idx])
                n = 2 * n_radial[idx] - m

                W += c_Zernike[idx] * self._annular_zernike(n=n, l=l_azimuthal[idx])

            # Avoid undefined phase outside pupil by multiplying with pupil
            phase = 2 * np.pi * W
            self.H = self.pupil * np.exp(1j * phase)


    def _annular_zernike(self, n, l, outside_value=0.0):
        """
        Calculate a real-valued, RMS-normalized annular Zernike polynomial.

        Parameters
        ----------
        n : int
            Radial index. Must satisfy n >= 0.

        l : int
            Signed azimuthal index. It must satisfy

                abs(l) <= n
                n - abs(l) is even

            Angular convention:
                l > 0 : cos(l * theta)
                l < 0 : sin(abs(l) * theta)
                l = 0 : no angular dependence

        epsilon : float
            Central obscuration ratio. Must satisfy

                0 <= epsilon < 1

            The annular pupil is defined by

                epsilon <= rho <= 1.

        rho : numpy.ndarray
            Two-dimensional array containing normalized radial coordinates.

        theta : numpy.ndarray
            Two-dimensional array containing azimuthal angles in radians.
            Must have the same shape as rho.

        outside_value : float, optional
            Value assigned outside the annular pupil. Default is 0.0.
            Use np.nan instead when a masked-looking output is preferred.

        Returns
        -------
        Z : numpy.ndarray
            Two-dimensional array containing the annular Zernike polynomial.

        Notes
        -----
        The polynomial is normalized so that

            mean(Z**2 over the clear annular pupil) = 1

        in the continuous-area sense.

        The radial polynomial is generated by orthonormalizing the monomials

            rho**m, rho**(m+2), ..., rho**n

        with respect to the annular radial inner product

            <f, g> = 2/(1-epsilon**2)
                     * integral_epsilon^1 f(rho) g(rho) rho d rho.

        For epsilon = 0, the result reduces to the conventional RMS-normalized
        Zernike polynomial.
        """

        # ---------------------------------------------------------------
        # Validate scalar inputs
        # ---------------------------------------------------------------
        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer.")

        if not isinstance(l, (int, np.integer)):
            raise TypeError("l must be an integer.")

        if n < 0:
            raise ValueError("n must be nonnegative.")

        m = abs(l)

        if m > n:
            raise ValueError("The indices must satisfy abs(l) <= n.")

        if (n - m) % 2 != 0:
            raise ValueError("The indices must satisfy n - abs(l) = even.")

        if not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite.")

        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must satisfy 0 <= epsilon < 1.")

        # ---------------------------------------------------------------
        # Powers appearing in the radial polynomial:
        #
        # R_n^m(rho) = c_0 rho^m + c_1 rho^(m+2) + ... + c_q rho^n
        # ---------------------------------------------------------------
        q = (n - m) // 2
        powers = m + 2 * np.arange(q + 1)

        # ---------------------------------------------------------------
        # Construct the exact Gram matrix of the radial monomials.
        #
        # G_ij = <rho^p_i, rho^p_j>
        #
        #      = 2/(1-epsilon^2)
        #        * integral_epsilon^1 rho^(p_i+p_j+1) d rho
        #
        #      = 2[1-epsilon^(p_i+p_j+2)]
        #        / [(1-epsilon^2)(p_i+p_j+2)]
        # ---------------------------------------------------------------
        p_sum = powers[:, None] + powers[None, :]

        gram_matrix = (
                2.0
                * (1.0 - self.epsilon ** (p_sum + 2))
                / ((1.0 - self.epsilon ** 2) * (p_sum + 2))
        )

        # ---------------------------------------------------------------
        # Modified Gram-Schmidt orthonormalization.
        #
        # Each coefficient vector represents a polynomial in the basis
        #
        # [rho^m, rho^(m+2), ..., rho^n].
        # ---------------------------------------------------------------
        orthonormal_coefficients = []

        for k in range(q + 1):
            coefficient = np.zeros(q + 1, dtype=float)
            coefficient[k] = 1.0

            # Reorthogonalize twice for improved numerical stability.
            for _ in range(2):
                for previous in orthonormal_coefficients:
                    projection = previous @ gram_matrix @ coefficient
                    coefficient -= projection * previous

            norm_squared = coefficient @ gram_matrix @ coefficient

            if norm_squared <= 0.0 or not np.isfinite(norm_squared):
                raise RuntimeError(
                    "Numerical orthogonalization failed. "
                    "This may occur for very high polynomial orders."
                )

            coefficient /= np.sqrt(norm_squared)

            # Choose a positive coefficient for the highest radial power.
            # This makes the sign convention agree with ordinary Zernikes.
            if coefficient[k] < 0:
                coefficient *= -1.0

            orthonormal_coefficients.append(coefficient)

        radial_coefficients = orthonormal_coefficients[-1]

        # ---------------------------------------------------------------
        # Evaluate the radial polynomial.
        # ---------------------------------------------------------------
        radial = np.zeros_like(self.rho, dtype=float)

        for coefficient, power in zip(radial_coefficients, powers):
            radial += coefficient * self.rho ** power

        # ---------------------------------------------------------------
        # Add the angular dependence.
        #
        # The sqrt(2) factor for l != 0 produces unit RMS over the complete
        # annular pupil because the angular mean of cos^2 or sin^2 is 1/2.
        # ---------------------------------------------------------------
        if l > 0:
            angular = np.sqrt(2.0) * np.cos(m * self.theta)
        elif l < 0:
            angular = np.sqrt(2.0) * np.sin(m * self.theta)
        else:
            angular = 1.0

        Z = radial * angular

        # ---------------------------------------------------------------
        # Keep only the clear annular pupil.
        # ---------------------------------------------------------------
        result = np.full((self.N, self.N), outside_value, dtype=float)
        result[self.pupil] = Z[self.pupil]

        return result

    def _fourier_mask_aberration(self, UF):
        # Apply Fourier-plane mask
        UF_filtered = UF * self.H

        return UF_filtered

    def _fourier_mask_basis(self, UF, n, l):
        # ============================================================
        # Fourier-plane mask: pupil + aberration phase
        # ============================================================
        W = self._annular_zernike(n=n, l=l)

        # Avoid undefined phase outside pupil by multiplying with pupil
        phase = 2 * np.pi * self.basis_factor * W

        H = self.pupil * np.exp(1j * phase)

        # Apply Fourier-plane mask
        UF_filtered = UF * H

        return UF_filtered

    def forward(self, U_in, n = 0, l = 0):
        UF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_in)))

        if self.mode == "basis":
            UF = self._fourier_mask_basis(UF, n, l)

        if self.mode == "aberration":
            UF = self._fourier_mask_aberration(UF)

        if self.mode == "reference":
            UF = self._fourier_mask_aberration(UF)

        # Fourier transform from Fourier plane to output plane
        U_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(UF)))

        return U_out