import numpy as np
# from scipy.ndimage import zoom
from scipy.signal import czt
import torch
# from scipy.ndimage import gaussian_filter

class Stage3(torch.nn.Module):
    def __init__(self, config, mode = "grating"):
        super(Stage3, self).__init__()
        # ============================================================
        # Physical parameters
        # ============================================================
        self.lam = config.wavelength  # wavelength [m]
        self.k = 2 * np.pi / self.lam

        self.N = config.N_lattice * config.N_super_lattice
        M = config.f2 / config.f1
        self.dx_img = config.super_lattice_period / config.N_lattice * M
        self.up_sample = config.up_sample_grating
        self.mode = mode

        # L_img = self.dx_img * self.N
        # print()
        # print("Stage 3: the simulation parameters for the grating diffraction")
        # print("dx_img =", "{:.4g}".format(self.dx_img * 1e9), "nm")
        # print("L_img =", "{:.4g}".format(L_img * 1e6), "um")

        # OFF mode, neglect the misalignment of hole array and of grating
        self.error_mode = config.error_mode

        # The distance from the image plane to grating plane,
        # In ZJ's assumption, it is equal to one Talbot distance
        # shift_z_grating is the distance from the object plane to the mask, unit of [m].
        self.z_grating = config.z_grating - config.shift_z_grating
        self.dx_g = config.dx_grating
        
        # Orientation of the grating
        # Axis of rotation in the xy plane.
        # Examples:
        #   (1, 0): plane A rotated about x axis
        #   (0, 1): plane A rotated about y axis
        #   (1, 1): plane A rotated about diagonal axis
        # can't be (0,0)
        self.tilt_axis_x_grating = config.tilt_axis_x_grating
        self.tilt_axis_y_grating = config.tilt_axis_y_grating

        self.tilted_angle_grating = config.tilted_angle_grating * np.pi / 180  # unit of degree

        # Parameters of the grating
        self.grating_type = config.grating_type # Square or Diamond
        self.grating_period = config.grating_period # unit of [m]

        # The duty cycle of grating. Its value should be within (0, 100)
        # Useless for the diamond grating
        self.duty_cycle = config.duty_cycle / 100

        # generate the grating
        if (self.grating_type == "Diamond"):
            self.grating = self._diamond_grating()
            # print("Use the diamond grating")
        else:
            self.grating = self._square_grating()
            # print("Use the square grating")

        # xy shift, unit of [m] in config
        # Should be shorter than one period
        self.shift_x_grating = int(
            config.shift_x_grating / self.dx_g
        )  # unit of pixel
        self.shift_y_grating = int(config.shift_y_grating / self.dx_g)
        self.grating = np.roll(self.grating, (self.shift_x_grating, self.shift_y_grating), axis=(1, 0))

        # camera coordinates
        self.z_cam = config.z_cam
        self.dx_cam = config.dx_cam
        
        # Orientation of the camera
        self.tilt_axis_x_cam = config.tilt_axis_x_cam
        self.tilt_axis_y_cam = config.tilt_axis_y_cam
        self.tilted_angle_cam = config.tilted_angle_cam * np.pi / 180  # unit of degree

        # print("Number of CCD pixels = ", self.N)
        # print("dx_camera =", "{:.4g}".format(self.dx_cam * 1e6), "um")
        # print("L_camera =", "{:.4g}".format(self.N * self.dx_cam * 1e3), "mm")

        """
        1. From the image plane to the grating
        """
        # image coordinates
        x = (np.arange(self.N) - self.N // 2) * self.dx_img
        self.X, self.Y = np.meshgrid(x, x)

        # Fresnel pre-quadratic phase on the image plane
        self.q1_img = np.exp(1j * np.pi / (self.lam * self.z_grating) * (self.X**2 + self.Y**2))

        # grating coordinates
        x_g = (np.arange(self.N*self.up_sample) - self.N*self.up_sample//2) * self.dx_g
        self.X_g, self.Y_g = np.meshgrid(x_g, x_g)

        # Fresnel post-quadratic phase on grating plane
        self.q2_g = np.exp(1j * np.pi / (self.lam * self.z_grating) * (self.X_g**2 + self.Y_g**2))

        # Fresnel propagation prefactor, from the image plane to the grating
        self.prefactor_g = np.exp(1j * self.k * self.z_grating) / (1j * self.lam * self.z_grating)

        # Required spatial frequencies on the camera plane:
        # fx = x2 / (lambda z), fy = y2 / (lambda z)
        self.f0_g = x_g[0] / (self.lam * self.z_grating)
        self.df_g = self.dx_g / (self.lam * self.z_grating)

        """
        2. From the grating to the camera
        """
        # Fresnel pre-quadratic phase on the image plane
        self.q1_g = np.exp(1j * np.pi / (self.lam * self.z_cam) * (self.X_g**2 + self.Y_g**2))
        
        # camera coordinates
        x_cam = (np.arange(self.N) - self.N // 2) * self.dx_cam
        X_cam, Y_cam = np.meshgrid(x_cam, x_cam)

        # Fresnel post-quadratic phase on destination plane
        self.q2_cam = np.exp(1j * np.pi / (self.lam * self.z_cam) * (X_cam**2 + Y_cam**2))

        # Fresnel propagation prefactor
        self.prefactor_cam = np.exp(1j * self.k * self.z_cam) / (1j * self.lam * self.z_cam)

        # Required spatial frequencies on the camera plane:
        # fx = x2 / (lambda z), fy = y2 / (lambda z)
        self.f0_cam = x_cam[0] / (self.lam * self.z_cam)
        self.df_cam = self.dx_cam / (self.lam * self.z_cam)

    def _grating2cam(self, U0):
        # Dowmsample the optical field
        # scale_ratio = 1 / self.downsampling_factor
        # U0 = zoom(U0, (scale_ratio, scale_ratio), order=1)
        # U0 = np.pad(U0, [(self.N_cam, self.N_cam), (self.N_cam, self.N_cam)], mode="constant")
        
        # Normalize the rotation axis in the xy plane
        axis = np.array(
            [self.tilt_axis_x_cam, self.tilt_axis_y_cam], dtype=float
        )
        axis = axis / np.linalg.norm(axis)
        ax, ay = axis

        # This vector is perpendicular to the rotation axis.
        px = -ay
        py = ax

        # Height of tilted image plane.
        # For small tilt:
        #     z_A(x,y) = tan(theta) * distance_perpendicular_to_axis
        z_A = np.tan(self.tilted_angle_cam) * (px * self.X_g + py * self.Y_g)

        # Local phase correction:
        # Field on tilted plane A: U_A(x,y) approximately equals U_0(x,y) * exp(i k0 z_A)
        # Therefore, U_0(x,y) approximately equals U_A(x,y) * exp(-i k0 z_A)
        U_0 = U0 * np.exp(-1j * self.k * z_A)
        
        # Shifted/scaled Fresnel diffraction
        g = U_0 * self.q1_g

        # 2D Fourier transform evaluated on arbitrary output grid.
        # First along x, then along y.
        Gx = self._czt_uniform_freq_1d(g, self.dx_g, self.f0_cam, self.df_cam, self.N, axis=1)
        Gxy = self._czt_uniform_freq_1d(Gx, self.dx_g, self.f0_cam, self.df_cam, self.N, axis=0)

        U_out = self.prefactor_cam * self.q2_cam * Gxy
        
        return U_out

    def _img2grating(self, U_A):
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
        axis = np.array([self.tilt_axis_x_grating, self.tilt_axis_y_grating], dtype=float)
        axis = axis / np.linalg.norm(axis)

        ax, ay = axis

        # This vector is perpendicular to the rotation axis.
        px = -ay
        py = ax

        # Height of tilted plane A.
        # For small tilt:
        #     z_A(x,y) = tan(theta) * distance_perpendicular_to_axis
        z_A = np.tan(self.tilted_angle_grating) * (px * self.X + py * self.Y)

        # Local phase correction:
        # Field on tilted plane A: U_A(x,y) approximately equals U_0(x,y) * exp(i k0 z_A)
        # Therefore, U_0(x,y) approximately equals U_A(x,y) * exp(-i k0 z_A)
        U_0 = U_A * np.exp(-1j * self.k * z_A)

        # Shifted/scaled Fresnel diffraction
        g = U_0 * self.q1_img

        # 2D Fourier transform evaluated on arbitrary output grid.
        # First along x, then along y.
        Gx = self._czt_uniform_freq_1d(g, self.dx_img, self.f0_g, self.df_g, self.N*self.up_sample, axis=1)
        Gxy = self._czt_uniform_freq_1d(Gx, self.dx_img, self.f0_g, self.df_g, self.N*self.up_sample, axis=0)

        U_out = self.prefactor_g * self.q2_g * Gxy

        return U_out

    def _czt_uniform_freq_1d(self, g, dx, f0, df, m, axis=-1):
        """
        Compute:
            G[k] = sum_n g[n] * exp(-i 2 pi f_k x_n) * dx

        where:
            x_n = (n - N/2) dx
            f_k = f0 + k df

        using scipy.signal.czt along one axis.
        """
        g = np.asarray(g, dtype=np.complex128)
        N = g.shape[axis]

        # scipy.signal.czt computes:
        # X[k] = sum_n x[n] * a^(-n) * w^(n k)
        a = np.exp(1j * 2 * np.pi * f0 * dx)
        w = np.exp(-1j * 2 * np.pi * df * dx)

        G = czt(g, m=m, w=w, a=a, axis=axis)

        # Correct for centered coordinates x_n = (n - N/2) dx
        k = np.arange(m)
        fk = f0 + k * df
        phase_center = np.exp(1j * 2 * np.pi * fk * (N / 2) * dx)

        shape = [1] * G.ndim
        shape[axis] = m
        G = G * phase_center.reshape(shape)

        return G * dx

    def _square_grating(self):
        """
        参数:
            N: int - 数组的大小（N x N）
            period: int - 周期的格子数（一个周期包含几个方格）
            square_size: int - 每个方格的大小（像素数）
        返回:
            numpy.ndarray - N x N 的数组，方格内为1，方格外为0
        """
        square_size = int(self.grating_period * self.duty_cycle / self.dx_g)
        period = int(self.grating_period / self.dx_g)

        # 计算每个位置在周期内的位置
        x = np.arange(self.N*self.up_sample)
        row, col = np.meshgrid(x, x)
        row_mod = row % period
        col_mod = col % period

        # 计算方格起始位置（居中对齐）
        start = (period - square_size) // 2  # 方格从周期中心开始
        end = start + square_size            # 方格占据1个格子单位

        # 创建方格图案：方格占据周期中心位置
        pattern = np.zeros((self.N*self.up_sample, self.N*self.up_sample))
        pattern[(row_mod >= start) & (row_mod < end) & (col_mod >= start) & (col_mod < end)] = 1

        return pattern.astype(int)

    def _diamond_grating(self):
        """
        生成周期性菱形图案，菱形的顶点为周期方格各边的中点
        参数:
            N: int - 数组的大小（N x N）
            period: int - 周期的格子数（一个周期包含几个方格）
            square_size: int - 每个方格的大小（像素数）
        返回:
            numpy.ndarray - N x N 的数组，菱形内为1，菱形外为0
        """
        period = int(self.grating_period / self.dx_g)

        # 生成行和列的坐标
        x = np.arange(self.N*self.up_sample)
        row, col = np.meshgrid(x, x)

        # 将坐标映射到周期内 [0, period)
        row_mod = row % period
        col_mod = col % period

        # 计算到周期内每个菱形中心的曼哈顿距离
        center = period // 2
        distance = np.abs(row_mod - center) + np.abs(col_mod - center)
        size = period // 2

        # 菱形内部：曼哈顿距离小于等于size
        pattern = distance <= size

        return pattern.astype(int)
    
    def forward(self, U_img):
        # From the image plane to the grating
        if self.mode == "grating":
            U_out = self._img2grating(U_img)

        if self.mode == "camera":
            U_grat = self._img2grating(U_img) * self.grating
            U_out = self._grating2cam(U_grat)

        # For testing
        # U_grat = U_img

        return U_out