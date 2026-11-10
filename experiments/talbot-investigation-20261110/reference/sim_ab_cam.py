import numpy as np
import matplotlib.pyplot as plt
from config import get_config_from_json
# from skimage.restoration import unwrap_phase
# from skimage.transform import resize
from datetime import datetime

from point2plate import Stage1
from obj_sys_ft import Stage2_ab
from grating2cam import Stage3
from source_grid import source_indices

def grid_grating(config):
    N = config.N_lattice * config.N_super_lattice # number of pixels
    dx = config.dx_grating * 2

    xout = (np.arange(N) - N // 2) * dx
    yout = xout.copy()

    return xout, yout

def grid_camera(config):
    N = config.N_lattice * config.N_super_lattice # number of pixels
    dx = config.dx_cam

    xout = (np.arange(N) - N // 2) * dx
    yout = xout.copy()

    return xout, yout

if __name__ == '__main__':
    filename = "gram_talbot_p067_64src.npy"
    save_file = True

    config_file = open("./config_file.json")
    config = get_config_from_json(config_file)

    stage1 = Stage1(config)
    stage2 = Stage2_ab(config, mode="aberration")
    stage3 = Stage3(config, mode = "camera")

    N = config.N_lattice * config.N_super_lattice
    # Padding changes the numerical window, not the physical 8x8 source set.
    indices = source_indices(config)

    I = np.zeros((N, N), dtype=float)
    for ii in indices:
        print("Row ", ii, ": ", datetime.now())
        for jj in indices:
            U_obj = stage1.forward(nx=ii, ny=jj)
            U_img = stage2.forward(U_obj)
            U_out = stage3.forward(U_img)
            I += np.abs(U_out) ** 2

    I = I / I.max()

    if save_file:
        with open(filename, "wb") as f:
            np.save(f, I)

    xout, yout = grid_camera(config)
    extent_out_um = [
        xout[0] * 1e3,
        xout[-1] * 1e3,
        yout[0] * 1e3,
        yout[-1] * 1e3,
    ]

    plt.figure(figsize=(6, 5))
    plt.imshow(I, extent=extent_out_um, cmap="hot")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Camera plane")
    plt.colorbar(label="Normalized intensity")
    # plt.axis([x_min, x_max, y_min, y_max])
    plt.tight_layout()
    plt.show()
