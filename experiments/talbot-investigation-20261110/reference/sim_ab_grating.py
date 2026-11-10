import numpy as np
import matplotlib.pyplot as plt
from config import get_config_from_json
from skimage.restoration import unwrap_phase
from skimage.transform import resize

def grid_grating(config):
    N = config.N_lattice * config.N_super_lattice # number of pixels
    dx = config.dx_grating * config.up_sample_grating

    xout = (np.arange(N) - N // 2) * dx
    yout = xout.copy()

    return xout, yout

if __name__ == '__main__':
    filename = "ab_test.npy"
    save_file = False

    config_file = open("./config_file.json")
    config = get_config_from_json(config_file)

    from point2plate import Stage1
    stage1 = Stage1(config)

    from obj_sys_ft import Stage2_ab
    # stage2 = Stage2_ab(config, mode="reference")
    stage2 = Stage2_ab(config, mode="aberration")

    from grating2cam import Stage3
    stage3 = Stage3(config)

    N = config.N_lattice * config.N_super_lattice
    U_obj = np.zeros((N, N))
    # U_obj[180, 180] = 1
    U_obj[N // 2, N // 2] = 1

    U_img = stage2.forward(U_obj)
    U_out = stage3.forward(U_img)
    phase_out = np.angle(U_out)
    unwrapped_phase = unwrap_phase(phase_out)
    W = resize(unwrapped_phase, (N, N), anti_aliasing=True)

    if save_file:
        with open(filename, "wb") as f:
            np.save(f, W)

    I = np.abs(U_out) ** 2
    I = I / I.max()

    xout, yout = grid_grating(config)
    extent_out_um = [
        xout[0] * 1e6,
        xout[-1] * 1e6,
        yout[0] * 1e6,
        yout[-1] * 1e6,
    ]

    x_max = 1
    x_min = -x_max
    y_max = x_max
    y_min = x_min

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(I, extent=extent_out_um, cmap="hot")
    plt.xlabel("X [um]")
    plt.ylabel("Y [um]")
    plt.title("Intensity")
    plt.colorbar(label="Normalized intensity")
    # plt.axis([x_min, x_max, y_min, y_max])
    # plt.show()

    plt.subplot(1, 2, 2)
    plt.imshow(W, extent=extent_out_um, cmap="jet")
    plt.xlabel("X [um]")
    plt.ylabel("Y [um]")
    plt.title("Wavefront")
    plt.colorbar(label="Phase [rad]")
    # plt.axis([x_min, x_max, y_min, y_max])
    plt.show()