import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from skimage.transform import resize
from datetime import datetime
from obj_sys_ft import Stage2_ab
from grating2cam import Stage3
from config import get_config_from_json

def grid_grating(config):
    N = config.N_lattice * config.N_super_lattice * config.up_sample_grating # number of pixels
    dx = config.dx_grating * 2

    xout = (np.arange(N) - N // 2) * dx
    yout = xout.copy()

    return xout, yout

if __name__ == '__main__':
    filename_wf = "basis_W.npy"
    save_wf = True
    filename_I = "basis_I.npy"
    save_I = True

    config_file = open('./config_file.json')
    config = get_config_from_json(config_file)

    N = config.N_lattice * config.N_super_lattice
    U_obj = np.zeros((N, N))
    U_obj[N//2, N//2] = 1

    stage2 = Stage2_ab(config, mode = "basis")
    stage3 = Stage3(config, mode = "basis")

    Nz = config.Nz
    N_poly = Nz**2 + 2*Nz + 1
    W = np.zeros((N_poly, N, N), dtype=float)
    I = np.zeros((N_poly, N, N), dtype=float)

    idx = 0
    for nn in range(Nz+1):
        print("Zernike polynomial radial degree ", nn, ": ", datetime.now())
        for ll in range(-nn, nn+1):
            # print("Radial degree = ", nn, "; Azimuthal degree = ", ll)
            mm = np.abs(ll)
            U_img = stage2.forward(U_obj, n = 2*nn-mm, l = ll)
            U_out = stage3.forward(U_img)

            phase_out = np.angle(U_out)
            unwrapped_phase = unwrap_phase(phase_out)
            W[idx, :, :] = resize(unwrapped_phase, (N, N), anti_aliasing=True)

            I_tmp = np.abs(U_out)**2
            I_tmp = I_tmp / I_tmp.max()
            I[idx, :, :] = resize(I_tmp, (N, N), anti_aliasing=True)

            idx += 1

    if save_wf:
        with open(filename_wf, "wb") as f:
            np.save(f, W)

    if save_I:
        with open(filename_I, "wb") as f:
            np.save(f, I)
    # ============================================================
    # Plot results
    print("Dimensions of the saved wavefront = ", W.shape)

    xout, yout = grid_grating(config)
    extent_out_um = [
        xout[0] * 1e6,
        xout[-1] * 1e6,
        yout[0] * 1e6,
        yout[-1] * 1e6,
    ]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.imshow(I[0,:,:], extent=extent_out_um, cmap="hot")
    plt.xlabel("X [um]")
    plt.ylabel("Y [um]")
    plt.title("Grating plane")
    plt.colorbar(label="Normalized intensity")

    plt.subplot(1, 2, 2)
    plt.imshow(W[0, :, :], extent=extent_out_um, cmap="jet")
    plt.xlabel("X [um]")
    plt.ylabel("Y [um]")
    plt.title("Grating plane")
    plt.colorbar(label="Phase [rad]")

    plt.tight_layout()
    plt.show()