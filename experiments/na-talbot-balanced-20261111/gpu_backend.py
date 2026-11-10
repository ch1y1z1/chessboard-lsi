"""Double-precision CUDA evaluation of the unchanged sampled operator."""
import time

import numpy as np
import torch

GPU_MEMORY_LIMIT_GIB = 16


class GPUCameraModel:
    """Keep all optical grids on the CPU; move only the composed matrices."""
    def __init__(self, cpu):
        assert torch.cuda.is_available()
        self.cpu = cpu
        self.device = torch.device("cuda:0")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.By = torch.as_tensor(cpu.By, dtype=torch.complex128, device=self.device)
        self.Bx = torch.as_tensor(cpu.Bx, dtype=torch.complex128, device=self.device)
        box = np.ix_(cpu.fy_indices, cpu.fx_indices)
        self.Z = torch.as_tensor(np.stack([z[box] for z in cpu.Z]),
                                 dtype=torch.float64, device=self.device)
        sources = np.zeros((len(cpu.config.sources), *cpu.pupil_box.shape), complex)
        sources[:, cpu.pupil_box] = cpu.source_spectra
        self.sources = torch.as_tensor(sources, dtype=torch.complex128, device=self.device)
        self.prefactor = complex(cpu.separable_prefactor)
        self.row_key = None
        self.n_forward = self.n_jacobian = 0
        self.seconds = 0.

    def __getattr__(self, name):
        return getattr(self.cpu, name)

    def _rows(self, rows):
        key = rows.tobytes()
        if key != self.row_key:
            yy, xx = np.divmod(rows, self.cpu.n)
            unique, inverse = np.unique(yy, return_inverse=True)
            self.row_By = self.By[torch.as_tensor(unique, device=self.device)]
            self.row_Bx = self.Bx[torch.as_tensor(xx, device=self.device)]
            self.row_inverse = torch.as_tensor(inverse, device=self.device)
            self.row_key = key

    def _transfer(self, spectrum, rows):
        if rows is None:
            value = self.By @ spectrum @ self.Bx.T
        else:
            temporary = self.row_By @ spectrum
            value = torch.sum(temporary[..., self.row_inverse, :]*self.row_Bx, dim=-1)
        return value.mul_(self.prefactor)

    @torch.inference_mode()
    def evaluate(self, coeffs, jacobian=False, rows=None):
        coeffs = np.asarray(coeffs, float)
        assert coeffs.shape == (len(self.cpu.indices),) and np.isfinite(coeffs).all()
        if rows is not None:
            rows = np.asarray(rows, int)
            assert rows.ndim == 1 and np.all((rows >= 0) & (rows < self.cpu.n**2))
            self._rows(rows)
        assert not jacobian or rows is not None, "Full-image Jacobians are intentionally disabled"
        started = time.perf_counter()
        phase = torch.tensordot(torch.as_tensor(coeffs, dtype=torch.float64,
                                              device=self.device), self.Z, dims=1)
        transfer = torch.exp(2j*np.pi*phase)
        shape = (self.cpu.n, self.cpu.n) if rows is None else (len(rows),)
        intensity = torch.zeros(shape, dtype=torch.float64, device=self.device)
        derivative = torch.zeros((len(coeffs), *shape), dtype=torch.float64,
                                 device=self.device) if jacobian else None
        for source in self.sources:
            spectrum = source*transfer
            field = self._transfer(spectrum, rows)
            intensity.add_(field.real.square()+field.imag.square())
            if jacobian:
                for j in range(0, len(coeffs), 8):
                    shifted = spectrum*(2j*np.pi*self.Z[j:j+8])
                    delta = self._transfer(shifted, rows)
                    derivative[j:j+8].add_(2*torch.real(torch.conj(field)*delta))
            del field
        image = intensity.cpu().numpy()
        result = (image, derivative.cpu().numpy()) if jacobian else image
        self.n_forward += 1
        self.n_jacobian += int(jacobian)
        self.seconds += time.perf_counter()-started
        return result
