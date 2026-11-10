"""Locate the actual 0/+1 and -1/+1 interference carriers independently."""
import gc
import resource

import numpy as np
from scipy import fft

from lsi.execution import exclusive_study
from lsi.forward import DiffractionModel
from talbot_common import RESULTS, Monitor, config, event, save


def main():
    resource.setrlimit(resource.RLIMIT_AS, (20*1024**3, 20*1024**3))
    with exclusive_study(), Monitor("diffraction_order_validation") as monitor:
        cfg = config("single")
        model = DiffractionModel(cfg, (7,), batch_sources=1, backend="czt")
        source = model.source_field((7,7))
        spectrum = fft.fftshift(fft.fft2(fft.ifftshift(source))) * model.pupil
        image = fft.fftshift(fft.ifft2(fft.ifftshift(spectrum)))
        grating_field = model.propagate_image(image, plane="grating")
        period = int(cfg.grating_period/cfg.dx_grating)
        size = int(cfg.grating_period*cfg.duty_cycle/100/cfg.dx_grating)
        start = (period-size)//2
        index = np.arange(model.ng)
        aperture = ((index % period >= start) & (index % period < start+size)).astype(float)
        cell = aperture[:period]
        fields = {}
        for order in (-1,0,1):
            monitor.update(stage="propagating_isolated_grating_order", order=order)
            coefficient = np.mean(cell*np.exp(-2j*np.pi*order*np.arange(period)/period))
            mask = aperture[:,None] * (
                coefficient*np.exp(2j*np.pi*order*index/period))[None,:]
            fields[order] = model.cc(grating_field*mask*model.pre_cam)*model.post_cam
            del mask
            gc.collect()
        focus_shift = (cfg.f2/cfg.f1)**2*cfg.shift_z
        actual_R = cfg.z_grating-focus_shift
        fundamental = cfg.n*cfg.dx_cam*actual_R/(cfg.grating_period*(actual_R+cfg.z_cam))
        result = dict(
            actual_R_m=actual_R, fundamental_geometric_bins=fundamental,
            opposite_orders_geometric_bins=2*fundamental,
            prior_formula_bins=cfg.n*cfg.grating_period*cfg.dx_cam/
                (cfg.wavelength*(actual_R+cfg.z_cam)),
            field_relation="U_m(x)=exp(i*2*pi*m*x/p-i*pi*lambda*D*m^2/p^2)*U_free(x-lambda*D*m/p)",
            quadratic_phase_carrier="(m-n)*N*dx_cam*R/(p*(R+D))",
            reason_prior_formula_fails="p/(lambda*T) coincides with opposite-order carrier only near R=p^2/(2*lambda), not the new full Talbot R.",
            pairs=[])
        profiles = {}
        middle = cfg.n//2
        for a,b in ((1,0),(1,-1)):
            cross = fields[a]*np.conj(fields[b])
            transform = abs(fft.fftshift(fft.fft2(cross)))
            profile = np.sum(transform[middle-5:middle+6],axis=0)
            expected = (a-b)*fundamental
            # Also retain the global peak: the test does not search only where expected.
            peak = int(np.argmax(profile))-middle
            result["pairs"].append(dict(
                orders=[a,b], expected_bins=expected, measured_global_peak_bins=peak,
                absolute_peak_error_bins=abs(abs(peak)-expected),
            ))
            profiles[f"cross_{a}_{b}"] = profile
        result["passed"] = all(row["absolute_peak_error_bins"] <= 2 for row in result["pairs"])
        save(RESULTS/"diffraction_order_validation.json",result)
        np.savez_compressed(RESULTS/"diffraction_order_validation.npz", **profiles)
        event("diffraction_order_validation", **result)
        assert result["passed"],result


if __name__ == "__main__":
    main()
