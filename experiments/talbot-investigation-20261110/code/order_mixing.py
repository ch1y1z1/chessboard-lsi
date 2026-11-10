"""Measure contamination of an isolated +/-1 cross term by all x orders.

Keep the complete y aperture and identical finite Fresnel/CZT operators.
Compare the filtered complex envelope, allowing one complex calibration gain.
"""
from pathlib import Path
import sys,json,gc
import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits
BASE=Path('/data/yk/shear-reference-lm/talbot-rerun-20261109')
sys.path.insert(0,str(BASE/'code'))
from lsi.forward import DiffractionModel
from lsi.execution import exclusive_study
from lsi.fourier import FourierExtractor,spectral_window
from talbot_ft import settings_for
from forward_compare import cases
ROOT=Path(__file__).resolve().parents[1]

def order_matrix(model,order):
    c=model.config;z=c.z_grating;xi=model.x_img[0];xg=model.x_g[0];xc=model.x_cam[0]
    period=int(c.grating_period/c.dx_grating);size=int(period*.5);start=(period-size)//2
    cell=((np.arange(period)>=start)&(np.arange(period)<start+size)).astype(float)
    coefficient=np.mean(cell*np.exp(-2j*np.pi*order*np.arange(period)/period))
    aperture=coefficient*np.exp(2j*np.pi*order*np.arange(model.ng)/period)
    pre=np.exp(1j*np.pi*xi**2/(c.wavelength*z))
    middle=np.exp(1j*np.pi*xg**2/(c.wavelength*z)+1j*np.pi*xg**2/(c.wavelength*c.z_cam))*aperture
    frequencies=model.fx_indices
    seeds=np.zeros((len(frequencies),model.n),complex);seeds[np.arange(len(frequencies)),frequencies]=1
    image=fft.fftshift(fft.ifft(fft.ifftshift(seeds,axes=-1),axis=-1),axes=-1)
    g=model.cg.plan(image*pre,axis=-1)*model.cg.factor
    cam=model.cc.plan(g*middle,axis=-1)*model.cc.factor
    cam*=np.exp(1j*np.pi*xc**2/(c.wavelength*c.z_cam))
    return np.ascontiguousarray(cam.T)

def main():
    results=[]
    for kind in ('single','multi64'):
        for name in ('quarter_matched','full_matched'):
            cfg=cases(kind)[name];m=DiffractionModel(cfg,(7,),batch_sources=1);m.prepare_sources()
            settings,geom=settings_for(cfg);ex=FourierExtractor(cfg,settings)
            center=(m.n//2+settings.carrier_x,m.n//2)
            win=spectral_window(m.n,center,settings);mask=ex.masks[0]
            del ex;gc.collect()
            bxminus=order_matrix(m,-1);bxplus=order_matrix(m,1)
            cross=np.zeros((m.n,m.n),complex)
            for spec1d in m.source_spectra:
                spec=np.zeros(m.pupil_box.shape,complex);spec[m.pupil_box]=spec1d
                uminus=(m.By@spec@bxminus.T)*m.separable_prefactor
                uplus=(m.By@spec@bxplus.T)*m.separable_prefactor
                cross+=uplus*np.conj(uminus)
            full=np.load(ROOT/'results'/f'{name}_{kind}'/'zero.npy')
            # The complete real pair intensity contributes cross+conj(cross).
            pair=2*np.real(cross)
            sf=fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(full))*win))
            sp=fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(pair))*win))
            a=sf[mask];b=sp[mask]
            gain=np.vdot(b,a)/np.vdot(b,b)
            row=dict(case=name,kind=kind,full_to_pair_envelope_norm=float(np.linalg.norm(a)/np.linalg.norm(b)),
                     best_complex_gain=[float(gain.real),float(gain.imag)],
                     pair_model_relative_residual=float(np.linalg.norm(a-gain*b)/np.linalg.norm(a)),
                     carrier=settings.carrier_x,mask_pixels=int(mask.sum()))
            results.append(row)
            print(json.dumps(row),flush=True)
            (ROOT/'results'/'order_mixing.json').write_text(json.dumps(results,indent=2)+'\n')
            del m,full,pair,cross,sf,sp,bxminus,bxplus;gc.collect()

if __name__=='__main__':
    with exclusive_study(),threadpool_limits(limits=4):main()
