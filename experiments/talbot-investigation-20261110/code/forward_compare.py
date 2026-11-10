"""Full sampled reference propagation; no changes to archived runs."""
import sys
from pathlib import Path
import json
import gc
import time
from dataclasses import dataclass, replace
import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits

BASE = Path('/data/yk/shear-reference-lm/talbot-rerun-20261109')
sys.path.insert(0, str(BASE/'code'))
from lsi.config import SystemConfig
from lsi.forward import DiffractionModel
from lsi.execution import exclusive_study
from talbot_common import config, truth_for
from talbot_ft import settings_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'

@dataclass(frozen=True)
class ExplicitConfig(SystemConfig):
    source_kind: str = 'single'

    @property
    def sources(self):
        nsuper = self.N_super_lattice
        if self.source_kind == 'single':
            return ((nsuper//2, nsuper//2),)
        first = (nsuper-8)//2
        return tuple((i,j) for i in range(first, first+8) for j in range(first, first+8))

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

def cases(kind):
    raw = config(kind).to_dict()
    raw['source_kind'] = kind
    full = ExplicitConfig(**raw)
    effective_quarter = 2*full.grating_period**2/full.wavelength/4
    radius_quarter = effective_quarter*full.z_cam/(full.z_cam-effective_quarter)
    focus = (full.f2/full.f1)**2*full.shift_z
    return {
        'legacy1200':replace(full,N_super_lattice=10,grating_period=1.34e-6,error_mode='OFF',shift_z=0,z_grating=134e-6),
        'legacy_padded':replace(full,grating_period=1.34e-6,error_mode='OFF',shift_z=0,z_grating=134e-6),
        'quarter_matched':replace(full,grating_period=1.34e-6),
        'full_matched':full,
        'quarter_p067':replace(full,z_grating=radius_quarter+focus),
    }

def diagnostics(image, cfg):
    settings, geom = settings_for(cfg)
    n = cfg.n; mid=n//2
    spec = np.abs(fft.fftshift(fft.fft2(image)))
    result = dict(peak=float(image.max()), total=float(image.sum()), image_rms=rms(image),
                  zero_fraction=float(np.mean(image<image.max()*1e-4)))
    profiles = {}
    for name, prof in [('x',spec[mid]),('y',spec[:,mid])]:
        profiles[name]=prof[mid:mid+min(mid,220)]/spec[mid,mid]
    bands={}
    for harmonic in (1,2,3,4):
        c=round(harmonic*geom['shadow_fundamental_bins']); h=3
        v=spec[mid-h:mid+h+1, mid+c-h:mid+c+h+1]
        bands[str(harmonic)]=dict(center=c,peak_over_dc=float(v.max()/spec[mid,mid]),
                                 root_power_over_dc=float(np.linalg.norm(v)/spec[mid,mid]))
    result.update(bands=bands, geometry=geom)
    return result, profiles, np.log10(np.maximum(spec[mid-200:mid+201,mid-200:mid+201]/spec[mid,mid],1e-9))

def main():
    OUT.mkdir(exist_ok=True)
    for kind in ('single','multi64'):
        for name,cfg in cases(kind).items():
            folder=OUT/f'{name}_{kind}'; folder.mkdir(exist_ok=True)
            if (folder/'metadata.json').exists():
                continue
            started=time.monotonic()
            print('start',name,kind,flush=True)
            model=DiffractionModel(cfg,tuple(range(1,36)),batch_sources=1)
            truth=truth_for(model)
            zero=model.evaluate(np.zeros(35)); actual=model.evaluate(truth)
            R=cfg.z_grating-(cfg.f2/cfg.f1)**2*cfg.shift_z if cfg.error_mode=='ON' else cfg.z_grating
            zeff=1/(1/R+1/cfg.z_cam); zT=2*cfg.grating_period**2/cfg.wavelength
            dz, pz, sz=diagnostics(zero,cfg)
            dt, pt, st=diagnostics(actual,cfg)
            np.save(folder/'zero.npy',zero); np.save(folder/'truth.npy',actual)
            np.savez_compressed(folder/'spectra.npz',zero_x=pz['x'],zero_y=pz['y'],truth_x=pt['x'],truth_y=pt['y'],zero_log=sz,truth_log=st)
            save(folder/'metadata.json',dict(case=name,kind=kind,config=cfg.to_dict(),sources=cfg.sources,
                 R_m=R,zeff_m=zeff,zT_m=zT,talbot_fraction=zeff/zT,
                 shear_continuous_pixels=cfg.wavelength*(R+cfg.z_cam)/cfg.grating_period/cfg.dx_cam,
                 truth_coeffs=truth.tolist(),truth_rms=rms(model.wavefront(truth)[model.pupil]),
                 zero=dz,truth=dt,seconds=time.monotonic()-started))
            print('done',name,kind,'fraction',zeff/zT,'seconds',time.monotonic()-started,flush=True)
            del model,zero,actual;gc.collect()

if __name__=='__main__':
    with exclusive_study(),threadpool_limits(limits=4): main()
