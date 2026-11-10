"""Independent 35-mode calibration of phase/sideband ablations.

Principal phase is a diagnostic for the small-aberration regime, not a general
replacement for phase unwrapping. No thresholds selected using the truth.
"""
import sys,json,time,gc
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy import fft,ndimage
from threadpoolctl import threadpool_limits
BASE=Path('/data/yk/shear-reference-lm/talbot-rerun-20261109')
sys.path.insert(0,str(BASE/'code'))
from lsi.forward import DiffractionModel
from lsi.fourier import FourierExtractor,spectral_window
from lsi.basis import basis
from lsi.execution import exclusive_study
from talbot_common import config,truth_for
from talbot_ft import settings_for
from talbot_rerun import metrics
ROOT=Path(__file__).resolve().parents[1]

def save(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

class SmallPhaseFeatures:
    def __init__(self,cfg,zero):
        self.cfg=cfg
        settings,geom=settings_for(cfg)
        self.templates=[]
        # q=2 retains the original windows, masks and differential basis.
        for q in (2,1):
            s=settings if q==2 else replace(settings,carrier_x=round(geom['shadow_fundamental_bins']),carrier_y=round(geom['shadow_fundamental_bins']))
            e=FourierExtractor(cfg,s)
            if q==1:
                x=(np.arange(cfg.n)-cfg.n//2)*cfg.dx_cam
                xx,yy=np.meshgrid(x,x)
                radius=(cfg.z_grating+cfg.z_cam)*cfg.NA_out*cfg.f1/cfg.f2
                rho=np.hypot(xx,yy)/radius;theta=np.arctan2(yy,xx)
                # Adjacent orders differ by one beam displacement, with the
                # measurement sampled at the undisplaced camera coordinates.
                e.masks=[e.pupil & np.roll(e.pupil,e.shear,axis) for axis in (1,0)]
                rows=[[],[]]
                for j in range(1,36):
                    z=basis((j,),rho,theta,cfg.epsilon,cfg.Nz)[0]
                    for k,axis in enumerate((1,0)):
                        rows[k].append((np.roll(z,e.shear,axis)-z)[e.masks[k]])
                e.A=np.vstack([np.stack(row,axis=1) for row in rows])
            inverse=np.linalg.pinv(e.A)
            image=zero-zero.min();refspec=fft.fftshift(fft.fft2(image/image.max()))
            windows=[];refs=[]
            for k,axis in enumerate((1,0)):
                carrier=s.carrier_x
                center=(cfg.n//2+(carrier if axis==1 else 0),cfg.n//2+(carrier if axis==0 else 0))
                win=spectral_window(cfg.n,center,s);windows.append(win)
                refs.append(fft.ifft2(fft.ifftshift(np.roll(refspec*win,-int(carrier-1),axis))))
            self.templates.append((q,s,e.masks,inverse,windows,refs))
            del e;gc.collect()

    def __call__(self,image):
        image=image-image.min();spec=fft.fftshift(fft.fft2(image/image.max()))
        result={};diags={}
        for q,s,masks,inverse,windows,refs in self.templates:
            values=[];fractions=[]
            for k,axis in enumerate((1,0)):
                sig=fft.ifft2(fft.ifftshift(np.roll(spec*windows[k],-int(s.carrier_x-1),axis)))
                phase=np.where(masks[k],np.angle(sig*np.conj(refs[k])),0.)
                fractions.append(float(np.mean(abs(phase[masks[k]])>2.8)))
                values.append(ndimage.gaussian_filter(phase,10)[masks[k]])
            result[f'q{q}_principal']=inverse@np.concatenate(values)
            diags[f'q{q}_principal']=fractions
        return result,diags

def main():
    for kind in ('single','multi64'):
        folder=ROOT/'results'/f'ablation_{kind}';folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():continue
        cfg=config(kind);model=DiffractionModel(cfg,tuple(range(1,36)),batch_sources=1)
        zero=np.load(BASE/'results'/kind/'zero_raw.npy')
        clean=np.load(BASE/'results'/kind/'clean.npy');noisy=np.load(BASE/'results'/kind/'observed_40db.npy')
        feature=SmallPhaseFeatures(cfg,zero);truth=truth_for(model)
        path=folder/'features.json';data=json.loads(path.read_text()) if path.exists() else {}
        jobs=[('zero',None),('truth_clean',None),('truth_40db',None)]
        for j in range(35):
            for sign in (1,-1):
                c=np.zeros(35);c[j]=sign*.02
                jobs.append((f'j{j+1:02d}{"p" if sign>0 else "m"}',c))
        for key,c in jobs:
            if key in data:continue
            start=time.monotonic()
            image=zero if key=='zero' else clean if key=='truth_clean' else noisy if key=='truth_40db' else model.evaluate(c)
            f,d=feature(image)
            data[key]=dict(features={k:v.tolist() for k,v in f.items()},near_branch_fraction=d)
            save(path,data)
            print(kind,key,round(time.monotonic()-start,2),flush=True)
        summary={}
        for variant in data['zero']['features']:
            b=np.array(data['zero']['features'][variant])
            R=np.stack([(np.array(data[f'j{j:02d}p']['features'][variant])-np.array(data[f'j{j:02d}m']['features'][variant]))/.04 for j in range(1,36)],axis=1)
            errors=[];even=[];odd=[]
            for j in range(1,36):
                p=np.array(data[f'j{j:02d}p']['features'][variant])-b;m=np.array(data[f'j{j:02d}m']['features'][variant])-b
                even.append(np.linalg.norm(p+m)/2);odd.append(np.linalg.norm(p-m)/2)
                for sign,f in [(1,p),(-1,m)]:
                    target=np.zeros(35);target[j-1]=sign*.02
                    errors.append(float(max(abs(np.linalg.lstsq(R,f,rcond=None)[0]-target))))
            result=dict(rank=int(np.linalg.matrix_rank(R)),condition=float(np.linalg.cond(R)),
                        train_max=max(errors),even_odd_ratio=float(np.linalg.norm(even)/np.linalg.norm(odd)))
            np.savez_compressed(folder/f'{variant}_calibration.npz',baseline=b,response=R)
            for cond in ('clean','40db'):
                c=np.linalg.lstsq(R,np.array(data['truth_'+cond]['features'][variant])-b,rcond=None)[0]
                result[cond]=metrics(model,c,truth)
            summary[variant]=result
        save(folder/'summary.json',summary)
        print('SUMMARY',kind,json.dumps({k:{'train':v['train_max'],'clean':v['clean']['raw_rms_waves'],'40db':v['40db']['raw_rms_waves']} for k,v in summary.items()}),flush=True)
        del model,feature,zero,clean,noisy;gc.collect()

if __name__=='__main__':
    with exclusive_study(),threadpool_limits(limits=4):main()
