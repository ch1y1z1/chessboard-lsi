"""Ablate phase-branch and mask-boundary effects on the identical CCD data."""
from pathlib import Path
import sys,json,gc,time,os
import numpy as np
from scipy import fft,ndimage
from skimage.restoration import unwrap_phase
from threadpoolctl import threadpool_limits
BASE=Path('/data/yk/shear-reference-lm/talbot-rerun-20261109')
sys.path.insert(0,str(BASE/'code'))
from lsi.forward import DiffractionModel
from lsi.fourier import FourierExtractor,spectral_window
from lsi.execution import exclusive_study
from talbot_common import config,truth_for
from talbot_ft import settings_for
from talbot_rerun import metrics
ROOT=Path(__file__).resolve().parents[1]

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def save(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

class Features:
    def __init__(self,cfg,zero):
        self.extractor=FourierExtractor(cfg,settings_for(cfg)[0]); e=self.extractor
        self.inverse=np.linalg.pinv(e.A)
        self.windows=[];self.reference=[]
        z=zero-zero.min();z=z/z.max();sz=fft.fftshift(fft.fft2(z))
        for axis in (1,0):
            carrier=e.settings.carrier_x
            center=(e.n//2+(carrier if axis==1 else 0),e.n//2+(carrier if axis==0 else 0))
            w=spectral_window(e.n,center,e.settings)
            self.windows.append(w)
            self.reference.append(fft.ifft2(fft.ifftshift(np.roll(sz*w,-int(carrier-1),axis))))
        self.denominators=[ndimage.gaussian_filter(m.astype(float),10) for m in e.masks]

    def __call__(self,image,detail=False):
        e=self.extractor;n=e.n
        im=image-image.min(); spec=fft.fftshift(fft.fft2(im/im.max()))
        values={k:[] for k in ('legacy','anchored','principal','masked','masked_nosmooth')}
        diagnostics=[];maps={}
        for k,axis in enumerate((1,0)):
            sig=fft.ifft2(fft.ifftshift(np.roll(spec*self.windows[k],-int(e.settings.carrier_x-1),axis)))
            mask=e.masks[k]
            wrapped=np.where(mask,np.angle(sig*np.conj(self.reference[k])),0.)
            unwrapped=unwrap_phase(wrapped,rng=0)
            # A global 2pi branch is arbitrary; select the branch nearest the
            # principal differential phase over the same independent mask.
            turns=int(np.rint(np.median((unwrapped[mask]-wrapped[mask])/(2*np.pi))))
            anchored=np.where(mask,unwrapped-turns*2*np.pi,0.)
            masked=unwrap_phase(np.ma.array(wrapped,mask=~mask),rng=0)
            masked=np.asarray(masked.filled(0))
            mt=int(np.rint(np.median((masked[mask]-wrapped[mask])/(2*np.pi))))
            masked=np.where(mask,masked-mt*2*np.pi,0.)
            candidates=dict(legacy=ndimage.gaussian_filter(unwrapped,10),
                            anchored=ndimage.gaussian_filter(anchored,10),
                            principal=ndimage.gaussian_filter(wrapped,10),
                            masked=ndimage.gaussian_filter(masked,10)/np.maximum(self.denominators[k],1e-100),
                            masked_nosmooth=masked)
            for key,array in candidates.items():values[key].append(array[mask])
            diagnostics.append(dict(axis=axis,turns=turns,masked_turns=mt,
                unwrapped_difference_rms=rms(unwrapped[mask]-wrapped[mask]),
                branch_pixels_fraction=float(np.mean(abs(unwrapped[mask]-wrapped[mask])>np.pi)),
                principal_abs_max=float(abs(wrapped[mask]).max()),
                ref_amp_percentiles=np.percentile(abs(self.reference[k])[mask],[0,1,5,50,99,100]).tolist()))
            if detail and k==0:
                maps=dict(wrapped=wrapped[::2,::2],legacy=candidates['legacy'][::2,::2],
                          masked=candidates['masked'][::2,::2],mask=mask[::2,::2],
                          amplitude=abs(sig)[::2,::2])
        return {key:self.inverse@np.concatenate(v) for key,v in values.items()},diagnostics,maps

def main():
    for kind in ('single','multi64'):
        folder=ROOT/'results'/f'phase_{kind}';folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():continue
        cfg=config(kind);model=DiffractionModel(cfg,tuple(range(1,36)),batch_sources=1)
        z=np.load(BASE/'results'/kind/'zero_raw.npy')
        feature=Features(cfg,z);truth=truth_for(model)
        clean=np.load(BASE/'results'/kind/'clean.npy')
        noisy=np.load(BASE/'results'/kind/'observed_40db.npy')
        checkpoint=folder/'features.json'
        data=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        jobs=[('zero',np.zeros(35)),('truth_clean',None),('truth_40db',None)]
        for j in range(35):
            for sign in (1,-1):
                c=np.zeros(35);c[j]=sign*.02
                jobs.append((f'j{j+1:02d}{"p" if sign>0 else "m"}',c))
        if os.environ.get("PROBE_ONLY"):
            jobs=[job for job in jobs if job[0] in ("zero","truth_clean","truth_40db","j12p","j12m","j07p","j07m")]
        for key,coef in jobs:
            if key in data:continue
            start=time.monotonic()
            image=z if key=='zero' else clean if key=='truth_clean' else noisy if key=='truth_40db' else model.evaluate(coef)
            f,d,m=feature(image,detail=key in ('j12p','j12m','truth_clean'))
            data[key]=dict(features={k:v.tolist() for k,v in f.items()},diagnostics=d)
            if m:np.savez_compressed(folder/f'{key}_maps.npz',**m)
            save(checkpoint,data)
            print(kind,key,round(time.monotonic()-start,2),[(x['turns'],x['branch_pixels_fraction']) for x in d],flush=True)
        if os.environ.get("PROBE_ONLY"):
            del model,feature,z,clean,noisy;gc.collect()
            continue
        summary={}
        for variant in data['zero']['features']:
            baseline=np.array(data['zero']['features'][variant])
            R=np.stack([(np.array(data[f'j{j:02d}p']['features'][variant])-np.array(data[f'j{j:02d}m']['features'][variant]))/.04 for j in range(1,36)],axis=1)
            residual=[];even=[];odd=[]
            for j in range(1,36):
                p=np.array(data[f'j{j:02d}p']['features'][variant])-baseline
                m=np.array(data[f'j{j:02d}m']['features'][variant])-baseline
                even.append(np.linalg.norm((p+m)/2));odd.append(np.linalg.norm((p-m)/2))
                for sign,f in [(1,p),(-1,m)]:
                    target=np.zeros(35);target[j-1]=sign*.02
                    residual.append(np.max(abs(np.linalg.lstsq(R,f,rcond=None)[0]-target)))
            result=dict(condition=float(np.linalg.cond(R)),rank=int(np.linalg.matrix_rank(R)),
                        even_odd_ratio=float(np.linalg.norm(even)/np.linalg.norm(odd)),train_max=float(max(residual)))
            for cond in ('clean','40db'):
                c=np.linalg.lstsq(R,np.array(data['truth_'+cond]['features'][variant])-baseline,rcond=None)[0]
                result[cond]=metrics(model,c,truth)
            summary[variant]=result
        save(folder/'summary.json',summary)
        print('SUMMARY',kind,json.dumps({k:{'train':v['train_max'],'clean':v['clean']['raw_rms_waves'],'40db':v['40db']['raw_rms_waves']} for k,v in summary.items()}),flush=True)
        del model,feature,z,clean,noisy;gc.collect()

if __name__=='__main__':
    with exclusive_study(),threadpool_limits(limits=4):main()
