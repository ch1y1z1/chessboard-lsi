"""Hold-outs, noise repeats, and provenance checks for the FT diagnosis."""
from pathlib import Path
import sys,json,gc,hashlib
import numpy as np
from skimage.restoration import unwrap_phase
from scipy import fft
from threadpoolctl import threadpool_limits
BASE=Path('/data/yk/shear-reference-lm/talbot-rerun-20261109')
sys.path.insert(0,str(BASE/'code'))
from lsi.forward import DiffractionModel
from lsi.execution import exclusive_study
from talbot_common import config,truth_for
from talbot_rerun import metrics
from calibrate_ablation import SmallPhaseFeatures
ROOT=Path(__file__).resolve().parents[1]

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def main():
    report={}
    for kind in ('single','multi64'):
        cfg=config(kind);m=DiffractionModel(cfg,tuple(range(1,36)),batch_sources=1)
        zero=np.load(BASE/'results'/kind/'zero_raw.npy');clean=np.load(BASE/'results'/kind/'clean.npy')
        truth=truth_for(m);features=SmallPhaseFeatures(cfg,zero)
        cal={}
        for tag in ('q1_principal','q2_principal'):
            a=np.load(ROOT/'results'/f'ablation_{kind}'/f'{tag}_calibration.npz');cal[tag]=(a['baseline'],a['response'])
        def fit(image):
            f,d=features(image);result={}
            for tag,raw in f.items():
                b,R=cal[tag];estimate=np.linalg.lstsq(R,raw-b,rcond=None)[0]
                result[tag]=estimate
            return result,d
        tests=[];random=np.random.default_rng(1807)
        for amp in (.005,.02,.05):
            coeff=random.normal(size=35);coeff*=amp/rms(m.wavefront(coeff)[m.pupil])
            image=m.evaluate(coeff);estimated,diag=fit(image)
            row=dict(truth_rms=amp,results={tag:metrics(m,c,coeff) for tag,c in estimated.items()},near_branch=diag)
            tests.append(row);print('holdout',kind,amp,{t:r['raw_rms_waves'] for t,r in row['results'].items()},flush=True)
        repeats=[]
        for seed in (41,42,43,44,45):
            noisy=clean+.01*rms(clean)*np.random.default_rng(seed).standard_normal(clean.shape)
            estimates,d=fit(noisy)
            repeats.append(dict(seed=seed,results={tag:metrics(m,c,truth) for tag,c in estimates.items()}))
        # Check whether a standard unwrap would change q1 for the main truth.
        image=clean-clean.min();spec=fft.fftshift(fft.fft2(image/image.max()))
        q,s,masks,inverse,windows,refs=features.templates[1]
        checks=[]
        for k,axis in enumerate((1,0)):
            sig=fft.ifft2(fft.ifftshift(np.roll(spec*windows[k],-int(s.carrier_x-1),axis)))
            phase=np.where(masks[k],np.angle(sig*np.conj(refs[k])),0.)
            unwrapped=unwrap_phase(phase,rng=0)
            checks.append(dict(axis=axis,max_abs_principal=float(abs(phase[masks[k]]).max()),
                               principal_unwrap_max_difference=float(abs(unwrapped-phase).max())))
        archived_zero=zero
        rerun_zero=np.load(ROOT/'results'/f'full_matched_{kind}'/'zero.npy')
        raw_truth=np.load(ROOT/'results'/f'full_matched_{kind}'/'truth.npy')
        report[kind]=dict(holdouts=tests,noise_repeats=repeats,q1_main_truth_unwrap_check=checks,
                         archived_zero_relative_L2=float(np.linalg.norm(rerun_zero-archived_zero)/np.linalg.norm(archived_zero)),
                         archived_clean_relative_L2=float(np.linalg.norm(raw_truth/raw_truth.max()-clean)/np.linalg.norm(clean)))
        (ROOT/'results'/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        del m,features,zero,clean;gc.collect()
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))}
    (ROOT/'results'/'code_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    with exclusive_study(),threadpool_limits(limits=4):main()
