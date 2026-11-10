"""Recompute saved wavefront metrics from coefficients on the exact pupil."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence'/'new-code'))
from lsi.basis import basis
cfg=json.loads((ROOT/'results'/'full_matched_single'/'metadata.json').read_text())['config']
n=1680;dx=cfg['super_lattice_period']/cfg['N_lattice']
freq=np.fft.fftshift(np.fft.fftfreq(n,dx));r=cfg['NA_out']/cfg['wavelength']
select=abs(freq)<r*1.03;v=freq[select]/r
x,y=np.meshgrid(freq[select],freq[select]);rho=cfg['wavelength']*np.hypot(x,y)/cfg['NA_out'];epsilon=cfg['NA_in']/cfg['NA_out']
pupil=(rho>=epsilon)&(rho<=1)
Z=basis(tuple(range(1,36)),rho,np.arctan2(y,x),epsilon,5)[:,pupil]
# Separate BLAS/LAPACK builds perturb the high-order annular Gram
# orthogonalization at ~1e-11 relative; audit below the displayed precision.
checks=[]
def check(label,yes):
    checks.append(dict(label=label,passed=bool(yes)))
    if not yes:raise AssertionError(label)
check('full evaluation pupil has 52428 samples',pupil.sum()==52428)
def metric(label,row):
    c=np.array(row['coefficients_waves_36']);t=np.array(row['truth_coefficients_waves_36'])
    check(label+' piston',c[0]==t[0]==0)
    w=(c[1:]-t[1:])@Z
    check(label+' raw RMS',np.isclose(np.sqrt(np.mean(w*w)),row['raw_rms_waves'],rtol=1e-9,atol=1e-12))
    check(label+' raw PV',np.isclose(np.ptp(w),row['raw_pv_waves'],rtol=1e-9,atol=1e-12))
    check(label+' pupil count',row['evaluation_pixels']==int(pupil.sum()))
for kind in ('single','multi64'):
    summary=json.loads((ROOT/'results'/f'ablation_{kind}'/'summary.json').read_text())
    for tag,v in summary.items():
        check(kind+tag+' rank',v['rank']==35)
        for cond in ('clean','40db'):metric(kind+tag+cond,v[cond])
validation=json.loads((ROOT/'results'/'validation.json').read_text())
for kind,v in validation.items():
    check(kind+' archived zero reproduced',v['archived_zero_relative_L2']<1e-12)
    check(kind+' archived clean reproduced',v['archived_clean_relative_L2']<1e-12)
    for j,row in enumerate(v['holdouts']):
        for tag,m in row['results'].items():metric(kind+f'holdout{j}'+tag,m)
    for row in v['noise_repeats']:
        for tag,m in row['results'].items():metric(kind+f"seed{row['seed']}"+tag,m)
for folder in sorted((ROOT/'results').glob('*_matched_*')):
    p=folder/'metadata.json'
    if not p.exists():continue
    m=json.loads(p.read_text());target=.25 if m['case']=='quarter_matched' else 1
    check(folder.name+' fraction',abs(m['talbot_fraction']-target)<1e-12)
result=dict(passed=all(c['passed'] for c in checks),count=len(checks),checks=checks)
(ROOT/'results'/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],count=result['count'])))
