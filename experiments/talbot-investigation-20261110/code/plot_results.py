from pathlib import Path
import json,sys,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence'/'new-code'))
from lsi.basis import basis
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
out=ROOT/'figures'
summaries={k:json.loads((ROOT/'results'/f'ablation_{k}'/'summary.json').read_text()) for k in ('single','multi64')}
old={'single':(.011403571,.011408989),'multi64':(.017186011,.020614244)}
rows=[]
for kind in summaries:
    for cond,idx in [('clean',0),('40db',1)]:
        rows.append(dict(source=kind,pipeline='archived_q2_OLS_unwrap',condition=cond,rms_waves=old[kind][idx]))
        for tag,result in summaries[kind].items():
            rows.append(dict(source=kind,pipeline=tag,condition=cond,rms_waves=result[cond]['raw_rms_waves']))
with (ROOT/'results'/'ablation_metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,kind in zip(axs,summaries):
    for label,values,style,marker in [
        ('Archived q2 OLS + unwrap',old[kind],'-','o'),
        ('q2 OLS + principal phase',[summaries[kind]['q2_principal'][c]['raw_rms_waves'] for c in ('clean','40db')],'--','s'),
        ('q1 + adjacent basis + principal phase',[summaries[kind]['q1_principal'][c]['raw_rms_waves'] for c in ('clean','40db')],':','^')]:
        ax.plot([0,1],values,ls=style,marker=marker,label=label)
    ax.set(title=f'{kind}: identical new-Talbot observations',xticks=[0,1],xticklabels=['clean','40 dB, seed 42'],ylabel='Full-pupil raw wavefront RMS error (waves)',yscale='log',ylim=(1e-4,.05))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'ft_ablation.png',dpi=180);fig.savefig(out/'ft_ablation.pdf');plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for row,kind in enumerate(summaries):
    cfg=json.loads((ROOT/'results'/f'full_matched_{kind}'/'metadata.json').read_text())['config']
    n=1680;dx=cfg['super_lattice_period']/cfg['N_lattice']
    v=np.fft.fftshift(np.fft.fftfreq(n,dx));limit=cfg['NA_out']/cfg['wavelength']
    keep=np.abs(v)<limit*1.03;x=v[keep]/limit
    xx,yy=np.meshgrid(x,x);rho=np.hypot(xx,yy);theta=np.arctan2(yy,xx)
    pupil=(rho>=cfg['NA_in']/cfg['NA_out'])&(rho<=1)
    Z=basis(tuple(range(1,36)),rho,theta,cfg['NA_in']/cfg['NA_out'],5)
    result=summaries[kind]['q1_principal']['40db'];truth=np.array(result['truth_coefficients_waves_36'][1:]);est=np.array(result['coefficients_waves_36'][1:])
    wt=np.tensordot(truth,Z,axes=1);we=np.tensordot(est-truth,Z,axes=1)
    for col,array,title,vlim in [(0,wt,'True wavefront',.2),(1,wt+we,'q1 FT reconstructed',.2),(2,we,'Error (own scale)',.003)]:
        im=axs[row,col].imshow(np.where(pupil,array,np.nan),origin='lower',extent=(x[0],x[-1],x[0],x[-1]),cmap='RdBu_r',vmin=-vlim,vmax=vlim,interpolation='nearest')
        axs[row,col].set(title=f'{kind}\n{title}',xlabel='Normalized pupil x',ylabel='Normalized pupil y')
        fig.colorbar(im,ax=axs[row,col],label='waves',shrink=.8)
fig.savefig(out/'q1_wavefronts_40db.png',dpi=180);fig.savefig(out/'q1_wavefronts_40db.pdf');plt.close(fig)

validation=ROOT/'results'/'validation.json'
if validation.exists():
    d=json.loads(validation.read_text())
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,kind in zip(axs,d):
        for tag,style,marker in [('q2_principal','--','s'),('q1_principal','-','o')]:
            ax.plot([r['truth_rms'] for r in d[kind]['holdouts']],[r['results'][tag]['raw_rms_waves'] for r in d[kind]['holdouts']],ls=style,marker=marker,label=tag)
        ax.set(xlabel='Independent mixed truth RMS (waves)',ylabel='Full-pupil raw RMS error (waves)',title=f'{kind}: independent mixtures, seed 1807',yscale='log',xscale='log')
        ax.set_xticks([.005,.02,.05],labels=['0.005','0.020','0.050']);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'holdout_errors.png',dpi=180);fig.savefig(out/'holdout_errors.pdf');plt.close(fig)
