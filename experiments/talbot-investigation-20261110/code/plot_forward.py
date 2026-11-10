from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
cases=['quarter_matched','full_matched']
kinds=['single','multi64']
fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for i,kind in enumerate(kinds):
    for j,name in enumerate(cases):
        folder=ROOT/'results'/f'{name}_{kind}'
        a=np.load(folder/'truth.npy');a=a/a.mean()
        extent=np.array([-a.shape[0]/2,a.shape[0]/2,-a.shape[0]/2,a.shape[0]/2])*.01
        ax=axs[i,j*2]
        im=ax.imshow(a,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=10,interpolation='nearest')
        ax.set(title=f'{kind}: {"1/4" if j==0 else "1"} Talbot — full CCD',xlabel='x (mm)',ylabel='y (mm)')
        ax=axs[i,j*2+1]
        ax.imshow(a,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=10,interpolation='nearest')
        ax.set(title='Same image, off-axis crop',xlim=(2.5,4.5),ylim=(-1,1),xlabel='x (mm)',ylabel='y (mm)')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Intensity / whole-CCD mean; shared scale, saturated above 10',shrink=.75)
fig.savefig(OUT/'ccd_controlled.png',dpi=180);fig.savefig(OUT/'ccd_controlled.pdf');plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for row,kind in enumerate(kinds):
    for j,name in enumerate(cases):
        f=ROOT/'results'/f'{name}_{kind}';d=json.loads((f/'metadata.json').read_text());s=np.load(f/'spectra.npz')
        ax=axs[row,j]
        im=ax.imshow(s['zero_log'],origin='lower',extent=(-200,200,-200,200),cmap='magma',vmin=-6,vmax=0,interpolation='nearest')
        k=d['zero']['geometry']['shadow_fundamental_bins']
        for q,color,style in [(1,'cyan','--'),(2,'white',':')]:
            ax.axvline(q*k,color=color,ls=style,lw=.8)
        ax.set(title=f'{kind}: {"1/4" if j==0 else "1"} Talbot',xlabel='Horizontal FFT bin',ylabel='Vertical FFT bin')
        axs[row,2].semilogy(np.arange(len(s['zero_x'])),s['zero_x'],label=f'{"1/4" if j==0 else "1"} Talbot',ls='-' if j==0 else '--')
    axs[row,2].set(xlabel='Horizontal FFT bin',ylabel='FFT amplitude / DC',ylim=(1e-6,1),xlim=(0,200),title=f'{kind}: zero-wavefront axis spectrum')
    axs[row,2].legend();axs[row,2].grid(alpha=.2)
fig.colorbar(im,ax=axs[:,:2].ravel().tolist(),label='log10(FFT amplitude / DC)',shrink=.75)
fig.savefig(OUT/'spectra_controlled.png',dpi=180);fig.savefig(OUT/'spectra_controlled.pdf');plt.close(fig)

# Uniform plane-wave control isolates Talbot phase from pupils and finite windows.
n=1024;period=128
t=((np.arange(n)%period)>=period//4)&((np.arange(n)%period)<3*period//4)
freq=np.fft.fftfreq(n)*period
fractions=np.linspace(0,1,401)
carpet=np.array([abs(np.fft.ifft(np.fft.fft(t)*np.exp(-2j*np.pi*q*freq**2)))**2 for q in fractions])
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
axs[0].imshow(carpet[:,:2*period],origin='lower',extent=(0,2,0,1),aspect='auto',cmap='gray',vmin=0,vmax=1,interpolation='nearest')
axs[0].axhline(.25,color='#E69F00',ls='--');axs[0].set(xlabel='x / period',ylabel='z / full Talbot length',title='Plane-wave binary 50% duty control (1D)')
for q,ls in [(0,'-'),(.25,'--'),(1,':')]:
    idx=np.argmin(abs(fractions-q));axs[1].plot(np.arange(2*period)/period,carpet[idx,:2*period],ls=ls,label=f'z/zT={q:g}')
axs[1].set(xlabel='x / period',ylabel='Intensity (incident = 1)',title='Full Talbot returns mask; 1/4 has uniform intensity')
axs[1].legend();axs[1].grid(alpha=.2)
fig.savefig(OUT/'plane_wave_control.png',dpi=180);fig.savefig(OUT/'plane_wave_control.pdf');plt.close(fig)
np.savez_compressed(ROOT/'results'/'plane_wave_control.npz',fractions=fractions,carpet=carpet)
