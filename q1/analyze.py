"""Export traceable tables, figures and LaTeX numbers from actual solver output."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/carbon-mpl')
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
BASE=Path(__file__).resolve().parent
D=BASE/'data'; F=BASE/'figures'; R=BASE/'report'
F.mkdir(exist_ok=True)
records=[]
for p in D.glob('L*.json'):
    records.extend(json.loads(p.read_text()).values())
records.sort(key=lambda x:(x['L'],x['c']))
roots=[]
for p in D.glob('roots_L*.json'):
    roots.extend(json.loads(p.read_text()))
roots.sort(key=lambda x:x['L'])
byL={L:[r for r in records if r['L']==L] for L in sorted({r['L'] for r in records})}
def get(L,c):return next(r for r in byL[L] if abs(r['c']-c)<1e-9)
def csvwrite(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
csvwrite(D/'energies.csv',['L','c','E2','E31','E32','ER','B'],records)
csvwrite(D/'crossings.csv',['L','cstar','ER','xtol'],roots)
maxres=max(max(r['residual3'][i] for i in r['selected']) for r in records)
maxsym=max(max(e[i] for i in r['selected']) for r in records for e in r['symmetry'].values())
maxorth=max(r['orthogonality'] for r in records)
assert maxres<1e-7 and maxsym<1e-6 and maxorth<1e-10
assert all(r['selected']==[0,1] for r in records)
bench=get(20,5)
assert abs(bench['E31']+4.771)<.0005 and abs(bench['E32']+1.109)<.0005
# Increasing c lowers all energies on the coarse grids; roots have a resolved sign change.
for L in [8,12,16,20]:
    grid=[get(L,float(c)) for c in np.arange(3.5,7.001,.25)]
    for key in ['E2','E31','E32','ER']:
        assert np.all(np.diff([r[key] for r in grid])<0)
for root in roots:
    assert abs(root['ER'])<1e-7
    a,b=root['bracket']
    assert get(root['L'],a)['ER']*get(root['L'],b)['ER']<0
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
 'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
colors=['#7f8c8d','#b5a0c6','#c58b62','#477ca8','#2d857a','#3b4865','#222222']
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(F/f'{name}.{ext}',dpi=240)
    plt.close(fig)
def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=9)
    ax.text(-.16,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)

fig,axs=plt.subplots(2,2,figsize=(7.12,5.0),layout='constrained')
ax=axs[0,0]
grid=[get(20,float(c)) for c in np.arange(3.5,7.001,.25)]
for key,name,col,marker in [('E31',r'$E_{3,1}$','#477ca8','o'),('E32',r'$E_{3,2}$','#c58b62','s'),('E2',r'$E_{\rm thr}$','#444444','^')]:
    ax.plot([r['c'] for r in grid],[r[key] for r in grid],label=name,color=col,marker=marker,ms=2.8,lw=1.1)
ax.set(xlabel=r'Attraction strength $c$',ylabel=r'Energy ($t=1$)')
ax.legend();label(ax,'a',r'Low-energy spectrum ($L=20$)')
ax=axs[0,1]
show=[8,12,16,20,24,28,32]
for L,col in zip(show,colors):
    if L not in byL:continue
    rr=[r for r in byL[L] if 3.5<=r['c']<=4.5]
    ax.plot([r['c'] for r in rr],[r['ER'] for r in rr],'.-',ms=2.4,lw=1,color=col,label=f'{L}')
ax.axhline(0,color='black',ls='--',lw=.7)
ax.set(xlim=(3.5,4.5),ylim=(-.11,.065),xlabel=r'Attraction strength $c$',ylabel=r'$E_R=E_{3,2}-E_{\rm thr}$')
ax.legend(title=r'$L$',ncol=2,fontsize=6.5,title_fontsize=7,loc='lower left',columnspacing=.8)
label(ax,'b','Threshold crossing depends on box size')
ax=axs[1,0]
ax.plot([r['L'] for r in roots],[r['cstar'] for r in roots],'o-',color='#477ca8',ms=4,lw=1)
ax.set(xlabel=r'Lattice size $L$',ylabel=r'Finite-box crossing $c^\star(L)$')
label(ax,'c','Crossing approaches a large-box value')
ax=axs[1,1]
conv=[get(L,5) for L in byL if any(abs(r['c']-5)<1e-9 for r in byL[L])]
last=conv[-1]
for key,name,col,marker in [('E2',r'$E_{\rm thr}$','#444444','^'),('E32',r'$E_{3,2}$','#c58b62','s')]:
    vals=[abs(r[key]-last[key]) for r in conv[:-1]]
    ax.semilogy([r['L'] for r in conv[:-1]],vals,marker+'-',label=name,color=col,ms=3,lw=1)
ax.set(xlabel=r'Lattice size $L$',ylabel=rf'$|E(L)-E({last["L"]})|$')
ax.legend();label(ax,'d',r'Energy convergence at $c=5$')
save(fig,'spectrum_convergence')

states=np.load(D/'states_L20_c5.npz')
L=20
q=np.arange(L);q=np.where(q<L//2,q,q-L)
x,y=np.meshgrid(q,q,indexing='ij');r2=x*x+y*y
marginals={key:np.sum(states[key]**2,axis=(2,3)) for key in ['ground','excited']}
rms={k:float(np.sqrt(np.sum(p*r2))) for k,p in marginals.items()}
csvwrite(D/'pair_marginals.csv',['x','y','P_ground','P_excited'],[
 dict(x=int(x[i,j]),y=int(y[i,j]),P_ground=marginals['ground'][i,j],P_excited=marginals['excited'][i,j]) for i in range(L) for j in range(L)])
radii=np.sqrt(np.unique(r2))
cdfs={k:np.array([np.sum(p[r2<=rad*rad+1e-10]) for rad in radii]) for k,p in marginals.items()}
csvwrite(D/'pair_cdf.csv',['r','ground','excited'],[dict(r=float(rad),ground=float(cdfs['ground'][i]),excited=float(cdfs['excited'][i])) for i,rad in enumerate(radii)])
fig=plt.figure(figsize=(7.12,2.35),layout='constrained')
gs=fig.add_gridspec(1,4,width_ratios=[1,1,.06,1.15])
axs=[fig.add_subplot(gs[0,j]) for j in [0,1,3]]
for ax,key,letter,title in zip(axs[:2],['ground','excited'],['a','b'],[r'Ground state $0^+_1$',r'Hoyle-like state $0^+_2$']):
    im=ax.imshow(np.fft.fftshift(marginals[key]).T,origin='lower',extent=(-10.5,9.5,-10.5,9.5),cmap='viridis',norm=LogNorm(vmin=1e-8,vmax=1),interpolation='nearest')
    ax.set(xlabel=r'$r_{1x}/a$',ylabel=r'$r_{1y}/a$',xticks=[-10,-5,0,5],yticks=[-10,-5,0,5])
    label(ax,letter,title)
fig.colorbar(im,cax=fig.add_subplot(gs[0,2]),label=r'Pair probability $P(\mathbf{r}_1)$',ticks=[1,1e-4,1e-8])
ax=axs[2]
for key,name,col in [('ground',r'$0^+_1$','#477ca8'),('excited',r'$0^+_2$','#c58b62')]:
    ax.step(radii,cdfs[key],where='post',label=name,color=col,lw=1.2)
ax.set(xlabel=r'Pair separation $r/a$',ylabel=r'$\Pr(|\mathbf{r}_1|\leq r)$',xlim=(0,10),ylim=(0,1.03))
ax.legend(loc='lower right');label(ax,'c','Spatial extent')
save(fig,'state_structure')
# Machine-generated numerical snippets prevent report/data transcription drift.
lines=[]
for root in roots:
    r=get(root['L'],5)
    lines.append(f"{r['L']} & {r['E2']:.9f} & {r['E31']:.9f} & {r['E32']:.9f} & {root['cstar']:.7f} \\\\")
(R/'convergence_rows.tex').write_text('\n'.join(lines)+'\n')
header = r'\begin{tabular}{rrrrr}\toprule $L$ & $E_2$ ($c=5$) & $E_{3,1}$ ($c=5$) & $E_{3,2}$ ($c=5$) & $c^\star(L)$ \\ \midrule'
(R/'convergence_table.tex').write_text(header+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
def macro(name,val):return '\\newcommand{\\'+name+'}{'+val+'}\n'
text=''
for name,key in [('Etwo','E2'),('Eground','E31'),('Eexcited','E32'),('Erel','ER'),('Binding','B')]:
    text+=macro(name,f'{bench[key]:.9f}')
text+=macro('CstarMax',f"{roots[-1]['cstar']:.7f}")
text+=macro('Lmax',str(roots[-1]['L']))
text+=macro('LastRootChange',f"{abs(roots[-1]['cstar']-roots[-2]['cstar']):.6f}")
text+=macro('RmsGround',f"{rms['ground']:.3f}")+macro('RmsExcited',f"{rms['excited']:.3f}")
for name,val in [('MaxResidual',maxres),('MaxSymmetry',maxsym),('MaxOrtho',maxorth)]:
    mant,exp=f'{val:.2e}'.split('e');text+=macro(name,mant+'\\times10^{'+str(int(exp))+'}')
text+=macro('Nsolves',str(len(records)))
(R/'numbers.tex').write_text(text)
summary=dict(solves=len(records),max_target_residual=maxres,max_target_symmetry_error=maxsym,max_orthogonality_error=maxorth,
             all_targets_are_lowest_two=True,benchmark=bench,pair_rms=rms,largest_box_root=roots[-1],
             last_root_change=abs(roots[-1]['cstar']-roots[-2]['cstar']),
             interpretation='Last-box difference is a convergence diagnostic, not an error bound or confidence interval.')
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='benchmark'},indent=2))
