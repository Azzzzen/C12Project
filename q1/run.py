"""Reproducible Q1 scan. Cached scalar results are written after each solve."""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from model import geometry, solve, symmetry_maps
BASE = Path(__file__).resolve().parent
DATA = BASE/'data'
DATA.mkdir(exist_ok=True)

def calculate(L, cs, roots=False, tol=1e-10, k_start=4):
    path = DATA/f'L{L}.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    T2,F2,G2 = geometry(L,2)
    T3,F3,G3 = geometry(L,3)
    maps = symmetry_maps(L)
    def evaluate(c):
        key = f'{c:.10f}'
        if key in records:
            return records[key]['ER']
        start=time.monotonic()
        e2,u2,r2,s2=solve(T2,F2,G2,c,k=1,tol=tol)
        k=k_start
        while True:
            e,u,r,s=solve(T3,F3,G3,c,k=k,tol=tol)
            errors={name: np.linalg.norm(u[ix]-u,axis=0) for name,ix in maps.items()}
            symmetric=np.max(list(errors.values()),axis=0)<1e-6
            chosen=np.flatnonzero(symmetric)[:2]
            if len(chosen)==2: break
            k*=2
            if k>32: raise RuntimeError('Target sector not resolved')
        j0,j1=chosen
        result=dict(L=L,c=float(c),E2=float(e2[0]),E31=float(e[j0]),E32=float(e[j1]),
                    ER=float(e[j1]-e2[0]),B=float(e2[0]-e[j0]),
                    low_energies=e.tolist(),selected=chosen.tolist(),
                    residual2=float(r2[0]),residual3=r.tolist(),
                    symmetry={name:v.tolist() for name,v in errors.items()},
                    slopes=[float(s2[0]),float(s[j0]),float(s[j1])],
                    orthogonality=float(np.linalg.norm(u.T@u-np.eye(k))),tol=tol)
        records[key]=result
        path.write_text(json.dumps(records,indent=2)+'\n')
        if abs(c-5)<1e-12 and L==20:
            np.savez_compressed(DATA/'states_L20_c5.npz',ground=u[:,j0].reshape((L,)*4),excited=u[:,j1].reshape((L,)*4))
        print(f'L={L} c={c:.8f}: E2={e2[0]:.9f} E31={e[j0]:.9f} E32={e[j1]:.9f} ER={result["ER"]:.9f}; {time.monotonic()-start:.1f}s',flush=True)
        return result['ER']
    for c in cs: evaluate(float(c))
    if roots:
        xs=sorted(float(c) for c in cs)
        intervals=[(a,b) for a,b in zip(xs[:-1],xs[1:]) if evaluate(a)*evaluate(b)<0]
        rr=[]
        for a,b in intervals:
            root=brentq(evaluate,a,b,xtol=2e-7)
            rr.append({'L':L,'cstar':root,'ER':evaluate(root),'bracket':[a,b],'xtol':2e-7})
        (DATA/f'roots_L{L}.json').write_text(json.dumps(rr,indent=2)+'\n')
        print('ROOTS',rr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--L',nargs='+',type=int,default=[8,12,16,20])
    p.add_argument('--cs',nargs='+',type=float)
    p.add_argument('--roots',action='store_true')
    p.add_argument('--k',type=int,default=4)
    a=p.parse_args()
    for L in a.L: calculate(L,a.cs if a.cs is not None else np.arange(3.5,7.001,.25),a.roots,k_start=a.k)
