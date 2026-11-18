"""Independent tests of kinetic normalization, symmetry and eigensolutions."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from model import geometry,solve,symmetry_maps
out={}
L=3
T2,F2,G2=geometry(L,2)
T3,F3,G3=geometry(L,3)
# Exact plane-wave dispersions derived independently from individual momenta.
p=2*np.pi*np.arange(L)/L
q=np.array(np.meshgrid(p,p,indexing='ij')).reshape(2,-1).T
eps=lambda k: 2*np.sum(1-np.cos(k),axis=-1)
exact2=np.sort(2*eps(q))
exact3=np.sort(np.array([eps(k)+eps(l)+eps(k+l) for k in q for l in q]))
out['free_spectrum_two_max_error']=float(np.max(abs(np.linalg.eigvalsh(T2.toarray())-exact2)))
out['free_spectrum_three_max_error']=float(np.max(abs(np.linalg.eigvalsh(T3.toarray())-exact3)))
H=T3+diags(G3-5*F3)
out['hermiticity_max_error']=float(abs(H-H.T).max())
out['constant_free_state_residual']=float(np.linalg.norm(T3@np.ones(L**4)))
maps=symmetry_maps(L)
rng=np.random.default_rng(17)
x=rng.normal(size=L**4)
out['commutator_relative_errors']={name:float(np.linalg.norm((H@x)[ix]-H@x[ix])/np.linalg.norm(H@x)) for name,ix in maps.items()}
e,u,r,s=solve(T3,F3,G3,5,k=4,tol=1e-12)
out['dense_vs_sparse_max_error']=float(np.max(abs(e-np.linalg.eigvalsh(H.toarray())[:4])))
# Hellmann-Feynman slope vs central finite difference at nondegenerate low levels.
L=8
h=1e-4
slopes=[]
finite=[]
for n,k in [(2,1),(3,2)]:
 T,F,G=geometry(L,n)
 e,u,r,s=solve(T,F,G,5,k=k,tol=1e-12)
 ep=solve(T,F,G,5+h,k=k,tol=1e-12)[0]
 em=solve(T,F,G,5-h,k=k,tol=1e-12)[0]
 slopes.extend(s.tolist());finite.extend(((ep-em)/(2*h)).tolist())
out['HF_slopes']=slopes
out['finite_difference_slopes']=finite
out['HF_max_error']=float(np.max(abs(np.array(slopes)-finite)))
# Repeat production benchmark with stricter tolerance and independent seed.
L=20
T,F,G=geometry(L,3)
e,u,r,s=solve(T,F,G,5,k=4,tol=1e-12,seed=73)
records=json.loads((Path(__file__).parent/'data/L20.json').read_text())
base=records['5.0000000000']
out['tight_tolerance_seed_energy_difference']=float(np.max(abs(e[:2]-[base['E31'],base['E32']])) )
out['tight_tolerance_residuals']=r.tolist()
out['benchmark_three_absolute_errors']=abs(e[:2]-[-4.771,-1.109]).tolist()
out['benchmark_two_absolute_error']=abs(base['E2']-(-.821))
assert out['free_spectrum_three_max_error']<1e-10
assert out['free_spectrum_two_max_error']<1e-10
assert out['hermiticity_max_error']==0
assert max(out['commutator_relative_errors'].values())<1e-12
assert out['dense_vs_sparse_max_error']<1e-10
assert out['HF_max_error']<1e-7
assert out['tight_tolerance_seed_energy_difference']<1e-9
assert max(out['benchmark_three_absolute_errors'])<.0005
assert out['benchmark_two_absolute_error']<.0005
out['passed']=True
(Path(__file__).parent/'data/validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
