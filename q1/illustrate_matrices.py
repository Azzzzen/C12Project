"""Pedagogical matrices and stencils; verify Kronecker construction independently."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/carbon-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from model import geometry
BASE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['pdf','svg','png']:
  fig.savefig(BASE/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
# Figure contract: the mixed diagonal stencil comes from motion of particle 3.
# Schematic-led quantitative figure; coordinates are a local x-direction slice,
# not particle positions or a replacement one-dimensional model.
fig,ax=plt.subplots(figsize=(4.5,2.5),layout='constrained')
for x in [-1,0,1]:
 for y in [-1,0,1]:
  ax.scatter(x,y,s=35,color='#dadfe2',zorder=2)
styles=[((1,0),'#477ca8',r'Particle 1: $(\pm1,0)$'),((0,1),'#c58b62',r'Particle 2: $(0,\pm1)$'),((1,1),'#2d857a',r'Particle 3: $(\pm1,\pm1)$')]
for (x,y),col,lab in styles:
 for sign in [-1,1]:
  ax.annotate('',xy=(sign*x,sign*y),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=1.7,shrinkA=4,shrinkB=3))
 ax.plot([],[],color=col,lw=1.7,label=lab)
ax.scatter(0,0,s=48,color='#333333',zorder=3)
ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),xticks=[-1,0,1],yticks=[-1,0,1],xlabel=r'$\Delta r_{1x}/a$',ylabel=r'$\Delta r_{2x}/a$')
ax.set_aspect('equal');ax.grid(color='#eeeeee',lw=.5)
ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
save(fig,'relative_hops')

L=3;n=L*L
I=np.eye(L)
S=np.zeros((L,L))
for i in range(L):S[i,(i+1)%L]=1
Sx=np.kron(S,I);Sy=np.kron(I,S)
Id=np.eye(n)
K2=8*Id-2*(Sx+Sx.T+Sy+Sy.T)
K3=12*np.eye(n*n)
for U in [Sx,Sy]:
 K3-=np.kron(U+U.T,Id)+np.kron(Id,U+U.T)+np.kron(U,U)+np.kron(U.T,U.T)
T2,F2,G2=geometry(L,2);T3,F3,G3=geometry(L,3)
assert np.array_equal(K2,T2.toarray())
assert np.array_equal(K3,T3.toarray())
# Independently assemble all 9x9 blocks, matching the displayed expanded matrix.
block=np.zeros_like(K3)
for a in range(n):
 for b in range(n):
  if a==b:
   B=12*Id-(Sx+Sx.T+Sy+Sy.T)
  else:
   B=np.zeros((n,n))
   for U in [Sx,Sx.T,Sy,Sy.T]:
    if U[a,b]: B-=Id+U
  block[a*n:(a+1)*n,b*n:(b+1)*n]=B
assert np.array_equal(block,K3)
expected=[0,1,2,3,6,9,10,18,20,27,30,54,60]
assert np.flatnonzero(K3[0]).tolist()==expected
np.savetxt(BASE/'data/kinetic_two_L3.csv',K2,delimiter=',',fmt='%d')
np.savetxt(BASE/'data/kinetic_three_L3.csv',K3,delimiter=',',fmt='%d')
# Generate the full 9x9 H2 matrix with symbolic t and potential values.
rows=[]
for i,row in enumerate(K2.astype(int)):
 vals=[]
 for j,v in enumerate(row):
  vals.append(r'8t+v_{%d%d}'%(i//L,i%L) if i==j else ('-2t' if v==-2 else '0'))
 rows.append(' & '.join(vals)+r' \\')
(BASE/'report/two_body_matrix.tex').write_text(r'\begin{pmatrix}'+'\n'+'\n'.join(rows)+'\n'+r'\end{pmatrix}'+'\n')
# Generate full 9x9 block structure of H3 (each entry is 9x9).
rows=[]
for i in range(n):
 vals=[]
 for j in range(n):
  val=r'A_{%d%d}'%(i//L,i%L) if i==j else '0'
  if i!=j:
   for U,label in [(Sx,'B_x'),(Sx.T,r'B_x^T'),(Sy,'B_y'),(Sy.T,r'B_y^T')]:
    if U[i,j]:val=label
  vals.append(val)
 rows.append(' & '.join(vals)+r' \\')
(BASE/'report/three_body_blocks.tex').write_text(r'\begin{pmatrix}'+'\n'+'\n'.join(rows)+'\n'+r'\end{pmatrix}'+'\n')
fig,axs=plt.subplots(1,2,figsize=(7.12,2.65),layout='constrained')
colors=ListedColormap(['#477ca8','#ffffff','#c58b62'])
norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
for ax,M,label,title in zip(axs,[K2,K3],['a','b'],[r'Two-body $T_2$: $9\times9$',r'Three-body $T_3$: $81\times81$']):
 cat=np.ones_like(M);cat[M<0]=0;cat[M>0]=2
 ax.imshow(cat,cmap=colors,norm=norm,interpolation='nearest',origin='upper')
 ax.set(xlabel='Column index',ylabel='Row index',title=title)
 ax.text(-.21,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
 if len(M)==9:
  ax.set(xticks=[0,4,8],yticks=[0,4,8])
  ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True)
  ax.grid(which='minor',color='#dddddd',lw=.3);ax.tick_params(which='minor',length=0)
  for i in range(9):
   for j in range(9):
    if M[i,j]:ax.text(j,i,str(int(M[i,j])),ha='center',va='center',fontsize=6,color='white' if M[i,j]<0 else 'black')
 else:
  ax.set(xticks=[0,20,40,60,80],yticks=[0,20,40,60,80])
  for k in np.arange(8.5,81,9):
   ax.axhline(k,color='#bfc4c7',lw=.35);ax.axvline(k,color='#bfc4c7',lw=.35)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#477ca8',label='Hopping (negative)'),Patch(facecolor='white',edgecolor='#cccccc',label='Zero'),Patch(color='#c58b62',label='Diagonal (positive)')],loc='outside lower center',ncol=3,frameon=False,fontsize=8)
save(fig,'matrix_structure')
qa={'L':L,'two_body_kronecker_max_error':float(abs(K2-T2.toarray()).max()),'three_body_kronecker_max_error':float(abs(K3-T3.toarray()).max()),'three_body_block_max_error':float(abs(block-K3).max()),'three_body_first_row_columns':expected,'two_body_nnz':int(np.count_nonzero(K2)),'three_body_nnz':int(np.count_nonzero(K3)),'passed':True}
(BASE/'data/matrix_validation.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa,indent=2))
