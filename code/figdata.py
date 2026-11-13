# Data for the three figures (saved as .npz); plotting is separate.
import numpy as np, networkx as nx
from p_alpha import p
rng=np.random.default_rng(21)
# Fig 1: p(alpha) curve + Pareto MC at k=1000
al=np.linspace(0.02,1.995,200); pc=np.array([p(a) for a in al])
X=np.array([[1,1,1],[1,-1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]])  # (ab,bc,ac) for the 6 rankings
def cyc(alpha,k,R):
    b=(1-rng.random((R,k)))**(-1/alpha); r=rng.integers(6,size=(R,k)); M=np.einsum('rk,rkj->rj',b,X[r]); s=np.sign(M)
    return ((s[:,0]==s[:,1])&(s[:,1]==-s[:,2])).mean()
mcA=np.array([0.6,1.0,1.2,1.5,1.8]); mcP=np.array([cyc(a,1000,20000) for a in mcA])
# Fig 2: P(dictator) vs k
ks=np.array([5,10,20,50,100,200,500,1000]); alphas=np.array([1.2,1.5,1.8]); D=np.zeros((3,len(ks)))
for i,a in enumerate(alphas):
    for j,k in enumerate(ks):
        R=int(4e6//k); b=(1-rng.random((R,k)))**(-1/a); mx=b.max(1); D[i,j]=(mx>b.sum(1)-mx).mean()
np.savez('outputs/figdata12.npz',al=al,pc=pc,mcA=mcA,mcP=mcP,ks=ks,alphas=alphas,D=D)
print(mcP, D)
