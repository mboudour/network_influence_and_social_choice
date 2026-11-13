# Step 2 check with iid Pareto(alpha) weights (x_m=1), impartial culture on 3 alternatives.
# (A) P(dictator) vs prediction k * P(beta > (k-1) mu) = k ((k-1) mu)^(-alpha), mu = alpha/(alpha-1)
# (B) P(cycle) vs Guilbaud 0.0877 (alpha>2) and below it (alpha<2)
import numpy as np
rng=np.random.default_rng(11)
PER=np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])  # position of alt a,b,c
# pairwise sign vectors for each ranking: (a>b, b>c, a>c)
X=np.array([[1 if p[0]<p[1] else -1, 1 if p[1]<p[2] else -1, 1 if p[0]<p[2] else -1] for p in PER])
guil=0.25-1.5/np.pi*np.arcsin(1/3)
print("Guilbaud", round(guil,4))
for alpha in (1.2,1.5,1.8,2.5,4.0):
    mu=alpha/(alpha-1)
    for k in (10,100,1000):
        R=20000 if k<1000 else 4000
        b=(1-rng.random((R,k)))**(-1/alpha)
        mx=b.max(1); dic=(mx>b.sum(1)-mx).mean()
        pred=k*((k-1)*mu)**(-alpha)
        r=rng.integers(6,size=(R,k)); M=np.einsum('rk,rkj->rj',b,X[r])
        s=np.sign(M); cyc=((s[:,0]==s[:,1])&(s[:,1]==-s[:,2])).mean()
        print(f"alpha={alpha} k={k:5d}  P(dict)={dic:.4f} pred={min(pred,1):.4f}  P(cycle)={cyc:.4f}")
