# Item 2 check: Green weights vs degree on a large power-law configuration model.
# beta_z = kappa * sum_{i~z, i interior} u_i/d_i,  u = 1 + kappa * W_OO^T u  (kappa = s g < 1)
# Lemma: kappa*sum_{i~z} 1/d_i <= beta_z <= kappa/(1-kappa) * d_z ;  claim beta_z/d_z -> kappa*E[r*], tail index preserved.
import numpy as np, networkx as nx, scipy.sparse as sp
rng=np.random.default_rng(2); n=200000; gam=2.5; kappa=0.5
deg=np.floor((1-rng.random(n))**(-1/(gam-1))*2).astype(int); deg=np.minimum(deg,int(n**0.5))
if deg.sum()%2: deg[0]+=1
G=nx.Graph(nx.configuration_model(deg,seed=3)); G.remove_edges_from(nx.selfloop_edges(G))
G=G.subgraph(max(nx.connected_components(G),key=len)).copy(); G=nx.convert_node_labels_to_integers(G)
A=nx.to_scipy_sparse_array(G,format="csr",dtype=float); d=np.asarray(A.sum(1)).ravel(); N=len(d)
B=[];D=[];LO=[];HI=[]
for rep in range(40):
    Z=rng.choice(N,15,replace=False); mask=np.ones(N); mask[Z]=0
    r=np.zeros(N)
    for _ in range(80):                       # r = (1 + kappa * A_OO r)/d on interior; contraction in sup norm
        r=mask*(1+kappa*(A@(mask*r)))/d
    for z in Z:
        nb=A[[z]].indices; nb=nb[mask[nb]>0]
        B.append(kappa*r[nb].sum()); D.append(d[z]); LO.append(kappa*(1/d[nb]).sum()); HI.append(kappa/(1-kappa)*len(nb))
B,D,LO,HI=map(np.array,(B,D,LO,HI))
print("N=",N," bounds hold:", bool(np.all(LO<=B+1e-12)), bool(np.all(B<=HI+1e-12)))
print("corr(log beta, log d) =", np.corrcoef(np.log(B),np.log(D))[0,1])
np.savez("outputs/figdata3.npz",B=B,D=D,LO=LO,HI=HI)
for lo,hi in ((1,3),(3,10),(10,30),(30,100),(100,1e9)):
    s=(D>=lo)&(D<hi)
    if s.sum(): print(f"d in [{lo},{hi}): n={s.sum():4d}  mean beta/d = {np.mean(B[s]/D[s]):.4f}  sd = {np.std(B[s]/D[s]):.4f}")
def hill(x,k): x=np.sort(x)[::-1]; return 1/np.mean(np.log(x[:k]/x[k]))
print("Hill tail index (top 60): beta", round(hill(B,60),2), " degree", round(hill(D,60),2), " theory alpha =", gam-1)
