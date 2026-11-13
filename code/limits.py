# Item 3: order of limits.  Exact P_n(dictator) (zealots committed jointly) vs
#  (a) iid-from-empirical prediction: sample k vertices, single-zealot weights beta~ (no interaction)
#  (b) asymptotic C k^{1-alpha} = k * L * ((k-1) mu)^{-alpha}
import numpy as np, networkx as nx, sys
gam=2.5; al=gam-1; kappa=0.5; rng=np.random.default_rng(int(sys.argv[2]))
n=int(sys.argv[1]); SETS=int(sys.argv[3])
deg=np.floor((1-rng.random(n))**(-1/al)*2).astype(int)
if deg.sum()%2: deg[0]+=1
G=nx.Graph(nx.configuration_model(deg,seed=int(rng.integers(1e9)))); G.remove_edges_from(nx.selfloop_edges(G))
G=G.subgraph(max(nx.connected_components(G),key=len)).copy(); G=nx.convert_node_labels_to_integers(G)
A=nx.to_scipy_sparse_array(G,format="csr",dtype=float); d=np.asarray(A.sum(1)).ravel(); N=len(d)
def r_of(mask):
    r=np.zeros(N)
    for _ in range(45): r=mask*(1+kappa*(A@(mask*r)))/d
    return r
r0=r_of(np.ones(N)); bt=1+kappa*(A@r0)          # approx single-zealot weights for every vertex
mu=bt.mean(); cfac=np.mean((bt-1)[d>=10]/d[d>=10]); L=(2*cfac)**al   # tail of omega=1+beta ~ (cfac*d)
print(f"n={n} N={N} dmax={int(d.max())} mean deg={mu:.2f}")
for k in (3,5,9,15,30):
    ex=0; ap=0
    for _ in range(SETS):
        Z=rng.choice(N,k,replace=False); mask=np.ones(N); mask[Z]=0; r=r_of(mask)
        b=np.array([1+kappa*r[A[[z]].indices].sum() for z in Z]); ex+= b.max()>b.sum()-b.max()
    S=rng.choice(N,(20000,k)); B=bt[S]; ap=np.mean(B.max(1)>B.sum(1)-B.max(1))
    print(f" k={k:3d}  exact={ex/SETS:.3f}  iid-empirical={ap:.3f}  asymptotic={min(1,k*L*((k-1)*mu)**(-al)):.3f}")
