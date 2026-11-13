# Generalization test: P(induced rule is dictatorial) = P(max gain-weighted harmonic weight > sum of the others)
# and P(Condorcet cycle) for zealot rankings drawn uniformly (m=3), random zealot placement.
import numpy as np, networkx as nx, scipy.sparse as sp, scipy.sparse.linalg as spl, itertools, collections
S=0.8; B=0.5; rng=np.random.default_rng(3); n=2000
PER=list(itertools.permutations(range(3)))
def G_of(kind):
    if kind=="ER": G=nx.gnp_random_graph(n,6/n,seed=int(rng.integers(1e9)))
    else:
        gam=float(kind); w=np.arange(1,n+1)**(-1/(gam-1)); w*=6*n/w.sum()
        G=nx.expected_degree_graph(list(w),seed=int(rng.integers(1e9)),selfloops=False)
    G=G.subgraph(max(nx.connected_components(G),key=len)).copy(); return nx.convert_node_labels_to_integers(G)
out=collections.defaultdict(list)
for kind in ("ER","3.0","2.5","2.2"):
  for rep in range(6):
    G=G_of(kind); N=G.number_of_nodes(); A=nx.to_scipy_sparse_array(G,format="csr",dtype=float)
    d=np.asarray(A.sum(1)).ravel(); W=sp.diags(1/d)@A
    for k in (3,5,9,15):
      for trial in range(25):
        Z=rng.choice(N,k,replace=False); mask=np.ones(N,bool); mask[Z]=False; I=np.where(mask)[0]
        Wii=W[I][:,I]; Wiz=W[I][:,Z].toarray()
        # g = B/ρ(S Wii) ; ρ(Wii)<=1, approximate by 1 (row-stochastic minus small leak): use power iteration
        x=np.ones(len(I))
        for _ in range(200): x=Wii@x; lam=np.linalg.norm(x); x/=lam
        g=B/(S*lam)
        M=(sp.eye(len(I))-S*g*Wii).tocsc()
        beta=1+spl.splu(M).solve(S*g*Wiz).sum(0)
        dic = beta.max()>beta.sum()-beta.max()
        # cycle probability under uniform zealot rankings
        cyc=0; T=300
        for _ in range(T):
            prof=[PER[i] for i in rng.integers(6,size=k)]
            r=[]
            for a,b in ((0,1),(1,2),(0,2)):
                sig=np.array([1 if p.index(a)<p.index(b) else -1 for p in prof]); r.append(np.sign(beta@sig))
            cyc+= (r[0]==r[1]==-r[2]!=0)
        out[(kind,k)].append((dic,cyc/T,np.corrcoef(beta,d[Z])[0,1] if k>3 else np.nan))
print("graph  k  P(dictator)  P(cycle|uniform)  corr(beta,degree)")
for key in out:
    a=np.array(out[key],float); print(key, round(a[:,0].mean(),3), round(a[:,1].mean(),4), round(np.nanmean(a[:,2]),3))
