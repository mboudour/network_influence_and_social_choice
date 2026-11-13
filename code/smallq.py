# small-amplitude check: zealots at +-q, q->0, subcritical g; F should equal sign(beta_g . sigma),
# beta_g = column sums of (I - s g Wii)^{-1} s g Wiz (gain-weighted Green operator, Thm 1 of the LLM-BVP paper)
import itertools, numpy as np, networkx as nx
from larger_graphs import graph, S
rng=np.random.default_rng(7); res=[]
for kind in ("ER","BA","SF"):
  for rep in range(20):
    G=graph(kind,60); n=G.number_of_nodes(); A=nx.to_numpy_array(G); W=A/A.sum(1,keepdims=True)
    k=5; Z=list(rng.choice(n,k,replace=False)); I=[i for i in range(n) if i not in Z]
    Wii,Wiz=W[np.ix_(I,I)],W[np.ix_(I,Z)]
    gs=1/max(abs(np.linalg.eigvals(S*Wii)))
    for B in (0.5,0.9):
      g=B*gs; bw=np.linalg.solve(np.eye(len(I))-S*g*Wii,S*g*Wiz).sum(0)
      for q in (1.0,1e-2,1e-4):
        ok=True
        for sig in itertools.product([-1,1],repeat=k):
          if sig[0]<0: continue
          sz=q*np.array(sig,float); v=np.zeros(len(I))
          for _ in range(200000):
            nv=S*(Wii@np.tanh(g*v)+Wiz@np.tanh(g*sz))
            if np.max(np.abs(nv-v))<1e-14*max(q,1e-3): break
            v=nv
          ok&= np.sign(v.sum()+q*np.sum(sig))==np.sign(np.dot(1+bw,sig))
        res.append((kind,B,q,ok))
import collections
c=collections.Counter((k,B,q) for k,B,q,ok in res if ok); t=collections.Counter((k,B,q) for k,B,q,ok in res)
for key in sorted(t): print(key, c[key],"/",t[key])
