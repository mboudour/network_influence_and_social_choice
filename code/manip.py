# Item 2: probability that SOME committed agent can manipulate g = (ranked pairs on weighted margins) o F,
# 3 alternatives, impartial culture, k iid weights (Pareto alpha, or equal weights).
import numpy as np, sys
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
RANK=np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])        # rankings as orders (best first)
POS=np.argsort(RANK,1)                                                    # POS[r,x] = position of x
S=np.stack([np.sign(POS[:,1]-POS[:,0]),np.sign(POS[:,2]-POS[:,1]),np.sign(POS[:,2]-POS[:,0])],1)  # (ab,bc,ac): +1 if first preferred
def winner(M):            # M[...,3] margins (ab,bc,ac); ranked pairs = drop weakest edge if cyclic
    ab,bc,ac=M[...,0],M[...,1],M[...,2]
    cyc=((ab>0)&(bc>0)&(ac<0))|((ab<0)&(bc<0)&(ac>0))
    absM=np.abs(M); weakest=np.argmin(absM,-1)
    keep=np.ones(M.shape,bool)
    for e in range(3): keep[...,e]&=~(cyc&(weakest==e))
    # losses: a loses on ab<0 or ac<0; b loses ab>0 or bc<0; c loses bc>0 or ac>0
    la=keep[...,0]&(ab<0)|keep[...,2]&(ac<0); lb=keep[...,0]&(ab>0)|keep[...,1]&(bc<0); lc=keep[...,1]&(bc>0)|keep[...,2]&(ac>0)
    return np.where(~la,0,np.where(~lb,1,2))
def prob(alpha,k,R):
    hits=0
    for _ in range(R):
        b=np.ones(k) if alpha is None else (1-rng.random(k))**(-1/alpha)
        r=rng.integers(6,size=k); M=(b[:,None]*S[r]).sum(0); w=winner(M)
        Mp=M[None,None,:]+b[:,None,None]*(S[None,:,:]-S[r][:,None,:])    # (k,6,3)
        wp=winner(Mp)                                                      # (k,6)
        better=POS[r][:,None].repeat(6,1)                                 # positions under true ranking
        gain=np.take_along_axis(POS[r],wp,1)<POS[r][:,w][:,None]
        hits+=gain.any()
    return hits/R
if __name__=="__main__":
  for alpha in (None,3.0,2.5,1.8,1.5,1.2,0.8,0.5,0.3):
      row=[prob(alpha,k,4000 if k<=300 else 1500) for k in (11,31,101,301,1001)]
      print("equal" if alpha is None else alpha, " ".join(f"{x:.3f}" for x in row), flush=True)
