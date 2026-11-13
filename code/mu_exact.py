# Exact mu_k for finite k (Theorem 14): all agents are checked, but Lemma 13 lets us skip agents with
# 4*beta_j < min_h |h(M)|, who provably cannot change the outcome. No truncation, no remainder.
import numpy as np, sys, json
from manip import S, POS, winner
def hmin(M):
    a,b,c=M; return min(abs(a),abs(b),abs(c),abs(a-b),abs(a+b),abs(a-c),abs(a+c),abs(b-c),abs(b+c))
def mu(alpha,k,R,seed):
    rng=np.random.default_rng(seed); hits=0
    for _ in range(R):
        b=(1-rng.random(k))**(-1/alpha); r=rng.integers(6,size=k)
        M=(b[:,None]*S[r]).sum(0); w=winner(M)
        idx=np.nonzero(4*b>=hmin(M))[0]          # only these agents can change the outcome
        if idx.size==0: continue
        Mp=M[None,None,:]+b[idx,None,None]*(S[None,:,:]-S[r[idx]][:,None,:]); wp=winner(Mp); P=POS[r[idx]]
        hits+=(np.take_along_axis(P,wp,1)<P[:,w][:,None]).any()
    return hits/R
if __name__=="__main__":
    k=int(sys.argv[1]); R=int(sys.argv[2]); part=int(sys.argv[3])
    al=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.85,1.9,1.95,1.98]
    al=al[part::2]; out=[]
    for i,a in enumerate(al):
        out.append((a,mu(a,k,R,seed=5000+int(a*1000)+k))); print(out[-1],flush=True)
    json.dump(out,open(f"outputs/mu_k{k}_p{part}.json","w"))
