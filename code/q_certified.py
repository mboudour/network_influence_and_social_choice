# Certified Monte Carlo for q(alpha) (Theorem 14): LePage weights Gamma_j^{-1/alpha}, j<=J, + Gaussian remainder
# with the exact variance of the omitted terms (optional). Agents are checked in decreasing weight order until
# Lemma 13 certifies that no unchecked agent can change the outcome: min_h |h(M)| > 4 * (largest unchecked weight).
# Draws that cannot be certified within the J explicit agents are counted separately.
import numpy as np, sys, json
from manip import S, POS, winner
C=np.cov(S.T,bias=True); Lc=np.linalg.cholesky(C)
def hmin(M):
    a,b,c=M; return min(abs(a),abs(b),abs(c),abs(a-b),abs(a+b),abs(a-c),abs(a+c),abs(b-c),abs(b+c))
def q(alpha,J,R,rem=True,seed=0,block=40):
    rng=np.random.default_rng(seed); hits=0; unc=0; p=2/alpha
    for _ in range(R):
        G=np.cumsum(rng.exponential(size=J)); b=G**(-1/alpha); r=rng.integers(6,size=J)
        M=(b[:,None]*S[r]).sum(0)
        if rem: M=M+np.sqrt(G[-1]**(1-p)/(p-1))*(Lc@rng.standard_normal(3))
        w=winner(M); hm=hmin(M); got=False; t=0
        while t<J:
            sl=slice(t,min(t+block,J))
            Mp=M[None,None,:]+b[sl,None,None]*(S[None,:,:]-S[r[sl]][:,None,:]); wp=winner(Mp); P=POS[r[sl]]
            if (np.take_along_axis(P,wp,1)<P[:,w][:,None]).any(): got=True; break
            t=sl.stop
            nxt=b[t] if t<J else G[-1]**(-1/alpha)
            if hm>4*nxt: break
        if got: hits+=1
        elif t>=J and hm<=4*G[-1]**(-1/alpha): unc+=1
    return hits/R, unc/R
if __name__=="__main__":
    out={}
    al=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.85,1.9,1.95,1.98,1.995]
    out["main"]=[(a,)+q(a,3000,20000,True,seed=1000+i) for i,a in enumerate(al)]
    print("main done",flush=True)
    sens=[]
    for a in (1.5,1.8,1.9,1.95):
        for J in (1000,3000,10000):
            for rem in (True,False):
                sens.append((a,J,rem)+q(a,J,8000,rem,seed=7)); print(sens[-1],flush=True)
    out["sens"]=sens
    json.dump(out,open("outputs/q_certified.json","w"))
