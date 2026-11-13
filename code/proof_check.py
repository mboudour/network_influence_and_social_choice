# Check: p(alpha) = 1/4 - (3/4) P(|L + (alpha/2) V| < log 2),
# L standard logistic; V = Laplace mixture: rate (2k-1)/2 w.p. 8/((2k-1)^2 pi^2)  (CF tanh(pi t)/(pi t))
import numpy as np
from p_alpha import p   # earlier closed-form integral
rng=np.random.default_rng(0); M=4_000_000
L=rng.logistic(size=M)
K=np.arange(1,20001); w=8/((2*K-1)**2*np.pi**2); w/=w.sum()
k=rng.choice(K,size=M,p=w); V=rng.laplace(scale=2/(2*k-1))
for al in (0.3,1.0,1.5,1.8,1.999):
    q=np.mean(np.abs(L+al/2*V)<np.log(2))
    print(f"alpha={al}: decomposition {0.25-0.75*q:.5f}   integral {p(al):.5f}")
