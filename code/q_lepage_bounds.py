# Certified bounds for the limit q(alpha) from the LePage series (Theorem 14(ii)): explicit points j<=J,
# Gaussian remainder with the exact variance of the omitted points. Lower bound = manipulations found among
# explicit agents; upper bound adds draws that Lemma 13 cannot certify against the omitted agents.
import json, sys
from q_certified import q
al=[0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.5,1.6,1.7,1.8,1.9,1.95,1.98]
out=[]
for i,a in enumerate(al):
    lo,unc=q(a,10000,10000,True,seed=2000+i); out.append((a,lo,lo+unc)); print(out[-1],flush=True)
    json.dump(out,open("outputs/q_lepage_bounds.json","w"))
