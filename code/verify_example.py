# Checks the five-agent manipulation used in the proof of Theorem 14(iii).
import numpy as np
from manip import S, POS, winner
names = ["abc", "acb", "bac", "bca", "cab", "cba"]
w = np.array([1, 0.95, 0.85, 0.8, 0.75])
prof = [names.index(r) for r in ("bac", "bca", "abc", "abc", "cab")]
M = (w[:, None] * S[prof]).sum(0)
Mp = M + w[0] * (S[names.index("bca")] - S[prof[0]])
hmin = lambda z: min(abs(z[0]), abs(z[1]), abs(z[2]), *[abs(z[e] + s * z[f]) for e in range(3) for f in range(e + 1, 3) for s in (1, -1)])
print("margins (ab, bc, ac):", M, "winner:", "abc"[winner(M)])
print("after misreport bca:", Mp, "winner:", "abc"[winner(Mp)])
print("true ranking of agent 1: b > a > c; distance to boundaries:", hmin(M), hmin(Mp))
