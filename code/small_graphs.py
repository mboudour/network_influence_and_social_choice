"""Option 2 small-graph check.
Pairwise channel-mediated FJ (tanh), k zealots at +/-1 per pair, neutral interior anchors,
interior starts at 0.  Per pair the outcome depends only on the zealot sign vector sigma,
so the network induces one pairwise rule F:{+-1}^k -> {-1,0,1} used for every pair.
We record: dictatorship, agreement with the linear harmonic weighted majority,
and whether some zealot ranking profile over 3 alternatives gives a collective cycle.
"""
import itertools, sys, json
import numpy as np
import networkx as nx
from networkx.generators.atlas import graph_atlas_g

S = 0.8                      # homogeneous susceptibility
BETAS = [None, 0.5, 0.9, 1.5, 3.0]   # None = linear (identity channel); else g = beta * g*_C
TOL, TMAX = 1e-12, 200000
PERMS3 = list(itertools.permutations(range(3)))
PAIRS3 = [(0, 1), (1, 2), (0, 2)]


def steady(Wii, Wiz, sig, g):
    v = np.zeros(Wii.shape[0])
    bz = Wiz @ (sig if g is None else np.tanh(g * sig))
    for t in range(TMAX):
        nv = S * (Wii @ (v if g is None else np.tanh(g * v)) + bz)
        if np.max(np.abs(nv - v)) < TOL:
            return nv, True
        v = nv
    return v, False


def outcomes(Wii, Wiz, k, g):
    """F for every sigma, two collective readings: sign of interior mean, majority vote of all agents."""
    Fm, Fv, ok = {}, {}, True
    for sig in itertools.product([-1, 1], repeat=k):
        if sig in Fm:
            continue
        v, conv = steady(Wii, Wiz, np.array(sig, float), g)
        ok &= conv
        m = v.sum() + float(np.sum(sig))   # interior declarations + committed declarations (q=1)
        fm = 0 if abs(m) < 1e-9 else int(np.sign(m))
        signs = [0 if abs(x) < 1e-9 else int(np.sign(x)) for x in v] + list(sig)
        tot = sum(signs)
        fv = int(np.sign(tot))
        neg = tuple(-x for x in sig)
        Fm[sig], Fm[neg] = fm, -fm
        Fv[sig], Fv[neg] = fv, -fv
    return Fm, Fv, ok


def dictator(F, k):
    for d in range(k):
        if all(F[s] == s[d] for s in F):
            return d
    return None


def has_cycle(F, k):
    """Is there a profile of zealot rankings on 3 alternatives giving a collective (weak) cycle?"""
    for prof in itertools.product(PERMS3, repeat=k):
        rel = {}
        for a, b in PAIRS3:
            sig = tuple(1 if p.index(a) < p.index(b) else -1 for p in prof)
            rel[(a, b)] = F[sig]
        # cycle 0>1>2>0 or reverse, strict
        if rel[(0, 1)] == 1 and rel[(1, 2)] == 1 and rel[(0, 2)] == -1:
            return True
        if rel[(0, 1)] == -1 and rel[(1, 2)] == -1 and rel[(0, 2)] == 1:
            return True
    return False


def run(nmin, nmax, k):
    rows = []
    for G in graph_atlas_g():
        n = G.number_of_nodes()
        if n < nmin or n > nmax or not nx.is_connected(G):
            continue
        A = nx.to_numpy_array(G)
        W = A / A.sum(1, keepdims=True)
        for Z in itertools.combinations(range(n), k):
            I = [i for i in range(n) if i not in Z]
            Wii, Wiz = W[np.ix_(I, I)], W[np.ix_(I, list(Z))]
            # every interior node must be reachable from some zealot (else it stays 0)
            H = G.subgraph(I)
            if any(not any(W[i, z] > 0 for i in comp for z in Z) for comp in nx.connected_components(H)):
                continue
            rho = max(abs(np.linalg.eigvals(S * Wii))) if len(I) else 0
            gstar = np.inf if rho == 0 else 1.0 / rho
            # linear harmonic weights of zealots
            Hm = np.linalg.solve(np.eye(len(I)) - S * Wii, S * Wiz)
            beta_w = Hm.sum(0)
            for B in BETAS:
                if B is not None and not np.isfinite(gstar):
                    continue
                g = None if B is None else B * gstar
                Fm, Fv, ok = outcomes(Wii, Wiz, k, g)
                rec = dict(n=n, m_edges=G.number_of_edges(), Z=Z, beta=B, conv=bool(ok),
                           wmax_share=float((1+beta_w).max() / (1+beta_w).sum()))
                for name, F in (("mean", Fm), ("vote", Fv)):
                    d = dictator(F, k)
                    lin = all(F[s] == (0 if abs(np.dot(1+beta_w, s)) < 1e-9 else int(np.sign(np.dot(1+beta_w, s)))) for s in F)
                    rec[name] = dict(dict=d, ties=sum(1 for s in F if F[s] == 0), lin=lin,
                                     cyc=has_cycle(F, k), F={"".join("+" if x > 0 else "-" for x in s): F[s] for s in F})
                rows.append(rec)
    return rows


if __name__ == "__main__":
    nmin, nmax, k, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rows = run(nmin, nmax, k)
    json.dump(rows, open(out, "w"))
    print(len(rows), "records ->", out)
