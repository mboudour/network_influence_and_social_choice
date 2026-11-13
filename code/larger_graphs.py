"""Does the nonlinear (tanh) induced rule F differ from the linear harmonic weighted majority on larger graphs?
ER, BA, Chung-Lu-type (configuration) graphs; k zealots placed at random or at top-degree nodes.
Outcome = sign of interior mean opinion.  Interior starts at 0.
"""
import itertools, sys, json
import numpy as np
import networkx as nx

S = 0.8
TOL, TMAX = 1e-11, 100000
rng = np.random.default_rng(1)


def steady(Wii, Wiz, sig, g):
    v = np.zeros(Wii.shape[0]); bz = Wiz @ np.tanh(g * sig)
    for _ in range(TMAX):
        nv = S * (Wii @ np.tanh(g * v) + bz)
        if np.max(np.abs(nv - v)) < TOL:
            return nv, True
        v = nv
    return v, False


def graph(kind, n):
    while True:
        if kind == "ER": G = nx.gnp_random_graph(n, 6 / n, seed=int(rng.integers(1e9)))
        elif kind == "BA": G = nx.barabasi_albert_graph(n, 3, seed=int(rng.integers(1e9)))
        elif kind == "SF":  # heavy-tailed Chung-Lu, exponent 2.5
            w = (np.arange(1, n + 1) ** (-1 / 1.5)); w *= 6 * n / w.sum()
            G = nx.expected_degree_graph(list(w), seed=int(rng.integers(1e9)), selfloops=False)
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        if G.number_of_nodes() > n // 2:
            return nx.convert_node_labels_to_integers(G)


def one(kind, n, k, place, betas):
    G = graph(kind, n); n = G.number_of_nodes()
    A = nx.to_numpy_array(G); W = A / A.sum(1, keepdims=True)
    deg = A.sum(1)
    Z = list(np.argsort(-deg)[:k]) if place == "hub" else list(rng.choice(n, k, replace=False))
    I = [i for i in range(n) if i not in Z]
    Wii, Wiz = W[np.ix_(I, I)], W[np.ix_(I, Z)]
    Hm = np.linalg.solve(np.eye(len(I)) - S * Wii, S * Wiz); bw = Hm.sum(0)
    gstar = 1 / max(abs(np.linalg.eigvals(S * Wii)))
    out = dict(kind=kind, n=n, k=k, place=place, share=float(bw.max() / bw.sum()), res=[])
    for B in betas:
        g = B * gstar; agree = True; conv = True; F = {}
        for sig in itertools.product([-1, 1], repeat=k):
            if sig[0] == -1: continue
            v, c = steady(Wii, Wiz, np.array(sig, float), g); conv &= c
            m = v.mean(); f = 0 if abs(m) < 1e-10 else int(np.sign(m))
            lin = np.dot(bw, sig); fl = 0 if abs(lin) < 1e-10 else int(np.sign(lin))
            F[sig] = f; agree &= (f == fl)
        dict_ = [d for d in range(k) if all(F[s] == s[d] for s in F)]
        out["res"].append(dict(beta=B, agree=bool(agree), conv=bool(conv), dict=dict_[0] if dict_ else None,
                               ties=sum(F[s] == 0 for s in F)))
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]); out = []
    for kind in ("ER", "BA", "SF"):
        for k in (3, 5):
            for place in ("random", "hub"):
                for _ in range(reps):
                    out.append(one(kind, 60, k, place, [0.5, 0.9, 1.5, 3.0, 10.0]))
    json.dump(out, open(sys.argv[2], "w")); print(len(out))
