# Network influence and social choice

Code and outputs for the numerical results in *Dictators, Cycles and Manipulation under Network Influence* (M. Boudourides).

Run every script from the repository root, e.g. `python code/p_alpha.py`. Outputs are written to `outputs/`. Random seeds are fixed inside the scripts.

## Independent-weight model (exact theorems)

| Script | Result |
|---|---|
| `code/p_alpha.py` | Closed form of the limiting cycle probability p(alpha) (Theorems 11-12, Table 1), with a check by direct stable sampling |
| `code/proof_check.py` | Numerical check of the representation p(alpha) = 1/4 - 3/4 P(\|L + (alpha/2) V\| < log 2) |
| `code/step2_check.py` | Dictatorship and cycle probabilities for independent Pareto weights |
| `code/figdata.py` | Data for the dictatorship and cycle figures |
| `code/manip.py` | Ranked pairs and the probability of a manipulable profile for k agents (Table 2) |
| `code/mu_exact.py` | Exact manipulation probability for k = 10^3, 10^4, 10^5 using the localisation lemma |
| `code/q_certified.py`, `code/q_lepage_bounds.py` | Bounds for the limit q(alpha) from the LePage series, with sensitivity to series length and remainder (Figure 3) |
| `code/verify_example.py` | The five-agent manipulation used in the positivity proof of Theorem 15(iii) |

## Network weights

| Script | Result |
|---|---|
| `code/green_degree.py` | Influence weights against degree on a configuration model (Lemma 8, Proposition 9, Figure 4) |
| `code/limits.py` | Dictatorship on configuration models, jointly clamped versus independent weights (Table 3) |
| `code/scalefree.py` | Dictatorship and cycles on Erdos-Renyi and Chung-Lu graphs (Table 4) |

## Exploratory runs outside the proved regime

| Script | Result |
|---|---|
| `code/small_graphs.py` | All connected graphs on 4-6 vertices, nonlinear rule at full committed strength and several gains |
| `code/larger_graphs.py`, `code/smallq.py` | Nonlinear rule versus weighted majority at finite and small committed strength |

`code/plotfigs.py` draws the figures from the saved outputs.

Requirements: see `requirements.txt` (Python 3.10 or later).
