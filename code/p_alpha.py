# Closed form for the limiting cycle probability under heavy-tailed weights (index alpha in (0,2)):
#   p(alpha) = 1 - (3/2) P(|A|/|B| < 2^{1/alpha}),  A,B iid standard SaS
# Sub-Gaussian representation + Lamperti law (a = alpha/2):
#   p(alpha) = 1 - (3/pi) * (sin(pi a)/(pi a)) * int_0^inf arctan((2/t)^{1/alpha}) / (t^2 + 2 t cos(pi a) + 1) dt
import numpy as np
from scipy.integrate import quad
from scipy.stats import levy_stable
pG = 0.25 - 1.5/np.pi*np.arcsin(1/3)
def p(alpha):
    a = alpha/2
    f = lambda t: np.arctan((2/t)**(1/alpha)) / (t*t + 2*t*np.cos(np.pi*a) + 1)
    I = quad(f, 0, 1, limit=400)[0] + quad(f, 1, np.inf, limit=400)[0]
    return 1 - 3/np.pi * np.sin(np.pi*a)/(np.pi*a) * I
if __name__ == "__main__":
    print("Guilbaud p_G =", round(pG, 6))
    for al in (0.05, 0.3, 0.6, 1.0, 1.2, 1.5, 1.8, 1.95, 1.999):
        print(f"alpha={al:6.3f}  p(alpha)={p(al):.6f}")
    # independent check by direct stable sampling
    rng = np.random.default_rng(5)
    for al in (1.2, 1.5, 1.8):
        A = levy_stable.rvs(al, 0, size=400000, random_state=rng); B = levy_stable.rvs(al, 0, size=400000, random_state=rng)
        print(f"MC alpha={al}: {1-1.5*np.mean(np.abs(A)/np.abs(B) < 2**(1/al)):.4f}")
    # monotonicity on a fine grid
    g = np.linspace(0.02, 1.998, 400); v = np.array([p(x) for x in g])
    print("monotone increasing on grid:", bool(np.all(np.diff(v) > 0)), " max p on grid:", v.max(), "< pG:", bool(v.max() < pG))
