"""Calibrate GR4J and GR4J-CemaNeige per catchment on the training block (2000-2011) by differential evolution.
Objective: L = 1 - (0.5 NSE + 0.5 NSE_log), the same balanced criterion used by the learned models."""
import sys, json, time
import numpy as np
from scipy.optimize import differential_evolution
from common import *
from hydro import gr4j, cemaneige

which = sys.argv[1]          # 'gr4j' or 'gr4jcn'
idx = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(NC)
B_GR = [(10, 3000), (-10, 10), (10, 1000), (0.5, 10)]
B_CN = [(0.0, 1.0), (0.0, 10.0)]

def make_obj(i):
    q = Q[i]; m = TRAIN & np.isfinite(q)
    qo = q[m]; eps = qo.mean() / 100
    lo = np.log(qo + eps)
    den1 = np.sum((qo - qo.mean()) ** 2); den2 = np.sum((lo - lo.mean()) ** 2)
    def obj(X):                      # X: (D, S) vectorised
        X = X.T if X.ndim == 2 else X[None, :]
        if which == 'gr4j':
            qs = gr4j(P[i], E[i], X[:, :4])
        else:
            liq = cemaneige(P[i], T[i], H[i], X[:, 4:6], train_mask=TRAIN)
            qs = gr4j(liq, np.broadcast_to(E[i], liq.shape), X[:, :4])
        qs = qs[:, m]
        n1 = 1 - np.sum((qs - qo) ** 2, 1) / den1
        n2 = 1 - np.sum((np.log(qs + eps) - lo) ** 2, 1) / den2
        L = 1 - (0.5 * n1 + 0.5 * n2)
        return L if L.size > 1 else float(L[0])
    return obj

res = {}
for i in idx:
    t0 = time.time()
    bounds = B_GR + (B_CN if which == 'gr4jcn' else [])
    r = differential_evolution(make_obj(i), bounds, popsize=15, maxiter=120, tol=1e-6, seed=42 + i,
                               init='sobol', vectorized=True, updating='deferred', polish=False,
                               mutation=(0.4, 1.0), recombination=0.85)
    res[IDS[i]] = dict(x=r.x.tolist(), L=float(r.fun), nfev=int(r.nfev))
    print(which, IDS[i], np.round(r.x, 3), round(r.fun, 4), f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open(f'out/params_{which}_{idx[0]}.json', 'w'), indent=1)
