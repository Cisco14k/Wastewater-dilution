"""Regional gradient-boosted trees (sklearn HistGradientBoosting; stands in for XGBoost).
GBM-A ('associational'): antecedent-forcing features + calendar harmonics, unconstrained.
GBM-C ('causally constrained'): only causal parents of discharge in the hydrological DAG (precipitation,
temperature, evaporative demand and their antecedent aggregates), no calendar proxies, with monotonicity
constraints dQ/dP >= 0 and dQ/dPET <= 0 on all precipitation and evaporative-demand aggregates."""
import sys, json, itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *

att = static_attributes()
S = att[STATIC_COLS].values.astype(float)
DOY = DATES.dayofyear.values

def rsum(x, w):
    c = np.cumsum(np.concatenate([np.zeros(x.shape[:-1] + (1,)), x], -1), -1)
    out = c[..., w:] - c[..., :-w]
    return np.concatenate([np.full(x.shape[:-1] + (w - 1,), np.nan), out], -1)

def features(Pa, Ta, Ea, calendar):
    """Pa, Ta, Ea: (NC, NT). Returns X (NC, NT, F), names, monotone signs."""
    f, nm, mono = [], [], []
    def add(x, n, s):
        f.append(x); nm.append(n); mono.append(s)
    add(Pa, 'P', 1)
    for l in (1, 2, 3):
        add(np.concatenate([np.full((NC, l), np.nan), Pa[:, :-l]], 1), f'P_lag{l}', 1)
    for w in (3, 7, 15, 30, 60, 90, 180, 365):
        add(rsum(Pa, w), f'P_sum{w}', 1)
    for w in (7, 30, 90, 180, 365):
        add(rsum(Ea, w), f'PET_sum{w}', -1)
    add(Ta, 'T', 0)
    for w in (7, 30, 90):
        add(rsum(Ta, w) / w, f'T_mean{w}', 0)
    sol = Pa * np.clip((3.0 - Ta) / 4.0, 0, 1)                    # solid-precipitation proxy
    for w in (30, 90, 180):
        add(rsum(sol, w), f'Psolid_sum{w}', 0)
    add(rsum(np.maximum(Ta, 0), 30), 'PDD_30', 0)
    if calendar:
        add(np.broadcast_to(np.sin(2 * np.pi * DOY / 365.25), (NC, NT)), 'sin_doy', 0)
        add(np.broadcast_to(np.cos(2 * np.pi * DOY / 365.25), (NC, NT)), 'cos_doy', 0)
    for j, c in enumerate(STATIC_COLS):
        add(np.broadcast_to(S[:, j:j + 1], (NC, NT)), c, 0)
    return np.stack(f, -1), nm, mono

EPS = np.nanmean(np.where(TRAIN, Q, np.nan), 1) / 100
Y = np.log(Q + EPS[:, None])

def fit(variant, Xf, mono, params):
    m_tr = (TRAIN[None, :] & np.isfinite(Q)) & np.all(np.isfinite(Xf), -1)
    m_va = (VALID[None, :] & np.isfinite(Q))
    Xtr, ytr = Xf[m_tr], Y[m_tr]
    Xva, yva = Xf[m_va], Y[m_va]
    cst = mono if variant == 'C' else None
    mdl = HistGradientBoostingRegressor(loss='squared_error', max_iter=1500, early_stopping=True,
                                        validation_fraction=None, n_iter_no_change=50, scoring='loss',
                                        monotonic_cst=cst, random_state=0, **params)
    # sklearn uses validation_fraction; to use our chronological validation block we pass it via X_val
    mdl.fit(Xtr, ytr, X_val=Xva, y_val=yva)
    return mdl

def predict(mdl, Xf):
    yh = mdl.predict(Xf.reshape(-1, Xf.shape[-1])).reshape(Xf.shape[:2])
    return np.exp(yh) - EPS[:, None]

def balanced(sim, mask):
    v = []
    for i in range(NC):
        m = mask & np.isfinite(Q[i]); o = Q[i, m]; s = np.maximum(sim[i, m], 0)
        n1 = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
        lo = np.log(o + EPS[i]); ls = np.log(s + EPS[i])
        n2 = 1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
        v.append(0.5 * n1 + 0.5 * n2)
    return np.median(v)

if __name__ == '__main__':
    grid = [dict(learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=ms, l2_regularization=1.0)
            for lr, ln, ms in itertools.product([0.03, 0.08], [15, 63], [50, 200])]
    report = {}
    for variant in ('A', 'C'):
        Xf, nm, mono = features(P, T, E, calendar=(variant == 'A'))
        best = None
        for g in grid:
            mdl = fit(variant, Xf, mono, g)
            sim = predict(mdl, Xf)
            v = balanced(sim, VALID)
            print(variant, g, mdl.n_iter_, round(v, 4), flush=True)
            if best is None or v > best[0]:
                best = (v, g, mdl, sim)
        v, g, mdl, sim = best
        np.save(f'out/sim_gbm{variant}.npy', sim)
        import pickle; pickle.dump(mdl, open(f'out/gbm{variant}.pkl', 'wb'))
        report[variant] = dict(params=g, n_iter=int(mdl.n_iter_), valid_balanced=float(v), features=nm)
        print('BEST', variant, g, v)
    json.dump(report, open('out/gbm_report.json', 'w'), indent=1)
