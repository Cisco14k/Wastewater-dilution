"""Double/debiased machine learning (Chernozhukov et al., 2018) estimate of the causal effect of seasonal
precipitation and temperature anomalies on low-flow-season monthly discharge.

Unit: catchment c x year y x target month m (June-October), 2000-2018.
Outcome Y: log mean discharge in month m.
Treatments (window W = months m-2..m): D_P = log(P_W / climatological P_W); D_T = T_W - climatological T_W [degC].
Confounders X: calendar month, catchment identity and static attributes, antecedent P and T anomalies
(months m-5..m-3 and m-11..m-6), and log discharge anomaly in month m-3 (pre-treatment storage proxy).
Partially linear model with cross-fitting; folds grouped by year (all catchments of one year share a fold),
because a given year's weather is shared across catchments. Standard errors are cluster-robust by year."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *

MONTHS = (6, 7, 8, 9, 10)
YRS = np.arange(2000, 2019)
_mon = DATES.month.values; _yr = DATES.year.values
# month index for each day: months since Jan 1999
MIDX = (_yr - 1999) * 12 + (_mon - 1)
NM = MIDX.max() + 1

def monthly(x, how='mean'):
    """x (NC, NT) -> (NC, NM) monthly mean (nan if < 25 valid days for discharge)."""
    out = np.full((x.shape[0], NM), np.nan)
    for k in range(NM):
        d = MIDX == k
        v = x[:, d]
        cnt = np.isfinite(v).sum(1)
        out[:, k] = np.where(cnt >= min(25, d.sum()), np.nanmean(v, 1), np.nan)
    return out

def window(xm, k, a, b, agg='sum'):
    """aggregate monthly series over months k-a..k-b (inclusive)."""
    s = xm[:, k - a:k - b + 1]
    return s.sum(1) if agg == 'sum' else s.mean(1)

_att = static_attributes()

def build_units(Qsim=None, Pm=None, Tm=None):
    """Assemble the unit-level table. Qsim lets the same design run on simulated discharge."""
    Pm = monthly(P) if Pm is None else Pm
    Tm = monthly(T) if Tm is None else Tm
    Qm = monthly(Q if Qsim is None else np.where(np.isfinite(Q), Qsim, np.nan))
    rows = []
    for y in YRS:
        for m in MONTHS:
            k = (y - 1999) * 12 + (m - 1)
            if k - 11 < 0:
                continue
            rows.append((y, m, k))
    recs = []
    # climatologies over 2000-2018 of the window sums, per catchment and target month
    clim = {}
    for m in MONTHS:
        ks = [(y - 1999) * 12 + (m - 1) for y in YRS]
        clim[m] = dict(
            PW=np.mean([window(Pm, k, 2, 0) for k in ks], 0), TW=np.mean([window(Tm, k, 2, 0, 'mean') for k in ks], 0),
            PA1=np.mean([window(Pm, k, 5, 3) for k in ks], 0), TA1=np.mean([window(Tm, k, 5, 3, 'mean') for k in ks], 0),
            PA2=np.mean([window(Pm, k, 11, 6) for k in ks], 0), TA2=np.mean([window(Tm, k, 11, 6, 'mean') for k in ks], 0),
            Q3=np.nanmean([np.log(Qm[:, k - 3]) for k in ks], 0), QY=np.nanmean([np.log(Qm[:, k]) for k in ks], 0))
    for y, m, k in rows:
        c = clim[m]
        dP = np.log(window(Pm, k, 2, 0) / c['PW'])
        dT = window(Tm, k, 2, 0, 'mean') - c['TW']
        a1 = np.log(window(Pm, k, 5, 3) / c['PA1']); t1 = window(Tm, k, 5, 3, 'mean') - c['TA1']
        a2 = np.log(window(Pm, k, 11, 6) / c['PA2']); t2 = window(Tm, k, 11, 6, 'mean') - c['TA2']
        q3 = np.log(Qm[:, k - 3]) - c['Q3']
        yv = np.log(Qm[:, k]) - c['QY']
        for i in range(NC):
            recs.append(dict(c=i, id=IDS[i], year=y, month=m, k=k, Y=yv[i], DP=dP[i], DT=dT[i],
                             PA1=a1[i], TA1=t1[i], PA2=a2[i], TA2=t2[i], Q3=q3[i]))
    df = pd.DataFrame(recs)
    for col in STATIC_COLS:
        df[col] = _att[col].values[df.c.values]
    return df.dropna(subset=['Y', 'Q3']).reset_index(drop=True)

XCOLS = ['month', 'PA1', 'TA1', 'PA2', 'TA2', 'Q3'] + STATIC_COLS

def _learner(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=seed)

def residualise(df, reps=5, nfold=5, seed=0, learner=None, xcols=None):
    """Cross-fitted residuals of Y, DP, DT on X (+ catchment one-hot). Returns list over repetitions."""
    learner = learner or _learner
    X = np.column_stack([df[xcols or XCOLS].values, np.eye(NC)[df.c.values]])
    years = df.year.values
    out = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        yperm = rng.permutation(YRS)
        fold_of = {y: i % nfold for i, y in enumerate(yperm)}
        f = np.array([fold_of[y] for y in years])
        res = {}
        for col in ('Y', 'DP', 'DT'):
            v = df[col].values; e = np.empty_like(v)
            for k in range(nfold):
                tr, te = f != k, f == k
                mdl = learner(seed + r).fit(X[tr], v[tr])
                e[te] = v[te] - mdl.predict(X[te])
            res[col] = e
        out.append(res)
    return out

def plm(res_list, df, Z=None):
    """Final-stage OLS of Y~ on (DP~, DT~) [optionally interacted with centred moderators Z].
    Returns median-aggregated coefficients and cluster(year)-robust SEs."""
    ests, ses = [], []
    for res in res_list:
        D = [res['DP'], res['DT']]
        if Z is not None:
            for z in Z.T:
                D += [res['DP'] * z, res['DT'] * z]
        Dm = np.column_stack(D)
        y = res['Y']
        XtX = Dm.T @ Dm
        b = np.linalg.solve(XtX, Dm.T @ y)
        u = y - Dm @ b
        meat = np.zeros_like(XtX)
        for yy in np.unique(df.year.values):
            s = df.year.values == yy
            g = Dm[s].T @ u[s]
            meat += np.outer(g, g)
        G = len(np.unique(df.year.values))
        V = np.linalg.solve(XtX, np.linalg.solve(XtX, meat).T) * G / (G - 1)
        ests.append(b); ses.append(np.sqrt(np.diag(V)))
    ests, ses = np.array(ests), np.array(ses)
    b = np.median(ests, 0)
    # Chernozhukov et al. (2018) median aggregation of variance across splits
    se = np.sqrt(np.median(ses ** 2 + (ests - b) ** 2, 0))
    return b, se

def per_catchment(res_list, df):
    """Catchment-specific effects from pooled residuals (group-average treatment effects)."""
    out = []
    for i in range(NC):
        s = df.c.values == i
        bs = []
        for res in res_list:
            Dm = np.column_stack([res['DP'][s], res['DT'][s]])
            bs.append(np.linalg.lstsq(Dm, res['Y'][s], rcond=None)[0])
        out.append(np.median(bs, 0))
    return np.array(out)

if __name__ == '__main__':
    df = build_units()
    print(df.shape, df[['Y', 'DP', 'DT', 'Q3']].describe().round(3))
    res = residualise(df)
    b, se = plm(res, df)
    print('theta_P (dlogQ/dlogP) =', b[0].round(3), '+-', se[0].round(3))
    print('theta_T (dlogQ/degC)  =', b[1].round(3), '+-', se[1].round(3))
    # naive OLS for contrast
    Dm = np.column_stack([np.ones(len(df)), df.DP, df.DT]); print('naive OLS', np.linalg.lstsq(Dm, df.Y, rcond=None)[0].round(3))
