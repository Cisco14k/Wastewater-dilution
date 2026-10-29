"""Shared configuration: data loading, chronological split, attributes."""
import numpy as np, pandas as pd

D = np.load('data/camels_fr19.npz', allow_pickle=True)
P, T, E, Q, H = D['P'], D['T'], D['E'], D['Q'], D['H']
DATES = pd.to_datetime(D['dates'])
IDS = list(D['ids'])
LAT = D['lat']
NC, NT = P.shape

YEAR = DATES.year.values
WARM = YEAR == 1999                      # forcing-only warm-up, never scored
TRAIN = (YEAR >= 2000) & (YEAR <= 2011)  # 12 years
VALID = (YEAR >= 2012) & (YEAR <= 2014)  # 3 years: early stopping / hyper-parameters
TEST = (YEAR >= 2015) & (YEAR <= 2018)   # 4 years: scored once
CAL = (YEAR >= 2000) & (YEAR <= 2014)    # conceptual models calibrate on train only; flag kept for clarity

def static_attributes():
    """Climate and physiographic attributes computed from forcing only, training years only (no discharge)."""
    att = pd.read_csv('data/attributes_raw.csv')
    m = TRAIN
    att['p_mean'] = P[:, m].mean(1)
    att['pet_mean'] = E[:, m].mean(1)
    att['aridity'] = att.pet_mean / att.p_mean
    att['t_mean'] = T[:, m].mean(1)
    snow = []
    from hydro import band_forcing
    for i in range(NC):
        Pb, Tb = band_forcing(P[i], T[i], H[i])
        fs = np.clip((3.0 - Tb) / 4.0, 0, 1)
        snow.append((Pb * fs)[:, m].sum() / Pb[:, m].sum())
    att['snow_frac'] = snow
    # precipitation seasonality: amplitude of first harmonic of monthly P climatology / mean
    mon = DATES.month.values
    seas = []
    for i in range(NC):
        mp = np.array([P[i, m & (mon == k)].mean() for k in range(1, 13)])
        seas.append((mp.max() - mp.min()) / mp.mean())
    att['p_seasonality'] = seas
    att['log_area'] = np.log10(att.area)
    att['elev_range'] = att.elev_max - att.elev_min
    return att

def hydro_signatures():
    """Discharge-based signatures over training years (descriptive only; never model inputs)."""
    from scipy.signal import lfilter
    att = static_attributes()
    bfi, rr, q95 = [], [], []
    for i in range(NC):
        q = Q[i].copy(); m = TRAIN & np.isfinite(q)
        qi = pd.Series(q).interpolate(limit_direction='both').values
        # Lyne-Hollick, alpha 0.925, 3 passes (Ladson et al., 2013)
        b = qi.copy()
        for k in range(3):
            x = b if k % 2 == 0 else b[::-1]
            qf = np.zeros_like(x)
            for t in range(1, len(x)):
                qf[t] = 0.925 * qf[t-1] + (1 + 0.925) / 2 * (x[t] - x[t-1])
                qf[t] = min(max(qf[t], 0), x[t])
            bb = x - qf
            b = bb if k % 2 == 0 else bb[::-1]
        bfi.append(np.nansum(b[m]) / np.nansum(qi[m]))
        rr.append(np.nanmean(q[m]) / P[i, m].mean())
        q95.append(np.nanpercentile(q[m], 5))
    att['bfi'] = bfi; att['runoff_ratio'] = rr; att['q95_train'] = q95
    return att

STATIC_COLS = ['p_mean', 'pet_mean', 'aridity', 't_mean', 'snow_frac', 'p_seasonality',
               'log_area', 'elev_med', 'elev_range', 'lat', 'lon']
