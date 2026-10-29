"""Like-for-like causal discrepancy of each model: DML with outcome Y_obs - Y_sim.
Because the partially linear model is linear in Y, theta(Y_obs - Y_sim) = theta_obs - theta_M^DML: the same estimator
is applied to observations and simulations, so estimator bias (e.g. shrinkage of catchment effects) cancels.
Also: concentration error attributable to the causal discrepancy in the hot-dry scenario."""
import json
import numpy as np, pandas as pd
from common import *
import dml

R = json.load(open('out/results.json'))
S = np.load('out/sims_base.npz')
sig = hydro_signatures()
MODELS = list(R['causal'].keys())
MODS = R['moderators']
Zc = sig[MODS].values; Zc = (Zc - Zc.mean(0)) / Zc.std(0)
units = dml.build_units()
rng = np.random.default_rng(11)

def theta_c(b):
    thP = b[0] + sum(b[2 + 2 * j] * Zc[:, j] for j in range(len(MODS)))
    thT = b[1] + sum(b[3 + 2 * j] * Zc[:, j] for j in range(len(MODS)))
    return thP, thT

def amp(d, dr0=10.0):
    return (dr0 + 1) / (dr0 * np.exp(np.asarray(d)) + 1)

out = {}
for m in MODELS:
    sim = S[m.replace('-', '_')]
    us = dml.build_units(Qsim=sim).merge(units[['c', 'year', 'month']], on=['c', 'year', 'month'])
    uo = units.merge(us[['c', 'year', 'month', 'Y']].rename(columns={'Y': 'Ysim'}), on=['c', 'year', 'month'])
    uo['Y'] = uo['Y'] - uo['Ysim']           # discrepancy outcome
    res = dml.residualise(uo, reps=5)
    b, se = dml.plm(res, uo)
    bh, _ = dml.plm(res, uo, Z=Zc[uo.c.values])
    dP, dT = theta_c(bh)
    # year-cluster bootstrap of the projection
    yrs = np.unique(uo.year.values); idx = {y: np.where(uo.year.values == y)[0] for y in yrs}
    r0 = res[0]
    Dm = np.column_stack([r0['DP'], r0['DT']] + sum([[r0['DP'] * z, r0['DT'] * z] for z in Zc[uo.c.values].T], []))
    bs = []
    for _ in range(1000):
        ii = np.concatenate([idx[y] for y in rng.choice(yrs, len(yrs))])
        bs.append(np.linalg.lstsq(Dm[ii], r0['Y'][ii], rcond=None)[0])
    bs = np.array(bs)
    dPb = np.array([theta_c(x)[0] for x in bs]); dTb = np.array([theta_c(x)[1] for x in bs])
    ciP = np.percentile(dPb, [2.5, 97.5], 0); ciT = np.percentile(dTb, [2.5, 97.5], 0)
    # scenario: model's true September response vs response corrected by the measured discrepancy
    scen_m = np.array(R['causal'][m]['scen'])
    corr = dP * np.log(0.8) + dT * 2.0           # theta_obs - theta_M applied to the scenario anomalies
    corr_b = dPb * np.log(0.8) + dTb * 2.0
    a_m = amp(scen_m); a_obs = amp(scen_m + corr)
    a_obs_b = amp(scen_m[None, :] + corr_b)
    out[m] = dict(delta_P=float(b[0]), delta_P_se=float(se[0]), delta_T=float(b[1]), delta_T_se=float(se[1]),
                  delta_P_c=dP.tolist(), delta_T_c=dT.tolist(), delta_P_c_ci=ciP.tolist(), delta_T_c_ci=ciT.tolist(),
                  n_sig_P_pos=int((ciP[0] > 0).sum()), n_sig_P_neg=int((ciP[1] < 0).sum()),
                  n_sig_T_pos=int((ciT[0] > 0).sum()), n_sig_T_neg=int((ciT[1] < 0).sum()),
                  mean_abs_delta_P=float(np.mean(np.abs(dP))), mean_abs_delta_T=float(np.mean(np.abs(dT))),
                  amp_model=a_m.tolist(), amp_obsimplied=a_obs.tolist(),
                  amp_obsimplied_ci=np.percentile(a_obs_b, [2.5, 97.5], 0).tolist(),
                  amp_err_pct=(100 * (a_m - a_obs) / a_obs).tolist(),
                  inc_err_pct=(100 * ((a_m - 1) - (a_obs - 1)) / np.maximum(a_obs - 1, 1e-6)).tolist())
    print(m, 'deltaP %.3f±%.3f deltaT %.4f±%.4f' % (b[0], se[0], b[1], se[1]),
          'sigP+', out[m]['n_sig_P_pos'], 'sigP-', out[m]['n_sig_P_neg'], 'sigT+', out[m]['n_sig_T_pos'], 'sigT-', out[m]['n_sig_T_neg'],
          'mean|dP| %.3f' % out[m]['mean_abs_delta_P'], 'median ampErr %.1f%%' % np.median(out[m]['amp_err_pct']), flush=True)
json.dump(out, open('out/discrepancy.json', 'w'), indent=1)
