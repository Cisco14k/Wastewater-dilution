"""Main analysis: predictive skill, observational causal benchmark (DML), synthetic validation,
model interventional effects, causal error and the dilution scenario. Writes out/results.json + CSVs."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from common import *
from hydro import nse, kge, lognse, pbias
import dml
from interventions import Sim, interventional_effects, scenario

MODELS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['GR4J', 'GR4J-CN', 'GBM-A', 'GBM-C', 'LSTM', 'LSTM-CR']
rng = np.random.default_rng(7)
sim = Sim()
att = hydro_signatures()

# ------------------------------------------------------------------ base simulations
def clim_sim():
    doy = DATES.dayofyear.values
    out = np.empty((NC, NT))
    for d in range(1, 367):
        win = (np.abs(((doy - d + 183) % 365) - 183) <= 7)
        v = np.nanmean(np.where(TRAIN & win, Q, np.nan), 1)
        out[:, doy == d] = v[:, None]
    return out

SIMS = {'CLIM': clim_sim()}
for m in MODELS:
    SIMS[m] = sim.run(m, P, T, E)
np.savez_compressed('out/sims_base.npz', **{k.replace('-', '_'): v for k, v in SIMS.items()})

# ------------------------------------------------------------------ predictive skill (test block, 2015-2018)
q_tr = np.where(TRAIN, Q, np.nan)
Q95_TR = np.nanpercentile(q_tr, 5, axis=1)        # flow exceeded 95% of the time (training)
rows = []
for name, S in SIMS.items():
    for i in range(NC):
        o = np.where(TEST, Q[i], np.nan); s = np.where(TEST, S[i], np.nan)
        m = np.isfinite(o)
        k, r, a, b = kge(o, s)
        q95o = np.nanpercentile(o[m], 5); q95s = np.nanpercentile(s[m], 5)
        defo = np.sum(o[m] < Q95_TR[i]); defs = np.sum(s[m] < Q95_TR[i])
        # late-summer (Aug-Oct) mean flow error: the dilution-critical period
        lsm = m & np.isin(DATES.month.values, (8, 9, 10))
        rows.append(dict(model=name, id=IDS[i], NSE=nse(o, s), KGE=k, r=r, alpha=a, beta=b,
                         logNSE=lognse(o, s, eps=np.nanmean(q_tr[i]) / 100), PBIAS=pbias(o, s),
                         Q95_bias=100 * (q95s - q95o) / q95o, deficit_err=defs - defo, deficit_obs=defo,
                         lowseason_bias=100 * (np.nanmean(s[lsm]) - np.nanmean(o[lsm])) / np.nanmean(o[lsm])))
skill = pd.DataFrame(rows)
skill.to_csv('out/skill_test.csv', index=False)
# validation-block skill too (for the supplement)
rows = []
for name, S in SIMS.items():
    for blk, msk in (('train', TRAIN), ('valid', VALID)):
        for i in range(NC):
            o = np.where(msk, Q[i], np.nan); s = np.where(msk, S[i], np.nan)
            rows.append(dict(model=name, block=blk, id=IDS[i], NSE=nse(o, s), KGE=kge(o, s)[0],
                             logNSE=lognse(o, s, eps=np.nanmean(q_tr[i]) / 100)))
pd.DataFrame(rows).to_csv('out/skill_trainvalid.csv', index=False)

# ------------------------------------------------------------------ observational causal benchmark
units = dml.build_units()
RES_OBS = dml.residualise(units, reps=5)
b_obs, se_obs = dml.plm(RES_OBS, units)
naive = np.linalg.lstsq(np.column_stack([np.ones(len(units)), units.DP, units.DT]), units.Y, rcond=None)[0]
# heterogeneity: best linear projection on standardised moderators
MODS = ['snow_frac', 'bfi', 'aridity']
Zc = att[MODS].values; Zc = (Zc - Zc.mean(0)) / Zc.std(0)
Zu = Zc[units.c.values]
b_het, se_het = dml.plm(RES_OBS, units, Z=Zu)

def theta_c(b):
    """catchment-specific effects implied by BLP coefficients b = [P0, T0, P_z1, T_z1, ...]."""
    thP = b[0] + sum(b[2 + 2 * j] * Zc[:, j] for j in range(len(MODS)))
    thT = b[1] + sum(b[3 + 2 * j] * Zc[:, j] for j in range(len(MODS)))
    return thP, thT

def boot_blp(res_list, df, Z, B=1000):
    """year-cluster bootstrap of the final stage (residuals held fixed)."""
    yrs = np.unique(df.year.values)
    idx_by_year = {y: np.where(df.year.values == y)[0] for y in yrs}
    res = res_list[0]
    D = [res['DP'], res['DT']]
    for z in Z.T:
        D += [res['DP'] * z, res['DT'] * z]
    Dm = np.column_stack(D)
    out = []
    for _ in range(B):
        ys = rng.choice(yrs, len(yrs))
        ii = np.concatenate([idx_by_year[y] for y in ys])
        out.append(np.linalg.lstsq(Dm[ii], res['Y'][ii], rcond=None)[0])
    return np.array(out)

thP_obs, thT_obs = theta_c(b_het)
BB = boot_blp(RES_OBS, units, Zu)
thP_bs = np.array([theta_c(b)[0] for b in BB]); thT_bs = np.array([theta_c(b)[1] for b in BB])
gate_obs = dml.per_catchment(RES_OBS, units)          # noisier catchment-specific estimates (supplement)

# ------------------------------------------------------------------ model worlds: interventions + DML on simulations
causal = {}
unit_eff = units[['c', 'id', 'year', 'month']].copy()
for name in MODELS:
    eP, eT, base = interventional_effects(sim, name, units)
    unit_eff[f'{name}_P'] = eP; unit_eff[f'{name}_T'] = eT
    us = dml.build_units(Qsim=base)
    # align: same unit set as observations
    us = us.merge(units[['c', 'year', 'month']], on=['c', 'year', 'month'])
    rs = dml.residualise(us, reps=3)
    b_m, se_m = dml.plm(rs, us)
    bh_m, _ = dml.plm(rs, us, Z=Zc[us.c.values])
    tP_m, tT_m = theta_c(bh_m)
    # true model effects: per catchment means and the PLM-weighted pooled average
    aP_c = np.array([np.nanmean(eP[units.c.values == i]) for i in range(NC)])
    aT_c = np.array([np.nanmean(eT[units.c.values == i]) for i in range(NC)])
    wP = RES_OBS[0]['DP'] ** 2; wT = RES_OBS[0]['DT'] ** 2
    causal[name] = dict(
        ace_P=float(np.nanmean(eP)), ace_T=float(np.nanmean(eT)),
        ace_P_w=float(np.nansum(wP * eP) / np.sum(wP[np.isfinite(eP)])),
        ace_T_w=float(np.nansum(wT * eT) / np.sum(wT[np.isfinite(eT)])),
        ace_P_c=aP_c.tolist(), ace_T_c=aT_c.tolist(),
        dml_sim_P=float(b_m[0]), dml_sim_P_se=float(se_m[0]), dml_sim_T=float(b_m[1]), dml_sim_T_se=float(se_m[1]),
        dml_sim_P_c=tP_m.tolist(), dml_sim_T_c=tT_m.tolist(),
        frac_wrong_sign_P=float(np.mean(eP < 0)),
        causal_err_P=float(np.mean(np.abs(aP_c - thP_obs))), causal_err_T=float(np.mean(np.abs(aT_c - thT_obs))),
        scen=scenario(sim, name).tolist())
    print(name, {k: (round(v, 3) if isinstance(v, float) else '') for k, v in causal[name].items() if not isinstance(v, list)}, flush=True)
unit_eff.to_csv('out/unit_effects.csv', index=False)

# observational scenario: hot-dry Jul-Sep window -> September flow, from catchment effects
dPs, dTs = np.log(0.8), 2.0
scen_obs = thP_obs * dPs + thT_obs * dTs
scen_obs_bs = thP_bs * dPs + thT_bs * dTs
out = dict(n_units=int(len(units)), theta_obs=b_obs.tolist(), theta_obs_se=se_obs.tolist(), naive_ols=naive[1:].tolist(),
           blp=b_het.tolist(), blp_se=se_het.tolist(), moderators=MODS,
           thP_obs_c=thP_obs.tolist(), thT_obs_c=thT_obs.tolist(),
           thP_obs_c_ci=np.percentile(thP_bs, [2.5, 97.5], 0).tolist(), thT_obs_c_ci=np.percentile(thT_bs, [2.5, 97.5], 0).tolist(),
           gate_obs=gate_obs.tolist(), scen_obs=scen_obs.tolist(), scen_obs_ci=np.percentile(scen_obs_bs, [2.5, 97.5], 0).tolist(),
           causal=causal, ids=IDS)
json.dump(out, open('out/results.json', 'w'), indent=1)
print('theta_obs', b_obs, se_obs, 'naive', naive)
print('BLP', np.round(b_het, 3), np.round(se_het, 3))
