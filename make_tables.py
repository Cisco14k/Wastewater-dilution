"""Build tables (main + supplement) and the numbers dictionary used to fill manuscript placeholders."""
import json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from common import *

R = json.load(open('out/results.json'))
RB = json.load(open('out/robustness.json'))
sk = pd.read_csv('out/skill_test.csv')
tv = pd.read_csv('out/skill_trainvalid.csv')
sig = pd.read_csv('out/signatures.csv')
gb = json.load(open('out/gbm_report.json'))
MODELS = list(R['causal'].keys()); ALL = ['CLIM'] + MODELS
N = {}
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'

def amp(d, dr0=10.0):
    return (dr0 + 1) / (dr0 * np.exp(np.asarray(d)) + 1)

# ---------------- catchment numbers
N['AREA_MIN'] = f'{sig.area.min():.0f}'; N['AREA_MAX'] = f'{sig.area.max():,.0f}'; N['AREA_MED'] = f'{sig.area.median():.0f}'
N['ELEV_MIN'] = f'{sig.elev_med.min():.0f}'; N['ELEV_MAX'] = f'{sig.elev_med.max():,.0f}'
N['ARID_MIN'] = f2(sig.aridity.min()); N['ARID_MAX'] = f2(sig.aridity.max())
N['RR_MIN'] = f2(sig.runoff_ratio.min()); N['RR_MAX'] = f2(sig.runoff_ratio.max())
N['BFI_MIN'] = f2(sig.bfi.min()); N['BFI_MAX'] = f2(sig.bfi.max())
N['SNOW_MAX'] = f'{100 * sig.snow_frac.max():.0f}'
qc = 100 * np.mean(np.isfinite(Q), 1)
N['QCOMP_MIN'] = f'{qc.min():.1f}'; N['QCOMP_MEAN'] = f'{qc.mean():.1f}'

# ---------------- skill
med = sk.groupby('model').median(numeric_only=True)
for m in ALL:
    k = m.replace('-', '')
    for col in ('NSE', 'logNSE', 'KGE', 'Q95_bias', 'lowseason_bias', 'PBIAS'):
        N[f'{col}_{k}'] = f2(med.loc[m, col]) if 'bias' not in col.lower() else f'{med.loc[m, col]:+.0f}'
    N[f'N07_{k}'] = str(int((sk[sk.model == m].NSE > 0.7).sum()))
def paired(a, b, col='NSE'):
    x = sk[sk.model == a].set_index('id').loc[IDS, col].values; y = sk[sk.model == b].set_index('id').loc[IDS, col].values
    d = x - y
    return np.median(d), int((d > 0).sum()), wilcoxon(d).pvalue
pairs = {}
for a, b in [('LSTM', 'GR4J-CN'), ('GBM-A', 'GR4J-CN'), ('GR4J-CN', 'GR4J'), ('GBM-A', 'GBM-C'), ('LSTM', 'LSTM-CR'), ('LSTM', 'GBM-A')]:
    md, nb, p = paired(a, b)
    key = f'{a}_{b}'.replace('-', '')
    N[f'DN_{key}'] = f'{md:+.3f}'; N[f'NB_{key}'] = str(nb); N[f'P_{key}'] = f'{p:.3f}' if p >= 0.001 else '<0.001'
    pairs[(a, b)] = (md, nb, p)

# ---------------- causal benchmark
th, se = R['theta_obs'], R['theta_obs_se']
N['THP'] = f2(th[0]); N['THP_SE'] = f2(se[0]); N['THT'] = f3(th[1]); N['THT_SE'] = f3(se[1])
N['THP_LO'] = f2(th[0] - 1.96 * se[0]); N['THP_HI'] = f2(th[0] + 1.96 * se[0])
N['THT_LO'] = f3(th[1] - 1.96 * se[1]); N['THT_HI'] = f3(th[1] + 1.96 * se[1])
N['THT_PCT'] = f'{100 * (np.exp(th[1]) - 1):.1f}'
N['NAIVE_P'] = f2(R['naive_ols'][0]); N['NAIVE_T'] = f3(R['naive_ols'][1])
N['NUNITS'] = f"{R['n_units']:,}"
b, bs = np.array(R['blp']), np.array(R['blp_se'])
labs = ['P', 'T'] + [f'{k}_{z}' for z in R['moderators'] for k in ('P', 'T')]
for l, v, s in zip(labs, b, bs):
    N[f'BLP_{l}'] = f3(v); N[f'BLP_{l}_SE'] = f3(s)
tPc, tTc = np.array(R['thP_obs_c']), np.array(R['thT_obs_c'])
ciT = np.array(R['thT_obs_c_ci']); ciP = np.array(R['thP_obs_c_ci'])
N['THPC_MIN'] = f2(tPc.min()); N['THPC_MAX'] = f2(tPc.max())
N['THTC_MIN'] = f3(tTc.min()); N['THTC_MAX'] = f3(tTc.max())
N['THTC_ARGMIN'] = IDS[int(np.argmin(tTc))]; N['THTC_ARGMAX'] = IDS[int(np.argmax(tTc))]
N['N_T_NEG_SIG'] = str(int((ciT[1] < 0).sum())); N['N_T_POS_SIG'] = str(int((ciT[0] > 0).sum()))
N['THPC_ARGMIN'] = IDS[int(np.argmin(tPc))]; N['THPC_ARGMAX'] = IDS[int(np.argmax(tPc))]

# ---------------- synthetic validation + causal errors
rows_val, rows_ce = [], []
allx = {'P': [], 'T': []}; ally = {'P': [], 'T': []}
for m in MODELS:
    c = R['causal'][m]; k = m.replace('-', '')
    for t in 'PT':
        x = np.array(c[f'ace_{t}_c']); y = np.array(c[f'dml_sim_{t}_c'])
        allx[t] += list(x); ally[t] += list(y)
        N[f'ACE{t}_{k}'] = f2(c[f'ace_{t}_w']) if t == 'P' else f3(c[f'ace_{t}_w'])
        N[f'DMLS{t}_{k}'] = f2(c[f'dml_sim_{t}']) if t == 'P' else f3(c[f'dml_sim_{t}'])
        N[f'CE{t}_{k}'] = f2(c[f'causal_err_{t}']) if t == 'P' else f3(c[f'causal_err_{t}'])
    N[f'WRONGP_{k}'] = f'{100 * c["frac_wrong_sign_P"]:.1f}'
    rel = [abs(c['dml_sim_P'] - c['ace_P_w']) / abs(c['ace_P_w'])]
    rows_val.append([m, f2(c['ace_P_w']), f"{c['dml_sim_P']:.2f} ± {1.96 * c['dml_sim_P_se']:.2f}",
                     f3(c['ace_T_w']), f"{c['dml_sim_T']:.3f} ± {1.96 * c['dml_sim_T_se']:.3f}"])
for t in 'PT':
    x, y = np.array(allx[t]), np.array(ally[t])
    N[f'VAL_R_{t}'] = f2(np.corrcoef(x, y)[0, 1])
    N[f'VAL_MAE_{t}'] = f2(np.mean(np.abs(x - y))) if t == 'P' else f3(np.mean(np.abs(x - y)))
# pooled validation errors
pe = [abs(R['causal'][m]['dml_sim_P'] - R['causal'][m]['ace_P_w']) for m in MODELS]
te = [abs(R['causal'][m]['dml_sim_T'] - R['causal'][m]['ace_T_w']) for m in MODELS]
N['VAL_POOL_P_MAX'] = f2(max(pe)); N['VAL_POOL_T_MAX'] = f3(max(te))
cover_P = sum(abs(R['causal'][m]['dml_sim_P'] - R['causal'][m]['ace_P_w']) <= 1.96 * R['causal'][m]['dml_sim_P_se'] for m in MODELS)
cover_T = sum(abs(R['causal'][m]['dml_sim_T'] - R['causal'][m]['ace_T_w']) <= 1.96 * R['causal'][m]['dml_sim_T_se'] for m in MODELS)
N['COVER_P'] = str(cover_P); N['COVER_T'] = str(cover_T)

# skill vs causal error rank correlation (model level)
mnse = [med.loc[m, 'NSE'] for m in MODELS]
ceP = [R['causal'][m]['causal_err_P'] for m in MODELS]; ceT = [R['causal'][m]['causal_err_T'] for m in MODELS]
N['RHO_NSE_CEP'] = f2(spearmanr(mnse, ceP)[0]); N['RHO_NSE_CET'] = f2(spearmanr(mnse, ceT)[0])
# catchment-level correlation within models pooled
xs, es = [], []
for m in MODELS:
    xs += list(sk[sk.model == m].set_index('id').loc[IDS, 'NSE'].values)
    es += list(np.abs(np.array(R['causal'][m]['ace_P_c']) - tPc))
N['RHO_CATCH_P'] = f2(spearmanr(xs, es)[0])

# ---------------- scenario
so = np.array(R['scen_obs']); ao = amp(so)
N['SCEN_OBS_MED'] = f'{100 * (np.exp(np.median(so)) - 1):.0f}'
N['AMP_OBS_MED'] = f2(np.median(ao)); N['AMP_OBS_MIN'] = f2(ao.min()); N['AMP_OBS_MAX'] = f2(ao.max())
for m in MODELS:
    k = m.replace('-', '')
    sm = np.array(R['causal'][m]['scen']); am = amp(sm)
    N[f'SCEN_{k}'] = f'{100 * (np.exp(np.median(sm)) - 1):.0f}'
    N[f'AMP_{k}'] = f2(np.median(am))
    rel = 100 * (am - ao) / ao
    N[f'AMPERR_{k}'] = f'{np.median(rel):+.0f}'
    N[f'AMPABSERR_{k}'] = f'{np.median(np.abs(rel)):.0f}'
    # error in concentration *increase* (A-1)
    inc = 100 * ((am - 1) - (ao - 1)) / (ao - 1)
    N[f'INCERR_{k}'] = f'{np.median(inc):+.0f}'
N['DP_SD'] = None  # filled below
import dml
units = dml.build_units()
N['DP_SD'] = f2(units.DP.std()); N['DT_SD'] = f2(units.DT.std())
N['DP_MIN'] = f2(units.DP.min()); N['DT_MAX'] = f2(units.DT.max())

# ---------------- robustness
for tag, key in [('random forest', 'RF'), ('linear nuisance', 'LIN'), ('no storage proxy', 'NOQ3'),
                 ('GR4J-CN world, exact forcing', 'EXACT'), ('GR4J-CN world, 10% monthly P error', 'NOISY')]:
    N[f'ROB_{key}_P'] = f2(RB[tag]['theta_P']); N[f'ROB_{key}_T'] = f3(RB[tag]['theta_T'])
N['ATTEN'] = f'{100 * (1 - RB["GR4J-CN world, 10% monthly P error"]["theta_P"] / RB["GR4J-CN world, exact forcing"]["theta_P"]):.0f}'

# ---------------- LSTM / GBM training info
for v in ('base', 'cr'):
    vals = []
    for s in (0, 1, 2):
        lg = np.load(f'out/lstm_{v}_s{s}.npz')['log']
        vals.append((len(lg), lg[:, 2].max()))
    N[f'LSTM_{v}_EPOCHS'] = ', '.join(str(int(a)) for a, _ in vals)
    N[f'LSTM_{v}_VALID'] = f'{np.mean([b for _, b in vals]):.3f}'
N['GBMA_VALID'] = f3(gb['A']['valid_balanced']); N['GBMC_VALID'] = f3(gb['C']['valid_balanced'])

# ---------------- like-for-like discrepancies
DD = json.load(open('out/discrepancy.json'))
for m in MODELS:
    k = m.replace('-', '')
    d = DD[m]
    N[f'DP_{k}'] = f"{d['delta_P']:+.2f}"; N[f'DPSE_{k}'] = f"{d['delta_P_se']:.2f}"
    N[f'DT_{k}'] = f"{d['delta_T']:+.3f}"; N[f'DTSE_{k}'] = f"{d['delta_T_se']:.3f}"
    N[f'NSIGP_{k}'] = str(d['n_sig_P_neg'] + d['n_sig_P_pos']); N[f'NSIGPNEG_{k}'] = str(d['n_sig_P_neg'])
    N[f'NSIGT_{k}'] = str(d['n_sig_T_neg'] + d['n_sig_T_pos'])
    N[f'MADP_{k}'] = f"{d['mean_abs_delta_P']:.3f}"; N[f'MADT_{k}'] = f"{d['mean_abs_delta_T']:.3f}"
    am, ao_ = np.array(d['amp_model']), np.array(d['amp_obsimplied'])
    N[f'AMPM_{k}'] = f'{np.median(am):.2f}'; N[f'AMPO_{k}'] = f'{np.median(ao_):.2f}'
    N[f'AMPPP_{k}'] = f'{np.median(100 * (am - ao_)):+.1f}'
    N[f'AMPPPMAX_{k}'] = f'{np.max(100 * (am - ao_)):.0f}'
    # relative overestimate of the concentration increase, pooled over catchments (robust to near-zero changes)
    N[f'INCREL_{k}'] = f'{100 * (np.sum(am - 1) - np.sum(ao_ - 1)) / np.sum(ao_ - 1):+.0f}'
    N[f'PCTP_{k}'] = f"{100 * -d['delta_P'] / R['theta_obs'][0]:.0f}"
mad = [DD[m]['mean_abs_delta_P'] for m in MODELS]
N['RHO_NSE_MADP'] = f2(spearmanr(mnse, mad)[0])
N['RHO_NSE_MADP_P'] = f2(spearmanr(mnse, mad)[1])
allA = np.array([DD[m]['amp_obsimplied'] for m in MODELS])
N['AMPO_ALL_MED'] = f2(np.median(allA))
N['AMPO_SPREAD'] = f2(np.median(allA.max(0) - allA.min(0)))

# ================= TABLES
T = {}
T['models'] = dict(
    caption='**Table 1.** Models compared. "Causal information" lists what each model is told about the direction of cause and effect. All learned models are regional (one model for all 19 catchments); conceptual models are calibrated per catchment.',
    header=['Model', 'Type', 'Inputs', 'Causal information', 'Parameters'],
    widths=[1150, 1700, 2500, 2610, 1400],
    rows=[['CLIM', 'Seasonal climatology', 'Day of year', 'None (reference)', '–'],
          ['GR4J', 'Conceptual, local', 'P, PET', 'Mass balance, fixed structure', '4 per catchment'],
          ['GR4J-CN', 'Conceptual + snow, local', 'P, T, PET, hypsometry', 'As GR4J + degree-day snow', '6 per catchment'],
          ['GBM-A', 'Gradient boosting, regional', 'Forcing aggregates, calendar, attributes', 'None (associational)', f"{gb['A']['n_iter']} trees"],
          ['GBM-C', 'Gradient boosting, regional', 'Forcing aggregates, attributes (no calendar)', 'Causal parents only; monotone in P (+) and PET (−)', f"{gb['C']['n_iter']} trees"],
          ['LSTM', 'Recurrent network, regional', 'Daily P, T, PET, attributes', 'None', '12,145 × 3 seeds'],
          ['LSTM-CR', 'Recurrent network, regional', 'Daily P, T, PET, attributes', 'Penalty on wrong-signed P and T responses', '12,145 × 3 seeds']])

rows = []
for m in ALL:
    d = sk[sk.model == m]
    rows.append([m, f2(d.NSE.median()), f2(d.logNSE.median()), f2(d.KGE.median()), str(int((d.NSE > 0.7).sum())),
                 f'{d.Q95_bias.median():+.0f}', f'{d.lowseason_bias.median():+.0f}'])
T['skill'] = dict(caption='**Table 2.** Test-period (2015–2018) performance, medians across 19 catchments. Q95 bias and Aug–Oct bias are percentage errors of the flow exceeded 95% of the time and of mean August–October discharge.',
                  header=['Model', 'NSE', 'NSE (log Q)', 'KGE', 'n(NSE > 0.7)', 'Q95 bias (%)', 'Aug–Oct bias (%)'],
                  widths=[1300, 1100, 1300, 1100, 1400, 1500, 1660], rows=rows)

rows = [['Observations', '–', f"{th[0]:.2f} ± {1.96*se[0]:.2f}", '–', '–', f"{th[1]:.3f} ± {1.96*se[1]:.3f}", '–']]
for m in MODELS:
    c = R['causal'][m]; d = DD[m]
    rows.append([m, f2(c['ace_P_w']), f"{c['dml_sim_P']:.2f} ± {1.96*c['dml_sim_P_se']:.2f}",
                 f"{d['delta_P']:+.2f} ± {1.96*d['delta_P_se']:.2f}", f3(c['ace_T_w']),
                 f"{c['dml_sim_T']:.3f} ± {1.96*c['dml_sim_T_se']:.3f}", f"{d['delta_T']:+.3f} ± {1.96*d['delta_T_se']:.3f}"])
T['causal'] = dict(caption='**Table 3.** Causal effects of window precipitation (θ_{P}, elasticity, –) and temperature (θ_{T}, °C^{-1}) on low-flow-season discharge. ACE: the model\'s true do-intervention effect (weighted as in the estimator). DML: the estimate recovered from the model\'s simulated discharge (synthetic validation), or from observations (first row). δ = θ_{obs} − θ_{M}: like-for-like causal discrepancy, estimated with the same DML design applied to log Q_{obs} − log Q_{sim}; negative δ_{P} means the model over-responds to precipitation. Intervals are ± 1.96 cluster-robust standard errors.',
                   header=['Source', 'θ_{P} ACE', 'θ_{P} DML', 'δ_{P}', 'θ_{T} ACE', 'θ_{T} DML', 'δ_{T}'],
                   widths=[1360, 1000, 1400, 1400, 1000, 1600, 1600], rows=rows)

# ---- supplement tables
S = {}
S['S1'] = dict(caption='**Table S1.** Catchment attributes (training years 2000–2011 for climate and signatures). BFI: Lyne–Hollick baseflow index (α = 0.925, three passes); snow fraction: share of precipitation falling as snow in the elevation-band model.',
               header=['ID', 'Station (code)', 'Area (km²)', 'Elev. (m)', 'P (mm/yr)', 'PET/P', 'Q/P', 'BFI', 'Snow'],
               widths=[450, 3300, 800, 750, 800, 750, 700, 700, 760],
               rows=[[r.id, f'{r["name"]} ({r.code})', f'{r.area:.0f}', f'{r.elev_med:.0f}', f'{365.25*r.p_mean:.0f}', f2(r.aridity), f2(r.runoff_ratio), f2(r.bfi), f2(r.snow_frac)]
                     for _, r in sig.iterrows()])
def load_params(w):
    d = {}
    for f in glob.glob(f'out/params_{w}_*.json'):
        d.update(json.load(open(f)))
    return d
pg, pc = load_params('gr4j'), load_params('gr4jcn')
S['S2'] = dict(caption='**Table S2.** Calibrated parameters of GR4J and GR4J-CN (X1, X3 in mm; X2 in mm; X4 in days; CTG dimensionless; Kf in mm °C^{-1} d^{-1}) and training objective L (Eq. 1, lower is better).',
               header=['ID', 'GR4J X1', 'X2', 'X3', 'X4', 'L', 'GR4J-CN X1', 'X2', 'X3', 'X4', 'CTG', 'Kf', 'L'],
               widths=[400] + [720] * 12,
               rows=[[i] + [f'{v:.1f}' if j in (0, 2) else f'{v:.2f}' for j, v in enumerate(pg[i]['x'])] + [f3(pg[i]['L'])] +
                     [f'{v:.1f}' if j in (0, 2) else f'{v:.2f}' for j, v in enumerate(pc[i]['x'])] + [f3(pc[i]['L'])] for i in IDS])
piv = sk.pivot(index='id', columns='model', values='NSE').loc[IDS, ALL]
S['S3'] = dict(caption='**Table S3.** Test-period NSE for every catchment and model.', header=['ID'] + ALL,
               widths=[600] + [int(8760 / len(ALL))] * len(ALL), rows=[[i] + [f2(v) for v in piv.loc[i]] for i in IDS])
piv = sk.pivot(index='id', columns='model', values='logNSE').loc[IDS, ALL]
S['S4'] = dict(caption='**Table S4.** Test-period NSE of log-transformed discharge for every catchment and model.', header=['ID'] + ALL,
               widths=[600] + [int(8760 / len(ALL))] * len(ALL), rows=[[i] + [f2(v) for v in piv.loc[i]] for i in IDS])
tvm = tv.groupby(['model', 'block']).median(numeric_only=True)
S['S5'] = dict(caption='**Table S5.** Median NSE, log-NSE and KGE in the training (2000–2011) and validation (2012–2014) blocks.',
               header=['Model', 'NSE train', 'NSE valid', 'logNSE train', 'logNSE valid', 'KGE train', 'KGE valid'],
               widths=[1360] + [1350] * 6,
               rows=[[m, f2(tvm.loc[(m, 'train'), 'NSE']), f2(tvm.loc[(m, 'valid'), 'NSE']), f2(tvm.loc[(m, 'train'), 'logNSE']),
                      f2(tvm.loc[(m, 'valid'), 'logNSE']), f2(tvm.loc[(m, 'train'), 'KGE']), f2(tvm.loc[(m, 'valid'), 'KGE'])] for m in ALL])
S['S6'] = dict(caption='**Table S6.** Best linear projection of the observational causal effects on standardised moderators (coefficient ± cluster-robust standard error). Interaction terms give the change in θ per standard deviation of the moderator.',
               header=['Term', 'θ_{P}', 'θ_{T}'], widths=[3360, 3000, 3000],
               rows=[['Intercept (average catchment)', f"{b[0]:.3f} ± {bs[0]:.3f}", f"{b[1]:.3f} ± {bs[1]:.3f}"]] +
                    [[f'× {z}', f"{b[2+2*j]:.3f} ± {bs[2+2*j]:.3f}", f"{b[3+2*j]:.3f} ± {bs[3+2*j]:.3f}"] for j, z in enumerate(R['moderators'])])
gate = np.array(R['gate_obs'])
rows = []
for i, c in enumerate(IDS):
    rows.append([c, f"{tPc[i]:.2f} [{ciP[0][i]:.2f}, {ciP[1][i]:.2f}]", f"{tTc[i]:.3f} [{ciT[0][i]:.3f}, {ciT[1][i]:.3f}]", f2(gate[i, 0]), f3(gate[i, 1])] +
                [f"{R['causal'][m]['ace_P_c'][i]:.2f} / {R['causal'][m]['ace_T_c'][i]:.3f}" for m in MODELS])
S['S7'] = dict(caption='**Table S7.** Catchment causal effects. Observed: projection-based θ_{c} with 95% year-cluster bootstrap intervals, and the noisier catchment-only (group) estimates. Models: true do-intervention effects θ_{P} / θ_{T}.',
               header=['ID', 'Obs θ_{P} [95% CI]', 'Obs θ_{T} [95% CI]', 'Group θ_{P}', 'Group θ_{T}'] + MODELS,
               widths=[400, 1300, 1450, 700, 700] + [800] * len(MODELS), rows=rows)
S['S8'] = dict(caption='**Table S8.** Robustness of the observational benchmark and attenuation by forcing error.',
               header=['Variant', 'θ_{P} ± SE', 'θ_{T} ± SE'], widths=[4360, 2500, 2500],
               rows=[[k, f"{v['theta_P']:.2f} ± {v['se_P']:.2f}", f"{v['theta_T']:.3f} ± {v['se_T']:.3f}"] for k, v in RB.items()])
rows = []
sci = np.array(R['scen_obs_ci'])
for i, c in enumerate(IDS):
    rows.append([c, f"{100*(np.exp(so[i])-1):+.0f} [{100*(np.exp(sci[0][i])-1):+.0f}, {100*(np.exp(sci[1][i])-1):+.0f}]", f2(ao[i])] +
                [f"{100*(np.exp(R['causal'][m]['scen'][i])-1):+.0f} / {amp(R['causal'][m]['scen'][i]):.2f}" for m in MODELS])
rows = []
for i, c in enumerate(IDS):
    rows.append([c] + [f"{100*(np.exp(R['causal'][m]['scen'][i])-1):+.0f} / {DD[m]['amp_model'][i]:.2f} / {DD[m]['amp_obsimplied'][i]:.2f}" for m in MODELS])
S['S9'] = dict(caption='**Table S9.** Hot–dry scenario (July–September P −20%, T +2 °C) for every catchment and model: change in September discharge (%) / concentration amplification factor A simulated by the model / A after correcting the model response by its estimated causal discrepancy (baseline dilution ratio 10).',
               header=['ID'] + MODELS, widths=[560] + [1466] * len(MODELS), rows=rows)
import re as _re
def _mn(v):
    if not isinstance(v, str): return v
    v = _re.sub(r'(?<![\w.])[-−]0\.(0+)(?!\d*[1-9])', r'0.\1', v)   # -0.00 -> 0.00
    v = _re.sub(r'\+0\.(0+)(?!\d*[1-9])', r'0.\1', v)
    return _re.sub(r'(?<![\w])-(?=\d)', '−', v)
for D_ in (T, S):
    for t in D_.values():
        t['rows'] = [[_mn(c) for c in r] for r in t['rows']]
for k in list(N):
    N[k] = _mn(N[k]) if isinstance(N[k], str) else N[k]
json.dump(T, open('out/tables_main.json', 'w'), indent=1, ensure_ascii=False)
json.dump(S, open('out/tables_supp.json', 'w'), indent=1, ensure_ascii=False)
json.dump(N, open('out/numbers.json', 'w'), indent=1, ensure_ascii=False)
for k, v in N.items():
    print(k, v)
