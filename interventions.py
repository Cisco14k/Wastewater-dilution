"""Model simulators under arbitrary forcing, and window interventions matching the DML estimand."""
import os
import json, glob, pickle
import numpy as np
from common import *
from hydro import gr4j, cemaneige, oudin_pet
import dml

DOY = DATES.dayofyear.values
_mon = DATES.month.values; _yr = DATES.year.values

def dpet(Ta_new):
    """Oudin PET change implied by a temperature change (added to the dataset PET)."""
    return np.stack([oudin_pet(Ta_new[i], DOY, LAT[i]) - oudin_pet(T[i], DOY, LAT[i]) for i in range(NC)])

def load_params(which):
    d = {}
    for f in glob.glob(f'out/params_{which}_*.json'):
        d.update(json.load(open(f)))
    return np.array([d[i]['x'] for i in IDS])

class Sim:
    def __init__(self):
        self.pg = load_params('gr4j')
        try:
            self.pc = load_params('gr4jcn')
        except KeyError:
            self.pc = None
        self.gbm = {v: pickle.load(open(f'out/gbm{v}.pkl', 'rb')) for v in 'AC'}
        self.lstm = {}
    def gr4j(self, Pa, Ta, Ea):
        return gr4j(Pa, Ea, self.pg)
    def gr4jcn(self, Pa, Ta, Ea):
        liq = np.stack([cemaneige(Pa[i], Ta[i], H[i], self.pc[i:i + 1, 4:6], train_mask=TRAIN)[0] for i in range(NC)])
        return gr4j(liq, Ea, self.pc[:, :4])
    def gbm_(self, v, Pa, Ta, Ea):
        import gbm as G
        Xf, _, _ = G.features(Pa, Ta, Ea, calendar=(v == 'A'))
        return np.maximum(G.predict(self.gbm[v], Xf), 0)
    def lstm_(self, variant, Pa, Ta, Ea, seeds=(0, 1, 2)):
        import sys
        from lstm_np import LSTM
        if not hasattr(self, '_lstm_mod'):
            sys.argv = ['x', 'base', '0']
            src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_lstm.py')).read().split('rng = np.random.default_rng(seed)')[0]
            g = {}
            exec(src, g)
            self._lstm_mod = g
        g = self._lstm_mod
        X = g['build_X'](g['dyn_features'](Pa, Ta, Ea))
        ys = []
        for s in seeds:
            f = f'out/lstm_{variant}_s{s}.npz'
            z = np.load(f)
            m = LSTM(X.shape[-1], g['HID']); m.set({k: z[k] for k in ('W', 'b', 'v', 'c')})
            ys.append(m.forward(X, keep=False)[0])
        return np.mean(ys, 0)
    def run(self, name, Pa, Ta, Ea):
        if name == 'GR4J': return self.gr4j(Pa, Ta, Ea)
        if name == 'GR4J-CN': return self.gr4jcn(Pa, Ta, Ea)
        if name == 'GBM-A': return self.gbm_('A', Pa, Ta, Ea)
        if name == 'GBM-C': return self.gbm_('C', Pa, Ta, Ea)
        if name == 'LSTM': return self.lstm_('base', Pa, Ta, Ea)
        if name == 'LSTM-CR': return self.lstm_('cr', Pa, Ta, Ea)
        raise KeyError(name)

def window_mask(m, parity, months_back=2):
    """Days in months m-2..m of years with year % 2 == parity (handles windows crossing year start)."""
    k_target = (_yr - 1999) * 12 + (_mon - 1)
    msk = np.zeros(NT, bool)
    for y in range(2000, 2019):
        if y % 2 != parity:
            continue
        kt = (y - 1999) * 12 + (m - 1)
        msk |= (k_target >= kt - months_back) & (k_target <= kt)
    return msk

def interventional_effects(sim, name, units, dP=0.10, dT=1.0):
    """Model-world average causal effects for the DML estimand.
    Returns per-unit effects (aligned with `units`) for P (per unit log P) and T (per degC)."""
    base = sim.run(name, P, T, E)
    Qm0 = dml.monthly(base)
    eP = np.full(len(units), np.nan); eT = np.full(len(units), np.nan)
    for m in dml.MONTHS:
        for par in (0, 1):
            w = window_mask(m, par)
            sel = (units.month.values == m) & (units.year.values % 2 == par)
            ci = units.c.values[sel]; kk = units.k.values[sel]
            out = {}
            for tag, (Pa, Ta) in {'P+': (np.where(w, P * (1 + dP), P), T), 'P-': (np.where(w, P * (1 - dP), P), T),
                                  'T+': (P, np.where(w, T + dT, T)), 'T-': (P, np.where(w, T - dT, T))}.items():
                Ea = E + dpet(Ta) if tag[0] == 'T' else E
                Ea = np.maximum(Ea, 0)
                Qm = dml.monthly(sim.run(name, Pa, Ta, Ea))
                out[tag] = np.log(Qm[ci, kk]) - np.log(Qm0[ci, kk])
            eP[sel] = (out['P+'] - out['P-']) / (np.log(1 + dP) - np.log(1 - dP))
            eT[sel] = (out['T+'] - out['T-']) / (2 * dT)
    return eP, eT, base

def scenario(sim, name, m=9, fP=0.8, dT=2.0):
    """Hot-dry window scenario (months m-2..m, all years): returns per-catchment mean dlogQ in month m."""
    base = sim.run(name, P, T, E)
    w = window_mask(m, 0) | window_mask(m, 1)
    Ta = np.where(w, T + dT, T); Pa = np.where(w, P * fP, P)
    pert = sim.run(name, Pa, Ta, np.maximum(E + dpet(Ta), 0))
    Q0 = dml.monthly(base); Q1 = dml.monthly(pert)
    ks = [(y - 1999) * 12 + (m - 1) for y in range(2000, 2019)]
    return np.nanmean(np.log(Q1[:, ks]) - np.log(Q0[:, ks]), 1)
