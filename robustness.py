"""Robustness of the observational causal benchmark:
(1) random-forest nuisance learners; (2) no storage proxy in X; (3) linear (OLS) nuisance = conventional regression;
(4) attenuation from forcing error: in the GR4J-CN model world, compute D from precipitation perturbed by
    independent monthly multiplicative noise (CV 10 %), while discharge is simulated from the true forcing."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from common import *
import dml

units = dml.build_units()
out = {}
rf = lambda s: RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=1, random_state=s)
lin = lambda s: LinearRegression()
for tag, kw in [('HGB (main)', {}), ('random forest', dict(learner=rf)), ('linear nuisance', dict(learner=lin)),
                ('no storage proxy', dict(xcols=[c for c in dml.XCOLS if c != 'Q3']))]:
    res = dml.residualise(units, reps=3, **kw)
    b, se = dml.plm(res, units)
    out[tag] = dict(theta_P=float(b[0]), se_P=float(se[0]), theta_T=float(b[1]), se_T=float(se[1]))
    print(tag, out[tag], flush=True)

# forcing-error attenuation in a model world with known truth
from interventions import Sim
sim = Sim()
base = sim.run('GR4J-CN', P, T, E)
truth = json.load(open('out/results.json'))['causal']['GR4J-CN']['ace_P_w'] if False else None
rng = np.random.default_rng(3)
Pm_true = dml.monthly(P)
noise = np.exp(rng.normal(0, 0.10, Pm_true.shape) - 0.5 * 0.10 ** 2)
us_true = dml.build_units(Qsim=base).merge(units[['c', 'year', 'month']], on=['c', 'year', 'month'])
us_noisy = dml.build_units(Qsim=base, Pm=Pm_true * noise).merge(units[['c', 'year', 'month']], on=['c', 'year', 'month'])
for tag, us in [('GR4J-CN world, exact forcing', us_true), ('GR4J-CN world, 10% monthly P error', us_noisy)]:
    res = dml.residualise(us, reps=3)
    b, se = dml.plm(res, us)
    out[tag] = dict(theta_P=float(b[0]), se_P=float(se[0]), theta_T=float(b[1]), se_T=float(se[1]))
    print(tag, out[tag], flush=True)
json.dump(out, open('out/robustness.json', 'w'), indent=1)
