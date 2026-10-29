"""Assemble the 19 CAMELS-FR catchments (airGRdatasets v0.2.3) into one npz + attribute table."""
import glob, os, json
import numpy as np, pandas as pd
import rdaread as R
from hydro import oudin_pet

SRC = '/tmp/claude-0/-home-claude/277e979c-7d79-5a01-9b9e-645cafcc8533/scratchpad/dl/airGRdatasets/data'
IDMAP = {'A273011002':'A','A605102001':'B','B222001001':'C','E540031001':'D','E645651001':'E',
         'F439000101':'F','H010002001':'G','H120101001':'H','H622101001':'I','J171171001':'J',
         'J421191001':'K','K134181001':'L','K265401001':'M','K731261001':'N','V123521001':'O',
         'X031001001':'P','X045401001':'Q','Y643401001':'R','Y862000101':'S'}

rows, P, T, E, Q, H = [], [], [], [], [], []
dates = None
for code in sorted(IDMAP, key=lambda c: IDMAP[c]):
    d = R.to_py(list(R.read_rda(f'{SRC}/{code}.rda').values())[0])
    ts = d['TS']
    dt = pd.to_datetime(ts['Date'][1], unit='s').normalize()
    if dates is None:
        dates = dt
    assert (dt == dates).all()
    p = np.asarray(ts['Ptot'], float); t = np.asarray(ts['Temp'], float)
    e = np.asarray(ts['Evap'], float); q = np.asarray(ts['Qmmd'], float)
    q[q < 0] = np.nan
    lat = float(d['Meta']['Coor']['Y'][0]); lon = float(d['Meta']['Coor']['X'][0])
    hyp = np.asarray(d['Hypso'], float)
    # verify PET reproduces Oudin from T and latitude
    e2 = oudin_pet(t, dates.dayofyear.values, lat)
    rows.append(dict(id=IDMAP[code], code=code, code_h2=d['Meta']['Code']['H2'][0], name=d['Meta']['Name'][0],
                     lat=lat, lon=lon, area=float(d['Meta']['Area'][0]),
                     elev_min=hyp[0], elev_med=hyp[50], elev_max=hyp[100],
                     pet_check_maxabs=float(np.nanmax(np.abs(np.round(e2, 1) - e))),
                     pet_check_r=float(np.corrcoef(e2, e)[0, 1])))
    P.append(p); T.append(t); E.append(e); Q.append(q); H.append(hyp)
att = pd.DataFrame(rows)
P, T, E, Q, H = map(np.array, (P, T, E, Q, H))
np.savez_compressed('data/camels_fr19.npz', P=P, T=T, E=E, Q=Q, H=H,
                    dates=dates.values.astype('datetime64[D]'), ids=att.id.values, lat=att.lat.values)
att.to_csv('data/attributes_raw.csv', index=False)
print(att[['id', 'name', 'area', 'elev_med', 'pet_check_maxabs', 'pet_check_r']])
print('dates', dates[0], dates[-1], len(dates), 'Q completeness %', np.round(100 * np.mean(np.isfinite(Q), 1), 1))
