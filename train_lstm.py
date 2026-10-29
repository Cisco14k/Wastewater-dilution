"""Regional LSTM trained on all 19 catchments (training block 2000-2011), early stopping on 2012-2014.
Loss approximates 1 - (0.5 NSE + 0.5 NSE_log) per catchment (basin-normalised, Kratzert et al., 2019).
Optional causal regulariser (variant 'cr'): penalises wrong-signed interventional responses (see manuscript Sect. 3.4)."""
import sys, time, json
import numpy as np
from common import *
from lstm_np import LSTM
from hydro import oudin_pet

variant = sys.argv[1]              # 'base' or 'cr'
seed = int(sys.argv[2])
HID, SEQ, SPIN, BATCH = 48, 545, 365, 32
EPOCH_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
MAX_EPOCHS, PATIENCE = 45, 8
LAM = 2.0                          # weight of the causal regulariser

att = static_attributes()
S = att[STATIC_COLS].values.astype(float)
S = (S - S.mean(0)) / S.std(0)
DOY = DATES.dayofyear.values

def dyn_features(Pa, Ta, Ea):
    return np.stack([Pa, Ta, Ea], -1)

F = dyn_features(P, T, E)                                  # (NC, NT, 3)
mu = F[:, TRAIN | WARM].reshape(-1, 3).mean(0); sd = F[:, TRAIN | WARM].reshape(-1, 3).std(0)
def build_X(Fdyn, idx=None):
    Fn = (Fdyn - mu) / sd
    Sb = np.broadcast_to(S[:, None, :] if idx is None else S[idx][:, None, :], Fn.shape[:2] + (S.shape[1],))
    return np.concatenate([Fn, Sb], -1)
X_all = build_X(F)

# per-catchment loss normalisation (training data only)
qtr = np.where(TRAIN, Q, np.nan)
EPS = np.nanmean(qtr, 1) / 100
VAR_Q = np.nanvar(qtr, 1)
VAR_L = np.array([np.nanvar(np.log(qtr[i] + EPS[i])) for i in range(NC)])

# perturbed inputs for the causal regulariser: P x 1.1 and T + 1 degC (PET recomputed with Oudin)
dPET = np.stack([np.clip(oudin_pet(T[i] + 1, DOY, LAT[i]) - oudin_pet(T[i], DOY, LAT[i]), 0, None) for i in range(NC)])
X_P = build_X(dyn_features(P * 1.1, T, E))
X_T = build_X(dyn_features(P, T + 1.0, E + dPET))
SNOWY = att.snow_frac.values > 0.10       # warming response sign not constrained where snow is material

train_end = np.where(TRAIN)[0][-1]
starts_ok = np.arange(0, train_end - SEQ + 2)              # windows fully inside warm-up+training
rng = np.random.default_rng(seed)
model = LSTM(X_all.shape[-1], HID, seed=seed)

def loss_grad(y, cidx, tidx):
    q = Q[cidx[:, None], tidx]
    w = np.isfinite(q); w[:, :SPIN] = False
    qf = np.where(w, q, 1.0)
    e = EPS[cidx][:, None]
    n = w.sum()
    r1 = (y - qf); r2 = np.log(y + e) - np.log(qf + e)
    L = 0.5 * np.sum(w * r1 ** 2 / VAR_Q[cidx][:, None]) / n + 0.5 * np.sum(w * r2 ** 2 / VAR_L[cidx][:, None]) / n
    dy = (w * r1 / VAR_Q[cidx][:, None] + w * r2 / (y + e) / VAR_L[cidx][:, None]) / n
    return L, dy

def predict(Xc, chunk=19):
    y, _, _ = model.forward(Xc, keep=False)
    return y

def objective(ys, mask):
    out = []
    for i in range(NC):
        m = mask & np.isfinite(Q[i]); o = Q[i, m]; s = ys[i, m]
        n1 = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
        lo = np.log(o + EPS[i]); ls = np.log(s + EPS[i])
        n2 = 1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
        out.append(0.5 * n1 + 0.5 * n2)
    return np.median(out)

vmask = VALID
best, best_p, bad, lr = -np.inf, None, 0, 1e-3
t0 = time.time()
log = []
for ep in range(MAX_EPOCHS):
    tl = []
    for _ in range(EPOCH_BATCHES):
        cidx = rng.integers(0, NC, BATCH)
        st = rng.choice(starts_ok, BATCH)
        tidx = st[:, None] + np.arange(SEQ)[None, :]
        Xb = X_all[cidx[:, None], tidx]
        y, fw, _ = model.forward(Xb)
        L, dy = loss_grad(y, cidx, tidx)
        g = model.backward(dy, fw)
        if variant == 'cr':
            # causal regulariser: an increase in precipitation must not reduce discharge; in snow-free
            # catchments warming (higher evaporative demand) must not increase discharge.
            msk = np.zeros_like(y); msk[:, SPIN:] = 1.0
            yP, fwP, _ = model.forward(X_P[cidx[:, None], tidx])
            dP = (yP - y) / (y + EPS[cidx][:, None])            # relative response
            vP = np.maximum(0, -dP) * msk
            yT, fwT, _ = model.forward(X_T[cidx[:, None], tidx])
            dT = (yT - y) / (y + EPS[cidx][:, None])
            mT = msk * (~SNOWY[cidx])[:, None]
            vT = np.maximum(0, dT) * mT
            nn = msk.sum()
            R = (np.sum(vP ** 2) + np.sum(vT ** 2)) / nn
            L = L + LAM * R
            # gradients: d/dyP, d/dyT, d/dy (through both relative responses)
            denom = (y + EPS[cidx][:, None])
            gP = LAM * 2 * (-vP) / nn            # dR/d dP
            gT = LAM * 2 * vT / nn
            dyP = gP / denom; dyT = gT / denom
            dy_base = -gP * (yP + EPS[cidx][:, None]) / denom ** 2 - gT * (yT + EPS[cidx][:, None]) / denom ** 2
            g2 = model.backward(dyP, fwP); g3 = model.backward(dyT, fwT); g4 = model.backward(dy_base, fw)
            for k in g:
                g[k] = g[k] + g2[k] + g3[k] + g4[k]
        model.adam(g, lr)
        tl.append(L)
    ys = predict(X_all[:, :np.where(VALID)[0][-1] + 1])
    ys = np.pad(ys, ((0, 0), (0, NT - ys.shape[1])), constant_values=np.nan)
    v = objective(ys, vmask)
    log.append((ep, float(np.mean(tl)), float(v)))
    print(f'{variant} s{seed} ep{ep:02d} train {np.mean(tl):.4f} valid-median-balancedNSE {v:.4f} lr {lr:.1e} {time.time()-t0:.0f}s', flush=True)
    if v > best + 1e-4:
        best, best_p, bad = v, model.get(), 0
    else:
        bad += 1
        if bad % 4 == 0:
            lr *= 0.5
        if bad >= PATIENCE:
            break
model.set(best_p)
np.savez(f'out/lstm_{variant}_s{seed}.npz', **best_p, log=np.array(log))
# full continuous simulations: base and the regulariser's perturbations (for diagnostics)
y = predict(X_all)
np.save(f'out/sim_lstm_{variant}_s{seed}.npy', y)
print('done', best)
