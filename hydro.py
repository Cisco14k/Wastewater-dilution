"""Shared hydrological utilities: Oudin PET, metrics, GR4J and GR4J-CemaNeige (vectorised over parameter sets)."""
import numpy as np

# ---------------------------------------------------------------- PET (Oudin et al., 2005), as in airGR::PE_Oudin
def extraterrestrial_radiation(doy, lat_deg):
    """Daily extraterrestrial radiation [MJ m-2 d-1] (FAO-56)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))

def oudin_pet(temp, doy, lat_deg):
    re = extraterrestrial_radiation(np.asarray(doy, float), lat_deg)
    lam, rho = 2.45, 1000.0
    pet = re / (lam * rho) * (np.asarray(temp) + 5) / 100 * 1000
    return np.where(np.asarray(temp) + 5 > 0, pet, 0.0)

# ---------------------------------------------------------------- metrics
def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

def kge(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    r = np.corrcoef(o, s)[0, 1]; a = s.std() / o.std(); b = s.mean() / o.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), r, a, b

def lognse(o, s, eps=None):
    m = np.isfinite(o) & np.isfinite(s)
    if eps is None:
        eps = np.nanmean(o) / 100
    return nse(np.log(o[m] + eps), np.log(s[m] + eps))

def pbias(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    return 100 * (s[m].sum() - o[m].sum()) / o[m].sum()

# ---------------------------------------------------------------- GR4J (Perrin et al., 2003), vectorised over N parameter sets
def _uh_ordinates(x4, nmax):
    """SH1/SH2 S-curves -> UH1/UH2 ordinates, shape (N, nmax) and (N, 2*nmax)."""
    x4 = np.asarray(x4)[:, None]
    def sh1(t):
        return np.where(t <= 0, 0.0, np.where(t < x4, (t / x4) ** 2.5, 1.0))
    def sh2(t):
        return np.where(t <= 0, 0.0, np.where(t < x4, 0.5 * (t / x4) ** 2.5,
                        np.where(t < 2 * x4, 1 - 0.5 * np.clip(2 - t / x4, 0, None) ** 2.5, 1.0)))
    t1 = np.arange(1, nmax + 1)[None, :].astype(float)
    t2 = np.arange(1, 2 * nmax + 1)[None, :].astype(float)
    uh1 = sh1(t1) - sh1(t1 - 1)
    uh2 = sh2(t2) - sh2(t2 - 1)
    return uh1, uh2

def gr4j(P, E, params, s0=0.5, r0=0.5):
    """P, E: (n,) or (N, n) daily forcing [mm]. params: (N,4) X1..X4. Returns Q (N, n) and final storages."""
    params = np.atleast_2d(params)
    N = params.shape[0]
    X1, X2, X3, X4 = params.T
    P = np.broadcast_to(np.atleast_2d(P), (N, np.shape(P)[-1]))
    E = np.broadcast_to(np.atleast_2d(E), (N, np.shape(E)[-1]))
    n = P.shape[1]
    nmax = int(np.ceil(X4.max())) + 1
    uh1, uh2 = _uh_ordinates(X4, nmax)
    S = s0 * X1; Rr = r0 * X3
    q1 = np.zeros((N, nmax)); q2 = np.zeros((N, 2 * nmax))
    Q = np.empty((N, n))
    for t in range(n):
        p = P[:, t]; e = E[:, t]
        net = p >= e
        pn = np.where(net, p - e, 0.0); en = np.where(net, 0.0, e - p)
        sr = S / X1
        tp = np.tanh(np.minimum(pn / X1, 13.0))
        ps = X1 * (1 - sr ** 2) * tp / (1 + sr * tp)
        te = np.tanh(np.minimum(en / X1, 13.0))
        es = S * (2 - sr) * te / (1 + (1 - sr) * te)
        S = S + ps - es
        perc = S * (1 - (1 + (4.0 / 9.0 * S / X1) ** 4) ** -0.25)
        S = S - perc
        pr = perc + pn - ps
        q1 = np.roll(q1, -1, axis=1); q1[:, -1] = 0.0; q1 += 0.9 * pr[:, None] * uh1
        q2 = np.roll(q2, -1, axis=1); q2[:, -1] = 0.0; q2 += 0.1 * pr[:, None] * uh2
        f = X2 * (np.maximum(Rr, 0) / X3) ** 3.5
        Rr = np.maximum(0.0, Rr + q1[:, 0] + f)
        qr = Rr * (1 - (1 + (Rr / X3) ** 4) ** -0.25)
        Rr = Rr - qr
        qd = np.maximum(0.0, q2[:, 0] + f)
        Q[:, t] = qr + qd
    return Q

# ---------------------------------------------------------------- CemaNeige (Valery et al., 2014), simplified (no hysteresis), 5 elevation bands
GRAD_T = -0.0055   # degC per m (mean of airGR seasonal gradient table)
BETA_P = 0.0004    # exponential precipitation-elevation gradient [m-1], as airGR

def band_forcing(P, T, hypso, nbands=5):
    """Distribute catchment-mean P and T to equal-area elevation bands using hypsometry (101 quantiles)."""
    zmed = hypso[50]
    # band median elevations: quantiles at 10,30,50,70,90 %
    qs = np.linspace(100 / nbands / 2, 100 - 100 / nbands / 2, nbands)
    zb = np.interp(qs, np.arange(101), hypso)
    Tb = T[None, :] + GRAD_T * (zb[:, None] - zmed)
    w = np.exp(BETA_P * (np.minimum(zb, 4000) - np.minimum(zmed, 4000)))
    w = w / w.mean()                       # conserve catchment-mean precipitation
    Pb = P[None, :] * w[:, None]
    return Pb, Tb

def cemaneige(P, T, hypso, cn, train_mask=None):
    """cn: (N,2) [CTG in (0,1), Kf mm/degC/d]. Returns liquid water input to GR4J (N, n)."""
    cn = np.atleast_2d(cn); N = cn.shape[0]
    ctg, kf = cn[:, 0][:, None], cn[:, 1][:, None]
    Pb, Tb = band_forcing(P, T, hypso)
    nb, n = Pb.shape
    frac_solid = np.clip((3.0 - Tb) / 4.0, 0, 1)   # linear between -1 and 3 degC (USACE)
    Psol = Pb * frac_solid; Pliq = Pb - Psol
    msk = slice(None) if train_mask is None else train_mask
    gthr = 0.9 * np.maximum(Psol[:, msk].mean(1) * 365.25, 1e-3)  # 0.9 x mean annual solid precip per band
    G = np.zeros((N, nb)); eTG = np.zeros((N, nb))
    out = np.empty((N, n))
    for t in range(n):
        G = G + Psol[None, :, t]
        eTG = np.minimum(0.0, ctg * eTG + (1 - ctg) * Tb[None, :, t])
        fpot = np.where((eTG == 0) & (Tb[None, :, t] > 0), np.minimum(G, kf * Tb[None, :, t]), 0.0)
        gratio = np.minimum(G / gthr[None, :], 1.0)
        melt = (0.9 * gratio + 0.1) * fpot
        G = G - melt
        out[:, t] = (Pliq[None, :, t] + melt).mean(1)
    return out
