"""Single-layer LSTM rainfall-runoff model in NumPy with analytic backpropagation through time and Adam.
Output head: softplus(linear(h)) -> discharge [mm/d]."""
import numpy as np

def sig(x):
    return 0.5 * (1 + np.tanh(0.5 * x))

def softplus(x):
    return np.logaddexp(0, x)

class LSTM:
    def __init__(self, n_in, n_hid, seed=0):
        rng = np.random.default_rng(seed)
        k = 1 / np.sqrt(n_hid)
        self.p = {
            'W': rng.uniform(-k, k, (n_in + n_hid, 4 * n_hid)),
            'b': np.zeros(4 * n_hid),
            'v': rng.uniform(-k, k, (n_hid,)),
            'c': np.array([0.0]),
        }
        self.p['b'][n_hid:2 * n_hid] = 3.0          # forget-gate bias (Kratzert et al., 2019)
        self.H = n_hid; self.I = n_in
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.s = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X, keep=True, h0=None, c0=None):
        """X: (B, L, I). Returns y (B, L) and cache."""
        B, L, _ = X.shape; H = self.H
        W, b = self.p['W'], self.p['b']
        h = np.zeros((B, H)) if h0 is None else h0
        c = np.zeros((B, H)) if c0 is None else c0
        hs = np.empty((B, L, H))
        cache = [] if keep else None
        for t in range(L):
            z = np.concatenate([X[:, t], h], 1) @ W + b
            i = sig(z[:, :H]); f = sig(z[:, H:2*H]); g = np.tanh(z[:, 2*H:3*H]); o = sig(z[:, 3*H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            if keep:
                cache.append((h, c, i, f, g, o, tc))
            h, c = h_new, c_new
            hs[:, t] = h
        a = hs @ self.p['v'] + self.p['c'][0]
        y = softplus(a)
        return y, (X, hs, a, cache), (h, c)

    def backward(self, dy, fw):
        """dy: (B, L) dLoss/dy. Returns gradient dict."""
        X, hs, a, cache = fw
        B, L, I = X.shape; H = self.H
        W = self.p['W']
        da = dy * sig(a)                                  # d softplus
        g = {'v': np.einsum('bl,blh->h', da, hs), 'c': np.array([da.sum()]),
             'W': np.zeros_like(W), 'b': np.zeros_like(self.p['b'])}
        dh_out = da[:, :, None] * self.p['v'][None, None, :]
        dh_next = np.zeros((B, H)); dc_next = np.zeros((B, H))
        for t in range(L - 1, -1, -1):
            h_prev, c_prev, i, f, gg, o, tc = cache[t]
            dh = dh_out[:, t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            di = dc * gg; dg = dc * i; df = dc * c_prev
            dc_next = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - gg ** 2), do * o * (1 - o)], 1)
            xin = np.concatenate([X[:, t], h_prev], 1)
            g['W'] += xin.T @ dz
            g['b'] += dz.sum(0)
            dh_next = dz @ W[I:].T
        return g

    def adam(self, g, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
        sc = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in self.p:
            gk = g[k] * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.s[k] = b2 * self.s[k] + (1 - b2) * gk ** 2
            mh = self.m[k] / (1 - b1 ** self.t); sh = self.s[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(sh) + eps)
        return norm

    def get(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set(self, p):
        self.p = {k: v.copy() for k, v in p.items()}
