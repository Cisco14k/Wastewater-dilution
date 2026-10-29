"""Figures 3-7 from out/results.json and out/skill_test.csv."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import style
from style import MODEL_COL

R = json.load(open('out/results.json'))
sk = pd.read_csv('out/skill_test.csv')
sig = pd.read_csv('out/signatures.csv')
MODELS = list(R['causal'].keys())
ALL = ['CLIM'] + MODELS
IDS = R['ids']
DR0 = 10.0  # baseline September dilution ratio Q_river / Q_effluent

def amp(dlogq, dr0=DR0):
    """in-river concentration amplification for a conservative effluent constituent (negligible background)."""
    return (dr0 + 1) / (dr0 * np.exp(np.asarray(dlogq)) + 1)

# ---------------------------------------------------------------- Fig 3: predictive skill
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6))
for ax, (col, lab, lim) in zip(axs, [('NSE', 'NSE (–)', (-0.6, 1)), ('logNSE', 'NSE on log Q (–)', (-0.6, 1)),
                                     ('Q95_bias', 'Q95 bias (%)', (-100, 150)), ('lowseason_bias', 'Aug–Oct flow bias (%)', (-100, 150))]):
    data = [sk[sk.model == m][col].values for m in ALL]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color='k', lw=1))
    for p, m in zip(bp['boxes'], ALL):
        p.set_facecolor(MODEL_COL[m]); p.set_alpha(0.55); p.set_edgecolor('#333'); p.set_linewidth(0.6)
    for j, d in enumerate(data):
        x = np.full(len(d), j + 1) + np.linspace(-0.18, 0.18, len(d))
        dd = np.clip(d, lim[0], lim[1])
        ax.scatter(x, dd, s=4, color='#222', zorder=3, lw=0)
    ax.set_xticks(range(1, len(ALL) + 1)); ax.set_xticklabels(ALL, rotation=60, fontsize=6.3, ha='right')
    ax.set_ylim(*lim); ax.set_ylabel(lab, fontsize=7)
    if 'bias' in col:
        ax.axhline(0, color='#777', lw=0.6, ls='--')
for ax, t in zip(axs, 'abcd'):
    ax.text(-0.3, 1.04, f'({t})', transform=ax.transAxes, fontsize=8.5, fontweight='bold')
fig.tight_layout(w_pad=0.6)
fig.savefig('figs/fig3_skill.png', dpi=300)

# ---------------------------------------------------------------- Fig 4: synthetic validation of the DML estimator
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, k, lab in [(axs[0], 'P', 'precipitation effect θ$_P$ (–)'), (axs[1], 'T', 'temperature effect θ$_T$ (°C$^{-1}$)')]:
    allx, ally = [], []
    for m in MODELS:
        c = R['causal'][m]
        x = np.array(c[f'ace_{k}_c']); y = np.array(c[f'dml_sim_{k}_c'])
        ax.scatter(x, y, s=10, color=MODEL_COL[m], alpha=0.75, lw=0, label=m)
        ax.scatter(c[f'ace_{k}_w'], c[f'dml_sim_{k}'], s=60, marker='D', color=MODEL_COL[m], edgecolor='k', lw=0.7, zorder=5)
        allx += list(x); ally += list(y)
    lo, hi = min(allx + ally), max(allx + ally); pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#555', lw=0.7, ls='--')
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f'True model effect (do-intervention), {lab}', fontsize=6.8)
    ax.set_ylabel('DML estimate from simulated discharge', fontsize=6.8)
    r = np.corrcoef(allx, ally)[0, 1]
    ax.text(0.04, 0.93, f'catchment level: r = {r:.2f}', transform=ax.transAxes, fontsize=6.6)
axs[0].legend(fontsize=6, frameon=False, loc='lower right')
axs[0].text(-0.2, 1.03, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold')
axs[1].text(-0.2, 1.03, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold')
fig.tight_layout()
fig.savefig('figs/fig4_synthetic_validation.png', dpi=300)

# ================================================================ like-for-like discrepancy figures
D = json.load(open('out/discrepancy.json'))

# ---------------------------------------------------------------- Fig 5: observed benchmark + pooled discrepancies
fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.34)
for j, (k, mod, xl, yl) in enumerate([('P', 'bfi', 'Baseflow index (–)', 'Observed θ$_P$ (–)'),
                                      ('T', 'snow_frac', 'Snow fraction (–)', 'Observed θ$_T$ (°C$^{-1}$)')]):
    ax = fig.add_subplot(gs[0, j])
    x = sig[mod].values
    o = np.array(R[f'th{k}_obs_c']); ci = np.array(R[f'th{k}_obs_c_ci'])
    ax.errorbar(x, o, yerr=[o - ci[0], ci[1] - o], fmt='s', color='k', ms=3.5, lw=0.7, capsize=0)
    lab_pts = {'P': {'O': (-10, 0), 'L': (5, 0), 'Q': (5, -2), 'P': (5, 0), 'D': (-11, 3), 'E': (5, 0)},
               'T': {'O': (5, 0), 'Q': (-11, 0), 'P': (5, 0)}}[k]
    for xi, yi, lab in zip(x, o, IDS):
        if lab in lab_pts:
            ax.annotate(lab, (xi, yi), xytext=lab_pts[lab], textcoords='offset points', fontsize=6, color='#333', va='center')
    if k == 'T':
        ax.axhline(0, color='#999', lw=0.5, ls=':')
    ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
    ax.text(-0.22, 1.04, f'({"ab"[j]})', transform=ax.transAxes, fontsize=8.5, fontweight='bold')
for j, (k, yl) in enumerate([('P', 'δ$_P$ = θ$_{P,obs}$ − θ$_{P,M}$ (–)'), ('T', 'δ$_T$ = θ$_{T,obs}$ − θ$_{T,M}$ (°C$^{-1}$)')]):
    ax = fig.add_subplot(gs[1, j])
    for i, m in enumerate(MODELS):
        d, s_ = D[m][f'delta_{k}'], D[m][f'delta_{k}_se']
        ax.errorbar(i, d, yerr=1.96 * s_, fmt='o', color=MODEL_COL[m], ms=6, mec='k', mew=0.6, lw=1.2, capsize=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=40, ha='right', fontsize=6.5)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylabel(yl, fontsize=7)
    lo, hi = ax.get_ylim()
    ax.text(0.98, 0.04, 'model too sensitive ↓', transform=ax.transAxes, ha='right', fontsize=6, color='#666') if k == 'P' else \
        ax.text(0.98, 0.96, 'model too sensitive to warming ↑', transform=ax.transAxes, ha='right', va='top', fontsize=6, color='#666')
    ax.text(-0.22, 1.04, f'({"cd"[j]})', transform=ax.transAxes, fontsize=8.5, fontweight='bold')
fig.savefig('figs/fig5_causal_benchmark.png', dpi=300)

# ---------------------------------------------------------------- Fig 6: catchment-level discrepancy matrix
order = np.argsort(sig.bfi.values)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw=dict(wspace=0.55))
for ax, k, lab, vmax in [(axs[0], 'P', 'δ$_P$ (–)', 0.6), (axs[1], 'T', 'δ$_T$ (°C$^{-1}$)', 0.06)]:
    M = np.array([D[m][f'delta_{k}_c'] for m in MODELS]).T[order]
    lo = np.array([D[m][f'delta_{k}_c_ci'][0] for m in MODELS]).T[order]
    hi = np.array([D[m][f'delta_{k}_c_ci'][1] for m in MODELS]).T[order]
    sigm = (lo > 0) | (hi < 0)
    im = ax.imshow(M, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if sigm[r, c]:
                ax.plot(c, r, marker='o', ms=2.6, color='k')
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=6.5)
    ax.set_yticks(range(len(IDS))); ax.set_yticklabels([f'{IDS[i]}  ({sig.bfi.values[i]:.2f})' for i in order], fontsize=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03); cb.set_label(lab, fontsize=7); cb.ax.tick_params(labelsize=6)
axs[0].set_ylabel('Catchment (BFI), sorted by baseflow index', fontsize=7)
axs[0].text(-0.42, 1.02, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold')
axs[1].text(-0.42, 1.02, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold')
fig.savefig('figs/fig6_discrepancy_matrix.png', dpi=300)

# ---------------------------------------------------------------- Fig 7: skill vs causal fidelity, and dilution consequence
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.38))
ax = axs[0]
for m in MODELS:
    x = sk[sk.model == m].NSE.median()
    y = D[m]['mean_abs_delta_P']
    ax.scatter(x, y, s=34, color=MODEL_COL[m], edgecolor='k', lw=0.6, zorder=4)
ax.set_yscale('log')
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1, 0.2, 0.3])); ax.yaxis.set_major_formatter(FixedFormatter(['0.02', '0.03', '0.05', '0.10', '0.20', '0.30'])); ax.yaxis.set_minor_locator(NullLocator())
offs = {'GR4J': (6, -2), 'GR4J-CN': (6, -2), 'GBM-A': (6, -2), 'GBM-C': (6, -2), 'LSTM': (-8, 7), 'LSTM-CR': (-46, 0)}
for m in MODELS:
    x = sk[sk.model == m].NSE.median(); y = D[m]['mean_abs_delta_P']
    ax.annotate(m, (x, y), xytext=offs[m], textcoords='offset points', fontsize=6.5, va='center')
ax.set_xlabel('Median test NSE (–)', fontsize=7); ax.set_ylabel('Mean |δ$_P$| across catchments (–)', fontsize=7)
xl = ax.get_xlim(); ax.set_xlim(xl[0] - 0.01, xl[1] + 0.012)
ax.text(-0.25, 1.03, '(a)', transform=ax.transAxes, fontsize=8.5, fontweight='bold')
ax = axs[1]
err = [100 * (np.array(D[m]['amp_model']) - np.array(D[m]['amp_obsimplied'])) for m in MODELS]
bp = ax.boxplot(err, orientation='horizontal', patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color='k'))
for p, m in zip(bp['boxes'], MODELS):
    p.set_facecolor(MODEL_COL[m]); p.set_alpha(0.65)
for j, e in enumerate(err):
    ax.scatter(e, np.full(len(e), j + 1) + np.linspace(-0.2, 0.2, len(e)), s=4, color='#222', lw=0, zorder=3)
ax.set_yticks(range(1, len(MODELS) + 1)); ax.set_yticklabels(MODELS, fontsize=6.5)
ax.axvline(0, color='#555', lw=0.7, ls='--')
ax.set_xlabel('Error in projected September concentration\nincrease, model − causally corrected (% points)', fontsize=7)
ax.text(-0.3, 1.03, '(b)', transform=ax.transAxes, fontsize=8.5, fontweight='bold')
fig.savefig('figs/fig7_fidelity_dilution.png', dpi=300)
print('figures written')
