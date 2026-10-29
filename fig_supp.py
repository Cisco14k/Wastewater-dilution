"""Supplementary figures: S1 test-period hydrographs; S2 per-catchment test NSE heatmap."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import *
import style
from style import MODEL_COL

S = np.load('out/sims_base.npz')
sims = {k.replace('_', '-'): S[k] for k in S.files}
MODELS = ['GR4J-CN', 'GBM-A', 'GBM-C', 'LSTM', 'LSTM-CR']
pick = [('E', 'Nièvre (chalk, BFI 0.94)'), ('P', 'Durance (alpine, snow 43%)'), ('L', 'Arroux (rain-fed, BFI 0.45)')]
fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
win = (DATES >= '2017-05-01') & (DATES <= '2018-12-31')
for ax, (cid, lab) in zip(axs, pick):
    i = IDS.index(cid)
    ax.fill_between(DATES[win], 0, Q[i, win], color='#cfcfcf', lw=0, label='observed', step='mid')
    for m in MODELS:
        ax.plot(DATES[win], sims[m][i, win], color=MODEL_COL[m], lw=0.8, label=m)
    ax.set_yscale('log'); ax.set_ylabel('Q (mm d$^{-1}$)', fontsize=7)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(0.005, 1.03, f'({"abc"[axs.tolist().index(ax)]}) {cid}: {lab}', transform=ax.transAxes, fontsize=7.5, fontweight='bold')
    for yy in (2017, 2018):
        ax.axvspan(pd.Timestamp(f'{yy}-07-01'), pd.Timestamp(f'{yy}-10-31'), color=style.ACCENT_BG, zorder=0)
axs[0].legend(fontsize=6, ncol=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.32))
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.tight_layout(h_pad=1.6)
fig.savefig('figs/figS1_hydrographs.png', dpi=300)

sk = pd.read_csv('out/skill_test.csv')
ALL = ['CLIM', 'GR4J', 'GR4J-CN', 'GBM-A', 'GBM-C', 'LSTM', 'LSTM-CR']
piv = sk.pivot(index='id', columns='model', values='NSE').loc[IDS, ALL]
fig, ax = plt.subplots(figsize=(4.6, 5.6))
im = ax.imshow(np.clip(piv.values, -0.5, 1), cmap='RdYlBu', vmin=-0.5, vmax=1, aspect='auto')
for r in range(piv.shape[0]):
    for c in range(piv.shape[1]):
        v = piv.values[r, c]
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=5.4, color='k' if 0.0 < v < 0.85 else 'w')
ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL, rotation=45, ha='right', fontsize=6.5)
ax.set_yticks(range(len(IDS))); ax.set_yticklabels(IDS, fontsize=6.5)
cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label('Test NSE (clipped at −0.5)', fontsize=7); cb.ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig('figs/figS2_nse_heatmap.png', dpi=300)
print('ok')
