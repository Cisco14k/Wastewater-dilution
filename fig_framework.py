"""Figure 2: causal graph of the low-flow problem and the three-way evaluation design."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style

fig, ax = plt.subplots(figsize=(8.2, 4.1))
ax.set_xlim(0, 115); ax.set_ylim(0, 56); ax.axis('off')

def box(x, y, w, h, txt, fc, ec='#333', fs=7, bold=False, ls='-'):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                       fc=fc, ec=ec, lw=0.8, ls=ls)
    ax.add_patch(p)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')

def arr(x1, y1, x2, y2, c='#333', ls='-', lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, color=c, lw=lw,
                                 ls=ls, connectionstyle=f'arc3,rad={rad}'))

# ---- panel (a): DAG
ax.text(1, 54, '(a) Causal graph for low-flow-season discharge', fontsize=8, fontweight='bold')
box(10, 44, 15, 6, 'Season,\ncatchment\nattributes', '#eeeeee')
box(10, 29, 15, 6, 'Antecedent\nclimate\n(m−11…m−3)', '#eeeeee')
box(10, 14, 15, 6, 'Storage proxy\nlog Q(m−3)', '#eeeeee')
box(33, 41, 14, 6, 'P anomaly\n(m−2…m)', '#d6e9f8', bold=True)
box(33, 23, 14, 6, 'T anomaly\n(m−2…m)', '#fbe3cf', bold=True)
box(49, 30, 10, 5, 'PET', '#fbe3cf')
box(49, 17, 10, 5, 'Snow\nmelt', '#fbe3cf')
box(49, 44, 10, 5, 'Latent\nstorage', '#ffffff', ls='--')
box(64, 32, 11, 7, 'log Q\nmonth m', '#d9efe3', bold=True)
for (x1, y1, x2, y2) in [(17.5, 44, 25.5, 41.5), (17.5, 43, 25.5, 24), (17.5, 30, 25.5, 40), (17.5, 28.5, 25.5, 23.5),
                         (40, 41, 43.8, 43.5), (40, 23.5, 44, 29), (40, 22.5, 44, 18),
                         (54, 30, 58.3, 31.5), (54, 17.5, 58.3, 30), (54, 44, 58.3, 34.5), (40, 40.5, 58.3, 33.5)]:
    arr(x1, y1, x2, y2)
arr(17.5, 45.5, 58.3, 35, c='#888', rad=-0.25)
arr(17.5, 12.5, 66, 28.2, c='#333', rad=0.3)
arr(17.5, 29.5, 43.8, 44, c='#888', rad=-0.15)
ax.text(30, 2.5, 'Treatments: D$_P$, D$_T$   Confounders X: grey boxes\nθ = ∂E[log Q | do(D), X]/∂D  (partially linear DML)',
        ha='center', fontsize=6.6, color='#333')

# ---- panel (b): evaluation triangle
ax.text(73, 54, '(b) Three estimates of the same causal effect', fontsize=8, fontweight='bold')
box(92, 44, 26, 7, 'θ$_{obs}$: DML on observed\ndischarge (causal benchmark)', '#e5e5e5', fs=6.6)
box(79.5, 22, 17, 7, 'θ$_{M}^{DML}$: DML on\nsimulated discharge', '#f4ecf7', fs=6.6)
box(104, 22, 19, 7, 'ACE$_M$: do-intervention\non model forcing', '#f4ecf7', fs=6.6)
arr(88.6, 23, 93.9, 23, c=style.CAT[4]); arr(93.9, 21, 88.6, 21, c=style.CAT[4])
ax.text(92, 15.3, 'synthetic validation\n(known truth)', fontsize=6, color=style.CAT[4], ha='center')
arr(81, 26.2, 86, 40.2, c=style.CAT[3], lw=1.3)
ax.text(85.5, 32, 'causal discrepancy δ\n(same estimator on both)', fontsize=6.4, color=style.CAT[3], fontweight='bold', ha='left')
ax.text(92, 4, 'Water-quality use: September flow under a hot–dry\nsummer → effluent dilution and in-river concentration',
        ha='center', fontsize=6.6, color='#6b4e16',
        bbox=dict(boxstyle='round,pad=0.4', fc=style.ACCENT_BG, ec='#c9a35d', lw=0.6))
ax.plot([71, 71], [2, 52], color='#cccccc', lw=0.6)
fig.savefig('figs/fig2_framework.png', dpi=300)
