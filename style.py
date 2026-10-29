"""Common figure style (colour-blind-safe categorical palette, Okabe-Ito based)."""
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
CAT = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#000000']
SEA = '#e8f1f8'; LAND = '#fbfaf6'; LAND_OTHER = '#ededed'; RIVER = '#5d9cd1'; ACCENT_BG = '#fbefd5'
SEQ_CMAP = 'viridis'
MODEL_COL = {'CLIM': '#9e9e9e', 'GR4J': '#56B4E9', 'GR4J-CN': '#0072B2', 'GBM-A': '#E69F00',
             'GBM-C': '#D55E00', 'LSTM': '#CC79A7', 'LSTM-CR': '#7B3294', 'OBS-DML': '#000000'}
