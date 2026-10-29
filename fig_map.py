"""Figure 1: study area. Real geography (Natural Earth 10 m, IGN-derived regions from france-geojson)."""
import os
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from matplotlib.collections import PatchCollection, LineCollection
from common import *
import style

GEO = os.environ.get('GEO_DIR', 'geo')  # holds ne/geojson/*.geojson and fr/regions-version-simplifiee.geojson
sig = hydro_signatures()
sig.to_csv('out/signatures.csv', index=False)

def polys(geom):
    t, c = geom['type'], geom['coordinates']
    if t == 'Polygon':
        return [c]
    if t == 'MultiPolygon':
        return c
    return []

def lines(geom):
    t, c = geom['type'], geom['coordinates']
    if t == 'LineString':
        return [c]
    if t == 'MultiLineString':
        return c
    return []

def draw_polys(ax, feats, fc, ec, lw, z, bbox=None):
    patches = []
    for f in feats:
        for p in polys(f['geometry']):
            ring = np.asarray(p[0])
            if bbox is not None:
                if ring[:, 0].max() < bbox[0] or ring[:, 0].min() > bbox[1] or ring[:, 1].max() < bbox[2] or ring[:, 1].min() > bbox[3]:
                    continue
            patches.append(MPoly(ring, closed=True))
    ax.add_collection(PatchCollection(patches, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

countries = json.load(open(f'{GEO}/ne/geojson/ne_10m_admin_0_countries.geojson'))['features']
regions = json.load(open(f'{GEO}/fr/regions-version-simplifiee.geojson'))['features']
rivers = json.load(open(f'{GEO}/ne/geojson/ne_10m_rivers_europe.geojson'))['features'] + \
         json.load(open(f'{GEO}/ne/geojson/ne_10m_rivers_lake_centerlines.geojson'))['features']
lakes = json.load(open(f'{GEO}/ne/geojson/ne_10m_lakes.geojson'))['features'] + \
        json.load(open(f'{GEO}/ne/geojson/ne_10m_lakes_europe.geojson'))['features']

bbox = (-5.5, 10.0, 41.0, 51.5)
fig = plt.figure(figsize=(7.2, 7.6))
ax = fig.add_axes([0.0, 0.40, 0.62, 0.58])
ax.set_facecolor(style.SEA)
fr = [f for f in countries if f['properties'].get('ADM0_A3') == 'FRA']
oth = [f for f in countries if f['properties'].get('ADM0_A3') != 'FRA']
draw_polys(ax, oth, style.LAND_OTHER, '#b9b9b9', 0.4, 1, bbox)
draw_polys(ax, fr, style.LAND, '#7a7a7a', 0.6, 2, bbox)
draw_polys(ax, regions, 'none', '#bdbdbd', 0.35, 3)
segs = []
for f in rivers:
    for l in lines(f['geometry']):
        a = np.asarray(l)
        if a[:, 0].max() < bbox[0] or a[:, 0].min() > bbox[1] or a[:, 1].max() < bbox[2] or a[:, 1].min() > bbox[3]:
            continue
        segs.append(a)
ax.add_collection(LineCollection(segs, colors=style.RIVER, linewidths=0.55, zorder=4))
draw_polys(ax, lakes, style.RIVER, 'none', 0, 4, bbox)
sz = 18 + 110 * (np.log10(sig.area) - 1) / 2.7
sc = ax.scatter(sig.lon, sig.lat, s=sz, c=sig.bfi, cmap=style.SEQ_CMAP, vmin=0.45, vmax=0.95,
                edgecolor='k', linewidth=0.6, zorder=6)
offs = {'A': (5, 3), 'B': (5, -9), 'G': (-11, -6), 'H': (6, 1), 'P': (6, 2), 'Q': (-12, -5), 'O': (5, 2),
        'K': (-10, -6), 'J': (4, 4), 'E': (6, -3), 'L': (6, -4), 'R': (5, 3), 'S': (6, 0)}
for _, r in sig.iterrows():
    dx, dy = offs.get(r.id, (5, 3))
    ax.annotate(r.id, (r.lon, r.lat), xytext=(dx, dy), textcoords='offset points', fontsize=7.5,
                fontweight='bold', zorder=7, color='#111')
ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
ax.set_aspect(1 / np.cos(np.deg2rad(46.5)))
ax.set_xticks(range(-4, 10, 2)); ax.set_yticks(range(42, 52, 2))
ax.set_xticklabels([f'{abs(x)}°{"W" if x < 0 else "E"}' for x in range(-4, 10, 2)], fontsize=7)
ax.set_yticklabels([f'{y}°N' for y in range(42, 52, 2)], fontsize=7)
ax.text(-0.07, 1.0, '(a)', transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
cax = fig.add_axes([0.09, 0.455, 0.2, 0.012])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.set_label('Baseflow index (BFI)', fontsize=7)
cb.ax.tick_params(labelsize=6.5)
for a_, lab in [(100, '100'), (1000, '1000 km²')]:
    ax.scatter([], [], s=18 + 110 * (np.log10(a_) - 1) / 2.7, c='w', edgecolor='k', label=lab)
ax.legend(loc='upper right', fontsize=6.5, title='Area', title_fontsize=6.5, frameon=True, framealpha=0.9)
# scale bar 200 km
x0, y0 = 2.6, 41.45
dlon = 200 / (111.32 * np.cos(np.deg2rad(y0)))
ax.plot([x0, x0 + dlon], [y0, y0], 'k', lw=2, zorder=8); ax.text(x0 + dlon / 2, y0 + 0.15, '200 km', ha='center', fontsize=6.5)

# inset: Europe locator
ai = ax.inset_axes([-5.45, 43.05, 3.3, 3.2], transform=ax.transData)
eu = (-12, 32, 34, 62)
ai.set_facecolor(style.SEA)
draw_polys(ai, oth, style.LAND_OTHER, '#9a9a9a', 0.2, 1, eu); draw_polys(ai, fr, '#e0a458', '#555', 0.3, 2, eu)
ai.set_xlim(eu[0], eu[1]); ai.set_ylim(eu[2], eu[3]); ai.set_xticks([]); ai.set_yticks([])
ai.set_aspect(1 / np.cos(np.deg2rad(48)))

# (b) climate space
bx = fig.add_axes([0.71, 0.70, 0.27, 0.26])
bx.scatter(sig.aridity, sig.runoff_ratio, s=26, c=sig.snow_frac, cmap='Blues', vmin=0, vmax=0.4, edgecolor='k', lw=0.5)
for _, r in sig.iterrows():
    bx.annotate(r.id, (r.aridity, r.runoff_ratio), xytext=((2.5, -6) if r.id in 'QBL' else (2.5, 1.5)), textcoords='offset points', fontsize=6)
bx.set_xlabel('Aridity index PET/P (–)', fontsize=7); bx.set_ylabel('Runoff ratio Q/P (–)', fontsize=7)
bx.tick_params(labelsize=6.5)
bx.text(-0.25, 1.02, '(b)', transform=bx.transAxes, fontsize=9, fontweight='bold', va='top')
# (c) regimes
cx = fig.add_axes([0.71, 0.405, 0.27, 0.22])
mon = DATES.month.values
for i in range(NC):
    m = TRAIN
    mq = np.array([np.nanmean(Q[i, m & (mon == k)]) for k in range(1, 13)])
    c = style.CAT[2] if sig.snow_frac[i] > 0.10 else '#8c8c8c'
    cx.plot(range(1, 13), mq / mq.mean(), color=c, lw=1.6 if sig.snow_frac[i] > 0.10 else 0.8, alpha=0.9)
cx.axvspan(6, 10, color=style.ACCENT_BG, zorder=0)
cx.text(8, 0.12, 'low-flow season', fontsize=6, ha='center', va='bottom', color='#6b4e16')
cx.set_xticks(range(1, 13)); cx.set_xticklabels('JFMAMJJASOND', fontsize=6.5)
cx.set_ylabel('Q / annual mean (–)', fontsize=7); cx.tick_params(labelsize=6.5)
cx.text(-0.25, 1.02, '(c)', transform=cx.transAxes, fontsize=9, fontweight='bold', va='top')
cx.plot([], [], color=style.CAT[2], lw=1.6, label='snow fraction > 0.10'); cx.plot([], [], color='#8c8c8c', lw=0.8, label='other')
cx.legend(fontsize=5.8, loc='upper left', bbox_to_anchor=(0.0, 0.88), frameon=False)

# (d) chronological design and data availability
dx = fig.add_axes([0.09, 0.06, 0.89, 0.26])
avail = np.isfinite(Q).astype(float)
yrs = DATES.year.values + (DATES.dayofyear.values - 1) / 365.25
dx.imshow(avail, aspect='auto', cmap=matplotlib.colors.ListedColormap(['#222222', '#f4f4f4']),
          extent=[yrs[0], yrs[-1] + 1 / 365, NC - 0.5, -0.5], interpolation='nearest')
blocks = [(1999, 2000, 'warm-up', '#bbbbbb'), (2000, 2012, 'training (2000–2011)', style.CAT[0]),
          (2012, 2015, 'validation', style.CAT[1]), (2015, 2019, 'test (2015–2018)', style.CAT[3])]
for a, b, lab, c in blocks:
    dx.axvspan(a, b, ymin=1.0, ymax=1.09, color=c, clip_on=False)
    dx.text((a + b) / 2, -1.55, lab, ha='center', va='center', fontsize=6.5, color='white' if c != '#bbbbbb' else 'k',
            fontweight='bold', clip_on=False)
    dx.axvline(a, color='k', lw=0.4, ls=':')
for yy in range(2000, 2019):
    for mm in (6, 10):
        pass
dx.set_yticks(range(NC)); dx.set_yticklabels(IDS, fontsize=6)
dx.set_xlim(1999, 2019); dx.set_xticks(range(2000, 2020, 2)); dx.tick_params(labelsize=6.5)
dx.set_xlabel('Year', fontsize=7)
dx.text(-0.06, 1.09, '(d)', transform=dx.transAxes, fontsize=9, fontweight='bold', va='top')
fig.savefig('figs/fig1_study_area.png', dpi=300)
print(sig[['id', 'area', 'bfi', 'runoff_ratio', 'aridity', 'snow_frac']].round(3).to_string())
