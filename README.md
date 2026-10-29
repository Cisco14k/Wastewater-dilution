# Wastewater-dilution: causal benchmark for low-flow models

Code for the article *"Does a good fit mean a correct response? An observational causal benchmark for streamflow
models used to assess wastewater dilution in dry summers"* (Yeboah et al., submitted to Environmental Modelling & Software).

The code estimates the causal response of late-summer discharge to seasonal precipitation and temperature anomalies
with double machine learning (DML), builds six rainfall–runoff models (GR4J, GR4J-CemaNeige, two gradient-boosting
models, two LSTMs), compares each model's response with the observational benchmark, and propagates the differences
into effluent dilution under a hot–dry summer.

## Requirements

* Python 3.10 or later with NumPy, SciPy, pandas, scikit-learn and Matplotlib (`pip install -r requirements.txt`).
* No R installation and no GPU are needed: the R data files are read by `rdaread.py`, and the LSTMs are written in NumPy.
* Any standard desktop or laptop computer. GR4J calibration takes about 40 s per catchment; the LSTMs take longest.

## Data (all open)

| Data | Source | Licence |
|---|---|---|
| 19 CAMELS-FR catchments (daily P, T, PET, Q, 1999–2018) | airGRdatasets R package v0.2.3, https://doi.org/10.32614/CRAN.package.airGRdatasets (mirror: https://github.com/cran/airGRdatasets) | CC BY 4.0 |
| Full CAMELS-FR dataset (reference) | https://doi.org/10.57745/WH7FJR | Open access |
| Rivers, lakes, coastlines (Fig. 1 only) | Natural Earth 1:10 m, https://github.com/nvkelso/natural-earth-vector (folder `geojson/`) | Public domain |
| French regions (Fig. 1 only) | france-geojson, https://github.com/gregoiredavid/france-geojson | Licence Ouverte (Etalab) |

## Set-up

```bash
git clone https://github.com/Cisco14k/Wastewater-dilution.git
git clone --depth 1 https://github.com/cran/airGRdatasets.git
pip install -r Wastewater-dilution/requirements.txt
mkdir work && cd work && mkdir data out figs
export AIRGR_DATA=../airGRdatasets/data      # folder with the .rda files
export GEO_DIR=/path/to/geo                  # only for fig_map.py: geo/ne/geojson/*.geojson, geo/fr/regions-version-simplifiee.geojson
```

## Run (from the `work` folder; scripts are called with their path, e.g. `python ../Wastewater-dilution/build_data.py`)

1. `build_data.py` – assemble the 19 catchments into `data/camels_fr19.npz`
2. `calib_gr.py gr4j 0,1,...,18` and `calib_gr.py gr4jcn 0,1,...,18` – calibrate the conceptual models
3. `gbm.py` – train the two gradient-boosting models
4. `train_lstm.py base {0,1,2}` and `train_lstm.py cr {0,1,2}` – train the LSTMs (three seeds each)
5. `analysis.py`, `analysis2.py`, `robustness.py` – DML benchmark, model worlds, discrepancies, scenario, robustness
6. `fig_map.py`, `fig_framework.py`, `fig_results.py`, `fig_supp.py`, `make_tables.py` – figures and tables

`make_tables.py` writes `out/numbers.json`, from which every number quoted in the manuscript is taken.

## Files

| File | Purpose |
|---|---|
| `common.py` | data loading, chronological split (warm-up 1999, train 2000–2011, validation 2012–2014, test 2015–2018) |
| `hydro.py` | GR4J, CemaNeige, Oudin PET and skill scores |
| `dml.py` | partially linear DML with year-grouped cross-fitting and cluster-robust errors |
| `interventions.py` | do-interventions on model forcing and the hot–dry scenario |
| `gbm.py`, `lstm_np.py`, `train_lstm.py` | learned models |
| `rdaread.py` | reader for R `.rda` files |

## Licence

MIT (see `LICENSE`). Contact: Emmanuel Yeboah, emmanuelyeboah@henu.edu.cn
