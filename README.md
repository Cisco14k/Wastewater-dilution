# Code: causal benchmark for low-flow models

Python 3 (NumPy, SciPy, pandas, scikit-learn, Matplotlib). Data: CAMELS-FR subset from the airGRdatasets R package v0.2.3
(git clone https://github.com/cran/airGRdatasets; set the SRC path in build_data.py). Map layers: natural-earth-vector and
france-geojson (set GEO in fig_map.py).

Run in this order (from one working directory with subfolders data/, out/, figs/):

1. python build_data.py                      # assemble the 19 catchments
2. python calib_gr.py gr4j 0,1,...,18 ; python calib_gr.py gr4jcn 0,1,...,18
3. python gbm.py
4. python train_lstm.py base {0,1,2} ; python train_lstm.py cr {0,1,2}
5. python analysis.py ; python analysis2.py ; python robustness.py
6. python fig_map.py ; python fig_framework.py ; python fig_results.py ; python fig_supp.py ; python make_tables.py

make_tables.py writes out/numbers.json, from which every number quoted in the manuscript is taken.
