#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reliability_vs_axes.py -- are the items the six axes explain poorly simply the
unreliable ones?

For each of the 216 feature columns of output/master_clean.csv: the column is
left out, the first six principal components of the other 215 (z-scored) are
computed, and the column is predicted from those six scores and, separately,
from all 215 other columns, by ridge regression (five-fold cross-validated R2,
penalty chosen in-fold). Each R2 is set against the item's reliability
1 - e_j from output/rating_error_share.xlsx (run rating_error_share.py first).
R2 / reliability is the share of the item's reliable variance that is
predicted. It can exceed 1 when rating errors are correlated within a
questionnaire, because shared error is predictable from the other items.

Run from the project root:  python supplementary/reliability_vs_axes.py
Writes output/reliability_vs_axes.csv.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
pre = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__", "workctx_cx__")
c = pd.read_csv("output/master_clean.csv")
cols = [x for x in c.columns if x.startswith(pre)]
Z = c[cols].to_numpy(float); Z = (Z - Z.mean(0)) / Z.std(0)
def cv_r2(X, y):
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 30))).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rows = []
for j, col in enumerate(cols):
    rest = np.delete(Z, j, axis=1); rc = rest - rest.mean(0)
    _, _, vt = np.linalg.svd(rc, full_matrices=False)
    rows.append({"item": col, "R2 six axes": cv_r2(rc @ vt[:6].T, Z[:, j]), "R2 other 215": cv_r2(rest, Z[:, j])})
r = pd.DataFrame(rows)
rel = pd.read_excel("output/rating_error_share.xlsx", sheet_name="items")[["item", "block", "reliability 1 - e_j"]]
d = r.merge(rel, on="item"); assert len(d) == 216
d["six axes / reliability"] = d["R2 six axes"] / d["reliability 1 - e_j"]
d["other 215 / reliability"] = d["R2 other 215"] / d["reliability 1 - e_j"]
print("items:", len(d))
print("Spearman(reliability, R2 six axes) = %.2f; Pearson = %.2f" % (spearmanr(d["reliability 1 - e_j"], d["R2 six axes"])[0], np.corrcoef(d["reliability 1 - e_j"], d["R2 six axes"])[0, 1]))
print("Spearman(reliability, R2 other 215) = %.2f" % spearmanr(d["reliability 1 - e_j"], d["R2 other 215"])[0])
d["rel band"] = pd.cut(d["reliability 1 - e_j"], [0, .7, .8, .9, 1], labels=["<0.7", "0.7-0.8", "0.8-0.9", ">=0.9"])
print(d.groupby("rel band", observed=True)[["R2 six axes", "R2 other 215", "six axes / reliability"]].agg(["count", "median"]).round(2).to_string())
print("\nby block, medians:")
print(d.groupby("block")[["reliability 1 - e_j", "R2 six axes", "six axes / reliability", "R2 other 215", "other 215 / reliability"]].median().round(2).to_string())
low = d.sort_values("R2 six axes").head(15)
print("\n15 items least explained by the six axes:")
print(low[["item", "reliability 1 - e_j", "R2 six axes", "six axes / reliability", "R2 other 215"]].assign(item=lambda x: x["item"].str.slice(0, 60)).round(2).to_string(index=False))
d.to_csv("output/reliability_vs_axes.csv", index=False)