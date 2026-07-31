#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_orthogonality.py -- can O*NET information predict each external variable?
High R^2 => that external variable is collinear with skills (status-consistent);
near-zero R^2 => it is orthogonal, a dimension of its own.

Regressor: Ridge (RidgeCV to pick alpha). Ridge has a closed-form solution, is
stable under the heavy collinearity of O*NET features, and has no convergence
issues. Honest generalization R^2 comes from a plain 5-fold split done ONCE in
this function (inside each fold RidgeCV picks alpha by its own fast leave-one-
out CV) -- no nested cross_val_score wrapping.

All feature matrices are median-filled and z-scored before fitting.

For each target we report:
  A) all raw O*NET features
  B) leading O*NET PCA axes (3/6/10)
  C) per-block R^2 (which block tracks the target)
  D) incremental R^2 adding workctx_cx, ct, ete, jobzone to classic skills
  E) ETE distributions alone -> each external variable

Targets: union coverage, wage median, prestige, log employment.
Reads master_wide.xlsx. Terminal output only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

MASTER = Path("output/master_clean.xlsx")
SEED = 0
ALPHAS = np.logspace(-2, 4, 25)      # ridge penalty grid

CLASSIC_PREFIXES = ["skills_", "abilities_", "knowledge_", "workact_"]
CX_PREFIX = ["workctx_cx"]
CT_PREFIX = ["workctx_ct"]
ETE_PREFIXES = ["ete_"]
JOBZONE = ["jobzone"]


def cols_with(df, prefixes):
    out = []
    for c in df.columns:
        base = c.split("__", 1)[0]
        if any(c == p or c.startswith(p) or base.startswith(p.rstrip("_"))
               for p in prefixes):
            out.append(c)
    return out


def ridge_r2(X, y):
    """Honest 5-fold R^2 with RidgeCV (alpha chosen inside each fold).
    X is median-filled then z-scored (scaler fit on train fold only)."""
    Xf = pd.DataFrame(X).apply(lambda s: s.fillna(s.median())).values
    ok = ~np.isnan(y)
    Xf, yy = Xf[ok], y[ok]
    if len(yy) < 60 or Xf.shape[1] == 0:
        return np.nan
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds, truth = [], []
    for tr, te in kf.split(Xf):
        sc = StandardScaler().fit(Xf[tr])
        model = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xf[tr]), yy[tr])
        preds.append(model.predict(sc.transform(Xf[te])))
        truth.append(yy[te])
    return r2_score(np.concatenate(truth), np.concatenate(preds))


def main():
    df = pd.read_excel(MASTER).set_index("onet_soc")

    # targets already imputed in clean_master.py
    targets = {"union_cov_pct":  df["ext_union_cov_pct"].values,
               "wage_median":    df["ext_wage_median"].values,
               "prestige":       df["ext_prestige"].values,
               "log_employment": df["ext_employment_log"].values}

    onet_cols = [c for c in df.columns if "__" in c]
    Xonet = df[onet_cols]

    Xf = Xonet
    Xz = StandardScaler().fit_transform(Xf.values)
    pcs = PCA(n_components=10, random_state=SEED).fit_transform(Xz)

    classic = cols_with(Xonet, CLASSIC_PREFIXES)
    cx = cols_with(Xonet, CX_PREFIX)
    ct = cols_with(Xonet, CT_PREFIX)
    ete = cols_with(Xonet, ETE_PREFIXES)
    jz = [c for c in Xonet.columns if c == "jobzone"]
    print(f"feature counts: classic={len(classic)}, cx={len(cx)}, ct={len(ct)}, "
          f"ete={len(ete)}, jobzone={len(jz)}")

    for name, y in targets.items():
        print("\n" + "=" * 70)
        print(f"TARGET: {name}   (n={int((~np.isnan(y)).sum())})")
        print("=" * 70)

        print(f"  [A] all O*NET features ({len(onet_cols)} cols): "
              f"R^2 = {ridge_r2(Xonet.values, y):.3f}")
        for npc in (3, 6, 10):
            print(f"  [B] first {npc:2d} O*NET PCA axes         : "
                  f"R^2 = {ridge_r2(pcs[:, :npc], y):.3f}")
        print("  [C] per-block alone -> target:")
        for bn, cols in [("classic skills", classic), ("workctx_cx", cx),
                         ("workctx_ct", ct), ("ete", ete), ("jobzone", jz)]:
            if cols:
                print(f"       {bn:16s} ({len(cols):3d} cols): "
                      f"R^2 = {ridge_r2(Xonet[cols].values, y):.3f}")
        print("  [D] incremental (add blocks cumulatively):")
        cum = []
        for bn, cols in [("classic", classic), ("+workctx_cx", cx),
                         ("+workctx_ct", ct), ("+ete", ete), ("+jobzone", jz)]:
            cum += cols
            if cum:
                print(f"       {bn:16s} ({len(cum):3d} cols): "
                      f"R^2 = {ridge_r2(Xonet[cum].values, y):.3f}")
        print("  [E] ETE distributions alone -> target:")
        if ete:
            print(f"       ete ({len(ete)} cols): R^2 = {ridge_r2(Xonet[ete].values, y):.3f}")


if __name__ == "__main__":
    main()