#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wage_regression.py -- (personal-interest, not part of the study) what predicts a
job's pay, and what pays ON TOP OF skill?

  1. regress log(median wage) on skills alone (Ridge, honest CV R^2) -> how much
     of pay is 'explained by ability'.
  2. add institutional/economic variables (union, self-employment, separation
     rates, employment) and ETE -> how much EXTRA pay they explain beyond skills.
  3. residual analysis: log-wage minus skill-predicted log-wage. High positive
     residual = paid more than skills predict (rent / bargaining / scarcity);
     high negative = paid less than skills predict. List the top occupations at
     each end -- a personal map of 'where the money is relative to ability'.

Reads master_wide.xlsx. Terminal + wage_residuals.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

MASTER = Path("output/master_clean.xlsx")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
SEED = 0
ALPHAS = np.logspace(-2, 4, 25)


def ridge_oos(X, y):
    """out-of-sample predictions + R^2 via 5-fold; scaler fit on train only."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred, r2_score(y, pred)


def main():
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)

    # target: log median wage (already imputed in clean_master.py)
    med = df["ext_wage_median"]
    y = df["ext_wage_level_log"].values

    skill = [c for c in df.columns if "__" in c and not c.startswith("ete_")]
    ete = [c for c in df.columns if c.startswith("ete_")]
    Xs = df[skill].apply(pd.to_numeric, errors="coerce").fillna(df[skill].median()).values

    inst_cols = ["ext_union_cov_pct", "ext_self_employed_pct",
                 "ext_sep_exit_rate", "ext_sep_transfer_rate", "ext_employment"]
    inst_cols = [c for c in inst_cols if c in df.columns]
    I = df[inst_cols].apply(pd.to_numeric, errors="coerce")
    if "ext_employment" in I:
        I["ext_employment"] = np.log1p(I["ext_employment"])
    I = I.fillna(I.median()).values
    Xete = df[ete].apply(pd.to_numeric, errors="coerce").fillna(df[ete].median()).values

    print("=" * 60)
    print("WHAT PREDICTS (LOG) MEDIAN WAGE?  honest 5-fold R^2")
    print("=" * 60)
    pred_s, r2_s = ridge_oos(Xs, y)
    print(f"  skills only                : R^2 = {r2_s:.3f}")
    _, r2_si = ridge_oos(np.hstack([Xs, I]), y)
    print(f"  skills + institutions      : R^2 = {r2_si:.3f}  "
          f"(+{r2_si - r2_s:.3f})")
    _, r2_se = ridge_oos(np.hstack([Xs, Xete]), y)
    print(f"  skills + ETE               : R^2 = {r2_se:.3f}  "
          f"(+{r2_se - r2_s:.3f})")
    _, r2_all = ridge_oos(np.hstack([Xs, I, Xete]), y)
    print(f"  skills + institutions + ETE: R^2 = {r2_all:.3f}  "
          f"(+{r2_all - r2_s:.3f})")

    # individual institutional incremental value (each alone, added to skills)
    print("\n  incremental R^2 of each institutional variable over skills:")
    for j, c in enumerate(inst_cols):
        _, r2c = ridge_oos(np.hstack([Xs, I[:, [j]]]), y)
        print(f"     +{c:24s}: {r2c - r2_s:+.3f}")

    # residuals from the skills-only model = pay beyond ability
    resid = y - pred_s
    R = pd.DataFrame({"title": title.values, "log_wage": y,
                      "skill_pred": pred_s, "residual": resid,
                      "wage_median": med.values}, index=df.index)
    for c in inst_cols:
        R[c] = pd.to_numeric(df[c], errors="coerce").values

    print("\n" + "=" * 60)
    print("PAID MORE THAN SKILLS PREDICT (top +residual)")
    print("  = rent / scarcity / bargaining power / licensing")
    print("=" * 60)
    top = R.sort_values("residual", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r['residual']:+.2f}  ${r['wage_median']:>7,.0f}  {r['title'][:50]}")

    print("\n" + "=" * 60)
    print("PAID LESS THAN SKILLS PREDICT (top -residual)")
    print("  = crowded / low bargaining / mission-driven / vocational")
    print("=" * 60)
    bot = R.sort_values("residual").head(15)
    for _, r in bot.iterrows():
        print(f"  {r['residual']:+.2f}  ${r['wage_median']:>7,.0f}  {r['title'][:50]}")

    # do institutions correlate with the residual? (is 'extra pay' institutional?)
    print("\n" + "=" * 60)
    print("does the residual (pay beyond skill) track institutions?")
    print("=" * 60)
    for c in inst_cols:
        v = pd.to_numeric(df[c], errors="coerce").values
        ok = ~np.isnan(v)
        r = np.corrcoef(resid[ok], v[ok])[0, 1]
        print(f"  corr(residual, {c:24s}) = {r:+.3f}")

    R.to_csv(OUTDIR / "wage_residuals.csv")
    print(f"\n[ok] wrote wage_residuals.csv")


if __name__ == "__main__":
    main()