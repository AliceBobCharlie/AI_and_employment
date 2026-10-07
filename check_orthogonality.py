#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_orthogonality.py -- how much of each labour-market variable can the
description of the work predict? (Section 5.5)

Each target is regressed on the 216 feature columns by ridge regression, and
scored by five-fold cross-validated R2: the penalty is chosen and the scaler
fitted inside each training fold. A high R2 means the variable restates
occupational content; a low one means it carries information the features do
not contain.

Values that clean_master.py filled in are left out of each regression rather
than predicted, so every R2 is computed on published values only.

Sections:
  A  Table 5.5: R2 from all 216 features, and from each feature block alone
  B  the education, training and experience block column by column: median R2,
     counts below 0.3 and above 0.6, and the variance-weighted R2 of the block
  C  descriptive figures quoted in Section 5.5: employment, self-employment
     and union coverage, and union coverage's loading on the leading direction
     of the eight labour-market variables

Reads output/master.csv (to tell published values from filled ones) and
output/master_clean.csv. Writes output/orthogonality.xlsx and prints every sheet.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = Path("output/master.csv")
MASTER = Path("output/master_clean.csv")
OUT = Path("output/orthogonality.xlsx")
SEED = 0
ALPHAS = np.logspace(-2, 5, 30)

# The feature matrix: Importance for four blocks, the Context scale for Work Context.
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")

# The eight labour-market variables, each with the raw columns it is computed
# from (used to tell published values from filled ones).
WAGES = ["ext_wage_p10", "ext_wage_median", "ext_wage_p90"]
LABOUR = {
    "ext_wage_level_log": WAGES,
    "ext_wage_disp_p90p10": WAGES,
    "ext_prestige": ["ext_prestige"],
    "ext_sep_exit_rate": ["ext_sep_exit_rate"],
    "ext_sep_transfer_rate": ["ext_sep_transfer_rate"],
    "ext_union_cov_pct": ["ext_union_cov_pct"],
    "ext_self_employed_pct": ["ext_self_employed_pct"],
    "ext_employment_log": ["ext_employment"],
}

TARGETS = {
    "ext_prestige": "occupational prestige",
    "ext_wage_level_log": "log median wage",
    "ext_sep_exit_rate": "labour-force exit rate",
    "ext_sep_transfer_rate": "occupational transfer rate",
    "ext_union_cov_pct": "union coverage",
    "ext_self_employed_pct": "self-employment share",
    "ext_employment_log": "log employment",
    "ext_wage_disp_p90p10": "wage dispersion, p90/p10",
}


def cv_r2(X, y):
    """Out-of-fold R2 of a ridge regression, scaler and penalty fitted in-fold.
    Rows where y is NaN are left out."""
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    pred = np.empty_like(y)
    for train, test in KFold(5, shuffle=True, random_state=SEED).split(X):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def observed(clean, raw, col):
    """True for occupations whose value of `col` in the clean table rests on a
    published value, False where clean_master.py filled it in.

    A wage summary counts as published if any of its percentiles is (a wage at
    the OEWS cap is read from the source). Required education counts as
    published on either the RL or the RQ scale.
    """
    raw = raw.reindex(clean.index)
    if col in LABOUR:
        sources = LABOUR[col]
    elif col.startswith("ete_rl__"):
        sources = [c for c in raw.columns if c.startswith(("ete_rl__", "ete_rq__"))]
    elif col.startswith("ete_"):
        prefix = col.split("__", 1)[0] + "__"
        sources = [c for c in raw.columns if c.startswith(prefix)]
    else:
        sources = [col]
    return raw[sources].notna().any(axis=1).to_numpy()


def published(clean, raw, col):
    """The clean column with filled-in values set to NaN."""
    return np.where(observed(clean, raw, col), clean[col].to_numpy(float), np.nan)


def ete_by_column(clean, raw, X):
    rows = []
    for col in [c for c in clean.columns if c.startswith("ete_")]:
        y = published(clean, raw, col)
        rows.append({"column": col, "occupations": int((~np.isnan(y)).sum()),
                     "variance": np.nanvar(y), "R2": cv_r2(X, y)})
    return pd.DataFrame(rows)


def descriptives(clean, raw):
    raw = raw.reindex(clean.index)
    title = clean["title"]
    out = []

    emp = raw["ext_employment"].dropna()
    out += [("employment: occupations with a figure", len(emp)),
            ("employment: median", emp.median()),
            ("employment: minimum", emp.min()),
            ("employment: occupation with the minimum", title[emp.idxmin()]),
            ("employment: occupations above one million", int((emp > 1e6).sum())),
            ("employment: those occupations", "; ".join(title[emp[emp > 1e6].index]))]

    se = raw["ext_self_employed_pct"].dropna()
    top = se.sort_values(ascending=False).head(8)
    out += [("self-employment: occupations with a figure", len(se)),
            ("self-employment: median %", se.median()),
            ("self-employment: mean %", se.mean()),
            ("self-employment: maximum %", se.max()),
            ("self-employment: highest eight",
             "; ".join(f"{title[i]} {v:.1f}" for i, v in top.items())),
            ("self-employment: corr with log employment",
             clean["ext_self_employed_pct"].corr(clean["ext_employment_log"])),
            ("self-employment: corr with log median wage",
             clean["ext_self_employed_pct"].corr(clean["ext_wage_level_log"]))]

    un = raw["ext_union_cov_pct"].dropna() * 100     # stored as a share
    out += [("union: occupations with a figure", len(un)),
            ("union: median %", un.median()),
            ("union: lower quartile %", un.quantile(.25)),
            ("union: upper quartile %", un.quantile(.75)),
            ("union: share below 20%", (un < 20).mean()),
            ("union: occupations above 50%", int((un > 50).sum())),
            ("union: those occupations", "; ".join(title[un[un > 50].index])),
            ("union: distinct values (CPS classes behind them)", un.nunique())]

    # The eight labour-market variables decomposed on their own.
    L = clean[list(LABOUR)]
    Z = ((L - L.mean()) / L.std()).to_numpy()
    ev, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    lead = vec[:, -1] * np.sign(vec[:, -1].sum())
    j = list(LABOUR).index("ext_union_cov_pct")
    out += [("labour-market PCA: leading share of variance", ev[-1] / ev.sum()),
            ("labour-market PCA: union eigenvector weight", lead[j]),
            ("labour-market PCA: union correlation with the component",
             lead[j] * np.sqrt(ev[-1]))]
    return pd.DataFrame(out, columns=["quantity", "value"])


def main():
    clean = pd.read_csv(MASTER, index_col="onet_soc")
    raw = pd.read_csv(RAW, index_col="onet_soc", low_memory=False)
    feats = [c for c in clean.columns if c.startswith(FEATURE_PREFIXES)]
    X = clean[feats].to_numpy(float)
    # each feature block alone, e.g. the 55 Work Context columns
    blocks = {}
    for c in feats:
        blocks.setdefault(c.split("__", 1)[0], []).append(c)
    print(f"{len(clean)} occupations, {len(feats)} feature columns")

    # B first: the ETE row of Table 5.5 comes from it.
    ete = ete_by_column(clean, raw, X)
    weighted = (ete["R2"] * ete["variance"]).sum() / ete["variance"].sum()
    ete_summary = pd.DataFrame([
        ("columns", len(ete)),
        ("median R2", ete["R2"].median()),
        ("columns with R2 < 0.3", int((ete["R2"] < 0.3).sum())),
        ("columns with R2 > 0.6", int((ete["R2"] > 0.6).sum())),
        ("variance-weighted R2 of the block", weighted),
    ], columns=["quantity", "value"])

    rows = []
    for col, label in TARGETS.items():
        y = published(clean, raw, col)
        row = {"variable": label, "occupations": int((~np.isnan(y)).sum()),
               "all features": cv_r2(X, y)}
        for b, cols in blocks.items():
            row[b] = cv_r2(clean[cols].to_numpy(float), y)
        rows.append(row)
        print(f"  {label}: done")
    rows.append({"variable": "education, training and experience (variance-weighted)",
                 "occupations": int(ete["occupations"].min()), "all features": weighted})
    table = pd.DataFrame(rows).sort_values("all features", ascending=False)

    sheets = {"table_5_5": table, "ete_summary": ete_summary,
              "ete_by_column": ete.sort_values("R2"),
              "descriptives": descriptives(clean, raw)}
    with pd.ExcelWriter(OUT) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.max_colwidth", 200,
                           "display.float_format", "{:.3f}".format):
        for name, frame in sheets.items():
            print(f"\n=== {name} ===")
            if name in ("ete_summary", "descriptives"):
                for q, v in frame.itertuples(index=False):
                    print(f"  {q}: {v:.3f}" if isinstance(v, float) else f"  {q}: {v}")
            else:
                print(frame.to_string(index=False))
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
