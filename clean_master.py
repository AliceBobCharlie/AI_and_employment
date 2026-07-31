#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_master.py -- the single place where imputation and transformation happen.

`master_wide.xlsx` deliberately holds the raw published values, gaps and all.
This script turns it into `master_clean.xlsx`, an analysis-ready table with no
missing cells, so that every downstream script reads the same filled numbers
instead of repeating (and slowly diverging from) the same fill logic.

What it does, and why:

  WAGES. OEWS suppresses annual wages above $239,200. Two kinds of gap need
  different treatment, told apart by whether the occupation has a p10 value:
    - a high percentile missing while p10 is present  -> top-coded. The true
      value is known to lie ABOVE the cap, so filling at the median would put
      a surgeon below their own 10th percentile. Filled at the cap.
    - every wage percentile missing                   -> the occupation is
      absent from OEWS altogether. No anchor to extrapolate from, so filled at
      the column median.
  The three wage ratios are then RECOMPUTED from the filled levels, which keeps
  them >= 1 and consistent with the levels they are built from.

  EMPLOYMENT. Spans orders of magnitude (max/median ~ 69), so a log version is
  added; the raw column is kept for figures.

  DERIVED WAGE COLUMNS. `ext_wage_level_log` and `ext_wage_disp_p90p10` collapse
  the nine collinear wage columns into a level and a dispersion, which is what
  the rotated PCA uses so that wages cannot dominate an axis by column count.

  EVERYTHING ELSE. Remaining gaps (the 16 occupations with no ETE rows, prestige,
  separation rates) are filled at the column median.

What it deliberately does NOT do: standardization. Z-scoring stays inside each
analysis script, because `test_orthogonality.py` must fit its scaler on the
training folds only, and baking it in here would leak test-fold information.

Reads output/master_wide.xlsx -> writes output/master_clean.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path

IN = Path("output/master_wide.xlsx")
OUT = Path("output/master_clean.xlsx")

OEWS_CAP = 239200.0          # OEWS annual-wage top-code
WAGE_LEVELS = ["ext_wage_median", "ext_wage_mean", "ext_wage_p10",
               "ext_wage_p25", "ext_wage_p75", "ext_wage_p90"]


def fill_wages(df):
    """Top-coded highs -> cap; occupations absent from OEWS -> column median."""
    if "ext_wage_p10" not in df.columns:
        return df, 0, 0
    has_p10 = df["ext_wage_p10"].notna()
    n_cap = n_med = 0
    for col in [c for c in WAGE_LEVELS if c in df.columns]:
        na = df[col].isna()
        cap_rows = na & has_p10          # value exists but was suppressed as '#'
        med_rows = na & ~has_p10         # occupation not in OEWS at all
        df.loc[cap_rows, col] = OEWS_CAP
        df.loc[med_rows, col] = df[col].median()
        n_cap += int(cap_rows.sum())
        n_med += int(med_rows.sum())

    # ratios recomputed from the filled levels, never filled directly
    if {"ext_wage_p90", "ext_wage_p10"} <= set(df.columns):
        df["ext_wage_p90p10"] = df["ext_wage_p90"] / df["ext_wage_p10"]
    if {"ext_wage_p90", "ext_wage_median"} <= set(df.columns):
        df["ext_wage_p90p50"] = df["ext_wage_p90"] / df["ext_wage_median"]
    if {"ext_wage_median", "ext_wage_p10"} <= set(df.columns):
        df["ext_wage_p50p10"] = df["ext_wage_median"] / df["ext_wage_p10"]
    return df, n_cap, n_med


def main():
    df = pd.read_excel(IN)
    id_cols = [c for c in ["onet_soc", "title", "soc6"] if c in df.columns]
    feats = [c for c in df.columns if c not in id_cols]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    before = int(df[feats].isna().sum().sum())

    df, n_cap, n_med = fill_wages(df)
    print(f"wages   : {n_cap} top-coded cells -> ${OEWS_CAP:,.0f}, "
          f"{n_med} whole-row cells -> column median")

    # derived columns used by the analyses
    if "ext_employment" in df.columns:
        df["ext_employment_log"] = np.log1p(df["ext_employment"])
    if "ext_wage_median" in df.columns:
        df["ext_wage_level_log"] = np.log(df["ext_wage_median"])
    if {"ext_wage_p90", "ext_wage_p10"} <= set(df.columns):
        df["ext_wage_disp_p90p10"] = df["ext_wage_p90"] / df["ext_wage_p10"]

    # everything still missing -> column median
    feats = [c for c in df.columns if c not in id_cols]
    still = df[feats].isna().sum()
    for c in still[still > 0].index:
        print(f"median  : {c:24s} {int(still[c]):3d} cells")
    df[feats] = df[feats].fillna(df[feats].median())

    left = int(df[feats].isna().sum().sum())
    print(f"\nfilled {before} missing cells; {left} remain (should be 0)")
    print(f"clean table: {df.shape[0]} occupations x {len(feats)} variables")
    df.to_excel(OUT, index=False)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()