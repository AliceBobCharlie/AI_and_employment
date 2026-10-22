#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_lmv.py -- the labour-market table on the rows of the feature matrix,
with its gaps filled and variables derived:

    output/labour.csv + output/master_clean.csv  ->  output/labour_clean.csv

  ROWS. The occupations of output/master_clean.csv, in the same order. The
  feature matrix decides which occupations are analysed; this table follows.
  All medians below are taken over these rows.

  EDUCATION. Required education is published on RL for some occupations and
  on RQ for the others, never both. The two are merged into the RL columns,
  category by category (the twelve categories match in number and order).

  HIGHEST CATEGORY. The categories of each education, training and experience
  scale sum to 100, so the highest is dropped.

  SELF-EMPLOYMENT. Occupations absent from BLS table 1.2 are set to 0.

  WAGES. Missing wage cells take the column median; the three wage ratios are
  then computed from the filled levels.

  DERIVED. ext_employment_log, ext_wage_level_log, ext_wage_disp_p90p10, and
  the three wage ratios.

  EVERYTHING ELSE still missing takes the column median.

Not done here: standardisation, which each analysis does for itself.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path("output/labour.csv")
ROWS = Path("output/master_clean.csv")
OUT = Path("output/labour_clean.csv")

ID_COLUMNS = ["onet_soc", "title"]
ETE_PREFIXES = ["ete_rl", "ete_rw", "ete_pt", "ete_oj"]
WAGE_LEVELS = ["ext_wage_median", "ext_wage_mean", "ext_wage_p10",
               "ext_wage_p25", "ext_wage_p75", "ext_wage_p90"]


def columns_with(df, prefix):
    return [c for c in df.columns if c.startswith(prefix + "__")]


def keep_feature_rows(df):
    """Keep the occupations of the feature matrix, in its order."""
    rows = pd.read_csv(ROWS, usecols=["onet_soc"])["onet_soc"]
    absent = rows[~rows.isin(df["onet_soc"])]
    if len(absent):
        raise ValueError(f"{len(absent)} occupations of {ROWS} are not in {SRC}")
    print(f"rows    : {len(rows)} occupations of {ROWS} kept, of {len(df)} in {SRC}")
    return df.set_index("onet_soc").loc[rows].reset_index()


def merge_education(df):
    """Fold RQ into RL, category by category; an occupation has one or the other."""
    rq_cols = columns_with(df, "ete_rq")
    n_rq = int(df[rq_cols].notna().any(axis=1).sum())
    for rq in rq_cols:
        rl = "ete_rl__" + rq.split("__", 1)[1]
        clash = df[rl].notna() & df[rq].notna()
        if clash.any():
            raise ValueError(f"{int(clash.sum())} occupations have both {rl} and {rq}")
        df[rl] = df[rl].fillna(df[rq])
    print(f"educ    : RQ merged into RL for {n_rq} occupations")
    return df.drop(columns=rq_cols)


def drop_highest_category(df):
    dropped = []
    for prefix in ETE_PREFIXES:
        cols = columns_with(df, prefix)
        top = max(int(re.search(r"__c(\d+)$", c).group(1)) for c in cols)
        dropped += [c for c in cols if c.endswith(f"__c{top}")]
    print(f"ete     : highest category dropped from each scale ({len(dropped)} columns)")
    return df.drop(columns=dropped)


def fill_wages(df):
    """Missing wage cells -> column median; then the ratios from the filled levels."""
    no_wage = df[WAGE_LEVELS].isna().all(axis=1)
    missing = df[WAGE_LEVELS].isna()
    n_whole = int(missing[no_wage].sum().sum())
    n_partial = int(missing[~no_wage].sum().sum())
    n_cap = int((df[WAGE_LEVELS] == 239200).sum().sum())
    df[WAGE_LEVELS] = df[WAGE_LEVELS].fillna(df[WAGE_LEVELS].median())
    df["ext_wage_p90p10"] = df["ext_wage_p90"] / df["ext_wage_p10"]
    df["ext_wage_p90p50"] = df["ext_wage_p90"] / df["ext_wage_median"]
    df["ext_wage_p50p10"] = df["ext_wage_median"] / df["ext_wage_p10"]
    print(f"wages   : {n_cap} cells at the OEWS cap (read from the source); "
          f"{n_whole} cells in {int(no_wage.sum())} occupations with no annual wage "
          f"and {n_partial} other cells -> column median")
    return df


def clean(df):
    df = keep_feature_rows(df)
    df = merge_education(df)
    df = drop_highest_category(df)
    ete = [c for p in ETE_PREFIXES for c in columns_with(df, p)]
    ext = [c for c in df.columns if c.startswith("ext_")]
    df = df[ID_COLUMNS + ete + ["soc6"] + ext].copy()

    values = ete + ext
    df[values] = df[values].apply(pd.to_numeric, errors="coerce")
    df = df.copy()               # column-wise assignment above fragments the frame
    before = int(df[values].isna().sum().sum())

    n_self = int(df["ext_self_employed_pct"].isna().sum())
    df["ext_self_employed_pct"] = df["ext_self_employed_pct"].fillna(0)
    print(f"selfemp : {n_self} occupations absent from BLS table 1.2 -> 0")

    df = fill_wages(df)
    df = df.assign(
        ext_employment_log=np.log1p(df["ext_employment"]),
        ext_wage_level_log=np.log(df["ext_wage_median"]),
        ext_wage_disp_p90p10=df["ext_wage_p90"] / df["ext_wage_p10"],
    )

    values = [c for c in df.columns if c not in ID_COLUMNS + ["soc6"]]
    still = df[values].isna()
    for prefix in ETE_PREFIXES:
        cols = columns_with(df, prefix)
        rows = int(still[cols].any(axis=1).sum())
        if rows:
            print(f"median  : {prefix:22s} {rows:3d} occupations, "
                  f"{int(still[cols].sum().sum())} cells")
    for c in [c for c in values if c.startswith("ext_") and still[c].any()]:
        print(f"median  : {c:22s} {int(still[c].sum()):3d} cells")
    df[values] = df[values].fillna(df[values].median())

    left = int(df[values].isna().sum().sum())
    print(f"filled {before} missing cells; {left} remain (should be 0)")
    return df


def main():
    print(f"reading {SRC}")
    df = clean(pd.read_csv(SRC, low_memory=False))
    ete = sum(len(columns_with(df, p)) for p in ETE_PREFIXES)
    ext = len([c for c in df.columns if c.startswith("ext_")])
    print(f"clean table: {df.shape[0]} occupations; "
          f"{ete} education/training/experience, {ext} external columns")
    df.to_csv(OUT, index=False)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()