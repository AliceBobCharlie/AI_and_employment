#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_master.py -- choose the rows and columns of the feature matrix and fill
its gaps:

    output/master.csv  ->  output/master_clean.csv

  ROWS. Keeps the occupations rated in all five feature blocks: Abilities,
  Skills, Knowledge and Work Activities on the Importance scale, Work Context
  on the Context scale. The rows kept here are also the rows of
  output/labour_clean.csv (see clean_labour.py).

  COLUMNS. Importance for four blocks; for Work Context, CX for the items
  rated on it and the category percentages (CTP) of the two working-time
  items, Duration of Typical Work Week and Work Schedules, without the
  standard category of each (40 hours, regular). The Level scale and Work
  Context CT and CXP stay in output/master.csv only.

  GAPS. Any cell still missing (the CTP cells of the occupations without
  them) takes the column median.

Not done here: standardisation, which each analysis does for itself.
"""

import pandas as pd
from pathlib import Path

SRC = Path("output/master.csv")
OUT = Path("output/master_clean.csv")

ID_COLUMNS = ["onet_soc", "title"]
FEATURE_PREFIXES = ["skills_im", "abilities_im", "knowledge_im", "workact_im", "workctx_cx"]
WORKING_TIME = "workctx_ctp"     # in the feature matrix; not used to select rows
CTP_STANDARD = {"Duration of Typical Work Week": "c2",    # 40 hours
                "Work Schedules": "c1"}                   # regular


def columns_with(df, prefix):
    return [c for c in df.columns if c.startswith(prefix + "__")]


def keep_rated(df):
    """Keep occupations rated in all five feature blocks; report the others."""
    has = pd.DataFrame({p: df[columns_with(df, p)].notna().any(axis=1)
                        for p in FEATURE_PREFIXES})
    keep = has.all(axis=1)
    none = ~has.any(axis=1)
    partial = ~keep & ~none
    print(f"rows    : {int(keep.sum())} kept, {int(none.sum())} with no ratings dropped, "
          f"{int(partial.sum())} partly rated dropped")
    for i in df.index[partial]:
        present = ", ".join(p for p in FEATURE_PREFIXES if has.at[i, p])
        print(f"          dropped {df.at[i, 'onet_soc']} {df.at[i, 'title']} (has only {present})")
    return df[keep].reset_index(drop=True)


def working_time_columns(df):
    """The CTP columns kept: every category but the standard one of each item."""
    return [c for c in columns_with(df, WORKING_TIME)
            if c.split("__")[2] != CTP_STANDARD[c.split("__")[1]]]


def clean(df):
    df = keep_rated(df)
    features = [c for p in FEATURE_PREFIXES for c in columns_with(df, p)]
    features += working_time_columns(df)
    df = df[ID_COLUMNS + features].copy()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    missing = df[features].isna()
    for prefix in FEATURE_PREFIXES + [WORKING_TIME]:
        cols = [c for c in features if c.startswith(prefix + "__")]
        rows = int(missing[cols].any(axis=1).sum())
        if rows:
            print(f"median  : {prefix:22s} {rows:3d} occupations, "
                  f"{int(missing[cols].sum().sum())} cells")
    df[features] = df[features].fillna(df[features].median())
    print(f"filled {int(missing.sum().sum())} missing cells; "
          f"{int(df[features].isna().sum().sum())} remain (should be 0)")
    return df


def main():
    print(f"reading {SRC}")
    df = clean(pd.read_csv(SRC, low_memory=False))
    print(f"clean table: {df.shape[0]} occupations; {df.shape[1] - len(ID_COLUMNS)} feature columns")
    df.to_csv(OUT, index=False)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()