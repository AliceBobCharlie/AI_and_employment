#!/usr/bin/env python3
"""
clean_master.py -- drop columns and handle missing values in the feature table.

Input:  output/master.csv
Output: output/master_clean.csv

Columns: Skills, Knowledge, Abilities and Work Activities keep only Importance
(IM); Work Context keeps only CX and CTP. CTP covers the two working-time items,
Duration of Typical Work Week and Work Schedules, without the standard category
of each (40 hours, regular).

Missing values: an occupation missing a whole block is dropped. Scattered gaps
(in CTP) are filled with the column median.
"""

import sys

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
    """Keep occupations rated in all five feature blocks."""
    has = pd.DataFrame({p: df[columns_with(df, p)].notna().any(axis=1)
                        for p in FEATURE_PREFIXES})
    keep = has.all(axis=1)
    print(f"[ok] rows: kept {int(keep.sum())}, "
          f"dropped {int((~keep).sum())} missing a whole block")
    return df[keep].reset_index(drop=True)


def working_time_columns(df):
    """The CTP columns kept: every category but the standard one of each item."""
    return [c for c in columns_with(df, WORKING_TIME)
            if c.split("__")[2] != CTP_STANDARD[c.split("__")[1]]]


def clean(df):
    df = keep_rated(df)
    features = [c for p in FEATURE_PREFIXES for c in columns_with(df, p)]
    features += working_time_columns(df)
    dropped = df.shape[1] - len(ID_COLUMNS) - len(features)
    df = df[ID_COLUMNS + features].copy()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    print(f"[ok] columns: kept {len(features)} features, dropped {dropped}")

    filled = int(df[features].isna().sum().sum())
    df[features] = df[features].fillna(df[features].median())
    remaining = int(df[features].isna().sum().sum())
    if remaining:
        sys.exit(f"[fail] gaps: {remaining} cells still missing after median fill")
    print(f"[ok] gaps: filled {filled} cells with column medians")
    return df


def main():
    try:
        df = pd.read_csv(SRC, low_memory=False)
    except OSError as err:
        sys.exit(f"[fail] read {SRC}: {err}")
    df = clean(df)
    df.to_csv(OUT, index=False)
    print(f"[ok] wrote {OUT}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()