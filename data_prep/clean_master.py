#!/usr/bin/env python3
"""
clean_master.py -- drop the occupations without ratings, drop one category of
each percentage item, and fill the remaining gaps.

Input:  data/prepared/master.csv
Output: data/prepared/master_clean.csv

keep_rated() keeps the occupations with a value in every block of ROW_PREFIXES:
Importance for Abilities, Skills, Knowledge and Work Activities, and CX for
Work Context. is_redundant() marks one category of each item whose
percentages sum to 100: CXP_DROPPED in every CXP item, and the category in
CTP_DROPPED in each CTP item. Those columns are dropped, and so is any column
empty on every kept occupation, with a warning. Every remaining missing cell
takes its column median. Columns keep the order of master.csv.
"""

import sys

import pandas as pd
from pathlib import Path

SRC = Path("data/prepared/master.csv")
OUT = Path("data/prepared/master_clean.csv")

ID_COLUMNS = ["onet_soc", "title"]
ROW_PREFIXES = ["skills_im", "abilities_im", "knowledge_im", "workact_im", "workctx_cx"]
CXP_DROPPED = "c1"                                        # lowest category
CTP_DROPPED = {"Duration of Typical Work Week": "c2",     # 40 hours
               "Work Schedules": "c1"}                    # regular


def columns_with(columns, prefix):
    return [c for c in columns if c.startswith(prefix + "__")]


def keep_rated(df):
    """Rows with a value in every block of ROW_PREFIXES."""
    has = pd.DataFrame({p: df[columns_with(df.columns, p)].notna().any(axis=1)
                        for p in ROW_PREFIXES})
    keep = has.all(axis=1)
    print(f"[ok] rows: kept {int(keep.sum())}, "
          f"dropped {int((~keep).sum())} missing a whole block")
    return df[keep].reset_index(drop=True)


def is_redundant(col):
    """True for the dropped category of a CXP or CTP item."""
    parts = col.split("__")
    if parts[0] == "workctx_cxp":
        return parts[2] == CXP_DROPPED
    if parts[0] == "workctx_ctp":
        if parts[1] not in CTP_DROPPED:
            sys.exit(f"[fail] columns: no dropped category set for CTP item '{parts[1]}'")
        return parts[2] == CTP_DROPPED[parts[1]]
    return False


def clean(df):
    df = keep_rated(df)
    all_features = [c for c in df.columns if c not in ID_COLUMNS]
    features = [c for c in all_features if not is_redundant(c)]
    redundant = len(all_features) - len(features)
    df = df[ID_COLUMNS + features].copy()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    empty = [c for c in features if df[c].isna().all()]
    if empty:
        print(f"[warn] columns: {len(empty)} empty on every kept occupation, dropped")
        features = [c for c in features if c not in empty]
        df = df[ID_COLUMNS + features]
    print(f"[ok] columns: kept {len(features)}, dropped {redundant} redundant categories")

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