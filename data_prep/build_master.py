#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py -- join five O*NET blocks (Abilities, Skills, Knowledge, Work
Activities, Work Context) into one table: one row per occupation, one column
per feature.

Input:  O*NET 31.0 text files in data/raw/onet_31_0 (download first)
Output: data/prepared/master.csv

Blocks are read one at a time; each is joined and released before the next.

Column names: <block>_<scale>__<Element Name>, plus __c<category> for scales
with categories, e.g. abilities_lv__Oral Comprehension,
workctx_cxp__Public Speaking__c3.
"""

import sys

import numpy as np
import pandas as pd
from pathlib import Path

ONET_DIR = Path("data/raw/onet_31_0")
OUT = Path("data/prepared/master.csv")

# Column prefix -> the O*NET files that hold the block.
BLOCKS = {
    "abilities": ["Abilities"],
    "skills":    ["Essential Skills", "Transferable Skills"],
    "knowledge": ["Knowledge"],
    "workact":   ["Work Activities"],
    "workctx":   ["Work Context"],
}

RATING_COLUMNS = {
    "O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
    "Category": "category", "Data Value": "value",
}


def read_text(path, columns):
    """Read only the named columns of a tab-separated O*NET file, renamed."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                     usecols=lambda c: c in columns)
    return df.rename(columns=columns)


def occupations():
    """Every O*NET-SOC code in Occupation Data, with its title, as the index."""
    try:
        occ = read_text(ONET_DIR / "Occupation Data.txt",
                        {"O*NET-SOC Code": "onet_soc", "Title": "title"})
    except (OSError, ValueError, KeyError) as err:
        sys.exit(f"[fail] occupations: {err}")
    print(f"[ok] occupations: {len(occ)} codes")
    return occ.set_index("onet_soc")


def block_wide(prefix, stems):
    """One block as an occupation x column table of published values."""
    paths = [ONET_DIR / f"{stem}.txt" for stem in stems]
    df = pd.concat([read_text(p, RATING_COLUMNS) for p in paths], ignore_index=True)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "category" in df.columns:
        category = pd.to_numeric(df["category"], errors="coerce")
    else:
        category = pd.Series(np.nan, index=df.index)
    suffix = category.map(lambda c: "" if pd.isna(c) else f"__c{int(c)}")
    df["column"] = prefix + "_" + df["scale"].str.lower() + "__" + df["element"] + suffix

    return df.pivot(index="soc", columns="column", values="value")


def join_blocks(wide, blocks):
    """Left-join each block onto the table of occupations; stop at the first failure."""
    for prefix, stems in blocks.items():
        try:
            block = block_wide(prefix, stems)
        except (OSError, ValueError, KeyError) as err:
            sys.exit(f"[fail] {prefix}: {err}")
        unknown = block.index.difference(wide.index)
        wide = wide.join(block, how="left")
        joined = len(block) - len(unknown)
        print(f"[ok] {prefix}: joined {joined} occupations, {block.shape[1]} columns")
        if len(unknown):
            print(f"[warn] {prefix}: {len(unknown)} codes not in Occupation Data, dropped")
    return wide.rename_axis("onet_soc").reset_index()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    M = join_blocks(occupations(), BLOCKS)
    M.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[ok] wrote {OUT}: {M.shape[0]} rows x {M.shape[1]} columns")


if __name__ == "__main__":
    main()