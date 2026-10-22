#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py -- assemble the raw table of occupational features:

    data_raw/onet/db_31_0_text  ->  output/master.csv

One row per O*NET-SOC code listed in O*NET's Occupation Data, rated or not.
One column per block, scale, element and (where the scale has them) category,
holding the Data Value as O*NET publishes it. The blocks are Abilities,
Skills, Knowledge, Work Activities and Work Context.

This script only reads and joins. It drops no occupation and no scale, fills
no gap and derives no variable; that is done in clean_master.py. It keeps
Importance and Level for Abilities, Skills, Knowledge and Work Activities, and
CX, CT, CXP and CTP for Work Context. Ratings that O*NET flags Not Relevant or
Recommend Suppress are kept as published; the flags are not carried over.

Column names: <block>_<scale>__<Element Name>, with __c<category> added for
scales that have categories, e.g. abilities_lv__Oral Comprehension,
workctx_cxp__Public Speaking__c3.

The labour-market data are assembled separately, by build_lmv.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ONET_DIR = Path("data_raw") / "onet" / "db_31_0_text"
OUT = Path("output/master.csv")

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
    occ = read_text(ONET_DIR / "Occupation Data.txt",
                    {"O*NET-SOC Code": "onet_soc", "Title": "title"})
    print(f"[O*NET] {ONET_DIR}: {len(occ)} O*NET-SOC codes in Occupation Data")
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

    duplicated = df.duplicated(["soc", "column"])
    wide = df[~duplicated].pivot(index="soc", columns="column", values="value")
    return wide, [p.name for p in paths], int(duplicated.sum())


def join_blocks(wide, blocks):
    """Left-join each block onto the table of occupations, reporting what was read."""
    for prefix, stems in blocks.items():
        block, files, duplicates = block_wide(prefix, stems)
        print(f"  [{prefix}] {', '.join(files)}: {block.shape[1]} columns, {len(block)} codes"
              + (f"  ({duplicates} duplicate ratings, first kept)" if duplicates else ""))
        unknown = block.index.difference(wide.index)
        if len(unknown):
            print(f"    note: {len(unknown)} rated codes not in Occupation Data: "
                  f"{', '.join(unknown)}")
        wide = wide.join(block, how="left")
    return wide.rename_axis("onet_soc").reset_index()


def main():
    OUT.parent.mkdir(exist_ok=True)
    M = join_blocks(occupations(), BLOCKS)
    M.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n[ok] wrote {OUT}: {M.shape[0]} rows x {M.shape[1]} columns")
    print("coverage, flags and missingness: run check_data.py")


if __name__ == "__main__":
    main()