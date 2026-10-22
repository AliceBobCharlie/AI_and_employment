#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_lmv.py -- data quality of the labour-market variables.

For the occupations of the feature matrix, on raw values as published (before
clean_labour.py fills anything): which occupations miss which external
variable, and each variable's distribution and skew.

Run after build_labour.py and clean_master.py, from the project root. Reads
output/labour.csv and output/master_clean.csv (for the rows). Writes
output/labour_data_quality.xlsx and prints every table in full.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("output/labour.csv")
ROWS = Path("output/master_clean.csv")
OUT = Path("output/labour_data_quality.xlsx")


def show(df, floatfmt="{:.3f}"):
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", None, "display.max_colwidth", 70,
                           "display.float_format", floatfmt.format):
        print(df.to_string())


def external(L, kept):
    ext = [c for c in L.columns if c.startswith("ext_")]
    X = L.loc[kept, ext].apply(pd.to_numeric, errors="coerce")
    na = X.isna()
    per_var = na.sum().rename("missing").to_frame()
    missing = na[na.any(axis=1)].astype(int)
    missing.columns = [c.replace("ext_", "") for c in ext]
    missing.insert(0, "title", L.loc[missing.index, "title"])
    missing["n missing"] = missing.iloc[:, 1:].sum(axis=1)
    missing = missing.sort_values("n missing", ascending=False)
    desc = pd.DataFrame({"n": X.count(), "mean": X.mean(), "sd": X.std(), "min": X.min(),
                         "q25": X.quantile(.25), "median": X.median(), "q75": X.quantile(.75),
                         "max": X.max(), "skew": X.skew(),
                         "max/median": X.max() / X.median().replace(0, np.nan)})
    print(f"occupations missing at least one variable: {len(missing)} of {len(X)}; "
          f"missing all but at most one: {int((missing['n missing'] >= len(ext) - 1).sum())}; "
          f"missing one or two: {int((missing['n missing'] <= 2).sum())}")
    show(per_var)
    print("\noccupations with a missing value (1 = missing):")
    show(missing.replace({0: ""}), "{:.0f}")
    print()
    show(desc, "{:,.2f}")
    skewed = desc.index[(desc["skew"].abs() > 1) | (desc["max/median"] > 5)]
    print(f"\nright-skewed (|skew| > 1 or max/median > 5): {list(skewed) or 'none'}")
    return {"external_missing": missing.reset_index(),
            "external_describe": desc.reset_index(names="variable")}


def main():
    L = pd.read_csv(RAW, index_col="onet_soc", low_memory=False)
    kept = pd.read_csv(ROWS, index_col="onet_soc", usecols=["onet_soc"]).index
    print(f"[read] {RAW}: {len(L)} codes; {ROWS}: {len(kept)} occupations kept")

    sheets = external(L, kept)
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()