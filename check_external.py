#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_external.py -- inspect the external (ext_*) variables in the master table:
  A) exactly which occupations are missing each external variable (enumerated,
     with SOC code + title -- counts are small so we list them)
  B) descriptive statistics for each external variable (n, mean, sd, min,
     quartiles, max, skew) to decide whether any need a log transform before
     standardized PCA.

Reads master_wide.xlsx. Terminal output + external_report.xlsx (two sheets).
"""

import numpy as np
import pandas as pd
from pathlib import Path

MASTER = Path("output/master_wide.xlsx")
OUT = Path("output/qc_external.xlsx")


def main():
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    ext = [c for c in df.columns if c.startswith("ext_")]
    if not ext:
        print("no ext_* columns found"); return
    X = df[ext].apply(pd.to_numeric, errors="coerce")

    # ---------------------------------------------------------------- A. missing
    print("=" * 70)
    print("A. MISSING external values -- occupation x variable matrix")
    print("=" * 70)

    na = X.isna()
    short = [c.replace("ext_", "") for c in ext]     # short column labels

    # keep only occupations missing at least one external variable
    row_has_miss = na.any(axis=1)
    miss_idx = X.index[row_has_miss]

    # build the matrix: 1 = missing, "" = present; add title + total
    mat = na.loc[miss_idx, ext].astype(int)
    mat.columns = short
    mat.insert(0, "title", title.loc[miss_idx])
    mat["_n_missing"] = mat[short].sum(axis=1)
    mat = mat.sort_values("_n_missing", ascending=False)

    per_var = na[ext].sum()
    print(f"\noccupations missing >=1 external var : {len(miss_idx)} / {len(X)}")
    print("\nper-variable missing counts:")
    for c in ext:
        print(f"  {c.replace('ext_',''):16s} {int(per_var[c]):3d}")

    # display the matrix with 1 for missing, blank for present
    show = mat.copy()
    for c in short:
        show[c] = show[c].map({1: "1", 0: ""})
    print("\n--- missing matrix (1 = missing) ---")
    with pd.option_context("display.max_rows", None, "display.width", 240,
                           "display.max_colwidth", 44):
        print(show.to_string())

    # whole-row vs sporadic: a row missing (almost) all external vars is
    # structural non-coverage; a row missing one or two is sporadic
    n_ext = len(ext)
    whole = (mat["_n_missing"] >= n_ext - 1).sum()
    print(f"\noccupations missing (almost) ALL external vars "
          f"(>= {n_ext-1}/{n_ext}): {whole}")
    print(f"occupations missing only 1-2: {(mat['_n_missing'] <= 2).sum()}")

    # ---------------------------------------------------------------- B. describe
    print("\n" + "=" * 70)
    print("B. DESCRIPTIVE STATISTICS (to judge need for log transform)")
    print("=" * 70)
    desc = pd.DataFrame({
        "n":      X.count(),
        "mean":   X.mean(),
        "sd":     X.std(),
        "min":    X.min(),
        "q25":    X.quantile(.25),
        "median": X.median(),
        "q75":    X.quantile(.75),
        "max":    X.max(),
        "skew":   X.skew(),
        "max/med": X.max() / X.median().replace(0, np.nan),
    })
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:,.2f}"):
        print("\n" + desc.to_string())

    print("\nnote: |skew| > ~1 or a large max/median ratio suggests a right tail")
    print("      where a log transform would help before z-scoring for PCA.")
    flag = desc.index[(desc["skew"].abs() > 1) | (desc["max/med"] > 5)]
    if len(flag):
        print(f"\nvariables that look right-skewed: {list(flag)}")
    else:
        print("\nno variable looks strongly skewed on these thresholds.")

    with pd.ExcelWriter(OUT) as xw:
        desc.to_excel(xw, sheet_name="describe")
        mat.to_excel(xw, sheet_name="missing_matrix")
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()