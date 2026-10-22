#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all_scales.py -- what do the axes look like when every rating scale goes into
the matrix?

The paper's matrix is Importance for Abilities, Skills, Knowledge and Work
Activities, and the Context mean (CX) for Work Context: 216 columns. This
script builds a matrix from all the rating scales, on the same 910
occupations:

  IM    Importance for Skills, Abilities, Knowledge and Work Activities (161)
  LV    Level for Skills, Abilities and Work Activities (128). Knowledge
        Level is left out.
  CXP   Work Context category percentages. Each item's five percentages sum
        to 100, so one category per item is dropped; the lowest ("never",
        "not important at all") is the one dropped, which keeps the remaining
        columns pointing the same way as the item's mean.
  CTP   Category percentages for Duration of Typical Work Week (less than
        40 hours, 40 hours, more than 40) and Work Schedules (regular,
        irregular, seasonal). The baseline category of each, 40 hours and
        regular, is dropped, so the remaining four columns are departures
        from a standard week. Percentages rather than the CT mean, because
        Work Schedules is a nominal item and the mean of Duration cannot
        tell part-time from long hours.

The CXP columns are weighted by one half, so that the many category columns
do not dominate the decomposition; the four CTP columns keep full weight.
Columns are z-scored, CXP columns are multiplied by sqrt(1/2), and PCA is
done on the result. The first
N_AXES components are varimax-rotated together, not in tiers.

Printed for each axis:
  - its share of the matrix's variance;
  - the TOP_COLS columns with the largest positive and negative loadings.
    Loadings are correlations between a column and the axis scores, so
    weighted and unweighted columns are on one scale;
  - the TOP_OCC occupations with the highest and lowest scores;
  - the correlation of its scores with the paper's axes R1-R6, if
    output/occupation_coordinates.csv exists.

Axes are ordered by variance and signed so that the larger pole is positive.
The signs carry no claim about substitutability.

Descriptive only: no resampling. Run after clean_master.py (and pca_rotated.py
for the comparison with R1-R6):
    python supplementary/all_scales.py
Paths are resolved from the project root, so the script runs from any
directory. Reads output/master_clean.csv and output/master.csv. Prints only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pca_rotated import rotate_tier, standardize  # noqa: E402

CLEAN = ROOT / "output" / "master_clean.csv"
RAW = ROOT / "output" / "master.csv"
AXES = ROOT / "output" / "occupation_coordinates.csv"

N_AXES = 6              # components rotated together
CXP_WEIGHT = 0.5        # variance weight of every CXP column
CXP_DROP = "c1"         # CXP category left out of each item: the lowest
CTP_DROP = {"Duration of Typical Work Week": "c2",    # 40 hours
            "Work Schedules": "c1"}                   # regular
CTP_NAMES = {"Duration of Typical Work Week": {"c1": "less than 40 hours",
                                               "c3": "more than 40 hours"},
             "Work Schedules": {"c2": "irregular", "c3": "seasonal"}}
TOP_COLS = 12
TOP_OCC = 10

IM = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")
LV = ("skills_lv__", "abilities_lv__", "workact_lv__")


# --------------------------------------------------------------------------- #
def load():
    """The matrix on the occupations of the clean table. Importance comes
    from the clean table; Level and the Work Context category percentages
    from the raw one, with missing cells filled by the column median."""
    clean = pd.read_csv(CLEAN, index_col="onet_soc")
    raw = pd.read_csv(RAW, index_col="onet_soc", low_memory=False).reindex(clean.index)

    def cols(src, *prefixes):
        return src[[c for c in src.columns if c.startswith(prefixes)]]

    cxp = cols(raw, "workctx_cxp__")
    cxp = cxp[[c for c in cxp.columns if not c.endswith("__" + CXP_DROP)]]
    ctp = cols(raw, "workctx_ctp__")
    ctp = ctp[[c for c in ctp.columns if c.split("__")[2] != CTP_DROP[c.split("__")[1]]]]
    df = pd.concat([cols(clean, *IM), cols(raw, *LV), cxp, ctp], axis=1)
    n_missing = int(df.isna().sum().sum())
    df = df.fillna(df.median())
    df = df.loc[:, df.std() > 0]
    if n_missing:
        print(f"[note] {n_missing} missing cells filled with the column median")
    return df, clean["title"]


def block(col):
    """'skills_im__Speaking' -> 'IM'; 'workctx_cxp__...' -> 'CXP'."""
    return col.split("__", 1)[0].split("_")[1].upper()


def label(col):
    """'workctx_cxp__Spend Time Sitting__c5' -> 'Spend Time Sitting [CXP c5]';
    'workctx_ctp__Work Schedules__c3' -> 'Work Schedules: seasonal [CTP]'."""
    parts = col.split("__")
    if block(col) == "CTP":
        return f"{parts[1]}: {CTP_NAMES[parts[1]][parts[2]]} [CTP]"
    cat = f" {parts[2]}" if len(parts) > 2 else ""
    return f"{parts[1]} [{block(col)}{cat}]"


def decompose(Zw):
    """Eigenvalues (n-1 denominator), eigenvectors and unit-variance scores."""
    Zc = Zw - Zw.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    ev = s ** 2 / (len(Zc) - 1)
    V = vt.T
    return ev, V, Zc @ V / np.sqrt(ev)


# --------------------------------------------------------------------------- #
def main():
    df, titles = load()
    blocks = df.columns.map(block)
    is_cxp = df.columns.str.startswith("workctx_cxp__")
    w = np.where(is_cxp, np.sqrt(CXP_WEIGHT), 1.0)

    Z = standardize(df.to_numpy(float))
    ev, V, T = decompose(Z * w)
    total = ev.sum()

    print(f"all-scales matrix: {Z.shape[0]} occupations x {Z.shape[1]} columns")
    var_share = pd.Series(w ** 2, index=blocks).groupby(level=0).sum() / (w ** 2).sum()
    counts = pd.Series(blocks).value_counts()
    print("columns and share of the matrix's variance by block:")
    for b in ("IM", "LV", "CXP", "CTP"):
        print(f"  {b:4s} {counts[b]:4d} columns  {100 * var_share[b]:5.1f}%")

    print(f"\nunrotated components, % of variance (first 10): "
          + " ".join(f"{100 * e / total:.1f}" for e in ev[:10]))
    print(f"components 1-{N_AXES} together: {100 * ev[:N_AXES].sum() / total:.1f}%")

    idx = list(range(N_AXES))
    L, R = rotate_tier(V, ev, idx)
    S = T[:, idx] @ R
    flip = np.where((L ** 3).sum(axis=0) < 0, -1.0, 1.0)        # larger pole positive
    L, S = L * flip, S * flip
    share = (L ** 2).sum(axis=0) / total

    Sz = (S - S.mean(axis=0)) / S.std(axis=0)
    corr = Z.T @ Sz / len(Z)                                    # column-score correlations
    names = [label(c) for c in df.columns]
    titles = titles.to_numpy()

    paper = None
    if AXES.exists():
        paper = pd.read_csv(AXES, index_col=0).reindex(df.index)
        paper = paper[[c for c in paper.columns if c.startswith("R")]]

    for j in range(N_AXES):
        print("\n" + "=" * 70)
        print(f"AXIS W{j + 1}: {100 * share[j]:.1f}% of variance")
        print("=" * 70)
        if paper is not None:
            r = [np.corrcoef(S[:, j], paper[c])[0, 1] for c in paper.columns]
            print("  r with the paper's axes: "
                  + "  ".join(f"{c} {v:+.2f}" for c, v in zip(paper.columns, r)))
        for sign, order in (("+", np.argsort(-corr[:, j])), ("-", np.argsort(corr[:, j]))):
            print(f"  {sign} pole, columns:")
            for i in order[:TOP_COLS]:
                print(f"     {corr[i, j]:+.2f}  {names[i]}")
        for sign, order in (("+", np.argsort(-S[:, j])), ("-", np.argsort(S[:, j]))):
            print(f"  {sign} pole, occupations:")
            for i in order[:TOP_OCC]:
                print(f"     {S[i, j]:+.2f}  {titles[i]}")


if __name__ == "__main__":
    main()