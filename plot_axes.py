#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_axes.py -- export the data the interactive figures are drawn from.

The figures themselves are drawn in the browser by plotly.js on the paper's web
page, so this script only assembles one small table: each occupation's position
on the three rotated axes, plus the two variables the figures show as colour
and marker size.

Axis signs are fixed by the anchors in pca_rotated.py, so '+' always means
harder to automate:

  R1  embodiment     abstract/symbolic (-)   <->  physical/hands-on (+)
  R2  cognitive load routine execution (-)   <->  complex reasoning (+)
  R3  interpersonal  technical/systems (-)   <->  care/people (+)

Reads output/rotated_axes.csv (from pca_rotated.py) and output/master_clean.xlsx.
Writes output/plot_data.csv.
"""

import pandas as pd
from pathlib import Path

AXES_CSV = Path("output/rotated_axes.csv")
MASTER = Path("output/master_clean.xlsx")
OUT = Path("output/plot_data.csv")


def main():
    axes = pd.read_csv(AXES_CSV, index_col="onet_soc")
    master = pd.read_excel(MASTER, index_col="onet_soc",
                           usecols=["onet_soc", "ext_union_cov_pct", "ext_employment"])

    df = axes[["title", "R1", "R2", "R3"]].join(master, how="inner")
    df = df.rename(columns={"ext_union_cov_pct": "union",
                            "ext_employment": "employment"})

    if len(df) != len(axes):
        raise ValueError(f"{len(axes) - len(df)} occupations in {AXES_CSV} "
                         f"have no row in {MASTER}")

    df.to_csv(OUT, float_format="%.6g")
    print(f"[ok] wrote {OUT}: {len(df)} occupations")


if __name__ == "__main__":
    main()
