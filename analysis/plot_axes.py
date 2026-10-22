#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_axes.py -- export the data the interactive figures are drawn from.

The figures themselves are drawn in the browser by plotly.js on the paper's web
page, so this script only assembles two small tables.

  output/plot_data.csv     one row per occupation: its position on the six
                           rotated axes and its SOC major group. No
                           labour-market variable is included; the figures
                           show the axes only.
  output/axis_loadings.csv one row per feature variable: its name, the block
                           of O*NET's Content Model it belongs to, and its
                           loading on each of the six rotated axes.

Axis signs are fixed by the marker variables in pca_rotated.py.

Reads output/rotated_axes.csv and output/rotated_loadings.csv (from
pca_rotated.py). Writes the two files above.
"""

import pandas as pd
from pathlib import Path

AXES_CSV = Path("output/rotated_axes.csv")
LOADINGS_CSV = Path("../output/rotated_loadings.csv")
OUT = Path("../output/plot_data.csv")
OUT_LOADINGS = Path("../output/axis_loadings.csv")

AXES = ["R1", "R2", "R3", "R4", "R5", "R6"]

# Column prefix -> block of the Content Model, in the order the figures show them.
BLOCKS = {"skills_im__": "Skills", "abilities_im__": "Abilities",
          "knowledge_im__": "Knowledge", "workact_im__": "Work Activities",
          "workctx_cx__": "Work Context", "workctx_ctp__": "Work Context"}

# The working-time items enter as category shares (see clean_master.py); these
# are the names the figures show for them.
CTP_NAMES = {"Duration of Typical Work Week__c1": "Work week under 40 hours (share)",
             "Duration of Typical Work Week__c3": "Work week over 40 hours (share)",
             "Work Schedules__c2": "Irregular work schedule (share)",
             "Work Schedules__c3": "Seasonal work schedule (share)"}

# SOC 2018 major groups, by the first two digits of the O*NET-SOC code.
SOC_GROUPS = {
    "11": "Management", "13": "Business and financial operations",
    "15": "Computer and mathematical", "17": "Architecture and engineering",
    "19": "Life, physical and social science", "21": "Community and social service",
    "23": "Legal", "25": "Educational instruction and library",
    "27": "Arts, design, entertainment, sports and media",
    "29": "Healthcare practitioners and technical", "31": "Healthcare support",
    "33": "Protective service", "35": "Food preparation and serving",
    "37": "Building and grounds cleaning and maintenance",
    "39": "Personal care and service", "41": "Sales and related",
    "43": "Office and administrative support", "45": "Farming, fishing and forestry",
    "47": "Construction and extraction", "49": "Installation, maintenance and repair",
    "51": "Production", "53": "Transportation and material moving",
}


def occupations():
    axes = pd.read_csv(AXES_CSV, index_col="onet_soc")
    df = axes[["title"] + [a for a in AXES if a in axes.columns]].copy()

    codes = df.index.str[:2]
    unknown = sorted(set(codes) - set(SOC_GROUPS))
    if unknown:
        raise ValueError(f"SOC major groups not in SOC_GROUPS: {unknown}")
    df.insert(1, "soc_group", codes.map(SOC_GROUPS))
    return df


def loadings():
    L = pd.read_csv(LOADINGS_CSV, index_col=0)
    block = pd.Series(L.index, index=L.index).map(
        lambda c: next((b for p, b in BLOCKS.items() if c.startswith(p)), None))
    if block.isna().any():
        raise ValueError(f"columns in no known block: {list(L.index[block.isna()])}")
    out = L[[a for a in AXES if a in L.columns]].copy()
    out.insert(0, "block", block)
    names = [c.split("__", 1)[1] for c in L.index]
    out.insert(0, "variable", [CTP_NAMES.get(n, n) for n in names])
    return out


def main():
    df = occupations()
    df.to_csv(OUT, float_format="%.6g")
    print(f"[ok] wrote {OUT}: {len(df)} occupations")

    L = loadings()
    L.to_csv(OUT_LOADINGS, index=False, float_format="%.4f")
    print(f"[ok] wrote {OUT_LOADINGS}: {len(L)} variables")


if __name__ == "__main__":
    main()