#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_quality.py -- data-quality scan of the raw O*NET blocks, restricted to the
894 occupations actually kept for analysis. For each block it quantifies three
quality signals so we can see which blocks are low-quality:

  - Recommend Suppress == 'Y' : rating is statistically unreliable
  - Not Relevant       == 'Y' : descriptor doesn't apply to the occupation
  - Domain Source             : is the block OBSERVED (Incumbent / Analyst /
                                Occupational Expert) or GENERATED (Machine
                                Learning / AI/Expert)?

For flags we report the share of rows flagged (per scale where relevant). For
Domain Source we report the share that is model-generated. A block that is
heavily suppressed, heavily not-relevant, or largely generated is low quality.

Reads the 894 SOC list from output/master_wide.xlsx and the raw block files from
RAW_DIR. Terminal output + quality_scan.xlsx.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data_raw/onet/O*NET_30_2_excel")
MASTER = Path("output/master_wide.xlsx")
OUT = Path("output/qc_onet_quality.xlsx")

# raw block files that carry occupation ratings
BLOCK_FILES = [
    "Abilities", "Skills", "Knowledge", "Work Activities", "Work Context",
    "Work Values", "Work Styles", "Interests",
    "Education, Training, and Experience", "Job Zones",
]

GENERATED_PAT = re.compile(r"machine learning|ai/", re.I)


def soc_list():
    df = pd.read_excel(MASTER, usecols=["onet_soc"])
    return set(df["onet_soc"].astype(str))


def scan_block(name, keep):
    path = RAW / f"{name}.xlsx"
    if not path.exists():
        print(f"  [skip] {name}: file not found")
        return None
    df = pd.read_excel(path)
    code_col = "O*NET-SOC Code"
    if code_col not in df.columns:
        print(f"  [skip] {name}: no SOC code column")
        return None
    df = df[df[code_col].astype(str).isin(keep)]        # restrict to the 894
    n = len(df)
    if n == 0:
        return None

    rec = {"block": name, "rows": n}

    # --- Recommend Suppress ---
    if "Recommend Suppress" in df.columns:
        rs = df["Recommend Suppress"].fillna("(blank)")
        y = (rs == "Y").sum()
        rec["suppress_Y_%"] = round(y / n * 100, 2)
        rec["suppress_blank_%"] = round((rs == "(blank)").sum() / n * 100, 2)
    else:
        rec["suppress_Y_%"] = np.nan
        rec["suppress_blank_%"] = np.nan

    # --- Not Relevant ---
    if "Not Relevant" in df.columns:
        nr = df["Not Relevant"].fillna("(blank)")
        rec["not_relevant_Y_%"] = round((nr == "Y").sum() / n * 100, 2)
    else:
        rec["not_relevant_Y_%"] = np.nan

    # --- Domain Source ---
    if "Domain Source" in df.columns:
        src = df["Domain Source"].astype(str)
        gen = src.str.contains(GENERATED_PAT).sum()
        rec["generated_%"] = round(gen / n * 100, 2)
        rec["top_source"] = src.value_counts().index[0]
    else:
        rec["generated_%"] = np.nan
        rec["top_source"] = "(none)"

    return rec


def main():
    keep = soc_list()
    print(f"scanning quality over {len(keep)} analysis occupations\n")

    rows = []
    for name in BLOCK_FILES:
        r = scan_block(name, keep)
        if r:
            rows.append(r)
            print(f"[ok] {name}")

    Q = pd.DataFrame(rows).set_index("block")
    cols = ["rows", "suppress_Y_%", "suppress_blank_%", "not_relevant_Y_%",
            "generated_%", "top_source"]
    Q = Q[[c for c in cols if c in Q.columns]]

    print("\n" + "=" * 78)
    print("DATA-QUALITY SCAN (per block, over the 894 analysis occupations)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + Q.to_string())

    # quality verdicts
    print("\n--- flags ---")
    print("suppress_Y_%    : share of rows O*NET says are statistically unreliable")
    print("not_relevant_Y_%: share of rows marked 'doesn't apply' to the occupation")
    print("generated_%     : share produced by a model (Machine Learning / AI)")

    print("\n--- likely LOW-QUALITY blocks ---")
    for b, r in Q.iterrows():
        reasons = []
        if pd.notna(r.get("generated_%")) and r["generated_%"] > 50:
            reasons.append(f"{r['generated_%']:.0f}% model-generated")
        if pd.notna(r.get("not_relevant_Y_%")) and r["not_relevant_Y_%"] > 10:
            reasons.append(f"{r['not_relevant_Y_%']:.0f}% not-relevant")
        if pd.notna(r.get("suppress_Y_%")) and r["suppress_Y_%"] > 5:
            reasons.append(f"{r['suppress_Y_%']:.0f}% suppressed")
        if reasons:
            print(f"  {b:36s}: {', '.join(reasons)}")

    Q.to_excel(OUT)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()