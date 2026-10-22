#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corner_table.py -- Table 6.1: the occupations furthest into the two corners of
the three-axis space.

Section 6.1 conjectures that the positive pole of every axis is the less
substitutable one, which makes the least substitutable occupations those
positive on all three axes and the most substitutable those negative on all
three. This script lists the occupations deepest in each of those two corners.

Depth into a corner is the smallest of the three coordinates, taken with the
corner's sign. An occupation ranks high only if it is far from the boundary on
every axis at once. Distance from the origin would not do: it ranks an
occupation extreme on one axis and close to zero on another as deep in the
corner, when it is barely inside it.

Reads output/rotated_axes.csv. Prints a Markdown table to paste into
paper_draft.qmd.
"""

import pandas as pd
from pathlib import Path

AXES = Path("output/rotated_axes.csv")
AXIS_COLS = ["R1", "R2", "R3"]
TOP_N = 10

CORNERS = [
    ("Positive on all three axes", +1),
    ("Negative on all three axes", -1),
]


def signed(value):
    """+1.23 or −1.23, with a true minus sign to match the paper's tables."""
    return f"{value:+.2f}".replace("-", "−")


def corner(axes, sign):
    """Occupations inside the corner, deepest first."""
    signed = axes[AXIS_COLS] * sign
    inside = axes[(signed > 0).all(axis=1)].copy()
    inside["depth"] = signed.loc[inside.index].min(axis=1)
    return inside.sort_values("depth", ascending=False)


def main():
    axes = pd.read_csv(AXES, index_col="onet_soc")

    print("| occupation | physical intensity | judgement | person-facing |")
    # Pandoc sets column widths from the length of these dashes when a row is
    # long; equal dashes would squeeze the occupation titles into a quarter.
    print("|" + "-" * 52 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 16 + "|")
    for label, sign in CORNERS:
        inside = corner(axes, sign)
        print(f"| **{label}** ({len(inside)} occupations) | | | |")
        for _, r in inside.head(TOP_N).iterrows():
            print(f"| {r['title']} | {signed(r['R1'])} | {signed(r['R2'])} "
                  f"| {signed(r['R3'])} |")


if __name__ == "__main__":
    main()
