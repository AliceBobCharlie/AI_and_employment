#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
block_weights.py -- does the structure depend on how much weight each block
carries? (Appendix B)

In the paper's matrix every column has variance 1, so a block weighs in
proportion to its number of columns: Work Context 55, Abilities 52, Work
Activities 41, Skills 35, Knowledge 33. Each scheme below z-scores the 216
columns as before and then multiplies every column of block b by w_b:

  equal columns   w_b = 1, the paper's matrix, as a reference row
  equal blocks    w_b = 1 / sqrt(p_b), so every block has the same total
                  variance whatever its number of columns
  MFA             w_b = 1 / sqrt(lambda_1 of block b), as in multiple factor
                  analysis: every block's leading eigenvalue becomes 1, so no
                  block can dominate the first component through its own
                  internal correlation

The weights are computed once on all 910 occupations and held fixed in the
split-halves. Each scheme is then analysed exactly as an input variant in
input_variants.py: parallel analysis, components reproducing one by one,
nested-subspace stability, the paper's tiers rotated and tested, and
agreement with the paper's six axes.

Run after pca_rotated.py, from the project root:
    python supplementary/block_weights.py
Reads output/master_clean.csv, output/master.csv and output/rotated_axes.csv.
Writes output/block_weights.xlsx and prints every sheet.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from input_variants import analyse, load, matrix, report, IM, CX  # noqa: E402
from pca_rotated import standardize                              # noqa: E402

OUT = Path("output/block_weights.xlsx")


def leading_eigenvalue(Z):
    return np.linalg.svd(Z, compute_uv=False)[0] ** 2 / (len(Z) - 1)


def main():
    clean, raw, paper = load()
    X = matrix(clean, raw, IM + CX)
    blocks = np.array([c.split("__")[0] for c in X.columns])
    Z = standardize(X.to_numpy(float))

    schemes = {"equal columns (the paper's matrix)": {b: 1.0 for b in set(blocks)},
               "equal blocks": {b: 1 / np.sqrt((blocks == b).sum()) for b in set(blocks)},
               "MFA": {b: 1 / np.sqrt(leading_eigenvalue(Z[:, blocks == b])) for b in set(blocks)}}

    rows = []
    for name, wb in schemes.items():
        w = np.array([wb[b] for b in blocks])
        share = {b: (w[blocks == b] ** 2).sum() / (w ** 2).sum() for b in sorted(wb)}
        print(f"[{name}] share of total variance by block: "
              + ", ".join(f"{b} {s:.1%}" for b, s in share.items()))
        rows.append(analyse(name, X.to_numpy(float), paper, w))
    report(rows, OUT)


if __name__ == "__main__":
    main()
