#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_rotation.py -- the six components rotated together.

The paper rotates in two tiers: varimax within components 1-3, and varimax
within components 4-6. This script rotates all six components in one varimax
and reports the result on its own terms, for readers who want to look at it.

The axes J1-J6 are numbered in their own order of variance, and each is
signed so that its larger pole (by the sum of cubed loadings) is positive.

  axes          each axis's variance share and its strongest variables at
                each pole
  loadings      every feature variable: its block of O*NET's Content Model
                and its loading on J1-J6
  occupations   every occupation: its SOC major group and its score on J1-J6
                (unit variance)

Run after pca_rotated.py, from the project root:
    python supplementary/joint_rotation.py
Reads output/master_clean.csv. Writes output/joint_rotation.xlsx, one sheet
per table above, and prints the first.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import OUTDIR, SEED, load_and_prep, rotate_tier, tier_scores, banner  # noqa: E402
from plot_axes import BLOCKS, CTP_NAMES, SOC_GROUPS  # noqa: E402

OUT = OUTDIR / "joint_rotation.xlsx"
K = 6                 # components rotated together
TOP = 10              # variables listed per pole


def short(col):
    name = col.split("__", 1)[1]
    return CTP_NAMES.get(name, name)


def block(col):
    return next(b for prefix, b in BLOCKS.items() if col.startswith(prefix))


def poles(L, feats, k):
    """The TOP strongest variables at each pole of axis k."""
    s = pd.Series(L[:, k], index=[short(f) for f in feats]).sort_values()
    pos = "; ".join(f"{n} {v:+.2f}" for n, v in s[::-1].head(TOP).items())
    neg = "; ".join(f"{n} {v:+.2f}" for n, v in s.head(TOP).items())
    return pos, neg


def main():
    Xz, feats, title, _ = load_and_prep()

    pca = PCA(random_state=SEED).fit(Xz)
    V, lam = pca.components_.T, pca.explained_variance_
    idx = list(range(K))
    L, R = rotate_tier(V, lam, idx)                  # variance order
    S = tier_scores(pca, Xz, idx, R)
    share = (L ** 2).sum(axis=0) / Xz.shape[1]

    # each axis signed so that its larger pole is positive
    sign = np.where((L ** 3).sum(axis=0) < 0, -1.0, 1.0)
    L, S = L * sign, S * sign
    jnames = [f"J{j + 1}" for j in range(K)]

    axes = []
    for j in range(K):
        pos, neg = poles(L, feats, j)
        axes.append({"axis": jnames[j], "variance share": share[j], "+ pole": pos, "- pole": neg})
    axes = pd.DataFrame(axes)

    loadings = pd.DataFrame(L, columns=jnames)
    loadings.insert(0, "block", [block(f) for f in feats])
    loadings.insert(0, "variable", [short(f) for f in feats])

    occupations = pd.DataFrame(S, index=title.index, columns=jnames)
    occupations.insert(0, "soc_group", title.index.str[:2].map(SOC_GROUPS))
    occupations.insert(0, "title", title)

    banner(f"JOINT VARIMAX OF COMPONENTS 1-{K}")
    for _, r in axes.iterrows():
        print(f"\n  {r['axis']} ({r['variance share']:.1%})\n    + {r['+ pole']}\n    - {r['- pole']}")

    with pd.ExcelWriter(OUT) as writer:
        axes.to_excel(writer, sheet_name="axes", index=False)
        loadings.to_excel(writer, sheet_name="loadings", index=False)
        occupations.to_excel(writer, sheet_name="occupations")
    print(f"\n[ok] wrote {OUT}: {len(loadings)} variables, {len(occupations)} occupations")


if __name__ == "__main__":
    main()