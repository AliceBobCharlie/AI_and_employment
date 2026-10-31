#!/usr/bin/env python3
"""
joint_rotation.py -- rotate PC1-6 together by one varimax and describe the six
axes, J1-J6.

Input:  data/prepared/master_clean.csv
Output: output/joint_rotation/axes.csv         each axis: variance share and the
                                               TOP strongest columns at each pole
        output/joint_rotation/loadings.csv     every column: its Content Model block
                                               and its loading on J1-J6
        output/joint_rotation/occupations.csv  every occupation: its SOC major group
                                               and its score on J1-J6 (unit variance)

load_and_prep() from pca_rotated.py z-scores the FEATURE_PREFIXES columns.
sklearn's PCA is fitted once. rotate_group() from pca_rotated.py applies
varimax to the loadings of the first K components and sorts the axes by
variance; group_scores() gives their scores. Each axis is flipped so that
the sum of its cubed loadings is positive. poles() lists the TOP columns with
the largest and smallest loadings, named by short() (CTP_NAMES for the CTP
columns). block() maps each column to its block by BLOCKS; SOC_GROUPS names
each occupation's major group by the first two digits of its code.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import OUTDIR, SEED, group_scores, load_and_prep, rotate_group   # noqa: E402

OUT = OUTDIR / "joint_rotation"
K = 6                 # components rotated together
TOP = 10              # columns listed per pole

BLOCKS = {"skills_im__": "Skills", "abilities_im__": "Abilities",
          "knowledge_im__": "Knowledge", "workact_im__": "Work Activities",
          "workctx_cx__": "Work Context", "workctx_ctp__": "Work Context"}

CTP_NAMES = {"Duration of Typical Work Week__c1": "Work week under 40 hours (share)",
             "Duration of Typical Work Week__c3": "Work week over 40 hours (share)",
             "Work Schedules__c2": "Irregular work schedule (share)",
             "Work Schedules__c3": "Seasonal work schedule (share)"}

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


def short(col):
    name = col.split("__", 1)[1]
    return CTP_NAMES.get(name, name)


def block(col):
    return next(b for prefix, b in BLOCKS.items() if col.startswith(prefix))


def poles(L, feats, k):
    """The TOP columns with the largest and the smallest loadings on axis k."""
    s = pd.Series(L[:, k], index=[short(f) for f in feats]).sort_values()
    pos = "; ".join(f"{n} {v:+.2f}" for n, v in s[::-1].head(TOP).items())
    neg = "; ".join(f"{n} {v:+.2f}" for n, v in s.head(TOP).items())
    return pos, neg


def main():
    Xz, feats, title = load_and_prep()
    pca = PCA(random_state=SEED).fit(Xz)
    V, ev = pca.components_.T, pca.explained_variance_
    idx = list(range(K))
    L, R = rotate_group(V, ev, idx)
    S = group_scores(pca, Xz, idx, R)
    share = (L ** 2).sum(axis=0) / Xz.shape[1]
    sign = np.where((L ** 3).sum(axis=0) < 0, -1.0, 1.0)
    L, S = L * sign, S * sign
    names = [f"J{j + 1}" for j in range(K)]

    axes = pd.DataFrame([{"axis": names[j], "variance share": share[j],
                          **dict(zip(("+ pole", "- pole"), poles(L, feats, j)))} for j in range(K)])
    loadings = pd.DataFrame(L, columns=names)
    loadings.insert(0, "block", [block(f) for f in feats])
    loadings.insert(0, "column", [short(f) for f in feats])
    occupations = pd.DataFrame(S, index=title.index, columns=names)
    occupations.insert(0, "soc_group", title.index.str[:2].map(SOC_GROUPS))
    occupations.insert(0, "title", title)

    OUT.mkdir(parents=True, exist_ok=True)
    axes.to_csv(OUT / "axes.csv", index=False)
    loadings.to_csv(OUT / "loadings.csv", index=False)
    occupations.to_csv(OUT / "occupations.csv")
    print(f"[ok] rotated PC1-{K} together; wrote axes.csv, loadings.csv and occupations.csv to {OUT}/")


if __name__ == "__main__":
    main()