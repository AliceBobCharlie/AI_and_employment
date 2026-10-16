#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_rotation.py -- what if the six components are rotated together?

The paper rotates in two tiers: varimax within components 1-3 (R1-R3), and
varimax within components 4-6 (R4-R6). Rotating all six components in one varimax is the obvious
alternative. Varimax then trades variance between the tiers, and may move
part of R1-R3 into an axis of the second tier. This script asks whether it
does, what moves, and whether R1-R3 keep their meaning without it.

  1. THE JOINT AXES. Varimax within components 1-6 together. Each axis's variance share, its strongest variables at each pole, and its
     split-half congruence (each half decomposed and rotated from scratch,
     200 splits, as in pca_rotated.py).
  2. AGAINST THE TWO-TIER AXES. Loading congruence and score correlation of
     every joint axis with every paper axis R1-R6, and a one-to-one pairing
     (Hungarian, on loading congruence). Each joint axis is named J1-J6 after
     the paper axis it pairs with and signed to agree with it.
  3. WHAT MOVED. For each of R1-R3: the variables whose loading changes most
     between the paper axis and its joint counterpart, with their loading on
     the other joint axes, so that a variable moving out of an axis can be
     followed to where it went.
  4. THE MANAGEMENT QUESTION. The SOC management group (11-) on every axis
     of both solutions: its mean score, and how many of its occupations
     are in the top tenth. If the joint rotation moves management out of
     R2 or R3, management occupations score lower on J2 or J3 than on R2 or
     R3, and the axis that took them up correlates with R2 or R3.
  5. DOES THE READING HOLD? For each of R1-R3 and its joint counterpart, the
     occupations at each end, side by side.
  6. THE LABOUR-MARKET VARIABLES (the held-out columns of pca_rotated.py) on
     both solutions: each one's correlation with every axis, and the R2 of an
     ordinary least-squares regression on R1-R3, on R1-R6 and on J1-J6 (all
     axes are uncorrelated with unit variance, so a coefficient is the
     correlation and R2 is the sum of the squared correlations). Both
     solutions span components 1-6, so the two R2 on six axes are equal. Then the median wage in the four quarters of the physical x
     judgement plane, on both solutions (Section 5.7).

Run after pca_rotated.py, from the project root:
    python supplementary/joint_rotation.py
Reads output/master_clean.csv, output/rotated_loadings.csv and
output/rotated_axes.csv. Writes output/joint_rotation.xlsx and prints every
table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import (OUTDIR, SEED, AXES, LOADINGS, MASTER, ECON,  # noqa: E402
                         load_and_prep, rotate_tier,
                         tier_scores, rotated_stability, congruence, banner)

OUT = OUTDIR / "joint_rotation.xlsx"
K = 6                 # components rotated together
TOP = 10              # variables listed per pole
TOP_SHARE = 0.10      # "top tenth" of occupations on an axis
N_ENDS = 8            # occupations listed at each end of an axis
MANAGEMENT = "11"     # SOC major group


def short(col):
    return col.split("__", 1)[1]


def poles(L, feats, k):
    """The TOP strongest variables at each pole of axis k."""
    s = pd.Series(L[:, k], index=[short(f) for f in feats]).sort_values()
    pos = "; ".join(f"{n} {v:+.2f}" for n, v in s[::-1].head(TOP).items())
    neg = "; ".join(f"{n} {v:+.2f}" for n, v in s.head(TOP).items())
    return pos, neg


def main():
    Xz, feats, title, supp = load_and_prep()
    paper_L = pd.read_csv(LOADINGS, index_col=0).reindex(feats)
    paper_S = pd.read_csv(AXES, index_col=0).reindex(title.index)
    names = [c for c in paper_L.columns if c.startswith("R")]
    P_L = paper_L[names].to_numpy()
    P_S = paper_S[names].to_numpy()

    # ---- 1. the joint axes
    pca = PCA(random_state=SEED).fit(Xz)
    V, lam = pca.components_.T, pca.explained_variance_
    idx = list(range(K))
    L, R = rotate_tier(V, lam, idx)                  # variance order
    S = tier_scores(pca, Xz, idx, R)
    share = (L ** 2).sum(axis=0) / Xz.shape[1]
    p05, med = rotated_stability(Xz, [idx])          # same variance order

    # ---- 2. pair with the paper's axes and name and sign accordingly
    C = np.array([[congruence(L[:, j], P_L[:, i]) for i in range(len(names))] for j in range(K)])
    rows, cols = linear_sum_assignment(-C)
    pair = dict(zip(rows, cols))
    sign = np.array([np.sign(L[:, j] @ P_L[:, pair[j]]) or 1.0 for j in range(K)])
    L, S = L * sign, S * sign
    jnames = [f"J{pair[j] + 1}" for j in range(K)]

    banner(f"1. JOINT VARIMAX OF COMPONENTS 1-{K}, in order of variance")
    joint = pd.DataFrame({
        "axis": jnames, "variance share": share,
        "paired with": [names[pair[j]] for j in range(K)],
        "loading congruence with it": [C[j, pair[j]] for j in range(K)],
        "score correlation with it": [np.corrcoef(S[:, j], P_S[:, pair[j]])[0, 1] for j in range(K)],
        "split-half congruence p05": p05, "split-half median": med})
    print(joint.round(3).to_string(index=False))
    print("\n  paper axes, variance share: " + ", ".join(
        f"{n} {v:.1%}" for n, v in zip(names, (P_L ** 2).sum(axis=0) / Xz.shape[1])))

    banner("1b. STRONGEST VARIABLES OF EACH JOINT AXIS")
    pole_rows = []
    for j in range(K):
        pos, neg = poles(L, feats, j)
        pole_rows.append({"axis": jnames[j], "+ pole": pos, "- pole": neg})
        print(f"\n  {jnames[j]} ({share[j]:.1%})\n    + {pos}\n    - {neg}")

    # ---- 2. full matrices
    loading_cong = pd.DataFrame(C, index=jnames, columns=names)
    score_corr = pd.DataFrame([[np.corrcoef(S[:, j], P_S[:, i])[0, 1] for i in range(len(names))]
                               for j in range(K)], index=jnames, columns=names)
    banner("2. EVERY JOINT AXIS AGAINST EVERY PAPER AXIS")
    print("  loading congruence (unsigned)")
    print(loading_cong.round(2).to_string())
    print("\n  score correlation (signed; joint axes signed to agree with their pair)")
    print(score_corr.round(2).to_string())

    # ---- 3. what moved out of R1-R3
    banner("3. WHAT MOVED: largest loading changes from R1-R3 to their joint counterparts")
    moved = []
    for i, n in enumerate(names[:3]):
        j = rows[list(cols).index(i)]
        diff = L[:, j] - P_L[:, i]
        for v in np.argsort(-np.abs(diff))[:TOP]:
            others = {jn: L[v, jj] for jj, jn in enumerate(jnames) if jj != j}
            dest = max(others, key=lambda a: abs(others[a]))
            moved.append({"paper axis": n, "variable": short(feats[v]),
                          "paper loading": P_L[v, i], "joint loading": L[v, j],
                          "change": diff[v], "largest other joint loading":
                          f"{dest} {others[dest]:+.2f}"})
    moved = pd.DataFrame(moved)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        for n, g in moved.groupby("paper axis", sort=False):
            print(f"\n  {n}")
            print(g.drop(columns="paper axis").round(2).to_string(index=False))

    # ---- 4. management occupations on both solutions
    banner("4. MANAGEMENT OCCUPATIONS (SOC 11) ON BOTH SOLUTIONS")
    mgmt = title.index.str.startswith(MANAGEMENT + "-")
    cut = int(round(TOP_SHARE * len(S)))
    mg = []
    for label, M, axn in (("paper", P_S, names), ("joint", S, jnames)):
        for k, a in enumerate(axn):
            top = np.argsort(-M[:, k])[:cut]
            mg.append({"solution": label, "axis": a, "management mean score": M[mgmt, k].mean(),
                       f"management share of top {TOP_SHARE:.0%}": mgmt[top].mean(),
                       "management share of all occupations": mgmt.mean()})
    mg = pd.DataFrame(mg)
    print(f"  {mgmt.sum()} management occupations of {len(mgmt)}")
    print(mg.round(3).to_string(index=False))

    # ---- 5. the occupations at each end, paper axis beside its joint counterpart
    banner("5. OCCUPATIONS AT EACH END: paper axis and its joint counterpart")
    ends = []
    for i, n in enumerate(names[:3]):
        j = rows[list(cols).index(i)]
        for label, v in ((n, P_S[:, i]), (jnames[j], S[:, j])):
            order = np.argsort(v)
            ends.append({"axis": label,
                         "+ end": "; ".join(title.iloc[order[::-1][:N_ENDS]]),
                         "- end": "; ".join(title.iloc[order[:N_ENDS]])})
    ends = pd.DataFrame(ends)
    for _, r in ends.iterrows():
        print(f"\n  {r['axis']}\n    + {r['+ end']}\n    - {r['- end']}")

    # ---- 6. labour-market variables on both solutions
    banner("6. LABOUR-MARKET VARIABLES: correlations and R2 on both solutions")
    by_name = {jn: S[:, j] for j, jn in enumerate(jnames)}
    J = np.column_stack([by_name[f"J{i + 1}"] for i in range(K)])          # J1..J6 in order
    jcols = [f"J{i + 1}" for i in range(K)]

    def r2(y, A):
        A = np.column_stack([np.ones(len(y)), A])
        fit = A @ np.linalg.lstsq(A, y, rcond=None)[0]
        return 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    lm = []
    for c in [c for c in ECON if c in supp.columns]:
        y = supp[c].to_numpy(float)
        row = {"variable": c.replace("ext_", "")}
        row.update({n: np.corrcoef(y, P_S[:, i])[0, 1] for i, n in enumerate(names)})
        row.update({n: np.corrcoef(y, J[:, i])[0, 1] for i, n in enumerate(jcols)})
        row.update({"R2 on R1-R3": r2(y, P_S[:, :3]), "R2 on R1-R6": r2(y, P_S),
                    "R2 on J1-J6": r2(y, J)})
        lm.append(row)
    lm = pd.DataFrame(lm)
    with pd.option_context("display.width", 250):
        print(lm.round(2).to_string(index=False))

    wage = pd.read_csv(MASTER, index_col="onet_soc")["ext_wage_median"].reindex(title.index).to_numpy()
    quarters = []
    for label, x, y in (("paper R1 x R2", P_S[:, 0], P_S[:, 1]), ("joint J1 x J2", J[:, 0], J[:, 1])):
        hi_x, hi_y = x > np.median(x), y > np.median(y)
        cell = {(a, b): np.median(wage[(hi_x == a) & (hi_y == b)]) for a in (False, True) for b in (False, True)}
        n = {(a, b): int(((hi_x == a) & (hi_y == b)).sum()) for a in (False, True) for b in (False, True)}
        quarters.append({
            "plane": label,
            "abstract, procedural": cell[(False, False)], "abstract, judgement": cell[(False, True)],
            "physical, procedural": cell[(True, False)], "physical, judgement": cell[(True, True)],
            "n (ap, aj, pp, pj)": f"{n[(False, False)]}, {n[(False, True)]}, {n[(True, False)]}, {n[(True, True)]}",
            "gap along judgement": (cell[(False, True)] + cell[(True, True)] - cell[(False, False)] - cell[(True, False)]) / 2,
            "gap along physical": (cell[(True, False)] + cell[(True, True)] - cell[(False, False)] - cell[(False, True)]) / 2})
    quarters = pd.DataFrame(quarters)
    print("\n  median wage in the four quarters (split at each axis's median)")
    with pd.option_context("display.width", 250):
        print(quarters.round(0).to_string(index=False))

    with pd.ExcelWriter(OUT) as writer:
        joint.to_excel(writer, sheet_name="joint_axes", index=False)
        pd.DataFrame(pole_rows).to_excel(writer, sheet_name="joint_poles", index=False)
        loading_cong.to_excel(writer, sheet_name="loading_congruence")
        score_corr.to_excel(writer, sheet_name="score_correlation")
        moved.to_excel(writer, sheet_name="what_moved", index=False)
        mg.to_excel(writer, sheet_name="management", index=False)
        ends.to_excel(writer, sheet_name="ends", index=False)
        lm.to_excel(writer, sheet_name="labour_market", index=False)
        quarters.to_excel(writer, sheet_name="wage_quarters", index=False)
        pd.DataFrame(L, index=feats, columns=jnames).to_excel(writer, sheet_name="joint_loadings")
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
