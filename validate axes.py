#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_axes.py -- are the three axes what we say they are?

The axes were named by reading their loadings, which is unavoidably a judgement
call, and every substantive claim rests on those names being right. In
particular the central claim -- that pay tracks cognitive load rather than the
manual/mental divide -- only holds if R1 and R2 really are two different things
rather than two versions of the same one. The data would be unchanged if the
names were wrong; the argument would not survive.

Three checks, each using something the PCA did not see:

  1. O*NET'S OWN TAXONOMY. Element IDs encode the Content Model hierarchy, in
     which experts grouped abilities into cognitive, psychomotor, physical and
     sensory, and work activities into information input, mental processes, work
     output and interacting with others. That grouping took no part in the
     analysis, so it is a fair yardstick. If R1 is a work medium, physical and
     psychomotor abilities should sit at one end of it; if R2 is cognitive load
     rather than a second abstraction axis, mental-process activities should
     load on R2 and not mainly on R1.

  2. THE R1 x R2 QUADRANTS. The naming claims R1 and R2 are separable, which
     requires the off-diagonal cells to be populated: jobs that are physical AND
     cognitively demanding, and jobs that are abstract AND routine. If every
     demanding job turns out to be abstract, "cognitive load" is a misnomer and
     the axis is measuring abstraction twice. Median wage per quadrant then
     shows directly whether pay follows load or follows medium.

  3. ENDPOINTS. The occupations at the extremes of each axis, which is the
     plainest way to see whether a label fits.

Reads output/rotated_loadings.csv, output/rotated_axes.csv,
output/master_clean.xlsx and the raw O*NET blocks (for Element IDs).
Terminal only.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ONET_DIR = Path("data_raw/onet/O*NET_30_2_excel")
LOADINGS = Path("output/rotated_loadings.csv")
SCORES = Path("output/rotated_axes.csv")
MASTER = Path("output/master_clean.xlsx")

BLOCK_FILES = {"skills_im": "Skills", "abilities_im": "Abilities",
               "knowledge_im": "Knowledge", "workact_im": "Work Activities",
               "workctx_cx": "Work Context"}

AXIS = {"R1": "embodiment  (+ physical / - abstract)",
        "R2": "cognitive load  (+ complex / - routine)",
        "R3": "interpersonal  (+ people / - technical)"}

# Fallback labels for the O*NET hierarchy, used if Content Model Reference.xlsx
# is not present. These are the expert groupings, not ours.
FALLBACK = {
    "1.A.1": "Cognitive abilities", "1.A.2": "Psychomotor abilities",
    "1.A.3": "Physical abilities", "1.A.4": "Sensory abilities",
    "2.A.1": "Basic skills: content", "2.A.2": "Basic skills: process",
    "2.B.1": "Social skills", "2.B.2": "Complex problem solving",
    "2.B.3": "Technical skills", "2.B.4": "Systems skills",
    "2.B.5": "Resource management skills",
    "4.A.1": "Information input", "4.A.2": "Mental processes",
    "4.A.3": "Work output", "4.A.4": "Interacting with others",
}


def element_categories():
    """Map each element name to its O*NET Content Model category."""
    ref = {}
    cmr = ONET_DIR / "Content Model Reference.xlsx"
    if cmr.exists():
        c = pd.read_excel(cmr)
        idc = next(x for x in c.columns if "element id" in x.lower())
        nmc = next(x for x in c.columns if "element name" in x.lower())
        ref = dict(zip(c[idc].astype(str), c[nmc].astype(str)))

    name2cat = {}
    for prefix, stem in BLOCK_FILES.items():
        f = ONET_DIR / f"{stem}.xlsx"
        if not f.exists():
            continue
        d = pd.read_excel(f, usecols=lambda c: "Element" in str(c)).drop_duplicates()
        idc = next(x for x in d.columns if "ID" in x)
        nmc = next(x for x in d.columns if "Name" in x)
        for eid, nm in zip(d[idc].astype(str), d[nmc].astype(str)):
            cat_id = ".".join(eid.split(".")[:3])
            label = ref.get(cat_id) or FALLBACK.get(cat_id) or cat_id
            name2cat[(prefix, nm)] = f"{label}"
    return name2cat


def check_taxonomy(L):
    print("=" * 72)
    print("1. O*NET'S OWN TAXONOMY vs the axes")
    print("   (mean loading per expert category; the grouping is O*NET's, not ours)")
    print("=" * 72)
    name2cat = element_categories()
    rows = []
    for col in L.index:
        if "__" not in col:          # the econ/institutional columns have no
            continue                 # O*NET element behind them
        prefix, nm = col.split("__", 1)
        cat = name2cat.get((prefix, nm))
        if cat is None:
            continue
        rows.append({"category": cat, **{a: L.loc[col, a] for a in L.columns}})
    if not rows:
        print("  could not map element names to categories "
              f"(is {ONET_DIR} present?)")
        return
    T = pd.DataFrame(rows).groupby("category").agg(["mean", "size"])
    axes = list(L.columns)
    out = pd.DataFrame({a: T[(a, "mean")] for a in axes})
    out["n"] = T[(axes[0], "size")]
    out = out.sort_values(axes[0], ascending=False)
    with pd.option_context("display.width", 200):
        print("\n" + out.round(2).to_string())

    print("\n  what to look for:")
    print("   - physical / psychomotor / sensory abilities high on R1, cognitive low")
    print("     -> R1 is a work medium, as claimed")
    print("   - mental processes and complex problem solving high on R2 while")
    print("     staying near zero on R1 -> load is separate from medium, which is")
    print("     what the wage claim depends on")
    print("   - interacting with others / social skills high on R3")


def check_quadrants(S, wage):
    print("\n" + "=" * 72)
    print("2. THE R1 x R2 QUADRANTS -- are the off-diagonal cells real?")
    print("=" * 72)
    d = S.join(wage.rename("wage"))
    hi1, hi2 = d["R1"] > d["R1"].median(), d["R2"] > d["R2"].median()
    cells = {
        "physical + demanding": hi1 & hi2,
        "physical + routine": hi1 & ~hi2,
        "abstract + demanding": ~hi1 & hi2,
        "abstract + routine": ~hi1 & ~hi2,
    }
    print(f"\n{'quadrant':24s} {'n':>5s} {'median wage':>13s}")
    for k, m in cells.items():
        print(f"{k:24s} {int(m.sum()):5d} {d.loc[m, 'wage'].median():13,.0f}")

    print("\n  most typical occupations in each quadrant "
          "(furthest from the centre):")
    for k, m in cells.items():
        sub = d[m].copy()
        sub["dist"] = np.hypot(sub["R1"] - d["R1"].median(),
                               sub["R2"] - d["R2"].median())
        top = sub.nlargest(6, "dist")
        print(f"\n  {k}")
        for _, r in top.iterrows():
            print(f"     ${r['wage']:>7,.0f}  {r['title'][:56]}")

    w = {k: d.loc[m, "wage"].median() for k, m in cells.items()}
    load_gap = ((w["physical + demanding"] + w["abstract + demanding"]) / 2 -
                (w["physical + routine"] + w["abstract + routine"]) / 2)
    medium_gap = ((w["abstract + demanding"] + w["abstract + routine"]) / 2 -
                  (w["physical + demanding"] + w["physical + routine"]) / 2)
    print("\n  " + "-" * 68)
    print(f"  pay gap along cognitive load : {load_gap:+,.0f}")
    print(f"  pay gap along work medium    : {medium_gap:+,.0f}")
    print("  The claim is that the first is the large one. If instead the medium")
    print("  gap dominates, pay is tracking manual-vs-mental after all and the")
    print("  interpretation has to change.")


def check_endpoints(S):
    print("\n" + "=" * 72)
    print("3. ENDPOINTS -- do the extreme occupations fit the labels?")
    print("=" * 72)
    for a in [c for c in S.columns if c.startswith("R")]:
        print(f"\n{a}: {AXIS.get(a, '')}")
        print("  + end:")
        for _, r in S.nlargest(8, a).iterrows():
            print(f"     {r[a]:+.2f}  {r['title'][:58]}")
        print("  - end:")
        for _, r in S.nsmallest(8, a).iterrows():
            print(f"     {r[a]:+.2f}  {r['title'][:58]}")


def congruence(a, b):
    """Tucker's phi: how nearly two loading vectors point the same way."""
    den = np.sqrt((a @ a) * (b @ b))
    return abs(a @ b) / den if den > 0 else 0.0


def varimax(Phi, q=100, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d
        Lm = Phi @ R
        u, sv, vt = np.linalg.svd(
            Phi.T @ (Lm**3 - (1.0 / p) * Lm @ np.diag(np.diag(Lm.T @ Lm))))
        R = u @ vt; d = np.sum(sv)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def rotated_loadings(X, cols, k):
    """The same recipe as pca_rotated.py: PCA -> varimax -> order -> sign anchor."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(random_state=0).fit(Xz)
    Lr = varimax(pca.components_[:k].T * np.sqrt(pca.explained_variance_[:k]))
    order = np.argsort(-(Lr ** 2).sum(axis=0))
    Lr = Lr[:, order]
    idx = {c: i for i, c in enumerate(cols)}
    for j, a in enumerate(["abilities_im__Manual Dexterity",
                           "skills_im__Complex Problem Solving",
                           "workact_im__Assisting and Caring for Others"][:k]):
        if a in idx and Lr[idx[a], j] < 0:
            Lr[:, j] *= -1
    return pd.DataFrame(Lr, index=cols, columns=[f"R{j+1}" for j in range(k)])


def check_robustness(L, m):
    """Do the axes survive changing how the economic variables are handled?
    They contribute a few percent of each axis, so they should not be shaping
    them -- but that is worth demonstrating rather than asserting."""
    print("\n" + "=" * 72)
    print("4. ROBUSTNESS -- are the axes an artefact of the wage handling?")
    print("=" * 72)
    skill = [c for c in m.columns if "__" in c and not c.startswith("ete_")]
    ete = [c for c in m.columns if c.startswith("ete_")]
    inst = [c for c in ["ext_union_cov_pct", "ext_self_employed_pct",
                        "ext_sep_exit_rate", "ext_sep_transfer_rate",
                        "ext_employment_log"] if c in m.columns]
    wage2 = [c for c in ["ext_wage_level_log", "ext_wage_disp_p90p10"]
             if c in m.columns]
    wage9 = [c for c in m.columns if c.startswith("ext_wage_")
             and c not in wage2]

    k = L.shape[1]
    variants = {
        "main (wages as 2 columns)": skill + ete + wage2 + inst,
        "no economic variables at all": skill + ete,
        "all nine raw wage columns": skill + ete + wage9 + inst,
    }
    print(f"\n{'variant':32s} {'cols':>5s}   congruence with the published axes")
    for name, cols in variants.items():
        Lv = rotated_loadings(m[cols].values, cols, k)
        shared = [c for c in L.index if c in Lv.index]
        cong = [congruence(L.loc[shared, f"R{j+1}"].values,
                           Lv.loc[shared, f"R{j+1}"].values) for j in range(k)]
        print(f"{name:32s} {len(cols):5d}   " +
              "  ".join(f"R{j+1}={c:.3f}" for j, c in enumerate(cong)))
    print("\n  Above ~0.95 means the same axis. If dropping the economic")
    print("  variables entirely leaves the axes intact, they are a property of")
    print("  the skill and preparation data, and no choice about wages made them.")


def main():
    L = pd.read_csv(LOADINGS, index_col=0)
    S = pd.read_csv(SCORES, index_col=0)
    m = pd.read_excel(MASTER).set_index("onet_soc")
    wage = m["ext_wage_median"].reindex(S.index)

    check_taxonomy(L)
    check_quadrants(S, wage)
    check_endpoints(S)
    check_robustness(L, m)


if __name__ == "__main__":
    main()