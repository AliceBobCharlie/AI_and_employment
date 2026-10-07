#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_tier1.py -- are the tier-1 axes R1-R3 what we say they are?

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

  3. ENDPOINTS. The occupations at the extremes of R1, R2 and R3, which is the
     plainest way to see whether a label fits.

Then:

  4. ROBUSTNESS. Tier 1 is refitted with the labour-market variables added
     to the 216 feature columns, with wages as two summaries and as all nine
     wage columns, and compared with the published axes.

  5. POLES. The facts the paper states about the poles of R1 and R2: how many
     columns load beyond +/-0.5, composites of the physical and psychomotor
     abilities and of the language skills, and the positions of named
     occupations.

  6. CLUSTERS AGAINST THE AXES. The parts of Section 5.2 that need the axes:
     how the k = 2 split of occupations from cluster_onet.py lines up with
     them; which expert categories the two skill clusters hold and where they
     sit on R1; and the SOC major groups in the space of the three leading
     components -- distances within and between groups, the silhouette of the
     groups, and how much of the feature matrix the groups reconstruct
     compared with the components.

The tier-2 axes are checked by validate_tier2.py, which first removes what
tier 1 can predict of them.

Run after pca_rotated.py and cluster_onet.py. Reads the tier-1 columns of
output/rotated_loadings.csv and output/rotated_axes.csv, output/master_clean.csv, output/cluster_assignments.csv,
output/cluster_skills.csv and the raw O*NET 31.0 blocks (for Element IDs).
Terminal only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pca_rotated import (MASTER, FEATURE_PREFIXES, AXES, LOADINGS, TIER1_ANCHORS,
                        congruence, refit_loadings)

ONET_DIR = Path("data_raw/onet/db_31_0_text")
CLUSTERS = Path("output/cluster_assignments.csv")
SKILL_CLUSTERS = Path("output/cluster_skills.csv")

BLOCK_FILES = {"skills_im": ["Essential Skills", "Transferable Skills"],
               "abilities_im": ["Abilities"], "knowledge_im": ["Knowledge"],
               "workact_im": ["Work Activities"], "workctx_cx": ["Work Context"]}

# Occupations whose positions the paper quotes.
NAMED = {"37-2011.00": "janitors", "35-9021.00": "dishwashers",
         "35-3031.00": "waiters", "41-2011.00": "cashiers",
         "27-3043.05": "poets", "43-9081.00": "proofreaders",
         "23-1012.00": "judicial law clerks", "43-2011.00": "switchboard operators",
         "27-3092.00": "court reporters", "43-9022.00": "word processors and typists",
         "43-9061.00": "general office clerks"}
LANGUAGE_SKILLS = ["skills_im__Reading Comprehension", "skills_im__Active Listening",
                   "skills_im__Writing", "skills_im__Speaking"]

AXIS = {"R1": "physical intensity  (+ physical / - symbolic)",
        "R2": "judgement  (+ judgement / - procedure)",
        "R3": "person-facing  (+ people / - things and systems)"}

# Fallback labels for the O*NET hierarchy, used if Content Model Reference.txt
# is not present: the release 31.0 names. These are the expert groupings, not ours.
FALLBACK = {
    "1.A.1": "Cognitive abilities", "1.A.2": "Psychomotor abilities",
    "1.A.3": "Physical abilities", "1.A.4": "Sensory abilities",
    "2.A.1": "Essential skills: foundational areas",
    "2.A.2": "Essential skills: ways of working",
    "2.B.1": "Social skills", "2.B.2": "Complex problem solving skills",
    "2.B.3": "Technical skills", "2.B.4": "Systems skills",
    "2.B.5": "Resource management skills",
    "2.C.1": "Business and management", "2.C.2": "Manufacturing and production",
    "2.C.3": "Engineering and technology", "2.C.4": "Mathematics and science",
    "2.C.5": "Health services", "2.C.6": "Education and training",
    "2.C.7": "Arts and humanities", "2.C.8": "Law and public safety",
    "2.C.9": "Communications", "2.C.10": "Transportation",
    "4.A.1": "Information input", "4.A.2": "Mental processes",
    "4.A.3": "Work output", "4.A.4": "Interacting with others",
    "4.C.1": "Interpersonal relationships", "4.C.2": "Physical work conditions",
    "4.C.3": "Structural job characteristics",
}


def read_text(path, columns):
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                       usecols=lambda c: c in columns)


def element_categories():
    """Map each element name to its O*NET Content Model category."""
    ref = {}
    cmr = ONET_DIR / "Content Model Reference.txt"
    if cmr.exists():
        c = read_text(cmr, ["Element ID", "Element Name"])
        ref = dict(zip(c["Element ID"], c["Element Name"]))

    name2cat = {}
    for prefix, stems in BLOCK_FILES.items():
        for stem in stems:
            f = ONET_DIR / f"{stem}.txt"
            if not f.exists():
                continue
            d = read_text(f, ["Element ID", "Element Name"]).drop_duplicates()
            for eid, nm in zip(d["Element ID"], d["Element Name"]):
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


def check_robustness(L, m, tiers, heading="4. ROBUSTNESS"):
    """Would the axes change if the labour-market variables took part in
    forming them? They are a handful of columns against 216, so they should
    not -- but that is worth demonstrating rather than asserting. tiers gives
    the components of the axes in L (e.g. [[0, 1, 2]] for R1-R3). Returns the
    congruences."""
    print("\n" + "=" * 72)
    print(f"{heading} -- do the labour-market variables change the axes?")
    print("=" * 72)
    feats = features(m)
    inst = [c for c in ["ext_union_cov_pct", "ext_self_employed_pct",
                        "ext_sep_exit_rate", "ext_sep_transfer_rate",
                        "ext_employment_log"] if c in m.columns]
    wage2 = [c for c in ["ext_wage_level_log", "ext_wage_disp_p90p10"]
             if c in m.columns]
    wage9 = [c for c in m.columns if c.startswith("ext_wage_")
             and c not in wage2]

    variants = {
        "+ labour market, wages as 2 columns": feats + wage2 + inst,
        "+ labour market, all nine wage columns": feats + wage9 + inst,
    }
    print(f"\n{'variant':40s} {'cols':>5s}   congruence with the published axes")
    rows = []
    for name, cols in variants.items():
        Lv = refit_loadings(m[cols].values, cols, tiers)
        shared = [c for c in L.index if c in Lv.index]
        cong = {a: congruence(L.loc[shared, a].values, Lv.loc[shared, a].values)
                for a in L.columns}
        rows.append({"variant": name, "columns": len(cols), **cong})
        print(f"{name:40s} {len(cols):5d}   " +
              "  ".join(f"{a}={c:.3f}" for a, c in cong.items()))
    print("\n  Above ~0.95 means the same axis. If adding the labour-market")
    print("  variables leaves the axes intact, they are a property of the")
    print("  description of the work, and no choice about wages made them.")
    return pd.DataFrame(rows)


def check_poles(L, S, m):
    print("\n" + "=" * 72)
    print("5. POLES -- the facts the paper states about R1 and R2")
    print("=" * 72)
    r1 = L["R1"]
    print(f"\n  R1: {int((r1 > 0.5).sum())} of {len(r1)} columns load above +0.5, "
          f"{int((r1 < -0.5).sum())} below -0.5; most negative: "
          f"{r1.idxmin().split('__')[1]} ({r1.min():+.2f})")

    name2cat = element_categories()
    feats = features(m)
    Z = (m[feats] - m[feats].mean()) / m[feats].std()
    body = [c for c in feats if name2cat.get(tuple(c.split("__", 1))) in
            ("Psychomotor abilities", "Physical abilities")]
    if body:
        comp = Z[body].mean(axis=1).reindex(S.index)
        print(f"  composite of the {len(body)} physical and psychomotor abilities: "
              f"r = {np.corrcoef(comp, S['R1'])[0, 1]:+.3f} with R1")
    else:
        print("  (no category map: physical/psychomotor composite skipped)")
    lang = [c for c in LANGUAGE_SKILLS if c in Z.columns]
    comp = Z[lang].mean(axis=1).reindex(S.index)
    print(f"  composite of the {len(lang)} language skills "
          f"({', '.join(c.split('__')[1] for c in lang)}): "
          f"r = {np.corrcoef(comp, S['R1'])[0, 1]:+.3f} with R1")
    lo = r1[(r1 > -0.53) & (r1 < -0.47)].sort_values()
    print("  columns loading between -0.53 and -0.47 on R1: "
          + "; ".join(f"{k.split('__')[1]} {v:+.2f}" for k, v in lo.items()))

    print("\n  named occupations (R1, R2):")
    for soc, name in NAMED.items():
        if soc in S.index:
            print(f"     {name:28s} R1 {S.at[soc, 'R1']:+.2f}  R2 {S.at[soc, 'R2']:+.2f}"
                  f"   ({S.at[soc, 'title']})")
    print(f"\n  R2 10th percentile: {S['R2'].quantile(0.10):+.2f}")
    office = S.index.str.startswith("43-")
    print(f"  SOC major group 43 (office and administrative support): "
          f"{int((S.loc[office, 'R2'] < 0).sum())} of {int(office.sum())} negative on R2, "
          f"{int((S.loc[office, 'R2'] < S['R2'].quantile(0.10)).sum())} below the 10th percentile")


def features(m):
    return [c for c in m.columns if c.startswith(FEATURE_PREFIXES)]


def mean_distances(X, groups):
    """Mean Euclidean distance between occupations in the same group and in
    different groups."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    same = groups[:, None] == groups[None, :]
    iu = np.triu_indices(len(X), k=1)
    return D[iu][same[iu]].mean(), D[iu][~same[iu]].mean()


def check_clusters(L, S, m):
    print("\n" + "=" * 72)
    print("6. CLUSTERS AGAINST THE AXES (Section 5.2)")
    print("=" * 72)

    # the k = 2 split of occupations
    if CLUSTERS.exists():
        k2 = pd.read_csv(CLUSTERS, index_col=0)["cluster_k2"].reindex(S.index)
        diag = S["R1"] - S["R2"]
        print(f"\n  k = 2 partition: |r| = {abs(np.corrcoef(k2, diag)[0, 1]):.2f} with R1 - R2, "
              f"{abs(np.corrcoef(k2, S['R1'])[0, 1]):.2f} with R1, "
              f"{abs(np.corrcoef(k2, S['R2'])[0, 1]):.2f} with R2, "
              f"{abs(np.corrcoef(k2, S['R3'])[0, 1]):.2f} with R3")
    else:
        print(f"\n  ({CLUSTERS} not found; run cluster_onet.py first)")

    # the two skill clusters: expert categories and position on R1
    if SKILL_CLUSTERS.exists():
        sk = pd.read_csv(SKILL_CLUSTERS)
        name2cat = element_categories()
        sk["category"] = [name2cat.get(tuple(c.split("__", 1)), "?") for c in sk["column"]]
        print("\n  skill clusters by O*NET expert category:")
        tab = pd.crosstab(sk["category"], sk["cluster"])
        print("\n" + tab.loc[tab.sum(axis=1).sort_values(ascending=False).index].to_string())
        sk["R1"] = L["R1"].reindex(sk["column"]).to_numpy()
        print("\n  mean R1 loading per skill cluster:")
        for c, g in sk.groupby("cluster"):
            print(f"    cluster {c} (n={len(g):3d}): R1 = {g['R1'].mean():+.2f}")
    else:
        print(f"\n  ({SKILL_CLUSTERS} not found; run cluster_onet.py first)")

    # SOC major groups in the space of the three leading components
    Z = StandardScaler().fit_transform(m[features(m)].to_numpy(float))
    pca = PCA(random_state=0).fit(Z)
    T3 = pca.transform(Z)[:, :3]                      # unscaled scores, 3 components
    groups = np.array([s[:2] for s in m.index])
    n_groups = len(set(groups))
    within, between = mean_distances(T3, groups)
    print(f"\n  {n_groups} SOC major groups; mean distance within {within:.2f}, "
          f"between {between:.2f}, ratio {within / between:.3f}")
    print(f"  silhouette of the major groups: {silhouette_score(T3, groups):.3f} "
          f"in 3 components, {silhouette_score(Z, groups):.3f} in all {Z.shape[1]} features")
    share = pca.explained_variance_ratio_
    k = n_groups - 1
    means = pd.DataFrame(Z).groupby(groups).transform("mean").to_numpy()
    r2_groups = (means ** 2).sum() / (Z ** 2).sum()
    print(f"  reconstruction of the feature matrix: 3 components {share[:3].sum():.1%}; "
          f"{n_groups} major groups {r2_groups:.1%}; "
          f"{k} components (same degrees of freedom as the groups) {share[:k].sum():.1%}")


def main():
    # tier 1 is the axes the tier-1 anchors sign, R1-R3
    tier1 = [f"R{j + 1}" for j in range(len(TIER1_ANCHORS))]
    L = pd.read_csv(LOADINGS, index_col=0)[tier1]
    S = pd.read_csv(AXES, index_col=0)[["title"] + tier1]
    m = pd.read_csv(MASTER, index_col="onet_soc")
    wage = m["ext_wage_median"].reindex(S.index)

    check_taxonomy(L)
    check_quadrants(S, wage)
    check_endpoints(S)
    check_robustness(L, m, [list(range(L.shape[1]))])
    check_poles(L, S, m)
    check_clusters(L, S, m)


if __name__ == "__main__":
    main()