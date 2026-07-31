#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_onet.py -- is the reported polarization a property of workplace skills,
or of the way the skill matrix is constructed?

Alabdulkareem et al. (Sci. Adv. 2018) build a "Skillscape": skills are nodes,
edges are pairwise complementarity, and the network splits into two communities,
social-cognitive and sensory-physical. The construction has three steps that are
easy to overlook: the occupation x skill ratings are binarised by revealed
comparative advantage (RCA > 1), complementarity is defined as the minimum
conditional probability of two skills co-occurring, and the resulting network is
thresholded before communities are read off. The paper itself notes that this
produces a bimodal distribution of complementarity, unlike other applications of
RCA -- which is exactly the observation worth interrogating, because a bimodal
edge-weight distribution is what makes two communities visible.

This script holds the data fixed and varies only the construction:

  PART A  occupations. Do occupations themselves fall into groups, under either
          continuous or binarised ratings? This is what licenses describing them
          with continuous axes rather than types.

  PART B  skills. Rebuilds skill-skill similarity two ways -- the RCA-and-
          minimum-conditional-probability route, and a straightforward
          correlation across occupations on the published values -- then
          compares how bimodal each is (Sarle's coefficient) and how separable
          the resulting skill clusters are (silhouette). If the two communities
          appear only via the first route, the polarization is a feature of the
          pipeline rather than of skills.

Not a replication: community detection on a thresholded graph is replaced by
clustering on the similarity matrix, so the comparison between representations
is like for like. The point is the contrast between the two columns, not the
absolute numbers.

Reads output/master_clean.xlsx -> terminal, output/cluster_*.png,
output/cluster_assignments.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

MASTER = Path("output/master_clean.xlsx")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
K_RANGE = range(2, 11)
THETA_CUT = 0.6          # the threshold used to draw the published network
SEED = 0

# The four rating blocks that describe what a job requires. Work Context is left
# out here: it records the conditions work happens under, not skills, and was
# not part of the original skill network.
SKILL_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")


# --------------------------------------------------------------------------- #
def rca_binarise(M):
    """RCA > 1: does this occupation lean on this skill more than the labour
    market does on average? Ratings are positive, so the shares are well defined."""
    M = M.astype(float)
    rca = (M / M.sum(1, keepdims=True)) / (M.sum(0, keepdims=True) / M.sum())
    return (rca > 1).astype(float)


def theta_min_conditional(B):
    """Complementarity as in the Skillscape: for each pair of skills, the
    smaller of the two conditional probabilities that one is used given the
    other. Takes the binarised matrix."""
    co = B.T @ B                       # skill x skill co-occurrence counts
    use = B.sum(0)                     # how many occupations use each skill
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = co / use[None, :]       # P(row | column)
    cond = np.nan_to_num(cond)
    theta = np.minimum(cond, cond.T)
    np.fill_diagonal(theta, 1.0)
    return theta


def bimodality(v):
    """Sarle's bimodality coefficient. Above ~0.555 points to two modes; the
    uniform distribution sits at that value and the normal well below it."""
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g ** 2 + 1) / denom


def offdiag(S):
    iu = np.triu_indices_from(S, k=1)
    return S[iu]


# --------------------------------------------------------------------------- #
def part_a_occupations(M):
    print("\n" + "=" * 66)
    print("PART A -- do OCCUPATIONS fall into groups?")
    print("=" * 66)
    cont = StandardScaler().fit_transform(M)
    binr = rca_binarise(M)
    pcs = PCA(n_components=10, random_state=SEED).fit_transform(cont)

    curves = {}
    for label, X in [("continuous", cont), ("binarised (RCA>1)", binr),
                     ("continuous, 10 PCs", pcs)]:
        sil = {}
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            sil[k] = silhouette_score(X, km.labels_)
        best = max(sil, key=sil.get)
        curves[label] = sil
        print(f"  {label:22s} best k={best}, silhouette={sil[best]:.3f}")
    print("\n  Silhouettes far below 0.5 in every representation: occupations sit")
    print("  on a continuum, and any partition of them is imposed rather than found.")
    return cont, curves


def describe_clusters(names, lab, prefixes):
    """Which expert categories do the two skill clusters correspond to, and do
    they line up with the embodiment axis? If the skills split into the two
    groups the published work reports, and that split coincides with R1, then
    a dichotomy in skill space and a continuum in occupation space are two
    readings of one structure rather than competing claims."""
    try:
        from validate_axes import element_categories
        name2cat = element_categories()
    except Exception as e:
        print(f"  (cannot map to O*NET categories: {e})")
        return
    cats = [name2cat.get((p, n), "?") for p, n in zip(prefixes, names)]

    print("\n  cluster composition by O*NET expert category:")
    tab = pd.crosstab(pd.Series(cats, name="category"),
                      pd.Series(lab, name="cluster"))
    tab = tab.loc[tab.sum(1).sort_values(ascending=False).index]
    print("\n" + tab.to_string())

    load = Path("output/rotated_loadings.csv")
    if load.exists():
        L = pd.read_csv(load, index_col=0)
        key = [f"{p}__{n}" for p, n in zip(prefixes, names)]
        r1 = pd.Series([L["R1"].get(k, np.nan) for k in key])
        print("\n  mean R1 (embodiment) loading per cluster:")
        for c in sorted(set(lab)):
            m = np.array(lab) == c
            print(f"    cluster {c} (n={m.sum():3d}): R1 = {r1[m].mean():+.2f}")
        print("  Clusters separated on R1 mean the skill dichotomy IS the")
        print("  embodiment axis, seen from the skill side instead of the job side.")


def part_b_skills(M, names, prefixes):
    print("\n" + "=" * 66)
    print("PART B -- do SKILLS fall into two clusters, and does that depend")
    print("          on how skill-skill similarity is built?")
    print("=" * 66)

    B = rca_binarise(M)
    theta = theta_min_conditional(B)                     # the published route
    corr = np.corrcoef(StandardScaler().fit_transform(M).T)  # published values

    print(f"  binarised matrix keeps {B.mean()*100:.1f}% of cells as 1")

    results = {}
    for label, S in [("RCA + min conditional prob.", theta),
                     ("correlation on published values", corr)]:
        w = offdiag(S)
        bc = bimodality(w)
        D = 1 - S
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0, None)
        try:                       # sklearn >= 1.4
            lab = AgglomerativeClustering(
                n_clusters=2, metric="precomputed",
                linkage="average").fit_predict(D)
        except TypeError:          # older sklearn calls the argument 'affinity'
            lab = AgglomerativeClustering(
                n_clusters=2, affinity="precomputed",
                linkage="average").fit_predict(D)
        sil = silhouette_score(D, lab, metric="precomputed")
        sizes = np.bincount(lab)
        print(f"\n  {label}")
        print(f"    similarity spread : min {w.min():+.2f}  median {np.median(w):+.2f}"
              f"  max {w.max():+.2f}")
        print(f"    bimodality coeff. : {bc:.3f}"
              f"   ({'bimodal' if bc > 0.555 else 'not bimodal'}; 0.555 = uniform)")
        print(f"    2-cluster split   : sizes {sizes.tolist()}, "
              f"silhouette {sil:.3f}")
        results[label] = dict(w=w, bc=bc, sil=sil, lab=lab)
        if label.startswith("correlation"):
            describe_clusters(names, lab, prefixes)

    # the published network is read after thresholding, so check that step too
    keep = theta > THETA_CUT
    np.fill_diagonal(keep, False)
    frac = keep.sum() / (keep.size - len(keep))
    print(f"\n  thresholding theta at {THETA_CUT} keeps {frac*100:.1f}% of skill pairs")
    print("  (the published network is drawn after this step, so whatever the")
    print("   threshold removes is not visible in the published figure)")

    a, b = results["RCA + min conditional prob."], results["correlation on published values"]
    print("\n  " + "-" * 62)
    print(f"  bimodality : {a['bc']:.3f} via RCA vs {b['bc']:.3f} on published values"
          f"   (difference {a['bc'] - b['bc']:+.3f})")
    print(f"  silhouette : {a['sil']:.3f} via RCA vs {b['sil']:.3f} on published values"
          f"   (difference {a['sil'] - b['sil']:+.3f})")
    print("  A large positive difference means the two-cluster picture is made by")
    print("  the construction. A small one means it is in the skills themselves.")
    return results


def figures(curves, results):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, sil in curves.items():
        ax.plot(list(K_RANGE), [sil[k] for k in K_RANGE], "o-", label=label)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.text(K_RANGE[-1], 0.51, "well separated", ha="right", fontsize=8, color="grey")
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette")
    ax.set_title("Occupations: cluster separability by representation")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTDIR / "cluster_occupations.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (label, r) in zip(axes, results.items()):
        ax.hist(r["w"], bins=60, color="#4C72B0")
        ax.set_title(f"{label}\nbimodality {r['bc']:.3f}", fontsize=10)
        ax.set_xlabel("pairwise skill similarity"); ax.set_ylabel("pairs")
    fig.suptitle("Where the two communities come from: the edge-weight distribution",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "cluster_skill_similarity.png", dpi=130); plt.close(fig)


def main():
    df = pd.read_excel(MASTER).set_index("onet_soc")
    skill = [c for c in df.columns if c.startswith(SKILL_PREFIXES)]
    M = df[skill].values
    names = [c.split("__", 1)[1] for c in skill]
    prefixes = [c.split("__", 1)[0] for c in skill]
    print(f"{M.shape[0]} occupations x {len(skill)} skill ratings")

    cont, curves = part_a_occupations(M)
    results = part_b_skills(M, names, prefixes)
    figures(curves, results)

    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(cont)
    pd.DataFrame({"title": df["title"], "cluster_k2": km.labels_},
                 index=df.index).to_csv(OUTDIR / "cluster_assignments.csv")
    print("\n[ok] wrote cluster_occupations.png, cluster_skill_similarity.png, "
          "cluster_assignments.csv")


if __name__ == "__main__":
    main()