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

  PART C  local structure beyond tier 1. If the components after the first
          three were unstable because they describe features only a few
          occupations have (a licence, a clinical setting), the occupations
          should form a small, separate group along some direction of those
          components. k-means and the silhouette are poor at finding that: with
          one large and one small group, k-means tends to cut the large group
          in half. Kurtosis is not: a small group off to one side makes a heavy
          tail (high excess kurtosis), two groups of similar size make a flat or
          two-humped distribution (negative excess kurtosis). So for each
          subspace -- PC1-3, PC4-6, PC4-6 with the curvature of PC1-3 removed,
          PC7-14, PC4-14 and PC1-14 -- the scores are whitened and the
          directions of largest and smallest excess kurtosis are searched for
          (projection pursuit with kurtosis as the index: L-BFGS on the unit
          sphere, upwards and downwards, from 30 random starts). Searching more dimensions finds more
          extreme values by chance, so each subspace is compared with the same
          search on Gaussian data of the same size and dimension. For the
          heaviest-tailed direction: the share of occupations beyond 3 sd, the
          kurtosis after dropping the five most extreme occupations (a lone
          outlier collapses, a group does not), and the occupations in the
          tail; for the flattest: Sarle's bimodality coefficient. Each
          component PC1-14 is also scored on its own.

Not a replication: community detection on a thresholded graph is replaced by
clustering on the similarity matrix, so the comparison between representations
is like for like. The point is the contrast between the two columns, not the
absolute numbers.

Nothing here uses the rotated axes, which are estimated afterwards
(pca_rotated.py); Part C uses the unrotated components, so it does not depend
on any rotation.
The comparisons that do need them -- how the two-cluster split of occupations
and the two skill clusters line up with the axes, and the SOC major groups in
the space of the leading components -- are in validate_tier1.py, which reads the
two assignment files written here.

Reads output/master_clean.csv -> terminal, output/cluster_occupations.png,
output/cluster_skill_similarity.png, output/cluster_assignments.csv (the k = 2
split of occupations), output/cluster_skills.csv (the two skill clusters on
the correlation construction), output/cluster_kurtosis.xlsx (Part C)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

MASTER = Path("output/master_clean.csv")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
K_RANGE = range(2, 11)
THETA_CUT = 0.6          # the threshold used to draw the published network
SEED = 0
N_STARTS = 30            # random starts for the kurtosis search (Part C)
N_NULL = 30              # Gaussian samples for the search's null (Part C)
N_COMPONENTS = 14        # components above the parallel-analysis null
TAIL_SD = 3              # a Gaussian puts 0.27% of cases beyond 3 sd
N_TRIM = 5               # extreme occupations dropped in the outlier check
SUBSPACES = {"PC1-3 (tier 1)": (0, 3), "PC4-6 (tier 2)": (3, 6),
             "PC4-6 without curvature of PC1-3": (3, 6), "PC7-14": (6, 14),
             "PC4-14 (all after tier 1)": (3, 14), "PC1-14": (0, 14)}

# The four rating blocks that describe what a job requires. Work Context is left
# out here: it records the conditions work happens under, not skills, and was
# not part of the original skill network.
SKILL_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")
# Part A uses the feature matrix the axes are estimated from: the four blocks
# above on Importance, plus Work Context on the Context scale.
FEATURE_PREFIXES = SKILL_PREFIXES + ("workctx_cx__",)


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

    print("\n  silhouette by k")
    print(f"  {'k':22s}" + "".join(f"{k:>7d}" for k in K_RANGE))
    for label, sil in curves.items():
        print(f"  {label:22s}" + "".join(f"{sil[k]:>7.3f}" for k in K_RANGE))

    print("\n  Silhouettes far below 0.5 in every representation: occupations sit")
    print("  on a continuum, and any partition of them is imposed rather than found.")
    return cont, curves


def part_b_skills(M):
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


# --------------------------------------------------------------------------- #
def whiten(Y):
    """Centred scores rotated to uncorrelated unit-variance coordinates."""
    Y = Y - Y.mean(axis=0)
    u, s, vt = np.linalg.svd(Y, full_matrices=False)
    return u * np.sqrt(len(Y))


def climb(W, w, sign):
    """Local maximum (sign +1) or minimum (sign -1) of the excess kurtosis of
    W @ w over unit vectors w, from the start w, by L-BFGS on w / |w|. For
    whitened W every unit direction has mean 0 and variance 1, so the excess
    kurtosis is E[(W w)^4] - 3 and its gradient on the sphere is
    4 (I - w w') E[W (W w)^3]."""
    def objective(v):
        r = np.linalg.norm(v)
        u = v / r
        y = W @ u
        g = (W * (y ** 3)[:, None]).mean(axis=0)
        grad = 4 * (g - (g @ u) * u) / r
        return -sign * ((y ** 4).mean() - 3), -sign * grad
    v = minimize(objective, w, jac=True, method="L-BFGS-B").x
    return v / np.linalg.norm(v)


def kurtosis_search(W, rng):
    """The directions of largest and smallest excess kurtosis of W @ w over
    unit vectors w (W whitened), the best of N_STARTS random starts each.
    Returns (largest kurtosis, its direction), (smallest, its direction)."""
    hi, lo = (-np.inf, None), (np.inf, None)
    for _ in range(N_STARTS):
        w0 = rng.standard_normal(W.shape[1])
        w0 /= np.linalg.norm(w0)
        for sign in (1, -1):
            w = climb(W, w0, sign)
            k = kurtosis(W @ w, fisher=True)
            if sign == 1 and k > hi[0]:
                hi = (k, w)
            if sign == -1 and k < lo[0]:
                lo = (k, w)
    return hi, lo


def tail_report(y):
    """Orient y so its longer tail is positive; share beyond TAIL_SD and the
    kurtosis without the N_TRIM most extreme values on that side."""
    if skew(y) < 0:
        y = -y
    keep = np.argsort(y)[:-N_TRIM]
    return y, (np.abs(y) > TAIL_SD).mean(), kurtosis(y[keep], fisher=True)


def part_c_local(cont, titles, feats):
    print("\n" + "=" * 66)
    print("PART C -- is there LOCAL structure (a small separate group) after tier 1?")
    print("=" * 66)
    rng = np.random.default_rng(SEED)
    n = len(cont)
    _, s, vt = np.linalg.svd(cont - cont.mean(axis=0), full_matrices=False)
    T = (cont - cont.mean(axis=0)) @ vt.T[:, :N_COMPONENTS] / (s[:N_COMPONENTS] / np.sqrt(n))

    # each component on its own, against Gaussian samples of the same size
    null_one = np.array([kurtosis(rng.standard_normal(n), fisher=True) for _ in range(2000)])
    rows = []
    for j in range(N_COMPONENTS):
        y = T[:, j]
        rows.append({"component": f"PC{j + 1}", "skewness": skew(y),
                     "excess kurtosis": kurtosis(y, fisher=True),
                     f"share beyond {TAIL_SD} sd": (np.abs(y) > TAIL_SD).mean(),
                     "bimodality coefficient": bimodality(y)})
    comp = pd.DataFrame(rows)
    print(f"\n  each component on its own (Gaussian, n = {n}: excess kurtosis "
          f"p2.5-p97.5 {np.percentile(null_one, 2.5):+.2f} to {np.percentile(null_one, 97.5):+.2f}, "
          f"{2 * norm.sf(TAIL_SD):.2%} beyond {TAIL_SD} sd)")
    print(comp.round(3).to_string(index=False))

    null_cache, rows, tails = {}, [], []
    for name, (a, z) in SUBSPACES.items():
        Y = T[:, a:z]
        if "without curvature" in name:
            basis = PolynomialFeatures(3, include_bias=True).fit_transform(T[:, :3])
            Q, _ = np.linalg.qr(basis)
            Y = Y - Q @ (Q.T @ Y)
        W = whiten(Y)
        d = W.shape[1]
        (k_hi, w_hi), (k_lo, w_lo) = kurtosis_search(W, rng)
        if d not in null_cache:
            hi, lo, trim = [], [], []
            for _ in range(N_NULL):
                G = whiten(rng.standard_normal((n, d)))
                (kh, wh), (kl, _) = kurtosis_search(G, rng)
                hi.append(kh)
                lo.append(kl)
                trim.append(tail_report(G @ wh)[2])
            null_cache[d] = (np.array(hi), np.array(lo), np.array(trim))
        null_hi, null_lo, null_trim = null_cache[d]
        y, beyond, trimmed = tail_report(W @ w_hi)
        rows.append({"subspace": name, "dimensions": d,
                     "largest excess kurtosis": k_hi, "Gaussian null p95": np.percentile(null_hi, 95),
                     f"share beyond {TAIL_SD} sd": beyond,
                     f"kurtosis without {N_TRIM} most extreme": trimmed,
                     "Gaussian null p95 (trimmed)": np.percentile(null_trim, 95),
                     "smallest excess kurtosis": k_lo, "Gaussian null p05": np.percentile(null_lo, 5),
                     "bimodality coefficient (flattest direction)": bimodality(W @ w_lo)})
        rows[-1]["occupations per 0.5 sd from +1 sd (a gap or a second hump marks a group)"] = \
            " ".join(str(c) for c in np.histogram(y, bins=np.arange(1, 8.5, 0.5))[0])
        r = np.array([np.corrcoef(y, cont[:, j])[0, 1] for j in range(cont.shape[1])])
        rows[-1]["columns most correlated with the tail"] = "; ".join(
            f"{feats[j].split('__')[1]} {r[j]:+.2f}" for j in np.argsort(-r)[:5])
        for rank, i in enumerate(np.argsort(-y)[:10], 1):
            tails.append({"subspace": name, "rank": rank, "occupation": titles[i],
                          "score (sd)": y[i]})
    sub = pd.DataFrame(rows)
    print("\n  most and least kurtotic direction in each subspace, against the same")
    print(f"  search on Gaussian data ({N_NULL} samples per dimension)")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(sub.round(3).to_string(index=False))
    tails = pd.DataFrame(tails)
    print("\n  heaviest-tailed direction: occupations per 0.5 sd from +1 sd, and the")
    print("  columns most correlated with it")
    for _, r in sub.iterrows():
        print(f"    {r['subspace']}: {r.iloc[-2]}\n      {r.iloc[-1]}")
    print("\n  occupations furthest along the heaviest-tailed direction:")
    for name, g in tails.groupby("subspace", sort=False):
        print(f"    {name}: " + "; ".join(f"{t[:34]} ({v:.1f})" for t, v
                                           in zip(g["occupation"], g["score (sd)"])))
    print("\n  Kurtosis well above the null that survives dropping the most extreme")
    print("  occupations means variance concentrated in a few occupations. Whether")
    print("  they are a separate group or the end of a long tail is in the counts:")
    print("  a group shows as a gap or a second hump, a tail as counts that fall")
    print("  steadily.")
    with pd.ExcelWriter(OUTDIR / "cluster_kurtosis.xlsx") as writer:
        comp.to_excel(writer, sheet_name="components", index=False)
        sub.to_excel(writer, sheet_name="subspaces", index=False)
        tails.to_excel(writer, sheet_name="tail_occupations", index=False)


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
    df = pd.read_csv(MASTER, index_col="onet_soc")
    # Part A asks about OCCUPATIONS, so it uses the same 216-column feature
    # matrix the axes are estimated from -- otherwise the silhouettes and the
    # axes would describe different spaces.
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    F = df[feats].values
    # Part B asks about SKILLS, where the object of comparison is the 161
    # element ratings; Work Context is a condition of the job, not a skill.
    skill = [c for c in df.columns if c.startswith(SKILL_PREFIXES)]
    M = df[skill].values
    print(f"{F.shape[0]} occupations x {len(feats)} features "
          f"(of which {len(skill)} skill ratings)")

    cont, curves = part_a_occupations(F)
    results = part_b_skills(M)
    part_c_local(cont, df["title"].to_numpy(), feats)
    figures(curves, results)

    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(cont)
    pd.DataFrame({"title": df["title"], "cluster_k2": km.labels_},
                 index=df.index).to_csv(OUTDIR / "cluster_assignments.csv")
    pd.DataFrame({"column": skill,
                  "cluster": results["correlation on published values"]["lab"]}
                 ).to_csv(OUTDIR / "cluster_skills.csv", index=False)
    print("\n[ok] wrote cluster_occupations.png, cluster_skill_similarity.png, "
          "cluster_assignments.csv, cluster_skills.csv, cluster_kurtosis.xlsx")


if __name__ == "__main__":
    main()