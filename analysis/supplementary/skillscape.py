#!/usr/bin/env python3
"""
skillscape.py -- rebuild the skill-skill similarity of Alabdulkareem et al.
(Sci. Adv. 2018) and a correlation-based similarity from the same ratings, and
compare how bimodal each is and how well each splits into two skill clusters.

Input:  data/prepared/master_clean.csv
Output: output/skillscape/similarity.csv    one row per similarity
        output/skillscape/skill_clusters.csv
        output/skillscape/edge_weights.png

The ratings are the SKILL_PREFIXES columns of master_clean.csv (Importance for
Skills, Abilities, Knowledge and Work Activities).

1. rca_binarise() sets each occupation-skill cell to 1 where the revealed
   comparative advantage exceeds 1. theta_min_conditional() turns the binary
   matrix into a skill-skill similarity: for each pair, the smaller of the two
   conditional probabilities of co-occurrence.
2. The second similarity is the correlation between skills across
   occupations, on the z-scored ratings.
3. For each similarity, bimodality() gives Sarle's bimodality coefficient of
   the off-diagonal values, and average-linkage clustering on 1 - similarity
   splits the skills into two clusters, scored by the silhouette.
4. The share of skill pairs with theta above THETA_CUT is recorded.
5. similarity.csv holds one row per similarity; skill_clusters.csv the two
   cluster labels of every skill; edge_weights.png the histograms of the
   off-diagonal values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import MASTER, OUTDIR   # noqa: E402

CLEAN = MASTER
OUT = OUTDIR / "skillscape"

SKILL_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")
THETA_CUT = 0.6          # threshold on theta


def rca_binarise(M):
    """1 where (M_ij / row sum) / (column sum / total) > 1, else 0."""
    M = M.astype(float)
    rca = (M / M.sum(1, keepdims=True)) / (M.sum(0, keepdims=True) / M.sum())
    return (rca > 1).astype(float)


def theta_min_conditional(B):
    """For each pair of columns of the binary matrix B, the smaller of the two
    conditional probabilities of co-occurrence; 1 on the diagonal."""
    co = B.T @ B
    use = B.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = co / use[None, :]
    cond = np.nan_to_num(cond)
    theta = np.minimum(cond, cond.T)
    np.fill_diagonal(theta, 1.0)
    return theta


def bimodality(v):
    """Sarle's bimodality coefficient of v."""
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def offdiag(S):
    return S[np.triu_indices_from(S, k=1)]


def two_clusters(S):
    """Average-linkage labels on 1 - S and their silhouette."""
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0.0)
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(D)
    return lab, silhouette_score(D, lab, metric="precomputed")


def plot(weights):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (label, w) in zip(axes, weights.items()):
        ax.hist(w, bins=60, color="#4C72B0")
        ax.set_title(f"{label}\nbimodality {bimodality(w):.3f}", fontsize=10)
        ax.set_xlabel("pairwise skill similarity")
        ax.set_ylabel("pairs")
    fig.tight_layout()
    fig.savefig(OUT / "edge_weights.png", dpi=130)
    plt.close(fig)


def main():
    try:
        df = pd.read_csv(CLEAN, index_col="onet_soc")
    except OSError as err:
        sys.exit(f"[fail] read {CLEAN}: {err}")
    skills = [c for c in df.columns if c.startswith(SKILL_PREFIXES)]
    M = df[skills].to_numpy(float)
    print(f"[ok] read {CLEAN}: {M.shape[0]} occupations x {M.shape[1]} skill columns")

    B = rca_binarise(M)
    theta = theta_min_conditional(B)
    similarities = {"RCA + min conditional prob.": theta,
                    "correlation on published values": np.corrcoef(StandardScaler().fit_transform(M).T)}
    rows, labels, weights = [], {}, {}
    for label, S in similarities.items():
        w = offdiag(S)
        lab, sil = two_clusters(S)
        sizes = np.bincount(lab)
        rows.append({"similarity": label, "min": w.min(), "median": np.median(w), "max": w.max(),
                     "bimodality coefficient": bimodality(w),
                     "cluster sizes": f"{sizes[0]} {sizes[1]}", "silhouette": sil})
        labels[label] = lab
        weights[label] = w
    keep = theta > THETA_CUT
    np.fill_diagonal(keep, False)
    table = pd.DataFrame(rows)
    table["share of binarised cells = 1"] = [B.mean(), np.nan]
    table[f"share of pairs with theta > {THETA_CUT}"] = [keep.sum() / (keep.size - len(keep)), np.nan]

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "similarity.csv", index=False)
    pd.DataFrame({"column": skills,
                  "cluster, RCA + min conditional prob.": labels["RCA + min conditional prob."],
                  "cluster, correlation": labels["correlation on published values"]}
                 ).to_csv(OUT / "skill_clusters.csv", index=False)
    plot(weights)
    print(f"[ok] wrote similarity.csv, skill_clusters.csv and edge_weights.png to {OUT}/")


if __name__ == "__main__":
    main()