#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subspace_stability.py -- how stable is the subspace spanned by the first k
principal components, for k = 1 to 14, and where are the seams between tiers?

The paper's criterion (pca_rotated.py) matches components one at a time by
Tucker's congruence. That is sensitive to rotation inside a block of nearly
equal eigenvalues: two halves can find the same plane and still disagree on
which two lines to draw in it. Principal angles between subspaces are not
sensitive to that rotation, so a block of near-degenerate components shows up
as stable at the k that completes it and unstable at the k that cuts it.

Procedure:
  1. The full matrix is standardised and decomposed; eigenvalues are reported
     with North's ratio, gap to the next eigenvalue over the eigenvalue's
     sampling error at half the sample size, lambda * sqrt(2 / (n/2)).
  2. N_SPLIT times, the occupations are split at random into two halves. Each
     half is standardised on its own and decomposed on its own, so that
     nothing estimated on one half is used for the other.
  3. For each k, the cosines of the principal angles between the two halves'
     leading-k eigenvector subspaces are the singular values of V1[:, :k]' V2[:, :k].
     Two summaries are kept:
       min cosine           the worst-aligned direction (the strict measure;
                            it falls with k even without any seam, because
                            more angles means a more extreme worst one)
       mean squared cosine  average overlap, ||V1k' V2k||_F^2 / k
  4. Null reference: each column of the full matrix is permuted independently
     (destroying all correlation, keeping every column's distribution), and
     steps 2-3 are repeated N_NULL times. This gives what the two measures
     take, at each k, when there is no structure at all.
  5. Single components: each of the first 14 half-1 eigenvectors is matched to
     its best half-2 counterpart among the first 14 by Tucker's congruence,
     as in pca_rotated.py, to show which components are identified one by one.
  6. Noise floor at half size: the eigenvalues of each half, averaged over
     splits, against the 95th percentile of eigenvalues of column-permuted
     halves (parallel analysis at n/2). Where a half-sample eigenvalue sits
     below that floor, the half-sample eigenvector is noise, and a failure of
     the curve there means 'not resolvable at this sample size', not 'no seam'.

Input: the paper's matrix, the 910 occupations and 216 feature columns of
output/master_clean.csv (Importance for Abilities, Skills, Knowledge and Work
Activities, Context for Work Context).

Run from the project root:  python supplementary/subspace_stability.py
Writes output/subspace_stability.xlsx and output/subspace_stability.png, and
prints every sheet.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("output/master_clean.csv")
OUT_XLSX = Path("output/subspace_stability.xlsx")
OUT_PNG = Path("output/subspace_stability.png")

SEED = 0
N_SPLIT = 100
N_NULL = 30
K_MAX = 14
THRESHOLD = 0.90          # the paper's stability threshold, applied to p05

FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")


# --------------------------------------------------------------------------- #
def standardize(X):
    """z-score with the population sd, as StandardScaler does in pca_rotated.py."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def eigen(X):
    """Eigenvalues (n-1 denominator) and eigenvectors (columns) of standardized X."""
    Z = standardize(X)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    return s ** 2 / (len(Z) - 1), vt.T


def subspace_cosines(A, B):
    """Cosines of the principal angles between the column spaces of A and B
    (both with orthonormal columns)."""
    return np.linalg.svd(A.T @ B, compute_uv=False)


def split_statistics(X, rng, n_splits):
    """Per split: min cosine and mean squared cosine for k = 1..K_MAX, best-match
    congruence of each single component, and both halves' eigenvalues."""
    n = len(X)
    min_cos = np.empty((n_splits, K_MAX))
    msq_cos = np.empty((n_splits, K_MAX))
    single = np.empty((n_splits, K_MAX))
    half_ev = np.empty((2 * n_splits, K_MAX + 1))
    for s in range(n_splits):
        perm = rng.permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2:]
        e1, V1 = eigen(X[h1])
        e2, V2 = eigen(X[h2])
        half_ev[2 * s], half_ev[2 * s + 1] = e1[: K_MAX + 1], e2[: K_MAX + 1]
        for k in range(1, K_MAX + 1):
            c = subspace_cosines(V1[:, :k], V2[:, :k])
            min_cos[s, k - 1] = c.min()
            msq_cos[s, k - 1] = (c ** 2).mean()
        single[s] = np.abs(V1[:, :K_MAX].T @ V2[:, :K_MAX]).max(axis=1)
    return min_cos, msq_cos, single, half_ev


def permuted(X, rng):
    return np.column_stack([rng.permutation(col) for col in X.T])


def load_matrix():
    clean = pd.read_csv(CLEAN)
    cols = [c for c in clean.columns if c.startswith(FEATURE_PREFIXES)]
    X = clean[cols].to_numpy(float)
    if np.isnan(X).any():
        raise ValueError(f"{int(np.isnan(X).sum())} missing cells")
    return X


# --------------------------------------------------------------------------- #
def analyse(X, rng):
    n, p = X.shape
    ev, _ = eigen(X)
    print(f"[{CLEAN}] {n} occupations x {p} columns: {N_SPLIT} splits, {N_NULL} null runs")

    min_cos, msq_cos, single, half_ev = split_statistics(X, rng, N_SPLIT)
    # One fresh permutation per null run, each split once.
    null_runs = [split_statistics(permuted(X, rng), rng, 1) for _ in range(N_NULL)]
    null_min = np.vstack([r[0] for r in null_runs])
    null_msq = np.vstack([r[1] for r in null_runs])
    null_ev = np.vstack([r[3] for r in null_runs])

    half_error = np.sqrt(2 / (n / 2))
    rows = []
    for k in range(1, K_MAX + 1):
        j = k - 1
        rows.append({
            "k": k,
            "eigenvalue k (full sample)": ev[j],
            "gap k to k+1 / half-sample error": (ev[j] - ev[j + 1]) / (ev[j] * half_error),
            "min cos p05": np.percentile(min_cos[:, j], 5),
            "min cos median": np.median(min_cos[:, j]),
            "mean sq cos p05": np.percentile(msq_cos[:, j], 5),
            "mean sq cos median": np.median(msq_cos[:, j]),
            "null min cos median": np.median(null_min[:, j]),
            "null mean sq cos median": np.median(null_msq[:, j]),
            "null mean sq cos p95": np.percentile(null_msq[:, j], 95),
            "single component k: congruence p05": np.percentile(single[:, j], 5),
            "single component k: congruence median": np.median(single[:, j]),
            "half-sample eigenvalue k (mean)": half_ev[:, j].mean(),
            "half-sample null eigenvalue k (p95)": np.percentile(null_ev[:, j], 95),
        })
    table = pd.DataFrame(rows)
    table["half-sample eigenvalue above noise floor"] = (
        table["half-sample eigenvalue k (mean)"] > table["half-sample null eigenvalue k (p95)"])
    # A seam at k: the leading-k subspace is better aligned than both neighbours.
    m = table["mean sq cos median"].to_numpy()
    seam = [False] * K_MAX
    for j in range(K_MAX):
        left = m[j - 1] if j > 0 else -np.inf
        right = m[j + 1] if j < K_MAX - 1 else -np.inf
        seam[j] = m[j] > left and m[j] > right
    table["local peak (seam)"] = seam
    table["min cos p05 >= 0.90"] = table["min cos p05"] >= THRESHOLD
    return table


def plot(t):
    """Nested-subspace curves, with the null and the half-sample noise floor marked."""
    k = t["k"]
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_xticks(range(1, K_MAX + 1))
        ax.set_xlabel("k (leading components in the subspace)", color=muted, fontsize=9)
        below = t.loc[~t["half-sample eigenvalue above noise floor"], "k"]
        if len(below):
            ax.axvspan(below.min() - 0.5, K_MAX + 0.5, color=grid, alpha=0.6, linewidth=0)
    ax1.plot(k, t["mean sq cos median"], color=blue, lw=2, marker="o", ms=6, label="split-half median")
    ax1.fill_between(k, t["mean sq cos p05"], t["mean sq cos median"], color=blue, alpha=0.15,
                     linewidth=0, label="5th percentile to median")
    ax1.plot(k, t["null mean sq cos median"], color=muted, lw=1.5, ls="--", label="no structure (null)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Mean squared cosine, leading-k subspace", color=ink, fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.12))
    ax2.plot(k, t["min cos p05"], color=orange, lw=2, marker="o", ms=6, label="subspace, min cosine p05")
    ax2.plot(k, t["single component k: congruence p05"], color=aqua, lw=2, marker="s", ms=6,
             label="component k alone, congruence p05")
    ax2.axhline(THRESHOLD, color=muted, lw=1, ls=":")
    ax2.text(K_MAX - 0.6, THRESHOLD + 0.015, "paper threshold 0.90", color=muted, fontsize=8,
             va="bottom", ha="right")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Strict measures (5th percentile over splits)", color=ink, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.86))
    fig.suptitle("Split-half stability of nested subspaces, O*NET 31.0, 910 occupations x 216 (IM/CX)",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Grey band: k where the half-sample eigenvalue falls below the 95th percentile "
             "of column-permuted halves, so the half-sample eigenvector is noise.",
             color=muted, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_PNG, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    table = analyse(load_matrix(), rng)
    OUT_XLSX.parent.mkdir(exist_ok=True)
    table.to_excel(OUT_XLSX, index=False)
    plot(table)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print(f"\n[ok] wrote {OUT_XLSX} and {OUT_PNG}")


if __name__ == "__main__":
    main()