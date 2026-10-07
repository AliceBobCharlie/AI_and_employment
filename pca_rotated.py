#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_rotated.py -- the dimensions of occupational space and their rotated axes
(Sections 5.3 and 5.4).

Key design choices:
  - ONLY the O*NET feature blocks enter the decomposition: Abilities, Skills,
    Knowledge and Work Activities on the Importance scale, plus Work Context on
    the Context scale. 216 columns.
  - The labour-market variables (wages, employment, union coverage, self-
    employment, separation rates, prestige) and the education/training/
    experience distributions are HELD OUT. They are correlated against the
    finished axes as supplementary variables. This is what licenses the
    comparisons later: an axis cannot be said to predict wages if wages helped
    to build it. Wages appear in that comparison as a level (log median) and a
    dispersion (p90/p10) rather than as nine collinear percentiles.
  - everything z-scored.

Procedure:
  1. Eigenvalues, scree and Horn's parallel analysis.
  2. Component by component: bootstrap and split-half congruence of each
     unrotated component (Table 5.3). The components that pass one by one form
     tier 1.
  3. Nested subspaces: split-half principal angles between the leading-k
     subspaces, k = 1 to 14, with North's eigenvalue-gap ratio, a permutation
     null and the half-sample noise floor (Figure 5.1). A block of nearly equal
     eigenvalues fails component by component but holds as a subspace.
  4. Tier 2: the components after tier 1 are varimax-rotated among themselves,
     for every candidate end of the tier up to N_PC. Tier 2 ends at the largest
     candidate for which every rotated axis, of both tiers, reproduces on
     independent halves (split-half congruence p05 >= 0.90).
  5. Each tier is varimax-rotated on its own, so no variance moves between
     tiers, and each axis is signed by a marker variable. R1-R3 are therefore
     the same axes whatever tier 2 turns out to be.
  6. Loadings, block composition, supplementary variables against all axes,
     and a promax refit of tier 1.

A supplementary variable that correlates weakly with every axis is independent
of the content of the work; that is a statement about independence, not about
importance, and Section 5.6 of the paper makes the distinction with a ridge
regression instead.

Reads output/master_clean.csv. Writes output/rotated_axes.csv,
output/rotated_loadings.csv, output/oblique_pattern.csv,
output/pca_stability.xlsx, output/rotated_scree.png and the paper's
Figure 5.1, assets/subspace_stability.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MASTER = Path("output/master_clean.csv")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
FIGURE = Path("assets/subspace_stability.png")   # Figure 5.1; assets/ is published
SEED = 0
N_PC = 8          # components examined for stability
TOP = 12
B = 200
N_SPLIT = 50      # split-halves for the component-by-component check
N_SPLIT_SUB = 100 # split-halves for nested subspaces and rotated axes
N_NULL = 30       # permuted matrices for the subspace null
K_MAX = 14        # nested subspaces examined
STABLE_THRESH = 0.90
N_PERM = 50       # permutations for parallel analysis

# The feature matrix: Importance for four blocks, the Context scale for Work Context.
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")

# Sign anchors. Tier 1: '+' is the pole conjectured harder to substitute.
#   R1 + = physically intensive, R2 + = judgement, R3 + = person-facing.
TIER1_ANCHORS = ["abilities_im__Manual Dexterity",
                 "skills_im__Complex Problem Solving",
                 "workact_im__Assisting and Caring for Others"]
# Tier 2: each anchor names the axis it loads on most, and the axes are
# reported in this order. The signs carry no claim about substitutability.
#   R4 + = fixed site (clinical), R5 + = fixed correct standard,
#   R6 + = commercial.
TIER2_ANCHORS = ["knowledge_im__Medicine and Dentistry",
                 "workctx_cx__Importance of Being Exact or Accurate",
                 "knowledge_im__Sales and Marketing"]


# --------------------------------------------------------------------------- #
def load_and_prep():
    """The clean table is already imputed and carries the derived wage columns
    (see clean_master.py), so this only selects columns and standardizes.

    ONLY the O*NET descriptor blocks enter the decomposition. Education,
    training and experience and the economic/institutional columns are held out
    and correlated against the axes afterwards, so that nothing the axes are
    later compared with has helped to form them."""
    df = pd.read_csv(MASTER, index_col="onet_soc")
    title = df["title"]
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]

    # held out of the PCA, used only as supplementary variables below. Wages
    # appear as a level and a dispersion: the nine raw wage columns are
    # collinear and comparing all nine against every axis says nothing extra.
    ete = [c for c in df.columns if c.startswith("ete_")]
    econ = [c for c in ["ext_wage_level_log", "ext_wage_disp_p90p10",
                        "ext_prestige", "ext_sep_exit_rate",
                        "ext_sep_transfer_rate", "ext_union_cov_pct",
                        "ext_self_employed_pct", "ext_employment_log"]
            if c in df.columns]
    supp = df[econ + ete]

    Xz = StandardScaler().fit_transform(df[feats].values)
    print(f"feature matrix {Xz.shape[0]} x {len(feats)}")
    print(f"held out as supplementary: {len(econ)} labour-market columns, "
          f"{len(ete)} ETE columns")
    return Xz, feats, title, supp


def block_of(col):
    """'abilities_im__Oral Comprehension' -> 'abilities_im'."""
    return col.split("__", 1)[0]


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


# ------------------------- stability (unrotated) -------------------------- #
def congruence(a, b):
    den = np.sqrt((a @ a) * (b @ b))
    return np.abs(a @ b) / den if den > 0 else 0.0


def best_match(ref, cand):
    return np.array([max(congruence(ref[i], cand[j]) for j in range(cand.shape[0]))
                     for i in range(ref.shape[0])])


def pca_load(M, k):
    return PCA(n_components=k, random_state=SEED).fit(M).components_


def stability(X, ref, k):
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    boot = np.full((B, k), np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        try: boot[b] = best_match(ref, pca_load(X[idx], k))
        except Exception: pass
    sh = np.full((N_SPLIT, k), np.nan)
    for s in range(N_SPLIT):
        perm = rng.permutation(n); h1, h2 = perm[:n//2], perm[n//2:]
        try: sh[s] = best_match(pca_load(X[h1], k), pca_load(X[h2], k))
        except Exception: pass
    banner(f"STABILITY of UNROTATED components (boot B={B}, split x{N_SPLIT})")
    print(f"{'':4s} {'boot_p05':>9s} {'split_p05':>9s}  verdict")
    last = 0
    for j in range(k):
        bp = np.nanpercentile(boot[:, j], 5); sp = np.nanpercentile(sh[:, j], 5)
        ok = bp >= STABLE_THRESH and sp >= STABLE_THRESH
        if ok and last == j: last = j + 1
        print(f"PC{j+1:<2d} {bp:9.3f} {sp:9.3f}  {'STABLE' if ok else 'not stable on its own'}")
    print(f"\n-> the first {last} components reproduce one by one: tier 1")
    return last


def eigen_table(pca, p):
    ev, share = pca.explained_variance_, pca.explained_variance_ratio_
    cum = np.cumsum(share)
    banner("EIGENVALUES")
    print(f"{'':4s} {'eigenvalue':>10s} {'% var':>7s} {'cum %':>7s}")
    for j in range(10):
        print(f"PC{j+1:<2d} {ev[j]:10.2f} {share[j]*100:7.1f} {cum[j]*100:7.1f}")
    print(f"components to reach 90% of variance: {int(np.argmax(cum >= .9)) + 1}")


def parallel_analysis(Xz, ev):
    """Horn's parallel analysis: each column permuted independently, which keeps
    every column's distribution and destroys only the relations between them."""
    rng = np.random.default_rng(SEED)
    n, p = Xz.shape
    null = np.empty((N_PERM, 30))
    for b in range(N_PERM):
        Xp = rng.permuted(Xz, axis=0)
        Xp = Xp - Xp.mean(axis=0)
        null[b] = np.linalg.svd(Xp, compute_uv=False)[:30] ** 2 / (n - 1)
    q95 = np.percentile(null, 95, axis=0)
    above = ev[:30] > q95
    k = int(np.argmin(above)) if not above.all() else 30
    banner(f"PARALLEL ANALYSIS ({N_PERM} column-wise permutations, 95th percentile)")
    print(f"largest null eigenvalue {q95[0]:.2f}; component {k}: real {ev[k-1]:.2f} "
          f"vs null {q95[k-1]:.2f}; component {k+1}: real {ev[k]:.2f} vs null {q95[k]:.2f}")
    print(f"-> {k} components exceed the null, together {ev[:k].sum() / ev.sum():.1%} of variance")
    return k


# ------------------------- nested subspaces ------------------------------- #
def standardize(X):
    """z-score with the population sd, as StandardScaler does."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def eigen(X):
    """Eigenvalues (n-1 denominator) and eigenvectors (columns) of X z-scored."""
    Z = standardize(X)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    return s ** 2 / (len(Z) - 1), vt.T


def subspace_cosines(A, B):
    """Cosines of the principal angles between the column spaces of A and B
    (both with orthonormal columns; Björck and Golub 1973)."""
    return np.linalg.svd(A.T @ B, compute_uv=False)


def split_statistics(X, rng, n_splits):
    """Per split: the smallest principal-angle cosine and the mean squared
    cosine between the halves' leading-k subspaces for k = 1..K_MAX, the
    best-match congruence of each single component, and both halves'
    eigenvalues. Each half is standardised and decomposed on its own."""
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


def subspace_stability(X):
    """Split-half stability of the leading-k subspaces, against a null in
    which every column is permuted independently. North's ratio is the gap to
    the next eigenvalue over the eigenvalue's sampling error at half the
    sample size, lambda * sqrt(2 / (n/2)) (North et al. 1982)."""
    rng = np.random.default_rng(SEED)
    n = len(X)
    ev, _ = eigen(X)
    min_cos, msq_cos, single, half_ev = split_statistics(X, rng, N_SPLIT_SUB)
    null_runs = [split_statistics(np.column_stack([rng.permutation(c) for c in X.T]), rng, 1)
                 for _ in range(N_NULL)]
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
    table["min cos p05 >= 0.90"] = table["min cos p05"] >= STABLE_THRESH
    banner(f"NESTED SUBSPACES (split x{N_SPLIT_SUB}, null x{N_NULL})")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    return table


# ------------------------- rotation --------------------------------------- #
def promax(A, m=4):
    """Oblique rotation of an already varimax-rotated loading matrix A.
    Returns the pattern matrix and the factor correlation matrix.
    Follows the standard Hendrickson-White construction."""
    Q = A * np.abs(A) ** (m - 1)          # sharpened target
    U = np.linalg.lstsq(A, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    pattern = A @ U
    Uinv = np.linalg.inv(U)
    Phi = Uinv @ Uinv.T
    dg = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(dg, dg)          # to correlation form
    return pattern, Phi


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax rotation of a loading matrix Phi (p x k)."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vt
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


def rotate_tier(V, ev, idx):
    """Varimax within the components idx (eigenvectors as columns of V).
    Returns loadings and the rotation matrix, axes ordered by variance."""
    L, R = varimax(V[:, idx] * np.sqrt(ev[idx]))
    order = np.argsort(-(L ** 2).sum(axis=0))
    return L[:, order], R[:, order]


def rotated_stability(X, tiers):
    """Split-half congruence of the rotated axes. On each half the
    decomposition and every tier's rotation are redone; each full-sample axis
    is matched to the most congruent half-axis of the same tier, and the two
    halves' versions are compared. Returns p05 and median per axis."""
    rng = np.random.default_rng(SEED)
    ev, V = eigen(X)
    full = np.hstack([rotate_tier(V, ev, idx)[0] for idx in tiers])
    bounds = np.cumsum([0] + [len(idx) for idx in tiers])
    vals = np.empty((N_SPLIT_SUB, full.shape[1]))
    for s in range(N_SPLIT_SUB):
        perm = rng.permutation(len(X))
        halves = []
        for h in (perm[: len(X) // 2], perm[len(X) // 2:]):
            e, Vh = eigen(X[h])
            H = np.hstack([rotate_tier(Vh, e, idx)[0] for idx in tiers])
            matched = []
            for t in range(len(tiers)):
                cand = H[:, bounds[t]:bounds[t + 1]]
                for j in range(bounds[t], bounds[t + 1]):
                    c = [congruence(full[:, j], cand[:, i]) for i in range(cand.shape[1])]
                    matched.append(cand[:, int(np.argmax(c))])
            halves.append(np.column_stack(matched))
        vals[s] = [congruence(halves[0][:, j], halves[1][:, j]) for j in range(full.shape[1])]
    return np.percentile(vals, 5, axis=0), np.median(vals, axis=0)


def choose_tier2(X, k1):
    """Tier 2 is components k1+1..k2. Every candidate k2 from k1+2 to N_PC is
    tried (a tier of one component would be component-by-component again);
    tier 2 ends at the largest k2 whose rotated axes, and tier 1's, all
    reproduce at p05 >= STABLE_THRESH."""
    banner("TIER 2: rotated split-half congruence for each candidate end k2 (p05)")
    rows, k2 = [], k1
    for end in range(k1 + 2, N_PC + 1):
        tiers = [list(range(k1)), list(range(k1, end))]
        p05, med = rotated_stability(X, tiers)
        ok = bool((p05 >= STABLE_THRESH).all())
        if ok: k2 = end
        rows.append({"tier 2 = components": f"{k1 + 1}-{end}", "all axes pass": ok,
                     **{f"axis {j + 1} p05": v for j, v in enumerate(p05)},
                     **{f"axis {j + 1} median": v for j, v in enumerate(med)}})
        print(f"  components {k1 + 1}-{end}: " + " ".join(f"{v:.3f}" for v in p05)
              + f"  {'all pass' if ok else 'fails'}")
    print(f"\n-> tier 2 = components {k1 + 1}-{k2}" if k2 > k1 else "\n-> no tier 2")
    return k2, pd.DataFrame(rows)


def sign_and_order(L, R, feats, anchors, offset, verbose=True):
    """Sign each axis so its anchor loads positive. For tier 1 the anchors
    follow the variance order; for tier 2 each anchor picks the axis it loads
    on most and fixes the reporting order. If the anchors do not fit the tier,
    the axes keep their variance order and the larger pole is made positive."""
    fidx = {f: i for i, f in enumerate(feats)}
    k = L.shape[1]
    rows = [fidx.get(a) for a in anchors]
    if offset == 0:
        assign = list(range(k))
    else:
        assign = [int(np.argmax(np.abs(L[r]))) for r in rows] if None not in rows else []
    if len(anchors) != k or None in rows or len(set(assign)) != k:
        if verbose:
            print(f"[warn] anchors do not fit the tier starting at R{offset + 1}; "
                  "variance order, larger pole positive")
        sign = np.where(np.abs(L.min(axis=0)) > L.max(axis=0), -1.0, 1.0)
        return L * sign, R * sign
    L, R = L[:, assign].copy(), R[:, assign].copy()
    for j, (a, r) in enumerate(zip(anchors, rows)):
        v = L[r, j]
        if v < 0:
            L[:, j] *= -1
            R[:, j] *= -1
        if verbose:
            print(f"  axis R{offset + j + 1} anchored on '{a}' (loading {abs(v):.2f}, "
                  f"{'flipped' if v < 0 else 'kept'})")
    return L, R


# ------------------------- figure ----------------------------------------- #
def plot_subspaces(t, k1, k2):
    """Figure 5.1: for each k, the 5th percentile over split-halves of (a) the
    smallest principal-angle cosine between the two halves' leading-k
    subspaces and (b) the congruence of component k on its own. Tiers and the
    half-sample noise floor are marked."""
    k = t["k"].to_numpy()
    sub = t["min cos p05"].to_numpy()
    one = t["single component k: congruence p05"].to_numpy()
    ink, secondary, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(8.0, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=secondary, labelsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    below = k[~t["half-sample eigenvalue above noise floor"].to_numpy()]
    if len(below):
        ax.axvspan(below.min() - 0.5, K_MAX + 0.5, color=grid, alpha=0.5, linewidth=0)
        ax.text(below.min(), 0.04, "below\nnoise floor", color=secondary, fontsize=8, ha="center")
    edges = [0, k1] + ([k2] if k2 > k1 else [])
    for x in edges[1:]:
        ax.axvline(x + 0.5, color=muted, linewidth=0.8, linestyle=(0, (2, 3)))
    for t_no, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
        ax.text((a + b + 1) / 2, 1.035, f"tier {t_no}", color=secondary, fontsize=9, ha="center")
    ax.axhline(STABLE_THRESH, color=muted, linewidth=1, linestyle=":")
    ax.text(K_MAX + 0.4, STABLE_THRESH + 0.012, f"{STABLE_THRESH:.2f}", color=secondary,
            fontsize=8, ha="right", va="bottom")

    ax.plot(k, sub, color=blue, linewidth=2, marker="o", markersize=7,
            markeredgecolor=surface, markeredgewidth=1.5, label="leading-k subspace (smallest cosine)")
    ax.plot(k, one, color=orange, linewidth=2, marker="s", markersize=6.5,
            markeredgecolor=surface, markeredgewidth=1.5, label="component k on its own (congruence)")
    if k2 > k1:   # label the end of tier 2 and the point before it
        ax.annotate(f"{sub[k2 - 1]:.2f}", (k2, sub[k2 - 1]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=ink)
        ax.annotate(f"{sub[k2 - 2]:.2f}", (k2 - 1, sub[k2 - 2]), xytext=(9, -3),
                    textcoords="offset points", ha="left", va="center", fontsize=8, color=ink)

    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, K_MAX + 1))
    ax.set_xlabel("k", color=secondary, fontsize=9)
    ax.set_ylabel(f"5th percentile over {N_SPLIT_SUB} split-halves", color=secondary, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=200, facecolor=surface)
    plt.close(fig)


# ------------------------- reports ---------------------------------------- #
def print_loadings(Ld, var_share):
    banner("VARIMAX-ROTATED AXES (tiers rotated separately) -- loadings")
    for j, name in enumerate(Ld.columns):
        s = Ld[name]
        print(f"\nRotated axis {name}  (var share {var_share[j]:.1%})")
        print("  + end:")
        for n, v in s.sort_values(ascending=False).head(TOP).items():
            if v > 0.15: print(f"     {v:+.2f}  {n}")
        print("  - end:")
        for n, v in s.sort_values().head(TOP).items():
            if v < -0.15: print(f"     {v:+.2f}  {n}")


def print_block_composition(Ld, feats):
    banner("BLOCK COMPOSITION of rotated axes (% of axis SS by block)")
    tag = np.array([block_of(c) for c in feats])
    rows = []
    for name in Ld.columns:
        sq = Ld[name].to_numpy() ** 2
        rows.append({t: sq[tag == t].sum() / sq.sum() for t in sorted(set(tag))})
    rows.append({t: (tag == t).mean() for t in sorted(set(tag))})
    comp = pd.DataFrame(rows, index=list(Ld.columns) + ["share of columns"])
    print("\n" + (comp * 100).round(1).to_string())


def print_supplementary(Sdf, supp):
    banner("SUPPLEMENTARY VARIABLES against the rotated axes\n"
           "(none of these helped form the axes; low everywhere = independent\n"
           " of the content of the work, not necessarily unimportant)")
    corr = pd.DataFrame({n: supp.apply(lambda v: v.corr(Sdf[n])) for n in Sdf.columns})
    with pd.option_context("display.width", 200):
        print(corr.round(2).to_string())


def promax_check(Lr, feats, names):
    """Does the data prefer orthogonal axes, or does varimax impose them?
    Promax starts from the tier-1 solution and lets the axes tilt."""
    k = Lr.shape[1]
    Lp, Phi = promax(Lr)
    for j in range(k):
        if np.dot(Lr[:, j], Lp[:, j]) < 0:
            Lp[:, j] *= -1
            Phi[j, :] *= -1; Phi[:, j] *= -1
    np.fill_diagonal(Phi, 1.0)
    banner("OBLIQUE ROTATION CHECK (promax, tier 1): are the axes orthogonal by\n"
           "preference of the data, or only by constraint?")
    print("\nfactor correlations:")
    print(pd.DataFrame(Phi, index=names, columns=names).round(3).to_string())
    off = max(abs(Phi[i, j]) for i in range(k) for j in range(k) if i < j)
    print(f"\nlargest |correlation| between axes: {off:.3f}")
    print("congruence with the varimax solution:")
    for j in range(k):
        c = congruence(Lr[:, j], Lp[:, j])
        print(f"  {names[j]}: {c:.3f}  {'same axis' if c >= 0.95 else 'DIFFERS - inspect'}")
    if off < 0.20:
        print("\n-> the data places the axes close to orthogonal on its own.")
    elif off < 0.35:
        print("\n-> mild correlation; orthogonality is a simplification, not a")
        print("   distortion. Report these correlations.")
    else:
        print("\n-> substantial correlation; report the oblique solution instead.")
    pd.DataFrame(Lp, index=feats, columns=names).to_csv(OUTDIR / "oblique_pattern.csv")


# --------------------------------------------------------------------------- #
def main():
    Xz, feats, title, supp = load_and_prep()

    pca = PCA(random_state=SEED).fit(Xz)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), ev[:20], "o-"); a1.set_title("Scree")
    a2.plot(range(1, 21), cum[:20], "o-"); a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative"); fig.tight_layout()
    fig.savefig(OUTDIR / "rotated_scree.png", dpi=130); plt.close(fig)
    print(f"\nPC1={ev[0]:.1%}, PC1-4={cum[3]:.1%}, to90%={int(np.argmax(cum>=.9))+1} PCs")

    eigen_table(pca, Xz.shape[1])
    parallel_analysis(Xz, pca.explained_variance_)

    # tier 1: components that reproduce one by one
    k1 = max(stability(Xz, pca.components_[:N_PC], N_PC), 2)   # rotate at least 2
    # tier 2: the block after it, judged as a subspace and by its rotated axes
    sub = subspace_stability(Xz)
    k2, tier2_table = choose_tier2(Xz, k1)
    plot_subspaces(sub, k1, k2)
    tiers = [list(range(k1))] + ([list(range(k1, k2))] if k2 > k1 else [])

    # rotate each tier on its own, sign by anchors. Loadings carry a
    # sqrt(eigenvalue) scaling (p x k).
    V, lam = pca.components_.T, pca.explained_variance_
    Ls, Rs = [], []
    for t_no, idx in enumerate(tiers):
        L, R = rotate_tier(V, lam, idx)
        L, R = sign_and_order(L, R, feats, [TIER1_ANCHORS, TIER2_ANCHORS][t_no], idx[0])
        Ls.append(L); Rs.append(R)
    Lr = np.hstack(Ls)
    k = Lr.shape[1]
    names = [f"R{j+1}" for j in range(k)]
    var_share = (Lr ** 2).sum(axis=0) / Xz.shape[1]   # each variable standardized to var 1
    Ld = pd.DataFrame(Lr, index=feats, columns=names)
    print_loadings(Ld, var_share)
    print_block_composition(Ld, feats)

    # Rotated component scores. NOT Xz @ Lr: loadings carry a sqrt(eigenvalue)
    # scaling, and because the eigenvalues differ, using them as weights makes
    # the axes correlated (an orthogonal rotation should leave them
    # uncorrelated). Standardize the unrotated PC scores first, then apply each
    # tier's rotation matrix to its own components.
    T = pca.transform(Xz) / np.sqrt(lam)
    S = np.hstack([T[:, idx] @ R for idx, R in zip(tiers, Rs)])
    off = np.corrcoef(S.T) - np.eye(k)
    print(f"\n  max |corr| between rotated axes: {np.abs(off).max():.3f} (should be ~0)")
    Sdf = pd.DataFrame(S, columns=names, index=title.index)
    print_supplementary(Sdf, supp)

    promax_check(Ls[0], feats, names[:k1])

    p05, med = rotated_stability(Xz, tiers)
    rot = pd.DataFrame({"axis": names, "split-half congruence p05": p05, "median": med})
    banner("ROTATED AXES: split-half congruence")
    print(rot.round(3).to_string(index=False))

    out = Sdf.copy()
    out.insert(0, "title", title.values)
    out.to_csv(OUTDIR / "rotated_axes.csv")
    Ld.to_csv(OUTDIR / "rotated_loadings.csv")
    with pd.ExcelWriter(OUTDIR / "pca_stability.xlsx") as writer:
        sub.to_excel(writer, sheet_name="nested_subspaces", index=False)
        tier2_table.to_excel(writer, sheet_name="tier2_candidates", index=False)
        rot.to_excel(writer, sheet_name="rotated_axes", index=False)
    print(f"\n[ok] wrote rotated_axes.csv, rotated_loadings.csv, oblique_pattern.csv, "
          f"pca_stability.xlsx, rotated_scree.png and {FIGURE}")


if __name__ == "__main__":
    main()
