#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_rotated.py -- the dimensions of occupational space and their rotated axes,
in two tiers (Sections 5.3 and 5.5, Figure 5.1).

Key design choices:
  - ONLY the O*NET feature blocks enter the decomposition: Abilities, Skills,
    Knowledge and Work Activities on the Importance scale, plus Work Context on
    the Context scale. 216 columns, z-scored.
  - The labour-market variables and the education, training and experience
    distributions are HELD OUT and correlated against the finished axes. An
    axis cannot be said to predict wages if wages helped to build it. Wages
    appear as a level (log median) and a dispersion (p90/p10) rather than as
    nine collinear percentiles.

Tier 1 (Section 5.3):
  1. Eigenvalues, scree and Horn's parallel analysis (reported, not used to
     choose: the count depends on how many items measure the same thing).
  2. Component by component: bootstrap and split-half congruence of each
     unrotated component, PC1 to PC8. The leading components that pass one by
     one (p05 >= 0.90 on both) form tier 1.
  3. Tier 1 is varimax-rotated on its own and signed by marker variables
     (TIER1_ANCHORS). No later component enters this rotation, so R1-R3 do not
     depend on what tier 2 turns out to be. Loadings, block composition,
     supplementary variables, split-half congruence of the rotated axes and a
     promax refit (are the axes orthogonal by preference of the data?).

Tier 2 (Section 5.5):
  4. The eigenvalue gaps over the eigenvalue's sampling error at half the
     sample size, lambda * sqrt(2 / (n/2)) (North et al. 1982). A ratio near
     or below 2 means two components can swap or mix between samples.
  5. Nested subspaces: split-half principal angles between the leading-k
     subspaces, k = 1 to 14, with a permutation null and the half-sample noise
     floor (Figure 5.1).
  6. Candidate blocks after tier 1, components k1+1 to k2 for k2 = k1+2 ... 8.
     Each half is decomposed on its own and the principal angles between the
     two halves' block subspaces are computed. A block's agreement is the root
     mean square of the cosines (on the scale of a single congruence; for one
     component it is the congruence). Tier 2 ends at the largest k2 whose RMS
     cosine has p05 >= 0.90 over 100 split-halves. The rotated axes of each
     candidate are reported too; they are not used to choose.
  7. The same block measure for every contiguous block of two or more
     components after tier 2, up to component 14.
  8. Tier 2 is varimax-rotated on its own, signed and ordered by marker
     variables (TIER2_ANCHORS). Loadings, block composition, supplementary
     variables, split-half congruence of the rotated axes.

Tier-2 loadings read directly here still mix in the curvature of tier 1;
validate_tier2.py removes it before the axes are interpreted, and
tier2_residual_stability.py checks that what is left still reproduces.

The helpers (varimax, congruence, the tier rotation, the subspace measures,
the anchors) are imported by the other scripts from here.

Reads output/master_clean.csv. Writes output/rotated_axes.csv (scores R1-R6),
output/rotated_loadings.csv, output/oblique_pattern.csv,
output/pca_stability.xlsx, output/scree.png and the paper's Figure 5.1,
assets/subspace_stability.png, and prints every table.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MASTER = Path("output/master_clean.csv")
OUTDIR = Path("output")
AXES = OUTDIR / "rotated_axes.csv"              # scores R1-R6
LOADINGS = OUTDIR / "rotated_loadings.csv"
OUT = OUTDIR / "pca_stability.xlsx"
FIGURE = Path("assets/subspace_stability.png")   # Figure 5.1; assets/ is published

SEED = 0
STABLE_THRESH = 0.90
K_MAX = 14          # nested subspaces examined
N_SPLIT_SUB = 100   # split-halves for subspaces and rotated axes
TOP = 12            # loadings printed per pole
N_PC = 8            # components examined one by one; last end of a tier-2 candidate
B = 200             # bootstrap resamples
N_SPLIT = 50        # split-halves for the component-by-component check
N_NULL = 30         # permuted matrices for the nested-subspace null
N_PERM = 50         # permutations for parallel analysis

# The feature matrix: Importance for four blocks, the Context scale for Work Context.
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")
# Job-oriented blocks (what the work involves); the rest describe the worker.
JOB_PREFIXES = ("workact_im__", "workctx_cx__")

# Labour-market columns correlated against the axes. Wages appear as a level
# and a dispersion: the nine raw wage columns are collinear.
ECON = ["ext_wage_level_log", "ext_wage_disp_p90p10", "ext_prestige",
        "ext_sep_exit_rate", "ext_sep_transfer_rate", "ext_union_cov_pct",
        "ext_self_employed_pct", "ext_employment_log"]

# Sign anchors. Tier 1: '+' is the pole conjectured harder to substitute.
#   R1 + = physically intensive, R2 + = judgement, R3 + = person-facing.
TIER1_ANCHORS = ["abilities_im__Manual Dexterity",
                 "skills_im__Complex Problem Solving",
                 "workact_im__Assisting and Caring for Others"]
# Tier 2: each anchor names the axis it loads on most, and the axes are
# reported in this order. The signs carry no claim about substitutability.
#   R4 + = medicine and dentistry, R5 + = being exact or accurate,
#   R6 + = sales and marketing.
TIER2_ANCHORS = ["knowledge_im__Medicine and Dentistry",
                 "workctx_cx__Importance of Being Exact or Accurate",
                 "knowledge_im__Sales and Marketing"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_prep():
    """Select the feature columns and z-score them. The clean table is already
    imputed and carries the derived wage columns (see clean_master.py).
    Returns the z-scored matrix, the feature names, the titles and the held-out
    supplementary columns (labour market and ETE)."""
    df = pd.read_csv(MASTER, index_col="onet_soc")
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    ete = [c for c in df.columns if c.startswith("ete_")]
    econ = [c for c in ECON if c in df.columns]
    supp = df[econ + ete]
    Xz = StandardScaler().fit_transform(df[feats].values)
    print(f"feature matrix {Xz.shape[0]} occupations x {len(feats)} columns (from {MASTER})")
    print(f"held out as supplementary: {len(econ)} labour-market columns, "
          f"{len(ete)} ETE columns")
    return Xz, feats, df["title"], supp


def block_of(col):
    """'abilities_im__Oral Comprehension' -> 'abilities_im'."""
    return col.split("__", 1)[0]


def banner(text):
    print("\n" + "=" * 66)
    print(text)
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Decomposition and agreement measures
# --------------------------------------------------------------------------- #
def standardize(X):
    """z-score with the population sd, as StandardScaler does."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def eigen(X):
    """Eigenvalues (n-1 denominator) and eigenvectors (columns) of X z-scored."""
    Z = standardize(X)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    return s ** 2 / (len(Z) - 1), vt.T


def pca_load(M, k):
    return PCA(n_components=k, random_state=SEED).fit(M).components_


def congruence(a, b):
    """Tucker's congruence coefficient, unsigned."""
    den = np.sqrt((a @ a) * (b @ b))
    return np.abs(a @ b) / den if den > 0 else 0.0


def best_match(ref, cand):
    """For each row of ref, the congruence of the most congruent row of cand."""
    return np.array([max(congruence(ref[i], cand[j]) for j in range(cand.shape[0]))
                     for i in range(ref.shape[0])])


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


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
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


def rotate_tier(V, ev, idx):
    """Varimax within the components idx (eigenvectors as columns of V).
    Returns loadings and the rotation matrix, axes ordered by variance."""
    L, R = varimax(V[:, idx] * np.sqrt(ev[idx]))
    order = np.argsort(-(L ** 2).sum(axis=0))
    return L[:, order], R[:, order]


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


def tier_scores(pca, Xz, idx, R):
    """Rotated scores of one tier. NOT Xz @ L: loadings carry a
    sqrt(eigenvalue) scaling, and because the eigenvalues differ, using them as
    weights makes the axes correlated. The unrotated PC scores are
    standardised first, then the tier's rotation matrix is applied."""
    T = pca.transform(Xz) / np.sqrt(pca.explained_variance_)
    return T[:, idx] @ R


def match_axes(ref, cand):
    """Columns of cand paired one to one with the columns of ref, maximising
    the total congruence (Hungarian assignment). Pairing only: no axis is
    rotated or adjusted."""
    C = np.array([[congruence(ref[:, j], cand[:, i]) for i in range(cand.shape[1])]
                  for j in range(ref.shape[1])])
    _, cols = linear_sum_assignment(-C)
    return cand[:, cols]


def rotated_stability(X, tiers):
    """Split-half congruence of the rotated axes. On each half the
    decomposition and every tier's varimax rotation are redone from scratch,
    with nothing pulling the halves towards each other or towards the full
    sample. The full-sample axes are used only to label the halves' axes: each
    is paired one to one with a half-axis of the same tier, and then the two
    halves' versions of each axis are compared. Returns p05 and median per
    axis."""
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
                ref = full[:, bounds[t]:bounds[t + 1]]
                matched.append(match_axes(ref, H[:, bounds[t]:bounds[t + 1]]))
            halves.append(np.hstack(matched))
        vals[s] = [congruence(halves[0][:, j], halves[1][:, j]) for j in range(full.shape[1])]
    return np.percentile(vals, 5, axis=0), np.median(vals, axis=0)


def refit_loadings(X, cols, tiers):
    """The recipe of this script applied to another matrix:
    PCA, varimax within each tier, signed and ordered by the same anchors.
    tiers is a list of component index lists, tier 1 first if present.
    Columns are named by their position: a tier starting at component 4 gives
    R4, R5, ..."""
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED).fit(Xz)
    V, lam = pca.components_.T, pca.explained_variance_
    blocks, names = [], []
    for idx in tiers:
        anchors = TIER1_ANCHORS if idx[0] == 0 else TIER2_ANCHORS
        L, R = rotate_tier(V, lam, idx)
        L, _ = sign_and_order(L, R, cols, anchors, idx[0], verbose=False)
        blocks.append(L)
        names += [f"R{idx[0] + j + 1}" for j in range(L.shape[1])]
    return pd.DataFrame(np.hstack(blocks), index=cols, columns=names)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def print_loadings(Ld, var_share, title):
    banner(title)
    for j, name in enumerate(Ld.columns):
        s = Ld[name]
        print(f"\nRotated axis {name}  (var share {var_share[j]:.1%})")
        print("  + end:")
        for n, v in s.sort_values(ascending=False).head(TOP).items():
            if v > 0.15:
                print(f"     {v:+.2f}  {n}")
        print("  - end:")
        for n, v in s.sort_values().head(TOP).items():
            if v < -0.15:
                print(f"     {v:+.2f}  {n}")


def block_composition(Ld, feats):
    """Share of each axis's summed squared loadings falling in each block,
    with the blocks' shares of columns for comparison."""
    banner("BLOCK COMPOSITION of rotated axes (% of axis SS by block)")
    tag = np.array([block_of(c) for c in feats])
    rows = []
    for name in Ld.columns:
        sq = Ld[name].to_numpy() ** 2
        rows.append({t: sq[tag == t].sum() / sq.sum() for t in sorted(set(tag))})
    rows.append({t: (tag == t).mean() for t in sorted(set(tag))})
    comp = pd.DataFrame(rows, index=list(Ld.columns) + ["share of columns"])
    print("\n" + (comp * 100).round(1).to_string())
    return comp


def supplementary_correlations(Sdf, supp):
    banner("SUPPLEMENTARY VARIABLES against the rotated axes\n"
           "(none of these helped form the axes; low everywhere = independent\n"
           " of the content of the work, not necessarily unimportant)")
    corr = pd.DataFrame({n: supp.apply(lambda v: v.corr(Sdf[n])) for n in Sdf.columns})
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(corr.round(2).to_string())
    return corr


# --------------------------------------------------------------------------- #
# Tier 1
# --------------------------------------------------------------------------- #
def eigen_table(pca):
    ev, share = pca.explained_variance_, pca.explained_variance_ratio_
    t = pd.DataFrame({"component": [f"PC{j + 1}" for j in range(20)],
                      "eigenvalue": ev[:20], "% variance": 100 * share[:20],
                      "cumulative %": 100 * np.cumsum(share)[:20]})
    banner("EIGENVALUES")
    print(t.head(14).round(2).to_string(index=False))   # the 14 above the noise null
    print(f"components to reach 90% of variance: "
          f"{int(np.argmax(np.cumsum(share) >= .9)) + 1}")
    return t


def parallel_analysis(Xz, ev):
    """Horn's parallel analysis: each column permuted independently, which keeps
    every column's distribution and destroys only the relations between them."""
    rng = np.random.default_rng(SEED)
    n = len(Xz)
    null = np.empty((N_PERM, 30))
    for b in range(N_PERM):
        Xp = rng.permuted(Xz, axis=0)
        Xp = Xp - Xp.mean(axis=0)
        null[b] = np.linalg.svd(Xp, compute_uv=False)[:30] ** 2 / (n - 1)
    q95 = np.percentile(null, 95, axis=0)
    above = ev[:30] > q95
    k = int(np.argmin(above)) if not above.all() else 30
    banner(f"PARALLEL ANALYSIS ({N_PERM} column-wise permutations, 95th percentile)")
    print(f"-> {k} components exceed the null, together {ev[:k].sum() / ev.sum():.1%} of variance")
    print("   (the count depends on how many items measure the same thing, so it")
    print("   is reported, not used to choose the tiers)")
    return pd.DataFrame({"component": range(1, 31), "eigenvalue": ev[:30],
                         "null p95": q95, "above null": above})


def component_stability(X, ref):
    """Bootstrap and split-half best-match congruence of each unrotated
    component. Returns the table and k1, the number of leading components
    that pass one after another."""
    rng = np.random.default_rng(SEED)
    n, k = len(X), len(ref)
    boot = np.full((B, k), np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = best_match(ref, pca_load(X[idx], k))
    split = np.full((N_SPLIT, k), np.nan)
    for s in range(N_SPLIT):
        perm = rng.permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2:]
        split[s] = best_match(pca_load(X[h1], k), pca_load(X[h2], k))
    t = pd.DataFrame({"component": [f"PC{j + 1}" for j in range(k)],
                      "bootstrap p05": np.percentile(boot, 5, axis=0),
                      "split-half p05": np.percentile(split, 5, axis=0),
                      "split-half median": np.median(split, axis=0)})
    t["passes"] = (t["bootstrap p05"] >= STABLE_THRESH) & (t["split-half p05"] >= STABLE_THRESH)
    k1 = int(np.argmin(t["passes"])) if not t["passes"].all() else k
    banner(f"STABILITY of UNROTATED components (bootstrap B={B}, split-half x{N_SPLIT})")
    print(t.round(3).to_string(index=False))
    print(f"\n-> the first {k1} components reproduce one by one: tier 1")
    return t, k1


def promax_check(L, feats, names):
    """Does the data prefer orthogonal axes, or does varimax impose them?
    Promax starts from the varimax solution and lets the axes tilt."""
    k = L.shape[1]
    Lp, Phi = promax(L)
    for j in range(k):
        if np.dot(L[:, j], Lp[:, j]) < 0:
            Lp[:, j] *= -1
            Phi[j, :] *= -1
            Phi[:, j] *= -1
    np.fill_diagonal(Phi, 1.0)
    banner("OBLIQUE ROTATION CHECK (promax, m = 4): are the axes orthogonal by\n"
           "preference of the data, or only by constraint?")
    phi = pd.DataFrame(Phi, index=names, columns=names)
    print("\nfactor correlations:\n" + phi.round(3).to_string())
    off = max(abs(Phi[i, j]) for i in range(k) for j in range(k) if i < j)
    cong = [congruence(L[:, j], Lp[:, j]) for j in range(k)]
    print(f"\nlargest |correlation| between axes: {off:.3f}")
    print("congruence with the varimax solution:")
    for name, c in zip(names, cong):
        print(f"  {name}: {c:.3f}  {'same axis' if c >= 0.95 else 'DIFFERS - inspect'}")
    if off < 0.20:
        print("\n-> the data places the axes close to orthogonal on its own.")
    elif off < 0.35:
        print("\n-> mild correlation; orthogonality is a simplification, not a")
        print("   distortion. Report these correlations.")
    else:
        print("\n-> substantial correlation; report the oblique solution instead.")
    pd.DataFrame(Lp, index=feats, columns=names).to_csv(OUTDIR / "oblique_pattern.csv")
    phi["congruence with varimax"] = cong
    return phi



# --------------------------------------------------------------------------- #
# Tier 2
# --------------------------------------------------------------------------- #
def eigen_gaps(X):
    """Eigenvalue, share and North's gap ratio for k = 1 .. K_MAX."""
    n = len(X)
    ev, _ = eigen(X)
    half_error = np.sqrt(2 / (n / 2))
    k = np.arange(1, K_MAX + 1)
    t = pd.DataFrame({"k": k, "eigenvalue": ev[:K_MAX],
                      "% variance": 100 * ev[:K_MAX] / ev.sum(),
                      "gap k to k+1 / half-sample error":
                          (ev[:K_MAX] - ev[1:K_MAX + 1]) / (ev[:K_MAX] * half_error)})
    banner("EIGENVALUE GAPS (North's ratio at half the sample size)")
    print(t.round(2).to_string(index=False))
    print("  below about 2: the two components are not separated by the sample")
    return t


def subspace_stability(X):
    """Split-half stability of the leading-k subspaces, against a null in
    which every column is permuted independently."""
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
    banner(f"NESTED SUBSPACES (split x{N_SPLIT_SUB}, null x{N_NULL})")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    return table


def block_cosines(X, blocks):
    """Principal-angle cosines between the two halves' subspaces spanned by
    the components in each block (a, z), zero-based and z exclusive: block
    (3, 6) is components 4-6. Returns {block: array (splits, z - a)}."""
    rng = np.random.default_rng(SEED)
    n = len(X)
    out = {b: np.empty((N_SPLIT_SUB, b[1] - b[0])) for b in blocks}
    for s in range(N_SPLIT_SUB):
        perm = rng.permutation(n)
        _, V1 = eigen(X[perm[: n // 2]])
        _, V2 = eigen(X[perm[n // 2:]])
        for a, z in blocks:
            out[(a, z)][s] = subspace_cosines(V1[:, a:z], V2[:, a:z])
    return out


def block_row(block, cos):
    a, z = block
    rms = np.sqrt((cos ** 2).mean(axis=1))
    return {"components": f"{a + 1}-{z}", "size": z - a,
            "RMS cosine p05": np.percentile(rms, 5),
            "RMS cosine median": np.median(rms),
            "smallest cosine p05": np.percentile(cos.min(axis=1), 5),
            "RMS cosine p05 >= 0.90": np.percentile(rms, 5) >= STABLE_THRESH}


def choose_tier2(X, k1):
    """Tier 2 is components k1+1 .. k2, k2 the largest candidate whose block
    RMS cosine has p05 >= STABLE_THRESH. The rotated axes of each candidate
    are checked as well."""
    ends = range(k1 + 2, N_PC + 1)
    later = [(a, z) for a in range(k1, K_MAX) for z in range(a + 2, K_MAX + 1)]
    blocks = [(0, k1)] + [(k1, e) for e in ends] + later
    cos = block_cosines(X, list(dict.fromkeys(blocks)))

    banner(f"TIER 2 CANDIDATES: block split-half agreement (x{N_SPLIT_SUB})")
    ref = block_row((0, k1), cos[(0, k1)])
    print(f"  for reference, tier 1 as a block (components 1-{k1}): "
          f"RMS cosine p05 {ref['RMS cosine p05']:.3f}")
    rows, k2 = [], k1
    for e in ends:
        row = block_row((k1, e), cos[(k1, e)])
        p05, med = rotated_stability(X, [list(range(k1, e))])
        row["all rotated axes p05 >= 0.90"] = bool((p05 >= STABLE_THRESH).all())
        row.update({f"rotated axis {j + 1} p05": v for j, v in enumerate(p05)})
        row.update({f"rotated axis {j + 1} median": v for j, v in enumerate(med)})
        if row["RMS cosine p05 >= 0.90"]:
            k2 = e
        rows.append(row)
    cand = pd.DataFrame(rows)
    show = ["components", "RMS cosine p05", "RMS cosine median", "smallest cosine p05",
            "all rotated axes p05 >= 0.90"]
    print(cand[show].round(3).to_string(index=False))
    print("  rotated axes, split-half congruence p05:")
    for _, r in cand.iterrows():
        vals = [r[c] for c in cand.columns if c.endswith(" p05") and c.startswith("rotated")]
        print(f"    components {r['components']}: "
              + " ".join(f"{v:.3f}" for v in vals if pd.notna(v)))
    if k2 > k1:
        print(f"\n-> tier 2 = components {k1 + 1}-{k2}")
        chosen = cand.loc[cand["components"] == f"{k1 + 1}-{k2}"].iloc[0]
        if not chosen["all rotated axes p05 >= 0.90"]:
            print("[warn] the block holds but not all of its rotated axes do")
    else:
        print("\n-> no block after tier 1 holds: no tier 2")

    after = k2 if k2 > k1 else k1
    rest = pd.DataFrame([block_row(b, cos[b]) for b in later if b[0] >= after])
    banner(f"BLOCKS AFTER TIER 2: every contiguous block of 2+ components "
           f"from {after + 1} to {K_MAX}")
    if len(rest):
        best = rest.loc[rest["RMS cosine p05"].idxmax()]
        print(f"  {int(rest['RMS cosine p05 >= 0.90'].sum())} of {len(rest)} blocks pass; "
              f"highest RMS cosine p05 {best['RMS cosine p05']:.3f} "
              f"(components {best['components']})")
        print(rest.sort_values("RMS cosine p05", ascending=False).head(10)
              .round(3).to_string(index=False))
    return k2, pd.concat([pd.DataFrame([ref]), cand], ignore_index=True), rest


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


# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(exist_ok=True)
    Xz, feats, title, supp = load_and_prep()
    pca = PCA(random_state=SEED).fit(Xz)
    V, lam = pca.components_.T, pca.explained_variance_
    share = pca.explained_variance_ratio_
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), share[:20], "o-")
    a1.set_title("Scree")
    a2.plot(range(1, 21), np.cumsum(share)[:20], "o-")
    a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative")
    fig.tight_layout()
    fig.savefig(OUTDIR / "scree.png", dpi=130)
    plt.close(fig)

    # ---------------- tier 1 ----------------
    eig = eigen_table(pca)
    pa = parallel_analysis(Xz, lam)
    comp, k1 = component_stability(Xz, pca.components_[:N_PC])
    k1 = max(k1, 2)                       # rotate at least two components
    idx1 = list(range(k1))

    banner(f"TIER 1: varimax within components 1-{k1}")
    L1, R1 = rotate_tier(V, lam, idx1)
    L1, R1 = sign_and_order(L1, R1, feats, TIER1_ANCHORS, 0)
    names1 = [f"R{j + 1}" for j in range(k1)]
    var1 = (L1 ** 2).sum(axis=0) / Xz.shape[1]   # each column has variance 1
    print(f"  tier 1 holds {var1.sum():.1%} of the variance: "
          + ", ".join(f"{n} {v:.1%}" for n, v in zip(names1, var1)))
    print_loadings(pd.DataFrame(L1, index=feats, columns=names1), var1, "TIER 1 AXES -- loadings")
    phi = promax_check(L1, feats, names1)
    p05_1, med_1 = rotated_stability(Xz, [idx1])

    # ---------------- tier 2 ----------------
    gaps = eigen_gaps(Xz)
    sub = subspace_stability(Xz)
    k2, cand, rest = choose_tier2(Xz, k1)
    plot_subspaces(sub, k1, k2)

    Ls, Ss, names, var_share = [L1], [tier_scores(pca, Xz, idx1, R1)], list(names1), list(var1)
    p05, med = list(p05_1), list(med_1)
    if k2 > k1:
        idx2 = list(range(k1, k2))
        banner(f"TIER 2: varimax within components {k1 + 1}-{k2}")
        L_var, R2 = rotate_tier(V, lam, idx2)          # variance order
        L2, R2 = sign_and_order(L_var, R2, feats, TIER2_ANCHORS, k1)
        names2 = [f"R{k1 + j + 1}" for j in range(len(idx2))]
        var2 = (L2 ** 2).sum(axis=0) / Xz.shape[1]
        print(f"  tier 2 holds {var2.sum():.1%} of the variance: "
              + ", ".join(f"{n} {v:.1%}" for n, v in zip(names2, var2)))
        print_loadings(pd.DataFrame(L2, index=feats, columns=names2), var2,
                       "TIER 2 AXES -- loadings (curvature of tier 1 not yet removed)")
        # rotated_stability reports the axes in variance order; the anchors
        # may order them differently, so map each named axis back
        p05_2, med_2 = rotated_stability(Xz, [idx2])
        order = [int(np.argmax([congruence(L2[:, j], L_var[:, i]) for i in range(len(idx2))]))
                 for j in range(len(idx2))]
        Ls.append(L2)
        Ss.append(tier_scores(pca, Xz, idx2, R2))
        names += names2
        var_share += list(var2)
        p05 += list(p05_2[order])
        med += list(med_2[order])

    # ---------------- both tiers ----------------
    Ld = pd.DataFrame(np.hstack(Ls), index=feats, columns=names)
    comp_blocks = block_composition(Ld, feats)
    S = pd.DataFrame(np.hstack(Ss), columns=names, index=title.index)
    off = np.abs(np.corrcoef(S.to_numpy().T) - np.eye(len(names))).max()
    print(f"\n  max |corr| between rotated axes: {off:.3f} (should be ~0)")
    supp_corr = supplementary_correlations(S, supp)
    rot = pd.DataFrame({"axis": names, "tier": [1] * k1 + [2] * (len(names) - k1),
                        "variance share": var_share,
                        "split-half congruence p05": p05, "median": med})
    banner("ROTATED AXES: split-half congruence (each tier rotated on its own)")
    print(rot.round(3).to_string(index=False))

    out = S.copy()
    out.insert(0, "title", title.values)
    out.to_csv(AXES)
    Ld.to_csv(LOADINGS)
    with pd.ExcelWriter(OUT) as writer:
        eig.to_excel(writer, sheet_name="eigenvalues", index=False)
        pa.to_excel(writer, sheet_name="parallel_analysis", index=False)
        comp.to_excel(writer, sheet_name="component_stability", index=False)
        gaps.to_excel(writer, sheet_name="eigen_gaps", index=False)
        sub.to_excel(writer, sheet_name="nested_subspaces", index=False)
        cand.to_excel(writer, sheet_name="tier2_candidates", index=False)
        rest.to_excel(writer, sheet_name="blocks_after_tier2", index=False)
        rot.to_excel(writer, sheet_name="rotated_axes", index=False)
        Ld.to_excel(writer, sheet_name="loadings")
        comp_blocks.to_excel(writer, sheet_name="block_composition")
        supp_corr.to_excel(writer, sheet_name="supplementary")
        phi.to_excel(writer, sheet_name="promax_tier1")
    print(f"\n[ok] wrote {AXES}, {LOADINGS}, oblique_pattern.csv, scree.png, {OUT} and {FIGURE}")


if __name__ == "__main__":
    main()
