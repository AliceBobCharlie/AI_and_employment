#!/usr/bin/env python3
"""
pca_rotated.py -- the six rotated axes.

Reads output/master_clean.csv and z-scores its 220 feature columns. Runs one
PCA on all occupations, then rotates two groups of components by varimax, each
on its own: STABLE_COMPONENTS (PC1-3) into R1-R3 and STABLE_SUBSPACE (PC4-6)
into R4-R6. R1-R3 are ordered by variance; R4-R6 are ordered by their anchor
features. Every axis is signed so its anchor loads positive. Each occupation's
scores are its standardised PC scores times the group's rotation matrix; the
script stops if any two axes' scores correlate.

Writes to output/:
  occupation_coordinates.csv   each occupation's title and scores on R1-R6
  axis_loadings.csv            each feature's loadings on R1-R6
  axis_block_composition.csv   each axis's squared loadings, as shares by
                               feature block, and each block's share of columns
  axis_vs_components.csv       the congruence of R4-R6 with PC4-6

STABLE_COMPONENTS and STABLE_SUBSPACE are set from the results of
resampling_stability.py, which warns if what it finds differs from them.
Other scripts import the helpers and constants below from this module.
"""

import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MASTER = Path("output/master_clean.csv")
OUTDIR = Path("output")
AXES = OUTDIR / "occupation_coordinates.csv"
LOADINGS = OUTDIR / "axis_loadings.csv"
BLOCK_COMPOSITION = OUTDIR / "axis_block_composition.csv"
VS_COMPONENTS = OUTDIR / "axis_vs_components.csv"
CSV_FLOAT = "%.10g"     # 10 significant digits, so run-to-run noise does not change tracked files

SEED = 0
STABLE_THRESH = 0.90    # p05 of congruence needed to call a component, block or axis stable
K_MAX = 15              # components examined by the stability checks
N_SPLIT_SUB = 200       # resamples per stability check

# Zero-based component indices, each group rotated on its own:
# R1-R3 within PC1-3, R4-R6 within PC4-6.
STABLE_COMPONENTS = [0, 1, 2]
STABLE_SUBSPACE = [3, 4, 5]
ROTATION_GROUPS = [STABLE_COMPONENTS, STABLE_SUBSPACE]

FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__", "workctx_ctp__")

# Each group's axes are signed so their anchor loads positive.
# The stable components' anchors follow the variance order:
# R1 + = physically intensive, R2 + = judgement, R3 + = person-facing.
COMPONENT_ANCHORS = ["abilities_im__Manual Dexterity",
                 "skills_im__Complex Problem Solving",
                 "workact_im__Assisting and Caring for Others"]
# The stable subspace's anchors pick the axis they load on most and set the
# order: R4 + = medicine and dentistry, R5 + = being exact or accurate,
# R6 + = sales and marketing.
SUBSPACE_ANCHORS = ["knowledge_im__Medicine and Dentistry",
                 "workctx_cx__Importance of Being Exact or Accurate",
                 "knowledge_im__Sales and Marketing"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_prep():
    """Select the feature columns and z-score them. Returns the z-scored
    matrix, the feature names and the titles."""
    df = pd.read_csv(MASTER, index_col="onet_soc")
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    Xz = StandardScaler().fit_transform(df[feats].values)
    print(f"[ok] read {MASTER}: {Xz.shape[0]} occupations x {len(feats)} features")
    return Xz, feats, df["title"]


def block_of(col):
    """'abilities_im__Oral Comprehension' -> 'abilities_im'. The working-time
    columns (CTP) count with the rest of Work Context, as 'workctx_cx'."""
    block = col.split("__", 1)[0]
    return "workctx_cx" if block == "workctx_ctp" else block


def banner(text):
    """Section heading for console output; kept for the scripts that import it."""
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


def congruence(a, b):
    """Tucker's congruence coefficient, unsigned."""
    den = np.sqrt((a @ a) * (b @ b))
    return np.abs(a @ b) / den if den > 0 else 0.0


def subspace_cosines(A, B):
    """Cosines of the principal angles between the column spaces of A and B
    (both with orthonormal columns)."""
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
    Returns the pattern matrix, the factor correlation matrix and the
    transformation U (pattern = A U; oblique scores = varimax scores U^-T).
    Used by check_curvature.py."""
    Q = A * np.abs(A) ** (m - 1)
    U = np.linalg.lstsq(A, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    pattern = A @ U
    Uinv = np.linalg.inv(U)
    Phi = Uinv @ Uinv.T
    dg = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(dg, dg)
    return pattern, Phi, U


def rotate_group(V, ev, idx):
    """Varimax within the components idx (eigenvectors as columns of V).
    Returns loadings and the rotation matrix, axes ordered by variance."""
    L, R = varimax(V[:, idx] * np.sqrt(ev[idx]))
    order = np.argsort(-(L ** 2).sum(axis=0))
    return L[:, order], R[:, order]


def sign_and_order(L, R, feats, anchors, offset, verbose=True):
    """Sign each axis so its anchor loads positive. For the group starting at
    the first component (offset 0) the anchors follow the variance order;
    otherwise each anchor picks the axis it loads on most and fixes the order.
    If the anchors do not fit the group, the axes
    keep their variance order and the larger pole is made positive, with a
    warning when verbose."""
    fidx = {f: i for i, f in enumerate(feats)}
    k = L.shape[1]
    rows = [fidx.get(a) for a in anchors]
    if offset == 0:
        assign = list(range(k))
    else:
        assign = [int(np.argmax(np.abs(L[r]))) for r in rows] if None not in rows else []
    if len(anchors) != k or None in rows or len(set(assign)) != k:
        if verbose:
            print(f"[warn] anchors do not fit the group starting at R{offset + 1}; "
                  "variance order, larger pole positive")
        sign = np.where(np.abs(L.min(axis=0)) > L.max(axis=0), -1.0, 1.0)
        return L * sign, R * sign
    L, R = L[:, assign].copy(), R[:, assign].copy()
    for j, r in enumerate(rows):
        if L[r, j] < 0:
            L[:, j] *= -1
            R[:, j] *= -1
    return L, R


def group_scores(pca, Xz, idx, R):
    """Rotated scores of one group: the unrotated PC scores are standardised
    first, then the group's rotation matrix is applied (not Xz @ L)."""
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


def group_anchors(idx):
    return COMPONENT_ANCHORS if idx[0] == 0 else SUBSPACE_ANCHORS


def axis_names(idx):
    return [f"R{i + 1}" for i in idx]


def rotate_groups(V, ev, feats, groups, verbose=True):
    """Loadings and rotation matrices of every group, rotated, signed and
    ordered by the anchors. Returns a list of (L, R) per group."""
    out = []
    for idx in groups:
        L, R = rotate_group(V, ev, idx)
        out.append(sign_and_order(L, R, feats, group_anchors(idx), idx[0], verbose))
    return out


# --------------------------------------------------------------------------- #
# Descriptions of the axes
# --------------------------------------------------------------------------- #
def block_composition(Ld, feats):
    """Share of each axis's summed squared loadings falling in each block,
    with the blocks' shares of columns for comparison."""
    tag = np.array([block_of(c) for c in feats])
    rows = []
    for name in Ld.columns:
        sq = Ld[name].to_numpy() ** 2
        rows.append({t: sq[tag == t].sum() / sq.sum() for t in sorted(set(tag))})
    rows.append({t: (tag == t).mean() for t in sorted(set(tag))})
    return pd.DataFrame(rows, index=list(Ld.columns) + ["share of columns"])


def vs_components(L, V, idx):
    """Congruence of each rotated axis of a group with the group's unrotated components."""
    return pd.DataFrame([[congruence(L[:, j], V[:, i]) for i in idx] for j in range(len(idx))],
                        index=axis_names(idx), columns=[f"PC{i + 1}" for i in idx])


# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(exist_ok=True)
    Xz, feats, title = load_and_prep()
    pca = PCA(random_state=SEED).fit(Xz)
    V, lam = pca.components_.T, pca.explained_variance_

    groups = rotate_groups(V, lam, feats, ROTATION_GROUPS)
    names = [n for idx in ROTATION_GROUPS for n in axis_names(idx)]
    Ld = pd.DataFrame(np.hstack([L for L, _ in groups]), index=feats, columns=names)
    S = pd.DataFrame(np.hstack([group_scores(pca, Xz, idx, R)
                                for idx, (_, R) in zip(ROTATION_GROUPS, groups)]),
                     columns=names, index=title.index)
    print("[ok] rotated " + " and ".join(
        f"{axis_names(idx)[0]}-{axis_names(idx)[-1]} within PC{idx[0] + 1}-{idx[-1] + 1}"
        for idx in ROTATION_GROUPS))

    off = np.abs(np.corrcoef(S.to_numpy().T) - np.eye(len(names))).max()
    if off > 1e-6:
        sys.exit(f"[fail] rotated scores correlate (max |r| = {off:.2e}); check the rotation")

    out = S.copy()
    out.insert(0, "title", title.values)
    out.to_csv(AXES, float_format=CSV_FLOAT)
    Ld.to_csv(LOADINGS, float_format=CSV_FLOAT)
    block_composition(Ld, feats).to_csv(BLOCK_COMPOSITION, float_format=CSV_FLOAT)
    vs_components(groups[1][0], V, STABLE_SUBSPACE).to_csv(VS_COMPONENTS, float_format=CSV_FLOAT)
    print(f"[ok] wrote {AXES}, {LOADINGS}, {BLOCK_COMPOSITION}, {VS_COMPONENTS}")


# Old names, still imported by other scripts.
TIER1_ANCHORS = COMPONENT_ANCHORS
TIER2_ANCHORS = SUBSPACE_ANCHORS
rotate_tier = rotate_group
tier_scores = group_scores


if __name__ == "__main__":
    main()