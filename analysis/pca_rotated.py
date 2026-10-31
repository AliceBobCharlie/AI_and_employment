#!/usr/bin/env python3
"""
pca_rotated.py -- the six rotated axes R1-R6.

ROTATION_GROUPS (PC1-3 and PC4-6) is fixed here, copied from the results of
resampling_stability.py. check_groups() compares it with that script's
dimensions.csv and warns if they differ or the file is missing.

load_and_prep() reads output/master_clean.csv and z-scores the FEATURE_PREFIXES
columns. sklearn's PCA is fitted once. For each group, rotate_group() applies
varimax() to the loadings (eigenvectors x sqrt(eigenvalues)) and sorts the
axes by variance; sign_by_anchors() flips each axis so its anchor in ANCHORS
loads positive. group_scores() multiplies the standardised PC scores by the
group's rotation matrix.

Writes output/axis_loadings.csv (feature x axis) and
output/occupation_coordinates.csv (occupation x axis, with titles).
"""

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
DIMENSIONS = OUTDIR / "resampling_stability" / "dimensions.csv"
CSV_FLOAT = "%.10g"     # 10 significant digits, so run-to-run noise does not change tracked files
SEED = 0

# ============================================================================ #
# FIXED BY HAND from resampling_stability.py (output/resampling_stability/
# dimensions.csv). Rerun that script when master_clean.csv changes, and update
# these if it finds other components. Zero-based component indices.
STABLE_COMPONENTS = [0, 1, 2]     # PC1-3 -> R1-R3
STABLE_SUBSPACE = [3, 4, 5]       # PC4-6 -> R4-R6
# ============================================================================ #
ROTATION_GROUPS = [STABLE_COMPONENTS, STABLE_SUBSPACE]

FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__", "workctx_ctp__")

# One anchor per axis, in variance order within its group; the axis is
# signed so its anchor loads positive.
COMPONENT_ANCHORS = ["abilities_im__Manual Dexterity",                     # R1
                     "skills_im__Complex Problem Solving",                 # R2
                     "workact_im__Assisting and Caring for Others"]        # R3
SUBSPACE_ANCHORS = ["knowledge_im__Medicine and Dentistry",                # R4
                    "workctx_cx__Importance of Being Exact or Accurate",   # R5
                    "knowledge_im__Sales and Marketing"]                   # R6
ANCHORS = [COMPONENT_ANCHORS, SUBSPACE_ANCHORS]


def load_and_prep():
    """The z-scored FEATURE_PREFIXES columns, their names and the titles."""
    df = pd.read_csv(MASTER, index_col="onet_soc")
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    Xz = StandardScaler().fit_transform(df[feats].values)
    print(f"[ok] read {MASTER}: {Xz.shape[0]} occupations x {len(feats)} features")
    return Xz, feats, df["title"]


def group_label(idx):
    return f"PC{idx[0] + 1}-{idx[-1] + 1}" if idx else "none"


def check_groups():
    """Warn if ROTATION_GROUPS differs from what resampling_stability.py found."""
    if not DIMENSIONS.exists():
        print(f"[warn] {DIMENSIONS} not found: ROTATION_GROUPS not checked; "
              "run resampling_stability.py")
        return
    found = pd.read_csv(DIMENSIONS)["found"].tolist()
    fixed = [group_label(idx) for idx in ROTATION_GROUPS]
    if found != fixed:
        print(f"[warn] ROTATION_GROUPS is {', '.join(fixed)} but resampling_stability.py "
              f"found {', '.join(found)}: update STABLE_COMPONENTS and STABLE_SUBSPACE")


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax rotation of a loading matrix Phi (p x k). Returns the
    rotated loadings and the rotation matrix."""
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


def rotate_group(V, ev, idx):
    """Varimax on the loadings of components idx; axes sorted by variance.
    Returns loadings and rotation matrix."""
    L, R = varimax(V[:, idx] * np.sqrt(ev[idx]))
    order = np.argsort(-(L ** 2).sum(axis=0))
    return L[:, order], R[:, order]


def sign_by_anchors(L, R, feats, anchors, label="", verbose=True):
    """Flip each axis j so anchors[j] loads positive. Warns if an anchor
    loads most on another axis of the group."""
    rows = [feats.index(a) for a in anchors]
    if verbose and [int(np.argmax(np.abs(L[r]))) for r in rows] != list(range(len(rows))):
        print(f"[warn] {label}: an anchor loads most on another axis; check the axis names")
    sign = np.where(L[rows, range(len(rows))] < 0, -1.0, 1.0)
    return L * sign, R * sign


def axis_names(idx):
    return [f"R{i + 1}" for i in idx]


def rotate_groups(V, ev, feats, groups, verbose=True):
    """rotate_group() then sign_by_anchors() for each group; a list of (L, R)."""
    out = []
    for idx, anchors in zip(groups, ANCHORS):
        L, R = rotate_group(V, ev, idx)
        out.append(sign_by_anchors(L, R, feats, anchors, group_label(idx), verbose))
    return out


def group_scores(pca, Xz, idx, R):
    """Standardised PC scores of components idx times the rotation matrix R."""
    T = pca.transform(Xz) / np.sqrt(pca.explained_variance_)
    return T[:, idx] @ R


def main():
    check_groups()
    OUTDIR.mkdir(exist_ok=True)
    Xz, feats, title = load_and_prep()
    pca = PCA(random_state=SEED).fit(Xz)
    V, ev = pca.components_.T, pca.explained_variance_

    groups = rotate_groups(V, ev, feats, ROTATION_GROUPS)
    names = [n for idx in ROTATION_GROUPS for n in axis_names(idx)]
    loadings = pd.DataFrame(np.hstack([L for L, _ in groups]), index=feats, columns=names)
    scores = pd.DataFrame(np.hstack([group_scores(pca, Xz, idx, R)
                                     for idx, (_, R) in zip(ROTATION_GROUPS, groups)]),
                          columns=names, index=title.index)
    print("[ok] rotated " + " and ".join(
        f"{axis_names(idx)[0]}-{axis_names(idx)[-1]} within {group_label(idx)}"
        for idx in ROTATION_GROUPS))

    scores.insert(0, "title", title.values)
    scores.to_csv(AXES, float_format=CSV_FLOAT)
    loadings.to_csv(LOADINGS, float_format=CSV_FLOAT)
    print(f"[ok] wrote {AXES}, {LOADINGS}")


# ============================================================================ #
# Not used above; imported by other scripts. Remove each when its users no
# longer import it.
# ============================================================================ #
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


def match_axes(ref, cand):
    """Columns of cand reordered to pair one to one with the columns of ref,
    maximising the total congruence (Hungarian assignment)."""
    C = np.array([[congruence(ref[:, j], cand[:, i]) for i in range(cand.shape[1])]
                  for j in range(ref.shape[1])])
    _, cols = linear_sum_assignment(-C)
    return cand[:, cols]


def block_of(col):
    """'abilities_im__Oral Comprehension' -> 'abilities_im'; CTP columns -> 'workctx_cx'."""
    block = col.split("__", 1)[0]
    return "workctx_cx" if block == "workctx_ctp" else block


if __name__ == "__main__":
    main()