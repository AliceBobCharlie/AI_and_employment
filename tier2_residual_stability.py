#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tier2_residual_stability.py -- does the second tier still reproduce once the
curvature of the first tier is taken out of it? (Section 5.5)

Part of components 4-6 is a non-linear function of components 1-3 (squares,
products, bends; see check_curvature.py section B). validate_tier2.py removes
that part before reading R4-R6. This script asks whether what is left is still
a stable three-dimensional block, or whether the block was stable only because
of the curvature it carried.

Everything is done separately in each half of a split, so the two halves share
nothing: each half is z-scored and decomposed on its own, its own PC1-3 define
its own curvature, and the curvature is removed with its own fit. The halves'
results are then compared with principal angles, and agreement is the root
mean square (RMS) of the cosines, p05 over 100 split-halves, judged against
0.90 as in pca_rotated.py. The original block 4-6 is scored on the same splits
as the paired baseline.

Curvature is removed in two ways:
  cubic   least squares on an intercept, PC1-3 and all their squares,
          products and cubes (19 regressors besides the intercept), in sample
  kNN     the linear part of PC1-3 removed exactly, then the out-of-fold
          prediction (5 folds) of 20-nearest-neighbour regression in the space
          of the raw PC1-3 scores
Both are linear in the target given the predictors, so removing curvature
from rotated axes is the same as rotating after removing it: neither the
rotation within tier 1 nor that within tier 2 affects the subspace results.

  A  THE RESIDUAL OF THE BLOCK. The PC4-6 scores (unit variance) have the
     curvature removed. The covariances of the residual scores with the 216
     columns give the residual's directions in column space -- what
     validate_tier2.py interprets. The two halves' three directions are
     compared as a subspace (A) and, after varimax within 4-6 on each half,
     axis by axis (A-axes): each full-sample axis R4-R6 is matched to the most
     congruent residual axis of each half, and the halves' versions are
     compared by congruence.

  B  THE DATA WITHOUT THE CURVATURE. Every column has PC1-3 and their
     curvature removed (the same two methods), and the residual matrix is
     decomposed again, unstandardised. If tier 2 is more than curvature, the
     residual matrix should lead with a stable three-dimensional block
     (B 1-3) that is the space of the original components 4-6, and the
     boundary should stay after the third component: the block 1-4 should
     fail, and so should the components after it (B 4-11, the old 7-14).
     The eigenvalues of the full-sample residual matrix and North's gap ratio
     show where the boundary is.

Also reported, on the full sample: how much of the 4-6 block each method
removes (with the share an in-sample cubic fit removes from permuted
predictors, which is what it removes by chance), and principal cosines
between the residual spaces and the original 4-6.

A smaller block after removal can be less stable simply because it is
smaller: removing variance lowers the eigenvalues while the sampling noise
stays. The gap ratios show whether that is what happened.

Run after pca_rotated.py. Reads output/master_clean.csv.
Writes output/tier2_residual_stability.xlsx and prints every sheet.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from pca_rotated import (MASTER, OUTDIR, SEED, STABLE_THRESH, N_SPLIT_SUB,
                         FEATURE_PREFIXES, TIER1_ANCHORS, TIER2_ANCHORS, standardize,
                         subspace_cosines, congruence, varimax, sign_and_order, banner)

OUT = OUTDIR / "tier2_residual_stability.xlsx"
K1 = len(TIER1_ANCHORS)          # tier 1: components 1-3
K2 = K1 + len(TIER2_ANCHORS)     # tier 2: components 4-6
METHODS = ("cubic", "kNN")
KNN_K = 20
N_FOLDS = 5
N_PERM = 20                      # permutations for the chance removal
B_BLOCKS = {"B 1-3": (0, 3), "B 1-2": (0, 2), "B 1-4": (0, 4), "B 4-11": (3, 11)}


# --------------------------------------------------------------------------- #
# One sample (a half, or the whole)
# --------------------------------------------------------------------------- #
def decompose(X):
    """z-score, eigenvalues (n-1 denominator), eigenvectors as columns,
    unit-variance scores and raw scores."""
    Z = standardize(X)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    ev, V = s ** 2 / (len(Z) - 1), vt.T
    raw = Z @ V
    return Z, ev, V, raw / np.sqrt(ev), raw


def project_out(Y, B):
    """Y minus its least-squares fit on the columns of B."""
    Q, _ = np.linalg.qr(B)
    return Y - Q @ (Q.T @ Y)


def remove_curvature(Y, T1, T1_raw, method):
    """Y with PC1-3 and their curvature removed (see the docstring)."""
    linear = np.column_stack([np.ones(len(Y)), T1])
    if method == "cubic":
        return project_out(Y, np.column_stack(
            [linear, PolynomialFeatures(3, include_bias=False).fit_transform(T1)[:, K1:]]))
    Y = project_out(Y, linear)
    pred = np.empty_like(Y)
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(T1_raw):
        pred[test] = KNeighborsRegressor(KNN_K).fit(T1_raw[train], Y[train]).predict(T1_raw[test])
    return Y - pred


def orthonormal(W):
    Q, _ = np.linalg.qr(W)
    return Q


def analyse(X):
    """Everything compared between halves, for one sample."""
    Z, ev, V, T, raw = decompose(X)
    T1, T1_raw = T[:, :K1], raw[:, :K1]
    Y = T[:, K1:K2]
    _, R = varimax(V[:, K1:K2] * np.sqrt(ev[K1:K2]))
    out = {"raw block": V[:, K1:K2], "Z": Z, "V": V, "ev": ev, "rotation": R, "Y": Y}
    for m in METHODS:
        Yr = remove_curvature(Y, T1, T1_raw, m)
        W = Z.T @ Yr / len(Z)                  # covariances with the columns
        out[f"A {m}"] = orthonormal(W)
        out[f"A-axes {m}"] = W @ R             # one column per rotated axis
        out[f"Yr {m}"] = Yr
        E = remove_curvature(Z, T1, T1_raw, m)
        _, s, vt = np.linalg.svd(E - E.mean(axis=0), full_matrices=False)
        out[f"B {m}"] = vt.T
        out[f"B ev {m}"] = s ** 2 / (len(Z) - 1)
    return out


def rms(c):
    return np.sqrt((c ** 2).mean())


# --------------------------------------------------------------------------- #
# Full sample
# --------------------------------------------------------------------------- #
def full_sample(X, feats):
    full = analyse(X)
    Y = full["Y"]

    banner("VARIANCE OF THE 4-6 BLOCK REMOVED AS CURVATURE (full sample)")
    Z, ev, V, T, raw = decompose(X)
    rng = np.random.default_rng(SEED)
    chance = np.mean([1 - (remove_curvature(Y, rng.permutation(T[:, :K1]), None, "cubic") ** 2).sum()
                      / (Y ** 2).sum() for _ in range(N_PERM)])
    rows = [{"method": m, "share of block variance removed":
             1 - (full[f"Yr {m}"] ** 2).sum() / (Y ** 2).sum()} for m in METHODS]
    rows.append({"method": "cubic, PC1-3 rows permuted (chance)",
                 "share of block variance removed": chance})
    removed = pd.DataFrame(rows)
    print(removed.round(3).to_string(index=False))

    # anchored order and signs of R4-R6, as in pca_rotated.py
    L_var = V[:, K1:K2] * np.sqrt(ev[K1:K2]) @ full["rotation"]
    L, _ = sign_and_order(L_var, full["rotation"], feats, TIER2_ANCHORS, K1, verbose=False)
    order = [int(np.argmax([congruence(L[:, j], L_var[:, i]) for i in range(K2 - K1)]))
             for j in range(K2 - K1)]
    axes = {m: full[f"A-axes {m}"][:, order] for m in METHODS}

    banner("RESIDUAL SPECTRUM (full sample, B): eigenvalues of the data without\n"
           "PC1-3 and their curvature, with North's gap ratio at half the sample size")
    half_error = np.sqrt(2 / (len(X) / 2))
    spec = pd.DataFrame({"position": range(1, 13),
                         "original eigenvalue (component k+3)": ev[K1:K1 + 12]})
    for m in METHODS:
        e = full[f"B ev {m}"][:13]
        spec[f"{m} eigenvalue"] = e[:12]
        spec[f"{m} gap to next / half-sample error"] = (e[:12] - e[1:13]) / (e[:12] * half_error)
    spec["original gap to next / half-sample error"] = (
        (ev[K1:K1 + 12] - ev[K1 + 1:K1 + 13]) / (ev[K1:K1 + 12] * half_error))
    print(spec.round(2).to_string(index=False))

    banner("SAME SPACE? principal cosines with the original components 4-6 (full sample)")
    rows = []
    for m in METHODS:
        for name, sub in ((f"A {m}", full[f"A {m}"]), (f"B 1-3 {m}", full[f"B {m}"][:, :3])):
            c = subspace_cosines(full["raw block"], sub)
            # where the residual space lies in the original components
            share = (V.T @ sub) ** 2
            rows.append({"space": name, "cosines": ", ".join(f"{v:.3f}" for v in c),
                         "RMS cosine": rms(c),
                         "share in PC4-6": share[K1:K2].sum() / 3,
                         "share in PC7-14": share[K2:14].sum() / 3,
                         "share beyond PC14": share[14:].sum() / 3})
        c = subspace_cosines(full[f"A {m}"], full[f"B {m}"][:, :3])
        rows.append({"space": f"A {m} against B 1-3 {m}",
                     "cosines": ", ".join(f"{v:.3f}" for v in c), "RMS cosine": rms(c)})
    same = pd.DataFrame(rows)
    print(same.round(3).to_string(index=False))
    return full, axes, removed, spec, same


# --------------------------------------------------------------------------- #
# Split-halves
# --------------------------------------------------------------------------- #
def split_halves(X, full_axes):
    rng = np.random.default_rng(SEED)
    n = len(X)
    sub = {}         # measure -> list of (rms, min) per split
    axis = {}        # (method, axis) -> list of congruences
    single = {m: [] for m in METHODS}
    names = [f"R{K1 + j + 1}" for j in range(K2 - K1)]
    for _ in range(N_SPLIT_SUB):
        perm = rng.permutation(n)
        h1, h2 = analyse(X[perm[: n // 2]]), analyse(X[perm[n // 2:]])

        def add(name, A, B):
            c = subspace_cosines(A, B)
            sub.setdefault(name, []).append((rms(c), c.min()))

        add("original 4-6", h1["raw block"], h2["raw block"])
        for m in METHODS:
            add(f"A {m}", h1[f"A {m}"], h2[f"A {m}"])
            for label, (a, z) in B_BLOCKS.items():
                add(f"{label} {m}", h1[f"B {m}"][:, a:z], h2[f"B {m}"][:, a:z])
            single[m].append(np.abs(h1[f"B {m}"][:, :4].T @ h2[f"B {m}"][:, :4]).max(axis=1))
            matched = []
            for h in (h1, h2):
                cand = h[f"A-axes {m}"]
                matched.append([cand[:, int(np.argmax([congruence(full_axes[m][:, j], cand[:, i])
                                                       for i in range(cand.shape[1])]))]
                                for j in range(len(names))])
            for j, a in enumerate(names):
                axis.setdefault((m, a), []).append(congruence(matched[0][j], matched[1][j]))

    banner(f"SPLIT-HALF AGREEMENT (x{N_SPLIT_SUB}): subspaces, RMS principal cosine")
    rows = []
    for name, v in sub.items():
        v = np.array(v)
        rows.append({"measure": name, "RMS cosine p05": np.percentile(v[:, 0], 5),
                     "RMS cosine median": np.median(v[:, 0]),
                     "smallest cosine p05": np.percentile(v[:, 1], 5),
                     "passes (RMS p05 >= 0.90)": np.percentile(v[:, 0], 5) >= STABLE_THRESH})
    subs = pd.DataFrame(rows)
    print(subs.round(3).to_string(index=False))

    banner("SPLIT-HALF AGREEMENT: residual axes one by one (A-axes) and the\n"
           "leading components of the residual matrix one by one (B)")
    rows = [{"measure": f"A-axes {m}", "axis": a, "congruence p05": np.percentile(v, 5),
             "congruence median": np.median(v)} for (m, a), v in axis.items()]
    for m in METHODS:
        s = np.array(single[m])
        rows += [{"measure": f"B {m}", "axis": f"component {j + 1}",
                  "congruence p05": np.percentile(s[:, j], 5),
                  "congruence median": np.median(s[:, j])} for j in range(s.shape[1])]
    ax = pd.DataFrame(rows)
    print(ax.round(3).to_string(index=False))
    return subs, ax


def main():
    df = pd.read_csv(MASTER, index_col="onet_soc")
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    X = df[feats].to_numpy(float)
    print(f"{X.shape[0]} occupations x {X.shape[1]} feature columns (from {MASTER}); "
          f"tier 1 = components 1-{K1}, tier 2 = components {K1 + 1}-{K2}")
    full, axes, removed, spec, same = full_sample(X, feats)
    subs, ax = split_halves(X, axes)
    with pd.ExcelWriter(OUT) as writer:
        removed.to_excel(writer, sheet_name="variance_removed", index=False)
        spec.to_excel(writer, sheet_name="residual_spectrum", index=False)
        same.to_excel(writer, sheet_name="same_space", index=False)
        subs.to_excel(writer, sheet_name="split_half_subspaces", index=False)
        ax.to_excel(writer, sheet_name="split_half_axes", index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
