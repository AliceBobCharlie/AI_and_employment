#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_curvature.py -- are the axes related non-linearly: within tier 1, and
how much of what comes after each tier is curvature of the tiers before it,
and is what is left still structure? (Sections 5.3 and 5.5)

PCA describes linear structure. If occupations lie on a curved surface -- and
the 1-5 rating scales have a floor, a common cause of curvature -- PCA needs
extra linear directions to describe the bend, and those directions carry no
information of their own: they are functions of the leading components. The
classic case is the arch or horseshoe effect. The same question is asked at
both tier boundaries, after component 3 and after component 6, and first
among the tier-1 axes themselves.

Everything is computed on the paper's matrix: the 910 occupations and 216
feature columns of output/master_clean.csv (Importance for Abilities, Skills,
Knowledge and Work Activities, Context for Work Context), z-scored with the
population sd as in pca_rotated.py. PC scores are divided by the square root
of their eigenvalue, so every score has unit variance.

Two rules hold throughout.
  - "How much can be predicted" is always an out-of-fold R2 (five folds,
    repeated four times) with a permutation null: the target is permuted
    against the predictors, N_PERM times, and the 95th percentile of R2 is
    what chance gives.
  - "Take it out" is always an in-sample least-squares fit, with a baseline
    from the same fit on row-permuted predictors (what the regressors remove
    by chance), or an out-of-fold nearest-neighbour fit as the check.
  - The polynomial degree keeps the regressors few against half the sample
    (455): cubic in PC1-3 (19 regressors), quadratic in PC1-6 (27; cubic
    would be 83).
Any polynomial in PC1-k is also a polynomial of the same degree in any
rotation of them, so the varimax rotations do not affect any of this.

Models for the predictions:
  quadratic   the predictors, their squares and pairwise products, ridge
  cubic       all monomials up to degree 3, ridge
  kNN         k-nearest-neighbour regression on the raw scores (distances in
              the data), k chosen within each training fold from 10, 20, 40
Ridge penalties are chosen within each training fold.

PART 0 -- WITHIN TIER 1: do R1, R2 and R3 depend on one another beyond the
           linear? Varimax makes them uncorrelated, not independent, and the
           promax refit lets them correlate (about -0.2 to +0.3); both are
           tested. R1-R3 are rebuilt here exactly as in pca_rotated.py, and
           kNN distances are in the unit-variance axis scores.
  0a  The conditional mean: each axis predicted from each other axis alone
      and from the other two, by the three models, with nulls. Under varimax
      the linear part is zero by construction; under promax it is the factor
      correlation.
  0b  Beyond the mean: distance correlation of each pair (zero only under
      independence), before and after removing the linear relation, with
      permutation p-values (the smallest possible p is 1 / (N_PERM + 1));
      and whether the spread of one axis changes along another (R2 of the
      absolute kNN residual, cubic, out of fold, null).
  0c  The shape: R2 by decile of R1 (mean, p10, median, p90, sd, share
      positive); the four corner cells of the R1 x R2 quarters against the
      910 / 16 independence would give, with examples; mean and sd of R3 over
      the R1 x R2 terciles.
  0d  The stereotype that physical work needs less judgement: is it about
      physical effort or physical skill? Mean z-scores of exertion items
      (strength, stamina, standing, bending, handling objects), skilled
      physical items (depth perception, control precision, reaction time,
      operating vehicles and equipment, inspecting, hazardous equipment) and
      judgement items (complex problem solving, judgement and decision
      making, deductive reasoning, critical thinking, making decisions); their
      correlations with R1-R3, and judgement against each kind of physical
      work holding the other fixed (partial correlation).

PART 1 -- THE BOUNDARY AT 3: is the second tier curvature of the first?
  1a  R4, R5 and R6 (varimax within 4-6, as in the paper) predicted from R1-R3
      by the three models, with nulls; the 4-6 block is also scored as a whole
      (one R2 over the three targets, which does not depend on the rotation
      within it).
  1b  Which bends: each square and product of R1-R3, the share of its
      variance lying in the span of PC1-3, PC4-6, PC7-14 and PC15-216 (in
      sample), against the share a random vector would have, m / (n - 1) for
      a span of m directions. A term in PC1-3 is mostly skewness of the axes
      themselves; a term in PC4-6 is tier-1 curvature absorbed by tier 2.
  1c  Take it out and test again. Everything is done separately in each half
      of a split, so the halves share nothing: each half is z-scored and
      decomposed on its own, and its own PC1-3 define and remove its own
      curvature (cubic least squares; or the linear part removed exactly and
      then a 20-nearest-neighbour fit out of fold). The halves are compared by
      principal angles, agreement being the root mean square (RMS) of the
      cosines, p05 over N_SPLIT_SUB split-halves, against 0.90 as in
      pca_rotated.py; the original 4-6 block is scored on the same splits.
        A  the PC4-6 scores with the curvature removed; the covariances of
           the residual scores with the 216 columns are the residual's
           directions, which validate_tier2.py interprets. Compared as a
           subspace, and axis by axis: each half is varimax-rotated on its
           own, its residual axes are paired one to one with the full-sample
           R4-R6 (labelling only), and the halves' versions are compared.
        B  every column with PC1-3 and their curvature removed, the residual
           matrix decomposed again (unstandardised). Tier 2 is more than
           curvature if the residual matrix leads with a stable
           three-dimensional block (B 1-3) in the space of the original 4-6,
           and the boundary after it holds (B 1-4 and B 4-11 fail).
      Also, on the full sample: the share of the 4-6 block each method
      removes, with the chance share; the eigenvalues of the residual matrix
      with North's gap ratio; and principal cosines between the residual
      spaces and the original 4-6.

PART 2 -- THE BOUNDARY AT 6: are components 7-14 curvature of 1-6?
  2a  Each of PC7 ... PC14 predicted from PC1 ... PC6 by the three models
      (quadratic 27 terms, cubic 83), with nulls, and the 7-14 block as a
      whole. Honesty check: PCA refitted on each training fold and the test
      occupations projected onto it, so nothing about a test occupation
      enters the components it is scored on. Robustness: the block R2
      without the 2 percent of occupations furthest from the centre in
      PC1-6. Positive control: synthetic data with the real data's
      six-component structure and no curvature, then censored at each
      column's 10th or 25th percentile to create a floor; the test should
      find nothing without the floor and something with it.
  2b  Spectrum after removing curvature: each column regressed on [PC1-6,
      their squares and products]; the eigenvalues of the residual matrix
      against the eigenvalues after removing PC1-6 linearly (eigenvalues 7,
      8, ...), and against the same removal with row-permuted squares and
      products. The shrinkage beyond that baseline is what curvature accounts
      for. No stability test is repeated here: no block within 7-14 holds
      even with the curvature in (pca_rotated.py), and removing variance
      cannot make one hold.
  2c  Which bends: the same span shares as 1b for all 21 squares and products
      of R1-R6, sorted by the share in PC7-14.

PART 3 -- ROBUSTNESS: is the curvature the floor of the rating scales?
      Each column replaced by the normal scores of its ranks, which removes
      most of the non-linearity a floor produces while keeping the order of
      occupations. Eigenvalues, parallel analysis, the principal angles
      between the original and transformed leading-k subspaces, and the
      nested split-half stability curve of pca_rotated.py on the transformed
      matrix.

Where components 7-14 load by block (candidate method factors) is in
rating_error.py, with the rest of the rating-error analysis.

Run from the project root:  python check_curvature.py
Writes output/curvature_check.xlsx and prints every sheet.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pca_rotated import (STABLE_THRESH, N_SPLIT_SUB, TIER1_ANCHORS, TIER2_ANCHORS,
                         split_statistics, subspace_cosines, varimax, congruence,
                         sign_and_order, match_axes, rotate_tier, promax)

CLEAN = Path("output/master_clean.csv")
OUT = Path("output/curvature_check.xlsx")

SEED = 0
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")
N_LEAD = 6               # components whose curvature is tested
TARGETS = range(6, 14)   # PC7 ... PC14 (zero-based)
N_FOLDS, N_REPEATS = 5, 4
N_PERM = 50              # permutations for the nulls in A, B and C
N_PERM_KNN = 20          # kNN is slower; fewer permutations
N_SPLIT_D = 50           # split-half runs for the transformed matrix in D
ALPHAS = np.logspace(-4, 4, 30)
KNN_K = [10, 20, 40]
FLOORS = [0.0, 0.10, 0.25]
K1 = len(TIER1_ANCHORS)          # tier 1: components 1-3
K2 = K1 + len(TIER2_ANCHORS)     # tier 2: components 4-6
REMOVAL_METHODS = ("cubic", "kNN")
REMOVAL_KNN_K = 20               # neighbours for the kNN removal in 1c
B_BLOCKS = {"B 1-3": (0, 3), "B 1-2": (0, 2), "B 1-4": (0, 4), "B 4-11": (3, 11)}


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #
def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def pca(Z):
    """Eigenvalues (n-1 denominator), eigenvectors (columns), unit-variance scores."""
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    ev = s ** 2 / (len(Z) - 1)
    V = vt.T
    return ev, V, (Zc @ V) / np.sqrt(ev)


def tiered_scores(ev, V, T, return_loadings=False):
    """R1-R6: varimax within PCs 1-3 and within 4-6, scores rotated as in
    pca_rotated.py (standardised unrotated scores times the rotation).
    Within each tier the axes are in the order varimax returns them."""
    scores, loadings = [], []
    for idx in ([0, 1, 2], [3, 4, 5]):
        L, R = varimax(V[:, idx] * np.sqrt(ev[idx]))
        scores.append(T[:, idx] @ R)
        loadings.append(L)
    if return_loadings:
        return np.hstack(scores), np.hstack(loadings)
    return np.hstack(scores)


# --------------------------------------------------------------------------- #
# Models and cross-validated R2
# --------------------------------------------------------------------------- #
def model(kind):
    if kind == "quadratic":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             RidgeCV(alphas=ALPHAS))
    if kind == "cubic":
        return make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(),
                             RidgeCV(alphas=ALPHAS))
    if kind == "kNN":
        return GridSearchCV(KNeighborsRegressor(), {"n_neighbors": KNN_K},
                            cv=KFold(3, shuffle=True, random_state=SEED))
    raise ValueError(kind)


def oof_r2(kind, X, Y, seed=SEED, repeats=N_REPEATS):
    """Out-of-fold R2 per target column of Y, one value per repeat.
    Returns an array (repeats, n_targets)."""
    Y = np.atleast_2d(Y.T).T
    out = np.empty((repeats, Y.shape[1]))
    for r in range(repeats):
        pred = np.empty_like(Y)
        for train, test in KFold(N_FOLDS, shuffle=True, random_state=seed + r).split(X):
            for j in range(Y.shape[1]):
                m = model(kind).fit(X[train], Y[train, j])
                pred[test, j] = m.predict(X[test])
        out[r] = 1 - ((Y - pred) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return out


def block_r2(kind, X, Y, seed=SEED, repeats=N_REPEATS):
    """One R2 for a block of targets: 1 - total residual SS / total SS. It does
    not depend on how the targets are rotated within the block."""
    vals = []
    for r in range(repeats):
        pred = np.empty_like(Y)
        for train, test in KFold(N_FOLDS, shuffle=True, random_state=seed + r).split(X):
            for j in range(Y.shape[1]):
                pred[test, j] = model(kind).fit(X[train], Y[train, j]).predict(X[test])
        vals.append(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum())
    return np.array(vals)


def null_p95(kind, X, y, rng, n_perm):
    """95th percentile of single-repeat out-of-fold R2 with y permuted."""
    vals = [oof_r2(kind, X, rng.permutation(y), repeats=1)[0, 0] for _ in range(n_perm)]
    return np.percentile(vals, 95)


def refit_in_fold(Xraw, kind, seed=SEED):
    """Test A with the PCA refitted inside each training fold. Returns
    out-of-fold R2 per target (PC7-14 of each fold's own decomposition) and
    for the 7-14 block as a whole."""
    n = len(Xraw)
    pred = np.empty((n, len(TARGETS)))
    actual = np.empty((n, len(TARGETS)))
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=seed).split(Xraw):
        mu, sd = Xraw[train].mean(axis=0), Xraw[train].std(axis=0)
        Ztr, Zte = (Xraw[train] - mu) / sd, (Xraw[test] - mu) / sd
        ev, V, Ttr = pca(Ztr)
        Tte = ((Zte - Ztr.mean(axis=0)) @ V) / np.sqrt(ev)
        for j, t in enumerate(TARGETS):
            m = model(kind).fit(Ttr[:, :N_LEAD], Ttr[:, t])
            pred[test, j] = m.predict(Tte[:, :N_LEAD])
            actual[test, j] = Tte[:, t]
    ss_res = ((actual - pred) ** 2).sum(axis=0)
    ss_tot = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot, 1 - ss_res.sum() / ss_tot.sum()


# --------------------------------------------------------------------------- #
# Part 0: within tier 1
# --------------------------------------------------------------------------- #
AXIS_NAMES = ["R1", "R2", "R3"]
# Item composites for 0d: two kinds of physical work, and judgement.
EXERTION = ["abilities_im__Static Strength", "abilities_im__Dynamic Strength",
            "abilities_im__Trunk Strength", "abilities_im__Stamina",
            "workctx_cx__Spend Time Standing", "workctx_cx__Spend Time Bending or Twisting Your Body",
            "workact_im__Handling and Moving Objects"]
SKILLED_PHYSICAL = ["abilities_im__Depth Perception", "abilities_im__Control Precision",
                    "abilities_im__Reaction Time",
                    "workact_im__Operating Vehicles, Mechanized Devices, or Equipment",
                    "workact_im__Inspecting Equipment, Structures, or Materials",
                    "workctx_cx__Exposed to Hazardous Equipment"]
JUDGEMENT = ["skills_im__Complex Problem Solving", "skills_im__Judgment and Decision Making",
             "abilities_im__Deductive Reasoning", "skills_im__Critical Thinking",
             "workact_im__Making Decisions and Solving Problems"]


def tier1_scores(ev, V, T, cols):
    """R1-R3 as in pca_rotated.py (varimax within PC1-3, ordered by variance,
    signed by the anchors), and their promax version: pattern = A U, scores =
    varimax scores U^-T, standardised, each signed to agree with its varimax
    axis."""
    A, R = rotate_tier(V, ev, list(range(K1)))
    A, R = sign_and_order(A, R, cols, TIER1_ANCHORS, 0, verbose=False)
    S = T[:, :K1] @ R
    P, Phi, U = promax(A)
    So = S @ np.linalg.inv(U).T
    for j in range(K1):
        if A[:, j] @ P[:, j] < 0:
            So[:, j] *= -1
    So = (So - So.mean(axis=0)) / So.std(axis=0)
    return {"varimax": S, "promax": So}


def distance_correlation(x, y):
    """Szekely's distance correlation of two 1-d samples: zero only under
    independence, whatever the form of the dependence."""
    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()
    a, b = centred(x), centred(y)
    return np.sqrt(max((a * b).mean(), 0) / np.sqrt((a * a).mean() * (b * b).mean()))


def part_0_mean(scores, rng):
    """0a: each axis predicted from each other axis alone and from the other
    two, out of fold, with the permutation nulls."""
    rows = []
    for rot, M in scores.items():
        for j in range(K1):
            others = [i for i in range(K1) if i != j]
            for preds in [[i] for i in others] + [others]:
                X, y = M[:, preds], M[:, j]
                row = {"rotation": rot, "target": AXIS_NAMES[j],
                       "predictors": " + ".join(AXIS_NAMES[i] for i in preds),
                       "Pearson r": np.corrcoef(X[:, 0], y)[0, 1] if len(preds) == 1 else np.nan}
                for kind in ("quadratic", "cubic", "kNN"):
                    r2 = oof_r2(kind, X, y)[:, 0]
                    row[f"{kind} R2 mean"] = r2.mean()
                row["cubic null p95"] = null_p95("cubic", X, y, rng, N_PERM)
                row["kNN null p95"] = null_p95("kNN", X, y, rng, N_PERM_KNN)
                rows.append(row)
    return pd.DataFrame(rows)


def part_0_beyond_mean(scores, rng):
    """0b: dependence the conditional mean misses. Distance correlation of
    each pair, before and after removing the linear relation, with
    permutation p-values; and whether the spread of one axis changes with the
    other (R2 of the absolute kNN residual on the predictor, cubic, out of
    fold)."""
    rows = []
    for rot, M in scores.items():
        for i in range(K1):
            for j in range(K1):
                if i == j:
                    continue
                x, y = M[:, i], M[:, j]
                row = {"rotation": rot, "target": AXIS_NAMES[j], "predictor": AXIS_NAMES[i]}
                if i < j:
                    ry = y - np.polyval(np.polyfit(x, y, 1), x)
                    for label, v in (("", y), (" after removing linear", ry)):
                        d = distance_correlation(x, v)
                        null = np.array([distance_correlation(x, rng.permutation(v))
                                         for _ in range(N_PERM)])
                        row[f"distance corr{label}"] = d
                        row[f"null p95{label}"] = np.percentile(null, 95)
                        row[f"p{label}"] = ((null >= d).sum() + 1) / (N_PERM + 1)
                X = x[:, None]
                fit = np.empty_like(y)
                for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
                    fit[test] = model("kNN").fit(X[train], y[train]).predict(X[test])
                spread = np.abs(y - fit)
                row["spread: R2 of |residual| on predictor (cubic)"] = oof_r2("cubic", X, spread)[:, 0].mean()
                row["spread: null p95"] = null_p95("cubic", X, spread, rng, N_PERM)
                rows.append(row)
    return pd.DataFrame(rows)


def part_0_shape(scores, titles):
    """0c: the shape of the joint distribution. R2 by decile of R1; the four
    corner cells of the R1 x R2 quarters against the count independence would
    give; mean and sd of R3 over the R1 x R2 terciles."""
    deciles, corners, grid = [], [], []
    n = len(titles)
    for rot, M in scores.items():
        r1, r2, r3 = M[:, 0], M[:, 1], M[:, 2]
        d = pd.DataFrame({"bin": pd.qcut(r1, 10, labels=False), "R1": r1, "R2": r2})
        g = d.groupby("bin")
        t = g.agg(**{"R1 mean": ("R1", "mean"), "R2 mean": ("R2", "mean"),
                     "R2 p10": ("R2", lambda v: v.quantile(.1)), "R2 median": ("R2", "median"),
                     "R2 p90": ("R2", lambda v: v.quantile(.9)), "R2 sd": ("R2", "std"),
                     "share R2 > 0": ("R2", lambda v: (v > 0).mean())}).reset_index()
        deciles.append(t.assign(rotation=rot))
        q1, q2 = np.quantile(r1, [.25, .75]), np.quantile(r2, [.25, .75])
        for lab1, m1 in (("R1 top quarter", r1 >= q1[1]), ("R1 bottom quarter", r1 <= q1[0])):
            for lab2, m2 in (("R2 top quarter", r2 >= q2[1]), ("R2 bottom quarter", r2 <= q2[0])):
                m = m1 & m2
                far = np.argsort(-(np.abs(r1[m]) + np.abs(r2[m])))[:6]
                corners.append({"rotation": rot, "cell": f"{lab1} x {lab2}", "n": int(m.sum()),
                                "expected if independent": n / 16,
                                "examples": "; ".join(titles[m][far])})
        c = pd.DataFrame({"R1 tercile": pd.qcut(r1, 3, labels=["R1 low", "R1 mid", "R1 high"]),
                          "R2 tercile": pd.qcut(r2, 3, labels=["R2 low", "R2 mid", "R2 high"]),
                          "R3": r3})
        for stat in ("mean", "std"):
            t = c.pivot_table(index="R1 tercile", columns="R2 tercile", values="R3",
                              aggfunc=stat, observed=True).reset_index()
            grid.append(t.assign(rotation=rot, statistic=f"R3 {stat}"))
    return (pd.concat(deciles, ignore_index=True), pd.DataFrame(corners),
            pd.concat(grid, ignore_index=True))


def part_0_composites(Z, cols, scores):
    """0d: is the 'physical work needs less judgement' pattern about physical
    effort or about physical skill? Mean z-scores of exertion items, skilled
    physical items and judgement items; their correlations with R1-R3, and
    the correlation of judgement with each kind of physical work holding the
    other fixed."""
    idx = {c: j for j, c in enumerate(cols)}
    comp = {name: Z[:, [idx[c] for c in items if c in idx]].mean(axis=1)
            for name, items in (("exertion", EXERTION), ("skilled physical", SKILLED_PHYSICAL),
                                ("judgement", JUDGEMENT))}

    def partial(y, x, c):
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        ry = y - np.polyval(np.polyfit(c, y, 1), c)
        return np.corrcoef(rx, ry)[0, 1]

    S = scores["varimax"]
    rows = [{"composite": name, "items": "; ".join(c.split("__")[1] for c in items),
             **{f"r with {a}": np.corrcoef(comp[name], S[:, j])[0, 1]
                for j, a in enumerate(AXIS_NAMES)}}
            for name, items in (("exertion", EXERTION), ("skilled physical", SKILLED_PHYSICAL),
                                ("judgement", JUDGEMENT))]
    E, K, J = comp["exertion"], comp["skilled physical"], comp["judgement"]
    pairs = pd.DataFrame([
        {"pair": "exertion, skilled physical", "correlation": np.corrcoef(E, K)[0, 1],
         "held fixed": "", "partial correlation": np.nan},
        {"pair": "judgement, exertion", "correlation": np.corrcoef(J, E)[0, 1],
         "held fixed": "skilled physical", "partial correlation": partial(J, E, K)},
        {"pair": "judgement, skilled physical", "correlation": np.corrcoef(J, K)[0, 1],
         "held fixed": "exertion", "partial correlation": partial(J, K, E)},
    ])
    return pd.DataFrame(rows), pairs


# --------------------------------------------------------------------------- #
# Parts 1 to 3
# --------------------------------------------------------------------------- #
def section_a(ev, V, T, Xraw, rng, label="observed"):
    X_poly = T[:, :N_LEAD]
    X_knn = T[:, :N_LEAD] * np.sqrt(ev[:N_LEAD])        # raw scores: distances in the data
    rows = []
    fits = {"quadratic": oof_r2("quadratic", X_poly, T[:, list(TARGETS)]),
            "cubic": oof_r2("cubic", X_poly, T[:, list(TARGETS)]),
            "kNN": oof_r2("kNN", X_knn, T[:, list(TARGETS)])}
    for j, t in enumerate(TARGETS):
        row = {"data": label, "component": f"PC{t + 1}", "eigenvalue": ev[t]}
        for kind, r2 in fits.items():
            row[f"{kind} R2 mean"] = r2[:, j].mean()
            row[f"{kind} R2 min-max"] = f"{r2[:, j].min():.3f} to {r2[:, j].max():.3f}"
        row["quadratic null p95"] = null_p95("quadratic", X_poly, T[:, t], rng, N_PERM)
        row["kNN null p95"] = null_p95("kNN", X_knn, T[:, t], rng, N_PERM_KNN)
        rows.append(row)
    table = pd.DataFrame(rows)
    blk = {kind: block_r2(kind, X_knn if kind == "kNN" else X_poly, T[:, list(TARGETS)]).mean()
           for kind in fits}
    refit = {}
    if Xraw is not None:
        for kind in ("quadratic", "kNN"):
            per, whole = refit_in_fold(Xraw, kind)
            table[f"{kind} R2, PCA refitted in fold"] = per
            refit[kind] = whole
    return table, blk, refit


def section_b(ev, V, T, rng, cols):
    S, L = tiered_scores(ev, V, T, return_loadings=True)
    tier1, tier2 = S[:, :3], S[:, 3:]
    top = [cols[int(np.argmax(np.abs(L[:, 3 + j])))].split("__", 1)[1][:40] for j in range(3)]
    X_knn = T[:, :3] * np.sqrt(ev[:3])
    rows = []
    fits = {"quadratic": oof_r2("quadratic", tier1, tier2),
            "cubic": oof_r2("cubic", tier1, tier2),
            "kNN": oof_r2("kNN", X_knn, tier2)}
    for j in range(3):
        row = {"target": f"R{j + 4}, top-loading item: {top[j]}"}
        for kind, r2 in fits.items():
            row[f"{kind} R2 mean"] = r2[:, j].mean()
            row[f"{kind} R2 min-max"] = f"{r2[:, j].min():.3f} to {r2[:, j].max():.3f}"
        row["quadratic null p95"] = null_p95("quadratic", tier1, tier2[:, j], rng, N_PERM)
        row["kNN null p95"] = null_p95("kNN", X_knn, tier2[:, j], rng, N_PERM_KNN)
        rows.append(row)
    whole = {"target": "PCs 4-6 as a block (rotation-free)"}
    for kind in fits:
        r = block_r2(kind, X_knn if kind == "kNN" else tier1, T[:, 3:6])
        whole[f"{kind} R2 mean"] = r.mean()
        whole[f"{kind} R2 min-max"] = f"{r.min():.3f} to {r.max():.3f}"
    rows.append(whole)
    return pd.DataFrame(rows)


def residual_spectrum(Z, regressors):
    """Eigenvalues of Z after OLS on [intercept, regressors], per column."""
    A = np.column_stack([np.ones(len(Z)), regressors])
    Q, _ = np.linalg.qr(A)
    E = Z - Q @ (Q.T @ Z)
    return np.linalg.svd(E, compute_uv=False) ** 2 / (len(Z) - 1)


def quad_terms(T6):
    full = PolynomialFeatures(2, include_bias=False).fit_transform(T6)
    return full[:, T6.shape[1]:]          # the 21 squares and products only


def section_c(Z, T, rng):
    T6 = T[:, :N_LEAD]
    lin = residual_spectrum(Z, T6)
    quad = residual_spectrum(Z, np.column_stack([T6, quad_terms(T6)]))
    base = np.array([residual_spectrum(Z, np.column_stack([T6, quad_terms(rng.permutation(T6))]))
                     for _ in range(N_PERM)])
    rows = []
    for j in range(len(TARGETS)):
        b = base[:, j]
        rows.append({
            "position (original component)": f"{j + 1} (PC{j + 7})",
            "eigenvalue after removing PC1-6 linearly": lin[j],
            "after also removing their squares and products": quad[j],
            "baseline: permuted squares and products, mean": b.mean(),
            "baseline p05": np.percentile(b, 5),
            "shrinkage %": 100 * (1 - quad[j] / lin[j]),
            "baseline shrinkage %": 100 * (1 - b.mean() / lin[j]),
        })
    total = {"position (original component)": "sum over positions 1-8",
             "eigenvalue after removing PC1-6 linearly": lin[:8].sum(),
             "after also removing their squares and products": quad[:8].sum(),
             "baseline: permuted squares and products, mean": base[:, :8].sum(axis=1).mean(),
             "baseline p05": np.percentile(base[:, :8].sum(axis=1), 5)}
    total["shrinkage %"] = 100 * (1 - total["after also removing their squares and products"]
                                  / total["eigenvalue after removing PC1-6 linearly"])
    total["baseline shrinkage %"] = 100 * (1 - total["baseline: permuted squares and products, mean"]
                                           / total["eigenvalue after removing PC1-6 linearly"])
    rows.append(total)
    return pd.DataFrame(rows)


def normal_scores(X):
    n = len(X)
    return np.column_stack([norm.ppf((rankdata(col) - 0.5) / n) for col in X.T])


def parallel_count(Z, rng, n_perm=N_PERM):
    n = len(Z)
    ev = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False) ** 2 / (n - 1)
    null = np.array([np.linalg.svd(standardize(np.column_stack([rng.permutation(c) for c in Z.T])),
                                   compute_uv=False)[:30] ** 2 / (n - 1) for _ in range(n_perm)])
    above = ev[:30] > np.percentile(null, 95, axis=0)
    return int(np.argmin(above)) if not above.all() else 30


def section_d(Xraw, Z, rng):
    Zr = standardize(normal_scores(Xraw))
    ev, V, _ = pca(Z)
    evr, Vr, _ = pca(Zr)
    min_cos, msq_cos, single, _ = split_statistics(Zr, rng, N_SPLIT_D)
    min_o, msq_o, single_o, _ = split_statistics(Z, rng, N_SPLIT_D)
    rows = []
    for k in range(1, 15):
        c = subspace_cosines(V[:, :k], Vr[:, :k])
        rows.append({
            "k": k,
            "eigenvalue, original": ev[k - 1], "eigenvalue, normal scores": evr[k - 1],
            "original vs normal scores: min cosine": c.min(),
            "original vs normal scores: mean sq cosine": (c ** 2).mean(),
            "split-half mean sq cos median, original": np.median(msq_o[:, k - 1]),
            "split-half mean sq cos median, normal scores": np.median(msq_cos[:, k - 1]),
            "split-half min cos p05, original": np.percentile(min_o[:, k - 1], 5),
            "split-half min cos p05, normal scores": np.percentile(min_cos[:, k - 1], 5),
            "component k alone p05, original": np.percentile(single_o[:, k - 1], 5),
            "component k alone p05, normal scores": np.percentile(single[:, k - 1], 5),
        })
    table = pd.DataFrame(rows)
    pa = {"original": parallel_count(Z, rng), "normal scores": parallel_count(Zr, rng)}
    share = {"original, first 14": ev[:14].sum() / Z.shape[1],
             "normal scores, first 14": evr[:14].sum() / Z.shape[1]}
    return table, pa, share


def section_e(ev, V, rng):
    """Positive control on synthetic data shaped like the real six components."""
    L6 = V[:, :N_LEAD] * np.sqrt(ev[:N_LEAD])
    h = np.clip((L6 ** 2).sum(axis=1), 0, 0.99)
    n = 910
    rows = []
    for q in FLOORS:
        zl = rng.standard_normal((n, N_LEAD))
        X = zl @ L6.T + rng.standard_normal((n, L6.shape[0])) * np.sqrt(1 - h)
        if q > 0:
            X = np.maximum(X, np.quantile(X, q, axis=0))
        Zs = standardize(X)
        evs, Vs, Ts = pca(Zs)
        table, blk, _ = section_a(evs, Vs, Ts, None, rng, label=f"synthetic, floor at p{int(q * 100)}")
        rows.append(table.assign(**{"7-14 block quadratic R2": blk["quadratic"],
                                    "7-14 block kNN R2": blk["kNN"]}))
    return pd.concat(rows, ignore_index=True)


def span_r2(y, B):
    """In-sample R2 of y on the columns of B (with intercept)."""
    A = np.column_stack([np.ones(len(y)), B])
    Q, _ = np.linalg.qr(A)
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def section_g(ev, V, T, cols):
    """Which squares and products of the rotated axes R1-R6 carry the
    curvature: for each of the 21 terms, the share of its variance that lies in
    the span of PC1-6, of PC7-14 and of PC15-216 (in-sample). By chance alone a
    vector falls in a span of m directions in proportion m / (n - 1)."""
    S, L = tiered_scores(ev, V, T, return_loadings=True)
    top = [cols[int(np.argmax(np.abs(L[:, j])))].split("__", 1)[1] for j in range(N_LEAD)]
    names = [f"R{j + 1} ({top[j][:32]})" for j in range(N_LEAD)]
    n = len(T)
    rows = []
    for i in range(N_LEAD):
        for j in range(i, N_LEAD):
            term = S[:, i] * S[:, j]
            rows.append({
                "term": f"R{i + 1}^2" if i == j else f"R{i + 1} x R{j + 1}",
                "in span of PC1-3 (tier 1)": span_r2(term, T[:, :3]),
                "in span of PC4-6 (tier 2)": span_r2(term, T[:, 3:6]),
                "in span of PC7-14": span_r2(term, T[:, 6:14]),
                "in span of PC15-216": span_r2(term, T[:, 14:]),
            })
    table = pd.DataFrame(rows).sort_values("in span of PC7-14", ascending=False)
    chance = {"chance, span of 3": 3 / (n - 1), "chance, PC7-14": 8 / (n - 1),
              "chance, PC15-216": (T.shape[1] - 14) / (n - 1)}
    key = pd.DataFrame({"axis": names})
    return table, chance, key


def outlier_check(T, rng_seed=SEED, drop_share=0.02):
    """7-14 block R2 (quadratic) without the occupations most extreme in PC1-6,
    keeping the full-sample decomposition."""
    d2 = (T[:, :N_LEAD] ** 2).sum(axis=1)
    keep = d2 <= np.quantile(d2, 1 - drop_share)
    r = block_r2("quadratic", T[keep, :N_LEAD], T[keep][:, list(TARGETS)], seed=rng_seed)
    return r.mean(), int((~keep).sum())


# --------------------------------------------------------------------------- #
# Part 1c: remove the curvature of tier 1 and test again
# --------------------------------------------------------------------------- #
def decompose(X):
    """X z-scored, its eigenvalues and eigenvectors, unit-variance scores and
    raw scores."""
    Z = standardize(X)
    ev, V, T = pca(Z)
    return Z, ev, V, T, T * np.sqrt(ev)


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
        pred[test] = KNeighborsRegressor(REMOVAL_KNN_K).fit(T1_raw[train], Y[train]).predict(T1_raw[test])
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
    for m in REMOVAL_METHODS:
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
# Part 1c, full sample
# --------------------------------------------------------------------------- #
def part_1c_full_sample(X, feats):
    full = analyse(X)
    Y = full["Y"]

    _, ev, V, T, _ = decompose(X)
    rng = np.random.default_rng(SEED)
    chance = np.mean([1 - (remove_curvature(Y, rng.permutation(T[:, :K1]), None, "cubic") ** 2).sum()
                      / (Y ** 2).sum() for _ in range(N_PERM)])
    rows = [{"method": m, "share of block variance removed":
             1 - (full[f"Yr {m}"] ** 2).sum() / (Y ** 2).sum()} for m in REMOVAL_METHODS]
    rows.append({"method": "cubic, PC1-3 rows permuted (chance)",
                 "share of block variance removed": chance})
    removed = pd.DataFrame(rows)

    # anchored order and signs of R4-R6, as in pca_rotated.py
    L_var = V[:, K1:K2] * np.sqrt(ev[K1:K2]) @ full["rotation"]
    L, _ = sign_and_order(L_var, full["rotation"], feats, TIER2_ANCHORS, K1, verbose=False)
    order = [int(np.argmax([congruence(L[:, j], L_var[:, i]) for i in range(K2 - K1)]))
             for j in range(K2 - K1)]
    axes = {m: full[f"A-axes {m}"][:, order] for m in REMOVAL_METHODS}

    half_error = np.sqrt(2 / (len(X) / 2))
    spec = pd.DataFrame({"position": range(1, 13),
                         "original eigenvalue (component k+3)": ev[K1:K1 + 12]})
    for m in REMOVAL_METHODS:
        e = full[f"B ev {m}"][:13]
        spec[f"{m} eigenvalue"] = e[:12]
        spec[f"{m} gap to next / half-sample error"] = (e[:12] - e[1:13]) / (e[:12] * half_error)
    spec["original gap to next / half-sample error"] = (
        (ev[K1:K1 + 12] - ev[K1 + 1:K1 + 13]) / (ev[K1:K1 + 12] * half_error))

    rows = []
    for m in REMOVAL_METHODS:
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
    return axes, removed, spec, same


# --------------------------------------------------------------------------- #
# Part 1c, split-halves
# --------------------------------------------------------------------------- #
def part_1c_split_halves(X, full_axes):
    rng = np.random.default_rng(SEED)
    n = len(X)
    sub = {}         # measure -> list of (rms, min) per split
    axis = {}        # (method, axis) -> list of congruences
    single = {m: [] for m in REMOVAL_METHODS}
    names = [f"R{K1 + j + 1}" for j in range(K2 - K1)]
    for _ in range(N_SPLIT_SUB):
        perm = rng.permutation(n)
        h1, h2 = analyse(X[perm[: n // 2]]), analyse(X[perm[n // 2:]])

        def add(name, A, B):
            c = subspace_cosines(A, B)
            sub.setdefault(name, []).append((rms(c), c.min()))

        add("original 4-6", h1["raw block"], h2["raw block"])
        for m in REMOVAL_METHODS:
            add(f"A {m}", h1[f"A {m}"], h2[f"A {m}"])
            for label, (a, z) in B_BLOCKS.items():
                add(f"{label} {m}", h1[f"B {m}"][:, a:z], h2[f"B {m}"][:, a:z])
            single[m].append(np.abs(h1[f"B {m}"][:, :4].T @ h2[f"B {m}"][:, :4]).max(axis=1))
            matched = [match_axes(full_axes[m], h[f"A-axes {m}"]) for h in (h1, h2)]
            for j, a in enumerate(names):
                axis.setdefault((m, a), []).append(congruence(matched[0][:, j], matched[1][:, j]))

    rows = []
    for name, v in sub.items():
        v = np.array(v)
        rows.append({"measure": name, "RMS cosine p05": np.percentile(v[:, 0], 5),
                     "RMS cosine median": np.median(v[:, 0]),
                     "smallest cosine p05": np.percentile(v[:, 1], 5),
                     "passes (RMS p05 >= 0.90)": np.percentile(v[:, 0], 5) >= STABLE_THRESH})
    subs = pd.DataFrame(rows)

    rows = [{"measure": f"A-axes {m}", "axis": a, "congruence p05": np.percentile(v, 5),
             "congruence median": np.median(v)} for (m, a), v in axis.items()]
    for m in REMOVAL_METHODS:
        s = np.array(single[m])
        rows += [{"measure": f"B {m}", "axis": f"component {j + 1}",
                  "congruence p05": np.percentile(s[:, j], 5),
                  "congruence median": np.median(s[:, j])} for j in range(s.shape[1])]
    ax = pd.DataFrame(rows)
    return subs, ax


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    clean = pd.read_csv(CLEAN)
    cols = [c for c in clean.columns if c.startswith(FEATURE_PREFIXES)]
    Xraw = clean[cols].to_numpy(float)
    if np.isnan(Xraw).any():
        raise ValueError("missing cells in the feature matrix")
    Z = standardize(Xraw)
    ev, V, T = pca(Z)
    print(f"[data] {CLEAN}: {Z.shape[0]} occupations x {Z.shape[1]} columns")

    print("[0] within tier 1, varimax and promax")
    scores = tier1_scores(ev, V, T, cols)
    phi = np.corrcoef(scores["promax"].T)
    print("    promax score correlations R1-R2, R1-R3, R2-R3: "
          + ", ".join(f"{phi[i, j]:+.3f}" for i, j in ((0, 1), (0, 2), (1, 2))))
    mean0 = part_0_mean(scores, rng)
    beyond0 = part_0_beyond_mean(scores, rng)
    deciles0, corners0, grid0 = part_0_shape(scores, clean["title"].to_numpy())
    composites0, pairs0 = part_0_composites(Z, cols, scores)

    print("[1a] tier 2 from tier 1")
    b_table = section_b(ev, V, T, rng, cols)
    print("[1b, 2c] which squares and products carry the curvature")
    g_table, g_chance, g_key = section_g(ev, V, T, cols)
    g_key = g_key.assign(**{k: v for k, v in g_chance.items()})
    tier1_terms = g_table[~g_table["term"].str.contains("R[4-9]")].sort_values(
        "in span of PC4-6 (tier 2)", ascending=False)
    print("[1c] tier 2 with the curvature of tier 1 removed, split-halves")
    full_axes, removed, spectrum, same = part_1c_full_sample(Xraw, cols)
    subspaces, axes = part_1c_split_halves(Xraw, full_axes)

    print("[2a] harmonics of PC1-6 in PC7-14")
    a_table, a_block, a_refit = section_a(ev, V, T, Xraw, rng)
    trimmed, n_dropped = outlier_check(T)
    a_summary = pd.DataFrame([{
        "7-14 block R2, quadratic": a_block["quadratic"], "7-14 block R2, cubic": a_block["cubic"],
        "7-14 block R2, kNN": a_block["kNN"],
        "7-14 block R2, quadratic, PCA refitted in fold": a_refit["quadratic"],
        "7-14 block R2, kNN, PCA refitted in fold": a_refit["kNN"],
        f"7-14 block R2, quadratic, without the {n_dropped} most extreme occupations": trimmed}])
    print("[2a] positive control")
    e_table = section_e(ev, V, rng)
    print("[2b] spectrum after removing the curvature of PC1-6")
    c_table = section_c(Z, T, rng)

    print("[3] normal-score transform")
    d_table, d_pa, d_share = section_d(Xraw, Z, rng)
    d_summary = pd.DataFrame([{"parallel analysis, original": d_pa["original"],
                               "parallel analysis, normal scores": d_pa["normal scores"],
                               "variance in first 14, original": d_share["original, first 14"],
                               "variance in first 14, normal scores": d_share["normal scores, first 14"]}])

    sheets = {"0a_mean_dependence": mean0, "0b_beyond_the_mean": beyond0,
              "0c_R2_by_R1_decile": deciles0, "0c_R1_R2_corners": corners0,
              "0c_R3_on_R1_R2_grid": grid0, "0d_physical_composites": composites0,
              "0d_judgement_partials": pairs0,
              "1a_tier2_from_tier1": b_table, "1b_tier1_terms": tier1_terms,
              "1c_variance_removed": removed, "1c_residual_spectrum": spectrum,
              "1c_same_space": same, "1c_split_half_subspaces": subspaces,
              "1c_split_half_axes": axes,
              "2a_harmonics": a_table, "2a_block": a_summary, "2a_positive_control": e_table,
              "2b_residual_spectrum": c_table, "2c_all_terms": g_table,
              "2c_axes_and_chance": g_key,
              "3_normal_scores": d_table, "3_summary": d_summary}
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        for name, df in sheets.items():
            print(f"\n=== {name} ({len(df)} rows) ===")
            print(df.to_string(index=False))
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
