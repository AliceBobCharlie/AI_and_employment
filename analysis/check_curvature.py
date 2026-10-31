#!/usr/bin/env python3
"""
check_curvature.py -- measure how far the rotated axes and the later principal
components are non-linear functions of the earlier ones, and what is left
when that part is removed.

Input:  data/prepared/master_clean.csv, output/resampling_stability/resampled_spectra.npz (k_max and the
        split-half permutations)
Output: one CSV per table in output/check_curvature/, named by part (0a_..., 2b_...)

main() z-scores the FEATURE_PREFIXES columns. pca() gives the eigenvalues,
eigenvectors and unit-variance scores T. axis_scores() rebuilds R1-R6 with
rotate_groups() from pca_rotated.py: the scores are T on each group of
ROTATION_GROUPS times its rotation. Each section draws its random numbers
from its own generator seeded with SEED; section_a() and section_e() share
one.

Prediction: oof_r2() gives the out-of-fold R2 of each target (N_FOLDS folds,
N_REPEATS repeats); block_r2() gives one R2 for a block of targets. The models
in fit_predict() are ridge on quadratic or cubic polynomial terms (penalty per
target from ALPHAS) and nearest-neighbour regression (k per target from KNN_K
by inner three-fold cross-validation). null_p95() refits with the target
permuted N_PERM times and gives the 95th percentile of R2.

Part 0, R1-R3 against one another:
  0a  part_0_mean(): each axis predicted from each other axis and from the
      other two, by the three models, with null_p95().
  0b  part_0_beyond_mean(): distance_correlation() of each pair, before and
      after removing the linear fit, with permutation nulls; and the R2 of the
      absolute kNN residual of one axis on the other (cubic), with its null.
  0c  part_0_shape(): R2 by decile of R1; counts in the four corner cells of
      the R1 x R2 quarters against n / 16, with examples; mean and sd of R3 on
      the R1 x R2 tercile grid.
  0d  part_0_composites(): the mean z-score of the EXERTION, SKILLED_PHYSICAL
      and COMPLEX_REASONING items; their correlations with R1-R3, and the
      partial correlations of the reasoning composite with each physical one.

Part 1, R4-R6 against PC1-3:
  1a  section_b(): each of R4-R6 predicted from R1-R3 by the three models,
      with nulls, and PC4-6 as one block.
  1b  section_g(): each square and product of R1-R3, the share of its variance
      in the span of PC1-3, PC4-6, PC7 to PC k_max and the rest (span_r2()),
      with the share a random vector would have.
  1c  analyse() decomposes a sample and removes from PC4-6 (A) and from every
      column (B) the fit on PC1-3 and their cubic terms, or the linear fit and
      then an out-of-fold REMOVAL_KNN_K-nearest-neighbour fit
      (remove_curvature()). part_1c_full_sample() gives the share of the PC4-6
      variance removed, against the cubic fit on row-permuted PC1-3; the
      eigenvalues of B with North's gap ratio; and principal cosines between A,
      B 1-3 and the original PC4-6. part_1c_split_halves() runs analyse() on
      both halves of each split in split_perm and compares the halves: RMS and
      smallest principal cosine for the original PC4-6, A and the B_BLOCKS,
      against STABLE_THRESH; congruence of each varimax axis of A, paired with
      the full-sample axes by match_axes(); and of each of the first four
      components of B.
  1d  part_1d_axes_after_removal(): the square or product of R1-R3 most
      correlated with each of R4-R6; and, for each removal method, PC4-6 with
      the curvature removed, rotated again by varimax and paired with R4-R6,
      giving each new axis's congruence with the original and its share of the
      total variance.

Part 2, PC7 onwards against PC1-6:
  2a  section_a(): each of PC7 to PC k_max predicted from PC1-6 by the three
      models, with nulls, and refitted with the PCA estimated inside each
      training fold (refit_in_fold()). 2a_block.csv gives the block R2 of
      PC7 to PC k_max with the PCA refitted in each fold, and, from
      outlier_check(), without the occupations furthest from the centre in
      PC1-6. section_e() runs section_a() with the block R2 on synthetic data
      with the loadings of PC1-6, without curvature, and censored at each
      column's FLOORS quantile.
  2b  section_g(): the span shares of 1b for all squares and products of R1-R6.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pca_rotated import (FEATURE_PREFIXES, MASTER, OUTDIR, ROTATION_GROUPS, SEED,  # noqa: E402
                         congruence, match_axes, rotate_groups, subspace_cosines, varimax)
from resampling_stability import STABLE_THRESH, load_spectra                      # noqa: E402

OUT = OUTDIR / "check_curvature"
N_LEAD = 6               # PC1-6
K1, K2 = 3, 6            # R1-R3 are PC1-3; R4-R6 are PC4-6
N_FOLDS, N_REPEATS = 5, 4
N_PERM = 200             # permutations per null
ALPHAS = np.logspace(-4, 4, 30)
KNN_K = [10, 20, 40]
FLOORS = [0.0, 0.10, 0.25]
REMOVAL_METHODS = ("cubic", "kNN")
REMOVAL_KNN_K = 20
B_BLOCKS = {"B 1-3": (0, 3), "B 1-2": (0, 2), "B 1-4": (0, 4), "B 4-11": (3, 11)}
AXIS_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6"]

EXERTION = ["abilities_im__Static Strength", "abilities_im__Dynamic Strength",
            "abilities_im__Trunk Strength", "abilities_im__Stamina",
            "workctx_cx__Spend Time Standing", "workctx_cx__Spend Time Bending or Twisting Your Body",
            "workact_im__Handling and Moving Objects"]
SKILLED_PHYSICAL = ["abilities_im__Depth Perception", "abilities_im__Control Precision",
                    "abilities_im__Reaction Time",
                    "workact_im__Operating Vehicles, Mechanized Devices, or Equipment",
                    "workact_im__Inspecting Equipment, Structures, or Materials",
                    "workctx_cx__Exposed to Hazardous Equipment"]
COMPLEX_REASONING = ["skills_im__Complex Problem Solving", "skills_im__Judgment and Decision Making",
                     "abilities_im__Deductive Reasoning", "skills_im__Critical Thinking",
                     "workact_im__Making Decisions and Solving Problems"]


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


def axis_scores(ev, V, T, cols):
    """R1-R6 scores and loadings."""
    groups = rotate_groups(V, ev, cols, ROTATION_GROUPS, verbose=False)
    S = np.hstack([T[:, idx] @ R for idx, (_, R) in zip(ROTATION_GROUPS, groups)])
    return S, np.hstack([L for L, _ in groups])


# --------------------------------------------------------------------------- #
# Models and cross-validated R2
# --------------------------------------------------------------------------- #
def polynomial_model(kind):
    degree = {"quadratic": 2, "cubic": 3}[kind]
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                         RidgeCV(alphas=ALPHAS, alpha_per_target=True))


def knn_predict_all_k(X_train, Y_train, X_test):
    """Predictions {k: (n_test, n_targets)} for every k in KNN_K from one
    neighbour search."""
    idx = NearestNeighbors(n_neighbors=max(KNN_K)).fit(X_train).kneighbors(X_test, return_distance=False)
    running_sum = np.cumsum(Y_train[idx], axis=1)
    return {k: running_sum[:, k - 1, :] / k for k in KNN_K}


def knn_fit_predict(X_train, Y_train, X_test):
    """kNN predictions with k chosen per target by three-fold cross-validation
    on the training data (first k on ties)."""
    scores = np.zeros((len(KNN_K), Y_train.shape[1]))
    for inner_train, inner_test in KFold(3, shuffle=True, random_state=SEED).split(X_train):
        pred = knn_predict_all_k(X_train[inner_train], Y_train[inner_train], X_train[inner_test])
        y = Y_train[inner_test]
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        for i, k in enumerate(KNN_K):
            scores[i] += 1 - ((y - pred[k]) ** 2).sum(axis=0) / ss_tot
    best = np.argmax(scores, axis=0)
    pred = knn_predict_all_k(X_train, Y_train, X_test)
    return np.column_stack([pred[KNN_K[b]][:, j] for j, b in enumerate(best)])


def fit_predict(kind, X_train, Y_train, X_test):
    if kind == "kNN":
        return knn_fit_predict(X_train, Y_train, X_test)
    return polynomial_model(kind).fit(X_train, Y_train).predict(X_test).reshape(len(X_test), -1)


def oof_predictions(kind, X, Y, seed):
    pred = np.empty_like(Y)
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
        pred[test] = fit_predict(kind, X[train], Y[train], X[test])
    return pred


def oof_r2(kind, X, Y, seed=SEED, repeats=N_REPEATS):
    """Out-of-fold R2, array (repeats, n_targets)."""
    Y = np.atleast_2d(Y.T).T
    out = np.empty((repeats, Y.shape[1]))
    for r in range(repeats):
        pred = oof_predictions(kind, X, Y, seed + r)
        out[r] = 1 - ((Y - pred) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return out


def block_r2(kind, X, Y, seed=SEED, repeats=N_REPEATS):
    """1 - total residual SS / total SS over all targets, one value per repeat."""
    vals = []
    for r in range(repeats):
        pred = oof_predictions(kind, X, Y, seed + r)
        vals.append(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum())
    return np.array(vals)


def null_p95(kind, X, y, rng, n_perm):
    """95th percentile of one-repeat out-of-fold R2 over n_perm permutations of y."""
    Y_perm = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
    return np.percentile(oof_r2(kind, X, Y_perm, repeats=1)[0], 95)


def span_r2(y, B):
    """In-sample R2 of y on the columns of B, with intercept."""
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(y)), B]))
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def project_out(Y, B):
    """Y minus its least-squares fit on the columns of B."""
    Q, _ = np.linalg.qr(B)
    return Y - Q @ (Q.T @ Y)


def rms(c):
    return np.sqrt((c ** 2).mean())


# --------------------------------------------------------------------------- #
# Part 0
# --------------------------------------------------------------------------- #
def distance_correlation(x, y):
    """Szekely's distance correlation of two 1-d samples."""
    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()
    a, b = centred(x), centred(y)
    return np.sqrt(max((a * b).mean(), 0) / np.sqrt((a * a).mean() * (b * b).mean()))


def part_0_mean(M):
    rng = np.random.default_rng(SEED)
    rows = []
    for j in range(K1):
        others = [i for i in range(K1) if i != j]
        for preds in [[i] for i in others] + [others]:
            X, y = M[:, preds], M[:, j]
            row = {"target": AXIS_NAMES[j], "predictors": " + ".join(AXIS_NAMES[i] for i in preds),
                   "Pearson r": np.corrcoef(X[:, 0], y)[0, 1] if len(preds) == 1 else np.nan}
            for kind in ("quadratic", "cubic", "kNN"):
                row[f"{kind} R2 mean"] = oof_r2(kind, X, y)[:, 0].mean()
            row["cubic null p95"] = null_p95("cubic", X, y, rng, N_PERM)
            row["kNN null p95"] = null_p95("kNN", X, y, rng, N_PERM)
            rows.append(row)
    return pd.DataFrame(rows)


def part_0_beyond_mean(M):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(K1):
        for j in range(K1):
            if i == j:
                continue
            x, y = M[:, i], M[:, j]
            row = {"target": AXIS_NAMES[j], "predictor": AXIS_NAMES[i]}
            if i < j:
                ry = y - np.polyval(np.polyfit(x, y, 1), x)
                for label, v in (("", y), (" after removing linear", ry)):
                    d = distance_correlation(x, v)
                    null = np.array([distance_correlation(x, rng.permutation(v)) for _ in range(N_PERM)])
                    row[f"distance corr{label}"] = d
                    row[f"null p95{label}"] = np.percentile(null, 95)
                    row[f"p{label}"] = ((null >= d).sum() + 1) / (N_PERM + 1)
            X = x[:, None]
            fit = np.empty_like(y)
            for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
                fit[test] = knn_fit_predict(X[train], y[train, None], X[test])[:, 0]
            spread = np.abs(y - fit)
            row["spread: R2 of |residual| on predictor (cubic)"] = oof_r2("cubic", X, spread)[:, 0].mean()
            row["spread: null p95"] = null_p95("cubic", X, spread, rng, N_PERM)
            rows.append(row)
    return pd.DataFrame(rows)


def part_0_shape(M, titles):
    n = len(titles)
    r1, r2, r3 = M[:, 0], M[:, 1], M[:, 2]
    d = pd.DataFrame({"bin": pd.qcut(r1, 10, labels=False), "R1": r1, "R2": r2})
    deciles = d.groupby("bin").agg(**{
        "R1 mean": ("R1", "mean"), "R2 mean": ("R2", "mean"),
        "R2 p10": ("R2", lambda v: v.quantile(.1)), "R2 median": ("R2", "median"),
        "R2 p90": ("R2", lambda v: v.quantile(.9)), "R2 sd": ("R2", "std"),
        "share R2 > 0": ("R2", lambda v: (v > 0).mean())}).reset_index()
    corners = []
    q1, q2 = np.quantile(r1, [.25, .75]), np.quantile(r2, [.25, .75])
    for lab1, m1 in (("R1 top quarter", r1 >= q1[1]), ("R1 bottom quarter", r1 <= q1[0])):
        for lab2, m2 in (("R2 top quarter", r2 >= q2[1]), ("R2 bottom quarter", r2 <= q2[0])):
            m = m1 & m2
            far = np.argsort(-(np.abs(r1[m]) + np.abs(r2[m])))[:6]
            corners.append({"cell": f"{lab1} x {lab2}", "n": int(m.sum()),
                            "expected if independent": n / 16, "examples": "; ".join(titles[m][far])})
    c = pd.DataFrame({"R1 tercile": pd.qcut(r1, 3, labels=["R1 low", "R1 mid", "R1 high"]),
                      "R2 tercile": pd.qcut(r2, 3, labels=["R2 low", "R2 mid", "R2 high"]), "R3": r3})
    grid = pd.concat([c.pivot_table(index="R1 tercile", columns="R2 tercile", values="R3",
                                    aggfunc=stat, observed=True).reset_index().assign(statistic=f"R3 {stat}")
                      for stat in ("mean", "std")], ignore_index=True)
    return deciles, pd.DataFrame(corners), grid


def part_0_composites(Z, cols, M):
    idx = {c: j for j, c in enumerate(cols)}
    sets = {"exertion": EXERTION, "skilled physical": SKILLED_PHYSICAL,
            "complex reasoning": COMPLEX_REASONING}
    comp = {name: Z[:, [idx[c] for c in items if c in idx]].mean(axis=1) for name, items in sets.items()}

    def partial(y, x, c):
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        ry = y - np.polyval(np.polyfit(c, y, 1), c)
        return np.corrcoef(rx, ry)[0, 1]

    rows = [{"composite": name, "items": "; ".join(c.split("__")[1] for c in items),
             **{f"r with {a}": np.corrcoef(comp[name], M[:, j])[0, 1] for j, a in enumerate(AXIS_NAMES[:K1])}}
            for name, items in sets.items()]
    E, K, J = comp["exertion"], comp["skilled physical"], comp["complex reasoning"]
    pairs = pd.DataFrame([
        {"pair": "exertion, skilled physical", "correlation": np.corrcoef(E, K)[0, 1],
         "held fixed": "", "partial correlation": np.nan},
        {"pair": "complex reasoning, exertion", "correlation": np.corrcoef(J, E)[0, 1],
         "held fixed": "skilled physical", "partial correlation": partial(J, E, K)},
        {"pair": "complex reasoning, skilled physical", "correlation": np.corrcoef(J, K)[0, 1],
         "held fixed": "exertion", "partial correlation": partial(J, K, E)}])
    return pd.DataFrame(rows), pairs


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #
def section_b(ev, T, S, L, cols):
    rng = np.random.default_rng(SEED)
    first, second = S[:, :K1], S[:, K1:K2]
    top = [cols[int(np.argmax(np.abs(L[:, K1 + j])))].split("__", 1)[1][:40] for j in range(K2 - K1)]
    X_knn = T[:, :K1] * np.sqrt(ev[:K1])
    fits = {"quadratic": oof_r2("quadratic", first, second), "cubic": oof_r2("cubic", first, second),
            "kNN": oof_r2("kNN", X_knn, second)}
    rows = []
    for j in range(K2 - K1):
        row = {"target": f"{AXIS_NAMES[K1 + j]}, top-loading item: {top[j]}"}
        for kind, r2 in fits.items():
            row[f"{kind} R2 mean"] = r2[:, j].mean()
            row[f"{kind} R2 min-max"] = f"{r2[:, j].min():.3f} to {r2[:, j].max():.3f}"
        row["quadratic null p95"] = null_p95("quadratic", first, second[:, j], rng, N_PERM)
        row["kNN null p95"] = null_p95("kNN", X_knn, second[:, j], rng, N_PERM)
        rows.append(row)
    whole = {"target": "PC4-6 as a block"}
    for kind in fits:
        r = block_r2(kind, X_knn if kind == "kNN" else first, T[:, K1:K2])
        whole[f"{kind} R2 mean"] = r.mean()
        whole[f"{kind} R2 min-max"] = f"{r.min():.3f} to {r.max():.3f}"
    rows.append(whole)
    return pd.DataFrame(rows)


def section_g(T, S, L, cols, k_max):
    """Span shares of every square and product of R1-R6, the chance shares,
    and the top-loading item of each axis."""
    n = len(T)
    rows = []
    for i in range(N_LEAD):
        for j in range(i, N_LEAD):
            term = S[:, i] * S[:, j]
            rows.append({"term": f"R{i + 1}^2" if i == j else f"R{i + 1} x R{j + 1}",
                         "in span of PC1-3": span_r2(term, T[:, :3]),
                         "in span of PC4-6": span_r2(term, T[:, 3:6]),
                         f"in span of PC7-{k_max}": span_r2(term, T[:, 6:k_max]),
                         f"in span of PC{k_max + 1}-{T.shape[1]}": span_r2(term, T[:, k_max:])})
    table = pd.DataFrame(rows).sort_values(f"in span of PC7-{k_max}", ascending=False)
    top = [cols[int(np.argmax(np.abs(L[:, j])))].split("__", 1)[1] for j in range(N_LEAD)]
    key = pd.DataFrame({"axis": [f"R{j + 1} ({top[j][:32]})" for j in range(N_LEAD)]}).assign(**{
        "chance, span of 3": 3 / (n - 1), f"chance, PC7-{k_max}": (k_max - 6) / (n - 1),
        f"chance, PC{k_max + 1}-{T.shape[1]}": (T.shape[1] - k_max) / (n - 1)})
    return table, key


def decompose(X):
    """X z-scored, eigenvalues, eigenvectors, unit-variance and raw scores."""
    Z = standardize(X)
    ev, V, T = pca(Z)
    return Z, ev, V, T, T * np.sqrt(ev)


def remove_curvature(Y, T1, T1_raw, method):
    """Y minus its fit on PC1-3 and their cubic terms ("cubic"), or minus the
    linear fit and then an out-of-fold kNN fit on the raw PC1-3 scores ("kNN")."""
    linear = np.column_stack([np.ones(len(Y)), T1])
    if method == "cubic":
        return project_out(Y, np.column_stack(
            [linear, PolynomialFeatures(3, include_bias=False).fit_transform(T1)[:, K1:]]))
    Y = project_out(Y, linear)
    pred = np.empty_like(Y)
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(T1_raw):
        pred[test] = KNeighborsRegressor(REMOVAL_KNN_K).fit(T1_raw[train], Y[train]).predict(T1_raw[test])
    return Y - pred


def analyse(X):
    """For one sample: PC4-6 and its varimax rotation; for each removal
    method, A (PC4-6 with the curvature removed, as covariances with the
    columns) and B (the decomposition of every column with the curvature
    removed)."""
    Z, ev, V, T, raw = decompose(X)
    T1, T1_raw, Y = T[:, :K1], raw[:, :K1], T[:, K1:K2]
    _, R = varimax(V[:, K1:K2] * np.sqrt(ev[K1:K2]))
    out = {"raw block": V[:, K1:K2], "ev": ev, "V": V, "rotation": R, "Y": Y}
    for m in REMOVAL_METHODS:
        Yr = remove_curvature(Y, T1, T1_raw, m)
        W = Z.T @ Yr / len(Z)
        out[f"A {m}"] = np.linalg.qr(W)[0]
        out[f"A-axes {m}"] = W @ R
        out[f"Yr {m}"] = Yr
        E = remove_curvature(Z, T1, T1_raw, m)
        _, s, vt = np.linalg.svd(E - E.mean(axis=0), full_matrices=False)
        out[f"B {m}"] = vt.T
        out[f"B ev {m}"] = s ** 2 / (len(Z) - 1)
    return out


def part_1c_full_sample(X, L, k_max):
    """Returns the full-sample A-axes in the order of R4-R6 and three tables."""
    full = analyse(X)
    ev, V, Y = full["ev"], full["V"], full["Y"]
    _, _, _, T, _ = decompose(X)

    rng = np.random.default_rng(SEED)
    chance = np.mean([1 - (remove_curvature(Y, rng.permutation(T[:, :K1]), None, "cubic") ** 2).sum()
                      / (Y ** 2).sum() for _ in range(N_PERM)])
    removed = pd.DataFrame([{"method": m, "share of PC4-6 variance removed":
                             1 - (full[f"Yr {m}"] ** 2).sum() / (Y ** 2).sum()} for m in REMOVAL_METHODS]
                           + [{"method": "cubic, PC1-3 rows permuted",
                               "share of PC4-6 variance removed": chance}])

    L_var = V[:, K1:K2] * np.sqrt(ev[K1:K2]) @ full["rotation"]
    order = [int(np.argmax([congruence(L[:, K1 + j], L_var[:, i]) for i in range(K2 - K1)]))
             for j in range(K2 - K1)]
    axes = {m: full[f"A-axes {m}"][:, order] for m in REMOVAL_METHODS}

    half_error = np.sqrt(2 / (len(X) / 2))
    spec = pd.DataFrame({"position": range(1, 13), "original eigenvalue (component k+3)": ev[K1:K1 + 12]})
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
            share = (V.T @ sub) ** 2
            rows.append({"space": name, "cosines": ", ".join(f"{v:.3f}" for v in c), "RMS cosine": rms(c),
                         "share in PC4-6": share[K1:K2].sum() / 3,
                         f"share in PC7-{k_max}": share[K2:k_max].sum() / 3,
                         f"share beyond PC{k_max}": share[k_max:].sum() / 3})
        c = subspace_cosines(full[f"A {m}"], full[f"B {m}"][:, :3])
        rows.append({"space": f"A {m} against B 1-3 {m}",
                     "cosines": ", ".join(f"{v:.3f}" for v in c), "RMS cosine": rms(c)})
    return axes, removed, spec, pd.DataFrame(rows)


def part_1c_split_halves(X, full_axes, split_perm):
    n = len(X)
    sub, axis = {}, {}
    single = {m: [] for m in REMOVAL_METHODS}
    names = AXIS_NAMES[K1:K2]
    for perm in split_perm:
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

    subs = pd.DataFrame([{"measure": name, "RMS cosine p05": np.percentile(v[:, 0], 5),
                          "RMS cosine median": np.median(v[:, 0]),
                          "smallest cosine p05": np.percentile(v[:, 1], 5),
                          f"passes (RMS p05 >= {STABLE_THRESH})": np.percentile(v[:, 0], 5) >= STABLE_THRESH}
                         for name, v in ((k, np.array(v)) for k, v in sub.items())])
    rows = [{"measure": f"A-axes {m}", "axis": a, "congruence p05": np.percentile(v, 5),
             "congruence median": np.median(v)} for (m, a), v in axis.items()]
    for m in REMOVAL_METHODS:
        s = np.array(single[m])
        rows += [{"measure": f"B {m}", "axis": f"component {j + 1}", "congruence p05": np.percentile(s[:, j], 5),
                  "congruence median": np.median(s[:, j])} for j in range(s.shape[1])]
    return subs, pd.DataFrame(rows)


def part_1d_axes_after_removal(X, ev, T, S, L):
    _, _, _, _, raw = decompose(X)
    Z = standardize(X)
    L2 = L[:, K1:K2]
    names = AXIS_NAMES[K1:K2]
    terms = {(f"R{i + 1}^2" if i == j else f"R{i + 1} x R{j + 1}"): S[:, i] * S[:, j]
             for i in range(K1) for j in range(i, K1)}
    tc = pd.DataFrame({a: [np.corrcoef(v, S[:, K1 + k])[0, 1] for v in terms.values()]
                       for k, a in enumerate(names)}, index=list(terms))
    total = ev.sum()
    rows = []
    for k, a in enumerate(names):
        best = tc[a].abs().idxmax()
        rows.append({"axis": a, "strongest term": best, "r": tc.at[best, a],
                     "% of variance, original": (L2[:, k] ** 2).sum() / total * 100})
    for m in REMOVAL_METHODS:
        Yr = remove_curvature(T[:, K1:K2], T[:, :K1], raw[:, :K1], m)
        Lc, _ = varimax(Z.T @ Yr / len(Z))
        C = np.abs([[congruence(L2[:, j], Lc[:, i]) for i in range(K2 - K1)] for j in range(K2 - K1)])
        Lc = Lc[:, linear_sum_assignment(-C)[1]]
        for k in range(K2 - K1):
            rows[k][f"congruence after {m} removal"] = abs(congruence(L2[:, k], Lc[:, k]))
            rows[k][f"% of variance after {m} removal"] = (Lc[:, k] ** 2).sum() / total * 100
    return pd.DataFrame(rows), tc.rename_axis("term").reset_index()


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #
def refit_in_fold(Xraw, kind, k_max, seed=SEED):
    """Out-of-fold R2 of PC7 to PC k_max from PC1-6, with the z-scoring and
    PCA fitted on each training fold; per target and for the block."""
    targets = list(range(N_LEAD, k_max))
    n = len(Xraw)
    pred, actual = np.empty((n, len(targets))), np.empty((n, len(targets)))
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=seed).split(Xraw):
        mu, sd = Xraw[train].mean(axis=0), Xraw[train].std(axis=0)
        Ztr, Zte = (Xraw[train] - mu) / sd, (Xraw[test] - mu) / sd
        ev, V, Ttr = pca(Ztr)
        Tte = ((Zte - Ztr.mean(axis=0)) @ V) / np.sqrt(ev)
        pred[test] = fit_predict(kind, Ttr[:, :N_LEAD], Ttr[:, targets], Tte[:, :N_LEAD])
        actual[test] = Tte[:, targets]
    ss_res = ((actual - pred) ** 2).sum(axis=0)
    ss_tot = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot, 1 - ss_res.sum() / ss_tot.sum()


def section_a(ev, T, Xraw, rng, k_max, label="observed", with_block=False):
    targets = list(range(N_LEAD, k_max))
    X_poly = T[:, :N_LEAD]
    X_knn = T[:, :N_LEAD] * np.sqrt(ev[:N_LEAD])
    fits = {"quadratic": oof_r2("quadratic", X_poly, T[:, targets]),
            "cubic": oof_r2("cubic", X_poly, T[:, targets]),
            "kNN": oof_r2("kNN", X_knn, T[:, targets])}
    rows = []
    for j, t in enumerate(targets):
        row = {"data": label, "component": f"PC{t + 1}", "eigenvalue": ev[t]}
        for kind, r2 in fits.items():
            row[f"{kind} R2 mean"] = r2[:, j].mean()
            row[f"{kind} R2 min-max"] = f"{r2[:, j].min():.3f} to {r2[:, j].max():.3f}"
        row["quadratic null p95"] = null_p95("quadratic", X_poly, T[:, t], rng, N_PERM)
        row["kNN null p95"] = null_p95("kNN", X_knn, T[:, t], rng, N_PERM)
        rows.append(row)
    table = pd.DataFrame(rows)
    blk = ({kind: block_r2(kind, X_knn if kind == "kNN" else X_poly, T[:, targets]).mean() for kind in fits}
           if with_block else {})
    refit = {}
    if Xraw is not None:
        for kind in ("quadratic", "kNN"):
            per, whole = refit_in_fold(Xraw, kind, k_max)
            table[f"{kind} R2, PCA refitted in fold"] = per
            refit[kind] = whole
    return table, blk, refit


def outlier_check(T, k_max, drop_share=0.02):
    """Quadratic block R2 of PC7 to PC k_max without the drop_share of
    occupations furthest from the centre in PC1-6."""
    d2 = (T[:, :N_LEAD] ** 2).sum(axis=1)
    keep = d2 <= np.quantile(d2, 1 - drop_share)
    r = block_r2("quadratic", T[keep, :N_LEAD], T[keep][:, N_LEAD:k_max])
    return r.mean(), int((~keep).sum())


def section_e(ev, V, rng, n, k_max):
    """section_a() on synthetic data with the loadings of PC1-6 and
    independent noise, censored from below at each FLOORS quantile."""
    L6 = V[:, :N_LEAD] * np.sqrt(ev[:N_LEAD])
    h = np.clip((L6 ** 2).sum(axis=1), 0, 0.99)
    rows = []
    for q in FLOORS:
        X = rng.standard_normal((n, N_LEAD)) @ L6.T + rng.standard_normal((n, L6.shape[0])) * np.sqrt(1 - h)
        if q > 0:
            X = np.maximum(X, np.quantile(X, q, axis=0))
        evs, _, Ts = pca(standardize(X))
        table, blk, _ = section_a(evs, Ts, None, rng, k_max, label=f"synthetic, floor at p{int(q * 100)}",
                                    with_block=True)
        rows.append(table.assign(**{f"PC7-{k_max} block quadratic R2": blk["quadratic"],
                                    f"PC7-{k_max} block kNN R2": blk["kNN"]}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
def write(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False)


def main():
    try:
        m = pd.read_csv(MASTER, index_col="onet_soc")
    except OSError as err:
        sys.exit(f"[fail] read {MASTER}: {err}")
    cols = [c for c in m.columns if c.startswith(FEATURE_PREFIXES)]
    Xraw = m[cols].to_numpy(float)
    if np.isnan(Xraw).any():
        sys.exit("[fail] missing cells in the feature matrix")
    spectra = load_spectra()
    k_max = int(spectra["k_max"])
    Z = standardize(Xraw)
    ev, V, T = pca(Z)
    S, L = axis_scores(ev, V, T, cols)
    titles = m["title"].to_numpy()
    print(f"[ok] read {MASTER}: {Z.shape[0]} occupations x {Z.shape[1]} columns; k_max {k_max}")
    OUT.mkdir(parents=True, exist_ok=True)

    write("0a_mean_dependence", part_0_mean(S[:, :K1]))
    write("0b_beyond_the_mean", part_0_beyond_mean(S[:, :K1]))
    for name, t in zip(("0c_R2_by_R1_decile", "0c_R1_R2_corners", "0c_R3_on_R1_R2_grid"),
                       part_0_shape(S[:, :K1], titles)):
        write(name, t)
    composites, pairs = part_0_composites(Z, cols, S[:, :K1])
    write("0d_composites", composites)
    write("0d_partial_correlations", pairs)
    print("[ok] part 0: R1-R3")

    write("1a_R4_R6_from_R1_R3", section_b(ev, T, S, L, cols))
    terms, key = section_g(T, S, L, cols, k_max)
    write("1b_R1_R3_terms", terms[~terms["term"].str.contains("R[4-9]")]
          .sort_values("in span of PC4-6", ascending=False))
    full_axes, removed, spectrum, same = part_1c_full_sample(Xraw, L, k_max)
    write("1c_variance_removed", removed)
    write("1c_residual_spectrum", spectrum)
    write("1c_same_space", same)
    subspaces, axes = part_1c_split_halves(Xraw, full_axes, spectra["split_perm"])
    write("1c_split_half_subspaces", subspaces)
    write("1c_split_half_axes", axes)
    after, term_corr = part_1d_axes_after_removal(Xraw, ev, T, S, L)
    write("1d_axes_after_removal", after)
    write("1d_term_correlations", term_corr)
    print("[ok] part 1: R4-R6 against PC1-3")

    rng = np.random.default_rng(SEED)
    a_table, _, a_refit = section_a(ev, T, Xraw, rng, k_max)
    write("2a_harmonics", a_table)
    trimmed, n_dropped = outlier_check(T, k_max)
    band = f"PC7-{k_max}"
    write("2a_block", pd.DataFrame([{
        f"{band} block R2, quadratic, PCA refitted in fold": a_refit["quadratic"],
        f"{band} block R2, kNN, PCA refitted in fold": a_refit["kNN"],
        f"{band} block R2, quadratic, without the {n_dropped} most extreme occupations": trimmed}]))
    write("2a_positive_control", section_e(ev, V, rng, len(Z), k_max))
    write("2b_all_terms", terms)
    write("2b_axes_and_chance", key)
    print("[ok] part 2: PC7 onwards against PC1-6")
    print(f"[ok] wrote {len(list(OUT.glob('*.csv')))} tables to {OUT}/")


if __name__ == "__main__":
    main()