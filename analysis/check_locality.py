#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_locality.py -- is the structure beyond the six axes local?

Components after the sixth do not reproduce on independent halves of the
occupations. One reading is that they describe features only a few
occupations have (a licence, a clinical setting, a craft): real, but local.
This script tests that reading on the unrotated components of the paper's
matrix (220 feature columns, z-scored). Nothing here uses the rotated axes.

  PART A  the heaviest tail in each subspace. If a subspace holds a small
          group of occupations off to one side, some direction in it has a
          heavy tail (high excess kurtosis); two groups of similar size make a
          flat or two-humped distribution (negative excess kurtosis). k-means
          and the silhouette are poor at finding a small group next to a large
          one, which is why kurtosis is used. For each subspace -- PC1-3,
          PC4-6, PC4-6 with the curvature of PC1-3 removed, PC7-15, PC4-15 and
          PC1-15 -- the scores are whitened and the directions of largest and
          smallest excess kurtosis are searched for (projection pursuit: L-BFGS
          on the unit sphere, from N_STARTS random starts). Searching more
          dimensions finds more extreme values by chance, so each subspace is
          compared with the same search on Gaussian data of the same size and
          dimension. For the heaviest-tailed direction: the share of
          occupations beyond TAIL_SD, the kurtosis after dropping the N_TRIM
          most extreme occupations (a lone outlier collapses, a group does
          not), the occupation counts along the tail (a gap or a second hump
          marks a group, a steady fall a long tail) and the occupations in it.
          Each component PC1-15 is also scored on its own.

  PART B  every direction in PC7-15. Part A finds the single heaviest tail; one
          local direction does not make the whole band local. With the
          quadratic curvature of PC1-6 removed, the band is searched again,
          one direction at a time: the heaviest tail, then the heaviest in
          what is orthogonal to it, and so on until the eight directions span
          the band. Orthogonality is in the original metric, so the
          directions split the band's variance exactly and each one's share
          can be read off. Each rank is compared with the same sequence on
          Gaussian data of the same covariance, with and without the N_TRIM
          most extreme occupations; a direction above both is heavy-tailed.
          Also reported: the share of each direction's variance carried by
          its top tenth of occupations. The whole part is repeated with every
          column replaced by the normal scores of its ranks, which removes
          what a floor or a skewed scale does to a single item, so tails that
          survive come from how occupations are arranged.

  PART C  structure specific to a region. A direction can also be local by
          mattering in one part of the space and not elsewhere, without a
          heavy tail. Occupations are split at random into halves. On one
          half the six components are fitted, their quadratic curvature is
          removed from every column, and the occupations are grouped into k
          regions of the six-component space (k-means). The leading m
          directions of what remains are estimated either once for all
          occupations (global) or separately in each region (regional), and
          scored on the other half by the share of its remainder they capture.
          Regional directions that beat the global ones out of sample are
          structure specific to a region. Region labels shuffled within the
          same sizes show what splitting the data costs on its own.

Run after build_master.py and clean_master.py, from the project root:
    python check_locality.py
Reads output/master_clean.csv. Writes output/check_locality.xlsx and prints
every table.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, rankdata, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

from pca_rotated import MASTER, OUTDIR, SEED, FEATURE_PREFIXES, K_MAX, standardize, banner
from cluster_onet import bimodality

OUT = OUTDIR / "check_locality.xlsx"
N_COMPONENTS = K_MAX    # components above the parallel-analysis null
K_AXES = 6              # the two tiers
N_STARTS = 30           # random starts per kurtosis search
N_NULL = 200            # Gaussian samples for each search's null
TAIL_SD = 3             # a Gaussian puts 0.27% of cases beyond 3 sd
N_TRIM = 5              # extreme occupations dropped in the outlier check
TOP_SHARE = 0.10        # top share of occupations in Part B
SUBSPACES = {"PC1-3 (tier 1)": (0, 3), "PC4-6 (tier 2)": (3, 6),
             "PC4-6 without curvature of PC1-3": (3, 6), f"PC7-{K_MAX}": (6, K_MAX),
             f"PC4-{K_MAX} (all after tier 1)": (3, K_MAX), f"PC1-{K_MAX}": (0, K_MAX)}
BAND = (6, K_MAX)       # Part B: PC7 to the last component above the null
N_SPLITS = 20           # Part C: random halves
REGIONS = (4, 8)        # Part C: numbers of regions
M_DIRS = (1, 2, 4, 8)   # Part C: directions per model


# --------------------------------------------------------------------------- #
# Kurtosis search
# --------------------------------------------------------------------------- #
def scores(Z, k):
    """Unit-variance scores on the first k principal components of Z."""
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ vt.T[:, :k] / (s[:k] / np.sqrt(len(Z)))


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


def remove_curvature(Y, axes, degree):
    """Y minus its least-squares fit on the polynomial terms of axes."""
    basis = PolynomialFeatures(degree, include_bias=True).fit_transform(axes)
    Q, _ = np.linalg.qr(basis)
    return Y - Q @ (Q.T @ Y)


# --------------------------------------------------------------------------- #
# Part A: the heaviest tail in each subspace
# --------------------------------------------------------------------------- #
def tail_report(y):
    """Orient y so its longer tail is positive; share beyond TAIL_SD and the
    kurtosis without the N_TRIM most extreme values on that side."""
    if skew(y) < 0:
        y = -y
    keep = np.argsort(y)[:-N_TRIM]
    return y, (np.abs(y) > TAIL_SD).mean(), kurtosis(y[keep], fisher=True)


def part_a_subspaces(Z, titles, feats):
    banner("PART A -- the heaviest tail in each subspace")
    rng = np.random.default_rng(SEED)
    n = len(Z)
    T = scores(Z, N_COMPONENTS)

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
            Y = remove_curvature(Y, T[:, :3], degree=3)
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
        row = {"subspace": name, "dimensions": d,
               "largest excess kurtosis": k_hi, "Gaussian null p95": np.percentile(null_hi, 95),
               f"share beyond {TAIL_SD} sd": beyond,
               f"kurtosis without {N_TRIM} most extreme": trimmed,
               "Gaussian null p95 (trimmed)": np.percentile(null_trim, 95),
               "smallest excess kurtosis": k_lo, "Gaussian null p05": np.percentile(null_lo, 5),
               "bimodality coefficient (flattest direction)": bimodality(W @ w_lo)}
        row["occupations per 0.5 sd from +1 sd"] = \
            " ".join(str(c) for c in np.histogram(y, bins=np.arange(1, 8.5, 0.5))[0])
        r = np.array([np.corrcoef(y, Z[:, j])[0, 1] for j in range(Z.shape[1])])
        row["columns most correlated with the tail"] = "; ".join(
            f"{feats[j].split('__')[1]} {r[j]:+.2f}" for j in np.argsort(-r)[:5])
        rows.append(row)
        for rank, i in enumerate(np.argsort(-y)[:10], 1):
            tails.append({"subspace": name, "rank": rank, "occupation": titles[i],
                          "score (sd)": y[i]})
    sub = pd.DataFrame(rows)
    tails = pd.DataFrame(tails)

    print("\n  most and least kurtotic direction in each subspace, against the same")
    print(f"  search on Gaussian data ({N_NULL} samples per dimension)")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(sub.iloc[:, :-2].round(3).to_string(index=False))
    print("\n  heaviest-tailed direction: occupations per 0.5 sd from +1 sd, and the")
    print("  columns most correlated with it")
    for _, r in sub.iterrows():
        print(f"    {r['subspace']}: {r['occupations per 0.5 sd from +1 sd']}\n"
              f"      {r['columns most correlated with the tail']}")
    print("\n  occupations furthest along the heaviest-tailed direction:")
    for name, g in tails.groupby("subspace", sort=False):
        print(f"    {name}: " + "; ".join(f"{t[:34]} ({v:.1f})" for t, v
                                           in zip(g["occupation"], g["score (sd)"])))
    return comp, sub, tails


# --------------------------------------------------------------------------- #
# Part B: every direction in PC7 to PC K_MAX
# --------------------------------------------------------------------------- #
def heaviest_direction(Y, rng):
    """Unit vector a, in Y's own coordinates, along which Y @ a has the
    largest excess kurtosis. The search runs on whitened scores and is mapped
    back: whitened scores are Yc V diag(1/s) sqrt(n)."""
    Yc = Y - Y.mean(axis=0)
    u, s, vt = np.linalg.svd(Yc, full_matrices=False)
    (_, w), _ = kurtosis_search(u * np.sqrt(len(Y)), rng)
    a = vt.T @ (w / s)
    return a / np.linalg.norm(a)


def sequential_directions(Y, rng):
    """Orthonormal directions spanning Y's columns, each the heaviest-tailed
    in the orthogonal complement of those before it. Returns the scores on
    each direction, one column per direction, in the order found."""
    Y = Y - Y.mean(axis=0)
    basis = np.eye(Y.shape[1])
    out = []
    while basis.shape[1] > 1:
        a = heaviest_direction(Y @ basis, rng)
        out.append(Y @ basis @ a)
        q, _ = np.linalg.qr(np.column_stack([a, np.eye(len(a))]))
        basis = basis @ q[:, 1:]
    out.append((Y @ basis)[:, 0])
    return np.column_stack(out)


def tail_stats(y):
    """Excess kurtosis, the same without the N_TRIM most extreme values on
    the long side, and the share of the variance in the top TOP_SHARE."""
    y = y - y.mean()
    y = y if skew(y) >= 0 else -y
    keep = np.argsort(y)[:-N_TRIM]
    sq = np.sort(y ** 2)[::-1]
    top = sq[: int(round(TOP_SHARE * len(y)))].sum() / sq.sum()
    return y, kurtosis(y, fisher=True), kurtosis(y[keep], fisher=True), top


def gaussian_like(Y, rng):
    """Gaussian sample with the same size and covariance as Y."""
    Yc = Y - Y.mean(axis=0)
    return rng.multivariate_normal(np.zeros(Y.shape[1]), Yc.T @ Yc / len(Yc), size=len(Y))


def rank_normal(X):
    """Each column replaced by the normal scores of its ranks."""
    n = len(X)
    return np.column_stack([norm.ppf((rankdata(c) - 0.5) / n) for c in X.T])


def part_b_directions(Z, titles, label):
    banner(f"PART B -- every direction in PC7-{K_MAX}, curvature of PC1-6 removed ({label})")
    rng = np.random.default_rng(SEED)
    T = scores(Z, N_COMPONENTS)
    a, z = BAND
    Y = remove_curvature(T[:, a:z], T[:, :K_AXES], degree=2)
    S = sequential_directions(Y, rng)
    null = np.array([[tail_stats(s)[1:] for s in sequential_directions(gaussian_like(Y, rng), rng).T]
                     for _ in range(N_NULL)])
    total = S.var(axis=0).sum()
    rows = []
    for j in range(S.shape[1]):
        y, k, k_trim, top = tail_stats(S[:, j])
        rows.append({"input": label, "rank": j + 1,
                     "share of band variance": y.var() / total,
                     "excess kurtosis": k,
                     "Gaussian p95": np.percentile(null[:, j, 0], 95),
                     f"without {N_TRIM} most extreme": k_trim,
                     "Gaussian p95 (trimmed)": np.percentile(null[:, j, 1], 95),
                     f"variance in top {TOP_SHARE:.0%}": top,
                     "Gaussian median (top)": np.median(null[:, j, 2]),
                     "occupations at the tail": "; ".join(titles[i] for i in np.argsort(-y)[:5])})
    t = pd.DataFrame(rows)
    t["heavy-tailed"] = (t["excess kurtosis"] > t["Gaussian p95"]) & \
        (t[f"without {N_TRIM} most extreme"] > t["Gaussian p95 (trimmed)"])
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(t.drop(columns="occupations at the tail").round(3).to_string(index=False))
    print("\n  occupations at the long end of each direction:")
    for _, r in t.iterrows():
        print(f"    {r['rank']}: {r['occupations at the tail']}")
    print(f"\n  heavy-tailed: {t['heavy-tailed'].sum()} of {len(t)} directions, carrying "
          f"{t.loc[t['heavy-tailed'], 'share of band variance'].sum():.0%} of the band's variance")
    return t


# --------------------------------------------------------------------------- #
# Part C: structure specific to a region
# --------------------------------------------------------------------------- #
def remainder(Za, Zb):
    """Fit six components and their quadratic curvature on half a; return
    what is left of both halves and both halves' six-component scores."""
    mu = Za.mean(axis=0)
    _, _, vt = np.linalg.svd(Za - mu, full_matrices=False)
    V = vt[:K_AXES].T
    Ta, Tb = (Za - mu) @ V, (Zb - mu) @ V
    Ra, Rb = (Za - mu) - Ta @ V.T, (Zb - mu) - Tb @ V.T
    poly = PolynomialFeatures(2, include_bias=True).fit(Ta)
    coef = np.linalg.lstsq(poly.transform(Ta), Ra, rcond=None)[0]
    return Ra - poly.transform(Ta) @ coef, Rb - poly.transform(Tb) @ coef, Ta, Tb


def top_directions(R, m):
    _, _, vt = np.linalg.svd(R - R.mean(axis=0), full_matrices=False)
    return vt[:m].T


def score_models(Ra, Rb, la, lb, k, m):
    """Share of half b's remainder captured by m global directions and by m
    directions per region, both fitted on half a. Each region is centred on
    its own half-a mean in both models, so only the directions differ."""
    Ra, Rb = Ra.copy(), Rb.copy()
    for r in range(k):
        mu = Ra[la == r].mean(axis=0)
        Ra[la == r] -= mu
        Rb[lb == r] -= mu
    total = (Rb ** 2).sum()
    glob = ((Rb @ top_directions(Ra, m)) ** 2).sum() / total
    reg = sum(((Rb[lb == r] @ top_directions(Ra[la == r], m)) ** 2).sum()
              for r in range(k) if (la == r).sum() > m)
    return glob, reg / total


def part_c_regions(Z):
    banner("PART C -- structure specific to a region, scored out of sample")
    rng = np.random.default_rng(SEED)
    n = len(Z)
    rows = []
    for split in range(N_SPLITS):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        Ra, Rb, Ta, Tb = remainder(Z[a], Z[b])
        for k in REGIONS:
            km = KMeans(k, n_init=10, random_state=SEED + split).fit(Ta)
            la, lb = km.labels_, km.predict(Tb)
            sa, sb = rng.permutation(la), rng.permutation(lb)
            for m in M_DIRS:
                glob, reg = score_models(Ra, Rb, la, lb, k, m)
                _, shuffled = score_models(Ra, Rb, sa, sb, k, m)
                rows.append({"split": split, "regions": k, "directions": m, "global": glob,
                             "regional": reg, "regional, labels shuffled": shuffled})
    raw = pd.DataFrame(rows)
    raw["regional wins"] = raw["regional"] > raw["global"]
    t = raw.groupby(["regions", "directions"]).agg(
        {"global": "mean", "regional": "mean", "regional, labels shuffled": "mean",
         "regional wins": "mean"}).reset_index()
    t["regional minus global"] = t["regional"] - t["global"]
    print("  share of the held-out remainder (after six components and their quadratic")
    print(f"  curvature) captured, mean over {N_SPLITS} splits; 'regional wins' is the")
    print("  share of splits in which the regional model captures more")
    print(t.round(3).to_string(index=False))
    return t, raw


# --------------------------------------------------------------------------- #
def main():
    m = pd.read_csv(MASTER, index_col="onet_soc")
    feats = [c for c in m.columns if c.startswith(FEATURE_PREFIXES)]
    X = m[feats].to_numpy(float)
    Z = standardize(X)
    titles = m["title"].tolist()
    print(f"{len(Z)} occupations, {len(feats)} feature columns")

    comp, sub, tails = part_a_subspaces(Z, titles, feats)
    dirs = pd.concat([part_b_directions(Z, titles, "published ratings"),
                      part_b_directions(standardize(rank_normal(X)), titles,
                                        "rank-normal scores")])
    reg, reg_raw = part_c_regions(Z)

    with pd.ExcelWriter(OUT) as writer:
        comp.to_excel(writer, sheet_name="components", index=False)
        sub.to_excel(writer, sheet_name="subspaces", index=False)
        tails.to_excel(writer, sheet_name="tail_occupations", index=False)
        dirs.to_excel(writer, sheet_name=f"directions_7_{K_MAX}", index=False)
        reg.to_excel(writer, sheet_name="regions", index=False)
        reg_raw.to_excel(writer, sheet_name="regions_by_split", index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()