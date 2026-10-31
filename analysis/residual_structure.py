#!/usr/bin/env python3
"""
residual_structure.py -- split the variance of the feature matrix beyond PC6
into rating error, curvature of PC1-6, local signal and the remainder, and
search it for heavy-tailed directions.

Input:  data/prepared/master_clean.csv, output/resampling_stability/resampled_spectra.npz (for k_max), and
        the O*NET text files in the folder given as the first argument
        (default ONET_DIR)
Output: one CSV per table in output/residual_structure/:
        decomposition.csv   one row per segment and rho
        error.csv           error variance per segment and rho
        curvature.csv       R2 per segment and model
        local.csv           HDBSCAN cluster share and baselines, PC7 to PC k_max
        local_clusters.csv  members of the clusters found
        tail_components.csv each of PC1 to PC k_max on its own
        tail_subspaces.csv  heaviest and flattest direction per subspace
        tail_occupations.csv the ten occupations furthest along each heaviest direction

Steps 1 to 4 use the IM and CX columns of master_clean.csv (the CTP columns
are left out); step 5 uses all FEATURE_PREFIXES columns. Both are
z-scored. Each step draws from its own generator seeded with SEED.

Steps 1 to 4 split the principal components into segments by segments():
PC1-3, PC4-6, PC1-6, the band PC7 to PC k_max, the tail after PC k_max, and
all components after PC6. Step 1 covers every segment; steps 2 to 4 the
last three. A segment's variance is the sum of its eigenvalues. Each segment
is described by its raw scores, the z-scored matrix projected onto its
eigenvectors.

1. Error. load() reads, with read_block(), the rows of each O*NET file in
   BLOCKS on its analysis scale and without a category, for the occupations
   of master_clean.csv, and pivots the standard errors (SE) to an occupation
   x column table. sigma is the SE divided by the column's sd, with the
   column's RMS SE where none is published. error_covariance() builds the
   error covariance C for each rho in RHOS: the mean squared sigma on the
   diagonal, rho times the mean product of sigmas between columns of the same
   error group in BLOCKS, and 0 elsewhere. The error variance of a segment
   with eigenvectors V is trace(V' C V).
2. Curvature. block_r2() from check_curvature.py gives the out-of-fold R2 of each
   segment's raw scores from the PC1-6 scores (ridge on quadratic or cubic
   terms, or nearest neighbours), over all the segment's columns together. The
   segment's curvature share is that R2. The same R2 on N_CHECK simulated
   error matrices (simulate_error(), rho 0) is recorded as a check.
3. Local signal, band only. The band scores minus their out-of-fold
   quadratic prediction from step 2 are the residual Rb. cluster_share() runs
   HDBSCAN with MIN_CLUSTER on Rb and gives the share of Rb's variance
   explained by the cluster means (noise points count as 0). The same share is
   computed on N_SIM samples of two baselines: Gaussian data with the
   covariance of Rb; and, for each rho, Gaussian data with the covariance of
   Rb minus the band's error covariance (clipped at zero eigenvalues) plus
   simulated error projected onto the band. The local share is (observed share
   - baseline mean) x total variance of Rb / band variance.
4. decomposition.csv gives, per segment and rho, the shares of the segment's
   variance: error, curvature (quadratic model), local (error baseline, band
   only) and the remainder, 1 minus the others.

5. Heavy tails, tail_search(), on the unit-variance scores of PC1 to PC k_max
   (scores()). For each component: skewness, excess kurtosis, share beyond
   TAIL_SD and bimodality(), against the excess kurtosis of 2000 Gaussian
   samples of the same size. For each subspace in tail_subspaces() (PC1-3;
   PC4-6; PC4-6 without the cubic curvature of PC1-3; the band; the band
   without the quadratic curvature of PC1-6), the scores are whitened
   (whiten()) and kurtosis_search() finds the directions of largest and
   smallest excess kurtosis, the best of N_STARTS starts each, climbed by
   L-BFGS on the unit sphere (climb()). The same search on N_NULL whitened
   Gaussian samples of the same dimension gives the null. tail_report() gives,
   for the heaviest direction, the share beyond TAIL_SD and the kurtosis
   without the N_TRIM most extreme occupations; the occupation counts along
   it, the columns most correlated with it and the ten occupations at its end
   are recorded.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_curvature import block_r2, oof_predictions                   # noqa: E402
from pca_rotated import FEATURE_PREFIXES, MASTER, OUTDIR, SEED          # noqa: E402
from resampling_stability import load_spectra                           # noqa: E402

OUT = OUTDIR / "residual_structure"
N_LEAD = 6              # PC1-6
ONET_DIR = Path("data/raw/onet_31_0")
CLEAN = MASTER
RHOS = [0.0, 0.1, 0.3]  # error correlation within an error group
MODELS = ("quadratic", "cubic", "kNN")
N_SIM = 20              # samples per baseline of the cluster share
MIN_CLUSTER = 5         # smallest HDBSCAN cluster
N_CHECK = 5             # simulated error matrices in the curvature check
N_STARTS = 30           # random starts per kurtosis search
N_NULL = 200            # Gaussian samples per kurtosis null
TAIL_SD = 3             # cut-off for the share in the tail
N_TRIM = 5              # extreme occupations dropped in the outlier check


# Column prefix -> (files, analysis scale, error group)
BLOCKS = {
    "abilities": (["Abilities"], "IM", "analysts: abilities and skills"),
    "skills":    (["Essential Skills", "Transferable Skills"], "IM", "analysts: abilities and skills"),
    "knowledge": (["Knowledge"], "IM", "knowledge questionnaire"),
    "workact":   (["Work Activities"], "IM", "work activities questionnaire"),
    "workctx":   (["Work Context"], "CX", "work context questionnaire"),
}
COLUMNS = {"O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
           "Category": "category", "Standard Error": "se"}


# --------------------------------------------------------------------------- #
# Reading and error
# --------------------------------------------------------------------------- #
def read_block(prefix, socs):
    stems, scale, _ = BLOCKS[prefix]
    parts = []
    for stem in stems:
        path = ONET_DIR / f"{stem}.txt"
        parts.append(pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                                 usecols=lambda c: c in COLUMNS).rename(columns=COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    if "category" not in df.columns:
        df["category"] = np.nan
    df = df[(df["scale"] == scale) & df["category"].isna() & df["soc"].isin(socs)].copy()
    df["se"] = pd.to_numeric(df["se"], errors="coerce")
    df["column"] = prefix + "_" + scale.lower() + "__" + df["element"]
    return df


def load(clean):
    """The IM and CX columns of clean, their error groups, and the SE of every
    cell (NaN where none is published)."""
    rows = pd.concat([read_block(p, set(clean["onet_soc"])) for p in BLOCKS], ignore_index=True)
    rows = rows.drop_duplicates(["soc", "column"])
    cols = [c for c in clean.columns if c.startswith(tuple(f"{p}_{s.lower()}__"
                                                           for p, (_, s, _) in BLOCKS.items()))]
    se = rows.pivot(index="soc", columns="column", values="se").reindex(index=clean["onet_soc"], columns=cols)
    groups = np.array([BLOCKS[c.split("_")[0]][2] for c in cols])
    return cols, groups, se.to_numpy(float)


def error_covariance(S, groups, rho):
    """C in standardised units: diag = mean_i S_ij^2; within-group off-diagonal
    = rho * mean_i S_ij S_il."""
    M = (S.T @ S) / len(S)
    same = groups[:, None] == groups[None, :]
    C = np.where(same, rho * M, 0.0)
    np.fill_diagonal(C, np.diag(M))
    return C


def segments(k_max, p):
    """Label -> component indices. The last three are the segments beyond PC6."""
    return {"PC1-3": range(0, 3), "PC4-6": range(3, N_LEAD), "PC1-6": range(0, N_LEAD),
            f"PC7-{k_max}": range(N_LEAD, k_max), f"PC{k_max + 1}-{p}": range(k_max, p),
            f"PC7-{p}": range(N_LEAD, p)}


def simulate_error(sigma, groups, rho, rng):
    """One matrix of rating error: each cell normal with sd sigma, correlated
    rho within an error group of the same occupation."""
    names = list(dict.fromkeys(groups))
    gidx = np.array([names.index(g) for g in groups])
    shared = rng.standard_normal((len(sigma), len(names)))[:, gidx]
    unique = rng.standard_normal(sigma.shape)
    E = sigma * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * unique)
    return E - E.mean(axis=0)


def cluster_share(Y):
    """Share of the variance of Y explained by the means of its HDBSCAN
    clusters, and the cluster labels (-1 for noise)."""
    Yc = Y - Y.mean(axis=0)
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER, copy=True).fit_predict(Yc)
    M = np.zeros_like(Yc)
    for c in set(labels) - {-1}:
        M[labels == c] = Yc[labels == c].mean(axis=0)
    return (M ** 2).sum() / (Yc ** 2).sum(), labels


def gaussian(cov, n, rng):
    return rng.multivariate_normal(np.zeros(len(cov)), cov, size=n)


def psd(A):
    """A with negative eigenvalues set to zero."""
    w, U = np.linalg.eigh((A + A.T) / 2)
    return (U * np.clip(w, 0, None)) @ U.T


# --------------------------------------------------------------------------- #
# Step 5: heavy tails
# --------------------------------------------------------------------------- #
def scores(Z, k):
    """Unit-variance scores on the first k principal components of Z."""
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ vt.T[:, :k] / (s[:k] / np.sqrt(len(Z)))


def remove_curvature(Y, axes, degree):
    """Y minus its least-squares fit on the polynomial terms of axes."""
    Q, _ = np.linalg.qr(PolynomialFeatures(degree, include_bias=True).fit_transform(axes))
    return Y - Q @ (Q.T @ Y)


def bimodality(v):
    """Sarle's bimodality coefficient of v."""
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def whiten(Y):
    """Centred scores rotated to uncorrelated unit-variance coordinates."""
    Y = Y - Y.mean(axis=0)
    u, _, _ = np.linalg.svd(Y, full_matrices=False)
    return u * np.sqrt(len(Y))


def climb(W, w, sign):
    """Local maximum (sign +1) or minimum (sign -1) of the excess kurtosis of
    W @ w over unit vectors w, from the start w, by L-BFGS on w / |w|."""
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
    """(largest excess kurtosis, its direction), (smallest, its direction) of
    W @ w over unit vectors w, the best of N_STARTS random starts each."""
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


def tail_report(y):
    """y oriented so its longer tail is positive, the share beyond TAIL_SD and
    the kurtosis without the N_TRIM most extreme values on that side."""
    if skew(y) < 0:
        y = -y
    keep = np.argsort(y)[:-N_TRIM]
    return y, (np.abs(y) > TAIL_SD).mean(), kurtosis(y[keep], fisher=True)


def tail_subspaces(T, k_max):
    """Name -> scores of the subspace, in search order."""
    return {"PC1-3": T[:, 0:3], "PC4-6": T[:, 3:6],
            "PC4-6 without curvature of PC1-3": remove_curvature(T[:, 3:6], T[:, :3], degree=3),
            f"PC7-{k_max}": T[:, N_LEAD:k_max],
            f"PC7-{k_max} without curvature of PC1-6": remove_curvature(T[:, N_LEAD:k_max], T[:, :N_LEAD], degree=2)}


def tail_search(Z, titles, feats, k_max):
    rng = np.random.default_rng(SEED)
    n = len(Z)
    T = scores(Z, k_max)
    null_one = np.array([kurtosis(rng.standard_normal(n), fisher=True) for _ in range(2000)])
    comp = pd.DataFrame([{"component": f"PC{j + 1}", "skewness": skew(T[:, j]),
                          "excess kurtosis": kurtosis(T[:, j], fisher=True),
                          f"share beyond {TAIL_SD} sd": (np.abs(T[:, j]) > TAIL_SD).mean(),
                          "bimodality coefficient": bimodality(T[:, j])} for j in range(k_max)])
    comp["Gaussian p2.5"] = np.percentile(null_one, 2.5)
    comp["Gaussian p97.5"] = np.percentile(null_one, 97.5)

    null_cache, rows, tails = {}, [], []
    for name, Y in tail_subspaces(T, k_max).items():
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
        r = np.array([np.corrcoef(y, Z[:, j])[0, 1] for j in range(Z.shape[1])])
        rows.append({"subspace": name, "dimensions": d,
                     "largest excess kurtosis": k_hi, "Gaussian null p95": np.percentile(null_hi, 95),
                     f"share beyond {TAIL_SD} sd": beyond,
                     f"kurtosis without {N_TRIM} most extreme": trimmed,
                     "Gaussian null p95 (trimmed)": np.percentile(null_trim, 95),
                     "smallest excess kurtosis": k_lo, "Gaussian null p05": np.percentile(null_lo, 5),
                     "bimodality coefficient (flattest direction)": bimodality(W @ w_lo),
                     "occupations per 0.5 sd from +1 sd":
                         " ".join(str(c) for c in np.histogram(y, bins=np.arange(1, 8.5, 0.5))[0]),
                     "columns most correlated with the tail":
                         "; ".join(f"{feats[j].split('__')[1]} {r[j]:+.2f}" for j in np.argsort(-r)[:5])})
        tails += [{"subspace": name, "rank": rank, "occupation": titles[i], "score (sd)": y[i]}
                  for rank, i in enumerate(np.argsort(-y)[:10], 1)]
    return comp, pd.DataFrame(rows), pd.DataFrame(tails)


def main():
    global ONET_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onet_dir", nargs="?", default=str(ONET_DIR), help="O*NET text folder")
    ONET_DIR = Path(ap.parse_args().onet_dir)
    if not ONET_DIR.exists():
        sys.exit(f"[fail] {ONET_DIR} not found")
    try:
        clean = pd.read_csv(CLEAN)
    except OSError as err:
        sys.exit(f"[fail] read {CLEAN}: {err}")
    k_max = int(load_spectra()["k_max"])

    cols, groups, SE = load(clean)
    X = clean[cols].to_numpy(float)
    Z = StandardScaler().fit_transform(X)
    n, p = Z.shape
    rms = np.sqrt(np.nanmean(SE ** 2, axis=0))
    sigma = np.where(np.isnan(SE), rms, SE) / X.std(axis=0)
    print(f"[ok] read {n} occupations x {p} IM and CX columns, SE for "
          f"{np.isfinite(SE).mean():.0%} of cells; k_max {k_max}")

    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    ev, V = s ** 2 / n, vt.T
    raw = Z @ V
    T6 = raw[:, :N_LEAD] / np.sqrt(ev[:N_LEAD])
    segs = segments(k_max, p)
    band = f"PC7-{k_max}"
    beyond = list(segs)[3:]
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. error
    covs = {rho: error_covariance(sigma, groups, rho) for rho in RHOS}
    error = pd.DataFrame([{"segment": name, "components": len(idx), "variance": ev[list(idx)].sum(),
                           "rho": rho, "error variance": np.trace(V[:, idx].T @ C @ V[:, idx])}
                          for name, idx in segs.items() for rho, C in covs.items()])
    error["error share"] = error["error variance"] / error["variance"]
    error.to_csv(OUT / "error.csv", index=False)
    print("[ok] error variance per segment")

    # 2. curvature
    sims = [simulate_error(sigma, groups, 0.0, rng) for _ in range(N_CHECK)]
    rows = []
    for name in beyond:
        idx = segs[name]
        idx = list(idx)
        row = {"segment": name, "components": len(idx), "variance": ev[idx].sum()}
        for kind in MODELS:
            X_pred = raw[:, :N_LEAD] if kind == "kNN" else T6
            row[f"R2 {kind}"] = block_r2(kind, X_pred, raw[:, idx]).mean()
        row["R2 quadratic, simulated error (check)"] = np.mean(
            [block_r2("quadratic", T6, E @ V[:, idx], repeats=1)[0] for E in sims])
        rows.append(row)
        print(f"[ok] curvature: {name}")
    curvature = pd.DataFrame(rows)
    curvature.to_csv(OUT / "curvature.csv", index=False)

    # 3. local signal, band only
    b = list(segs[band])
    Rb = raw[:, b] - oof_predictions("quadratic", T6, raw[:, b], SEED)
    obs, labels = cluster_share(Rb)
    Rc = Rb - Rb.mean(axis=0)
    cov_rb = Rc.T @ Rc / n
    total = np.trace(cov_rb)
    found = sorted(set(labels) - {-1})
    rows = [{"baseline": "observed", "rho": np.nan, "share mean": obs, "share p95": np.nan,
             "clusters": len(found), "sizes": " ".join(str((labels == c).sum()) for c in found)}]
    base = np.array([cluster_share(gaussian(cov_rb, n, rng))[0] for _ in range(N_SIM)])
    rows.append({"baseline": "Gaussian, covariance of Rb", "rho": np.nan,
                 "share mean": base.mean(), "share p95": np.percentile(base, 95)})
    local_by_rho = {}
    for rho, C in covs.items():
        Cb = V[:, b].T @ C @ V[:, b]
        reliable = psd(cov_rb - Cb)
        base = np.array([cluster_share(gaussian(reliable, n, rng) + simulate_error(sigma, groups, rho, rng) @ V[:, b])[0]
                         for _ in range(N_SIM)])
        rows.append({"baseline": "reliable Gaussian plus simulated error", "rho": rho,
                     "share mean": base.mean(), "share p95": np.percentile(base, 95)})
        local_by_rho[rho] = (obs - base.mean()) * total / ev[b].sum()
    local = pd.DataFrame(rows).assign(**{"variance of Rb": total, "band variance": ev[b].sum()})
    local.to_csv(OUT / "local.csv", index=False)
    titles = clean["title"].to_numpy()
    pd.DataFrame([{"cluster": c, "occupation": titles[i]} for c in found for i in np.flatnonzero(labels == c)],
                 columns=["cluster", "occupation"]).to_csv(OUT / "local_clusters.csv", index=False)
    print(f"[ok] local signal: {band}")

    # 4. decomposition
    curv = curvature.set_index("segment")["R2 quadratic"]
    rows = []
    for _, e in error.iterrows():
        if e["segment"] not in beyond:
            continue
        row = {"segment": e["segment"], "components": e["components"], "variance": e["variance"],
               "rho": e["rho"], "error share": e["error share"],
               "curvature share": curv[e["segment"]],
               "local share": local_by_rho[e["rho"]] if e["segment"] == band else np.nan}
        row["remainder share"] = 1 - row["error share"] - row["curvature share"] - np.nan_to_num(row["local share"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "decomposition.csv", index=False)
    print("[ok] decomposition")

    # 5: all FEATURE_PREFIXES columns
    feats = [c for c in clean.columns if c.startswith(FEATURE_PREFIXES)]
    Zf = StandardScaler().fit_transform(clean[feats].to_numpy(float))
    comp, sub, tails = tail_search(Zf, list(titles), feats, k_max)
    comp.to_csv(OUT / "tail_components.csv", index=False)
    sub.to_csv(OUT / "tail_subspaces.csv", index=False)
    tails.to_csv(OUT / "tail_occupations.csv", index=False)
    print(f"[ok] heavy tails: {len(sub)} subspaces")
    print(f"[ok] wrote {len(list(OUT.glob('*.csv')))} tables to {OUT}/")


if __name__ == "__main__":
    main()