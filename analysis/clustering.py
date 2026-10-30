#!/usr/bin/env python3
"""
clustering.py -- cluster the occupations by k-means, place the SOC major
groups in the space of the leading components, and relate the k = 2
partition to the axes.

Input:  output/master_clean.csv, and output/occupation_coordinates.csv if it
        exists (for step 4)
Output: output/clustering/silhouette.csv         silhouette by k, per matrix
        output/clustering/assignments.csv        the k = 2 labels on the z-scored matrix
        output/clustering/soc_major_groups.csv
        output/clustering/k2_regression.csv      the k = 2 labels on R1-R6

main() reads the FEATURE_PREFIXES columns of MASTER and z-scores them.

1. silhouettes() runs KMeans (n_init 10) for each k in K_RANGE on the
   z-scored matrix and on its first N_PCS principal component scores, and
   records the silhouette.
2. main() saves the k = 2 labels on the z-scored matrix.
3. soc_major_groups() groups occupations by the first two digits of the code.
   On the first three principal component scores it gives the mean distance
   within and between groups (mean_distances()) and the silhouette; it also
   gives the silhouette on the z-scored matrix, and the share of the matrix's
   variance kept by the group means, by the first three components and by as
   many components as there are groups less one.
4. k2_regression() regresses the k = 2 labels (0 or 1) on R1-R6 from AXES by
   least squares with an intercept, and gives each coefficient, its standard
   error and the R2. If AXES does not exist, the step is skipped with a note.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pca_rotated import AXES, FEATURE_PREFIXES, MASTER, OUTDIR, SEED   # noqa: E402

OUT = OUTDIR / "clustering"
AX = ["R1", "R2", "R3", "R4", "R5", "R6"]
K_RANGE = range(2, 11)
N_PCS = 10              # components kept in the reduced matrix


def silhouettes(Z):
    matrices = {"z-scored": Z,
                f"z-scored, first {N_PCS} PCs": PCA(n_components=N_PCS, random_state=SEED).fit_transform(Z)}
    return pd.DataFrame([{"matrix": name, **{f"k = {k}": silhouette_score(
        X, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)) for k in K_RANGE}}
        for name, X in matrices.items()])


def mean_distances(X, groups):
    """Mean Euclidean distance between occupations in the same group and in
    different groups."""
    D = squareform(pdist(X))
    same = groups[:, None] == groups[None, :]
    iu = np.triu_indices(len(X), k=1)
    return D[iu][same[iu]].mean(), D[iu][~same[iu]].mean()


def soc_major_groups(Z, codes):
    pca = PCA(random_state=SEED).fit(Z)
    T3 = pca.transform(Z)[:, :3]
    groups = np.array([c[:2] for c in codes])
    n_groups = len(set(groups))
    within, between = mean_distances(T3, groups)
    means = pd.DataFrame(Z).groupby(groups).transform("mean").to_numpy()
    share = pca.explained_variance_ratio_
    return pd.DataFrame([{
        "SOC major groups": n_groups, "mean distance within": within,
        "mean distance between": between, "within / between": within / between,
        "silhouette, 3 components": silhouette_score(T3, groups),
        "silhouette, all features": silhouette_score(Z, groups),
        "feature variance reconstructed, 3 components": share[:3].sum(),
        "feature variance reconstructed, major groups": (means ** 2).sum() / (Z ** 2).sum(),
        f"feature variance reconstructed, {n_groups - 1} components": share[:n_groups - 1].sum()}])


def k2_regression(S, k2):
    """Least squares of k2 on the columns of S with an intercept."""
    X = np.column_stack([np.ones(len(S)), S.to_numpy()])
    y = np.asarray(k2, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return pd.DataFrame({"term": ["intercept"] + list(S.columns), "coefficient": beta,
                         "standard error": se, "R2": r2, "occupations": len(y)})


def main():
    try:
        m = pd.read_csv(MASTER, index_col="onet_soc")
    except OSError as err:
        sys.exit(f"[fail] read {MASTER}: {err}")
    feats = [c for c in m.columns if c.startswith(FEATURE_PREFIXES)]
    Z = StandardScaler().fit_transform(m[feats].to_numpy(float))
    print(f"[ok] read {MASTER}: {len(Z)} occupations x {len(feats)} features")
    OUT.mkdir(parents=True, exist_ok=True)

    silhouettes(Z).to_csv(OUT / "silhouette.csv", index=False)
    print("[ok] silhouettes")

    k2 = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Z)
    pd.DataFrame({"title": m["title"], "cluster_k2": k2}, index=m.index).to_csv(OUT / "assignments.csv")
    soc_major_groups(Z, list(m.index)).to_csv(OUT / "soc_major_groups.csv", index=False)
    print("[ok] k = 2 labels, SOC major groups")

    if AXES.exists():
        S = pd.read_csv(AXES, index_col=0).reindex(m.index)[AX]
        k2_regression(S, k2).to_csv(OUT / "k2_regression.csv", index=False)
        print("[ok] k = 2 labels regressed on R1-R6")
    else:
        print(f"[note] {AXES} not found: k2_regression.csv not written; run pca_rotated.py, then this again")
    print(f"[ok] wrote {len(list(OUT.glob('*.csv')))} tables to {OUT}/")


if __name__ == "__main__":
    main()