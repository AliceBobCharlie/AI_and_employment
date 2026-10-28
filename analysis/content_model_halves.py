#!/usr/bin/env python3
"""
content_model_halves.py -- decompose the job half and the worker half of the
feature matrix separately and compare their components and axes with each
other and with R1-R6.

Input:  output/master_clean.csv, output/occupation_coordinates.csv
Output: output/content_model_halves/stability.csv, recover_axes.csv,
        canonical.csv, half_axes.csv, job_x_worker_axes.csv

HALVES gives each half's column prefixes. For each half, in the order job,
worker, with one generator seeded with SEED:

1. min_cos_curve() decomposes both halves of N_SPLIT random splits of the
   occupations and takes pair_stats() from resampling_stability.py: the p05 of
   the smallest cosine of the leading-k subspaces, k = 1..K_HALF. k is the
   largest k with p05 >= STABLE_THRESH (stability.csv).
2. scores() decomposes the half on all occupations. r2() regresses each of
   R1-R6 on its first k unrotated scores (recover_axes.csv).
3. rotate_group() rotates its first k components; each axis is signed so its
   larger pole is positive. half_axes.csv lists each axis's share of variance,
   top five positive and top three negative loadings, and r with R1-R6.

canonical() gives the canonical correlations between the two halves' first
three scores and between their first k (canonical.csv). job_x_worker_axes.csv
holds the correlations between the job and the worker axes.
"""

import sys

import numpy as np
import pandas as pd
from pathlib import Path

from pca_rotated import SEED, eigen, rotate_group, standardize
from resampling_stability import STABLE_THRESH, pair_stats

CLEAN = Path("output/master_clean.csv")
AXES = Path("output/occupation_coordinates.csv")
OUT = Path("output/content_model_halves")

N_SPLIT = 200
K_HALF = 8
HALVES = {"job": ("workact_im__", "workctx_cx__", "workctx_ctp__"),
          "worker": ("skills_im__", "abilities_im__", "knowledge_im__")}


def scores(X):
    """Eigenvalues, eigenvectors and unit-variance component scores."""
    ev, V = eigen(X)
    return ev, V, standardize(X) @ V / np.sqrt(ev)


def min_cos_curve(X, rng):
    """p05 over N_SPLIT splits of the smallest cosine of the leading-k subspaces."""
    out = np.empty((N_SPLIT, K_HALF))
    for s in range(N_SPLIT):
        perm = rng.permutation(len(X))
        _, V1 = eigen(X[perm[: len(X) // 2]])
        _, V2 = eigen(X[perm[len(X) // 2:]])
        out[s] = pair_stats(V1, V2, K_HALF)[0]
    return np.percentile(out, 5, axis=0)


def r2(y, B):
    """In-sample R2 of y on the columns of B, with an intercept."""
    A = np.column_stack([np.ones(len(y)), B])
    Q, _ = np.linalg.qr(A)
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def canonical(A, B):
    """Canonical correlations between the columns of A and of B."""
    Qa, _ = np.linalg.qr(A - A.mean(axis=0))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0))
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def main():
    rng = np.random.default_rng(SEED)
    try:
        clean = pd.read_csv(CLEAN, index_col="onet_soc")
        axes = pd.read_csv(AXES, index_col=0).reindex(clean.index)
    except OSError as err:
        sys.exit(f"[fail] read: {err}")
    names = [c for c in axes.columns if c.startswith("R")]
    G = axes[names].to_numpy()
    print(f"[ok] read {CLEAN}: {len(clean)} occupations; {AXES}: {', '.join(names)}")

    stab, recover, desc, comp, half_axes = [], [], [], {}, {}
    for label, prefixes in HALVES.items():
        cols = [c for c in clean.columns if c.startswith(prefixes)]
        X = clean[cols].to_numpy(float)
        curve = min_cos_curve(X, rng)
        k = max([j + 1 for j in range(K_HALF) if curve[j] >= STABLE_THRESH], default=1)
        ev, V, T = scores(X)
        comp[label] = (k, T)
        stab.append({"half": label, "columns": len(cols), "reproducing leading k": k,
                     "variance in them": ev[:k].sum() / len(cols),
                     **{f"k={j + 1}": curve[j] for j in range(K_HALF)}})
        recover.append({"half": label, "components": k,
                        **{n: r2(G[:, i], T[:, :k]) for i, n in enumerate(names)}})

        L, R = rotate_group(V, ev, list(range(k)))
        S = T[:, :k] @ R
        sign = np.where(np.abs(L.min(axis=0)) > L.max(axis=0), -1.0, 1.0)
        L, S = L * sign, S * sign
        half_axes[label] = S
        tag = label[0].upper()
        for j in range(k):
            o = np.argsort(L[:, j])
            desc.append({"half axis": f"{tag}{j + 1}",
                         "share of half's variance": (L[:, j] ** 2).sum() / len(cols),
                         "+ items": "; ".join(f"{cols[i].split('__')[1]} {L[i, j]:.2f}" for i in o[::-1][:5]),
                         "- items": "; ".join(f"{cols[i].split('__')[1]} {L[i, j]:.2f}" for i in o[:3]),
                         **{f"corr {n}": np.corrcoef(G[:, i], S[:, j])[0, 1] for i, n in enumerate(names)}})
        print(f"[ok] {label} half: {len(cols)} columns, {k} reproducing components")

    (kj, Tj), (kw, Tw) = comp["job"], comp["worker"]
    cca = [{"job components": a, "worker components": b,
            **{f"canonical r {i + 1}": c for i, c in enumerate(canonical(Tj[:, :a], Tw[:, :b]))}}
           for a, b in ((3, 3), (kj, kw))]
    J, W = half_axes["job"], half_axes["worker"]
    cross = pd.DataFrame(np.corrcoef(J.T, W.T)[: J.shape[1], J.shape[1]:],
                         index=[f"J{i + 1}" for i in range(J.shape[1])],
                         columns=[f"W{i + 1}" for i in range(W.shape[1])])

    tables = {"stability": pd.DataFrame(stab), "recover_axes": pd.DataFrame(recover),
              "canonical": pd.DataFrame(cca), "half_axes": pd.DataFrame(desc),
              "job_x_worker_axes": cross.reset_index().rename(columns={"index": "job axis"})}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"[ok] wrote {len(tables)} tables to {OUT}/")


if __name__ == "__main__":
    main()