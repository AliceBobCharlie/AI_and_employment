#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_model_halves.py -- do the job and the worker descriptors of O*NET's
Content Model arrive at the same axes on their own? (Section 5.4)

The Content Model divides descriptors into those of the job and those of the
worker. The job half is Work Activities and Work Context (96 columns); the
worker half is Skills, Abilities and Knowledge (120). The halves share no
column, so agreement between them is agreement between two disjoint
descriptions of the same 910 occupations.

  1 stability    For each half on its own: the split-half stability of its
                 leading-k subspaces, k = 1..K_HALF (smallest principal-angle
                 cosine, 5th percentile over N_SPLIT splits). A half's
                 reproducing components are the leading k with p05 >= 0.90.
  2 recovery     R2 with which each half's reproducing components (unrotated,
                 so no rotation choice enters) reproduce each axis R1-R6 of the
                 full matrix. In-sample OLS, at most K_HALF regressors on 910
                 occupations.
  3 canonical    Canonical correlations between the two halves' leading
                 components: the leading three of each, and each half's
                 reproducing set.
  4 half axes    Each half's reproducing components varimax-rotated, the pole
                 with the larger extreme loading made positive; their top
                 items, their correlations with R1-R6, and the correlations
                 between the job-half and the worker-half axes.

Run after pca_rotated.py. Reads output/master_clean.csv and
output/rotated_axes.csv. Writes output/content_model_halves.xlsx and prints
every sheet.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from pca_rotated import eigen, standardize, rotate_tier, STABLE_THRESH, SEED

CLEAN = Path("output/master_clean.csv")
AXES = Path("output/rotated_axes.csv")
OUT = Path("output/content_model_halves.xlsx")

N_SPLIT = 100
K_HALF = 8
HALVES = {"job": ("workact_im__", "workctx_cx__"),
          "worker": ("skills_im__", "abilities_im__", "knowledge_im__")}


def scores(X):
    """Eigenvalues, eigenvectors and unit-variance component scores."""
    ev, V = eigen(X)
    return ev, V, standardize(X) @ V / np.sqrt(ev)


def min_cos_curve(X, rng):
    out = np.empty((N_SPLIT, K_HALF))
    for s in range(N_SPLIT):
        perm = rng.permutation(len(X))
        _, V1 = eigen(X[perm[: len(X) // 2]])
        _, V2 = eigen(X[perm[len(X) // 2:]])
        for k in range(1, K_HALF + 1):
            out[s, k - 1] = np.linalg.svd(V1[:, :k].T @ V2[:, :k], compute_uv=False).min()
    return np.percentile(out, 5, axis=0)


def r2(y, B):
    A = np.column_stack([np.ones(len(y)), B])
    Q, _ = np.linalg.qr(A)
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def canonical(A, B):
    Qa, _ = np.linalg.qr(A - A.mean(axis=0))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0))
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def main():
    rng = np.random.default_rng(SEED)
    clean = pd.read_csv(CLEAN, index_col="onet_soc")
    axes = pd.read_csv(AXES, index_col=0).reindex(clean.index)
    names = [c for c in axes.columns if c.startswith("R")]
    G = axes[names].to_numpy()
    print(f"[data] {len(clean)} occupations; full-matrix axes {', '.join(names)}")

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

        L, R = rotate_tier(V, ev, list(range(k)))
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

    (kj, Tj), (kw, Tw) = comp["job"], comp["worker"]
    cca = [{"job components": a, "worker components": b,
            **{f"canonical r {i + 1}": c for i, c in enumerate(canonical(Tj[:, :a], Tw[:, :b]))}}
           for a, b in ((3, 3), (kj, kw))]
    J, W = half_axes["job"], half_axes["worker"]
    cross = pd.DataFrame(np.corrcoef(J.T, W.T)[: J.shape[1], J.shape[1]:],
                         index=[f"J{i + 1}" for i in range(J.shape[1])],
                         columns=[f"W{i + 1}" for i in range(W.shape[1])])

    sheets = {"stability": pd.DataFrame(stab), "recover_axes": pd.DataFrame(recover),
              "canonical": pd.DataFrame(cca), "half_axes": pd.DataFrame(desc),
              "job_x_worker_axes": cross.reset_index().rename(columns={"index": "job axis"})}
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 140, "display.float_format", "{:.3f}".format):
        for name, df in sheets.items():
            print(f"\n=== {name} ({len(df)} rows) ===")
            print(df.to_string(index=False))
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
