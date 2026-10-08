#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input_variants.py -- does the structure depend on what goes into the matrix?
(Appendices A and B; the parallel-analysis table of Section 5.5)

The paper's matrix is Importance for Abilities, Skills, Knowledge and Work
Activities and Context for Work Context (IM/CX). Each variant below changes
the input and keeps everything else, on the same 910 occupations:

  IM/CX                the paper's matrix, as a reference row
  LV/CX                Level instead of Importance (Appendix A)
  IM+LV/CX             Importance and Level together (Appendix A)
  IM/CX no Knowledge   the Knowledge block left out
  CX only              Work Context alone
  job half             Work Activities and Work Context
  worker half          Skills, Abilities and Knowledge

Level ratings that O*NET flags Not Relevant are used as published; the flag
is not in the master table (see build_master.py).

For each variant (analyse() below, also used by block_weights.py):
  parallel analysis    Horn's, N_PERM column permutations, 95th percentile;
                       components above the null and their share of variance.
  one by one           split-half congruence of each unrotated component
                       (best match among the first K), 5th percentile; the
                       number of leading components that pass 0.90.
  subspaces            split-half smallest principal-angle cosine of the
                       leading-k subspace, 5th percentile, k = 1..K.
  rotated axes         the paper's tiers imposed (varimax within components
                       1-3 and within 4-6); each half's axes paired one to
                       one with the full-sample axes of the same tier;
                       split-half congruence of each rotated axis, 5th
                       percentile.
  against the paper    for each of the paper's axes R1-R6: the largest |r|
                       with the variant's rotated axes of the same tier, and
                       the R2 with which the variant's first three and first
                       six unrotated components reproduce it. Rotation-free
                       R2 is the fairer measure where a variant's tiers do not
                       line up with the paper's.

Every variant uses its own generator seeded with SEED, so a row does not
depend on which variants ran before it.

Run after pca_rotated.py, from the project root:
    python supplementary/input_variants.py
Reads output/master_clean.csv, output/master.csv (Level columns) and
output/rotated_axes.csv. Writes output/input_variants.xlsx and prints every
sheet.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import (congruence, rotate_tier, subspace_cosines, standardize,  # noqa: E402
                         match_axes, SEED, STABLE_THRESH)

CLEAN = Path("output/master_clean.csv")
RAW = Path("output/master.csv")
AXES = Path("output/rotated_axes.csv")
OUT = Path("output/input_variants.xlsx")

N_SPLIT = 100
N_PERM = 50
K = 8                      # components and subspaces examined
TIERS = [[0, 1, 2], [3, 4, 5]]

IM = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")
LV = ("skills_lv__", "abilities_lv__", "knowledge_lv__", "workact_lv__")
CX = ("workctx_cx__",)
VARIANTS = {
    "IM/CX (the paper's matrix)": IM + CX,
    "LV/CX": LV + CX,
    "IM+LV/CX": IM + LV + CX,
    "IM/CX no Knowledge": ("skills_im__", "abilities_im__", "workact_im__") + CX,
    "CX only": CX,
    "job half (Work Activities, Work Context)": ("workact_im__",) + CX,
    "worker half (Skills, Abilities, Knowledge)": ("skills_im__", "abilities_im__", "knowledge_im__"),
}


# --------------------------------------------------------------------------- #
def decompose(X, w):
    """z-score X, multiply column j by w[j], decompose. Eigenvalues (n-1
    denominator), eigenvectors as columns, unit-variance scores."""
    Z = standardize(X) * w
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    ev = s ** 2 / (len(Z) - 1)
    V = vt.T
    return ev, V, Z @ V / np.sqrt(ev)


def rotated(ev, V, T):
    """The paper's tiers, varimax within each: loadings and scores."""
    Ls, Ss = [], []
    for idx in TIERS:
        L, R = rotate_tier(V, ev, idx)
        Ls.append(L)
        Ss.append(T[:, idx] @ R)
    return np.hstack(Ls), np.hstack(Ss)


def parallel_analysis(X, w, ev, rng):
    Z = standardize(X) * w
    n = len(Z)
    null = np.empty((N_PERM, 40))
    for b in range(N_PERM):
        P = np.column_stack([rng.permutation(c) for c in Z.T])
        P = P - P.mean(axis=0)
        null[b] = np.linalg.svd(P, compute_uv=False)[:40] ** 2 / (n - 1)
    above = ev[:40] > np.percentile(null, 95, axis=0)
    k = int(np.argmin(above)) if not above.all() else 40
    return k, ev[:k].sum() / ev.sum()


def split_halves(X, w, full_L, rng):
    """Per split: smallest subspace cosine for k = 1..K, best-match congruence
    of each single component, and congruence of each rotated axis. Within each
    tier the halves' rotated axes are paired one to one with the full-sample
    axes (match_axes, as in pca_rotated.py), so no half-axis can stand in for
    two full-sample axes."""
    n = len(X)
    min_cos = np.empty((N_SPLIT, K))
    single = np.empty((N_SPLIT, K))
    rot = np.empty((N_SPLIT, full_L.shape[1]))
    for s in range(N_SPLIT):
        perm = rng.permutation(n)
        halves = [decompose(X[h], w) for h in (perm[: n // 2], perm[n // 2:])]
        (e1, V1, _), (e2, V2, _) = halves
        for k in range(1, K + 1):
            min_cos[s, k - 1] = subspace_cosines(V1[:, :k], V2[:, :k]).min()
        single[s] = np.abs(V1[:, :K].T @ V2[:, :K]).max(axis=1)
        matched = []
        for e, V, _ in halves:
            H = np.hstack([rotate_tier(V, e, idx)[0] for idx in TIERS])
            matched.append(np.hstack([match_axes(full_L[:, idx], H[:, idx]) for idx in TIERS]))
        rot[s] = [congruence(matched[0][:, j], matched[1][:, j]) for j in range(full_L.shape[1])]
    return (np.percentile(min_cos, 5, axis=0), np.percentile(single, 5, axis=0),
            np.percentile(rot, 5, axis=0))


def r2(y, B):
    A = np.column_stack([np.ones(len(y)), B])
    Q, _ = np.linalg.qr(A)
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def analyse(name, X, paper, w=None):
    """One row of every table for the matrix X (occupations x columns), with
    optional column weights w applied after z-scoring. `paper` holds the
    paper's six axis scores for the same occupations, in the same order."""
    rng = np.random.default_rng(SEED)
    w = np.ones(X.shape[1]) if w is None else np.asarray(w, float)
    if np.isnan(X).any():
        raise ValueError(f"{name}: {int(np.isnan(X).sum())} missing cells")
    ev, V, T = decompose(X, w)
    L, S = rotated(ev, V, T)
    pa, pa_share = parallel_analysis(X, w, ev, rng)
    curve, single, rot = split_halves(X, w, L, rng)
    one_by_one = next((j for j in range(K) if single[j] < STABLE_THRESH), K)

    names = list(paper.columns)
    G = paper.to_numpy()
    match = {}
    for t, idx in enumerate(TIERS):
        for j in idx:
            match[f"best |r| {names[j]}"] = max(abs(np.corrcoef(G[:, j], S[:, i])[0, 1]) for i in idx)
    summary = {"input": name, "columns": X.shape[1],
               "parallel analysis": pa, "variance in them": pa_share,
               "share of variance, components 1-3": ev[:3].sum() / ev.sum(),
               "share of variance, components 4-6": ev[3:6].sum() / ev.sum(),
               "components reproducing one by one": one_by_one,
               "subspace min cos p05, k=3": curve[2], "subspace min cos p05, k=6": curve[5],
               **{f"rotated p05 axis {j + 1}": v for j, v in enumerate(rot)},
               **match,
               **{f"R2 {n} from 3 PCs": r2(G[:, j], T[:, :3]) for j, n in enumerate(names)},
               **{f"R2 {n} from 6 PCs": r2(G[:, j], T[:, :6]) for j, n in enumerate(names)}}
    curves = {"input": name, **{f"k={k + 1}": curve[k] for k in range(K)}}
    singles = {"input": name, **{f"PC{k + 1}": single[k] for k in range(K)}}
    return summary, curves, singles


def load():
    clean = pd.read_csv(CLEAN, index_col="onet_soc")
    raw = pd.read_csv(RAW, index_col="onet_soc", low_memory=False).reindex(clean.index)
    paper = pd.read_csv(AXES, index_col=0).reindex(clean.index)
    paper = paper[[c for c in paper.columns if c.startswith("R")]]
    return clean, raw, paper


def matrix(clean, raw, prefixes):
    """Importance and Context columns from the clean table, Level from the raw one."""
    parts = []
    for p in prefixes:
        src = raw if "_lv__" in p else clean
        parts.append(src[[c for c in src.columns if c.startswith(p)]])
    return pd.concat(parts, axis=1)


def report(rows, out):
    sheets = {"summary": pd.DataFrame([r[0] for r in rows]),
              "subspace_curves": pd.DataFrame([r[1] for r in rows]),
              "single_components": pd.DataFrame([r[2] for r in rows])}
    out.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80, "display.max_rows", 300,
                           "display.float_format", "{:.3f}".format):
        for name, df in sheets.items():
            print(f"\n=== {name} ({len(df)} rows) ===")
            # the summary is wide: print it transposed, one column per input
            print((df.set_index("input").T if name == "summary" else df).to_string())
    print(f"\n[ok] wrote {out}")


def main():
    clean, raw, paper = load()
    rows = []
    for name, prefixes in VARIANTS.items():
        X = matrix(clean, raw, prefixes)
        print(f"[{name}] {X.shape[1]} columns")
        rows.append(analyse(name, X.to_numpy(float), paper))
    report(rows, OUT)


if __name__ == "__main__":
    main()