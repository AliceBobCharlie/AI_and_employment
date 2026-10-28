#!/usr/bin/env python3
"""
variants_experiment.py -- run the resampling experiment of
resampling_stability.py on other input matrices, rotate the stable components
and subspace each one finds, and compare the axes with R1-R6.

Input:  output/master_clean.csv, output/occupation_coordinates.csv
Output: output/variants_experiment/summary.csv   one row per variant
        output/variants_experiment/<variant>/    for each entry of VARIANTS:
          eigenvalues.csv ... dimensions.csv, resampled_spectra.npz,
          subspace_stability.png   as written by run_experiment()
          matrix.csv               columns and % of variance per prefix
          axes.csv                 each axis: group, % of variance, matched R
                                   axis and r, split-half and bootstrap congruence
          axes_vs_paper.csv        r of every axis with R1-R6
          canonical.csv            canonical correlations, all axes against R1-R6
          paper_axes_r2.csv        R2 of each of R1-R6 from all axes, and from
                                   the first 6 unrotated components
          column_correlations.csv  r of every column with every axis
          occupation_scores.csv    every occupation's score on every axis

VARIANTS gives each variant's column prefixes and weighting. For each, in
order:

1. matrix() takes the master_clean.csv columns with those prefixes, in the
   order of master_clean.csv, drops any with zero variance and z-scores them.
   weights() gives the column weights: 1; CXP_WEIGHT on the variance of every
   CXP column ("cxp"); the same total variance for every block, by
   1 / sqrt(columns in the block) ("equal blocks"); or 1 / sqrt(the block's
   leading eigenvalue) ("mfa"). block_of() from pca_rotated.py gives each
   column's block.
2. run_experiment() finds k_max, the stable components and the stable
   subspace, and tests the axes of each group, named A1, A2, ...
3. axes() computes the unit-variance component scores on the full sample and
   rotates the same groups with rotate_group().
   match() pairs the axes with R1-R6 by the largest total |r|
   (linear_sum_assignment) and flips each matched axis so its r is positive;
   an unmatched axis is flipped so the sum of its cubed loadings is positive.
4. The tables are written, and a row is added to summary.csv: the dimensions
   found and their % of variance, the axis matched to each of R1-R6 with its
   r, the R2 of each of R1-R6 from all the variant's axes and from its first 6
   unrotated components, and the canonical correlations between
   the variant's axes and R1-R6.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import block_of, rotate_group, standardize            # noqa: E402
from resampling_stability import found_groups, run_experiment          # noqa: E402

CLEAN = Path("output/master_clean.csv")
AXES = Path("output/occupation_coordinates.csv")
OUT = Path("output/variants_experiment")

CXP_WEIGHT = 0.5        # variance weight of every CXP column in "cxp" weighting
N_EIGEN = 60            # eigenvalues tested by the parallel analysis (at most the columns)

IM = ("skills_im__", "abilities_im__", "knowledge_im__", "workact_im__")
LV = ("skills_lv__", "abilities_lv__", "knowledge_lv__", "workact_lv__")
LV_NO_KNOWLEDGE = ("skills_lv__", "abilities_lv__", "workact_lv__")
CX = ("workctx_cx__", "workctx_ctp__")
CXP = ("workctx_cxp__", "workctx_ctp__")
VARIANTS = {   # folder: (description, prefixes, weighting)
    "im_cx": ("IM/CX (the paper's matrix)", IM + CX, None),
    "lv_cx": ("LV/CX", LV + CX, None),
    "lv_cx_no_knowledge": ("LV without Knowledge/CX", LV_NO_KNOWLEDGE + CX, None),
    "im_lv_cx": ("IM+LV/CX", IM + LV + CX, None),
    "im_lv_cx_no_knowledge": ("IM+LV without Knowledge/CX", IM + LV_NO_KNOWLEDGE + CX, None),
    "cxp_only": ("CXP and CTP only", CXP, None),
    "job_half": ("job half (Work Activities, Work Context)", ("workact_im__",) + CX, None),
    "worker_half": ("worker half (Skills, Abilities, Knowledge)", ("skills_im__", "abilities_im__", "knowledge_im__"), None),
    "equal_blocks": ("IM/CX, every block the same total variance", IM + CX, "equal blocks"),
    "mfa": ("IM/CX, every block's leading eigenvalue 1 (MFA)", IM + CX, "mfa"),
    "all_scales": ("IM, LV without Knowledge, CXP, CTP", IM + LV_NO_KNOWLEDGE + CXP, "cxp"),
}


def matrix(clean, prefixes):
    """The z-scored columns of clean with the prefixes, in clean's order, with nonzero variance."""
    df = clean[[c for c in clean.columns if c.startswith(prefixes)]]
    df = df.loc[:, df.std() > 0]
    return standardize(df.to_numpy(float)), list(df.columns)


def weights(Xz, feats, weighting):
    """Column weights of the weighting named in VARIANTS (None: all 1)."""
    if weighting is None:
        return None
    if weighting == "cxp":
        return np.where([c.startswith("workctx_cxp__") for c in feats], np.sqrt(CXP_WEIGHT), 1.0)
    blocks = np.array([block_of(c) for c in feats])
    if weighting == "equal blocks":
        wb = {b: 1 / np.sqrt((blocks == b).sum()) for b in set(blocks)}
    else:
        wb = {b: 1 / np.sqrt(np.linalg.svd(Xz[:, blocks == b], compute_uv=False)[0] ** 2 / (len(Xz) - 1))
              for b in set(blocks)}
    return np.array([wb[b] for b in blocks])


def axis_label(idx):
    return [f"A{i + 1}" for i in idx]


def axes(Xz, w, ev, V, groups):
    """Loadings and scores of every group, rotated by rotate_group() on the
    full sample, and the unit-variance component scores."""
    Zw = standardize(Xz) * (1.0 if w is None else w)
    T = Zw @ V / np.sqrt(ev[: V.shape[1]])
    Ls, Ss = [], []
    for idx in groups:
        L, R = rotate_group(V, ev, idx)
        Ls.append(L)
        Ss.append(T[:, idx] @ R)
    return np.hstack(Ls), np.hstack(Ss), T


def match(L, S, paper):
    """Flip the axes as in step 3; returns L, S, the matched R axis of each
    axis (or "") and the r matrix of the axes with R1-R6."""
    C = np.corrcoef(S.T, paper.to_numpy().T)[: S.shape[1], S.shape[1]:]
    rows, cols = linear_sum_assignment(-np.abs(C))
    matched = [""] * S.shape[1]
    sign = np.where((L ** 3).sum(axis=0) < 0, -1.0, 1.0)
    for i, j in zip(rows, cols):
        matched[i] = paper.columns[j]
        sign[i] = np.sign(C[i, j]) or 1.0
    return L * sign, S * sign, matched, C * sign[:, None]


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


def run_variant(folder, description, prefixes, weighting, clean, paper):
    """Steps 1-4 for one variant; returns its row of summary.csv."""
    out = OUT / folder
    Xz, feats = matrix(clean, prefixes)
    w = weights(Xz, feats, weighting)
    print(f"[ok] {folder}: {Xz.shape[1]} columns")
    res = run_experiment(Xz, feats, out, out / "resampled_spectra.npz", out / "subspace_stability.png",
                         w=w, n_eigen=min(N_EIGEN, Xz.shape[1]), names=axis_label)
    groups = found_groups(res["n_comp"], res["sub_end"])
    s = res["spectra"]
    L, S, T = axes(Xz, w, s["full_ev"], s["full_V"], groups)
    L, S, matched, C = match(L, S, paper)
    names = [n for idx in groups for n in axis_label(idx)]
    rot = res["tables"]["rotated_axes"].set_index("axis")
    r2s = [r2(paper[c].to_numpy(), S) for c in paper.columns]
    r2_pcs6 = [r2(paper[c].to_numpy(), T[:, :6]) for c in paper.columns]
    ev = s["full_ev"]
    wsq = np.ones(len(feats)) if w is None else w ** 2
    prefix = np.array([next(p for p in prefixes if c.startswith(p)) for c in feats])
    canon = canonical(S, paper.to_numpy())

    tables = {
        "matrix": pd.DataFrame({"prefix": list(dict.fromkeys(prefix)),
                                "columns": [int((prefix == p).sum()) for p in dict.fromkeys(prefix)],
                                "% of variance": [100 * wsq[prefix == p].sum() / wsq.sum()
                                                  for p in dict.fromkeys(prefix)]}),
        "axes": pd.DataFrame({
            "axis": names,
            "group": [g for g, idx in zip(("stable components", "stable subspace"), groups) for _ in idx],
            "% of variance": 100 * (L ** 2).sum(axis=0) / s["full_ev"].sum(),
            "matched R axis": matched,
            "r with matched R axis": [C[i, list(paper.columns).index(m)] if m else np.nan
                                      for i, m in enumerate(matched)],
            "split-half p05": rot.loc[names, "split-half p05"].to_numpy(),
            "split-half median": rot.loc[names, "split-half median"].to_numpy(),
            "bootstrap p05": rot.loc[names, "bootstrap p05"].to_numpy()}),
        "axes_vs_paper": pd.DataFrame(C, index=names, columns=paper.columns).reset_index(names="axis"),
        "canonical": pd.DataFrame({"pair": range(1, len(canon) + 1), "canonical r": canon}),
        "paper_axes_r2": pd.DataFrame({"paper axis": paper.columns, "R2 from all axes": r2s,
                                       "R2 from first 6 components": r2_pcs6}),
        "column_correlations": pd.DataFrame(Xz.T @ ((S - S.mean(0)) / S.std(0)) / len(Xz), columns=names)
                               .assign(column=feats, prefix=prefix)[["column", "prefix"] + names],
        "occupation_scores": pd.DataFrame(S, columns=names, index=clean.index)
                             .assign(title=clean["title"].to_numpy())[["title"] + names].reset_index(),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    by_r = {m: (names[i], C[i, list(paper.columns).index(m)]) for i, m in enumerate(matched) if m}
    sub = f"PC{res['n_comp'] + 1}-{res['sub_end']}" if res["sub_end"] > res["n_comp"] else "none"
    print(f"[ok] {folder}: stable components PC1-{res['n_comp']}, stable subspace {sub}")
    return {"variant": folder, "description": description, "columns": Xz.shape[1],
            "k_max": res["k_max"], "stable components": f"PC1-{res['n_comp']}",
            "stable subspace": sub, "axes": len(names),
            "% variance, first k_max": 100 * ev[: res["k_max"]].sum() / ev.sum(),
            "% variance, stable components": 100 * ev[: res["n_comp"]].sum() / ev.sum(),
            "% variance, stable subspace": 100 * ev[res["n_comp"]: res["sub_end"]].sum() / ev.sum(),
            **{f"{c} matched axis": by_r.get(c, ("", np.nan))[0] for c in paper.columns},
            **{f"{c} r": by_r.get(c, ("", np.nan))[1] for c in paper.columns},
            **{f"{c} R2 from all axes": v for c, v in zip(paper.columns, r2s)},
            **{f"{c} R2 from first 6 components": v for c, v in zip(paper.columns, r2_pcs6)},
            **{f"canonical r {i + 1}": v for i, v in enumerate(canon)}}


def main():
    try:
        clean = pd.read_csv(CLEAN, index_col="onet_soc")
        paper = pd.read_csv(AXES, index_col=0).reindex(clean.index)
    except OSError as err:
        sys.exit(f"[fail] read: {err}")
    paper = paper[[c for c in paper.columns if c.startswith("R")]]
    rows = [run_variant(folder, *spec, clean, paper) for folder, spec in VARIANTS.items()]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"[ok] wrote {OUT}/summary.csv and one folder per variant")


if __name__ == "__main__":
    main()