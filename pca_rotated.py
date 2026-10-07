#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_rotated.py -- an interpretable multi-axis PCA over the O*NET descriptor
blocks, designed to yield several STABLE, ROTATED (varimax) axes that each have
a clean meaning.

Key design choices (from earlier diagnostics):
  - ONLY the O*NET feature blocks enter the decomposition: Abilities, Skills,
    Knowledge and Work Activities on the Importance scale, plus Work Context on
    the Context scale. 216 columns.
  - The labour-market variables (wages, employment, union coverage, self-
    employment, separation rates, prestige) and the education/training/
    experience distributions are HELD OUT. They are correlated against the
    finished axes as supplementary variables. This is what licenses the
    comparisons later: an axis cannot be said to predict wages if wages helped
    to build it. Wages appear in that comparison as a level (log median) and a
    dispersion (p90/p10) rather than as nine collinear percentiles.
  - everything z-scored.
  - VARIMAX rotation is applied to the leading components so each axis loads on a
    few variables and is interpretable (raw PCs are variance-optimal, not
    interpretation-optimal).

Procedure: PCA -> scree + bootstrap/split-half stability on UNROTATED components
to choose k -> varimax-rotate the first k -> name axes by rotated loadings ->
block composition -> supplementary variables correlated against the axis scores.

A supplementary variable that correlates weakly with every axis is independent
of the content of the work; that is a statement about independence, not about
importance, and Section 5.5 of the paper makes the distinction with a ridge
regression instead.

Reads output/master_clean.csv. Terminal + output/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MASTER = Path("output/master_clean.csv")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
SEED = 0
N_PC = 8          # components examined for stability
TOP = 12
B = 200
N_SPLIT = 50
STABLE_THRESH = 0.90
N_PERM = 50       # permutations for parallel analysis

# The feature matrix: Importance for four blocks, the Context scale for Work Context.
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__")


# --------------------------------------------------------------------------- #
def load_and_prep():
    """The clean table is already imputed and carries the derived wage columns
    (see clean_master.py), so this only selects columns and standardizes.

    ONLY the O*NET descriptor blocks enter the decomposition. Education,
    training and experience and the economic/institutional columns are held out
    and correlated against the axes afterwards, so that nothing the axes are
    later compared with has helped to form them."""
    df = pd.read_csv(MASTER, index_col="onet_soc")
    title = df["title"]
    feats = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]

    # held out of the PCA, used only as supplementary variables below. Wages
    # appear as a level and a dispersion: the nine raw wage columns are
    # collinear and comparing all nine against every axis says nothing extra.
    ete = [c for c in df.columns if c.startswith("ete_")]
    econ = [c for c in ["ext_wage_level_log", "ext_wage_disp_p90p10",
                        "ext_prestige", "ext_sep_exit_rate",
                        "ext_sep_transfer_rate", "ext_union_cov_pct",
                        "ext_self_employed_pct", "ext_employment_log"]
            if c in df.columns]
    supp = df[econ + ete]

    Xz = StandardScaler().fit_transform(df[feats].values)
    print(f"feature matrix {Xz.shape[0]} x {len(feats)}")
    print(f"held out as supplementary: {len(econ)} labour-market columns, "
          f"{len(ete)} ETE columns")
    return Xz, feats, title, supp


def block_of(col):
    """'abilities_im__Oral Comprehension' -> 'abilities_im'."""
    return col.split("__", 1)[0]


# ------------------------- stability (unrotated) -------------------------- #
def congruence(a, b):
    den = np.sqrt((a @ a) * (b @ b))
    return np.abs(a @ b) / den if den > 0 else 0.0


def best_match(ref, cand):
    return np.array([max(congruence(ref[i], cand[j]) for j in range(cand.shape[0]))
                     for i in range(ref.shape[0])])


def pca_load(M, k):
    return PCA(n_components=k, random_state=SEED).fit(M).components_


def stability(X, ref, k):
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    boot = np.full((B, k), np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        try: boot[b] = best_match(ref, pca_load(X[idx], k))
        except Exception: pass
    sh = np.full((N_SPLIT, k), np.nan)
    for s in range(N_SPLIT):
        perm = rng.permutation(n); h1, h2 = perm[:n//2], perm[n//2:]
        try: sh[s] = best_match(pca_load(X[h1], k), pca_load(X[h2], k))
        except Exception: pass
    print("\n" + "=" * 66)
    print(f"STABILITY of UNROTATED components (boot B={B}, split x{N_SPLIT})")
    print("=" * 66)
    print(f"{'':4s} {'boot_p05':>9s} {'split_p05':>9s}  verdict")
    last = 0
    for j in range(k):
        bp = np.nanpercentile(boot[:, j], 5); sp = np.nanpercentile(sh[:, j], 5)
        ok = bp >= STABLE_THRESH and sp >= STABLE_THRESH
        if ok: last = j + 1
        print(f"PC{j+1:<2d} {bp:9.3f} {sp:9.3f}  {'STABLE' if ok else 'unstable'}")
    print(f"\n-> {last} stable axes (split-half criterion)")
    return last


def eigen_table(pca, p):
    ev, share = pca.explained_variance_, pca.explained_variance_ratio_
    cum = np.cumsum(share)
    print("\n" + "=" * 66)
    print("EIGENVALUES")
    print("=" * 66)
    print(f"{'':4s} {'eigenvalue':>10s} {'% var':>7s} {'cum %':>7s}")
    for j in range(10):
        print(f"PC{j+1:<2d} {ev[j]:10.2f} {share[j]*100:7.1f} {cum[j]*100:7.1f}")
    print(f"components to reach 90% of variance: {int(np.argmax(cum >= .9)) + 1}")


def parallel_analysis(Xz, ev):
    """Horn's parallel analysis: each column permuted independently, which keeps
    every column's distribution and destroys only the relations between them."""
    rng = np.random.default_rng(SEED)
    n, p = Xz.shape
    null = np.empty((N_PERM, 30))
    for b in range(N_PERM):
        Xp = rng.permuted(Xz, axis=0)
        Xp = Xp - Xp.mean(axis=0)
        null[b] = np.linalg.svd(Xp, compute_uv=False)[:30] ** 2 / (n - 1)
    q95 = np.percentile(null, 95, axis=0)
    above = ev[:30] > q95
    k = int(np.argmin(above)) if not above.all() else 30
    print("\n" + "=" * 66)
    print(f"PARALLEL ANALYSIS ({N_PERM} column-wise permutations, 95th percentile)")
    print("=" * 66)
    print(f"largest null eigenvalue {q95[0]:.2f}; component {k}: real {ev[k-1]:.2f} "
          f"vs null {q95[k-1]:.2f}; component {k+1}: real {ev[k]:.2f} vs null {q95[k]:.2f}")
    print(f"-> {k} components exceed the null, together {ev[:k].sum() / ev.sum():.1%} of variance")
    return k


# ------------------------- varimax ---------------------------------------- #
def promax(A, m=4):
    """Oblique rotation of an already varimax-rotated loading matrix A.
    Returns the pattern matrix and the factor correlation matrix.
    Follows the standard Hendrickson-White construction."""
    Q = A * np.abs(A) ** (m - 1)          # sharpened target
    U = np.linalg.lstsq(A, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    pattern = A @ U
    Uinv = np.linalg.inv(U)
    Phi = Uinv @ Uinv.T
    dg = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(dg, dg)          # to correlation form
    return pattern, Phi



def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax rotation of a loading matrix Phi (p x k)."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vt
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


# --------------------------------------------------------------------------- #
def main():
    Xz, feats, title, supp = load_and_prep()

    pca = PCA(random_state=SEED).fit(Xz)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), ev[:20], "o-"); a1.set_title("Scree")
    a2.plot(range(1, 21), cum[:20], "o-"); a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative"); fig.tight_layout()
    fig.savefig(OUTDIR / "rotated_scree.png", dpi=130); plt.close(fig)
    print(f"\nPC1={ev[0]:.1%}, PC1-4={cum[3]:.1%}, to90%={int(np.argmax(cum>=.9))+1} PCs")

    eigen_table(pca, Xz.shape[1])
    parallel_analysis(Xz, pca.explained_variance_)

    ref = pca.components_[:N_PC]
    k = stability(Xz, ref, N_PC)
    k = max(k, 2)          # rotate at least 2

    # varimax-rotate the first k components (scaled by sqrt eigenvalue = loadings)
    L = (pca.components_[:k].T * np.sqrt(pca.explained_variance_[:k]))  # p x k
    Lr, Rot = varimax(L)
    # order rotated axes by their explained variance (sum of squared loadings)
    ssq = (Lr ** 2).sum(axis=0)
    order = np.argsort(-ssq)
    Lr = Lr[:, order]; Rot = Rot[:, order]; ssq = ssq[order]

    # --- sign convention -------------------------------------------------- #
    # Eigenvectors are only defined up to sign, and varimax inherits that, so
    # the direction of each axis is arbitrary unless we pin it. Anchor each axis
    # on a marker variable and flip so the marker loads POSITIVE. Markers are
    # chosen so that '+' consistently means 'harder to automate':
    #   axis 1  + = embodied / physical work medium
    #   axis 2  + = high cognitive load
    #   axis 3  + = interpersonal / care
    # This also keeps signs stable if the data is rebuilt.
    ANCHORS = ["abilities_im__Manual Dexterity",
               "skills_im__Complex Problem Solving",
               "workact_im__Assisting and Caring for Others"]
    fidx = {f: i for i, f in enumerate(feats)}
    for j in range(min(k, len(ANCHORS))):
        a = ANCHORS[j]
        if a not in fidx:
            print(f"[warn] sign anchor missing: {a}")
            continue
        v = Lr[fidx[a], j]
        if v < 0:
            Lr[:, j] *= -1
            Rot[:, j] *= -1
        print(f"  axis {j+1} anchored on '{a}' (loading {abs(v):.2f}, "
              f"{'flipped' if v < 0 else 'kept'})")
    var_share = ssq / (Xz.shape[1])     # each variable standardized to var 1

    print("\n" + "=" * 66)
    print(f"VARIMAX-ROTATED AXES (k={k}) -- loadings")
    print("=" * 66)
    Ld = pd.DataFrame(Lr, index=feats, columns=[f"R{j+1}" for j in range(k)])
    for j in range(k):
        s = Ld[f"R{j+1}"]
        hi = s.sort_values(ascending=False).head(TOP)
        lo = s.sort_values().head(TOP)
        print(f"\nRotated axis R{j+1}  (var share {var_share[j]:.1%})")
        print("  + end:")
        for n, v in hi.items():
            if v > 0.15: print(f"     {v:+.2f}  {n}")
        print("  - end:")
        for n, v in lo.items():
            if v < -0.15: print(f"     {v:+.2f}  {n}")

    # block composition of rotated axes
    print("\n" + "=" * 66)
    print("BLOCK COMPOSITION of rotated axes (% of axis SS by block)")
    print("=" * 66)
    tag = np.array([block_of(c) for c in feats])
    rows = []
    for j in range(k):
        sq = Lr[:, j] ** 2
        rows.append({t: sq[tag == t].sum() / sq.sum() for t in sorted(set(tag))})
    comp = pd.DataFrame(rows, index=[f"R{j+1}" for j in range(k)])
    print("\n" + (comp * 100).round(1).to_string())

    # Rotated component scores. NOT Xz @ Lr: loadings carry a sqrt(eigenvalue)
    # scaling, and because the eigenvalues differ, using them as weights makes
    # the axes correlated (an orthogonal rotation should leave them
    # uncorrelated). Standardize the unrotated PC scores first, then apply the
    # same rotation matrix.
    T = pca.transform(Xz)[:, :k] / np.sqrt(pca.explained_variance_[:k])
    S = T @ Rot
    off = np.corrcoef(S.T) - np.eye(k)
    print(f"\n  max |corr| between rotated axes: {np.abs(off).max():.3f} "
          f"(should be ~0)")
    print("\n" + "=" * 66)
    print("SUPPLEMENTARY VARIABLES against the rotated axes")
    print("(none of these helped form the axes; low everywhere = independent")
    print(" of the content of the work, not necessarily unimportant)")
    print("=" * 66)
    names = [f"R{j+1}" for j in range(k)]
    Sdf = pd.DataFrame(S, columns=names, index=title.index)
    corr = pd.DataFrame(
        {n: supp.apply(lambda v: v.corr(Sdf[n])) for n in names})
    with pd.option_context("display.width", 200):
        print(corr.round(2).to_string())

    # ---- robustness: does the data prefer orthogonal axes, or does varimax
    # impose them? Promax starts from this solution and lets the axes tilt.
    Lp, Phi = promax(Lr)
    for j in range(k):
        if np.dot(Lr[:, j], Lp[:, j]) < 0:
            Lp[:, j] *= -1
            Phi[j, :] *= -1; Phi[:, j] *= -1
    np.fill_diagonal(Phi, 1.0)
    print("\n" + "=" * 66)
    print("OBLIQUE ROTATION CHECK (promax): are the axes orthogonal by")
    print("preference of the data, or only by constraint?")
    print("=" * 66)
    names = [f"R{j+1}" for j in range(k)]
    print("\nfactor correlations:")
    print(pd.DataFrame(Phi, index=names, columns=names).round(3).to_string())
    off = max(abs(Phi[i, j]) for i in range(k) for j in range(k) if i < j)
    print(f"\nlargest |correlation| between axes: {off:.3f}")
    print("congruence with the varimax solution:")
    for j in range(k):
        c = congruence(Lr[:, j], Lp[:, j])
        print(f"  R{j+1}: {c:.3f}  {'same axis' if c >= 0.95 else 'DIFFERS - inspect'}")
    if off < 0.20:
        print("\n-> the data places the axes close to orthogonal on its own.")
    elif off < 0.35:
        print("\n-> mild correlation; orthogonality is a simplification, not a")
        print("   distortion. Report these correlations.")
    else:
        print("\n-> substantial correlation; report the oblique solution instead.")
    pd.DataFrame(Lp, index=feats, columns=names).to_csv(
        OUTDIR / "oblique_pattern.csv")

    out = pd.DataFrame(S, columns=[f"R{j+1}" for j in range(k)], index=title.index)
    out.insert(0, "title", title.values)
    out.to_csv(OUTDIR / "rotated_axes.csv")
    Ld.to_csv(OUTDIR / "rotated_loadings.csv")
    print(f"\n[ok] wrote rotated_axes.csv, rotated_loadings.csv, oblique_pattern.csv")


if __name__ == "__main__":
    main()