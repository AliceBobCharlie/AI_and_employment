#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_rotated.py -- an interpretable multi-axis PCA over skills + institutional/
economic variables, designed to yield several STABLE, ROTATED (varimax) axes
that each have a clean meaning.

Key design choices (from earlier diagnostics):
  - WAGES are collapsed to 2 columns before the PCA: log(median) [wage level]
    and p90/p10 [wage dispersion]. The nine raw wage columns are collinear and
    were vote-stuffing the first component, making axes vague; 2 columns capture
    the wage information without dominating by count.
  - union, self-employment, separation rates DO enter the PCA (per request),
    even though union has low variance in the US -- we let the data place them.
  - employment -> log; ETE distributions kept; everything z-scored.
  - VARIMAX rotation is applied to the leading components so each axis loads on a
    few variables and is interpretable (raw PCs are variance-optimal, not
    interpretation-optimal).

Procedure: PCA -> scree + bootstrap/split-half stability on UNROTATED components
to choose k -> varimax-rotate the first k -> name axes by rotated loadings ->
block composition. Union/self-employment interpretation is read off the rotated
loadings (a low-variance but orthogonal variable will simply load weakly
everywhere -- 'independent but low discriminating power', which is the correct
US finding).

Reads master_wide.xlsx. Terminal + master_out/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MASTER = Path("output/master_clean.xlsx")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
SEED = 0
N_PC = 8          # components examined for stability
TOP = 12
B = 200
N_SPLIT = 50
STABLE_THRESH = 0.90


# --------------------------------------------------------------------------- #
def load_and_prep():
    """master_clean.xlsx is already imputed and carries the derived wage columns
    (see clean_master.py), so this only selects columns and standardizes."""
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)

    skill = [c for c in df.columns if "__" in c and not c.startswith("ete_")]
    ete = [c for c in df.columns if c.startswith("ete_")]
    # wages enter as a level and a dispersion only: the nine raw wage columns are
    # collinear and would dominate an axis by sheer count.
    econ = [c for c in ["ext_wage_level_log", "ext_wage_disp_p90p10",
                        "ext_employment_log", "ext_union_cov_pct",
                        "ext_self_employed_pct", "ext_sep_exit_rate",
                        "ext_sep_transfer_rate"] if c in df.columns]

    F = df[skill + ete + econ]
    prestige = df["ext_prestige"]          # supplementary, not in the PCA

    feats = list(F.columns)
    Xz = StandardScaler().fit_transform(F.values)
    print(f"matrix {Xz.shape[0]} x {len(feats)}  "
          f"(skills={len(skill)}, ete={len(ete)}, econ/inst={len(econ)})")
    return Xz, feats, title, prestige.values


def block_of(c):
    if c.startswith("ext_"):
        return "econ_inst"
    if c.startswith("ete_"):
        return "ete"
    return c.split("__", 1)[0]


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


# ------------------------- varimax ---------------------------------------- #
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
    Xz, feats, title, prestige = load_and_prep()

    pca = PCA(random_state=SEED).fit(Xz)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), ev[:20], "o-"); a1.set_title("Scree")
    a2.plot(range(1, 21), cum[:20], "o-"); a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative"); fig.tight_layout()
    fig.savefig(OUTDIR / "rotated_scree.png", dpi=130); plt.close(fig)
    print(f"\nPC1={ev[0]:.1%}, PC1-4={cum[3]:.1%}, to90%={int(np.argmax(cum>=.9))+1} PCs")

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

    # where the key institutional variables land (rotated loadings)
    print("\n" + "=" * 66)
    print("INSTITUTIONAL / ECONOMIC VARIABLES on the rotated axes")
    print("(low loadings everywhere = independent but low discriminating power)")
    print("=" * 66)
    ext_feats = [c for c in feats if c.startswith("ext_")]
    with pd.option_context("display.width", 200):
        print(Ld.loc[ext_feats].round(3).to_string())

    # supplementary prestige vs rotated axis scores
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
    print("SUPPLEMENTARY prestige correlation with rotated axes")
    print("=" * 66)
    for j in range(k):
        r = np.corrcoef(prestige, S[:, j])[0, 1]
        print(f"  R{j+1}: r = {r:+.3f}")

    out = pd.DataFrame(S, columns=[f"R{j+1}" for j in range(k)], index=title.index)
    out.insert(0, "title", title.values)
    out.to_csv(OUTDIR / "rotated_axes.csv")
    Ld.to_csv(OUTDIR / "rotated_loadings.csv")
    print(f"\n[ok] wrote rotated_axes.csv, rotated_loadings.csv")


if __name__ == "__main__":
    main()