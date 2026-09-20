#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_ete.py -- internal structure of the Education/Training/Experience (ETE)
block: does the 'preparation path' have its own interpretable axes, distinct
from the skill axes?

ETE is percent-distribution data across four scales (each scale's categories
summing to ~100, with the highest category already dropped in the build):
  RL = required level of education, RW = related work experience,
  PT = on-site/in-plant training, OJ = on-the-job training.

Procedure: PCA on the 37 ETE columns (z-scored) -> scree + bootstrap/split-half
stability -> varimax-rotate the stable axes -> name them by rotated loadings ->
then relate the ETE axes to the three skill axes (from skill_axes.csv if present,
else recomputed here) to see whether ETE structure is independent of skills.

Reads master_wide.xlsx (and master_out/skill_axes.csv if available).
Terminal + master_out/.
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
N_PC = 6
TOP = 12
B = 200
N_SPLIT = 50
STABLE_THRESH = 0.90


# varimax and congruence are defined once, in pca_rotated.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_rotated import congruence, varimax as _varimax


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    L, _ = _varimax(Phi, gamma=gamma, q=q, tol=tol)
    return L


def best_match(ref, cand):
    return np.array([max(congruence(ref[i], cand[j]) for j in range(cand.shape[0]))
                     for i in range(ref.shape[0])])


def pca_load(M, k):
    return PCA(n_components=k, random_state=SEED).fit(M).components_


def ete_scale(col):
    # ete_rl__... -> RL
    return col.split("__", 1)[0].replace("ete_", "").upper()


def main():
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    ete_cols = [c for c in df.columns if c.startswith("ete_")]
    E = df[ete_cols]            # already imputed in clean_master.py
    Ez = StandardScaler().fit_transform(E.values)
    print(f"ETE matrix: {Ez.shape[0]} occ x {Ez.shape[1]} cols")
    by_scale = {}
    for c in ete_cols:
        by_scale.setdefault(ete_scale(c), 0)
        by_scale[ete_scale(c)] += 1
    print("  columns by scale:", by_scale)

    pca = PCA(random_state=SEED).fit(Ez)
    ev = pca.explained_variance_ratio_; cum = np.cumsum(ev)
    print(f"\nPC1={ev[0]:.1%}, PC1-3={cum[2]:.1%}, PC1-6={cum[5]:.1%}")

    # stability
    rng = np.random.default_rng(SEED); n = Ez.shape[0]
    ref = pca.components_[:N_PC]
    boot = np.full((B, N_PC), np.nan); sh = np.full((N_SPLIT, N_PC), np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        try: boot[b] = best_match(ref, pca_load(Ez[idx], N_PC))
        except Exception: pass
    for s in range(N_SPLIT):
        perm = rng.permutation(n); h1, h2 = perm[:n//2], perm[n//2:]
        try: sh[s] = best_match(pca_load(Ez[h1], N_PC), pca_load(Ez[h2], N_PC))
        except Exception: pass
    print("\n" + "=" * 60)
    print(f"ETE AXIS STABILITY (boot B={B}, split x{N_SPLIT})")
    print("=" * 60)
    k = 0
    for j in range(N_PC):
        bp = np.nanpercentile(boot[:, j], 5); sp = np.nanpercentile(sh[:, j], 5)
        ok = bp >= STABLE_THRESH and sp >= STABLE_THRESH
        if ok: k = j + 1
        print(f"PC{j+1}  boot_p05={bp:.3f}  split_p05={sp:.3f}  "
              f"{'STABLE' if ok else 'unstable'}")
    print(f"-> {k} stable ETE axes")
    k = max(k, 2)

    # varimax rotate
    L = (pca.components_[:k].T * np.sqrt(pca.explained_variance_[:k]))
    Lr = varimax(L)
    ssq = (Lr ** 2).sum(axis=0); order = np.argsort(-ssq)
    Lr = Lr[:, order]
    Ld = pd.DataFrame(Lr, index=ete_cols, columns=[f"E{j+1}" for j in range(k)])

    print("\n" + "=" * 60)
    print(f"VARIMAX-ROTATED ETE AXES (k={k})")
    print("=" * 60)
    for j in range(k):
        s = Ld[f"E{j+1}"]
        hi = s.sort_values(ascending=False).head(TOP)
        lo = s.sort_values().head(TOP)
        print(f"\nETE axis E{j+1}")
        print("  + end:")
        for nm, v in hi.items():
            if v > 0.2: print(f"     {v:+.2f}  {nm}")
        print("  - end:")
        for nm, v in lo.items():
            if v < -0.2: print(f"     {v:+.2f}  {nm}")

    # composition by ETE scale
    print("\n" + "=" * 60)
    print("ETE axis composition by scale (% of axis SS)")
    print("=" * 60)
    tag = np.array([ete_scale(c) for c in ete_cols])
    rows = []
    for j in range(k):
        sq = Lr[:, j] ** 2
        rows.append({t: sq[tag == t].sum() / sq.sum() for t in sorted(set(tag))})
    print("\n" + (pd.DataFrame(rows, index=[f"E{j+1}" for j in range(k)]) * 100)
          .round(1).to_string())

    # relate ETE axes to the skill axes
    ete_scores = Ez @ Lr
    skill_path = OUTDIR / "skills_axes.csv"
    print("\n" + "=" * 60)
    print("ETE axes vs SKILL axes (correlation)")
    print("(low corr = ETE preparation structure independent of skills)")
    print("=" * 60)
    if skill_path.exists():
        sk = pd.read_csv(skill_path, index_col=0)
        sk = sk.reindex(df.index)
        skcols = [c for c in sk.columns if c.startswith("PC")]
        for j in range(k):
            rs = []
            for sc in skcols[:3]:
                v = sk[sc].values
                ok = ~np.isnan(v)
                rs.append(np.corrcoef(ete_scores[ok, j], v[ok])[0, 1])
            cells = "  ".join(f"{sc}={r:+.2f}" for sc, r in zip(skcols[:3], rs))
            print(f"  ETE E{j+1}: {cells}  (max|r|={max(abs(r) for r in rs):.2f})")
    else:
        print("  (skill_axes.csv not found; run pca_skills.py first to compare)")

    out = pd.DataFrame(ete_scores, columns=[f"E{j+1}" for j in range(k)],
                       index=title.index)
    out.insert(0, "title", title.values)
    out.to_csv(OUTDIR / "ete_axes.csv")
    print(f"\n[ok] wrote ete_axes.csv")


if __name__ == "__main__":
    main()