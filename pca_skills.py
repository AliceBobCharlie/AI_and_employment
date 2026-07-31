#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_skills.py -- PURE-SKILL PCA with external variables projected on passively.

The axes are built from O*NET skill features ONLY (the external variables do
NOT participate in forming the axes). Then each external variable is projected
onto the skill space -- i.e. correlated with each skill PC -- as a
'supplementary variable'. An external variable that correlates with a skill
axis is status-consistent with skills; one that correlates with none is
orthogonal: it points OUT of the skill space and is its own dimension.

This is the clean test of whether union / wage / prestige live inside the skill
structure or outside it, independent of feature-count effects.

Preprocessing matches pca_mixed.py (wage fill, employment log, median fill,
z-score) but only skill columns enter the PCA.

Outputs: skill scree, skill axis loadings + block composition, and the
supplementary correlation table (external variable x skill PC). Terminal +
master_out/. Also writes skill PC scores to master_out/skill_axes.csv.
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
N_PC = 8
TOP = 12



def load():
    """master_clean.xlsx is already imputed (see clean_master.py). The axes are
    built from the skill columns alone; the external variables come back
    separately so they can be projected on afterwards as supplementary
    variables, taking no part in forming the axes."""
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    onet = [c for c in df.columns if "__" in c]
    ext = [c for c in df.columns
           if c.startswith("ext_") and c not in {"ext_wage_level_log", "ext_wage_disp_p90p10", "ext_employment"}]

    Sz = StandardScaler().fit_transform(df[onet].values)
    E = df[ext]
    print(f"skill matrix {Sz.shape[0]} x {Sz.shape[1]} | external vars {E.shape[1]}")
    return Sz, onet, E, title


def block_of(c):
    return c.split("__", 1)[0]


def main():
    Sz, onet, E, title = load()
    pca = PCA().fit(Sz)
    ev, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    comp = pca.components_[:N_PC]
    scores = pca.transform(Sz)[:, :N_PC]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), ev[:20], "o-"); a1.set_title("Scree (pure skill)")
    a2.plot(range(1, 21), cum[:20], "o-"); a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative"); fig.tight_layout()
    fig.savefig(OUTDIR / "skills_scree.png", dpi=130); plt.close(fig)
    print(f"\nPC1={ev[0]:.1%}, PC1-3={cum[2]:.1%}, to90%={int(np.argmax(cum>=.9))+1} PCs")

    print("\n" + "=" * 70)
    print("PURE-SKILL AXIS LOADINGS")
    print("=" * 70)
    for k in range(N_PC):
        ld = pd.Series(comp[k], index=onet)
        hi = ld.sort_values(ascending=False).head(TOP)
        lo = ld.sort_values().head(TOP)
        print(f"\nPC{k+1} ({ev[k]:.1%})")
        print("  + end:")
        for n, v in hi.items(): print(f"     {v:+.2f}  {n}")
        print("  - end:")
        for n, v in lo.items(): print(f"     {v:+.2f}  {n}")

    tags = sorted({block_of(c) for c in onet})
    tag_of = np.array([block_of(c) for c in onet])
    print("\n" + "=" * 70)
    print("BLOCK COMPOSITION (% of each skill axis)")
    print("=" * 70)
    rows = [{t: (comp[k]**2)[tag_of == t].sum() for t in tags} for k in range(N_PC)]
    cdf = pd.DataFrame(rows, index=[f"PC{k+1}" for k in range(N_PC)])[tags]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + (cdf * 100).round(1).to_string())

    # supplementary projection: external variable x skill PC correlations
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: external variable x SKILL PC correlation")
    print("(correlates with a skill axis = status-consistent; correlates with")
    print(" none = orthogonal, a dimension outside skill space)")
    print("=" * 70)
    hdr = "  ".join(f"PC{k+1:>2d}" for k in range(N_PC))
    print(f"\n{'variable':22s} {hdr}   max|r|")
    Ez = E.copy()
    for c in Ez.columns:
        v = Ez[c]
        Ez[c] = (v - v.mean()) / v.std()
    for c in Ez.columns:
        col = Ez[c].values
        ok = ~np.isnan(col)
        rs = [np.corrcoef(col[ok], scores[ok, k])[0, 1] for k in range(N_PC)]
        maxr = max(abs(r) for r in rs)
        tag = "ORTHOGONAL" if maxr < 0.3 else ("partial" if maxr < 0.5 else "CONSISTENT")
        cells = "  ".join(f"{r:+.2f}" for r in rs)
        print(f"{c:22s} {cells}   {maxr:.2f} {tag}")

    out = pd.DataFrame(scores, columns=[f"PC{k+1}" for k in range(N_PC)],
                       index=title.index)
    out.insert(0, "title", title)
    out.to_csv(OUTDIR / "skills_axes.csv")
    print(f"\n[ok] wrote skills_axes.csv")


if __name__ == "__main__":
    main()