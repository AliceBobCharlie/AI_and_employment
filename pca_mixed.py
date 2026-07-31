#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_mixed.py -- MIXED PCA over the lean master table: skill features AND the
external social-structural variables (wages, employment, union, prestige)
standardized and put in one PCA, equal weight. With the skill side now pruned
to ~257 columns (single-scale IM, no redundant LV), the question is whether the
external variables -- especially union coverage -- can now surface as their own
axis instead of being drowned by skill-feature count.

Preprocessing (as fixed earlier):
  wages: top-coded highs (p10 present) -> OEWS cap; whole-row-missing (p10 also
         absent) -> column median; ratios recomputed from filled levels.
  employment -> log1p.  union, prestige -> as-is.  residual gaps -> median.
  everything z-scored, then one PCA.

Outputs: scree, per-axis loadings (ext_ marked), block composition with the
'external' share per PC, and each external variable's correlation with each PC.
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

MASTER = Path("master_wide.xlsx")
OUTDIR = Path("master_out"); OUTDIR.mkdir(exist_ok=True)
OEWS_CAP = 239200.0
N_PC = 10
TOP = 12

WAGE_LEVELS = ["ext_wage_median", "ext_wage_mean", "ext_wage_p10",
               "ext_wage_p25", "ext_wage_p75", "ext_wage_p90"]


def load_and_prep():
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    onet = [c for c in df.columns if "__" in c]
    ext = [c for c in df.columns if c.startswith("ext_")]

    W = df[[c for c in WAGE_LEVELS + ["ext_wage_p90p10", "ext_wage_p90p50",
            "ext_wage_p50p10"] if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    has_p10 = W["ext_wage_p10"].notna() if "ext_wage_p10" in W else pd.Series(False, index=df.index)
    tc = wr = 0
    for col in [c for c in WAGE_LEVELS if c in W.columns]:
        med = W[col].median(); na = W[col].isna()
        W.loc[na & has_p10, col] = OEWS_CAP; tc += int((na & has_p10).sum())
        W.loc[na & ~has_p10, col] = med;      wr += int((na & ~has_p10).sum())
    if {"ext_wage_p90", "ext_wage_p10"}.issubset(W.columns):
        W["ext_wage_p90p10"] = W["ext_wage_p90"] / W["ext_wage_p10"]
    if {"ext_wage_p90", "ext_wage_median"}.issubset(W.columns):
        W["ext_wage_p90p50"] = W["ext_wage_p90"] / W["ext_wage_median"]
    if {"ext_wage_median", "ext_wage_p10"}.issubset(W.columns):
        W["ext_wage_p50p10"] = W["ext_wage_median"] / W["ext_wage_p10"]
    print(f"wage fill: {tc} top-code -> cap, {wr} whole-row -> median")

    F = df[onet].apply(pd.to_numeric, errors="coerce").copy()
    for c in W.columns:
        F[c] = W[c]
    for c in ext:
        if c not in F.columns:
            F[c] = pd.to_numeric(df[c], errors="coerce")
    if "ext_employment" in F.columns:
        F["ext_employment"] = np.log1p(F["ext_employment"])
    n_missing = int(F.isna().sum().sum())
    F = F.fillna(F.median())
    feats = list(F.columns)
    print(f"median-filled {n_missing} cells | matrix {F.shape[0]} x {len(feats)} "
          f"({len(onet)} skill + {len([c for c in feats if c.startswith('ext_')])} external)")
    return F.values, feats, title


def block_of(c):
    if c.startswith("ext_"): return "external"
    return c.split("__", 1)[0]


def main():
    X, feats, title = load_and_prep()
    Xz = StandardScaler().fit_transform(X)
    full = PCA().fit(Xz)
    ev, cum = full.explained_variance_ratio_, np.cumsum(full.explained_variance_ratio_)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(range(1, 21), ev[:20], "o-"); a1.set_title("Scree (mixed)")
    a2.plot(range(1, 21), cum[:20], "o-"); a2.axhline(.9, ls="--", c="grey")
    a2.set_title("Cumulative"); fig.tight_layout()
    fig.savefig(OUTDIR / "mixed_scree.png", dpi=130); plt.close(fig)
    print(f"\nPC1={ev[0]:.1%}, PC1-3={cum[2]:.1%}, to90%={int(np.argmax(cum>=.9))+1} PCs")

    comp = full.components_[:N_PC]
    Xp = full.transform(Xz)[:, :N_PC]

    # loadings per axis (external variables marked)
    print("\n" + "=" * 70)
    print("MIXED-PCA AXIS LOADINGS (ext_ = external variable)")
    print("=" * 70)
    for k in range(N_PC):
        load = pd.Series(comp[k], index=feats)
        hi = load.sort_values(ascending=False).head(TOP)
        lo = load.sort_values().head(TOP)
        print(f"\nPC{k+1} ({ev[k]:.1%})")
        print("  + end:", "  ".join(f"{v:+.2f}:{n}" for n, v in hi.items() if v > 0.05)[:200])
        print("  - end:", "  ".join(f"{v:+.2f}:{n}" for n, v in lo.items() if v < -0.05)[:200])
        # explicitly flag any external variable in this axis's top loadings
        exthi = [(n, load[n]) for n in load.index if n.startswith("ext_") and abs(load[n]) > 0.10]
        if exthi:
            print("  external here:", ", ".join(f"{n}={v:+.2f}" for n, v in
                  sorted(exthi, key=lambda t: -abs(t[1]))))

    # block composition + external share per PC
    tags = sorted({block_of(c) for c in feats})
    tag_of = np.array([block_of(c) for c in feats])
    print("\n" + "=" * 70)
    print("BLOCK COMPOSITION (% of each axis variance) + external share")
    print("=" * 70)
    rows = []
    for k in range(N_PC):
        sq = comp[k] ** 2
        rows.append({t: sq[tag_of == t].sum() for t in tags})
    cdf = pd.DataFrame(rows, index=[f"PC{k+1}" for k in range(N_PC)])[tags]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + (cdf * 100).round(1).to_string())
    print("\nexternal share by PC:", {f"PC{k+1}": round(cdf.iloc[k]["external"]*100, 1)
                                       for k in range(N_PC)})

    # each external variable vs each PC
    print("\n" + "=" * 70)
    print("EACH EXTERNAL VARIABLE x PC (|r|>=0.4 flagged)")
    print("=" * 70)
    ext_idx = [i for i, c in enumerate(feats) if c.startswith("ext_")]
    hdr = "  ".join(f"PC{k+1:>2d}" for k in range(6))
    print(f"{'variable':22s} {hdr}")
    for i in ext_idx:
        rs = [np.corrcoef(Xz[:, i], Xp[:, k])[0, 1] for k in range(6)]
        cells = "  ".join(f"{r:+.2f}" for r in rs)
        flag = " *" if any(abs(r) >= 0.4 for r in rs) else ""
        print(f"{feats[i]:22s} {cells}{flag}")


if __name__ == "__main__":
    main()