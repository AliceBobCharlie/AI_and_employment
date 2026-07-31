#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mfa_analysis.py -- Multiple Factor Analysis (Escofier-Pages) over three groups
of variables describing the same 894 occupations, so that no group dominates the
global structure by sheer column count:

  group 1  SKILLS   : Skills/Abilities/Knowledge/WorkActivities IM + WorkContext
                      CX  (~257 cols, continuous ratings)
  group 2  ETE      : Education/Training/Experience percent distributions
                      (41 cols; compositional -- each scale's categories sum to
                      ~100, so this group carries a built-in collinearity)
  group 3  ECON_INST: wages, employment, union coverage, separation rates,
                      self-employment (labor-market / institutional variables)

MFA normalizes each group by its first singular value so each group has the same
maximum axial inertia; the question is whether ECON_INST (esp. union coverage)
introduces structure independent of skills, or is absorbed by the skill axes.

Prestige is NOT an active group (it is collinear with skills). It is projected
afterwards as a supplementary correlation, purely to help interpret axes.

Preprocessing before MFA (each variable then z-scored):
  wages: top-code highs (p10 present) -> OEWS cap; whole-row -> median; ratios
         recomputed from filled levels.  employment -> log1p.
  separation rates, self-employment, union, prestige -> as-is.
  residual gaps -> column median.

Prince notes respected: set random_state (randomized SVD); groups passed as a
dict {name:[cols]}; NaNs filled before fitting; print prince.__version__.

Outputs: eigenvalues, group contributions per axis, column_correlations
(loadings; where each variable lands, esp. union), partial axes (each group's
own structure vs the global axes), and supplementary prestige correlations.
Terminal + master_out/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

MASTER = Path("output/master_clean.xlsx")
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
SEED = 0
N_COMP = 8



def prep():
    """master_clean.xlsx is already imputed (see clean_master.py). Variables are
    standardized here; MFA then additionally weights each GROUP by its first
    singular value, which is what stops the 216-column skill group from
    outvoting the 14-column economic one."""
    df = pd.read_excel(MASTER).set_index("onet_soc")
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)

    skill = [c for c in df.columns if "__" in c and not c.startswith("ete_")]
    ete = [c for c in df.columns if c.startswith("ete_")]
    econ = [c for c in df.columns
            if c.startswith("ext_") and c != "ext_prestige"
            and c not in {"ext_wage_level_log", "ext_wage_disp_p90p10", "ext_employment"}]

    X = df[skill + ete + econ]
    X = (X - X.mean()) / X.std(ddof=0)
    print(f"standardized {X.shape[1]} variables")

    groups = {"SKILLS": skill, "ETE": ete, "ECON_INST": econ}
    # prestige is supplementary: it is collinear with skills, so letting it help
    # form the axes would just re-state the skill structure under another name.
    p = df["ext_prestige"]
    prestige = (p - p.mean()) / p.std(ddof=0)
    print("group sizes:", {k: len(v) for k, v in groups.items()})
    return X, groups, prestige, title


def main():
    import prince
    print("prince", prince.__version__)
    X, groups, prestige, title = prep()

    mfa = prince.MFA(n_components=N_COMP, n_iter=5, random_state=SEED)
    mfa = mfa.fit(X, groups=groups)

    # eigenvalues / variance explained
    print("\n" + "=" * 66)
    print("MFA EIGENVALUES / % variance")
    print("=" * 66)
    try:
        print(mfa.eigenvalues_summary.to_string())
    except Exception:
        ev = mfa.percentage_of_variance_ if hasattr(mfa, "percentage_of_variance_") else None
        print("percentage of variance:", ev)

    # row coordinates (occupation scores) -> save for downstream
    rc = mfa.row_coordinates(X)
    rc.columns = [f"Dim{i+1}" for i in range(rc.shape[1])]
    rc.insert(0, "title", title.values)
    rc.to_csv(OUTDIR / "mfa_row_coords.csv")

    # column correlations (loadings): where each variable lands.
    # In prince 0.20.x this is an ATTRIBUTE and its row index is a MultiIndex
    # (group_name, variable_name).
    print("\n" + "=" * 66)
    print("COLUMN CORRELATIONS (loadings) -- external variables on Dim1..4")
    print("does union / separation land on the same axes as skills, or its own?")
    print("=" * 66)
    cc = mfa.column_correlations.copy()
    cc.columns = [f"Dim{i+1}" for i in range(cc.shape[1])]
    ndim = min(4, cc.shape[1])
    dims = [f"Dim{i+1}" for i in range(ndim)]

    var_names = cc.index.get_level_values(-1)          # variable name level
    grp_names = cc.index.get_level_values(0)           # group name level
    is_ext = pd.Index(var_names).str.startswith("ext_")
    ext_view = cc[is_ext].copy()
    ext_view.index = pd.Index(var_names)[is_ext]       # flatten to variable name
    with pd.option_context("display.width", 200):
        print(ext_view[dims].round(3).to_string())

    # group-level contributions (manual): a group's contribution to a dimension
    # = sum of squared column correlations of its variables, normalized per dim.
    print("\n" + "=" * 66)
    print("GROUP CONTRIBUTIONS per axis (share of each dim explained by a group)")
    print("=" * 66)
    sq = cc[dims] ** 2
    G = sq.groupby(grp_names).sum()                    # sum within each group
    G = G / G.sum(axis=0)                               # normalize per dimension
    with pd.option_context("display.width", 200):
        print((G * 100).round(1).to_string())
    print("(each column sums to 100%; watch whether ECON_INST drives its own dim)")

    # partial axes (manual): each group's own PCA axes vs the global MFA dims.
    print("\n" + "=" * 66)
    print("PARTIAL AXES -- each group's own PC1..2 vs global MFA Dim1..4")
    print("(econ group axis NOT aligning with the skill-driven global dim =")
    print(" independent institutional structure)")
    print("=" * 66)
    from sklearn.decomposition import PCA as _PCA
    glob = rc[dims].values
    for gname, cols in groups.items():
        Xg = X[cols].values
        p = _PCA(n_components=2, random_state=SEED).fit_transform(Xg)
        for gi in range(2):
            rs = [np.corrcoef(p[:, gi], glob[:, d])[0, 1] for d in range(ndim)]
            cells = "  ".join(f"{r:+.2f}" for r in rs)
            print(f"  {gname:10s} ownPC{gi+1}: {cells}")

    # supplementary prestige correlation with the global axes
    if prestige is not None:
        print("\n" + "=" * 66)
        print("SUPPLEMENTARY: prestige correlation with MFA dimensions")
        print("=" * 66)
        for i in range(min(4, rc.shape[1] - 1)):
            r = np.corrcoef(prestige.values, rc[f"Dim{i+1}"].values)[0, 1]
            print(f"  Dim{i+1}: r = {r:+.3f}")

    print(f"\n[ok] wrote mfa_row_coords.csv")


if __name__ == "__main__":
    main()