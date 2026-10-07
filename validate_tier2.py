#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_tier2.py -- what the tier-2 axes R4-R6 measure once the curvature of
tier 1 is taken out (Section 5.5).

The tier-2 axes are uncorrelated with R1-R3 by construction, but not
independent of them: part of each is a non-linear function of tier 1 (a
square, a product, a bend), which PCA has to describe with extra linear
directions. Read directly, the loadings of R4-R6 mix that curvature with
whatever is new. So the order here is: remove first, then interpret.

  1. REMOVE THE CURVATURE OF TIER 1. Each tier-2 axis is predicted from R1-R3
     out of fold (five folds, repeated four times, the prediction averaged
     over the repeats), by three models with the settings of
     check_curvature.py part 1a:
       quadratic   R1-R3, their squares and products, ridge
       cubic       all monomials up to degree 3, ridge
       kNN         k-nearest neighbours in the space of the raw PC1-3 scores
                   (distances in the data), k chosen within each training
                   fold from 10, 20, 40
     The fitted part is the kNN prediction; the residual (axis minus fitted
     part) is what the rest of the script interprets. The cubic residual is
     reported alongside: if the two agree closely, the reading does not
     depend on the model. A linear fit is not needed: R4-R6 are exactly
     uncorrelated with R1-R3. Which squares and products drive the fitted
     part is shown by their correlations with each axis.
     Whether the predictability exceeds chance is tested in
     check_curvature.py part 1a (permutation nulls), and whether what is
     left still reproduces in part 1c; here it is only removed.

  2. WHAT THE RESIDUAL IS. The correlation of each of the 216 feature columns
     with the axis (equal to its loading), with the fitted part and with the
     residual. The strongest residual correlations name the axis; the
     strongest fitted-part correlations show what was removed. Also: the
     congruence of the residual correlations with the loadings, the mean
     residual correlation per O*NET expert category (needs the O*NET text
     files, as in validate_tier1.py), and the share of the summed squared
     correlations falling in the job half (Work Activities, Work Context)
     and in each block, against those halves' and blocks' shares of columns.
     For R4 two composites test the reading "manipulation versus navigation":
     the mean z-score of the fine-manipulation items and of the
     navigation/vehicle items.

  3. OCCUPATIONS. The occupations at each end of the residual, with their
     R1-R3; the mean R1-R3 of the top and bottom tenth (are both ends at the
     same place in tier 1?); and the middle fifth, with its largest
     occupations by employment.

  4. OUTSIDE VARIABLES. Correlations of the axis and of its residual with the
     labour-market columns, a mean required-education level (from the ETE
     Required Level of Education shares, categories 1-11), selected Work
     Context items bearing on the cost of error and on automation (these are
     feature columns, so they are partly what defines the axes), and, if the
     file is present, the Eloundou et al. (2024) exposure measures. The
     partial column correlates the axis with each variable after the
     variable's linear dependence on R1-R3 is removed.

  5. ROBUSTNESS. Tier 2 refitted with the labour-market variables added to
     the feature columns, as in validate_tier1.py.

Run after pca_rotated.py. Reads output/rotated_axes.csv,
output/rotated_loadings.csv, output/master_clean.csv,
optionally the O*NET text files and data_raw/eloundou_occ_level.csv (the
occupation-level file data/occ_level.csv of github.com/openai/GPTs-are-GPTs).
Writes output/validate_tier2.xlsx and prints every table.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from pca_rotated import (MASTER, OUTDIR, SEED, FEATURE_PREFIXES, JOB_PREFIXES, ECON,
                        TIER1_ANCHORS, AXES, LOADINGS, TOP, standardize, eigen,
                        block_of, banner, congruence)
from check_curvature import model, N_FOLDS, N_REPEATS
from validate_tier1 import element_categories, check_robustness

OUT = OUTDIR / "validate_tier2.xlsx"
ELOUNDOU = Path("data_raw/eloundou_occ_level.csv")
ELOUNDOU_COLUMNS = ["human_rating_alpha", "human_rating_beta",
                    "dv_rating_alpha", "dv_rating_beta"]
N_ENDS = 15          # occupations listed at each end
MIDDLE = 0.20        # share of occupations closest to zero

COMPOSITES = {"R4": {
    "manipulation": ["abilities_im__Finger Dexterity", "abilities_im__Arm-Hand Steadiness",
                     "abilities_im__Manual Dexterity", "abilities_im__Wrist-Finger Speed",
                     "workact_im__Handling and Moving Objects"],
    "navigation": ["abilities_im__Spatial Orientation", "abilities_im__Night Vision",
                   "abilities_im__Peripheral Vision", "abilities_im__Glare Sensitivity",
                   "abilities_im__Sound Localization",
                   "workact_im__Operating Vehicles, Mechanized Devices, or Equipment"]}}

WORK_CONTEXT_ITEMS = ["Consequence of Error", "Degree of Automation",
                      "Importance of Being Exact or Accurate",
                      "Importance of Repeating Same Tasks", "Freedom to Make Decisions",
                      "Frequency of Decision Making", "Time Pressure"]


def short(col):
    """'workctx_cx__Time Pressure' -> 'workctx: Time Pressure'."""
    blk, name = col.split("__", 1)
    return f"{blk.split('_')[0]}: {name[:52]}"


def corr(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[ok], b[ok])[0, 1], int(ok.sum())


# --------------------------------------------------------------------------- #
# 1. Remove the curvature of tier 1
# --------------------------------------------------------------------------- #
def oof_predict(kind, X, y):
    """Out-of-fold prediction averaged over N_REPEATS repeats of N_FOLDS
    folds, and the out-of-fold R2 of each repeat."""
    preds = np.empty((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        for train, test in KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(X):
            preds[r, test] = model(kind).fit(X[train], y[train]).predict(X[test])
    r2 = 1 - ((y - preds) ** 2).sum(axis=1) / ((y - y.mean()) ** 2).sum()
    return preds.mean(axis=0), r2


def remove_curvature(T1, T1_raw, S2):
    banner("1. REMOVE THE CURVATURE OF TIER 1\n"
           "   (out-of-fold R2 of each tier-2 axis predicted from R1-R3)")
    fitted, residual, rows = {}, {}, []
    for a in S2.columns:
        y = S2[a].to_numpy()
        quad, r2q = oof_predict("quadratic", T1, y)
        cub, r2c = oof_predict("cubic", T1, y)
        knn, r2k = oof_predict("kNN", T1_raw, y)
        fitted[a], residual[a] = knn, y - knn
        rows.append({"axis": a,
                     "quadratic R2": r2q.mean(), "cubic R2": r2c.mean(), "kNN R2": r2k.mean(),
                     "kNN R2 min-max": f"{r2k.min():.3f} to {r2k.max():.3f}",
                     "variance kept by kNN residual": residual[a].var() / y.var(),
                     "r(kNN residual, cubic residual)": np.corrcoef(y - knn, y - cub)[0, 1],
                     "r(kNN residual, axis)": np.corrcoef(y - knn, y)[0, 1]})
    fit = pd.DataFrame(rows)
    print(fit.round(3).to_string(index=False))
    print("\n  the residual (axis minus kNN prediction) is what sections 2-4 interpret")

    P = PolynomialFeatures(2, include_bias=False).fit(T1)
    names = [n.replace("x0", "R1").replace("x1", "R2").replace("x2", "R3").replace(" ", " x ")
             for n in P.get_feature_names_out()]
    Q = P.transform(T1)
    keep = [j for j, n in enumerate(names) if "^2" in n or " x " in n]
    terms = pd.DataFrame({a: [np.corrcoef(Q[:, j], S2[a])[0, 1] for j in keep]
                          for a in S2.columns}, index=[names[j] for j in keep])
    print("\n  squares and products of R1-R3, correlation with each axis:")
    print(terms.round(2).to_string())
    return pd.DataFrame(fitted, index=S2.index), pd.DataFrame(residual, index=S2.index), fit, terms


# --------------------------------------------------------------------------- #
# 2. What the residual is
# --------------------------------------------------------------------------- #
def item_correlations(Z, feats, S2, fitted, residual):
    banner("2. WHAT THE RESIDUAL IS -- item correlations")
    load = lambda s: np.array([np.corrcoef(s, Z[:, j])[0, 1] for j in range(Z.shape[1])])
    items = {}
    for a in S2.columns:
        items[f"{a} loading"] = load(S2[a].to_numpy())
        items[f"{a} fitted r"] = load(fitted[a].to_numpy())
        items[f"{a} residual r"] = load(residual[a].to_numpy())
    items = pd.DataFrame(items, index=feats)
    for a in S2.columns:
        c = congruence(items[f"{a} loading"].to_numpy(), items[f"{a} residual r"].to_numpy())
        print(f"\n{a}: congruence of residual correlations with the loadings {c:.3f}")
        for part in ("residual", "fitted"):
            s = items[f"{a} {part} r"].sort_values()
            print(f"  -- {part} {'(what is left)' if part == 'residual' else '(what was removed)'}")
            print("   + end: " + "\n          ".join(
                f"{v:+.2f}  {short(c)}" for c, v in s[::-1].head(TOP).items()))
            print("   - end: " + "\n          ".join(
                f"{v:+.2f}  {short(c)}" for c, v in s.head(TOP).items()))
    return items


def category_means(items, feats):
    banner("2b. MEAN RESIDUAL CORRELATION per O*NET expert category")
    name2cat = element_categories()
    cats = [name2cat.get(tuple(c.split("__", 1))) for c in feats]
    if not any(cats):
        print("  could not map element names to categories (O*NET text files absent); skipped")
        return pd.DataFrame()
    cols = [c for c in items.columns if c.endswith("residual r")]
    t = items[cols].assign(category=cats).dropna(subset=["category"])
    out = t.groupby("category")[cols].mean()
    out["n"] = t.groupby("category").size()
    with pd.option_context("display.width", 200):
        print(out.round(2).to_string())
    return out


def shares(items, feats):
    banner("2c. SHARE OF SUMMED SQUARED CORRELATIONS by half and block\n"
           "    (job half = Work Activities + Work Context)")
    blk = np.array([block_of(c) for c in feats])
    job = np.array([c.startswith(JOB_PREFIXES) for c in feats])
    rows = []
    for col in items.columns:
        sq = items[col].to_numpy() ** 2
        rows.append({"score": col, "job half": sq[job].sum() / sq.sum(),
                     **{b: sq[blk == b].sum() / sq.sum() for b in dict.fromkeys(blk)}})
    rows.append({"score": "share of columns", "job half": job.mean(),
                 **{b: (blk == b).mean() for b in dict.fromkeys(blk)}})
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    return t


def composites(Z, feats, S2, fitted, residual):
    rows = []
    fidx = {f: j for j, f in enumerate(feats)}
    for a, parts in COMPOSITES.items():
        if a not in S2.columns:
            continue
        comp = {n: Z[:, [fidx[c] for c in cols if c in fidx]].mean(axis=1)
                for n, cols in parts.items()}
        names = list(comp)
        comp[f"{names[0]} minus {names[1]}"] = comp[names[0]] - comp[names[1]]
        for n, v in comp.items():
            rows.append({"axis": a, "composite": n,
                         "r with axis": np.corrcoef(v, S2[a])[0, 1],
                         "r with fitted part": np.corrcoef(v, fitted[a])[0, 1],
                         "r with residual": np.corrcoef(v, residual[a])[0, 1]})
    t = pd.DataFrame(rows)
    if len(t):
        banner("2d. COMPOSITES for the reading of the axis")
        for a, parts in COMPOSITES.items():
            for n, cols in parts.items():
                print(f"  {a} {n}: " + "; ".join(c.split("__")[1] for c in cols))
        print("\n" + t.round(3).to_string(index=False))
    return t


# --------------------------------------------------------------------------- #
# 3. Occupations
# --------------------------------------------------------------------------- #
def occupations(title, T1df, S2, fitted, residual, employment, soc6):
    banner("3. OCCUPATIONS along the residual")
    occ = T1df.copy()
    occ.insert(0, "title", title)
    for a in S2.columns:
        occ[a] = S2[a]
        occ[f"{a} fitted"] = fitted[a]
        occ[f"{a} residual"] = residual[a]
    t1 = list(T1df.columns)
    tails = []
    for a in S2.columns:
        r = residual[a]
        show = ["title"] + t1 + [a, f"{a} residual"]
        for asc, end in ((False, "+"), (True, "-")):
            print(f"\n{a} residual, {end} end:")
            top = occ.sort_values(f"{a} residual", ascending=asc).head(N_ENDS)
            print(top[show].assign(title=top["title"].str.slice(0, 50))
                  .round(2).to_string(index=False))
        lo, hi = r.quantile(0.10), r.quantile(0.90)
        mid = r.abs() <= r.abs().quantile(MIDDLE)
        for label, mask in (("top tenth", r >= hi), ("bottom tenth", r <= lo),
                            ("middle fifth", mid)):
            tails.append({"axis": a, "group": label, "n": int(mask.sum()),
                          **{f"mean {c}": occ.loc[mask, c].mean() for c in t1},
                          "mean residual": r[mask].mean()})
        # employment is published per 6-digit SOC code, so O*NET occupations
        # sharing a code share one figure: list one occupation per code
        big = (occ.loc[mid].assign(emp=employment[mid], soc6=soc6[mid])
               .sort_values("emp", ascending=False).drop_duplicates("soc6").head(10))
        print(f"\n{a} middle fifth (|residual| smallest), largest by employment "
              "(one per 6-digit SOC): " + "; ".join(big["title"].str.slice(0, 40)))
    tails = pd.DataFrame(tails)
    print("\nmean tier-1 position of each end and of the middle:")
    print(tails.round(2).to_string(index=False))
    return occ, tails


# --------------------------------------------------------------------------- #
# 4. Outside variables
# --------------------------------------------------------------------------- #
def education_level(m):
    """Mean required-education category, 1-11, from the ETE shares."""
    cols = {int(c.rsplit("__c", 1)[1]): c for c in m.columns
            if c.startswith("ete_rl__")}
    if not cols:
        return None
    w = m[[cols[k] for k in sorted(cols)]].to_numpy(float)
    return (w * np.array(sorted(cols))).sum(axis=1) / w.sum(axis=1)


def outside_variables(m, T1, S2, residual):
    banner("4. OUTSIDE VARIABLES against the axis and its residual")
    ext = {c: m[c].to_numpy(float) for c in ECON if c in m.columns}
    edu = education_level(m)
    if edu is not None:
        ext["mean required education (ETE categories 1-11)"] = edu
    for item in WORK_CONTEXT_ITEMS:
        col = "workctx_cx__" + item
        if col in m.columns:
            ext["feature: " + item] = m[col].to_numpy(float)
    if ELOUNDOU.exists():
        el = pd.read_csv(ELOUNDOU).set_index("O*NET-SOC Code").reindex(m.index)
        for c in ELOUNDOU_COLUMNS:
            if c in el.columns:
                ext["Eloundou " + c] = el[c].to_numpy(float)
    else:
        print(f"  ({ELOUNDOU} not found: exposure measures skipped)")

    A = np.column_stack([np.ones(len(T1)), T1])
    rows = []
    for name, v in ext.items():
        ok = ~np.isnan(v)
        coef = np.linalg.lstsq(A[ok], v[ok], rcond=None)[0]
        v_part = np.full_like(v, np.nan)
        v_part[ok] = v[ok] - A[ok] @ coef
        row = {"variable": name, "n": int(ok.sum())}
        for a in S2.columns:
            y = S2[a].to_numpy()
            row[f"{a} r"] = corr(y, v)[0]
            row[f"{a} residual r"] = corr(residual[a].to_numpy(), v)[0]
            row[f"{a} partial on R1-R3"] = corr(y, v_part)[0]
        rows.append(row)
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(2).to_string(index=False))
    return t


# --------------------------------------------------------------------------- #
def main():
    m = pd.read_csv(MASTER, index_col="onet_soc")
    S = pd.read_csv(AXES, index_col=0)
    L = pd.read_csv(LOADINGS, index_col=0)
    t1 = [f"R{j + 1}" for j in range(len(TIER1_ANCHORS))]
    t2 = [c for c in S.columns if c.startswith("R") and c not in t1]
    if not t2:
        raise SystemExit("rotated_axes.csv has no tier-2 axes; run pca_rotated.py")
    m = m.reindex(S.index)
    feats = [c for c in m.columns if c.startswith(FEATURE_PREFIXES)]
    X = m[feats].to_numpy(float)
    Z = standardize(X)
    k1 = len(t1)
    ev, V = eigen(X)
    T1_raw = Z @ V[:, :k1]                 # raw PC1-3 scores: distances in the data
    T1 = S[t1].to_numpy()
    S2 = S[t2]
    print(f"{len(S)} occupations, {len(feats)} feature columns; tier 1 = {', '.join(t1)}, "
          f"tier 2 = {', '.join(t2)} (scores from {AXES})")

    fitted, residual, fit, terms = remove_curvature(T1, T1_raw, S2)
    items = item_correlations(Z, feats, S2, fitted, residual)
    cats = category_means(items, feats)
    share = shares(items, feats)
    comp = composites(Z, feats, S2, fitted, residual)
    occ, tails = occupations(m["title"], S[t1], S2, fitted, residual,
                             m["ext_employment"], m["soc6"])
    outside = outside_variables(m, T1, S2, residual)
    rob = check_robustness(L[t2], m, [list(range(k1, k1 + len(t2)))], heading="5. ROBUSTNESS")

    with pd.ExcelWriter(OUT) as writer:
        fit.to_excel(writer, sheet_name="curvature_fit", index=False)
        terms.to_excel(writer, sheet_name="quadratic_terms")
        items.to_excel(writer, sheet_name="item_correlations")
        cats.to_excel(writer, sheet_name="categories")
        share.to_excel(writer, sheet_name="shares", index=False)
        comp.to_excel(writer, sheet_name="composites", index=False)
        occ.to_excel(writer, sheet_name="occupations")
        tails.to_excel(writer, sheet_name="ends_and_middle", index=False)
        outside.to_excel(writer, sheet_name="outside_variables", index=False)
        rob.to_excel(writer, sheet_name="robustness", index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
