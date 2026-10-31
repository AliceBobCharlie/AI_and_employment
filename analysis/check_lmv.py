#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_lmv.py -- data quality of the labour-market variables.

For the occupations of the feature matrix, on raw values as published (before
clean_lmv.py fills anything): which occupations miss which external
variable, and each variable's distribution and skew.

ete_structure() describes how far the education, training and experience
(ETE) distributions of output/labour_clean.csv reduce to a few variables:
1. The share of the variance of the z-scored ETE columns in the first
   N_PCS principal components.
2. For each scale in ETE_SCALES, the category shares with the dropped
   highest category restored (100 minus the others, clipped at 0, rows
   scaled to sum to 1): the share of their variance in the first N_PCS
   components, and the R2 of all the shares together (share_r2()) from the
   mean category; the mean and its square; the mean and the sd of the
   categories; and the mean and the share in the two end categories.
3. The correlations of the four scale means, and the share of their
   variance in each of their principal components.
4. The R2 of the z-scored ETE columns, and of the restored shares, from the
   four scale means, and from the means and their squares.

wage_structure() describes how far the five wage quantiles of
output/labour_clean.csv (WAGE_QUANTILES, logged) reduce to a few variables,
on all occupations and without those with a quantile at the OEWS cap
(WAGE_CAP):
5. The share of the variance of the log quantiles in each principal
   component, and the loadings of the first three.
6. The R2 of the five log quantiles together (share_r2()) from the log
   median; the log median and the spread, log p90 - log p10; those and the
   skew, (log p90 - log p50) - (log p50 - log p10); and the mean of the five
   logs and the spread.

Run after build_lmv.py, clean_lmv.py and clean_master.py, from the project
root. Reads output/labour.csv, output/labour_clean.csv and
output/master_clean.csv (for the rows). Writes
output/labour_data_quality.xlsx and prints the tables of external().
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("output/labour.csv")
CLEAN = Path("output/labour_clean.csv")
ROWS = Path("output/master_clean.csv")
OUT = Path("output/labour_data_quality.xlsx")
ETE_SCALES = {"rl": "Required Level of Education", "rw": "Related Work Experience",
              "pt": "On-Site or In-Plant Training", "oj": "On-the-Job Training"}
N_PCS = 10              # components reported
WAGE_QUANTILES = ["ext_wage_p10", "ext_wage_p25", "ext_wage_median", "ext_wage_p75", "ext_wage_p90"]
WAGE_CAP = 239200       # OEWS top code


def show(df, floatfmt="{:.3f}"):
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", None, "display.max_colwidth", 70,
                           "display.float_format", floatfmt.format):
        print(df.to_string())


def external(L, kept):
    ext = [c for c in L.columns if c.startswith("ext_")]
    X = L.loc[kept, ext].apply(pd.to_numeric, errors="coerce")
    na = X.isna()
    per_var = na.sum().rename("missing").to_frame()
    missing = na[na.any(axis=1)].astype(int)
    missing.columns = [c.replace("ext_", "") for c in ext]
    missing.insert(0, "title", L.loc[missing.index, "title"])
    missing["n missing"] = missing.iloc[:, 1:].sum(axis=1)
    missing = missing.sort_values("n missing", ascending=False)
    desc = pd.DataFrame({"n": X.count(), "mean": X.mean(), "sd": X.std(), "min": X.min(),
                         "q25": X.quantile(.25), "median": X.median(), "q75": X.quantile(.75),
                         "max": X.max(), "skew": X.skew(),
                         "max/median": X.max() / X.median().replace(0, np.nan)})
    print(f"occupations missing at least one variable: {len(missing)} of {len(X)}; "
          f"missing all but at most one: {int((missing['n missing'] >= len(ext) - 1).sum())}; "
          f"missing one or two: {int((missing['n missing'] <= 2).sum())}")
    show(per_var)
    print("\noccupations with a missing value (1 = missing):")
    show(missing.replace({0: ""}), "{:.0f}")
    print()
    show(desc, "{:,.2f}")
    skewed = desc.index[(desc["skew"].abs() > 1) | (desc["max/median"] > 5)]
    print(f"\nright-skewed (|skew| > 1 or max/median > 5): {list(skewed) or 'none'}")
    return {"external_missing": missing.reset_index(),
            "external_describe": desc.reset_index(names="variable")}


def share_r2(P, D):
    """R2 of the columns of P together from [1, D] by least squares."""
    D = np.column_stack([np.ones(len(P)), D])
    e = P - D @ np.linalg.lstsq(D, P, rcond=None)[0]
    return 1 - (e ** 2).sum() / ((P - P.mean(axis=0)) ** 2).sum()


def cumulative_share(X):
    """Cumulative share of the variance of the columns of X in its principal components."""
    ev = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
    return np.cumsum(ev) / ev.sum()


def scale_shares(clean, scale):
    """The category shares of one ETE scale, the dropped highest category
    restored, clipped at 0, rows scaled to sum to 1."""
    cols = sorted([c for c in clean.columns if c.startswith(f"ete_{scale}__")],
                  key=lambda c: int(c.rsplit("__c", 1)[1]))
    X = clean[cols].to_numpy(float)
    P = np.clip(np.column_stack([X, 100 - X.sum(axis=1)]), 0, None)
    return P / P.sum(axis=1, keepdims=True)


def ete_structure(clean):
    cols = [c for c in clean.columns if c.startswith("ete_")]
    E = clean[cols].to_numpy(float)
    Ez = (E - E.mean(axis=0)) / E.std(axis=0)
    cum = cumulative_share(Ez)
    block = pd.DataFrame({"component": range(1, N_PCS + 1), "cumulative share": cum[:N_PCS]})

    rows, means, shares = [], {}, []
    for scale, name in ETE_SCALES.items():
        P = scale_shares(clean, scale)
        shares.append(P)
        k = np.arange(1, P.shape[1] + 1)
        m = P @ k
        sd = np.sqrt((P * (k - m[:, None]) ** 2).sum(axis=1))
        means[scale] = m
        cum = cumulative_share(P)
        rows.append({"scale": name, "categories": P.shape[1],
                     "R2, mean": share_r2(P, m),
                     "R2, mean and mean^2": share_r2(P, np.column_stack([m, m ** 2])),
                     "R2, mean and sd": share_r2(P, np.column_stack([m, sd])),
                     "R2, mean and end categories": share_r2(P, np.column_stack([m, P[:, 0] + P[:, -1]])),
                     **{f"share in first {j} PCs": cum[j - 1] for j in range(1, 6)}})
    scales = pd.DataFrame(rows)

    M = np.column_stack(list(means.values()))
    names = [ETE_SCALES[s] for s in means]
    corr = pd.DataFrame(np.corrcoef(M.T), index=names, columns=names).reset_index(names="scale mean")
    cum = cumulative_share((M - M.mean(axis=0)) / M.std(axis=0))
    means_pca = pd.DataFrame({"component": range(1, len(cum) + 1), "cumulative share": cum})

    S = np.hstack(shares)
    from_means = pd.DataFrame([
        {"target": "z-scored ETE columns", "R2, four scale means": share_r2(Ez, M),
         "R2, means and their squares": share_r2(Ez, np.column_stack([M, M ** 2]))},
        {"target": "restored category shares", "R2, four scale means": share_r2(S, M),
         "R2, means and their squares": share_r2(S, np.column_stack([M, M ** 2]))}])
    return {"ete_block_pca": block, "ete_scales": scales, "ete_scale_means_corr": corr,
            "ete_scale_means_pca": means_pca, "ete_from_means": from_means}


def wage_structure(clean):
    capped = (clean[WAGE_QUANTILES] >= WAGE_CAP).any(axis=1).to_numpy()
    samples = {"all occupations": np.ones(len(clean), bool), "no quantile at the cap": ~capped}
    pca, models = {}, []
    for label, keep in samples.items():
        W = np.log(clean.loc[keep, WAGE_QUANTILES].to_numpy(float))
        Wc = W - W.mean(axis=0)
        _, sv, vt = np.linalg.svd(Wc, full_matrices=False)
        ev = sv ** 2
        pca[label] = (ev / ev.sum(), vt)
        med, spread = W[:, 2], W[:, 4] - W[:, 0]
        skew = (W[:, 4] - W[:, 2]) - (W[:, 2] - W[:, 0])
        models.append({"sample": label, "occupations": int(keep.sum()),
                       "R2, log median": share_r2(W, med),
                       "R2, log median and spread": share_r2(W, np.column_stack([med, spread])),
                       "R2, log median, spread and skew": share_r2(W, np.column_stack([med, spread, skew])),
                       "R2, mean of the logs and spread": share_r2(W, np.column_stack([W.mean(axis=1), spread]))})
    rows = []
    for label, (share, vt) in pca.items():
        for j in range(len(share)):
            rows.append({"sample": label, "component": j + 1, "share": share[j],
                         "cumulative share": share[: j + 1].sum(),
                         **({f"loading, {q.replace('ext_wage_', '')}": vt[j, i] for i, q in enumerate(WAGE_QUANTILES)}
                            if j < 3 else {})})
    return {"wage_pca": pd.DataFrame(rows), "wage_models": pd.DataFrame(models)}


def main():
    L = pd.read_csv(RAW, index_col="onet_soc", low_memory=False)
    kept = pd.read_csv(ROWS, index_col="onet_soc", usecols=["onet_soc"]).index
    print(f"[read] {RAW}: {len(L)} codes; {ROWS}: {len(kept)} occupations kept")

    sheets = external(L, kept)
    clean = pd.read_csv(CLEAN, index_col="onet_soc")
    sheets.update(ete_structure(clean))
    print(f"[ok] ETE structure: {', '.join(t for t in sheets if t.startswith('ete_'))}")
    sheets.update(wage_structure(clean))
    print("[ok] wage structure: wage_pca, wage_models")
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()