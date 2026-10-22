#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data.py -- data quality of the occupational features.

Computed from the master tables and, for the flags and standard errors, from
the O*NET text files themselves (build_master.py does not carry the flags into
the master table). The labour-market variables are checked by
check_labour_data.py.

  A  codes        O*NET-SOC codes in Occupation Data; which feature blocks
                  each is rated in; the unrated codes by kind (military,
                  SOC "All Other" residuals, legislators, the rest listed by
                  name); codes rated in only some blocks.
  B  flags        For every feature block and scale, all rows of the text
                  files and the rows of the 910 occupations kept: Recommend
                  Suppress and Not Relevant counts and shares, and the median
                  published value of a rating flagged Not Relevant.
  C  standard errors
                  For the scales the analysis uses, the share of the 910
                  occupations' ratings that carry a standard error, by block
                  and by Domain Source.
  D  Importance against Level
                  For each of the 161 elements rated on both scales, the
                  correlation of Importance and Level across the 910
                  occupations; and how well the whole Level matrix is
                  predicted from the Importance matrix by ridge regression
                  (five-fold cross-validated R2, scaler and penalty fitted
                  in-fold, Level columns z-scored).
  E  Work Context categories
                  What the category percentages (CXP) of the 55 CX items hold
                  beyond their mean: how much of the category shares a cubic
                  in the mean explains, item by item; and the structure of
                  what is left, by PCA with a permutation null and
                  split-halves, its first component set against the
                  within-occupation spread of the ratings.
  F  the final matrix
                  Columns and missing cells, as published, per block of the
                  910 x 220 matrix (Table 4.1).

Run after build_master.py and clean_master.py, from the project root.
Reads output/master.csv, output/master_clean.csv and data_raw/onet/db_31_0_text,
one file at a time and only the columns needed. Writes
output/data_quality.xlsx and prints every table in full.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ONET_DIR = Path("data_raw/onet/db_31_0_text")
RAW = Path("output/master.csv")
CLEAN = Path("output/master_clean.csv")
OUT = Path("output/data_quality.xlsx")
SEED = 0

# Column prefix -> O*NET files, and the scale the analysis uses.
FEATURE_BLOCKS = {
    "abilities": (["Abilities"], "IM"),
    "skills":    (["Essential Skills", "Transferable Skills"], "IM"),
    "knowledge": (["Knowledge"], "IM"),
    "workact":   (["Work Activities"], "IM"),
    "workctx":   (["Work Context"], "CX"),
}
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__", "workctx_ctp__")
N_DRAWS = 200            # permutations and split-halves in section E
TEXT_COLUMNS = {
    "O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
    "Category": "category", "Data Value": "value", "Standard Error": "se",
    "Recommend Suppress": "suppress", "Not Relevant": "not_relevant",
    "Domain Source": "source",
}
LEGISLATORS = "11-1031.00"


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def show(df, floatfmt="{:.3f}"):
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", None, "display.max_colwidth", 70,
                           "display.float_format", floatfmt.format):
        print(df.to_string())


def read_block(prefix):
    """All rows of a block's text files, flags as booleans, values numeric.
    Category-scale breakdowns (rows with a category) are left out."""
    parts = []
    for stem in FEATURE_BLOCKS[prefix][0]:
        path = ONET_DIR / f"{stem}.txt"
        print(f"  reading {path}")
        parts.append(pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                                 usecols=lambda c: c in TEXT_COLUMNS).rename(columns=TEXT_COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    if "category" in df.columns:
        df = df[df["category"].isna()]
    for col in ("value", "se"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("suppress", "not_relevant"):
        df[col] = df[col].str.strip().str.upper().eq("Y") if col in df.columns else False
    df["block"] = prefix
    return df


# --------------------------------------------------------------------------- #
def section_codes(M):
    banner("A. CODES -- which O*NET-SOC codes are rated, and what the unrated ones are")
    rated = pd.DataFrame({b: M[[c for c in M.columns if c.startswith(f"{b}_")]].notna().any(axis=1)
                          for b in FEATURE_BLOCKS})
    n_rated = rated.sum(axis=1)
    code = M.index.to_series()
    four = code.str[3:7]
    kind = pd.Series("ordinary detailed occupation", index=M.index)
    kind[code.str.endswith(".00") & four.str.endswith("9")] = "SOC 'All Other' residual"
    kind[code.str.startswith("55-")] = "military"
    kind[code == LEGISLATORS] = "legislators"

    summary = pd.DataFrame([
        {"measure": "codes in Occupation Data", "value": len(M)},
        {"measure": "rated in all five feature blocks", "value": int((n_rated == 5).sum())},
        {"measure": "rated in no feature block", "value": int((n_rated == 0).sum())},
        {"measure": "rated in some blocks only", "value": int(((n_rated > 0) & (n_rated < 5)).sum())},
        {"measure": "ordinary detailed occupations (all codes not military, All Other or legislators)",
         "value": int((kind == "ordinary detailed occupation").sum())},
        {"measure": "ordinary detailed occupations not rated in all five",
         "value": int(((kind == "ordinary detailed occupation") & (n_rated < 5)).sum())},
    ])
    per_block = rated.sum().rename("codes rated").to_frame()
    unrated = M.loc[n_rated < 5, ["title"]].assign(
        kind=kind[n_rated < 5], blocks_rated=n_rated[n_rated < 5],
        which=rated[n_rated < 5].apply(lambda r: ", ".join(b for b in r.index if r[b]), axis=1))
    by_kind = (unrated[unrated["blocks_rated"] == 0].groupby("kind").size()
               .rename("unrated codes").to_frame())
    by_kind["share of unrated"] = by_kind["unrated codes"] / by_kind["unrated codes"].sum()

    show(summary); print(); show(per_block); print(); show(by_kind)
    print("\ncodes not rated in all five blocks, other than military and All Other:")
    show(unrated[~unrated["kind"].isin(["military", "SOC 'All Other' residual"])]
         .sort_values(["blocks_rated", "kind"]))
    return {"codes_summary": summary, "codes_per_block": per_block.reset_index(names="block"),
            "unrated_by_kind": by_kind.reset_index(), "codes_not_fully_rated": unrated.reset_index()}


def section_flags(rows, kept):
    banner("B. FLAGS -- Recommend Suppress and Not Relevant, per block and scale")
    out = []
    for (block, scale), g in rows.groupby(["block", "scale"]):
        for subset, h in (("all rows in the files", g), ("the 910 occupations kept", g[g["soc"].isin(kept)])):
            out.append({"block": block, "scale": scale, "rows of": subset, "rows": len(h),
                        "Recommend Suppress": int(h["suppress"].sum()),
                        "suppress share": h["suppress"].mean(),
                        "Not Relevant": int(h["not_relevant"].sum()),
                        "not relevant share": h["not_relevant"].mean(),
                        "median value, Not Relevant": h.loc[h["not_relevant"], "value"].median()})
    flags = pd.DataFrame(out)
    four = flags[(flags["block"] != "workctx") & flags["scale"].isin(["IM", "LV"])]
    totals = (four.groupby(["scale", "rows of"])[["rows", "Recommend Suppress", "Not Relevant"]]
              .sum().reset_index())
    totals["suppress share"] = totals["Recommend Suppress"] / totals["rows"]
    show(flags.set_index(["block", "scale", "rows of"]))
    print("\nAbilities, Skills, Knowledge and Work Activities together:")
    show(totals.set_index(["scale", "rows of"]))
    return {"flags": flags, "flags_four_blocks": totals}


def section_se(rows, kept):
    banner("C. STANDARD ERRORS -- share of the analysis ratings that carry one (910 occupations)")
    used = pd.concat([rows[(rows["block"] == b) & (rows["scale"] == s) & rows["soc"].isin(kept)]
                      for b, (_, s) in FEATURE_BLOCKS.items()])
    used = used.assign(has_se=used["se"].notna())
    by_block = used.groupby("block").agg(rows=("has_se", "size"), share_with_se=("has_se", "mean"))
    by_source = used.groupby(["block", "source"]).agg(rows=("has_se", "size"),
                                                      share_with_se=("has_se", "mean"))
    show(by_block); print(); show(by_source)
    print(f"\nall {len(used)} analysis ratings: {used['has_se'].mean():.1%} carry a standard error")
    return {"se_by_block": by_block.reset_index(), "se_by_source": by_source.reset_index()}


def cv_predictions(X, Y):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 30)))
    return cross_val_predict(model, X, Y, cv=KFold(5, shuffle=True, random_state=SEED))


def section_im_lv(M, kept):
    banner("D. IMPORTANCE AGAINST LEVEL (910 occupations)")
    D = M.loc[kept]
    pairs = []
    for b in ("abilities", "skills", "knowledge", "workact"):
        for c in [c for c in D.columns if c.startswith(f"{b}_im__")]:
            lv = c.replace("_im__", "_lv__", 1)
            if lv in D.columns:
                pairs.append({"block": b, "element": c.split("__", 1)[1], "im": c, "lv": lv,
                              "r(IM, LV)": np.corrcoef(D[c], D[lv])[0, 1]})
    P = pd.DataFrame(pairs)
    r = P["r(IM, LV)"]
    X = D[P["im"]].to_numpy(float)
    Y = D[P["lv"]].to_numpy(float)
    Yz = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    pred = cv_predictions(X, Yz)
    per_col = 1 - ((Yz - pred) ** 2).sum(axis=0) / (Yz ** 2).sum(axis=0)
    summary = pd.DataFrame([
        {"measure": "elements rated on both scales", "value": len(P)},
        {"measure": "missing cells, Importance / Level",
         "value": f"{int(np.isnan(X).sum())} / {int(np.isnan(Y).sum())}"},
        {"measure": "median r(IM, LV)", "value": r.median()},
        {"measure": "share of elements with r > 0.9", "value": (r > 0.9).mean()},
        {"measure": "lowest r(IM, LV)", "value": r.min()},
        {"measure": "Level matrix from Importance matrix: cross-validated R2 (all columns)",
         "value": 1 - ((Yz - pred) ** 2).sum() / (Yz ** 2).sum()},
        {"measure": "same, median over Level columns", "value": np.median(per_col)},
    ])
    P["R2 of LV column from all IM"] = per_col
    show(summary.set_index("measure"))
    print("\nby block:")
    show(P.groupby("block")["r(IM, LV)"].describe())
    print("\nthe 15 elements with the lowest r(IM, LV):")
    show(P.nsmallest(15, "r(IM, LV)")[["block", "element", "r(IM, LV)", "R2 of LV column from all IM"]]
         .set_index(["block", "element"]))
    return {"im_lv_summary": summary.astype({"value": str}),
            "im_lv_elements": P.drop(columns=["im", "lv"])}


def section_cxp(M, C):
    banner("E. WORK CONTEXT CATEGORIES -- what CXP holds beyond the CX mean")
    items = [c.split("__", 1)[1] for c in C.columns if c.startswith("workctx_cx__")]
    k = np.arange(1, 6)
    n = len(C)
    r_mean, r2_mean, r2_mean_sd, resid, sd, n_filled = [], [], [], [], [], 0
    for item in items:
        P = M.loc[C.index, [f"workctx_cxp__{item}__c{c}" for c in k]].to_numpy(float)
        n_filled += int(np.isnan(P).sum())
        P = np.where(np.isnan(P), np.nanmedian(P, axis=0), P)
        P = P / P.sum(axis=1, keepdims=True)
        mean = P @ k
        spread = np.sqrt((P * (k - mean[:, None]) ** 2).sum(axis=1))
        r_mean.append(np.corrcoef(mean, C[f"workctx_cx__{item}"])[0, 1])
        total = ((P - P.mean(axis=0)) ** 2).sum()

        def residual(D):
            D = np.column_stack([np.ones(n), D])
            return P - D @ np.linalg.lstsq(D, P, rcond=None)[0]
        e = residual(np.column_stack([mean, mean ** 2, mean ** 3]))
        r2_mean.append(1 - (e ** 2).sum() / total)
        e2 = residual(np.column_stack([mean, mean ** 2, mean ** 3, spread, spread * mean]))
        r2_mean_sd.append(1 - (e2 ** 2).sum() / total)
        resid.append(e[:, :4])               # four free shares per item
        sd.append(spread)
    E = np.hstack(resid)
    E = E[:, E.std(axis=0) > 1e-9]
    mean_sd = np.mean(sd, axis=0)

    def decompose(X):
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        _, s_, vt = np.linalg.svd(Z, full_matrices=False)
        return s_ ** 2 / (len(Z) - 1), vt.T, Z @ vt.T

    ev, V, T = decompose(E)
    rng = np.random.default_rng(SEED)
    null = np.array([decompose(rng.permuted(E, axis=0))[0][:10] for _ in range(N_DRAWS)])
    above = ev[:10] > np.percentile(null, 95, axis=0)
    split = []
    for _ in range(N_DRAWS):
        perm = rng.permutation(n)
        _, V1, _ = decompose(E[perm[: n // 2]])
        _, V2, _ = decompose(E[perm[n // 2:]])
        split.append(np.abs(V1[:, :3].T @ V2[:, :8]).max(axis=1))
    split = np.percentile(split, 5, axis=0)
    t = T[:, 0] / np.sqrt(ev[0])
    titles = C["title"].to_numpy()
    order = np.argsort(t)
    summary = pd.DataFrame([
        {"measure": "Work Context items rated on CX", "value": len(items)},
        {"measure": "CXP cells missing for the 910 (filled at the column median here)", "value": n_filled},
        {"measure": "r(CX, mean of the CXP categories), median over items", "value": np.median(r_mean)},
        {"measure": "category-share variance explained by a cubic in the mean, median", "value": np.median(r2_mean)},
        {"measure": "  10th percentile over items", "value": np.percentile(r2_mean, 10)},
        {"measure": "  90th percentile over items", "value": np.percentile(r2_mean, 90)},
        {"measure": "  with the within-occupation sd added", "value": np.median(r2_mean_sd)},
        {"measure": "residual beyond the mean: components above the permutation null",
         "value": int(np.argmin(above)) if not above.all() else 10},
        {"measure": "  component 1, % of residual variance", "value": 100 * ev[0] / ev.sum()},
        {"measure": "  component 2, % of residual variance", "value": 100 * ev[1] / ev.sum()},
        {"measure": "  split-half congruence p05, component 1", "value": split[0]},
        {"measure": "  split-half congruence p05, component 2", "value": split[1]},
        {"measure": "  split-half congruence p05, component 3", "value": split[2]},
        {"measure": "  r(component 1, mean within-occupation sd)", "value": np.corrcoef(t, mean_sd)[0, 1]},
        {"measure": "CXP columns with one category per item dropped", "value": 4 * len(items)},
    ])
    poles = pd.DataFrame({"low end of component 1": titles[order[:10]],
                          "high end of component 1": titles[order[::-1][:10]]})
    show(summary.set_index("measure"))
    print("\nthe two ends of the first component beyond the mean (sign arbitrary):")
    show(poles)
    return {"cxp_summary": summary.astype({"value": float}), "cxp_poles": poles}


def section_matrix(M, C):
    banner("F. THE FINAL MATRIX (Table 4.1)")
    feats = [c for c in C.columns if c.startswith(FEATURE_PREFIXES)]
    blocks = pd.Series([c.split("__")[0] for c in feats], index=feats)
    published = M.loc[C.index, feats].apply(pd.to_numeric, errors="coerce")
    T = pd.DataFrame({"columns": blocks.value_counts(),
                      "missing cells as published": published.isna().sum().groupby(blocks).sum()})
    T.loc["total"] = T.sum()
    print(f"{len(C)} occupations x {len(feats)} columns; missing cells take the "
          "column median in clean_master.py")
    show(T)
    return {"final_matrix": T.reset_index(names="block")}


# --------------------------------------------------------------------------- #
def main():
    M = pd.read_csv(RAW, index_col="onet_soc", low_memory=False)
    C = pd.read_csv(CLEAN, index_col="onet_soc")
    kept = C.index
    print(f"[read] {RAW}: {len(M)} codes; {CLEAN}: {len(kept)} occupations kept")

    sheets = {}
    sheets.update(section_codes(M))
    rows = pd.concat([read_block(b) for b in FEATURE_BLOCKS], ignore_index=True)
    sheets.update(section_flags(rows, kept))
    sheets.update(section_se(rows, kept))
    sheets.update(section_im_lv(M, kept))
    sheets.update(section_cxp(M, C))
    sheets.update(section_matrix(M, C))

    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()