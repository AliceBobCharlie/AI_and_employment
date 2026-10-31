#!/usr/bin/env python3
"""
check_data.py -- check the quality of the O*NET feature data: which codes are
rated, the flags and standard errors of the ratings, Importance against
Level, the Work Context category shares, and the missing cells of the feature
matrix.

Input:  output/master.csv, output/master_clean.csv, and the O*NET text files of
        the blocks in FEATURE_BLOCKS in ONET_DIR
Output: one CSV per table in output/data_quality/

main() reads RAW (every O*NET-SOC code) and CLEAN (the occupations kept), and
read_block() reads each block's text files: the rows without a category,
with value and standard error (SE) as numbers and the two flags as booleans.
The steps run in this order.

1. codes(): which of the five blocks each code in RAW is rated in. Each code
   is classed as military (55-), an SOC "All Other" residual (a .00 code whose
   last occupation digit is 9), legislators (LEGISLATORS) or an ordinary
   detailed occupation. Tables: counts, codes rated per block, the codes
   rated in no block by class, and every code not rated in all five.
2. flags(): for every block and scale, on all rows and on the rows of the
   kept occupations, the counts and shares of Recommend Suppress and Not
   Relevant, and the median value of the rows flagged Not Relevant; and the
   totals over the four blocks other than Work Context on IM and LV.
3. standard_errors(): the share of the analysis ratings (the scale of each
   block in FEATURE_BLOCKS, kept occupations) that carry a SE, by block, over
   all blocks, and by block and Domain Source.
4. item_error(): for every analysis column, on the kept occupations: its
   variance across occupations, the share with a SE, the mean SE^2 over the
   occupations that have one, the error share e = mean SE^2 / variance and
   the reliability 1 - e, also on Analyst and on Incumbent ratings alone;
   summarised by block and over all columns.
5. importance_vs_level(): for every element of the four blocks rated on both
   IM and LV, the correlation of the two across the kept occupations; the
   same summarised by block; and the out-of-fold R2 of the z-scored LV
   columns from all IM columns (cv_predictions(): ridge with scaler and
   penalty fitted in each of five folds), over all columns and per column.
6. work_context_categories(): for each CX item, its five category shares in
   RAW (missing cells at the column median, rows scaled to sum to 1).
   share_r2() gives the R2 of the five shares together from CX; from CX and
   CX^2; and from CX and the share in the two end categories (c1 + c5). The
   share of their variance in their first two principal components is
   recorded beside them. The table summarises the four over items.
7. final_matrix(): per block of the FEATURE_PREFIXES columns of CLEAN, the
   number of columns and the cells missing in RAW for the kept occupations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ONET_DIR = Path("data_raw/onet/db_31_0_text")
RAW = Path("output/master.csv")
CLEAN = Path("output/master_clean.csv")
OUT = Path("output/data_quality")
SEED = 0

# Column prefix -> O*NET files, and the scale of the analysis matrix.
FEATURE_BLOCKS = {
    "abilities": (["Abilities"], "IM"),
    "skills":    (["Essential Skills", "Transferable Skills"], "IM"),
    "knowledge": (["Knowledge"], "IM"),
    "workact":   (["Work Activities"], "IM"),
    "workctx":   (["Work Context"], "CX"),
}
FEATURE_PREFIXES = ("skills_im__", "abilities_im__", "knowledge_im__",
                    "workact_im__", "workctx_cx__", "workctx_ctp__")
TEXT_COLUMNS = {
    "O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
    "Category": "category", "Data Value": "value", "Standard Error": "se",
    "Recommend Suppress": "suppress", "Not Relevant": "not_relevant",
    "Domain Source": "source",
}
LEGISLATORS = "11-1031.00"


def read_block(prefix):
    """All rows of a block's text files without a category, value and se
    numeric, suppress and not_relevant boolean."""
    parts = []
    for stem in FEATURE_BLOCKS[prefix][0]:
        parts.append(pd.read_csv(ONET_DIR / f"{stem}.txt", sep="\t", dtype=str, encoding="utf-8",
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


def analysis_rows(rows, kept):
    """The rows on each block's analysis scale for the kept occupations."""
    return pd.concat([rows[(rows["block"] == b) & (rows["scale"] == s) & rows["soc"].isin(kept)]
                      for b, (_, s) in FEATURE_BLOCKS.items()])


# --------------------------------------------------------------------------- #
def codes(raw):
    rated = pd.DataFrame({b: raw[[c for c in raw.columns if c.startswith(f"{b}_")]].notna().any(axis=1)
                          for b in FEATURE_BLOCKS})
    n_rated = rated.sum(axis=1)
    code = raw.index.to_series()
    kind = pd.Series("ordinary detailed occupation", index=raw.index)
    kind[code.str.endswith(".00") & code.str[3:7].str.endswith("9")] = "SOC 'All Other' residual"
    kind[code.str.startswith("55-")] = "military"
    kind[code == LEGISLATORS] = "legislators"
    ordinary = kind == "ordinary detailed occupation"

    summary = pd.DataFrame([
        {"measure": "codes in Occupation Data", "value": len(raw)},
        {"measure": "rated in all five feature blocks", "value": int((n_rated == 5).sum())},
        {"measure": "rated in no feature block", "value": int((n_rated == 0).sum())},
        {"measure": "rated in some blocks only", "value": int(((n_rated > 0) & (n_rated < 5)).sum())},
        {"measure": "ordinary detailed occupations", "value": int(ordinary.sum())},
        {"measure": "ordinary detailed occupations not rated in all five",
         "value": int((ordinary & (n_rated < 5)).sum())},
    ])
    partial = n_rated < 5
    unrated = raw.loc[partial, ["title"]].assign(
        kind=kind[partial], blocks_rated=n_rated[partial],
        which=rated[partial].apply(lambda r: ", ".join(b for b in r.index if r[b]), axis=1))
    by_kind = unrated[unrated["blocks_rated"] == 0].groupby("kind").size().rename("unrated codes").to_frame()
    by_kind["share of unrated"] = by_kind["unrated codes"] / by_kind["unrated codes"].sum()
    return {"codes_summary": summary,
            "codes_per_block": rated.sum().rename("codes rated").to_frame().reset_index(names="block"),
            "unrated_by_kind": by_kind.reset_index(),
            "codes_not_fully_rated": unrated.sort_values(["blocks_rated", "kind"]).reset_index()}


def flags(rows, kept):
    out = []
    for (block, scale), g in rows.groupby(["block", "scale"]):
        for subset, h in (("all rows in the files", g),
                          (f"the {len(kept)} occupations kept", g[g["soc"].isin(kept)])):
            out.append({"block": block, "scale": scale, "rows of": subset, "rows": len(h),
                        "Recommend Suppress": int(h["suppress"].sum()),
                        "suppress share": h["suppress"].mean(),
                        "Not Relevant": int(h["not_relevant"].sum()),
                        "not relevant share": h["not_relevant"].mean(),
                        "median value, Not Relevant": h.loc[h["not_relevant"], "value"].median()})
    table = pd.DataFrame(out)
    four = table[(table["block"] != "workctx") & table["scale"].isin(["IM", "LV"])]
    totals = four.groupby(["scale", "rows of"])[["rows", "Recommend Suppress", "Not Relevant"]].sum().reset_index()
    totals["suppress share"] = totals["Recommend Suppress"] / totals["rows"]
    return {"flags": table, "flags_four_blocks": totals}


def standard_errors(rows, kept):
    used = analysis_rows(rows, kept)
    used = used.assign(has_se=used["se"].notna())
    by_block = used.groupby("block").agg(rows=("has_se", "size"), share_with_se=("has_se", "mean")).reset_index()
    by_block.loc[len(by_block)] = ["all", len(used), used["has_se"].mean()]
    by_source = used.groupby(["block", "source"]).agg(rows=("has_se", "size"),
                                                      share_with_se=("has_se", "mean")).reset_index()
    return {"se_by_block": by_block, "se_by_source": by_source}


def item_error(rows, kept):
    used = analysis_rows(rows, kept).drop_duplicates(["soc", "block", "element"])
    out = []
    for (block, element), g in used.groupby(["block", "element"]):
        var = g["value"].var(ddof=1)
        has = g["se"].notna()
        row = {"block": block, "element": element, "occupations": len(g),
               "variance across occupations": var, "share with SE": has.mean(),
               "mean SE^2": (g.loc[has, "se"] ** 2).mean()}
        row["error share e"] = row["mean SE^2"] / var
        row["reliability 1 - e"] = 1 - row["error share e"]
        for source in ("Analyst", "Incumbent"):
            h = g[has & (g["source"] == source)]
            row[f"error share, {source} ratings only"] = (h["se"] ** 2).mean() / var if len(h) else np.nan
        out.append(row)
    items = pd.DataFrame(out)
    e = items["error share e"]
    blocks = items.groupby("block").agg(items=("element", "size"), share_with_SE=("share with SE", "mean"),
                                        error_share_mean=("error share e", "mean"),
                                        error_share_median=("error share e", "median"),
                                        error_share_min=("error share e", "min"),
                                        error_share_max=("error share e", "max")).reset_index()
    blocks.loc[len(blocks)] = ["all", len(items), items["share with SE"].mean(), e.mean(), e.median(), e.min(), e.max()]
    return {"item_error_by_block": blocks, "item_error": items.sort_values("error share e", ascending=False)}


def cv_predictions(X, Y):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 30)))
    return cross_val_predict(model, X, Y, cv=KFold(5, shuffle=True, random_state=SEED))


def importance_vs_level(raw, kept):
    data = raw.loc[kept]
    pairs = []
    for block in ("abilities", "skills", "knowledge", "workact"):
        for im in [c for c in data.columns if c.startswith(f"{block}_im__")]:
            lv = im.replace("_im__", "_lv__", 1)
            if lv in data.columns:
                pairs.append({"block": block, "element": im.split("__", 1)[1], "im": im, "lv": lv,
                              "r(IM, LV)": np.corrcoef(data[im], data[lv])[0, 1]})
    elements = pd.DataFrame(pairs)
    r = elements["r(IM, LV)"]
    X = data[elements["im"]].to_numpy(float)
    Y = data[elements["lv"]].to_numpy(float)
    Yz = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    pred = cv_predictions(X, Yz)
    per_col = 1 - ((Yz - pred) ** 2).sum(axis=0) / (Yz ** 2).sum(axis=0)
    elements["R2 of LV column from all IM"] = per_col
    summary = pd.DataFrame([
        {"measure": "elements rated on both scales", "value": len(elements)},
        {"measure": "missing cells, Importance", "value": int(np.isnan(X).sum())},
        {"measure": "missing cells, Level", "value": int(np.isnan(Y).sum())},
        {"measure": "median r(IM, LV)", "value": r.median()},
        {"measure": "share of elements with r > 0.9", "value": (r > 0.9).mean()},
        {"measure": "lowest r(IM, LV)", "value": r.min()},
        {"measure": "Level matrix from Importance matrix: out-of-fold R2, all columns",
         "value": 1 - ((Yz - pred) ** 2).sum() / (Yz ** 2).sum()},
        {"measure": "same, median over Level columns", "value": np.median(per_col)},
    ])
    by_block = elements.groupby("block")["r(IM, LV)"].describe().reset_index()
    return {"im_lv_summary": summary, "im_lv_by_block": by_block,
            "im_lv_elements": elements.drop(columns=["im", "lv"]).sort_values("r(IM, LV)")}


def share_r2(P, D):
    """R2 of the columns of P together from [1, D] by least squares."""
    D = np.column_stack([np.ones(len(P)), D])
    e = P - D @ np.linalg.lstsq(D, P, rcond=None)[0]
    return 1 - (e ** 2).sum() / ((P - P.mean(axis=0)) ** 2).sum()


def work_context_categories(raw, clean):
    items = [c.split("__", 1)[1] for c in clean.columns if c.startswith("workctx_cx__")]
    rows, n_filled = [], 0
    for item in items:
        P = raw.loc[clean.index, [f"workctx_cxp__{item}__c{c}" for c in range(1, 6)]].to_numpy(float)
        n_filled += int(np.isnan(P).sum())
        P = np.where(np.isnan(P), np.nanmedian(P, axis=0), P)
        P = P / P.sum(axis=1, keepdims=True)
        cx = clean[f"workctx_cx__{item}"].to_numpy(float)
        ends = P[:, 0] + P[:, 4]
        ev = np.linalg.svd(P - P.mean(axis=0), compute_uv=False) ** 2
        rows.append({"item": item, "R2, CX": share_r2(P, cx),
                     "R2, CX and CX^2": share_r2(P, np.column_stack([cx, cx ** 2])),
                     "R2, CX and c1 + c5": share_r2(P, np.column_stack([cx, ends])),
                     "share in the first two principal components": ev[:2].sum() / ev.sum()})
    per_item = pd.DataFrame(rows)
    summary = (per_item.drop(columns="item").describe(percentiles=[0.1, 0.5, 0.9])
               .loc[["mean", "min", "10%", "50%", "90%", "max"]].reset_index(names="statistic"))
    summary["CXP cells missing, filled at the column median"] = n_filled
    return {"cxp_summary": summary, "cxp_items": per_item}


def final_matrix(raw, clean):
    feats = [c for c in clean.columns if c.startswith(FEATURE_PREFIXES)]
    blocks = pd.Series([c.split("__")[0] for c in feats], index=feats)
    published = raw.loc[clean.index, feats].apply(pd.to_numeric, errors="coerce")
    table = pd.DataFrame({"columns": blocks.value_counts(),
                          "missing cells in master.csv": published.isna().sum().groupby(blocks).sum()})
    table.loc["total"] = table.sum()
    return {"final_matrix": table.reset_index(names="block")}


# --------------------------------------------------------------------------- #
def main():
    try:
        raw = pd.read_csv(RAW, index_col="onet_soc", low_memory=False)
        clean = pd.read_csv(CLEAN, index_col="onet_soc")
    except OSError as err:
        sys.exit(f"[fail] {err}")
    if not ONET_DIR.exists():
        sys.exit(f"[fail] {ONET_DIR} not found")
    kept = clean.index
    print(f"[ok] read {RAW}: {len(raw)} codes; {CLEAN}: {len(kept)} occupations")
    rows = pd.concat([read_block(b) for b in FEATURE_BLOCKS], ignore_index=True)
    print(f"[ok] read {len(rows)} rows from the O*NET text files")

    OUT.mkdir(parents=True, exist_ok=True)
    steps = (("codes", lambda: codes(raw)), ("flags", lambda: flags(rows, kept)),
             ("standard errors", lambda: standard_errors(rows, kept)),
             ("item error", lambda: item_error(rows, kept)),
             ("Importance against Level", lambda: importance_vs_level(raw, kept)),
             ("Work Context categories", lambda: work_context_categories(raw, clean)),
             ("final matrix", lambda: final_matrix(raw, clean)))
    for label, step in steps:
        tables = step()
        for name, df in tables.items():
            df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"[ok] {label}: " + ", ".join(f"{name}.csv" for name in tables))
    print(f"[ok] wrote {len(list(OUT.glob('*.csv')))} tables to {OUT}/")


if __name__ == "__main__":
    main()