#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data.py -- the data-quality facts Sections 4.2 and 4.3 rely on.

build_master.py only reads and joins, and does not carry O*NET's quality
flags into the master table; clean_master.py only reports what it drops and
fills. Everything the paper says about the quality of the data is computed
here, from the master tables and, for the flags and standard errors, from the
O*NET text files themselves.

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
  E  Work Context Category scale
                  The two CT items against the 216 feature columns: their
                  closest CX items, the cross-validated R2 with which the 216
                  columns predict them, and the variance they would add that
                  the 216 columns do not already hold.
  F  the final matrix
                  Columns and missing cells per block of the 910 x 216
                  matrix (Table 4.1).
  G  labour-market variables
                  For the 910 occupations, raw values: which occupations miss
                  which variable, and each variable's distribution and skew.

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
                    "workact_im__", "workctx_cx__")
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


def section_ct(M, C):
    banner("E. WORK CONTEXT CATEGORY SCALE -- what the two CT items would add")
    feats = [c for c in C.columns if c.startswith(FEATURE_PREFIXES)]
    X = C[feats].to_numpy(float)
    ct = [c for c in M.columns if c.startswith("workctx_ct__")]
    rows, unique = [], 0.0
    for c in ct:
        y = M.loc[C.index, c].to_numpy(float)
        ok = ~np.isnan(y)
        r = pd.Series({f: np.corrcoef(C.loc[ok, f], y[ok])[0, 1]
                       for f in feats if f.startswith("workctx_cx__")})
        top = r.abs().sort_values(ascending=False).head(3).index
        pred = cv_predictions(X[ok], y[ok])
        r2 = 1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        unique += 1 - r2
        rows.append({"item": c.split("__", 1)[1], "occupations": int(ok.sum()),
                     "closest CX items (r)": "; ".join(f"{t.split('__', 1)[1]} {r[t]:+.2f}" for t in top),
                     "R2 from the 216 columns": r2})
    T = pd.DataFrame(rows)
    p = len(feats) + len(ct)
    summary = pd.DataFrame([{"measure": f"variance the CT items add beyond the 216 columns, "
                                        f"share of a {p}-column matrix", "value": unique / p},
                            {"measure": f"their share of a {p}-column matrix in total", "value": len(ct) / p}])
    show(T.set_index("item")); print(); show(summary.set_index("measure"))
    return {"ct_items": T, "ct_summary": summary}


def section_matrix(C):
    banner("F. THE FINAL MATRIX (Table 4.1)")
    feats = [c for c in C.columns if c.startswith(FEATURE_PREFIXES)]
    blocks = pd.Series([c.split("__")[0] for c in feats], index=feats)
    T = pd.DataFrame({"columns": blocks.value_counts(),
                      "missing cells": C[feats].isna().sum().groupby(blocks).sum()})
    T.loc["total"] = T.sum()
    print(f"{len(C)} occupations x {len(feats)} columns")
    show(T)
    return {"final_matrix": T.reset_index(names="block")}


def section_external(M, kept):
    banner("G. LABOUR-MARKET VARIABLES -- raw values for the 910 occupations")
    ext = [c for c in M.columns if c.startswith("ext_")]
    X = M.loc[kept, ext].apply(pd.to_numeric, errors="coerce")
    na = X.isna()
    per_var = na.sum().rename("missing").to_frame()
    missing = na[na.any(axis=1)].astype(int)
    missing.columns = [c.replace("ext_", "") for c in ext]
    missing.insert(0, "title", M.loc[missing.index, "title"])
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
    return {"external_missing": missing.reset_index(), "external_describe": desc.reset_index(names="variable")}


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
    sheets.update(section_ct(M, C))
    sheets.update(section_matrix(C))
    sheets.update(section_external(M, kept))

    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
