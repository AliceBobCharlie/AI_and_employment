#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py -- assemble the raw master table:

    data_raw/onet/db_31_0_text + the external sources  ->  output/master.csv

One row per O*NET-SOC code listed in O*NET's Occupation Data, rated or not.
One column per block, scale, element and (where the scale has them) category,
holding the Data Value as O*NET publishes it, followed by the labour-market
variables joined on the six-digit SOC code.

This script only reads and joins. It drops no occupation and no scale, fills
no gap and derives no variable; all of that is done in clean_master.py. In
particular it keeps Importance and Level for Abilities, Skills, Knowledge and
Work Activities; CX, CT, CXP and CTP for Work Context; and every category of
every education, training and experience scale, with required education on
both the RL and the RQ scale. Ratings that O*NET flags Not Relevant or
Recommend Suppress are kept as published; the flags are not carried over.

O*NET column names: <block>_<scale>__<Element Name>, with __c<category> added
for scales that have categories, e.g. abilities_lv__Oral Comprehension,
workctx_cxp__Public Speaking__c3, ete_rl__Required Level of Education__c6.

External columns (prefix ext_), and how their source codes are read:

  OEWS national, May 2024: employment, and the mean, median and 10th, 25th,
    75th and 90th percentiles of annual pay. OEWS marks a wage at or above
    $239,200 a year with "#"; it is read as 239200. The value is a lower
    bound, and since OEWS publishes no exact value at or above the cap, a
    value of exactly 239200 always means "at least this much". The other
    marks ("*" and "**", estimate not available; "~", fewer than 0.5 percent
    of establishments report the occupation) are read as missing.
  CPS 2024 union coverage, joined through the BLS National Employment Matrix
    crosswalk from CPS occupation codes to SOC codes.
  Occupational Prestige Ratings, averaged where several rated titles share a
    six-digit SOC code.
  BLS Employment Projections: labour-force exit and occupational transfer
    rates (table 1.10) and the self-employed share (table 1.2). In table 1.2 a
    blank or a dash means negligible self-employment and is read as 0.
    Occupations absent from the table are left missing.

Files are read one at a time, only the columns needed.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data_raw")
ONET_DIR = RAW / "onet" / "db_31_0_text"
OUT = Path("output/master.csv")

CROSSWALK = RAW / "nem-occcode-cps-crosswalk.xlsx"
OEWS_FILE = RAW / "national_M2024_dl.xlsx"
UNION_FILE = RAW / "occ_2024.xlsx"
PRESTIGE_FILE = RAW / "OccupationalPrestigeRatings.csv"
BLS_FILE = RAW / "bls_projections.xlsx"

# Column prefix -> the O*NET files that hold the block.
BLOCKS = {
    "abilities": ["Abilities"],
    "skills":    ["Essential Skills", "Transferable Skills"],
    "knowledge": ["Knowledge"],
    "workact":   ["Work Activities"],
    "workctx":   ["Work Context"],
    "ete":       ["Education", "Training and Experience"],
}

RATING_COLUMNS = {
    "O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
    "Category": "category", "Data Value": "value",
}

# OEWS columns kept, and their names in the master table.
OEWS_COLUMNS = {
    "TOT_EMP": "ext_employment", "A_MEAN": "ext_wage_mean",
    "A_MEDIAN": "ext_wage_median", "A_PCT10": "ext_wage_p10",
    "A_PCT25": "ext_wage_p25", "A_PCT75": "ext_wage_p75", "A_PCT90": "ext_wage_p90",
}
OEWS_TOP_CODE = "#"           # annual wage at or above the cap
OEWS_CAP = 239200
OEWS_NOT_AVAILABLE = {"*", "**", "~"}


def soc6(code):
    if pd.isna(code):
        return None
    m = re.match(r"(\d{2}-\d{4})", str(code))
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# O*NET
# --------------------------------------------------------------------------- #
def read_text(path, columns):
    """Read only the named columns of a tab-separated O*NET file, renamed."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                     usecols=lambda c: c in columns)
    return df.rename(columns=columns)


def block_wide(prefix):
    """One block as an occupation x column table of published values."""
    paths = [ONET_DIR / f"{stem}.txt" for stem in BLOCKS[prefix]]
    df = pd.concat([read_text(p, RATING_COLUMNS) for p in paths], ignore_index=True)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "category" in df.columns:
        category = pd.to_numeric(df["category"], errors="coerce")
    else:
        category = pd.Series(np.nan, index=df.index)
    suffix = category.map(lambda c: "" if pd.isna(c) else f"__c{int(c)}")
    df["column"] = prefix + "_" + df["scale"].str.lower() + "__" + df["element"] + suffix

    duplicated = df.duplicated(["soc", "column"])
    wide = df[~duplicated].pivot(index="soc", columns="column", values="value")
    return wide, [p.name for p in paths], int(duplicated.sum())


def report_block(prefix, wide, files, duplicates):
    """What was read. Coverage and missingness are reported by check_data.py."""
    print(f"  [{prefix}] {', '.join(files)}: {wide.shape[1]} columns, {len(wide)} codes"
          + (f"  ({duplicates} duplicate ratings, first kept)" if duplicates else ""))


def onet_wide():
    occ = read_text(ONET_DIR / "Occupation Data.txt",
                    {"O*NET-SOC Code": "onet_soc", "Title": "title"})
    wide = occ.set_index("onet_soc")
    print(f"[O*NET] {ONET_DIR}: {len(wide)} O*NET-SOC codes in Occupation Data")
    for prefix in BLOCKS:
        block, files, duplicates = block_wide(prefix)
        report_block(prefix, block, files, duplicates)
        unknown = block.index.difference(wide.index)
        if len(unknown):
            print(f"    note: {len(unknown)} rated codes not in Occupation Data: "
                  f"{', '.join(unknown)}")
        wide = wide.join(block, how="left")
    wide = wide.rename_axis("onet_soc").reset_index()
    wide["soc6"] = wide["onet_soc"].map(soc6)
    return wide


# --------------------------------------------------------------------------- #
# External sources
# --------------------------------------------------------------------------- #
def read_oews_value(series):
    """OEWS cell -> number: '#' -> the cap, the not-available marks -> missing."""
    s = series.astype(str).str.strip()
    s = s.where(s != OEWS_TOP_CODE, str(OEWS_CAP))
    s = s.where(~s.isin(OEWS_NOT_AVAILABLE))
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


def load_oews():
    df = pd.read_excel(OEWS_FILE, dtype=str)
    df = df[df["O_GROUP"].str.lower() == "detailed"].copy()
    df["soc6"] = df["OCC_CODE"].map(soc6)

    print("\n[OEWS] marks read per column: '#' -> 239200 | not available -> missing")
    for col, name in OEWS_COLUMNS.items():
        cell = df[col].astype(str).str.strip()
        n_top = int((cell == OEWS_TOP_CODE).sum())
        n_na = int(cell.isin(OEWS_NOT_AVAILABLE).sum())
        df[name] = read_oews_value(df[col])
        print(f"    {name:18s} {n_top:4d} top-coded | {n_na:4d} not available")

    return df[["soc6"] + list(OEWS_COLUMNS.values())].drop_duplicates("soc6")


def load_crosswalk():
    cw = pd.read_excel(CROSSWALK, dtype=str, skiprows=4).rename(columns={
        "National Employment Matrix code": "soc", "CPS code": "cps"})
    cw["soc6"] = cw["soc"].map(soc6)
    cw = cw[["soc6", "cps"]].dropna()
    cw["cps"] = cw["cps"].str.strip()
    return cw.drop_duplicates("soc6")


def load_union():
    raw = pd.read_excel(UNION_FILE, dtype=str, skiprows=2)
    raw.columns = [str(c).strip() for c in raw.columns]
    rename = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in ("coc", "code") or re.match(r"^coc", cl):
            rename[c] = "cps"
        elif ("cov" in cl and "%" in c) or "density" in cl or "percent" in cl:
            rename[c] = "ext_union_cov_pct"
    raw = raw.rename(columns=rename)
    keep = [c for c in ["cps", "ext_union_cov_pct"] if c in raw.columns]
    u = raw[keep].copy()
    u["cps"] = u["cps"].astype(str).str.strip()
    if "ext_union_cov_pct" in u:
        u["ext_union_cov_pct"] = pd.to_numeric(u["ext_union_cov_pct"], errors="coerce")
    return u.drop_duplicates("cps")


def load_prestige():
    p = pd.read_csv(PRESTIGE_FILE, dtype=str)
    code_col = next((c for c in p.columns if "onet" in c.lower() and "code" in c.lower()), None)
    rate_col = next((c for c in p.columns if "opr" in c.lower() and "rating" in c.lower()), None)
    p = p[[code_col, rate_col]].rename(columns={code_col: "c", rate_col: "ext_prestige"})
    p["ext_prestige"] = pd.to_numeric(p["ext_prestige"], errors="coerce")
    p["soc6"] = p["c"].map(soc6)
    return p.dropna(subset=["soc6"]).groupby("soc6", as_index=False)["ext_prestige"].mean()


def load_bls():
    def read_line_items(sheet):
        d = pd.read_excel(BLS_FILE, sheet_name=sheet, skiprows=1)
        d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
        code = next(c for c in d.columns if "code" in c.lower())
        typ = next(c for c in d.columns if "occupation" in c.lower() and "type" in c.lower())
        d = d[d[typ].astype(str).str.strip() == "Line item"].copy()
        d["soc6"] = d[code].map(soc6)
        return d

    def find(d, *kw):
        return next((c for c in d.columns if all(k in c.lower() for k in kw)), None)

    t110 = read_line_items("Table 1.10")
    S = pd.DataFrame({"soc6": t110["soc6"]})
    S["ext_sep_exit_rate"] = pd.to_numeric(t110[find(t110, "exit", "rate")], errors="coerce")
    S["ext_sep_transfer_rate"] = pd.to_numeric(t110[find(t110, "transfer", "rate")], errors="coerce")
    S = S.drop_duplicates("soc6")

    # In table 1.2 a blank or a dash is the published value 'negligible'.
    t12 = read_line_items("Table 1.2")
    T = pd.DataFrame({"soc6": t12["soc6"]})
    cell = t12[find(t12, "self employed")].astype(str).str.strip()
    T["ext_self_employed_pct"] = pd.to_numeric(
        cell.mask(cell == "—", "0"), errors="coerce").fillna(0)
    T = T.drop_duplicates("soc6")
    return S.merge(T, on="soc6", how="outer")


def join_external(onet):
    n0 = len(onet)
    M = onet.merge(load_oews(), on="soc6", how="left")
    M = M.merge(load_crosswalk(), on="soc6", how="left")
    M = M.merge(load_union(), on="cps", how="left")
    M = M.merge(load_prestige(), on="soc6", how="left")
    M = M.merge(load_bls(), on="soc6", how="left")
    M = M.drop(columns=["cps"], errors="ignore")
    assert len(M) == n0, f"row count changed {n0}->{len(M)}"
    return M


# --------------------------------------------------------------------------- #
def main():
    OUT.parent.mkdir(exist_ok=True)
    M = join_external(onet_wide())
    M.to_csv(OUT, index=False, encoding="utf-8")
    n_onet = len([c for c in M.columns if "__" in c])
    n_ext = len([c for c in M.columns if c.startswith("ext_")])
    print(f"\n[ok] wrote {OUT}: {M.shape[0]} rows x {M.shape[1]} columns "
          f"({n_onet} O*NET, {n_ext} external)")
    print("coverage, flags and missingness: run check_data.py")


if __name__ == "__main__":
    main()
